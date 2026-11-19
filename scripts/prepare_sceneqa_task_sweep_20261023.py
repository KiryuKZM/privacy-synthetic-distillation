#!/usr/bin/env python3
"""Prepare public SQA3D/ScanQA text QA diagnostics. No model/key/GPU access.

The fetch stage reads only bounded official public releases and writes receipts.
The offline stage never exports target answers, grounding IDs, or annotated pose.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/2026-10-23-llm-task-sweep/preparation/sceneqa'
SOURCES = BASE / 'sources'
SEED = 20261023
MAX_BYTES = 499_000_000
INITIAL = {
    'sqa3d-README.md': 'https://raw.githubusercontent.com/SilongYong/SQA3D/master/README.md',
    'sqa3d-LLM-README.md': 'https://raw.githubusercontent.com/SilongYong/SQA3D/master/LLM/README.md',
    'eval_sqa3d_llm.py': 'https://raw.githubusercontent.com/SilongYong/SQA3D/master/LLM/eval_sqa3d_llm.py',
    'sqa3d-dataset.md': 'https://raw.githubusercontent.com/SilongYong/SQA3D/master/assets/dataset.md',
    'scanqa-dataset.md': 'https://raw.githubusercontent.com/ATR-DBI/ScanQA/main/docs/dataset.md',
    'scanqa-README.md': 'https://raw.githubusercontent.com/ATR-DBI/ScanQA/main/README.md',
    'sqa3d-tree.json': 'https://api.github.com/repos/SilongYong/SQA3D/git/trees/master?recursive=1',
    'scanqa-tree.json': 'https://api.github.com/repos/ATR-DBI/ScanQA/git/trees/main?recursive=1',
    'sqa3d-zenodo.json': 'https://zenodo.org/api/records/7792397',
}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=canonical(value)+b'\n'
    if path.exists() and path.read_bytes()!=data: raise ValueError('Immutable output differs: '+str(path))
    if not path.exists():path.write_bytes(data)

def fetch_one(name,url):
    if '/' in name or '\\' in name: raise ValueError('Flat filename required')
    SOURCES.mkdir(parents=True,exist_ok=True)
    p=SOURCES/name;receipt=SOURCES/(name+'.receipt.json')
    if receipt.exists():
        old=json.loads(receipt.read_text(encoding='utf8'))
        if old['url']!=url or old['sha256']!=sha(p):raise ValueError('Cached source mismatch')
        return old
    req=urllib.request.Request(url,headers={'User-Agent':'public-research-sceneqa-preflight/1.0'})
    with urllib.request.urlopen(req,timeout=40) as r:
        length=r.headers.get('Content-Length')
        if length and int(length)>MAX_BYTES:raise ValueError('Source exceeds bounded asset cap')
        b=r.read(MAX_BYTES+1)
        if len(b)>MAX_BYTES:raise ValueError('Source exceeds bounded asset cap')
        meta={'url':url,'resolved_url':r.url,'etag':r.headers.get('ETag'),
              'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'path':p.relative_to(ROOT).as_posix()}
    if p.exists() and p.read_bytes()!=b:raise ValueError('Unreceipted cache differs')
    p.write_bytes(b);write(receipt,meta)
    return meta

def initial_fetch():
    output=[]
    for name,url in INITIAL.items():
        try: output.append(fetch_one(name,url));print(name,output[-1]['bytes'],flush=True)
        except Exception as e:print(name,type(e).__name__,str(e),flush=True)
    return output

class PrimitivesOnly(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError('No pickle globals permitted')

def restricted_pickle(path):
    return PrimitivesOnly(io.BytesIO(Path(path).read_bytes())).load()

def sqa_normalizer(source):
    """Extract only the official pure clean_answer function, never import runner."""
    tree=ast.parse(Path(source).read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='clean_answer')
    mod=ast.Module(body=[node],type_ignores=[]);space={'re':re}
    exec(compile(mod,str(source),'exec'),space)
    def clean(answer):
        return space['clean_answer']([{'answer':str(answer)}])[0]['answer']
    return clean

def strict_em(prediction,aliases,normalizer):
    """Exact normalized aliases; no substring/overlap or model-judge credit."""
    pred=normalizer(prediction)
    return bool(pred) and pred in {normalizer(a) for a in aliases if str(a).strip()}

@lru_cache(maxsize=1)
def answer_normalizer():
    return sqa_normalizer(SOURCES/'eval_sqa3d_llm.py')

def score_response(content,aliases):
    """Local objective schema+EM adapter; never inspect reasoning/fallback text."""
    try:
        obj=json.loads(content)
    except (json.JSONDecodeError,TypeError):
        return {'valid':False,'correct':False,'prediction':None,'reason':'INVALID_JSON'}
    if not isinstance(obj,dict) or set(obj)!={'answer'} or not isinstance(obj['answer'],str) or not obj['answer'].strip():
        return {'valid':False,'correct':False,'prediction':None,'reason':'INVALID_ANSWER_SCHEMA'}
    return {'valid':True,'correct':strict_em(obj['answer'],aliases,answer_normalizer()),
        'prediction':obj['answer'],'reason':'STRICT_NORMALIZED_ALIAS_EM'}

def inspect_sources():
    for name in ('sqa3d-tree.json','scanqa-tree.json'):
        path=SOURCES/name
        if not path.exists():continue
        obj=json.loads(path.read_text())
        print(name,'commit',obj.get('sha'))
        print(json.dumps([{k:x.get(k) for k in ('path','size','sha')} for x in obj.get('tree',[])
            if re.search('caption|sqa_task|ScanQA_v1|qa/|scene.*txt|metric|eval.*util',x['path'],re.I)],indent=2))
    path=SOURCES/'sqa3d-zenodo.json'
    if path.exists():
        obj=json.loads(path.read_text())
        print('zenodo',json.dumps([{k:x.get(k) for k in ('key','size','checksum','links')} for x in obj.get('files',[])],indent=2))

VLA=ROOT/'runs/2026-10-23-benchmark-discovery/research/selection-vla'

def physical_space(scene):
    if not re.fullmatch(r'scene\d{4}_\d{2}',scene):raise ValueError('Bad ScanNet identity')
    return scene.split('_')[0]

def independent_objects(scene):
    """All original observation fields, no region function or NYU remappings.

    VLA object indices are explicitly a different namespace from ScanNet target
    grounding IDs. No unverified cross-source instance join occurs here.
    """
    path=VLA/'full-csv'/(scene+'_object_result.csv')
    if not path.exists():return None
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    values=[]
    for row in rows:
        obj={'observation_index':int(row['object_id']),'category':row['raw_label']}
        obj['bbox']={key.removeprefix('object_bbox_'):float(row[key]) for key in row if key.startswith('object_bbox_')}
        obj['front_heading']=None if row['object_front_heading'] in (None,'_','') else float(row['object_front_heading'])
        colors=[]
        for index in (1,2,3):
            if row['object_color_scheme'+str(index)] in (None,'_',''):continue
            colors.append({'name':row['object_color_scheme'+str(index)],
                'rgb':[int(row['object_color_'+key+str(index)]) for key in 'rgb'],
                'proportion':float(row['object_color_scheme_percentage'+str(index)]),
                'average_color_distance':float(row['object_color_scheme_average_dist'+str(index)])})
        obj['dominant_colors']=colors
        if any(not math.isfinite(x) for x in obj['bbox'].values()):raise ValueError('Nonfinite bbox')
        values.append(obj)
    return {'source':'VLA independent pointcloud-derived object observations',
        'instance_namespace':'VLA sequential enumeration; NOT assumed ScanNet objectId',
        'geometry_units':'native ScanNet coordinates; physical metric interpretation awaits source calibration audit',
        'axis_convention':'native VLA coordinate frame; not viewer left/right',
        'objects':values,'source_binding':{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size}}

def caption_records(path):
    if Path(path).suffix=='.pkl':
        data=restricted_pickle(path)
        if not isinstance(data,dict):raise ValueError('Expected scene caption mapping')
        return {scene:[{'category':str(x[0]),'text':str(x[1])} for x in captions] for scene,captions in data.items()}
    data=json.loads(Path(path).read_text(encoding='utf8'));by_scene=defaultdict(list)
    for row in data:
        by_scene[row['scene_id']].append({'source_object_id':str(row.get('object_id','')),'text':row['description']})
    return dict(by_scene)

def jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=b''.join(canonical(x)+b'\n' for x in rows)
    if path.exists() and path.read_bytes()!=data:raise ValueError('Immutable jsonl differs')
    if not path.exists():path.write_bytes(data)

def question_kind(text):
    text=text.lower()
    if 'color' in text or 'colour' in text:return 'color'
    if 'made of' in text or 'material' in text:return 'material'
    if 'how many' in text:return 'count'
    if any(x in text for x in ('left','right','behind','front','between','closest','nearest','where','under','above','facing')):return 'spatial_or_reference'
    return 'other_semantic'

def component_split(scene_inputs,audit):
    """Exact duplicate observation/caption worlds and scan-prefix grouping."""
    parents={physical_space(s):physical_space(s) for s in scene_inputs}
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
        return x
    seen={};edges=[]
    for scene,value in sorted(scene_inputs.items()):
        # IDs and source-binding paths do not define observation content.
        observations=value.get('object_observations')
        reduced={'captions':[(k,[v['text'] for v in rows]) for k,rows in sorted(value['scene_captions'].items())],
            'objects':observations['objects'] if observations else None}
        fingerprint=hashlib.sha256(canonical(reduced)).hexdigest();space=physical_space(scene)
        if fingerprint in seen:
            other=seen[fingerprint];a,b=find(space),find(other)
            if a!=b:parents[max(a,b)]=min(a,b);edges.append([space,other,fingerprint])
        else:seen[fingerprint]=space
    groups=defaultdict(list)
    for scene in scene_inputs:groups[find(physical_space(scene))].append(scene)
    audit_components={find(g) for g in audit if g in parents}
    usable=[g for g in groups if g not in audit_components]
    usable.sort(key=lambda x:hashlib.sha256(canonical([SEED,'joint-sceneqa-roles',x])).hexdigest())
    # Scene/world selection is independent of QA labels and model outputs.
    if len(usable)<200:raise ValueError('Insufficient independent worlds')
    roles={g:('DevPool' if i<32 else 'ScreenPool' if i<132 else 'Fit') for i,g in enumerate(usable)}
    roles.update({g:'Audit' for g in audit_components})
    return {scene:roles[find(physical_space(scene))] for scene in scene_inputs},edges

SYSTEM=('Public academic text QA diagnostic on independently released ScanNet scene descriptions and object observations. '
    'Answer the question from the complete supplied scene context. Treat the situated description as the observer frame; '
    'raw geometry uses a source frame and is not automatically observer left/right. Scene captions and object observations '
    'are separate evidence sources; captions contain category/text pairs and have no instance-ID correspondence. '
    'Descriptions may be partial and do not assert exhaustive visibility. Use natural language semantics and supplied facts, '
    'do not invent material or visibility facts from category priors. Return exactly one JSON object {"answer":"short word or phrase"}. '
    'If evidence is insufficient, return {"answer":"unknown"}; it remains in the fixed denominator.')

def select_queries(records,role_by_scene,role,number,tag):
    eligible=[r for r in records if role_by_scene[r['scene_id']]==role]
    # At most one question per physical world in initial bounded Dev/Screen.
    eligible.sort(key=lambda r:hashlib.sha256(canonical([SEED,tag,r['record_id']])).hexdigest())
    chosen=[];seen=set()
    for record in eligible:
        world=physical_space(record['scene_id'])
        if world in seen:continue
        seen.add(world);chosen.append(record)
        if len(chosen)==number:break
    if len(chosen)!=number:raise ValueError(f'Only {len(chosen)} {role} questions for {tag}')
    return chosen

def self_check():
    normal=sqa_normalizer(SOURCES/'eval_sqa3d_llm.py')
    assert strict_em('2',['two'],normal)
    assert strict_em('Brown.',['brown'],normal)
    assert not strict_em('', ['brown'],normal)
    assert not strict_em('chair',['brown chair'],normal)
    assert not strict_em('brown table',['brown chair'],normal)
    assert score_response('{"answer":"2"}',['two'])['correct']
    assert not score_response('brown',['brown'])['valid']
    assert not score_response('{"answer":"brown","evidence":"extra"}',['brown'])['valid']
    assert not score_response('{"answer":"chair"}',['brown chair'])['correct']
    obj=independent_objects('scene0380_00')
    assert obj and obj['objects'] and all('region_label' not in x for x in obj['objects'])
    assert any(x['dominant_colors'] for x in obj['objects'])
    assert 'NOT assumed ScanNet objectId' in obj['instance_namespace']
    print('PASS strict answer aliases, no empty/substring credit, independent color fields, no false ID join')

def load_train_records():
    result={}
    archive=SOURCES/'sqa_task.zip'
    if archive.exists() and archive.stat().st_size==4552006:
        expected='0da8bb0ed16ea8094d433eeda397f45a'
        if hashlib.md5(archive.read_bytes()).hexdigest()!=expected:raise ValueError('Official SQA MD5 mismatch')
        with zipfile.ZipFile(archive) as z:
            # Only directory metadata is read for official heldout members.
            metadata=[{'name':i.filename,'bytes':i.file_size,'crc32':f'{i.CRC:08x}',
                       'decoded':('train' in i.filename and i.filename.endswith('.json'))} for i in z.infolist()]
            write(BASE/'sqa-archive-members.json',metadata)
            questions=[];annotations=[]
            for name in z.namelist():
                if 'train' not in name or not name.endswith('.json') or name.startswith('__MACOSX'):continue
                obj=json.loads(z.read(name))
                if 'questions' in obj:questions=obj['questions']
                elif 'annotations' in obj:annotations=obj['annotations']
            by_id={str(a['question_id']):a for a in annotations}
            if len(by_id)!=len(annotations):raise ValueError('Duplicate SQA question IDs')
            records=[]
            for q in questions:
                a=by_id[str(q['question_id'])]
                aliases=[v['answer'] for v in a['answers']]
                if not aliases:raise ValueError('Missing SQA aliases')
                records.append({'record_id':'SQA3D:'+str(q['question_id']),'scene_id':q['scene_id'],
                    'question':q['question'],'situation':q['situation'],
                    'alternative_situations':q.get('alternative_situation',[]),'aliases':aliases})
            result['SQA3D']=records
    train=SOURCES/'ScanQA_v1.0_train.json'
    if train.exists():
        obj=json.loads(train.read_text(encoding='utf8'))
        if not isinstance(obj,list):raise ValueError('ScanQA expected official list')
        result['ScanQA']=[{'record_id':'ScanQA:'+str(q['question_id']),'scene_id':q['scene_id'],
            'question':q['question'],'aliases':q['answers']} for q in obj]
    return result

def export_payload(record,worlds):
    # Explicit allowlist; target answers/groundings/annotated pose never enter.
    return {'scene_context':worlds[record['scene_id']],
        **({'situation':record['situation']} if 'situation' in record else {}),
        **({'alternative_situations':record['alternative_situations']} if 'alternative_situations' in record else {}),
        'question':record['question']}

def choose_donors(query,fit,count=2):
    tokens=lambda r:Counter(re.findall(r'[a-z0-9]+',(r.get('situation','')+' '+r['question']).lower()))
    querytokens=tokens(query)
    def score(record):
        other=tokens(record)
        return sum(querytokens[t]*other[t] for t in querytokens)/max(1,math.sqrt(sum(v*v for v in querytokens.values())*sum(v*v for v in other.values())))
    ordered=sorted(fit,key=lambda r:(-score(r),hashlib.sha256(canonical([SEED,'donors',r['record_id']])).hexdigest()))
    chosen=[];worlds={physical_space(query['scene_id'])}
    for donor in ordered:
        world=physical_space(donor['scene_id'])
        if world in worlds:continue
        worlds.add(world);chosen.append(donor)
        if len(chosen)==count:return chosen
    raise ValueError('Insufficient unique-world Fit donors')

def messages_for(record,worlds,fit,condition):
    system=SYSTEM
    if condition=='evidence2':
        system+=' Resolve referring expressions and the observer frame first, inspect color/material/count evidence for the referenced objects, then answer briefly. Fit demonstrations illustrate the required answer granularity; do not copy their facts into the current world.'
    messages=[{'role':'system','content':system}];donors=[]
    if condition=='evidence2':
        donors=choose_donors(record,fit)
        for donor in donors:
            messages.append({'role':'user','content':canonical(export_payload(donor,worlds)).decode()})
            messages.append({'role':'assistant','content':canonical({'answer':donor['aliases'][0]}).decode()})
    messages.append({'role':'user','content':canonical(export_payload(record,worlds)).decode()})
    return messages,[d['record_id'] for d in donors]

def prepare():
    records=load_train_records()
    if not records:raise ValueError('No complete official train release present')
    caption_sets={}
    expected_sizes={'scanrefer_captions_by_scene.pkl':4701136,'nr3d_captions_by_scene.pkl':2743334}
    for name,size in expected_sizes.items():
        path=SOURCES/name
        if path.exists() and path.stat().st_size==size:
            # Verify pinned Git blob, independent of HTTP transport success.
            metadata=json.loads((SOURCES/'sqa3d-tree.json').read_text())
            blob=next(x for x in metadata['tree'] if x['path']=='LLM/'+name)
            data=path.read_bytes();actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if actual!=blob['sha']:raise ValueError('Pinned Git caption blob mismatch')
            caption_sets[name]=caption_records(path)
    if not caption_sets:raise ValueError('No complete verified independent scene captions')
    scenes=sorted({r['scene_id'] for rows in records.values() for r in rows})
    worlds={}
    for scene in scenes:
        independent=independent_objects(scene)
        if independent:
            # Source bindings and source IDs remain solely in the local manifest.
            independent={k:v for k,v in independent.items() if k!='source_binding'}
        worlds[scene]={'scene_captions':{name:[{'caption_ref':f'{name.split("_")[0]}-{i+1}',
            'category':entry['category'],'text':entry['text']} for i,entry in enumerate(caption_sets[name].get(scene,[]))] for name in caption_sets},
            'object_observations':independent}
    old_audit=json.loads((VLA/'sample-retrieval.json').read_text())['selected_spaces']
    audit=set(old_audit)|{'scene0380'}
    roles,duplicates=component_split(worlds,audit)
    role_receipt={'seed':SEED,'scene_roles':roles,'audit_spaces':sorted(audit),
        'duplicate_world_edges':duplicates,'group_unit':'sceneNNNN physical scan prefix plus exact independent-world duplicates',
        'source_splits':'Only native official QA train files decoded; native val/test unopened',
        'cross_task_roles':'One shared role assignment for any overlapping SQA3D/ScanQA world',
        'old_classification_exposure':'Some same public ScanNet worlds have prior sceneType diagnostics; QA target answers have not been evaluated'}
    write(BASE/'roles.local.json',role_receipt)
    summary={'status':'PREPARED_ZERO_API_AWAITING_EXACT_PACKAGE','model_api_calls':0,'keys_read':False,
        'GPU':False,'official_heldout_qa_content_decoded':False,
        'shared_caption_pickle_decoded_globally':True,'heldout_caption_values_inspected_or_exported':False,
        'tasks':{},'independent_sources':list(caption_sets),
        'geometry_precision':'full source decimal values; no rounding/truncation',
        'information_limitations':['Captions describe selected objects and relations, not exhaustive perception',
            'Pointcloud color clusters may differ from natural-language perceived color',
            'No material, transparency, visibility or state is invented where not independently described',
            'Source geometry and caption IDs are NOT joined to target grounding IDs',
            'Annotated query position/rotation are excluded; situated textual description is the native LLM input'],
        'task_input_extension':'SQA3D primary situation plus all released alternative situation descriptions; all caption families and independent VLA observations rather than author random30-caption selection',
        'scorer':{'metric':'exact match after official SQA3D clean_answer normalization against ALL released aliases',
            'not_native_leaderboard':'No substring/one-token-overlap credit, no model judge; ScanQA additionally reuses this disclosed project metric'},
        'fewshot':'2 lexical cosine nearest Fit questions, one per independent scene, target worlds excluded; published public donor answers only'}
    for task,rows in records.items():
        task_dir=BASE/task;task_dir.mkdir(exist_ok=True)
        if len({r['record_id'] for r in rows})!=len(rows):raise ValueError('Duplicate QA IDs')
        fit=[r for r in rows if roles[r['scene_id']]=='Fit']
        dev=select_queries(rows,roles,'DevPool',16,task+'-dev')
        screen=select_queries(rows,roles,'ScreenPool',60,task+'-screen')
        for label,selected in (('Fit',fit),('Dev16',dev),('Screen60',screen)):
            if label=='Fit':
                jsonl(task_dir/(label+'-inputs.local.jsonl'),[{'record_id':r['record_id'],'group_id':physical_space(r['scene_id']),
                    'scene_id_local':r['scene_id'],'world_snapshot_ref':'../worlds.local.jsonl',
                    'question':r['question'],**({'situation':r['situation']} if 'situation' in r else {}),
                    **({'alternative_situations':r['alternative_situations']} if 'alternative_situations' in r else {})} for r in selected])
            else:
                jsonl(task_dir/(label+'-inputs.local.jsonl'),[{'record_id':r['record_id'],'group_id':physical_space(r['scene_id']),
                    'payload':export_payload(r,worlds)} for r in selected])
            write(task_dir/(label+'-gold.local.json'),{r['record_id']:r['aliases'] for r in selected})
        requestrows=[]
        for phase,selected in (('Dev16',dev),('Screen60',screen)):
            for record in selected:
                for condition in ('base0','evidence2'):
                    messages,donors=messages_for(record,worlds,fit,condition)
                    requestrows.append({'record_id':record['record_id'],'group_id':physical_space(record['scene_id']),
                        'phase':phase,'condition':condition,'messages':messages,'donor_ids_local':donors,
                        'messages_sha256':hashlib.sha256(canonical(messages)).hexdigest(),
                        'utf8_bytes':len(canonical(messages)),'conservative_input_token_bound':len(canonical(messages))})
        jsonl(task_dir/'requests.local.jsonl',requestrows)
        # Gold is read only locally after membership is selected; this descriptive
        # support audit does not remove, replace, or rank any questions.
        by_phase={}
        for phase,selected in (('Dev16',dev),('Screen60',screen)):
            by_phase[phase]={'questions':len(selected),'physical_worlds':len({physical_space(r['scene_id']) for r in selected}),
                'question_types_from_text_only':dict(Counter(question_kind(r['question']) for r in selected)),
                'no_scene_captions':sum(not any(worlds[r['scene_id']]['scene_captions'].values()) for r in selected),
                'no_object_observations':sum(worlds[r['scene_id']]['object_observations'] is None for r in selected)}
        summary['tasks'][task]={'native_train_questions':len(rows),'native_train_worlds':len({physical_space(r['scene_id']) for r in rows}),
            'role_question_counts':dict(Counter(roles[r['scene_id']] for r in rows)),
            'Fit_questions':len(fit),'selected':by_phase,'requests_prepared':len(requestrows),
            'max_input_bytes':max(r['utf8_bytes'] for r in requestrows),'max_actual_tokens':'PENDING root tokenizer audit',
            'question_filtering':'none; fixed hash queries retained irrespective of answer support or future teacher correctness'}
    jsonl(BASE/'worlds.local.jsonl',[{'scene_id':s,'group_id':physical_space(s),'role':roles[s],'payload':worlds[s]} for s in scenes])
    write(BASE/'preparation-summary.json',summary)
    write(BASE/'source-manifest.json',{'assets':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(SOURCES.iterdir()) if p.is_file() and not p.name.endswith('.receipt.json')],
        'SQA3D_commit':'a98da8fda65f81f614bdfed48c27aae34975c566',
        'source_csv_manifest':{'path':(VLA/'full-csv-summary.json').relative_to(ROOT).as_posix(),'sha256':sha(VLA/'full-csv-summary.json')},
        'source_preprocessor':{'path':(VLA/'scannet_preprocessing.py').relative_to(ROOT).as_posix(),'sha256':sha(VLA/'scannet_preprocessing.py')},
        'script':{'path':Path(__file__).relative_to(ROOT).as_posix(),'sha256':sha(__file__)}})
    print(json.dumps(summary,indent=2,ensure_ascii=False))

def audit():
    normal=sqa_normalizer(SOURCES/'eval_sqa3d_llm.py')
    roles=json.loads((BASE/'roles.local.json').read_text())['scene_roles']
    query_groups=set();fit_groups=set();counts={};token_details=[]
    try:
        from tokenizers import Tokenizer
        tokenizer_path=ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json'
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    except ImportError:
        tokenizer=None
    for directory in (BASE/'SQA3D',BASE/'ScanQA'):
        if not (directory/'requests.local.jsonl').exists():continue
        fit_rows=[json.loads(line) for line in (directory/'Fit-inputs.local.jsonl').read_text().splitlines()]
        fit_ids={r['record_id']:r['group_id'] for r in fit_rows}
        fit_gold=json.loads((directory/'Fit-gold.local.json').read_text())
        fit_groups|={r['group_id'] for r in fit_rows}
        requests=[json.loads(line) for line in (directory/'requests.local.jsonl').read_text().splitlines()]
        for request in requests:
            query_groups.add(request['group_id'])
            assert hashlib.sha256(canonical(request['messages'])).hexdigest()==request['messages_sha256']
            assert all(fit_ids[d]!=request['group_id'] for d in request['donor_ids_local'])
            assert len({fit_ids[d] for d in request['donor_ids_local']})==len(request['donor_ids_local'])
            assistant=[m for m in request['messages'] if m['role']=='assistant']
            assert len(assistant)==len(request['donor_ids_local'])
            for m,donor in zip(assistant,request['donor_ids_local']):
                assert json.loads(m['content'])=={'answer':fit_gold[donor][0]}
            for message in request['messages']:
                if message['role']!='user':continue
                payload=json.loads(message['content'])
                assert set(payload)<={'scene_context','situation','alternative_situations','question'}
                assert not any(x in payload for x in ('answer','answers','position','rotation','object_id','object_names'))
            if tokenizer:
                # Chat framing is estimated conservatively; provider bills actual usage.
                token_count=sum(len(tokenizer.encode(m['role']+'\n'+m['content']).ids)+12 for m in request['messages'])+8
                token_details.append({'task':directory.name,'record_id':request['record_id'],'phase':request['phase'],
                    'condition':request['condition'],'estimated_input_tokens':token_count})
        counts[directory.name]={'requests':len(requests),'Fit_questions':len(fit_ids)}
    assert not query_groups&fit_groups
    assert 'scene0380' not in query_groups|fit_groups
    assert strict_em('2',['two'],normal) and not strict_em('chair',['brown chair'],normal)
    result={'status':'PASS','model_api_calls':0,'counts':counts,'query_fit_group_overlap':0,
        'annotated_pose_or_grounding_target_in_query':False,'source_alias_scorer_fixtures':'PASS',
        'native_heldout_qa_files_decoded':False,'tokenizer_available':tokenizer is not None}
    if tokenizer:
        result.update({'tokenizer':{'path':tokenizer_path.relative_to(ROOT).as_posix(),'sha256':sha(tokenizer_path)},
            'estimated_input_token_total':sum(t['estimated_input_tokens'] for t in token_details),
            'max_estimated_input_tokens':max(t['estimated_input_tokens'] for t in token_details),
            'token_count_method':'cached tokenizer; per role/content encoding +12 framing tokens/message +8; not billed usage'})
        write(BASE/'token-details.local.json',token_details)
    write(BASE/'offline-audit.json',result)
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fetch-initial','fetch','inspect','prepare','self-check','audit'])
    p.add_argument('--name');p.add_argument('--url');args=p.parse_args()
    if args.command=='fetch-initial':initial_fetch()
    elif args.command=='fetch':print(json.dumps(fetch_one(args.name,args.url),ensure_ascii=False))
    elif args.command=='inspect':inspect_sources()
    elif args.command=='prepare':prepare()
    elif args.command=='self-check':self_check()
    elif args.command=='audit':audit()

if __name__=='__main__':main()
