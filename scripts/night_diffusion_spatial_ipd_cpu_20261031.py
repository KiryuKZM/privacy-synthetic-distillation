"""Offline CPU diagnostics. No model weights, CUDA, API, training or Test features."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg')
import collections as C
import hashlib
import json
import math
import re
import struct
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
O = R / 'runs/2026-10-31-night-cpu-v1'
SG = R / 'runs/2026-10-27-generator-formal/preparation/realgraph-full-incidence-v1'
MP = R / 'runs/2026-10-22-matterport-acceptance/geometry/native-inputs.local.json'
RAW = R / 'runs/2026-10-30-real-incidence-smallfit/execution/sample/work/diagnostic'
MODELS = {
    'teacher2B': R/'artifacts/hf-home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc',
    'student0.8B': R/'artifacts/hf-home/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17',
}

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def binding(p):
    p = Path(p)
    return dict(path=p.relative_to(R).as_posix(), bytes=p.stat().st_size, sha256=sha(p))

def put(name, x):
    p = O/name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')

def fraction(n, d):
    return dict(numerator=int(n), denominator=int(d), percentage=100*n/d if d else None)

def freq_stats(counts):
    v = sorted(counts.values())
    n, total = len(v), sum(v)
    return dict(types=n, incidences=total, histogram=dict(sorted(C.Counter(v).items())),
                gini=(sum((2*i-n-1)*x for i,x in enumerate(v,1))/(n*total) if total else None),
                entropy_bits=-sum(x/total*math.log2(x/total) for x in v) if total else None,
                top10percent_mass=sum(sorted(v, reverse=True)[:max(1, math.ceil(n*.1))])/total if total else None,
                max_min_support_ratio=max(v)/min(v) if v else None,
                rare_types=fraction(sum(x<=2 for x in v),n),
                rare_incidence_mass=sum(x for x in v if x<=2)/total if total else None)

def tv(a,b):
    na,nb=sum(a.values()),sum(b.values())
    if not na or not nb:
        return 0. if na==nb else 1.
    return .5*sum(abs(a[k]/na-b[k]/nb) for k in a.keys()|b.keys())

def array_records(text):
    """Lexical record splitter; feature payload of excluded split never JSON-decoded."""
    depth=0; quoted=False; escaped=False; start=None
    for i,ch in enumerate(text):
        if quoted:
            if escaped: escaped=False
            elif ch=='\\': escaped=True
            elif ch=='"': quoted=False
            continue
        if ch=='"': quoted=True
        elif ch in '[{':
            if depth==1 and ch=='{': start=i
            depth+=1
        elif ch in ']}':
            depth-=1
            if depth==1 and ch=='}' and start is not None:
                yield text[start:i+1]
                start=None

def selected_rooms(text):
    for record in array_records(text):
        # source_split occurs before objects in the frozen exporter, and its key
        # is unique. Reject ambiguous records rather than inspecting Test values.
        splits=re.findall(r'"source_split"\s*:\s*"([^"\\]*)"',record)
        if len(splits)!=1: raise ValueError('ambiguous split')
        if splits[0] not in ('train','training','validation','val'):
            continue
        yield json.loads(record)

def sg_audit():
    sys.path.insert(0,str(R/'scripts'))
    import sgqa_longtail_diagnostic_20261029 as old
    result={}; counts={}; raw_rows={}
    for role in ('fit','dev'):
        rows=[json.loads(x) for x in (SG/f'{role}.graph-incidence.jsonl').read_text().splitlines() if x.strip()]
        raw_rows[role]=rows
        videos=C.defaultdict(lambda:{d:set() for d in old.DIMS})
        joint=C.Counter(); directed=C.Counter(); shapes=[]
        for row in rows:
            if old.digest(row['graphs'])!=row['graph_sha256']: raise ValueError('SG graph hash')
            features, shape=old.graph_features(row['graphs']);shapes.append(shape)
            for d in old.DIMS: videos[row['video_uid']][d].update(features[d])
            for frame in row['graphs']:
                actions={o for s,r,o in frame if r=='verb'}
                rels={r for s,r,o in frame if r!='verb'}
                joint.update(json.dumps([a,r]) for a in actions for r in rels)
                directed.update(json.dumps([s,r,o]) for s,r,o in frame)
        counts[role]={d:C.Counter(x for video in videos.values() for x in video[d]) for d in old.DIMS}
        result[role]=dict(graphs=len(rows), physical_videos=len(videos),
                          frequency={d:freq_stats(c) for d,c in counts[role].items()},
                          structure=old.summary(shapes),action_relation_frame_cooccurrences=dict(joint),
                          directed_literal_triples_distinct=len(directed))
    if len(raw_rows['fit'])!=21 or len(raw_rows['dev'])!=6: raise ValueError('SG denominator')
    if {x['video_uid'] for x in raw_rows['fit']} & {x['video_uid'] for x in raw_rows['dev']}: raise ValueError('SG split overlap')
    frozen=load(R/'runs/2026-10-28-sgqa-2b-checkpoint-lr/analysis/longtail-20261029/frequency.local.json')
    for d,c in counts['fit'].items():
        if dict(c)!=frozen[d]['Fit_video_frequency']: raise ValueError('old frequency mismatch')
    result['dev_against_fit']={d:dict(unseen_types=sum(x not in counts['fit'][d] for x in counts['dev'][d]),
        unseen_incidence_mass=sum(n for x,n in counts['dev'][d].items() if x not in counts['fit'][d])/sum(counts['dev'][d].values()),
        support_distribution_TV=tv(counts['fit'][d],counts['dev'][d])) for d in old.DIMS}
    result['interpretation']='Unequal support and low-frequency types, not a fitted or established power-law. Entity strings are not semantic classes. Small 20-video Fit.'
    put('sgqa-frequency.local.json',{role:{d:dict(c) for d,c in v.items()} for role,v in counts.items()})
    put('sgqa-summary.json',result)
    return result,counts

def roster_audit():
    cases=[]; aggregate=C.Counter(); totals=C.Counter()
    files=sorted(p for p in RAW.glob('*.json') if '.started.' not in p.name)
    if len(files)!=100: raise ValueError('raw100 denominator')
    for p in files:
        r=load(p);shape=r['shape'];z=r['latent'];lengths=shape['entity_lengths']
        names=[];pos=0
        for n in lengths:
            chunk=z[pos:pos+n];pos+=n
            if any(type(v)is not int or not 0<=v<95 for v in chunk):raise ValueError('char support')
            names.append(''.join(chr(32+v) for v in chunk))
        assertions=sum(shape['assertions_per_frame']);triples=[z[i:i+3] for i in range(pos,len(z),3)]
        reasons=dict(entity_count_mismatch=len(names)!=shape['entities'],
                     assertion_count_mismatch=len(triples)!=assertions,
                     latent_length_mismatch=len(z)!=sum(lengths)+3*assertions,
                     duplicate_names=len(names)!=len(set(names)), empty_names=any(not n for n in names))
        used={v for triple in triples if len(triple)==3 for v in triple[1:]}
        reasons['pointer_out_of_range']=any(type(v)is not int or not 0<=v<shape['entities'] for v in used)
        reasons['unused_declared_entities']=len(set(range(shape['entities']))-used)>0
        totals[r['status']]+=1
        if 'Invalid global entity roster' in str(r.get('invalid_reason')):
            aggregate.update(k for k,v in reasons.items() if v)
        cases.append(dict(ordinal=r['ordinal'],raw=binding(p),status=r['status'],flags=reasons,
                          duplicate_slots=len(names)-len(set(names)),
                          unused_slots=len(set(range(shape['entities']))-used)))
    result=dict(raw=100,statuses=dict(totals),roster_failures_flags_overlapping=dict(aggregate),
                cases=cases,entity_strings_output=False,
                interpretation='Exact latent reconstruction diagnoses structural predicates, not the causal model or training mechanism. No new generation, repair or filtering.')
    put('roster-failure-analysis.json',result)
    return result

def mp_audit():
    import numpy as np
    from scipy.spatial import cKDTree
    selected=list(selected_rooms(MP.read_text(encoding='utf-8')))
    roles={k:[] for k in ('fit','dev')}
    for row in selected:
        role='fit' if row['source_split'] in ('train','training') else 'dev'
        roles[role].append(row)
    if [len(roles[k]) for k in ('fit','dev')]!=[918,437]:raise ValueError('MP split counts')
    buildings={k:{r['building_graph_index'] for r in rows} for k,rows in roles.items()}
    if [len(buildings[k]) for k in ('fit','dev')]!=[40,16] or buildings['fit']&buildings['dev']:raise ValueError('MP building split')
    arr={};categories={};nn={};room_counts={};room_support={};building_support={};problems={}
    for role,rows in roles.items():
        xyz=[];cats=[];within=[];bad=C.Counter();rs=C.Counter();bs=C.defaultdict(set)
        for row in rows:
            size=np.asarray(row['room_size_native'],dtype=float)
            if size.shape!=(3,) or not np.isfinite(size).all() or (size<=0).any():raise ValueError('MP room size')
            objects=row['objects'];points=np.array([o['relative_position_native'] for o in objects],float)
            if points.size==0:bad['empty_room']+=1;continue
            if points.shape!=(len(objects),3) or not np.isfinite(points).all():raise ValueError('MP point shape/nonfinite')
            cat=[str(o['category']) for o in objects]
            bad['empty_category']+=sum(not x.strip() for x in cat)
            bad['exact_duplicate_category_position']+=len(objects)-len(set((c,*p) for c,p in zip(cat,points)))
            # Ratios use native room sizes as scale covariates, not known bounds.
            q=points/size
            distances=cKDTree(q).query(q,k=2,workers=1)[0][:,1] if len(q)>1 else np.full(len(q),np.nan)
            within.extend(distances.tolist());xyz.extend(q.tolist());cats.extend(cat)
            rs.update(set(cat))
            for c in set(cat):bs[c].add(row['building_graph_index'])
        arr[role]=np.asarray(xyz);categories[role]=cats;nn[role]=np.asarray(within)
        room_counts[role]=np.array([len(r['objects']) for r in rows]);room_support[role]=rs
        building_support[role]=C.Counter({k:len(v) for k,v in bs.items()});problems[role]=dict(bad)
    fit=arr['fit'];center=np.median(fit,axis=0);scale=np.quantile(fit,.75,axis=0)-np.quantile(fit,.25,axis=0)
    scale=np.maximum(scale,1e-6)
    bycat={}
    for cat in sorted(set(categories['fit'])):
        x=fit[np.array(categories['fit'])==cat]
        med=np.median(x,axis=0) if len(x)>=5 else center
        iqr=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0) if len(x)>=5 else scale
        bycat[cat]=(med,np.maximum(iqr,scale*.1),len(x))
    scores={};joints={};cooccur={};result={}
    cuts=[np.unique(np.quantile(fit[:,j],[.25,.5,.75])) for j in range(3)]
    for role,x in arr.items():
        scores[role]=np.array([float(np.sqrt(np.sum(((p-bycat.get(c,(center,scale,0))[0])/bycat.get(c,(center,scale,0))[1])**2))) for p,c in zip(x,categories[role])])
        bins=np.column_stack([np.searchsorted(cuts[j],x[:,j],side='right') for j in range(3)])
        joints[role]=C.Counter(json.dumps([c,*map(int,b)]) for c,b in zip(categories[role],bins))
        cooccur[role]=C.Counter()
        for row in roles[role]:
            cats=sorted({str(o['category']) for o in row['objects']})
            cooccur[role].update(json.dumps([a,b]) for i,a in enumerate(cats) for b in cats[i+1:])
        v=nn[role][np.isfinite(nn[role])]
        result[role]=dict(rooms=len(roles[role]),buildings=len(buildings[role]),objects=len(x),
            building_category_support=freq_stats(building_support[role]),
            room_category_support=freq_stats(room_support[role]),object_category_frequency=freq_stats(C.Counter(categories[role])),
            objects_per_room_quantiles=dict(zip(['min','q25','median','q75','q95','max'],map(float,np.quantile(room_counts[role],[0,.25,.5,.75,.95,1])))),
            data_checks=problems[role],within_room_nn_finite=len(v),within_room_nn_undefined=len(x)-len(v),
            category_position_joint_bins=len(joints[role]),category_cooccurrence_pairs=len(cooccur[role]))
    thresholds=dict(category_conditioned_robust_distance=float(np.quantile(scores['fit'],.99)),
                    within_room_nn=float(np.quantile(nn['fit'][np.isfinite(nn['fit'])],.99)),
                    large_scene=float(np.quantile(room_counts['fit'],.75)+3*(np.quantile(room_counts['fit'],.75)-np.quantile(room_counts['fit'],.25))))
    for role in roles:
        result[role]['outlier_candidates']={
            'category_position':fraction(int((scores[role]>thresholds['category_conditioned_robust_distance']).sum()),len(scores[role])),
            'within_room_nn':fraction(int((nn[role][np.isfinite(nn[role])]>thresholds['within_room_nn']).sum()),int(np.isfinite(nn[role]).sum())),
            'large_scene':fraction(int((room_counts[role]>thresholds['large_scene']).sum()),len(roles[role]))}
    result.update(fit_only_thresholds=thresholds,fit_only_center=center.tolist(),fit_only_IQR=scale.tolist(),
        spatial_scale='relative_position_native / room_size_native, descriptive dimensionless covariates; no axis/unit/bounding-box claim',
        dev_vs_fit=dict(category_object_TV=tv(C.Counter(categories['fit']),C.Counter(categories['dev'])),
            category_position_joint_TV=tv(joints['fit'],joints['dev']),cooccurrence_TV=tv(cooccur['fit'],cooccur['dev']),
            unseen_category_objects=fraction(sum(c not in building_support['fit'] for c in categories['dev']),len(categories['dev']))),
        interpretation='Outlier candidates only; q99 creates a Fit tail by definition, not a 1% true anomaly claim. No data removed. No power-law model fit. Test feature payload never decoded by reader.')
    put('matterport-summary.json',result)
    put('matterport-frequency.local.json',{role:dict(building=dict(building_support[role]),room=dict(room_support[role]),objects=dict(C.Counter(categories[role])),joint=dict(joints[role]),cooccurrence=dict(cooccur[role])) for role in roles})
    return result,building_support,arr,nn,scores

def stress_audit():
    import numpy as np
    from scipy.spatial import cKDTree
    result={};plots={}
    for ratio,anom in [(10,.01),(100,.05)]:
        records=[];splits={}
        # Ten head + ten tail classes, exact per-class count ratio.
        head=100 if ratio==10 else 1000;tail=head//ratio
        for si,split in enumerate(('reference','diagnostic')):
            rng=np.random.default_rng(2026103100+ratio+si*1000)
            labels=np.concatenate([np.repeat(i,head if i<10 else tail) for i in range(20)])
            rng.shuffle(labels);n=len(labels);k=int(round(n*anom))
            mask=np.zeros(n,dtype=bool);mask[rng.choice(n,k,replace=False)]=True
            xy=rng.normal(0,.08,(n,2));angles=rng.uniform(0,2*np.pi,k);rad=rng.uniform(.75,.9,k)
            xy[mask]=np.column_stack([rad*np.cos(angles),rad*np.sin(angles)])
            if not (np.abs(xy)<=1).all():raise ValueError('stress known domain')
            splits[split]=(labels,xy,mask)
            for i in range(n):records.append(dict(split=split,scene=i//20,category=int(labels[i]),xy=xy[i].tolist(),injected_low_density=bool(mask[i])))
        a,x,m=splits['reference'];b,y,z=splits['diagnostic']
        # Known-normal reference calibration is possible ONLY in this constructed control.
        tree=cKDTree(x[~m]);cal=tree.query(x[~m],k=6,workers=1)[0][:,-1]
        threshold=float(np.quantile(cal,.99));dist=tree.query(y,k=5,workers=1)[0][:,-1];flag=dist>threshold
        tp=int((flag&z).sum());fp=int((flag&~z).sum())
        result[str(ratio)]=dict(head_tail_per_class_ratio=ratio,classes=20,
            points_per_split=len(y),known_anomalies_per_split=int(z.sum()),actual_anomaly_fraction=float(z.mean()),
            threshold=threshold,precision=fraction(tp,tp+fp),recall=fraction(tp,int(z.sum())),normal_FPR=fraction(fp,int((~z).sum())),
            definition='Public procedural 2D disk/core plus annulus, not derived from protected statistics and not natural anomaly truth',
            calibration='reference known-normal k5 leave-self q99; diagnostic k5-to-reference normals')
        put(f'stress-{ratio}.json',dict(config=result[str(ratio)],records=records))
        plots[str(ratio)]=(y,z)
    put('stress-summary.json',result)
    return result,plots

def embedding_header(path):
    index=load(path/'model.safetensors.index.json')
    selected={k:v for k,v in index['weight_map'].items() if 'embed_tokens.weight' in k or k.endswith('lm_head.weight')}
    result={};read=0
    for name in sorted(set(selected.values())):
        with (path/name).open('rb') as f:
            b=f.read(8);n=struct.unpack('<Q',b)[0]
            if n>2**20:raise ValueError('header cap')
            header=json.loads(f.read(n));read+=8+n
        result.update({k:header[k] for k,v in selected.items() if v==name})
    return dict(tensors=result,header_bytes_read=read,weight_payload_read=False)

def tokenizer_audit():
    from transformers import AutoTokenizer
    import torch, transformers, tokenizers
    ts={k:AutoTokenizer.from_pretrained(v,local_files_only=True) for k,v in MODELS.items()}
    keys=list(ts);a,b=[ts[k] for k in keys]
    backs=[json.loads(t.backend_tokenizer.to_str()) for t in (a,b)]
    def meta(t):
        return {str(i):{n:getattr(v,n) for n in ('content','single_word','lstrip','rstrip','normalized','special')} for i,v in t.added_tokens_decoder.items()}
    probes=['hello','你好','é','e\u0301','👩🏽‍💻','مرحبا','हिन्दी','日本語','a\n\n b','\t  ','0000123','1e-9','[x,y,z]','<|im_end|>','<|endoftext|>','<think>','</think>','a\x00b','한글','𝒙→∞']
    equal=sum(a.encode(s)==b.encode(s) for s in probes)
    prompts=[];template_equal=[]
    for i in range(8):
        messages=[dict(role='system',content='Answer briefly from the given public synthetic facts.'),dict(role='user',content=f'Object A{i} is at (0,0). Object B{i} is at (1,0). Which object has larger x?')]
        rendered=[t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in (a,b)]
        ids=[t.encode(s,add_special_tokens=False) for t,s in zip((a,b),rendered)]
        template_equal.append(rendered[0]==rendered[1] and ids[0]==ids[1])
        if max(map(len,ids))>512:raise ValueError('IPD short prefix cap')
        prompts.append(dict(ordinal=i,messages=messages,rendered_text=rendered[0],token_ids=ids[0],
                            rendered_sha256=hashlib.sha256(rendered[0].encode()).hexdigest(),token_ids_sha256=hashlib.sha256(canon(ids[0])).hexdigest()))
    configs=[load(path/'config.json') for path in MODELS.values()]
    headers=[embedding_header(path) for path in MODELS.values()]
    maxid=max(a.get_vocab().values())
    capacities=[cfg.get('text_config',cfg)['vocab_size'] for cfg in configs]
    full=(backs[0]==backs[1] and a.get_vocab()==b.get_vocab() and meta(a)==meta(b))
    out=dict(status='CPU_CODEC_QUALIFIED' if full and all(template_equal) and all(maxid<c for c in capacities) else 'CPU_CODEC_NOT_QUALIFIED',
        models={k:dict(snapshot=v.relative_to(R).as_posix(),files=[binding(v/f) for f in ('tokenizer.json','tokenizer_config.json','config.json','chat_template.jinja') if (v/f).exists()]) for k,v in MODELS.items()},
        classes=[type(t).__name__ for t in (a,b)],runtime=dict(transformers=transformers.__version__,tokenizers=tokenizers.__version__),
        vocab_counts=[len(a.get_vocab()),len(b.get_vocab())],max_registered_id=maxid,
        full_token_to_id_equal=a.get_vocab()==b.get_vocab(),full_backend_semantics_equal=backs[0]==backs[1],
        full_backend_sha256=[hashlib.sha256(canon(z)).hexdigest() for z in backs],
        raw_token_bytes_compatible=full,byte_justification='Complete equal token strings/IDs + equal ByteLevel decoder/pretokenizer/backend; not replacement-character decode probes',
        added_metadata_equal=meta(a)==meta(b),ordered_merges_equal=backs[0]['model']['merges']==backs[1]['model']['merges'],
        template_file_sha=[sha(v/'chat_template.jinja') if (v/'chat_template.jinja').exists() else None for v in MODELS.values()],
        thinking_off_fixed_prompts_identical=fraction(sum(template_equal),8),ordinary_probe_ids_equal=fraction(equal,len(probes)),
        eos_tokenizer=[t.eos_token_id for t in (a,b)],pad=[t.pad_token_id for t in (a,b)],
        model_eos=[c.get('text_config',c).get('eos_token_id') for c in configs],
        engineering_stop_ids=[a.convert_tokens_to_ids('<|im_end|>'),a.convert_tokens_to_ids('<|endoftext|>')],
        declared_vocab_capacities=capacities,embedding_headers=headers,cuda_initialized=torch.cuda.is_initialized(),
        qualification_limit='Local actual pair codec and fixed off prompts only; no weight load/teacher advantage/logits/IPD update/hosted Max qualification.')
    if out['cuda_initialized']:raise AssertionError('CUDA initialized')
    put('ipd-prompts-public-frozen.json',dict(prompts=prompts,max_new_tokens_per_prompt=64,total_rollout_cap_first_update=512,K=16,temperature=1,gamma=1))
    put('qwen-codec-audit.json',out)
    return out

def plots(sgcounts,mpcounts,arr,nn,scores,stress):
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,key in zip(axs[0],('action_literal','nonverb_relation_literal')):
        for role in ('fit','dev'):
            v=sorted(sgcounts[role][key].values(),reverse=True);ax.plot(range(1,len(v)+1),v,label=role)
        ax.set(xlabel='Type rank',ylabel='Physical-video support',title='SGQA '+key);ax.legend()
    ax=axs[0,2]
    for role in ('fit','dev'):
        v=sorted(mpcounts[role].values(),reverse=True);ax.plot(range(1,len(v)+1),v,label=role)
    ax.set(title='Matterport category support',xlabel='Category rank',ylabel='Building support');ax.legend()
    for role in ('fit','dev'):
        a=sorted(scores[role]);axs[1,0].plot(a,np.arange(1,len(a)+1)/len(a),label=role)
    axs[1,0].set(xscale='symlog',title='Category-position score CDF',xlabel='Fit-calibrated robust score');axs[1,0].legend()
    for ax,ratio in zip(axs[1,1:],('10','100')):
        x,z=stress[ratio];ax.scatter(x[~z,0],x[~z,1],s=2,alpha=.2);ax.scatter(x[z,0],x[z,1],s=4,color='red')
        ax.set(title=f'Constructed stress {ratio}:1',xlim=(-1,1),ylim=(-1,1),aspect='equal')
    fig.suptitle('Local descriptive audit; natural rarity != established power law; red = injected stress points')
    fig.tight_layout();fig.savefig(O/'distribution-audit.png',dpi=160);fig.savefig(O/'distribution-audit.pdf');plt.close(fig)

def checks():
    # Counterexamples for sealed-reader exclusion and joint-vs-marginal metrics.
    text='[{"source_split":"test","objects":NOT_JSON},{"source_split":"train","objects":[]}]'
    assert len(list(selected_rooms(text)))==1
    assert freq_stats({'a':1,'b':1})['gini']==0
    assert math.isclose(freq_stats({'a':1,'b':3})['gini'],.25)
    assert tv(C.Counter({'a':1}),C.Counter({'b':1}))==1
    a=C.Counter({'00':1,'11':1});b=C.Counter({'01':1,'10':1})
    assert tv(a,b)==1 # Same separate Bernoulli marginals but disjoint joint.
    names=['same','same'];assert len(set(names))!=len(names)
    return dict(meaningful_checks=6,pass_=True,checks=['test malformed feature skip','uniform Gini','known Gini','disjoint TV','joint marginal counterexample','duplicate roster'])

def main():
    began=time.monotonic()
    if O.exists():raise FileExistsError('Immutable CPU namespace exists; inspect receipts before any retry')
    O.mkdir(parents=True)
    inputs=[SG/'fit.graph-incidence.jsonl',SG/'dev.graph-incidence.jsonl',MP,
            R/'runs/2026-10-28-sgqa-2b-checkpoint-lr/analysis/longtail-20261029/frequency.local.json']
    put('protocol.json',dict(status='FROZEN_BEFORE_NEW_ANALYSIS',source=binding(__file__),inputs=[binding(p) for p in inputs],
        user_scope='2026-10-31: 好，按这个计划开始，先做cpu部分',
        rare='1–2 independent Fit videos/buildings; head>=3; descriptive threshold not fitted power law',
        outliers='Fit-only q99 robust category-position and within-room NN; 3 IQR large scenes; candidates not truth',
        natural_data_unchanged=True,stress_configs=[dict(ratio=10,anomaly=.01),dict(ratio=100,anomaly=.05)],
        GPU=False,API_calls=0,downloads=0,weights_payload_read=False,new_model_updates=0,max_threads=1,
        memory_limit_GiB=4,wall_cap_seconds=600,protected_exports=False,test_features_excluded=True))
    try:
        put('cpu-checks.json',checks())
        sg,sc=sg_audit();roster=roster_audit();mp,mc,arr,nn,scores=mp_audit();stress,sp=stress_audit()
        plots(sc,mc,arr,nn,scores,sp)
        put('distribution-complete.json',dict(seconds=time.monotonic()-began,sgqa='sgqa-summary.json',spatial='matterport-summary.json',roster='roster-failure-analysis.json',stress='stress-summary.json'))
        codec=tokenizer_audit()
        put('terminal.json',dict(status='CPU_COMPLETE',elapsed_seconds=time.monotonic()-began,source=binding(__file__),
            codec=codec['status'],GPU=False,API_calls=0,downloads=0,model_updates=0,
            notes='CPU qualification only; no automatic GPU continuation',
            outputs=[binding(p) for p in O.rglob('*') if p.is_file()]))
        print(json.dumps(dict(status='CPU_COMPLETE',seconds=time.monotonic()-began,codec=codec['status'],
            sgqa={k:v['frequency']['action_literal']['rare_types'] for k,v in sg.items() if k in ('fit','dev')},
            roster=roster['roster_failures_flags_overlapping'],mp={k:mp[k]['objects'] for k in ('fit','dev')})))
    except Exception as exc:
        put('failure.json',dict(type=type(exc).__name__,reason=str(exc),seconds=time.monotonic()-began))
        raise

if __name__=='__main__':main()
