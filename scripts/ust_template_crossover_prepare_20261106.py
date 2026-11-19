"""CPU-only, immutable training/RepairDev template crossover; no model or network.

Static template bundle = Data Schema + Note + optional N/A Analysis. Both
versions use identical whitespace normalization; complete Data and Question
sections are byte-identical. References and displayed numbers do not change.
"""
from pathlib import Path
from decimal import Decimal
import collections, hashlib, json, re, sys

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R/'scripts'))
import ust_dev_semantic_transforms_20261106 as t
O = R/'runs/2026-11-06-ust-autonomous-repair-v1/template-crossover'
PAIR = R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1'
DEV = R/'runs/2026-10-25-qwen-transfer-repair/data/USTBench.repair-dev32.requests.jsonl'

def load(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p); return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(name,value):
    with (O/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def split(text):
    # Independent simple section reader used for the evidence checks.
    bits=re.split(r'^## ([^\n]+)\n',text,flags=re.M)
    assert len(bits)%2==1 and not bits[0].strip()
    return {bits[i]:bits[i+1].strip() for i in range(1,len(bits),2)}
def metric(text):
    q=split(text)['Question']
    return 'distance' if 'average distance' in q else 'waiting_time' if 'average waiting time' in q else 'charging_time' if 'average charging time' in q else None

def independent_solve(text):
    s=split(text); field=metric(text); assert field
    data=re.split(r'(?im)^location ([1-4]):\s*\n',s['Data']);assert len(data)==9
    values={}
    for i in range(1,len(data),2):
        m=re.search(r'^- '+field+r':\s*(\S+)\s*$',data[i+1],re.M);assert m
        raw=re.sub(r'[mh]$','',m.group(1))
        if raw.lower()=='nan':return None
        values[int(data[i])]=Decimal(raw);assert values[int(data[i])].is_finite()
    opts=re.findall(r'^([ABCD])\.\s*([1-4](?:\s*>\s*[1-4]){3})\s*$',s['Question'],re.M)
    assert len(opts)==4 and set(values)=={1,2,3,4}
    good=[]
    for letter,body in opts:
        order=list(map(int,re.findall(r'[1-4]',body)));assert set(order)==set(values)
        if all(values[x]>=values[y] for x,y in zip(order,order[1:])):good.append(letter)
    return good

def swapped(question,template):
    original=t.sections(question)
    assert set(original) <= {'Task Description','Data Schema','Data','Analysis','Question','Note'}
    assert original['Task Description']==template['Task Description']
    out={}
    for name in ['Task Description','Data Schema','Data','Analysis','Question','Note']:
        if name in ('Data','Question'):out[name]=original[name]
        elif name in template:out[name]=template[name]
    return t.render_sections(out)

def main():
    O.mkdir(parents=True,exist_ok=False)
    real_path=PAIR/'real-reference-ce/training200.local.json';synth_path=PAIR/'synthetic-reference-ce/training200.local.json'
    real=[r for r in load(real_path) if r['metadata']['ability']=='st_understanding']
    synth=[r for r in load(synth_path) if r['metadata']['ability']=='st_understanding']
    dev=[json.loads(l) for l in DEV.open()];dev=[r for r in dev if r['metadata']['ability']=='st_understanding']
    dev_gold_path=DEV.with_name('USTBench.repair-dev32.gold.local.json');dev_gold=load(dev_gold_path)
    assert len(real)==len(synth)==68 and len(dev)==12
    real_t=t.sections(real[0]['question']);synth_t=t.sections(synth[0]['question'])
    assert real_t['Task Description']==synth_t['Task Description'] and synth_t['Analysis']=='N/A' and 'Analysis' not in real_t
    for pool,template in [(real,real_t),(dev,real_t),(synth,synth_t)]:
        for row in pool:
            s=t.sections(row['question'])
            for key in ['Task Description','Data Schema','Note','Analysis']:assert s.get(key)==template.get(key)
    queries=[];answers={};checks=[];exclusions=[]
    for cohort,pool in [('synthetic_train_understanding',synth),('RepairDev_understanding',dev)]:
        for row in pool:
            solved=independent_solve(row['question'])
            target=re.fullmatch(r'<Answer>([ABCD])</Answer>',row['target']).group(1) if cohort.startswith('synthetic') else dev_gold[row['id']]
            if solved is None:
                exclusions.append(dict(cohort=cohort,source_id=row['id'],reason='Nonfinite requested metric; this controlled finite arithmetic diagnostic does not define a NaN ordering',original_target=target))
                continue
            assert solved==[target],(row['id'],solved,target)
            before=split(t.canonical(row['question']))
            versions={name:swapped(row['question'],template) for name,template in [('real_template',real_t),('learned_template',synth_t)]}
            for name,text in versions.items():
                after=split(text)
                assert after['Data']==before['Data'] and after['Question']==before['Question']
                assert after['Task Description']==before['Task Description']
                assert independent_solve(text)==[target]
                qid=row['id']+'::template-crossover::'+name
                queries.append(dict(id=qid,source_id=row['id'],cohort=cohort,variant=name,question=text,metadata=dict(ability='st_understanding',goal=metric(text))))
                answers[qid]=dict(answer=target,source='existing training reference' if cohort.startswith('synthetic') else 'published RepairDev gold',independent_displayed_Decimal_verified=True)
            assert versions['real_template']!=versions['learned_template']
            assert versions['learned_template' if cohort.startswith('synthetic') else 'real_template']==t.canonical(row['question'])
            checks.append(dict(source_id=row['id'],cohort=cohort,Data_equal=True,Question_including_options_equal=True,Task_Description_equal=True,independent_answers_equal=True,target=target))
    assert len(queries)==158 and len(checks)==79 and len(exclusions)==1
    fixed=load(PAIR/'fixed8.local.json');assert len(fixed)==17
    save('queries.local.json',queries);save('answers.local.json',answers);save('fixed8.local.json',fixed)
    save('qualification.local.json',dict(status='PASS',paired_cases=79,variant_queries=158,checks=checks,excluded=exclusions,
        meaning_preserving_bundle='Data Schema travel_time line absent/present; Note output instruction; Analysis absent/N-A. This bundle is not an isolated cause claim.',
        no_GPU=True,no_training=True,no_API=True,no_Common_Audit_Final_Confirm_read=True))
    save('templates.local.json',dict(real={k:v for k,v in real_t.items() if k not in ('Data','Question')},learned={k:v for k,v in synth_t.items() if k not in ('Data','Question')},
        extraction=dict(real_source_id=real[0]['id'],learned_source_id=synth[0]['id'],all68_template_identity_verified=True)))
    paths=[Path(__file__),real_path,synth_path,DEV,dev_gold_path,PAIR/'fixed8.local.json',R/'scripts/ust_dev_semantic_transforms_20261106.py']
    save('protocol.json',dict(status='CPU_DATA_FROZEN_NO_MODEL_EXECUTION',source=bind(__file__),bindings=[bind(p) for p in paths],
        output_bindings=[bind(O/n) for n in ['queries.local.json','answers.local.json','fixed8.local.json','qualification.local.json','templates.local.json']],
        included=dict(synthetic_train_understanding=67,RepairDev_understanding=12),queries=158,
        intervention='static template bundle only; shared whitespace canonicalization; Data/Question/options byte-identical',
        inference='Not executed by this script. Root runner must separately freeze checkpoint, decoder and equal caps for both versions.',
        interpretation='Training resubstitution plus exposed Dev diagnostics; not independent generalization; no one-factor attribution within static template bundle',
        labels='Original targets retained, all79 verified against independent finite Decimal sorting, one NaN requested-metric train row excluded explicitly'))
    print(json.dumps(dict(status='PASS',queries=158,pairs=79,excluded=1,output=str(O))))
if __name__=='__main__':main()
