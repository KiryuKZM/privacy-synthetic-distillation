"""Freeze matched real/old/new inputs and untouched native Dev primary eval."""
from pathlib import Path
from decimal import Decimal
import json,hashlib,sys,itertools,random,collections,os
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_joint_generator_repair_20261107 as gen
from ust_joint_generator_residual_20261107 import ResidualRepair
from ust_joint_generator_conditional_calibration_20261107 import Calibrated
import audit_ust100_input_labels_20261024 as reader
import ust_dev_semantic_transforms_20261106 as transform
import ust_position_balanced_training_prepare_20261106 as ref
import ust_option_cycle_weak_data_v2_20261106 as weak
import ust_common_descriptive_20261106 as cm
import ust_tail_outlier_utility_20261104 as tail
from privacy_synth.ust_task_contract import NATIVE_UNDERSTANDING_DESCRIPTION as DESC,NATIVE_UNDERSTANDING_SCHEMA as SCHEMA,NATIVE_UNDERSTANDING_NOTE as NOTE,certify_native_ranking_reference
O=gen.O/'student-utility-v1';D=gen.base.D;B=R/'runs/2026-10-23-llm-task-sweep/preparation/text-qa/USTBench-POI-processQA';FIELDS=gen.F;LETTERS='ABCD';PERMS=list(itertools.permutations(range(4)));pins={}
def load(p,jsonl=False):
 p=Path(p);s=p.read_bytes();pins[str(p.relative_to(R))]=hashlib.sha256(s).hexdigest();return [json.loads(x) for x in s.splitlines() if x.strip()] if jsonl else json.loads(s)
def save(n,x):gen.save(O/n,x)
def key(*x):return hashlib.sha256(':'.join(map(str,x)).encode()).hexdigest()
def vals(q):
 data=reader.parse_data(reader.sections(q)['Data']);return [[None if data[i][f]['number']=='NaN' else float(data[i][f]['number']) for f in FIELDS] for i in range(1,5)]
def options(obs,metric,slot):
 j=FIELDS.index(metric);v=[Decimal('Infinity') if row[j] is None else Decimal(str(row[j])) for row in obs]
 good=[p for p in PERMS if all(v[a]>=v[b] for a,b in zip(p,p[1:]))];bad=[p for p in PERMS if p not in good]
 good=sorted(good,key=lambda p:key('good',slot,p));bad=sorted(bad,key=lambda p:key('bad',slot,p));seen=set();pick=[]
 for p in bad:
  sig=tuple(v[i] for i in p)
  if sig not in seen:seen.add(sig);pick.append(p)
  if len(pick)==3:break
 if len(pick)<3:return None
 answer=LETTERS[slot%4];result={};it=iter(pick)
 for a in LETTERS:result[a]=good[0] if a==answer else next(it)
 return result,answer

def render(obs,metric,opts):
 rows=[]
 for i,v in enumerate(obs,1):
  body=[]
  for j,(name,a) in enumerate(zip(FIELDS,v)):
   value='NaN' if a is None else format(a,'.3f' if j==0 else '.2f');unit='m' if j==2 else 'h' if j>=3 else '';body.append('- '+name+': '+value+unit)
  rows.append('location '+str(i)+':\n'+'\n'.join(body))
 head=next(k for k,v in ref.UNDER.items() if v==metric);question=head+'\n\n'+'\n'.join(a+'. '+' > '.join(str(i+1) for i in opts[a]) for a in LETTERS)
 return transform.render_sections({'Task Description':DESC,'Data Schema':SCHEMA,'Data':'\n\n'.join(rows),'Question':question,'Note':NOTE})
def main():
 plan=load(O/'study-plan.json');gp=load(gen.O/'protocol.json');fit=load(R/gp['fit']['path']);repair=Calibrated.__new__(Calibrated);repair.model=ResidualRepair(fit,'conditional_charge_and_wait');repair.tables=[(np.array(x['log_probe']),np.array(x['Fit_positive_log_inverse'])) for x in load(gen.O/'conditional_calibrated/tables.local.json')]
 fresh,raw=repair.sample(plan['new_pool_n'],plan['new_pool_seed']);save('new-generator-pool1000.local.json',fresh)
 oldraw=load(D/'full-schema-shared-calibrated-v2/raw-1000.local.jsonl',True);scenes=load(D/'full-schema-mainline-v3/scenes.local.json');allowed={r['source_raw_index'] for r in scenes if r['raw_generator']=='full-schema-shared-calibrated-v2' and '/train/' in r['id']}
 old=[r for r in oldraw if r['index'] in allowed];assert old and all(not r['failures'] for r in old)
 source=[r for r in load(R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1/real-reference-ce/training200.local.json') if r['metadata']['ability']=='st_understanding'];assert len(source)==68
 metrics=[certify_native_ranking_reference(r['question'],r['target'][8])['metric'] for r in source];assert collections.Counter(metrics)==dict(distance=42,charging_time=14,waiting_time=12)
 dataset={};allvariants={};checks={};allscan={}
 for arm,pool in [('real68',None),('old_shared_calibrated68',old),('conditional_calibrated68',fresh)]:
  source_rows=[];variants=[];proof=[];scans=[];used=set();order=sorted(pool,key=lambda r:key('pool',plan['new_pool_seed'],r['index'])) if pool is not None else None
  for slot,metric in enumerate(metrics):
   if pool is None:
    obs=vals(source[slot]['question']);opt=options(obs,metric,slot);sid=source[slot]['id'];assert opt is not None
   else:
    for row in order:
     if row['index'] in used:continue
     if row['failures']:scans.append(dict(slot=slot,index=row['index'],reason='structural_failure'));continue
     opt=options(row['observations'],metric,slot)
     if opt is None:scans.append(dict(slot=slot,index=row['index'],reason='insufficient_distinct_incorrect_orderings'));continue
     obs=row['observations'];used.add(row['index']);sid=arm+'/raw/'+str(row['index']);scans.append(dict(slot=slot,index=row['index'],reason='selected'));break
    else:raise RuntimeError('Fixed pool exhausted; no new generation/refill')
   opts,answer=opt;question=render(obs,metric,opts);cert=certify_native_ranking_reference(question,answer);assert cert['trusted_gold'];source_rows.append(dict(id=sid,question=question,metadata={'ability':'st_understanding','metric':metric},target='<Answer>'+answer+'</Answer>',source_slot=slot,original_source_id=source[slot]['id'] if pool is None else None))
   for offset in range(4):
    tr=transform.transform(question,{i:i for i in range(1,5)},LETTERS[offset:]+LETTERS[:offset]);ans=tr['old_to_new_letter'][answer];q=weak.clarify_native(tr['question']);cc=weak.certify_explicit_weak_reference(q,ans);ind=weak.independent_numeric(q);assert cc['trusted_gold'] and ind['compatible_answers']==[ans]
    vid=arm+f'/{slot:03d}/{offset}';v=dict(id=vid,source_id=sid,source_slot=slot,variant_index=offset,question=q,answer=ans,target='<Answer>'+ans+'</Answer>',metadata={'ability':'st_understanding','metric':metric},old_to_new_letter=tr['old_to_new_letter']);variants.append(v);proof.append(dict(id=vid,certificate=cc,independent=ind))
  assert len(variants)==272 and len(source_rows)==68
  slots=list(itertools.product(range(68),range(4)));random.Random(plan['source_shuffling_seed']).shuffle(slots);ids=[arm+f'/{i:03d}/{j}' for i,j in slots];schedule=[]
  for epoch in range(2):
   for bi in range(68):schedule.append(dict(step=epoch*68+bi+1,epoch=epoch+1,batch_index=bi,variant_ids=ids[bi*4:bi*4+4]))
  save(arm+'/source68.local.json',source_rows);save(arm+'/variants.local.json',variants);save(arm+'/schedule.local.json',schedule);save(arm+'/qualification.local.json',proof);save(arm+'/selection-log.local.json',scans)
  dataset[arm]=dict(source_cases=68,unique_generated_scene_indexes=len(used) if pool is not None else None,metric_counts=dict(collections.Counter(metrics)),variant_n=272,updates=136,source_exposures=8,has_displayed_ties=sum(proof[4*i]['independent']['has_displayed_ties'] for i in range(68)),has_target_NaN=sum(proof[4*i]['independent']['has_target_NaN'] for i in range(68)),rejected_scan_events=sum(r['reason']!='selected' for r in scans));allvariants[arm]=variants;checks[arm]=proof;allscan[arm]=scans
 original=load(B/'Dev.local.jsonl',True);dev=sorted([r for r in original if r['ability']=='st_understanding'],key=lambda r:r['id']);assert len(dev)==155
 role=load(B/'roles.json');assert all(r['id'] in role['roles']['Dev']['members'] for r in dev);assert not {r['id'] for r in dev}&{r['id'] for r in source}
 fitroles=load(R/'runs/2026-10-24-ust-sg-onpolicy/data/USTBench.membership.json');assert not {r['group'] for r in dev}&{r['group'] for r in fitroles['train_members']}
 thresholds=load(D/'tail-outlier-utility-v2/thresholds.local.json');queries=[];answers={};qual=[]
 for r in dev:
  q=r['question'];cert=certify_native_ranking_reference(q,r['gold']);loc,order=cm.derangements('new-Dev155:'+r['id']);tr=transform.transform(q,loc,order);nc=certify_native_ranking_reference(tr['question'],tr['old_to_new_letter'][r['gold']]);assert set(nc['compatible_answers'])=={tr['old_to_new_letter'][a] for a in cert['compatible_answers']};ff,_=tail.flags([np.nan if v is None else v for row in vals(q) for v in row],thresholds)
  inverse=transform.transform(tr['question'],{v:k for k,v in loc.items()},[tr['old_to_new_letter'][a] for a in LETTERS]);assert inverse['question']==transform.canonical(q)
  qual.append(dict(id=r['id'],certificate=cert,transform_checks=tr['checks'],flags=ff,inverse_restored=True))
  for variant,question,mapping,compatible in [('original',q,{a:a for a in LETTERS},cert['compatible_answers']),('joint',tr['question'],tr['old_to_new_letter'],nc['compatible_answers'])]:
   qid=r['id']+'::'+variant;queries.append(dict(id=qid,source_id=r['id'],group=r['group'],variant=variant,question=question,metric=cert['metric'],metadata={'ability':'st_understanding'},flags=ff,old_to_new_letter=mapping));answers[qid]=dict(answer=mapping[r['gold']],compatible=compatible)
 for arm,variants in allvariants.items():
  own=[]
  for r in variants:
   if r['variant_index']!=0:continue
   own.append(dict(id=r['id'],source_id=r['source_id'],group=r['source_id'],question=r['question'],metadata=r['metadata'],metric=r['metadata']['metric'],variant='own',flags={},old_to_new_letter={a:a for a in LETTERS}));answers[r['id']]=dict(answer=r['answer'],compatible=[r['answer']])
  save(arm+'/queries.local.json',queries+own)
 save('freshINIT/queries.local.json',queries);save('answers.local.json',answers);save('Dev155-qualification.local.json',qual);save('fixed8.local.json',load(R/'runs/2026-11-06-ust-autonomous-repair-v1/position-training-v2/fixed8.local.json'))
 save('data-summary.json',dict(status='ALL_DATA_FROZEN',arms=dataset,Dev=dict(source_cases=155,numeric_groups=len({r['group'] for r in dev}),ambiguous=sum(len(q['certificate']['compatible_answers'])!=1 for q in qual),metric_counts=dict(collections.Counter(q['certificate']['metric'] for q in qual)),flag_counts={f:sum(q['flags'][f] for q in qual) for f in qual[0]['flags']}),source_bindings=pins,source_script_sha256=gen.digest(__file__),API=0,GPU=0))
 print(json.dumps(load(O/'data-summary.json'),ensure_ascii=False)[:2500])
if __name__=='__main__':main()
