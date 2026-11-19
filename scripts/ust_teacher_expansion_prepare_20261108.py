"""Prepare a larger paired comparison locally; new original egress is pending."""
from pathlib import Path
import collections,hashlib,json,sys,time
import numpy as np
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import generator_discovery_ust_20261026 as g
import ust_missingness_data_20261107 as d
import ust_missingness_delivery_20261107 as audit
import ust_teacher_api_compare_20261108 as prior
S=R/'runs/2026-11-08-ust-teacher-expanded-v1';M=d.S;B=g.BASE
load=prior.load;save=prior.save;pin=prior.pin
THRESHOLD=R/'runs/2026-11-05-ust-environment-generator-ipd-v1/tail-outlier-utility-v2/thresholds.local.json'

def h(*parts):return hashlib.sha256(('teacherexpanded20261108:'+':'.join(map(str,parts))).encode()).hexdigest()
def numbers(q):
 obs,_=audit.sections(q);return [[None if row[f] is None else float(row[f]) for f in audit.FIELDS] for _,row in sorted(obs.items())]
def flags(obs,metric,threshold):
 j=audit.FIELDS.index(metric);v=[x[j] for x in obs];lo,hi=threshold[j]
 return (metric,any(x is None for x in v),len(set(v))<4,any(x is not None and x>0 and (x<lo or x>hi) for x in v))
def assignment(slots,candidates,eligible,cost):
 mat=np.full((len(slots),len(candidates)),1e9)
 for i,s in enumerate(slots):
  for j,c in enumerate(candidates):
   if eligible(s,c):mat[i,j]=cost(s,c)
 a,b=linear_sum_assignment(mat);assert len(a)==len(slots) and all(mat[i,j]<1e8 for i,j in zip(a,b)), 'Fixed matching infeasible'
 return {int(i):int(j) for i,j in zip(a,b)}
def main():
 assert not S.exists();S.mkdir();roles=load(B/'roles.json')['roles'];threshold=load(THRESHOLD)['positive_field_q01_q99']
 forbidden=set().union(*(set(v['groups']) for k,v in roles.items() if k!='Screen'))
 common=[g.picked(l,{'id','group'}) for l in (R/'runs/2026-10-24-ust100-sg60-modes/data-freeze/USTBench.requests.jsonl').open()];forbidden.update(r['group'] for r in common)
 # Other recorded train, validation, repair, teacher and generator memberships.
 files=[R/'runs/2026-10-24-ust-sg-onpolicy/data/USTBench.membership.json',R/'runs/2026-10-25-qwen-transfer-repair/data/USTBench.repair-membership.json',R/'runs/2026-10-26-generator-discovery/branches/UST/cpu-preparation-v2/roles.json']
 for p in files:
  x=load(p);rs=x['train_members']+x['validation_members'] if 'train_members' in x else [r for v in x['roles'].values() for r in (v['selected'] if isinstance(v,dict) else v)];forbidden.update(r['group'] for r in rs)
 candidates=[];excluded=[]
 for line in (B/'Screen.local.jsonl').open():
  meta=g.picked(line,{'id','group','ability'})
  if meta['ability']!='st_understanding' or meta['group'] in forbidden:continue
  r=json.loads(line);cert=d.old.certify_native_ranking_reference(r['question'],r['gold'])
  if not cert['trusted_gold']:excluded.append(dict(id=r['id'],reason='Original reference not unique under visible weak-order interpretation'));continue
  obs=numbers(r['question']);f=flags(obs,cert['metric'],threshold);candidates.append(dict(id=r['id'],group=r['group'],question=r['question'],answer=r['gold'],metric=cert['metric'],obs=obs,stratum=f))
 bymetric={m:[r for r in candidates if r['metric']==m] for m in ('distance','charging_time','waiting_time')}
 # Keep available missing-target coverage (at most three per time goal), then
 # deterministic assignment fills30 per goal without repeated numeric groups.
 chosen=[]
 for metric in ('charging_time','waiting_time'):
  for r in sorted([r for r in bymetric[metric] if r['stratum'][1]],key=lambda r:h('realmissing',r['id'])):
   if sum(x['metric']==metric for x in chosen)>=3:break
   if r['group'] not in {x['group'] for x in chosen}:chosen.append(r)
 used={r['group'] for r in chosen};groups=sorted({r['group'] for r in candidates}-used);lookup={}
 for r in sorted(candidates,key=lambda r:h('realrepresentative',r['id'])):lookup.setdefault((r['metric'],r['group']),r)
 slots=[(m,i) for m in ('distance','charging_time','waiting_time') for i in range(30-sum(x['metric']==m for x in chosen))]
 match=assignment(slots,groups,lambda s,c:(s[0],c) in lookup,lambda s,c:int(h('realassign',*s,c)[:12],16)/16**12)
 chosen += [lookup[(slots[i][0],groups[match[i]])] for i in range(len(slots))];chosen.sort(key=lambda r:h('realpresentation',r['id']))
 assert len(chosen)==len({r['group'] for r in chosen})==len({d.sig(r['obs']) for r in chosen})==90
 assert not ({r['group'] for r in chosen}&forbidden)
 # Reuse existing fixed-generator draws; no model fitting or resampling.
 poolpath=M/'data/fallback-fixed-pool2000.local.json';pool=load(poolpath);assert len(pool)==2000 and all(not r['failures'] for r in pool)
 excluded_numeric={d.sig(r['obs']) for name in d.ARMS for r in load(M/'data'/name/'source68.local.json')}
 for cohort in ('Select','Confirm'):excluded_numeric.update(d.sig(numbers(r['question'])) for r in load(M/'data'/f'{cohort}-queries.local.json'))
 excluded_numeric.update(d.sig(r['obs']) for r in chosen)
 eligible=[r for r in pool if d.sig(r['observations']) not in excluded_numeric];options={};profiles={}
 for row in eligible:
  for m in ('distance','charging_time','waiting_time'):
   opts=d.old.options(row['observations'],m,row['index'])
   if opts is not None:options[(row['index'],m)]=opts;profiles[(row['index'],m)]=flags(row['observations'],m,threshold)
 match=assignment(chosen,eligible,lambda r,c:profiles.get((c['index'],r['metric']))==tuple(r['stratum']),lambda r,c:int(h('syntheticassign',r['id'],c['index'])[:12],16)/16**12)
 generated=[]
 for i,r in enumerate(chosen):
  raw=eligible[match[i]];q,a,_=d.native(raw['observations'],r['metric'],raw['index'],options[(raw['index'],r['metric'])]);d.certify(q,a)
  generated.append(dict(id='fixedpool734/'+str(raw['index']),group='fixedpool734/'+str(raw['index']),question=q,answer=a,metric=r['metric'],obs=raw['observations'],stratum=r['stratum'],paired_real_id=r['id']))
 assert len({d.sig(x['obs']) for x in generated})==90
 queries=[];answers={};maps={};sysmsg='Solve the supplied urban spatiotemporal task from its data, objective and feedback. Follow the stated output format.'
 for index,(real,synth) in enumerate(zip(chosen,generated)):
  for cohort,item in [('public_original90_PENDING',real),('matched_synthetic90',synth)]:
   loc,order=d.old.cm.derangements('teacherexpanded:'+item['id']);tr=d.old.transform.transform(item['question'],loc,order);inverse=d.old.transform.transform(tr['question'],{v:k for k,v in loc.items()},[tr['old_to_new_letter'][a] for a in 'ABCD']);assert inverse['question']==d.old.transform.canonical(item['question'])
   for view,text,ans in [('original',item['question'],item['answer']),('joint',tr['question'],tr['old_to_new_letter'][item['answer']])]:
    d.certify(text,ans);ident=f'teachermore20261108/{cohort}/{index:03d}/{view}';body=dict(model=prior.MODEL,messages=[dict(role='system',content=sysmsg),dict(role='user',content=text)],temperature=0,enable_thinking=False,max_tokens=512,stream=False);bound=sum(len(m['content'].encode()) for m in body['messages'])+256
    queries.append(dict(id=ident,source_id=item['id'],group=item['group'],cohort=cohort,metric=item['metric'],view=view,pair=index,stratum=item['stratum'],body=body,input_reserve=bound));answers[ident]=ans
   maps[f'{cohort}/{index:03d}']=tr['old_to_new_letter']
 save(S/'requests-PENDING.local.json',queries);save(S/'answers.local.json',answers);save(S/'effective-maps.local.json',maps);save(S/'original-release-PENDING.local.json',chosen);save(S/'synthetic-selected.local.json',generated);save(S/'excluded-ambiguous-originals.local.json',excluded)
 oldcost=prior.budget()['cumulative_task_CNY'];newmax=sum(r['input_reserve']*12+512*36 for r in queries)/1e6
 save(S/'preparation-review.json',dict(status='LOCAL_PREPARED_AWAITING_NEW_ORIGINAL_EGRESS_EXCEPTION',source=pin(__file__),bindings=[pin(B/'roles.json'),pin(B/'Screen.local.jsonl'),pin(THRESHOLD),pin(poolpath)]+[pin(p) for p in files],real90_unique_numeric_groups=True,previous_recorded_training_fit_eval_overlap=0,original_Final_input_access=False,original_inputs_unchanged=True,original_eligible=len(candidates),excluded_nonunique=len(excluded),real_metric_counts=dict(collections.Counter(x['metric'] for x in chosen)),matched_strata=dict(collections.Counter(str(x['stratum']) for x in chosen)),matching='Exact metric / requested NaN / displayed tie / requested Fit quantile-tail indicator; no model outcome selection',synthetic_generator_unchanged=True,synthetic_prior_training_diagnostic_overlap=0,queries=360,request_order='Paired real/synthetic, each original/joint; all complete before scoring',model=prior.MODEL,new_GPU=0,new_HTTP=0,previous_research_CNY=oldcost,new_conservative_max_CNY=newmax,combined_conservative_CNY=oldcost+newmax,original_teacher_comparison_cap_CNY=20,requires_reduced_plan_if_combined_exceeds20=oldcost+newmax>20,task_remaining_before=500-oldcost,cheaper_alternative='Only expand synthetic held diagnostics and reuse existing public7; cannot establish real-domain equivalence',privacy_boundary='These90 were locally protected unused Screen-pool questions. New explicit public-proxy exception is required before external transmission. Keep Train,GeneratorFit,Dev,Common,Audit,Confirm,Final untouched. Log public exposure and never count these90 as private or independent final acceptance.',API_packets=[pin(S/n) for n in ['requests-PENDING.local.json','answers.local.json','effective-maps.local.json','original-release-PENDING.local.json','synthetic-selected.local.json']]))
 print('PREPARED90+90;360requests;budget',oldcost,newmax,oldcost+newmax,flush=True)
if __name__=='__main__':main()
