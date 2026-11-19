"""Frozen-generator paired missingness data, disjoint Select/Confirm worlds."""
from pathlib import Path
from decimal import Decimal
import collections,copy,hashlib,itertools,json,re,sys,time
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_joint_generator_student_data_20261107 as old
import ust_joint_generator_student_run_20261107 as previous
from ust_joint_generator_residual_20261107 import ResidualRepair
from ust_joint_generator_conditional_calibration_20261107 import Calibrated
S=R/'runs/2026-11-07-ust-missingness-repair-v1';P=old.O;G=old.gen.O;RULE='For waiting_time and charging_time, NaN (with or without the h unit) ranks above every finite value. Rank any NaN values first, then sort all remaining finite values from largest to smallest.'
ARMS=('coverage0_native','coverage8_native','coverage0_rule','coverage8_rule','coverage16_rule','real_native','real_rule')
load=previous.load;save=previous.save;bind=previous.bind

def sig(obs):return json.dumps(sorted(obs,key=lambda x:json.dumps(x)),sort_keys=True)
def hashkey(*x):return old.key('missingness-frozen',*x)
def missing_style(q,style):
 token={'bare':'NaN','unit':'NaNh','spaced':'NaN h'}[style]
 return re.sub(r'(?m)^([ \t]*-[ \t]*(?:waiting_time|charging_time):[ \t]*)NaN(?: ?h)?([ \t]*)$',lambda m:m.group(1)+token+m.group(2),q)
def native(obs,metric,slot,options=None):
 options=options or old.options(obs,metric,slot);assert options is not None;opts,answer=options
 return missing_style(old.render(obs,metric,opts),'bare'),answer,options

def finite_target(obs,metric):
 out=copy.deepcopy(obs);j=old.FIELDS.index(metric);finite=[Decimal(str(x[j])) for x in obs if x[j] is not None];assert finite
 value=float(Decimal(2)*max(finite)+Decimal(1))
 for row in out:
  if row[j] is None:row[j]=value
 return out

def clarify(q,representation):
 q=old.weak.clarify_native(q)
 if representation=='rule':q=q.replace('## Note\n\n','## Note\n\n'+RULE+'\n',1)
 return q

def strip_rule(q):
 if RULE in q:
  assert q.count(RULE)==1 and '## Note\n\n'+RULE+'\n' in q
  q=q.replace('## Note\n\n'+RULE+'\n','## Note\n\n',1)
 return q

def certify(q,answer,training=False):
 if training:
  q=strip_rule(q);cert=old.weak.certify_explicit_weak_reference(q,answer);numeric=old.weak.independent_numeric(q)
 else:
  cert=old.certify_native_ranking_reference(q,answer);numeric=old.weak.independent_numeric(old.weak.clarify_native(old.transform.canonical(q)))
 assert cert['trusted_gold'] and cert['compatible_answers']==[answer] and numeric['compatible_answers']==[answer]
 return dict(metric=cert['metric'],target_NaN=numeric['has_target_NaN'],displayed_ties=numeric['has_displayed_ties'],independent_unique=True)

def pool_row(row,pool):return dict(obs=row['observations'],world=f'{pool}/{row["index"]}',raw_index=row['index'],pool=pool)

def build_dataset(name,rows,coverage,representation):
 variants=[];sources=[];proof=[]
 for slot,item in enumerate(rows):
  q,answer,_=native(item['obs'],item['metric'],slot,item.get('options'));sources.append(dict(**item,source_slot=slot,answer=answer,question=q))
  for offset in range(4):
   letters='ABCD';tr=old.transform.transform(q,{i:i for i in range(1,5)},letters[offset:]+letters[:offset]);ans=tr['old_to_new_letter'][answer];question=clarify(tr['question'],representation);check=certify(question,ans,True);vid=f'{name}/{slot:03d}/{offset}'
   variants.append(dict(id=vid,source_id=item['world'],source_slot=slot,variant_index=offset,question=question,answer=ans,target='<Answer>'+ans+'</Answer>',metadata=dict(ability='st_understanding',metric=item['metric']),metric=item['metric'],old_to_new_letter=tr['old_to_new_letter'],source_pool=item['pool'],counterfactual=item.get('counterfactual',False)))
   proof.append(dict(id=vid,**check))
 assert len(variants)==272 and len({x['world'] for x in sources})==68 and len({sig(x['obs']) for x in sources})==68
 assert sum(x['target_NaN'] for x in proof)==coverage*4
 assert collections.Counter(x['metric'] for x in rows)==dict(distance=42,charging_time=14,waiting_time=12)
 assert collections.Counter(x['answer'] for x in variants)==dict(A=68,B=68,C=68,D=68)
 save(S/'data'/name/'source68.local.json',sources);save(S/'data'/name/'variants.local.json',variants);save(S/'data'/name/'qualification.local.json',proof)
 return dict(source_cases=68,variants=272,target_NaN_sources=coverage,representation=representation,metric_counts=dict(collections.Counter(x['metric'] for x in rows)),displayed_ties=sum(x['displayed_ties'] for x in proof)//4)

def build_eval(cohort,missing,ordinary,excluded):
 queries=[];answers={};proof=[]
 def add(q,answer,world,metric,case, family,spelling,view,proof_extra=None):
  cert=certify(q,answer);qid=f'{cohort}/{case}/{family}/{spelling}/{view}'
  queries.append(dict(id=qid,source_id=world,group=world,case_id=case,cohort=cohort,variant='synthetic_diagnostic',family=family,spelling=spelling,view=view,question=q,metric=metric,metadata=dict(ability='st_understanding'),flags={},old_to_new_letter={a:a for a in 'ABCD'}))
  answers[qid]=dict(answer=answer,compatible=[answer]);proof.append(dict(id=qid,**cert,**(proof_extra or {})))
 for i,item in enumerate(missing):
  assert sig(item['obs']) not in excluded
  for j,metric in enumerate(('waiting_time','charging_time')):
   slot=i*2+j;case=f'NaN{i:02d}-{metric}';q,ans,options=native(item['obs'],metric,slot);finite=finite_target(item['obs'],metric);fq,fa,_=native(finite,metric,slot,options);assert ans==fa
   loc,order=old.cm.derangements('fixed-missing-study/'+cohort+'/'+case)
   for view in ('original','joint'):
    if view=='original':base=q;fbase=fq;gold=ans
    else:
     tr=old.transform.transform(q,loc,order);fr=old.transform.transform(fq,loc,order);base=tr['question'];fbase=fr['question'];gold=tr['old_to_new_letter'][ans];assert fr['old_to_new_letter'][fa]==gold
     inv=old.transform.transform(base,{v:k for k,v in loc.items()},[tr['old_to_new_letter'][a] for a in 'ABCD']);assert inv['question']==old.transform.canonical(q)
    for style in ('bare','unit','spaced'):
     text=missing_style(base,style);assert missing_style(text,'bare')==missing_style(base,'bare');add(text,gold,item['world'],metric,case,'missing',style,view,dict(representation_inverse=True))
    add(fbase,gold,item['world'],metric,case,'finite_twin','native',view,dict(unchanged_ordinal_reference=True))
 goals=['distance']*20+['charging_time']*6+['waiting_time']*6
 for i,(item,metric) in enumerate(zip(ordinary,goals)):
  assert sig(item['obs']) not in excluded;case=f'ordinary{i:02d}';q,ans,_=native(item['obs'],metric,i);loc,order=old.cm.derangements('fixed-missing-study/'+cohort+'/'+case);tr=old.transform.transform(q,loc,order)
  add(q,ans,item['world'],metric,case,'ordinary','native','original');add(tr['question'],tr['old_to_new_letter'][ans],item['world'],metric,case,'ordinary','native','joint')
 assert len(queries)==256 and collections.Counter(q['family'] for q in queries)==dict(missing=144,finite_twin=48,ordinary=64)
 save(S/'data'/f'{cohort}-queries.local.json',queries);save(S/'data'/f'{cohort}-answers.local.json',answers);save(S/'data'/f'{cohort}-qualification.local.json',proof)
 return dict(n=256,missing_worlds=len(missing),ordinary_worlds=len(ordinary),missing_views=144,finite_views=112)

def main():
 plan=load(S/'study-plan.json')
 for pin in plan['generator_bindings']:assert bind(R/pin['path'])['sha256']==pin['sha256']
 pool=load(P/'new-generator-pool1000.local.json');assert len(pool)==1000 and all(not x['failures'] for x in pool)
 byindex={x['index']:x for x in pool};byloc={i:[] for i in range(4)}
 for row in pool:
  loc=[i for i,v in enumerate(row['observations']) if v[3] is None]
  if loc:assert len(loc)==1;byloc[loc[0]].append(row)
 for i in byloc:byloc[i].sort(key=lambda x:hashkey('missing-split',i,x['index']))
 train=[];select=[];confirm=[];confirm_quota=[4,3,3,2]
 for loc in range(4):
  rows=byloc[loc];assert len(rows)>=5+confirm_quota[loc]
  for i,metric in enumerate(('waiting_time','charging_time')):
   item=pool_row(rows[i],'pool714');item['metric']=metric;train.append(item)
  select += [pool_row(x,'pool714') for x in rows[2:5]]
  confirm += [pool_row(x,'pool714') for x in rows[5:5+confirm_quota[loc]]]
 assert len(train)==8 and len(select)==len(confirm)==12
 parent=load(P/'conditional_calibrated68/source68.local.json');real=load(P/'real68/source68.local.json');base=[]
 for row in parent:
  ix=int(row['id'].rsplit('/',1)[-1]);item=pool_row(byindex[ix],'pool714');assert item['obs']==old.vals(row['question']);item['metric']=row['metadata']['metric'];base.append(item)
 slots={m:sorted([i for i,r in enumerate(base) if r['metric']==m],key=lambda i:hashkey('training-slot',m,i)) for m in ('waiting_time','charging_time')}
 true8=copy.deepcopy(base)
 for metric in slots:
  chosen=[x for x in train if x['metric']==metric]
  for slot,item in zip(slots[metric][:4],chosen):true8[slot]=copy.deepcopy(item)
 control=copy.deepcopy(true8)
 for slot,item in enumerate(true8):
  _,_,options=native(item['obs'],item['metric'],slot);item['options']=options;control[slot]['options']=options
  if any(v[old.FIELDS.index(item['metric'])] is None for v in item['obs']):control[slot]['obs']=finite_target(item['obs'],item['metric']);control[slot]['counterfactual']=True
 # Reconstruct the exact fixed generator and assert every saved learned parameter.
 gp=load(G/'protocol.json');fit=load(R/gp['fit']['path']);model=ResidualRepair(fit,'conditional_charge_and_wait');saved=load(G/'conditional_charge_and_wait/fitted-parameters.local.json')
 assert np.array_equal(model.beta_charge,saved['charge_coefficients']) and np.array_equal(model.beta_wait,saved['wait_coefficients']) and np.array_equal(model.root,saved['root']) and all(np.array_equal(a,b) for a,b in zip(model.marg,saved['marginals']))
 cal=Calibrated.__new__(Calibrated);cal.model=model;cal.tables=[(np.array(x['log_probe']),np.array(x['Fit_positive_log_inverse'])) for x in load(G/'conditional_calibrated/tables.local.json')]
 fallback,_=cal.sample(plan['fallback']['frozen_generator_additional_pool_n'],plan['fallback']['seed']);save(S/'data/fallback-fixed-pool2000.local.json',fallback);assert all(not x['failures'] for x in fallback)
 excluded={sig(x['obs']) for x in base+train+select+confirm};fl={i:[] for i in range(4)}
 for row in fallback:
  loc=[i for i,v in enumerate(row['observations']) if v[3] is None]
  if len(loc)==1 and sig(row['observations']) not in excluded:fl[loc[0]].append(row)
 train16=[]
 for loc in range(4):
  ordered=sorted(fl[loc],key=lambda x:hashkey('fallback16',loc,x['index']));assert len(ordered)>=4
  for i,row in enumerate(ordered[:4]):item=pool_row(row,'pool734');item['metric']='waiting_time' if i<2 else 'charging_time';train16.append(item)
 true16=copy.deepcopy(base)
 for metric in slots:
  chosen=[x for x in train16 if x['metric']==metric]
  for slot,item in zip(slots[metric][:8],chosen):true16[slot]=item
 realrows=[dict(obs=old.vals(r['question']),world=r['id'],pool='original_real68',metric=r['metadata']['metric']) for r in real]
 excluded={sig(x['obs']) for x in base+train+train16+realrows}
 assert not ({sig(x['obs']) for x in select+confirm}&excluded)
 assert not {sig(x['obs']) for x in select}&{sig(x['obs']) for x in confirm}
 ordinary_pool=[pool_row(x,'pool714') for x in pool if all(v[3] is not None and v[4] is not None for v in x['observations']) and sig(x['observations']) not in excluded]
 ordinary_pool.sort(key=lambda x:hashkey('ordinary-holdout',x['raw_index']));ordinary_select=ordinary_pool[:32];ordinary_confirm=ordinary_pool[32:64];assert len({sig(x['obs']) for x in ordinary_select+ordinary_confirm})==64
 summary={}
 for name,rows,coverage,representation in [('coverage0_native',control,0,'native'),('coverage8_native',true8,8,'native'),('coverage0_rule',control,0,'rule'),('coverage8_rule',true8,8,'rule'),('coverage16_rule',true16,16,'rule'),('real_native',realrows,1,'native'),('real_rule',realrows,1,'rule')]:summary[name]=build_dataset(name,rows,coverage,representation)
 es={cohort:build_eval(cohort,missing,ordinary,excluded) for cohort,missing,ordinary in [('Select',select,ordinary_select),('Confirm',confirm,ordinary_confirm)]}
 save(S/'data/world-membership.local.json',dict(training8=train,training16=train16,Select_missing=select,Confirm_missing=confirm,Select_ordinary=ordinary_select,Confirm_ordinary=ordinary_confirm))
 # Original Dev text and published gold are copied exactly, not rewritten.
 save(S/'data/Dev310-queries.local.json',load(P/'freshINIT/queries.local.json'));aa=load(P/'answers.local.json');dev_ids={q['id'] for q in load(P/'freshINIT/queries.local.json')};save(S/'data/Dev310-answers.local.json',{k:v for k,v in aa.items() if k in dev_ids});save(S/'data/fixed8.local.json',load(P/'fixed8.local.json'))
 save(S/'data-summary.json',dict(status='FROZEN_ALL_DATA_CPU_ONLY',epoch=time.time(),datasets=summary,diagnostics=es,generator_parameters_exact=True,new_fixed_generator_draws=2000,no_generator_refitting_or_tuning=True,source=bind(__file__),parent_generator_bindings=plan['generator_bindings'],API=0,GPU=0,Final_access=False))
 print(json.dumps(summary,indent=2));print('DATA_COMPLETE_CPU_ONLY',flush=True)
if __name__=='__main__':main()
