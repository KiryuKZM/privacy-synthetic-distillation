"""Independent closed-study score, checkpoint and descriptive paired uncertainty."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import argparse,ast,collections,hashlib,json,math,re,sys,time
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_joint_generator_student_run_20261107 as run
O=run.O;load=run.load;bind=run.bind;save=run.save
ORIGINAL=R/'runs/2026-11-06-ust-retention-information-v1/independent-audit/real-own-runtime-and-closed-audit-v2.py'
def independent():
 nodes=[x for x in ast.parse(ORIGINAL.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='independent_prediction'];assert len(nodes)==1
 ns=dict(re=re);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ORIGINAL)+'::frozen_XML_only','exec'),ns);return ns['independent_prediction']
def stats(rows):
 out=dict(n=0,numeric_groups=0,correct=0,weak_compatible=0,valid=0,capped=0,predictions={});groups=set();pred=collections.Counter()
 for row in rows:
  out['n']+=1;groups.add(row['group']);pred[str(row['prediction'])]+=1
  for k in ('correct','weak_compatible','valid','capped'):out[k]+=int(row[k])
 out['numeric_groups']=len(groups);out['predictions']=dict(pred);return out
def contrast(a,b):
 aa={x['id']:x for x in a};bb={x['id']:x for x in b};assert set(aa)==set(bb)
 tab=collections.Counter((x['correct'],bb[k]['correct']) for k,x in aa.items())
 return dict(n=len(aa),before_correct=tab[1,0]+tab[1,1],after_correct=tab[0,1]+tab[1,1],retained=tab[1,1],lost=tab[1,0],gained=tab[0,1])
def interval(a,b,seed,replicates):
 bb={x['id']:x for x in b};groups=sorted({x['group'] for x in a});num=np.array([sum(int(bb[x['id']]['correct'])-int(x['correct']) for x in a if x['group']==g) for g in groups]);den=np.array([sum(x['group']==g for x in a) for g in groups]);ii=np.random.default_rng(seed).integers(0,len(groups),(replicates,len(groups)));boot=num[ii].sum(axis=1)/den[ii].sum(axis=1)
 return dict(n=len(a),dependency_groups=len(groups),difference=float(num.sum()/den.sum()),percentile95=np.quantile(boot,[.025,.975]).tolist())
def tests():
 import eval_ust_sg_qwen_modes_20261024 as parser
 pred=independent();cases=['','A','<Answer>A</Answer>','<Answer>Z</Answer>','<Answer> A </Answer>','<Answer>a</Answer>','<Answer>A</Answer><Answer>B</Answer>','<think><Answer>A</Answer>','<think><Answer>B</Answer></think><Answer>D</Answer>','```json\n{"answer":"A"}\n```','```\n<Answer>C</Answer>\n```','<Answer>\nA\n</Answer>']
 checks=0
 for txt in cases:
  for finish in ['stop','length','error']:
   for suffix in ['','<|im_end|>','<|endoftext|>']:
    val=pred(txt+suffix,finish);x=parser.classify(txt+suffix,'off','ust',{'ability':'st_understanding'},finish);assert val==x['prediction'] and (val in ('A','B','C','D'))==bool(x['valid']);checks+=1
 a=[dict(id=str(i),group=str(i//2),correct=i<2,weak_compatible=True,valid=True,capped=False,prediction='A') for i in range(4)];b=[dict(x,correct=i%2==0) for i,x in enumerate(a)]
 assert contrast(a,b)==dict(n=4,before_correct=2,after_correct=2,retained=1,lost=1,gained=1)
 assert stats(a)['correct']==2 and stats(a)['numeric_groups']==2 and interval(a,a,123,20)['percentile95']==[0.,0.]
 save(O/'closed-audit-qualification.json',dict(status='PASS_FICTIONAL_TESTS_ONLY',parser_cases=checks,summary_paired_bootstrap_fixtures=True,source=bind(__file__),independent_parser=bind(ORIGINAL),plan=bind(O/'closed-audit-plan.json'),no_real_model_raw_read=True))
 print('PASS',checks,'independent parser fixtures',flush=True)
def closed():
 cert=load(O/'closed-audit-qualification.json');assert cert['source']==bind(__file__) and cert['independent_parser']==bind(ORIGINAL) and cert['plan']==bind(O/'closed-audit-plan.json')
 run.verify();assert load(O/'terminal.json')['status']=='COMPLETE';result=load(O/'results.json');rawpins=load(O/'all1444-raw-frozen-before-score.json')['raw'];assert len(rawpins)==1444
 for pin in rawpins:assert bind(R/pin['path'])==pin
 import torch
 assert not torch.cuda.is_initialized();init=load(run.core.INIT/'manifest.json');model_specs=run.frozen_models();fit={}
 for arm in run.ARMS:
  u=O/arm;start=load(u/'initialization-verified.json');end=load(u/'fit-terminal.json');assert start['roles']==init['role_hashes'] and start['optimizer']==init['optimizer_hashes']['actor'] and end['actual_updates']==136
  assert all(end['roles'][k]==init['role_hashes'][k] for k in ('reference','discriminator'))
  schedule=load(u/'schedule.local.json');update_seconds=0
  for row in schedule:
   x=load(u/'updates'/f"{row['step']:04d}.json");assert x['variant_ids']==row['variant_ids'] and x['step']==row['step'] and x['optimizer_calls']==1 and x['micro_forward_backward_calls']==4 and x['clipping_calls']==1 and x['API']==0
   assert x['state_steps_before']==[row['step']-1]*48 and x['state_steps_after']==[row['step']]*48 and x['RNG']==start['RNG'] and math.isfinite(x['loss']) and math.isfinite(x['gradient_norm']);update_seconds+=x['update_seconds']
  for step in (34,68,102,136):
   folder=u/'checkpoints'/f'step{step:04d}';run.core.helper().checkpoint_pins(u,folder,step);opt=torch.load(folder/'optimizers.pt',map_location='cpu',weights_only=True)['actor'];assert len(opt['state'])==48 and all(int(x['step'].item())==step for x in opt['state'].values())
  fit[arm]=dict(updates=136,microexamples=544,update_seconds=update_seconds,fit_seconds=end['seconds'],unchanged_reference_discriminator=True,all4_checkpoints_verified=True)
 pred=independent();answers=load(O/'answers.local.json');grades={};scores={};behavior={};wall=0;gpu_forward=0;tokens=load(O/'tokens.local.json');all_raw=0
 for key in run.UNITS:
  qq=load(O/key/'queries.local.json');stored=load(O/key/'graded.local.json');rows=[];reload=load(O/key/'reload.json');t=load(O/key/'evaluation-terminal.json');spec=model_specs[key]
  assert reload['roles']==spec['roles']==t['roles'] and reload['optimizer']==spec['optimizer']==t['optimizer'] and reload['RNG']==t['RNG']
  for i,q in enumerate(qq):
   raw=load(O/key/'raw'/f'{i:04d}.json');assert raw['id']==q['id'] and raw['prompt_tokens']==len(tokens[q['id']]['prompt_ids']) and raw['cap']==128
   v=pred(raw['raw_generated'],raw['finish_reason']);a=answers[q['id']];row=dict(id=q['id'],source_id=q['source_id'],group=q['group'],variant=q['variant'],metric=q['metric'],flags=q['flags'],answer=a['answer'],compatible=a['compatible'],prediction=v,valid=v in ('A','B','C','D'),correct=v in ('A','B','C','D') and v==a['answer'],weak_compatible=v in ('A','B','C','D') and v in a['compatible'],capped=raw['finish_reason']=='length');assert row==stored[i];rows.append(row);wall+=raw['request_wall_seconds'];gpu_forward+=sum(raw['forward_GPU_seconds']);all_raw+=1
  grades[key]=rows;scores[key]={}
  for variant in ('original','joint','own'):
   sub=[x for x in rows if x['variant']==variant];scores[key][variant]=dict(overall=stats(sub),by_metric={m:stats([x for x in sub if x['metric']==m]) for m in ('distance','charging_time','waiting_time')},subgroups={f:stats([x for x in sub if x['flags'].get(f)]) for f in ('any_tail','upper_tail','lower_tail','time_NaN','statistical_outlier_candidate')})
  aa={x['source_id']:x for x in rows if x['variant']=='original'};bb={x['source_id']:x for x in rows if x['variant']=='joint'};maps={q['source_id']:q['old_to_new_letter'] for q in qq if q['variant']=='joint'};behavior[key]=dict(n=155,both_correct=sum(x['correct'] and bb[k]['correct'] for k,x in aa.items()),equivariant=sum(x['valid'] and bb[k]['valid'] and maps[k].get(x['prediction'])==bb[k]['prediction'] for k,x in aa.items()))
 assert all_raw==1444 and result['scores']==scores and result['behavior']==behavior
 uncertainty={};contrasts={};bp=load(O/'closed-audit-plan.json')['paired_bootstrap']
 for name in result['paired_contrasts']:
  before,after=name.split('__');contrasts[name]={};uncertainty[name]={}
  for variant in ('original','joint'):
   a=[x for x in grades[before] if x['variant']==variant];b=[x for x in grades[after] if x['variant']==variant];contrasts[name][variant]=contrast(a,b);uncertainty[name][variant]=interval(a,b,bp['seed'],bp['replicates'])
 assert contrasts==result['paired_contrasts'];assert not torch.cuda.is_initialized()
 save(O/'closed-independent-review.json',dict(status='PASS_ALL1444_AND_ALL3_TRAINING_ARMS',producer_results=bind(O/'results.json'),auditor=bind(__file__),grade_disagreements=0,all_aggregates_reproduced=True,fit=fit,evaluation_request_wall_seconds=wall,evaluation_forward_GPU_seconds=gpu_forward,paired_group_bootstrap=uncertainty,uncertainty_scope=bp['interpretation'],all_raw=1444,API=0))
 print('PASS_ALL1444;no_disagreements;checkpoint_schedule_valid',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','closed']);args=p.parse_args();globals()[args.action]()
