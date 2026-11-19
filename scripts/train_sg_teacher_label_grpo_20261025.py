#!/usr/bin/env python3
"""SG cached-teacher-label GRPO / continued SeqKD50; native frozen helper reuse."""
import argparse,gc,json,math,os,signal,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import train_sg_seqkd200_20261025 as seq
q=seq.q;t=seq.t;tiny=seq.tiny;probe=seq.probe;NEW=seq.NEW
BASE=NEW/'teacher-label-grpo/SGQA';PLAN=ROOT/'reports/qwen-transfer-repair-2026-10-25/teacher-label-grpo-sg-plan.md'
SOURCE_SHA='a0962e8e58a6fb7657606c826000f8a622ea9766b3ccf9ce68ea08a36b257c36'
if q.sha(ROOT/'scripts/train_sg_seqkd200_20261025.py')!=SOURCE_SHA:raise PermissionError('Frozen SeqKD helper changed')
ARMS=('label-grpo','continued-seqkd50');MILESTONES=[0,25,50];ARCHIVES=[0,10,20,25,30,40,50];STOP=False

def bind(p):return tiny.bind(p)
def bound(s):return tiny.bound(s)
def owned(out):
 raw=Path(out).absolute();p=raw.resolve()
 if p!=raw or raw.is_symlink() or p.name not in ARMS or p.parent.name not in ('08','2b') or p.parent.parent!=BASE.resolve():raise PermissionError('Owned SG size/arm path required')
 return p

def stop(*_):
 global STOP
 STOP=True

def stopped(out):return STOP or any(p.exists() for p in (NEW/'stop.json',NEW/'root-stop.json',NEW/'global-stop.json',BASE/'stop.json',out/'stop.json',out/'OOM_STOP.json'))

def first_families(rows):
 if len(rows)!=50 or len({r['id'] for r in rows})!=50:raise PermissionError('Exactly frozen PolicyTrain50 required')
 counts={};chosen=[]
 for row in rows:
  family=row['metadata']['question_type']
  if family not in counts:chosen.append(row)
  counts[family]=counts.get(family,0)+1
 if len(counts)!=10 or set(counts.values())!={5}:raise PermissionError('Policy50 must be10families x5')
 return chosen

def rewards_for(records,row,teacher):
 if len(records)!=8 or teacher.get('valid') is not True or teacher.get('prediction') is None:raise PermissionError('Exactly8 fresh responses and valid cached pseudo-label required')
 parsed=[q.classify(r['raw_generated'],'off','sgqa',row.get('metadata',{}),r['finish_reason']) for r in records]
 rewards=[float(v['valid'] and r['finish_reason']=='stop' and v['prediction']==teacher['prediction']) for r,v in zip(records,parsed)]
 return rewards,parsed

def diagnostic_summary(groups):
 if len(groups)!=10 or len({g['id'] for g in groups})!=10 or any(len(g['rewards'])!=8 for g in groups):raise PermissionError('All10x8 diagnostic groups freeze before qualification')
 mixed=sum(0<sum(g['rewards'])<8 for g in groups)
 return dict(qualified=mixed>=2,mixed_groups=mixed,total_groups=10,total_responses=80,teacher_matches=sum(sum(g['rewards']) for g in groups),invalid=sum(not x['valid'] for g in groups for x in g['parsed']),length_capped=sum(r['finish_reason']=='length' for g in groups for r in g['responses']),heuristic='atleast2of10 mixed; resource allocation only, not adequacy or promised gain',no_gold=True,no_resampling=True,optimizer_updates=0)

def starting_choices():
 freeze=seq.BASE/'both-size-selections-frozen.json';f=q.load(freeze)
 if f.get('all2_before_CommonEval') is not True or set(f['models'])!={'08','2b'}:raise PermissionError('Both SeqKD selections/reloads must freeze first')
 bindings=[bind(freeze)]
 for model in ('08','2b'):
  out=seq.BASE/model;selected=q.load(out/'selected.json');reloaded=q.load(out/'selected-reload.json');p=q.load(out/'protocol.json')
  seq.t.lib.verify_checkpoint(out,Path(selected['checkpoint']))
  if reloaded.get('status')!='PASS' or reloaded['selected']!=bind(out/'selected.json'):raise PermissionError('Exact selected W independently reloaded required')
  bindings += [bind(out/'protocol.json'),bind(out/'selected.json'),bind(out/'selected-reload.json'),selected['checkpoint_manifest'],*p['bindings']]
  for spec in f['models'][model].values():bound(spec)
 return bindings

def prepare(model,arm):
 if model not in ('08','2b') or arm not in ARMS:raise PermissionError('Fixed SG model/arm only')
 start_bindings=starting_choices();source=seq.BASE/model;sp=q.load(source/'protocol.json');selected=q.load(source/'selected.json');m,teacher,data,dev,fit,bindings=seq.data_inputs()
 rows=t.policy_rows(probe.RUN/'data/SGQA.manifest.json',data['selected'],'SGQA');diagnostic=first_families(rows)
 data=dict(selected=rows,teacher={r['id']:data['teacher'][r['id']] for r in rows},fixed8=data['fixed8'])
 out=owned(BASE/model/arm);q.immutable(out/'inputs.json',data)
 p=dict(schema='SG-cached-teacher-label-repair-v1',task='SGQA',scorer='sgqa',model=model,arm=arm,unit_path=sp['unit_path'],condition='matched8',lr=1e-5 if arm=='label-grpo' else 1e-4,base=sp['base'],base_assets=sp['base_assets'],runtime=sp['runtime'],seed=20261024,rollout_seed_derivation='sha256 legacy overnight-gad-v1/model/task/stage/step/id/index first8hex; actual seed saved per rollout',inputs=bind(out/'inputs.json'),fit16=sp['fit16'],development=dev,common_eval=sp['common_eval'],source_selection=bind(source/'selected.json'),source_checkpoint=selected['checkpoint_manifest'],source_output=str(source),training_order=[r['id'] for r in rows],diagnostic_order=[r['id'] for r in diagnostic],train_count=50,epochs=1,maximum_optimizer_attempts=50,group_size=8 if arm=='label-grpo' else 1,diagnostic_group_size=8,diagnostic_mixed_min=2,milestones=MILESTONES,selection_steps=[25,50],temperature=.8,top_k=0,top_p=1,rollout_max_new_tokens=256,greedy_max_new_tokens=256,thinking='off',beta=.001,clip=.2,clip_norm=1,weight_decay=0,optimizer='new AdamW moments reset',frozen_reference='exactselectedWclone',native_vocabulary=True,lora=sp['lora'],gold_in_loss=False,gold_in_reward=False,teacher_train_mode=teacher['mode'],teacher_final_only=True,checkpoint_each_complete_update=True,retention=ARCHIVES,output_bytes_max=8*2**30,new_api_calls=0,Confirm_read=False,Final_read=False,bindings=[bind(__file__),bind(ROOT/'scripts/train_sg_seqkd200_20261025.py'),bind(ROOT/'src/privacy_synth/onpolicy_gad.py'),bind(ROOT/'scripts/bird_gad_night_20261022.py'),bind(PLAN),bind(NEW/'user-goal-approval.json'),bind(out/'inputs.json'),sp['fit16'],*start_bindings,*bindings])
 # Reuse native tokenizer packing checks; no model weights or GPU allocation.
 q.setup_native();from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(p['base'],local_files_only=True,trust_remote_code=False)
 audit=[]
 for row in rows:
  prefix,target=probe.packing(tok,row,data['teacher'][row['id']],'matched8',data['fixed8']);audit.append(dict(id=row['id'],prompt_tokens=len(prefix),teacher_tokens=len(target)))
 q.immutable(out/'token-audit.json',dict(items=audit,tokenizer_only=True,gpu_used=False));p['bindings'].append(bind(out/'token-audit.json'));q.immutable(out/'protocol.json',p)
 return dict(status='PREPARED_ZERO_GPU',protocol=bind(out/'protocol.json'))

def verify(out):
 out=owned(out);p=q.load(out/'protocol.json');auth=q.load(out/'authorization.json')
 if auth.get('approved') is not True or auth.get('protocol_sha256')!=q.sha(out/'protocol.json'):raise PermissionError('Exact root arm authorization required')
 for spec in p['bindings']:bound(spec)
 lr=1e-5 if out.name=='label-grpo' else 1e-4
 if (p['task'],p['model'],p['arm'],p['lr'],p['maximum_optimizer_attempts'],p['milestones'],p['selection_steps'])!=('SGQA',out.parent.name,out.name,lr,50,MILESTONES,[25,50]):raise PermissionError('Fixed recipe changed')
 return p

def checkpoint_stage(out):return 'LABEL_GRPO' if out.name=='label-grpo' else 'CONTINUED_SEQKD50'

def cursor(out):
 out=owned(out)
 if tiny.unresolved(out):raise PermissionError('PARTIAL_UNSAFE unknown operation; never replay sampling/update')
 ptr=q.load(out/'work/latest.json');folder=Path(ptr['path'])
 if folder.is_symlink() or folder.resolve().parent!=(out/'work/checkpoints').resolve() or q.sha(folder/'manifest.json')!=ptr['manifest_sha256']:raise PermissionError('Owned checkpoint pointer differs')
 cm=t.lib.verify_checkpoint(out,folder);step=cm['step'];logs=sorted(int(f.stem) for f in (out/'work/updates').glob('*.json') if '.started.' not in f.name)
 if cm['stage']!=checkpoint_stage(out) or not 0<=step<=50 or logs!=list(range(1,step+1)):raise PermissionError('Checkpoint/log cursor differs')
 return folder,step

def save_boundary(out,step,torch,model,head,opt,metric=None):
 if t.size_tree(NEW)+64*2**20>8*2**30:raise RuntimeError('GLOBAL_REPAIR_OUTPUT_STAGING_8GIB_CAP')
 folder=out/'work/checkpoints'/f'step{step:04d}';logs=[q.load(f) for f in (out/'work/updates').glob('*.json') if '.started.' not in f.name]
 actual=sum(r['optimizer_updated'] for r in logs)+int(metric['optimizer_updated'] if metric else 0)
 tiny.SAVE_CHECKPOINT(out,folder,torch,model,head,dict(actor=opt),checkpoint_stage(out),step,actual_updates=actual);t.lib.verify_checkpoint(out,folder)
 if metric is not None:q.immutable(out/'work/updates'/f'{step:04d}.json',metric)
 t.atomic(out/'work/latest.json',dict(path=str(folder),manifest_sha256=q.sha(folder/'manifest.json')))
 ns=dict(tiny.__dict__,MILESTONES=ARCHIVES);types.FunctionType(tiny.retain.__code__,ns,'repair_retention')(out,step)
 return folder

def fresh_selected(p):
 selected=q.load(bound(p['source_selection']));source=Path(p['source_output']);cm=t.lib.verify_checkpoint(source,Path(selected['checkpoint']));torch,tok,model,head=t.lib.setup(Path(p['unit_path']),Path(selected['checkpoint']))
 if {r:t.lib.role_hash(model,r) for r in cm['role_hashes']}!=cm['role_hashes']:raise PermissionError('Selected W load differs')
 from peft import get_peft_model_state_dict,set_peft_model_state_dict
 state={k:v.detach().clone() for k,v in get_peft_model_state_dict(model,adapter_name='actor').items()};set_peft_model_state_dict(model,state,adapter_name='reference')
 if t.lib.role_hash(model,'reference')!=t.lib.role_hash(model,'actor'):raise PermissionError('Selected W KL clone differs')
 t.lib.activate(model,head,'actor');opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['lr'],weight_decay=0,foreach=False)
 if opt.state:raise PermissionError('New AdamW moments must be empty')
 return torch,tok,model,head,opt

def checkpoint_rows(*args,**kwargs):
 ns=dict(seq.__dict__,verify=verify,stopped=stopped);return types.FunctionType(seq.checkpoint_rows.__code__,ns,'repair_checkpoint_rows')(*args,**kwargs)

def evaluate_dev(out,p,step,torch,tok,model,head,data):
 folder=out/'development'/f'step{step:04d}';rows=t.lib.rows(p['development']['queries']);records=checkpoint_rows(out,folder,torch,tok,model,head,rows,data['fixed8'],'dev-step'+str(step))
 if records is None:return False
 if len(records)!=32 or {r['id'] for r in records}!={r['id'] for r in rows}:raise PermissionError('Full Dev32 freeze required')
 gold=q.load(bound(p['development']['gold']));by={r['id']:r for r in rows};scores=[probe.score_record(r,'sgqa',gold[r['id']],by[r['id']]['metadata']) for r in records]
 q.immutable(folder/'scores.json',dict(correct=sum(c for c,v in scores),valid=sum(v for c,v in scores),total=32,length_capped=sum(r['finish_reason']=='length' for r in records),freeze=bind(folder/'predictions-frozen.json'),step=step));return True

def sample_group(out,p,step,row,torch,tok,model,head,data,stage):
 t.context(out,stage,step)
 return t.rollouts(torch,tok,model,head,row,n=8,greedy=False,messages=data['fixed8']+[dict(role='user',content=row['question'])])

def diagnose(out):
 out=owned(out);p=verify(out)
 if p['arm']!='label-grpo':raise PermissionError('Diagnostic belongs only to label-policy arm')
 if tiny.unresolved(out):raise PermissionError('Unresolved diagnostic sampling; no replay')
 if (out/'diagnostic/qualification.json').exists():return q.load(out/'diagnostic/qualification.json')
 if stopped(out):return dict(status='PAUSED_BEFORE_DIAGNOSTIC')
 data=q.load(bound(p['inputs']));rows=first_families(data['selected']);torch,tok,model,head,opt=fresh_selected(p);groups=[]
 for i,row in enumerate(rows,1):
  file=out/'diagnostic/groups'/f'{i:04d}.json'
  if file.exists():
   value=q.load(file)
   if value['id']!=row['id'] or value['actor_before']!=t.lib.role_hash(model,'actor') or value['protocol_sha256']!=q.sha(out/'protocol.json'):raise PermissionError('Saved diagnostic identity differs')
  else:
   if stopped(out):return dict(status='PAUSED_DIAGNOSTIC_BOUNDARY',groups=i-1)
   q.immutable(file.with_suffix('.started.json'),dict(id=row['id'],protocol_sha256=q.sha(out/'protocol.json'),actor_before=t.lib.role_hash(model,'actor')))
   responses=sample_group(out,p,i,row,torch,tok,model,head,data,'label-qualification')
   rewards,parsed=rewards_for(responses,row,data['teacher'][row['id']]);value=dict(id=row['id'],family=row['metadata']['question_type'],responses=responses,rewards=rewards,parsed=parsed,actor_before=t.lib.role_hash(model,'actor'),teacher_prediction=data['teacher'][row['id']]['prediction'],protocol_sha256=q.sha(out/'protocol.json'))
   q.immutable(file,value)
  groups.append(value)
 q.immutable(out/'diagnostic/predictions-frozen.json',dict(complete=True,count=80,groups=[bind(out/'diagnostic/groups'/f'{i:04d}.json') for i in range(1,11)],protocol_sha256=q.sha(out/'protocol.json'),gold_not_read=True))
 summary=dict(status='DIAGNOSTIC_FROZEN',**diagnostic_summary(groups),freeze=bind(out/'diagnostic/predictions-frozen.json'));q.immutable(out/'diagnostic/qualification.json',summary);return summary

def policy_update(torch,model,head,opt,records,rewards,before_update=None):
 with torch.no_grad():
  model.eval();t.lib.activate(model,head,'actor');old=[t.lib.logprobs(torch,model,r).detach().cpu() for r in records]
  t.lib.activate(model,head,'reference');ref=[t.lib.logprobs(torch,model,r).detach().cpu() for r in records]
  advantages=t.lib.group_advantages(torch.tensor(rewards,device='cuda',dtype=torch.float32))
 for response,a,b in zip(records,old,ref):
  if len(a)!=len(response['response_ids']) or a.shape!=b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise RuntimeError('Actual response old/reference logprob shapes/nonfinite')
 saved=dict(old_log_probs=[v.tolist() for v in old],reference_log_probs=[v.tolist() for v in ref],advantages=advantages.cpu().tolist(),temperature=.8,logprobs_include_actual_EOS_only=True,k3_KL_before_update=sum(float(((b-a).exp()-(b-a)-1).mean()) for a,b in zip(old,ref))/len(records))
 if before_update is not None:before_update(saved)
 metric=t.lib.actor_step(torch,model,head,opt,records,old,ref,advantages)
 zero=all(float(x)==0 for x in saved['advantages']);metric.update(zero_label_advantages=zero,KL_only_optimizer_update=bool(zero and metric['optimizer_updated']),nonzero_label_signal=not zero)
 return metric,saved

def control_update(torch,tok,model,head,opt,row,teacher,p,data):
 prefix,target=probe.packing(tok,row,teacher,'matched8',data['fixed8']);t.lib.activate(model,head,'actor');model.train();opt.zero_grad(set_to_none=True);before=t.lib.role_hash(model,'actor');ids=torch.tensor([prefix+target],device='cuda')
 loss=-t.lib.response_log_probs(t.lib.hidden(model,ids),model.get_output_embeddings().weight,ids,len(prefix),temperature=1.).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1)
 if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite continued SeqKD loss/gradient')
 updated=float(norm)>0
 if updated:opt.step()
 opt.zero_grad(set_to_none=True);after=t.lib.role_hash(model,'actor')
 if updated and before==after:raise RuntimeError('Nonzero control gradient without parameter change')
 return dict(loss=float(loss.detach()),gradient_norm=float(norm),before=before,after=after,optimizer_updated=updated,teacher_tokens=len(target),input_tokens=len(prefix),temperature=1.,gold_in_loss=False)

def choose(out):
 verify(out);choices=[]
 for step in (25,50):
  v=q.load(out/'development'/f'step{step:04d}'/'scores.json');choices.append(((-v['correct'],-v['valid'],v['length_capped'],step),step,v))
 _,step,value=min(choices);folder=out/'work/checkpoints'/f'step{step:04d}';t.lib.verify_checkpoint(out,folder)
 result=dict(step=step,checkpoint=str(folder),checkpoint_manifest=bind(folder/'manifest.json'),development=value,baseline0_excluded=True,protocol=bind(out/'protocol.json'));q.immutable(out/'selected.json',result);return result

def run(out,resume=False):
 out=owned(out);p=verify(out);data=q.load(bound(p['inputs']))
 if stopped(out):return dict(status='PAUSED_BEFORE_MODEL')
 if tiny.unresolved(out):raise PermissionError('PARTIAL_UNSAFE unknown operation; no replay')
 if p['arm']=='label-grpo':
  qual=q.load(out/'diagnostic/qualification.json');freeze=q.load(bound(qual['freeze']))
  if freeze.get('complete') is not True or freeze.get('count')!=80:raise PermissionError('All80 diagnostic freeze needed')
  groups=[q.load(bound(s)) for s in freeze['groups']]
  if any(qual[k]!=v for k,v in diagnostic_summary(groups).items()):raise PermissionError('Qualification differs from frozen80')
  if not qual['qualified']:
   value=dict(status='SKIPPED_POLICY_NOT_MIXED',qualification=bind(out/'diagnostic/qualification.json'),optimizer_attempts=0,not_model_failure=True);q.immutable(out/'training-complete.json',value);return value
 start=0
 if resume:
  folder,start=cursor(out);a=q.load(out/f'resume-authorization-{start:04d}.json');used=out/f'resume-used-{start:04d}.json'
  if a.get('approved') is not True or a.get('checkpoint_manifest_sha256')!=q.sha(folder/'manifest.json') or a.get('protocol_sha256')!=q.sha(out/'protocol.json') or used.exists():raise PermissionError('Exact unused complete-boundary resume receipt needed')
  q.immutable(used,dict(authorization=bind(out/f'resume-authorization-{start:04d}.json')));torch,tok,model,head,opt=tiny.reload_saved(out,p,folder)
 else:
  if (out/'work').exists():raise PermissionError('Existing work requires explicit safe resume')
  (out/'work').mkdir();torch,tok,model,head,opt=fresh_selected(p)
  q.immutable(out/'optimizer-reset.json',dict(source_checkpoint=p['source_checkpoint'],new_state_empty=not bool(opt.state),lr=p['lr'],actor_sha256=t.lib.role_hash(model,'actor'),reference_sha256=t.lib.role_hash(model,'reference'),discriminator_unused=True))
  folder=save_boundary(out,0,torch,model,head,opt)
  del model,head,opt;gc.collect();torch.cuda.empty_cache();torch,tok,model,head,opt=tiny.reload_saved(out,p,folder)
 fixed={r:t.lib.role_hash(model,r) for r in ('reference','discriminator')};head_fixed=t.lib.nested_hash(head.state_dict())
 if start in MILESTONES and not evaluate_dev(out,p,start,torch,tok,model,head,data):return dict(status='PAUSED_DEV',step=start)
 for step in range(start+1,51):
  if stopped(out):return dict(status='PAUSED_AT_COMPLETE_CHECKPOINT',step=step-1)
  row=data['selected'][step-1];file=out/'work/updates'/f'{step:04d}.json';before=t.lib.role_hash(model,'actor');q.immutable(file.with_suffix('.started.json'),dict(id=row['id'],step=step,actor_before=before,protocol_sha256=q.sha(out/'protocol.json')))
  started=time.monotonic();extra={}
  if p['arm']=='label-grpo':
   responses=sample_group(out,p,step,row,torch,tok,model,head,data,'label-policy-fresh');rewards,parsed=rewards_for(responses,row,data['teacher'][row['id']])
   def persist_before_optimizer(saved):
    group=dict(id=row['id'],step=step,actor_before=before,reference_sha256=fixed['reference'],responses=responses,rewards=rewards,parsed=parsed,teacher_prediction=data['teacher'][row['id']]['prediction'],protocol_sha256=q.sha(out/'protocol.json'),**saved)
    q.immutable(out/'work/groups'/f'{step:04d}.json',group)
   metric,saved=policy_update(torch,model,head,opt,responses,rewards,persist_before_optimizer)
   extra=dict(group=bind(out/'work/groups'/f'{step:04d}.json'))
  else:metric=control_update(torch,tok,model,head,opt,row,data['teacher'][row['id']],p,data)
  if {r:t.lib.role_hash(model,r) for r in fixed}!=fixed or t.lib.nested_hash(head.state_dict())!=head_fixed:raise RuntimeError('Frozen reference/D/head changed')
  value=dict(id=row['id'],step=step,arm=p['arm'],optimizer_updated=metric['optimizer_updated'],actor=metric,lr=p['lr'],reference_discriminator_head_unchanged=True,gold_in_loss=False,gold_in_reward=False,update_seconds=time.monotonic()-started,**extra)
  folder=save_boundary(out,step,torch,model,head,opt,value);t.atomic(out/'heartbeat.json',dict(epoch=time.time(),stage='REPAIR_UPDATE_COMMITTED',arm=p['arm'],step=step))
  if step in (25,50):
   del model,head,opt;gc.collect();torch.cuda.empty_cache();torch,tok,model,head,opt=tiny.reload_saved(out,p,folder)
   if not evaluate_dev(out,p,step,torch,tok,model,head,data):return dict(status='PAUSED_DEV',step=step)
 selected=choose(out);logs=[q.load(out/'work/updates'/f'{i:04d}.json') for i in range(1,51)]
 if [v['id'] for v in logs]!=p['training_order']:raise PermissionError('All50 original policy prompts once')
 result=dict(status='TRAINING_COMPLETE',arm=p['arm'],model=p['model'],optimizer_attempts=50,actual_optimizer_updates=sum(v['optimizer_updated'] for v in logs),zero_label_advantage_groups=sum(v['actor'].get('zero_label_advantages',False) for v in logs),KL_only_updates=sum(v['actor'].get('KL_only_optimizer_update',False) for v in logs),nonzero_label_signal_groups=sum(v['actor'].get('nonzero_label_signal',False) for v in logs),selection=selected,new_api_calls=0,not_GAD=True)
 q.immutable(out/'training-complete.json',result);return result

def reload_selected(out):
 out=owned(out);p=verify(out);sel=q.load(out/'selected.json');folder=Path(sel['checkpoint']);torch,tok,model,head,opt=tiny.reload_saved(out,p,folder)
 fn=types.FunctionType(tiny.evaluate_milestone.__code__,dict(tiny.__dict__,stopped=stopped),'repair_selected_fit16')
 if not fn(out,p,sel['step'],torch,tok,model,head,q.load(bound(p['fit16']))):return dict(status='PAUSED_SELECTED_FIT16')
 q.immutable(out/'selected-reload.json',dict(status='PASS',selected=bind(out/'selected.json'),checkpoint_manifest=bind(folder/'manifest.json'),roles_optimizer_RNG_exact=True,fit16=bind(out/'measurements'/f"step{sel['step']:04d}"/'predictions-frozen.json')));return dict(status='SELECTED_RELOADED')

def evaluate_common(out):
 out=owned(out);p=verify(out);choices=q.load(BASE/'all-active-selections-frozen.json');key=p['model']+'/'+p['arm']
 if choices.get('all_active_before_CommonEval') is not True or key not in choices['active']:raise PermissionError('Every declared active selection must freeze first')
 for specs in choices['active'].values():
  for spec in specs.values():bound(spec)
 selected=q.load(out/'selected.json');torch,tok,model,head,opt=tiny.reload_saved(out,p,Path(selected['checkpoint']));rows=t.lib.rows(p['common_eval']['queries']);records=checkpoint_rows(out,out/'common',torch,tok,model,head,rows,[],'common')
 if records is None:return dict(status='PAUSED_COMMON')
 if len(records)!=60:raise PermissionError('Full Common60 required')
 return dict(status='COMMON_FROZEN_UNSCORED',freeze=bind(out/'common/predictions-frozen.json'))

def main():
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','diagnose','train','resume-check','reload','common']);ap.add_argument('--model',choices=['08','2b'],required=True);ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();out=BASE/a.model/a.arm
 if a.action=='prepare':result=prepare(a.model,a.arm)
 elif a.action=='resume-check':verify(out);f,s=cursor(out);result=dict(status='SAFE_RESUME',step=s,checkpoint=bind(f/'manifest.json'),gpu_used=False)
 else:
  from night_dual_20261022 import lock,cgroup_limits_verified
  import run_cityfm_repair_qwen_20261022 as guard
  if stopped(out):print(json.dumps(dict(status='PAUSED_BEFORE_GPU')));return
  with lock(out/'worker.lock'),lock(guard.HISTORICAL/'locks/gpu.lock'):
   cgroup_limits_verified()
   try:result={'diagnose':diagnose,'train':lambda o:run(o,a.resume),'reload':reload_selected,'common':evaluate_common}[a.action](out)
   except Exception as e:
    oom=type(e).__name__=='OutOfMemoryError' or 'out of memory' in str(e).lower();q.immutable(out/'failures'/f'{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),OOM=oom,unknown_started=tiny.unresolved(out),no_retry=True))
    if sum(q.load(f).get('OOM',False) for f in (out/'failures').glob('*.json'))>=2:q.immutable(out/'OOM_STOP.json',dict(reason='TWO_OOM_AFFECTED_PHASE_STOP'))
    raise
 print(json.dumps(result))
if __name__=='__main__':main()
