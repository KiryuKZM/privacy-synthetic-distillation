#!/usr/bin/env python3
"""Fixed Train16 offline teacher-CE memorization diagnostic; never main SeqKD/GAD."""
from __future__ import annotations
import argparse,ast,gc,json,math,os,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import probe_overnight_train_fit_20261025 as probe
import audit_overnight_training_replay_20261025 as audit
q=probe.q;t=probe.t;NEW=probe.NEW;BASE=NEW/'tiny-fit';STOP=False
MILESTONES=[0,16,32,64];LEARNING_RATES=[1e-6,1e-4]

def stop(*_):
 global STOP
 STOP=True
def stopped(out):return STOP or any(p.exists() for p in [NEW/'stop.json',NEW/'root-stop.json',NEW/'global-stop.json',BASE/'stop.json',out/'stop.json'])
def bind(p):return probe.bind(p)
def bound(s):return probe.bound(s)
def owned(out):
 raw=Path(out).absolute();p=raw.resolve()
 if raw!=p or raw.is_symlink() or not p.is_relative_to(BASE.resolve()):raise PermissionError('Owned new tiny-fit output required')
 return p
def recipe(task,model,condition,lr):
 if task not in ('USTBench','SGQA') or model not in ('08','2b') or condition not in ('zero-shot','matched8') or lr not in LEARNING_RATES:raise PermissionError('Explicit bounded tiny-fit recipe required')
 return BASE/task/model/(condition+'-lr'+format(lr,'.0e'))
def frozen_checkpoint_helper():
 # Reuse exact original serializer without mutating an imported module/global.
 if q.sha(t.IMPL)!=t.IMPL_SHA:raise PermissionError('Frozen checkpoint helper changed')
 tree=ast.parse(t.IMPL.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint');ns=dict(t.lib.__dict__)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(t.IMPL),'exec'),ns)
 return ns['checkpoint']
SAVE_CHECKPOINT=frozen_checkpoint_helper()

def prepare(task,model,condition,lr,decision_path):
 out=owned(recipe(task,model,condition,lr));decision=q.load(decision_path);ps=bound(decision['probe_scores']);scores=q.load(ps)
 if (decision.get('task'),decision.get('model'),decision.get('condition'))!=(task,model,condition) or (scores['task'],scores['model'])!=(task,model):raise PermissionError('Explicit same-unit probe-conditioned decision required')
 if not decision.get('reason'):raise PermissionError('Conditioning decision must explain probe evidence')
 if scores.get('posthoc_Train_only') is not True or scores.get('training_updates')!=0:raise PermissionError('Train-only diagnostic evidence required')
 pp=probe.verify();data_spec=pp['tasks'][task]['inputs'];data=q.load(bound(data_spec));unit=probe.RUN/'training'/task/model;unitp=q.load(unit/'protocol.json')
 bindings=[bind(__file__),bind(ROOT/'scripts/probe_overnight_train_fit_20261025.py'),bind(ROOT/'scripts/audit_overnight_training_replay_20261025.py'),bind(ROOT/'scripts/train_overnight_native_gad_20261025.py'),bind(t.IMPL),bind(NEW/'user-goal-approval.json'),bind(probe.OUT/'protocol.json'),bind(decision_path),decision['probe_scores'],data_spec,bind(unit/'protocol.json')]
 p=dict(schema='train16-cached-teacher-memorization-v1',task=task,model=model,condition=condition,lr=lr,unit_path=str(unit),bindings=bindings,probe_decision=bind(decision_path),inputs=data_spec,train_gold=pp['tasks'][task]['train_gold'],scorer=pp['tasks'][task]['scorer'],base=unitp['base'],base_assets=unitp['base_assets'],runtime=unitp['runtime'],seed=20261024,training_order=[r['id'] for r in data['selected']]*4,train_count=16,epochs=4,maximum_optimizer_attempts=64,group_size=1,teacher_final_only=True,gold_in_loss=False,no_teacher_filtering=True,native_vocabulary=True,thinking='off',lora=dict(r=8,alpha=16,target='original six full-attention layers q/k/v/o only',dropout=0),loss='full-native-vocab mean response teacher CE including native EOS; prompt target masked',clip_norm=1,weight_decay=0,optimizer='AdamW',milestones=MILESTONES,greedy_max_new_tokens=256,no_tuning_on_Eval=True,no_Eval_or_Final=True,new_api_calls=0,all_milestone_answers_freeze_before_Train_gold=True,checkpoint_each_complete_update=True,retention='init/16/32/64 +latest2; own verified payloads only',output_bytes_max=2*2**30,root_only_launch=True,diagnostic_not_generalization=True)
 q.immutable(out/'protocol.json',p);return dict(status='PREPARED_ZERO_GPU',protocol=bind(out/'protocol.json'),output=str(out))
def verify(out):
 out=owned(out);p=q.load(out/'protocol.json');a=q.load(out/'authorization.json')
 if a.get('approved') is not True or a.get('protocol_sha256')!=q.sha(out/'protocol.json'):raise PermissionError('Exact tiny-fit root authorization absent')
 for s in p['bindings']:bound(s)
 if p['maximum_optimizer_attempts']!=64 or p['milestones']!=MILESTONES or p['lr'] not in LEARNING_RATES:raise PermissionError('Fixed diagnostic scope changed')
 return p

def order_row(rows,step):
 if not 1<=step<=64 or len(rows)!=16:raise PermissionError('Fixed64/16 order required')
 return rows[(step-1)%16]
def unresolved(out):
 return [str(f) for f in out.rglob('*.started.json') if not f.with_name(f.name.replace('.started.json','.json')).exists()]
def cursor(out):
 out=owned(out)
 if unresolved(out):raise PermissionError('PARTIAL_UNSAFE unresolved operation; no replay')
 ptr=q.load(out/'work/latest.json');folder=Path(ptr['path'])
 if folder.is_symlink() or folder.resolve().parent!=(out/'work/checkpoints').resolve():raise PermissionError('Checkpoint pointer escaped own folder')
 if q.sha(folder/'manifest.json')!=ptr['manifest_sha256']:raise PermissionError('Latest checkpoint changed')
 cm=t.lib.verify_checkpoint(out,folder);step=cm['step'];logs=sorted(int(p.stem) for p in (out/'work/updates').glob('*.json') if '.started.' not in p.name)
 if logs!=list(range(1,step+1)):raise PermissionError('Optimizer worklog/pointer boundary differs')
 if cm['stage']!='TINY_FIT' or not 0<=step<=64:raise PermissionError('Unknown checkpoint stage')
 return folder,step

def retain(out,latest_step):
 base=out/'work/checkpoints';keep=set(MILESTONES)|{latest_step,max(0,latest_step-1)}
 for folder in base.iterdir():
  if int(folder.name.removeprefix('step')) in keep:continue
  if folder.is_symlink() or folder.resolve().parent!=base.resolve():raise PermissionError('Retention path escape')
  cm=t.lib.verify_checkpoint(out,folder);actual={str(x.relative_to(folder)) for x in folder.rglob('*') if x.is_file()}
  if actual!=set(cm['files'])|{'manifest.json'} or any(x.is_symlink() for x in folder.rglob('*')):raise PermissionError('Unexpected checkpoint payload')
  q.immutable(out/'work/tombstones'/(folder.name+'.json'),dict(manifest=cm,manifest_sha256=q.sha(folder/'manifest.json'),reason='fixed milestones+latest2'))
  shutil.rmtree(folder)

def save_boundary(out,step,torch,model,head,opt,metric=None):
 folder=out/'work/checkpoints'/f'step{step:04d}'
 if t.size_tree(out)+64*2**20>2*2**30:raise RuntimeError('Tiny-fit output staging exceeds2GiB')
 previous=[q.load(f) for f in (out/'work/updates').glob('*.json') if '.started.' not in f.name]
 actual=sum(int(r['optimizer_updated']) for r in previous)+int(metric['optimizer_updated'] if metric else 0)
 SAVE_CHECKPOINT(out,folder,torch,model,head,dict(actor=opt),'TINY_FIT',step,actual_updates=actual)
 cm=t.lib.verify_checkpoint(out,folder)
 if metric is not None:q.immutable(out/'work/updates'/f'{step:04d}.json',metric)
 t.atomic(out/'work/latest.json',dict(path=str(folder),manifest_sha256=q.sha(folder/'manifest.json')));retain(out,step)
 return folder

def reload_saved(out,p,folder):
 cm=t.lib.verify_checkpoint(out,folder);torch,tok,model,head=t.lib.setup(Path(p['unit_path']),folder);t.lib.activate(model,head,'actor');opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['lr'],weight_decay=0,foreach=False)
 t.lib.restore(torch,model,head,dict(actor=opt),out,folder)
 if {r:t.lib.role_hash(model,r) for r in cm['role_hashes']}!=cm['role_hashes']:raise PermissionError('Independent reload mismatch')
 return torch,tok,model,head,opt

def evaluate_milestone(out,p,step,torch,tok,model,head,data):
 folder=out/'measurements'/f'step{step:04d}';records=[];t.context(out,'tiny-fit-measurements',step);model.eval()
 for row in data['selected']:
  file=folder/(q.digest(row['id'])+'.json')
  if file.exists():records.append(q.load(file));continue
  if file.with_suffix('.started.json').exists():raise PermissionError('Unresolved diagnostic measurement; no replay')
  if stopped(out):return False
  q.immutable(file.with_suffix('.started.json'),dict(id=row['id'],step=step,protocol_sha256=q.sha(out/'protocol.json')))
  teacher=data['teacher'][row['id']];prefix,answer=probe.packing(tok,row,teacher,p['condition'],data['fixed8']);ids=torch.tensor([prefix+answer],device='cuda')
  t.lib.activate(model,head,'actor')
  with torch.inference_mode():lp=t.lib.response_log_probs(t.lib.hidden(model,ids),model.get_output_embeddings().weight,ids,len(prefix),temperature=1.);nll=float(-lp.mean());token_nll=(-lp).cpu().tolist()
  if not math.isfinite(nll):raise RuntimeError('Nonfinite diagnostic NLL')
  messages=row['messages'] if p['condition']=='zero-shot' else data['fixed8']+[dict(role='user',content=row['question'])]
  response=t.rollouts(torch,tok,model,head,row,n=1,greedy=True,messages=messages)[0];parsed=q.classify(response['raw_generated'],'off',p['scorer'],row.get('metadata',{}),response['finish_reason'])
  emitted=response['response_ids'];visible=emitted[:-1] if response['finish_reason']=='stop' else emitted
  record=dict(id=row['id'],step=step,**parsed,teacher_token_NLL=nll,teacher_token_NLLs=token_nll,teacher_target_ids=answer,teacher_tokens=len(answer),teacher_exact_text=tok.decode(visible,skip_special_tokens=False)==teacher['text'],teacher_prediction_agreement=bool(parsed['valid'] and parsed['prediction']==teacher['prediction']),raw_generated=response['raw_generated'],output_ids=emitted,finish_reason=response['finish_reason'],input_tokens=len(prefix),output_tokens=len(emitted),input_truncated=False,actor_sha256=t.lib.role_hash(model,'actor'),teacher_text_sha256=q.digest(teacher['text']),protocol_sha256=q.sha(out/'protocol.json'))
  q.immutable(file,record);records.append(record);del ids,lp;t.atomic(out/'heartbeat.json',dict(epoch=time.time(),stage='TRAIN_ONLY_MEASURE',step=step,id=row['id']))
 if len(records)!=16:raise PermissionError('Fixed16 measurement count')
 q.immutable(folder/'predictions-frozen.json',dict(complete=True,count=16,predictions=[bind(folder/(q.digest(r['id'])+'.json')) for r in data['selected']],protocol_sha256=q.sha(out/'protocol.json'),Train_gold_not_yet_read=True))
 q.immutable(folder/'teacher-only-summary.json',dict(mean_teacher_token_NLL=sum(r['teacher_token_NLL'] for r in records)/16,teacher_exact_text=sum(r['teacher_exact_text'] for r in records),teacher_prediction_agreement=sum(r['teacher_prediction_agreement'] for r in records),valid=sum(r['valid'] for r in records),length_capped=sum(r['finish_reason']=='length' for r in records),total=16,not_gold_accuracy=True))
 initial=out/'work/checkpoints/step0000/actor/adapter_model.safetensors';current=out/'work/checkpoints'/f'step{step:04d}'/'actor/adapter_model.safetensors'
 q.immutable(folder/'adapter-delta-from-init.json',audit.adapter_delta(initial,current));return True

def summarize(out):
 p=verify(out);data=q.load(bound(p['inputs']));paths=[out/'measurements'/f'step{s:04d}'/'predictions-frozen.json' for s in MILESTONES]
 if not all(f.exists() for f in paths):raise PermissionError('All4×16 frozen measurements required before Train gold')
 batches={}
 for step,freeze in zip(MILESTONES,paths):
  f=q.load(freeze);rs=[q.load(bound(s)) for s in f['predictions']]
  if f.get('complete') is not True or len(rs)!=16 or {r['id'] for r in rs}!={r['id'] for r in data['selected']}:raise PermissionError('Measurement membership differs')
  batches[step]=rs
 q.immutable(out/'all-measurements-frozen.json',dict(complete=True,count=64,milestones=[bind(f) for f in paths],protocol_sha256=q.sha(out/'protocol.json'),Train_gold_not_yet_read=True))
 gold=q.load(bound(p['train_gold']));by={r['id']:r for r in data['selected']};summary={}
 for step,rs in batches.items():
  scored=[probe.score_record(r,p['scorer'],gold[r['id']],by[r['id']]['metadata']) for r in rs];teach=[probe.score_record(data['teacher'][r['id']],p['scorer'],gold[r['id']],by[r['id']]['metadata']) for r in rs]
  summary[str(step)]=dict(**q.load(out/'measurements'/f'step{step:04d}'/'teacher-only-summary.json'),gold_correct=sum(c for c,v in scored),gold_valid=sum(v for c,v in scored),teacher_gold_correct=sum(c for c,v in teach),teacher_wrong_retained=sum(not c for c,v in teach))
 logs=[q.load(f) for f in sorted((out/'work/updates').glob('*.json')) if '.started.' not in f.name]
 if len(logs)!=64:raise PermissionError('Fixed64 optimizer attempts incomplete')
 result=dict(status='COMPLETE_DIAGNOSTIC',task=p['task'],model=p['model'],condition=p['condition'],lr=p['lr'],optimizer_attempts=64,actual_optimizer_updates=sum(r['optimizer_updated'] for r in logs),milestones=summary,all_measurements_freeze=bind(out/'all-measurements-frozen.json'),cached_teacher_offpolicy=True,not_main_SeqKD200=True,not_generalization_or_privacy=True,new_api_calls=0,teacher_wrong_not_filtered=True)
 q.immutable(out/'complete.json',result);return result

def run(out,resume=False):
 out=owned(out);p=verify(out);data=q.load(bound(p['inputs']));unit=Path(p['unit_path'])
 if stopped(out):return dict(status='PAUSED_BEFORE_MODEL')
 from night_dual_20261022 import lock,cgroup_limits_verified
 import run_cityfm_repair_qwen_20261022 as guard
 with lock(out/'worker.lock'),lock(guard.HISTORICAL/'locks/gpu.lock'):
  cgroup_limits_verified();start=0;folder=None
  if resume:
   folder,start=cursor(out);a=q.load(out/f'resume-authorization-{start:04d}.json')
   if a.get('approved') is not True or a.get('checkpoint_manifest_sha256')!=q.sha(folder/'manifest.json') or a.get('protocol_sha256')!=q.sha(out/'protocol.json'):raise PermissionError('Explicit boundary resume authorization required')
   if (out/f'resume-used-{start:04d}.json').exists():raise PermissionError('Boundary resume receipt already consumed; new reviewed authorization generation required')
   q.immutable(out/f'resume-used-{start:04d}.json',dict(authorization=bind(out/f'resume-authorization-{start:04d}.json')))
   torch,tok,model,head,opt=reload_saved(out,p,folder)
  else:
   if (out/'work').exists():raise PermissionError('Existing work requires safe CPU resume-check and explicit receipt')
   (out/'work').mkdir();torch,tok,model,head=t.lib.setup(unit);t.lib.activate(model,head,'actor');opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['lr'],weight_decay=0,foreach=False);folder=save_boundary(out,0,torch,model,head,opt)
   del model,head,opt;gc.collect();torch.cuda.empty_cache();torch,tok,model,head,opt=reload_saved(out,p,folder)
  fixed={r:t.lib.role_hash(model,r) for r in ['reference','discriminator']}
  if start in MILESTONES:
   q.immutable(out/'measurements'/f'step{start:04d}'/'independent-reload.json',dict(status='PASS_EXACT_ROLES_OPTIMIZER_RNG',checkpoint=bind(folder/'manifest.json'),logprob_error_not_measured=True))
   if not evaluate_milestone(out,p,start,torch,tok,model,head,data):return dict(status='PAUSED_AT_MEASUREMENT_BOUNDARY',step=start)
  for step in range(start+1,65):
   if stopped(out):return dict(status='PAUSED_AT_COMPLETE_CHECKPOINT',step=step-1)
   row=order_row(data['selected'],step);file=out/'work/updates'/f'{step:04d}.json';q.immutable(file.with_suffix('.started.json'),dict(id=row['id'],step=step,actor_before=t.lib.role_hash(model,'actor'),protocol_sha256=q.sha(out/'protocol.json')))
   prefix,answer=probe.packing(tok,row,data['teacher'][row['id']],p['condition'],data['fixed8']);t.lib.activate(model,head,'actor');model.train();opt.zero_grad(set_to_none=True);ids=torch.tensor([prefix+answer],device='cuda');before=t.lib.role_hash(model,'actor')
   loss=-t.lib.response_log_probs(t.lib.hidden(model,ids),model.get_output_embeddings().weight,ids,len(prefix),temperature=1.).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1)
   if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite tiny-fit objective/gradient')
   updated=float(norm)>0
   if updated:opt.step()
   opt.zero_grad(set_to_none=True);after=t.lib.role_hash(model,'actor')
   if updated and before==after:raise RuntimeError('Nonzero gradient but actor adapter unchanged')
   if {r:t.lib.role_hash(model,r) for r in fixed}!=fixed:raise RuntimeError('Tiny-fit changed frozen reference/D')
   metric=dict(step=step,id=row['id'],epoch=1+(step-1)//16,teacher_NLL_before_update=float(loss.detach()),gradient_norm_preclip=float(norm),optimizer_updated=updated,actor_before=before,actor_after=after,lr=opt.param_groups[0]['lr'],teacher_tokens=len(answer),input_tokens=len(prefix),gold_in_loss=False,reference_discriminator_unchanged=True)
   del ids,loss;folder=save_boundary(out,step,torch,model,head,opt,metric);t.atomic(out/'heartbeat.json',dict(epoch=time.time(),stage='TINY_FIT_UPDATE_COMMITTED',step=step,optimizer_updated=updated))
   if step in MILESTONES:
    del model,head,opt;gc.collect();torch.cuda.empty_cache();torch,tok,model,head,opt=reload_saved(out,p,folder)
    q.immutable(out/'measurements'/f'step{step:04d}'/'independent-reload.json',dict(status='PASS_EXACT_ROLES_OPTIMIZER_RNG',checkpoint=bind(folder/'manifest.json'),logprob_error_not_measured=True))
    if not evaluate_milestone(out,p,step,torch,tok,model,head,data):return dict(status='PAUSED_AT_MEASUREMENT_BOUNDARY',step=step)
  return summarize(out)

def main():
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','resume-check','summarize']);ap.add_argument('--task',choices=['USTBench','SGQA']);ap.add_argument('--model',choices=['08','2b']);ap.add_argument('--condition',choices=['zero-shot','matched8']);ap.add_argument('--lr',type=float,choices=LEARNING_RATES);ap.add_argument('--decision',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 if a.action=='prepare':
  if not all([a.task,a.model,a.condition,a.lr,a.decision]):ap.error('prepare requires task/model/condition/lr/decision')
  result=prepare(a.task,a.model,a.condition,a.lr,a.decision)
 else:
  if not a.output:ap.error('output required')
  out=owned(a.output)
  if a.action=='resume-check':verify(out);f,s=cursor(out);result=dict(status='SAFE_RESUME',step=s,checkpoint=bind(f/'manifest.json'),gpu_used=False)
  elif a.action=='summarize':result=summarize(out)
  else:result=run(out,a.resume)
 print(json.dumps(result))
if __name__=='__main__':main()


