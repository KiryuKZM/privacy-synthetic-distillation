#!/usr/bin/env python3
"""SG cached-teacher off-policy SeqKD200. Reuses frozen native trainer; no GAD."""
import argparse,ast,gc,json,math,os,signal,sys,time,types,weakref
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import tinyfit_qwen_train16_20261025 as tiny
q=tiny.q;t=tiny.t;probe=tiny.probe;NEW=tiny.NEW;BASE=NEW/'seqkd200/SGQA';PLAN=ROOT/'reports/qwen-transfer-repair-2026-10-25/seqkd200-sg-plan.md';STOP=False
TINY_SHA='9c41ec8e5e66e4929784c6e8a17151c0132afb8f310559501b27a1f557cea280';MILESTONES=[0,50,100,200]
if q.sha(ROOT/'scripts/tinyfit_qwen_train16_20261025.py')!=TINY_SHA:raise PermissionError('Frozen tiny-fit changed')
def bind(p):return tiny.bind(p)
def bound(s):return tiny.bound(s)
def owned(out):
 raw=Path(out).absolute();p=raw.resolve()
 if raw!=p or raw.is_symlink() or p.parent!=BASE.resolve() or p.name not in ['08','2b']:raise PermissionError('Owned SG SeqKD08/2b output only')
 return p
def stop(*_):
 global STOP
 STOP=True
LIVE={} 
def stop_requested(out):return STOP or any(p.exists() for p in [NEW/'stop.json',NEW/'root-stop.json',NEW/'global-stop.json',BASE/'stop.json',out/'stop.json',out/'OOM_STOP.json'])
def stopped(out):
 flag=stop_requested(out)
 if flag and LIVE.get('out')==out and LIVE.get('step',0)>0:
  latest=q.load(out/'work/latest.json')
  if t.lib.verify_checkpoint(out,Path(latest['path']))['step']<LIVE['step']:
   live=[LIVE[k]() for k in ['model','head','opt']]
   if any(x is None for x in live):raise PermissionError('Live boundary unavailable; no unsafe pause checkpoint')
   save_boundary(out,LIVE['step'],LIVE['torch'],*live)
 return flag
def data_inputs():
 manifest=probe.RUN/'data/SGQA.manifest.json';m=q.load(manifest);tm=probe.RUN/'teacher/SGQA/train/manifest.json';teacher=q.load(tm);rows=t.lib.rows(m['phases']['train']['queries']);refs=t.lib.rows(teacher['records_file']);by={r['id']:r for r in refs};prefix=q.load(bound(teacher['fixed8_prefix']))['messages']
 if len(rows)!=200 or len(by)!=200 or set(by)!={r['id'] for r in rows} or teacher.get('all_valid') is not True or any(not r['valid'] or not r['text'] or r['finish_reason']!='stop' for r in refs):raise PermissionError('Exactly200 valid references retained; no filtering')
 devpath=NEW/'data/SGQA.repair-development.manifest.json';dev=q.load(devpath)['dev32'];devrows=t.lib.rows(dev['queries']);fit=q.load(bound(q.load(probe.OUT/'protocol.json')['tasks']['SGQA']['inputs']))
 if len(devrows)!=32 or len({r['group'] for r in devrows})!=32 or {r['group'] for r in devrows}&{r['group'] for r in rows}:raise PermissionError('Fresh Dev32 group separation required')
 if any(r['messages'][0]!=prefix[0] or r['messages'][-1]['content']!=r['question'] for r in rows+devrows):raise PermissionError('Same system/fullfacts matched8 required')
 return m,teacher,dict(selected=rows,teacher=by,fixed8=prefix),dev,fit,[bind(manifest),m['phases']['train']['queries'],bind(tm),teacher['records_file'],teacher['fixed8_prefix'],bind(devpath),dev['queries'],dev['membership'],dev['input_freeze'],*teacher['bindings']]
def prepare(model):
 if model not in ['08','2b']:raise PermissionError('Only native08/2b')
 out=owned(BASE/model);m,teacher,data,dev,fit,bindings=data_inputs();unit=probe.RUN/'training/SGQA'/model;old=q.load(unit/'protocol.json');src=out/'inputs.json';fitpath=out/'fit16-inputs.json';q.immutable(src,data);q.immutable(fitpath,fit)
 # Tokenizer-only audit, no base weights/model/generation. Exactly same native chat template.
 q.setup_native();from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(old['base'],local_files_only=True,trust_remote_code=False);audit=[]
 for r in data['selected']:
  prefix,answer=probe.packing(tok,r,data['teacher'][r['id']],'matched8',data['fixed8']);audit.append(dict(id=r['id'],prompt_tokens=len(prefix),teacher_tokens=len(answer)))
 for r in t.lib.rows(dev['queries']):
  n=len(tok.encode(tok.apply_chat_template(data['fixed8']+[dict(role='user',content=r['question'])],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))
  if n+256>32768:raise PermissionError('Complete Dev exceeds context; no truncation')
  audit.append(dict(id=r['id'],prompt_tokens=n,dev=True))
 q.immutable(out/'token-audit.json',dict(items=audit,tokenizer_only=True,gpu_used=False))
 p=dict(schema='SG-native-offpolicy-SeqKD200-v1',task='SGQA',scorer='sgqa',model=model,unit_path=str(unit),condition='matched8',lr=1e-4,inputs=bind(src),fit16=bind(fitpath),development=dev,common_eval=m['common_eval'],base=old['base'],base_assets=old['base_assets'],runtime=old['runtime'],seed=20261024,training_order=[r['id'] for r in data['selected']],train_count=200,epochs=1,maximum_optimizer_attempts=200,milestones=MILESTONES,selection_steps=[50,100,200],selection='correct,valid,fewer capped,earlier; baseline0 never finetuned',teacher_final_only=True,teacher_train_mode=teacher['mode'],gold_in_loss=False,no_teacher_filtering=True,native_vocabulary=True,thinking='off',lora=dict(r=8,alpha=16,target='same6fullattention q/k/v/o',dropout=0),optimizer='AdamW',clip_norm=1,weight_decay=0,checkpoint_every_updates=5,checkpoint_on_pause=True,fit16_steps=[0,'selected'],Dev32_answers_per_model=128,fit_answers_per_model=32,CommonEval_answers_per_model=60,global_output_bytes_max=8*2**30,supervision='off-policy cached-teacher sequence KD, not GAD',Confirm64_read=False,Final_read=False,new_api_calls=0,bindings=[bind(__file__),bind(ROOT/'scripts/tinyfit_qwen_train16_20261025.py'),bind(ROOT/'scripts/probe_overnight_train_fit_20261025.py'),bind(ROOT/'scripts/train_overnight_native_gad_20261025.py'),bind(ROOT/'scripts/run_overnight_native_gad_20261025.py'),bind(PLAN),bind(NEW/'user-goal-approval.json'),bind(unit/'protocol.json'),bind(src),bind(fitpath),bind(out/'token-audit.json'),*bindings])
 q.immutable(out/'protocol.json',p);return dict(status='PREPARED_ZERO_GPU',protocol=bind(out/'protocol.json'))
def verify(out):
 out=owned(out);p=q.load(out/'protocol.json');a=q.load(out/'authorization.json')
 if not a.get('approved') or a.get('protocol_sha256')!=q.sha(out/'protocol.json'):raise PermissionError('Root exact SeqKD protocol authorization needed')
 for s in p['bindings']:bound(s)
 if (p['task'],p['model'],p['lr'],p['maximum_optimizer_attempts'],p['milestones'])!=('SGQA',out.name,1e-4,200,MILESTONES):raise PermissionError('Fixed SeqKD scope changed')
 return p

def order_row(rows,step):
 if len(rows)!=200 or not 1<=step<=200:raise PermissionError('Exactlyone Train200 epoch; no cycling')
 return rows[step-1]
def cursor(out):
 out=owned(out)
 if tiny.unresolved(out):raise PermissionError('PARTIAL_UNSAFE: unpaired operation; no replay')
 ptr=q.load(out/'work/latest.json');folder=Path(ptr['path'])
 if folder.is_symlink() or folder.resolve().parent!=(out/'work/checkpoints').resolve() or q.sha(folder/'manifest.json')!=ptr['manifest_sha256']:raise PermissionError('Checkpoint pointer/hash differs')
 cm=t.lib.verify_checkpoint(out,folder);step=cm['step'];logs=sorted(int(f.stem) for f in (out/'work/updates').glob('*.json') if '.started.' not in f.name)
 if cm['stage']!='SEQKD200' or not 0<=step<=200 or logs!=list(range(1,step+1)):raise PermissionError('Unsafe worklog/checkpoint cursor mismatch')
 return folder,step

def retain(out,latest):
 # Reuse exact safe retention implementation with this protocol's milestone set.
 generations=sorted(int(f.name.removeprefix('step')) for f in (out/'work/checkpoints').iterdir() if f.is_dir());prior=max([x for x in generations if x<latest]+[0])
 ns=dict(tiny.__dict__,MILESTONES=MILESTONES+[prior]);fn=types.FunctionType(tiny.retain.__code__,ns,'seqkd_retain');fn(out,latest)
def save_boundary(out,step,torch,model,head,opt,metric=None):
 save=step%5==0 or step in MILESTONES or stop_requested(out);folder=out/'work/checkpoints'/f'step{step:04d}'
 if save:
  if t.size_tree(NEW)+64*2**20>8*2**30:raise RuntimeError('GLOBAL_SEQKD_OUTPUT_STAGING_8GIB_CAP')
  old=[q.load(f) for f in (out/'work/updates').glob('*.json') if '.started.' not in f.name];actual=sum(r['optimizer_updated'] for r in old)+int(metric['optimizer_updated'] if metric else 0)
  tiny.SAVE_CHECKPOINT(out,folder,torch,model,head,dict(actor=opt),'SEQKD200',step,actual_updates=actual);t.lib.verify_checkpoint(out,folder)
 if metric is not None:q.immutable(out/'work/updates'/f'{step:04d}.json',metric)
 if save:t.atomic(out/'work/latest.json',dict(path=str(folder),manifest_sha256=q.sha(folder/'manifest.json')));retain(out,step)
 LIVE.update(out=out,step=step,torch=torch,model=weakref.ref(model),head=weakref.ref(head),opt=weakref.ref(opt))
 return folder if save else Path(q.load(out/'work/latest.json')['path'])


def checkpoint_rows(out,folder,torch,tok,model,head,rows,prefix,kind):
 records=[];p=verify(out);model.eval()
 for i,row in enumerate(rows):
  path=folder/(q.digest(row['id'])+'.json')
  if path.exists():records.append(q.load(path));continue
  if path.with_suffix('.started.json').exists():raise PermissionError('Unknown saved answer; no resampling')
  if stopped(out):return None
  q.immutable(path.with_suffix('.started.json'),dict(id=row['id'],protocol_sha256=q.sha(out/'protocol.json'),kind=kind));t.context(out,kind,i)
  messages=prefix+[dict(role='user',content=row['question'])] if kind!='common' else row['adapted_messages']
  response=t.rollouts(torch,tok,model,head,row,n=1,greedy=True,messages=messages)[0];parsed=q.classify(response['raw_generated'],'off','sgqa',row.get('metadata',{}),response['finish_reason'])
  record=dict(id=row['id'],**parsed,task='SGQA',model=p['model'],raw_generated=response['raw_generated'],output_ids=response['response_ids'],input_tokens=response['prompt_length'],output_tokens=len(response['response_ids']),finish_reason=response['finish_reason'],generation_seconds=response['generation_seconds'],max_new_tokens=256,thinking='off',input_truncated=False,protocol_sha256=q.sha(out/'protocol.json'),actor_sha256=t.lib.role_hash(model,'actor'))
  q.immutable(path,record);records.append(record);t.atomic(out/'heartbeat.json',dict(epoch=time.time(),stage=kind,id=row['id']))
 q.immutable(folder/'predictions-frozen.json',dict(complete=True,count=len(rows),predictions=[bind(folder/(q.digest(r['id'])+'.json')) for r in rows],protocol_sha256=q.sha(out/'protocol.json'),gold_not_yet_read=True));return records

def eval_dev(out,p,step,torch,tok,model,head,data):
 folder=out/'development'/f'step{step:04d}';rows=t.lib.rows(p['development']['queries']);records=checkpoint_rows(out,folder,torch,tok,model,head,rows,data['fixed8'],'dev-step'+str(step))
 if records is None:return False
 if len(records)!=32 or len({r['id'] for r in records})!=32:raise PermissionError('Fixed fresh Dev32 denominator')
 gold=q.load(bound(p['development']['gold']));by={r['id']:r for r in rows};pairs=[probe.score_record(r,'sgqa',gold[r['id']],by[r['id']]['metadata']) for r in records]
 q.immutable(folder/'scores.json',dict(correct=sum(c for c,v in pairs),valid=sum(v for c,v in pairs),total=32,length_capped=sum(r['finish_reason']=='length' for r in records),freeze=bind(folder/'predictions-frozen.json'),step=step,role='fresh-procedural-Dev32-not-Confirm'))
 if step==0:
  ns=dict(tiny.__dict__,stopped=stopped);f=types.FunctionType(tiny.evaluate_milestone.__code__,ns,'seqkd_fit16');return f(out,p,step,torch,tok,model,head,q.load(bound(p['fit16'])))
 return True

def choose(out):
 p=verify(out);candidates=[]
 for step in [50,100,200]:
  value=q.load(out/'development'/f'step{step:04d}'/'scores.json');candidates.append(((-value['correct'],-value['valid'],value['length_capped'],step),step,value))
 _,step,value=min(candidates);folder=out/'work/checkpoints'/f'step{step:04d}';t.lib.verify_checkpoint(out,folder)
 result=dict(step=step,checkpoint=str(folder),checkpoint_manifest=bind(folder/'manifest.json'),development=value,baseline0_excluded=True,protocol=bind(out/'protocol.json'));q.immutable(out/'selected.json',result);return result

def summarize(out):
 p=verify(out);logs=[q.load(f) for f in sorted((out/'work/updates').glob('*.json')) if '.started.' not in f.name]
 if len(logs)!=200 or [r['id'] for r in logs]!=p['training_order']:raise PermissionError('Exactlyone fixed Train200 epoch required')
 result=dict(status='TRAINING_COMPLETE_OFFPOLICY_SEQKD',task='SGQA',model=p['model'],optimizer_attempts=200,actual_optimizer_updates=sum(r['optimizer_updated'] for r in logs),no_GAD=True,no_API=True,baseline_dev=q.load(out/'development/step0000/scores.json'),selection=choose(out),not_Confirm_or_Final=True)
 q.immutable(out/'training-complete.json',result);return result

def specialize_run():
 source=ROOT/'scripts/tinyfit_qwen_train16_20261025.py'
 if q.sha(source)!=TINY_SHA:raise PermissionError('Frozen reuse source changed')
 node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
 class OnlyTwo(ast.NodeTransformer):
  ranges=0;epochs=0
  def visit_Call(self,n):
   self.generic_visit(n)
   if isinstance(n.func,ast.Name) and n.func.id=='range' and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value==65:n.args[1]=ast.Constant(201);self.ranges+=1
   return n
  def visit_keyword(self,n):
   self.generic_visit(n)
   if n.arg=='epoch' and ast.dump(n.value)==ast.dump(ast.parse('1+(step-1)//16',mode='eval').body):
    n.value=ast.Constant(1);self.epochs+=1
   return n
 tr=OnlyTwo();node=tr.visit(node)
 if (tr.ranges,tr.epochs)!=(1,1):raise PermissionError('Frozen run must have exactly2 documented adaptations')
 ns=dict(tiny.__dict__,owned=owned,verify=verify,stopped=stopped,cursor=cursor,order_row=order_row,save_boundary=save_boundary,evaluate_milestone=eval_dev,summarize=summarize,MILESTONES=MILESTONES)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),ns);return ns['run']
TRAIN=specialize_run()

def reload_selected(out):
 p=verify(out);selected=q.load(out/'selected.json');folder=Path(selected['checkpoint']);torch,tok,model,head,opt=tiny.reload_saved(out,p,folder)
 ns=dict(tiny.__dict__,stopped=stopped);f=types.FunctionType(tiny.evaluate_milestone.__code__,ns,'seqkd_selected_fit16')
 if not f(out,p,selected['step'],torch,tok,model,head,q.load(bound(p['fit16']))):return dict(status='PAUSED_SELECTED_FIT16')
 records=[q.load(bound(s)) for s in q.load(out/'measurements'/f"step{selected['step']:04d}"/'predictions-frozen.json')['predictions']];q.immutable(out/'selected-reload.json',dict(status='PASS',selected=bind(out/'selected.json'),checkpoint_manifest=bind(folder/'manifest.json'),roles_optimizer_RNG_exact=True,fit16_teacher_agreement=sum(r['teacher_prediction_agreement'] for r in records),fit16_valid=sum(r['valid'] for r in records),fit16_total=16,not_generalization=True));return dict(status='SELECTED_RELOADED')

def evaluate_common(out):
 p=verify(out);frozen=q.load(BASE/'both-size-selections-frozen.json')
 if frozen.get('all2_before_CommonEval') is not True or set(frozen['models'])!={'08','2b'}:raise PermissionError('Bothsize selection/reload freeze beforeCommon')
 for specs in frozen['models'].values():
  for spec in specs.values():bound(spec)
 sel=q.load(out/'selected.json');folder=Path(sel['checkpoint']);torch,tok,model,head,opt=tiny.reload_saved(out,p,folder);rows=t.lib.rows(p['common_eval']['queries']);records=checkpoint_rows(out,out/'common',torch,tok,model,head,rows,[], 'common')
 if records is None:return dict(status='PAUSED_COMMON_ANSWER_BOUNDARY')
 if len(records)!=60 or len({r['id'] for r in records})!=60:raise PermissionError('ExactlyCommon60')
 return dict(status='COMMON60_FROZEN_UNSCORED',freeze=bind(out/'common/predictions-frozen.json'))

def main():
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','resume-check','reload','common']);ap.add_argument('--model',choices=['08','2b'],required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();out=BASE/a.model
 if a.action=='prepare':result=prepare(a.model)
 elif a.action=='resume-check':verify(out);f,s=cursor(out);result=dict(status='SAFE_RESUME',step=s,checkpoint=bind(f/'manifest.json'),gpu_used=False)
 else:
  from night_dual_20261022 import lock,cgroup_limits_verified
  import run_cityfm_repair_qwen_20261022 as guard
  if stopped(out):print(json.dumps(dict(status='PAUSED_BEFORE_GPU')));return
  # TRAIN already owns these same guards; other GPU actions use them here.
  try:
   if a.action=='train':result=TRAIN(out,a.resume)
   else:
    with lock(out/'action.lock'),lock(guard.HISTORICAL/'locks/gpu.lock'):
     cgroup_limits_verified();result=reload_selected(out) if a.action=='reload' else evaluate_common(out)
  except Exception as e:
   oom=type(e).__name__=='OutOfMemoryError' or 'out of memory' in str(e).lower();q.immutable(out/'failures'/f'{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),OOM=oom,unknown_started=tiny.unresolved(out),no_retry=True))
   if sum(q.load(f).get('OOM',False) for f in (out/'failures').glob('*.json'))>=2:q.immutable(out/'OOM_STOP.json',dict(reason='TWO_OOM_AFFECTED_PHASE_STOP',no_retry=True))
   raise
 print(json.dumps(result))
if __name__=='__main__':main()


