#!/usr/bin/env python3
"""Versioned four-unit native GAD. Reuses frozen tensor/role/loader implementations."""
from __future__ import annotations
import argparse,contextlib,gc,hashlib,importlib.util,json,os,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import eval_ust_sg_qwen_modes_20261024 as q
RUN=ROOT/'runs/2026-10-25-overnight-onpolicy'
PLAN=ROOT/'reports/overnight-onpolicy-2026-10-25/plan.md'
PLAN_SHA='b091ae438f2d2c7c4b641f0161c2a1beb5004f4d4dbe8eb350dce00e60f4239b'
IMPL=ROOT/'scripts/train_ust_sg_native_gad_20261024.py'
IMPL_SHA='fd04244b6d288dd3b6412a87eaca46f319d7dfe950ae50346959b419b8c8ce16'
MODELS={'08':ROOT/'artifacts/hf-home/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17','2b':q.DEFAULT_MODEL}
if q.sha(IMPL)!=IMPL_SHA:raise PermissionError('Frozen reusable trainer changed')
spec=importlib.util.spec_from_file_location('_overnight_native_gad_impl',IMPL)
lib=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lib;spec.loader.exec_module(lib)
STOP=False;CTX={}
class SavedPause(RuntimeError):pass

def stop(*_):
 global STOP
 STOP=True

def paused():return STOP or any((RUN/n).exists() for n in ('stop.json','global-stop.json'))
def bind(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=q.sha(p))
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');t.replace(p)
def size_tree(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())
def unit_path(out):
 raw=Path(out).absolute();out=raw.resolve()
 if raw.is_symlink() or out!=raw or not out.is_relative_to((RUN/'training').resolve()):raise PermissionError('Owned unit directory required')
 return out

def policy_rows(manifest,tr,task):
 p=q.load(manifest)
 if 'policy_train50' in p:
  spec=p['policy_train50'];member=q.load(q.bound(spec['membership']));subset=lib.rows(spec['queries'])
  ids0=[str(r['id']) for r in subset]
  if ids0!=[str(r['id']) for r in member['selected']]:raise PermissionError('Policy membership order differs')
  by0={str(r['id']):r for r in tr}
  if any(str(r['id']) not in by0 or q.digest(r)!=q.digest(by0[str(r['id'])]) for r in subset):raise PermissionError('Policy row differs from original Train facts')
  p=dict(task=p['task'],count=spec['count'],ids=ids0)
 if p.get('task')!=task or p.get('count')!=50:raise PermissionError('Fixed PolicyTrain50 required')
 ids=[str(x) for x in p['ids']]
 if len(ids)!=50 or len(set(ids))!=50:raise PermissionError('Policy50 duplicate/count')
 by={str(r['id']):r for r in tr}
 if not set(ids)<=set(by):raise PermissionError('Policy outside Train200')
 rows=[by[k] for k in ids]
 if len({r['group'] for r in rows})!=50:raise PermissionError('Policy component overlap')
 from collections import Counter
 field='ability' if task=='USTBench' else 'question_type'
 counts=Counter(r['metadata'][field] for r in rows)
 if task=='USTBench' and counts!=dict(st_understanding=17,planning=17,reflection=16):raise PermissionError('UST fixed17/17/16 policy strata')
 if task=='SGQA' and (len(counts)!=10 or set(counts.values())!={5}):raise PermissionError('SG fixed10x5 policy strata')
 return rows

def protocol(model,manifest,teacher,policy_manifest):
 if model not in MODELS or q.sha(PLAN)!=PLAN_SHA:raise PermissionError('Model/approved plan changed')
 approval=q.load(RUN/'user-approval.json')
 if approval.get('approved') is not True or approval['plan']['sha256']!=PLAN_SHA:raise PermissionError('New execution approval required')
 q.bound(approval['plan']);m,tr,va,_=lib.data(manifest,teacher)
 pol=policy_rows(policy_manifest,tr,m['task'])
 assets=q.model_assets(MODELS[model])
 if assets!=m['native_assets'][model]:raise PermissionError('Model assets differ from frozen native revision')
 if len(tr)!=200 or len(va)!=32:raise PermissionError('Fixed Train200/Val32')
 return dict(schema='overnight-native-four-unit-v1',task=m['task'],model=model,base=str(MODELS[model]),base_assets=assets,
  manifest_path=str(Path(manifest).resolve()),manifest_sha256=q.sha(manifest),teacher_manifest_path=str(Path(teacher).resolve()),teacher_manifest_sha256=q.sha(teacher),
  policy_manifest_path=str(Path(policy_manifest).resolve()),policy_manifest_sha256=q.sha(policy_manifest),policy_ids=[str(r['id']) for r in pol],
  trainer_sha256=q.sha(__file__),dependencies={str(ROOT/x):q.sha(ROOT/x) for x in ['src/privacy_synth/onpolicy_gad.py','scripts/bird_gad_night_20261022.py','scripts/preflight_qwen35_lora.py','scripts/validate_qwen_thinking_budget_v3_20261024.py','scripts/run_ust_sg_fast_profile_20261024.py','scripts/night_dual_20261022.py','scripts/run_cityfm_repair_qwen_20261022.py']},reused_implementation=bind(IMPL),runtime=q.runtime_bindings(),approval=bind(RUN/'user-approval.json'),plan=bind(PLAN),
  train_count=200,validation_count=32,policy_groups=50,group_size=8,actor_sft_epochs=1,sequential_discriminator_warmup_epochs=1,
  validation_steps=[0,25,50],selection_steps=[25,50],selection='correct then valid then fewer capped then earlier',new_lora_r=8,lora_alpha=16,lora_dropout=0,
  rollout_max_new_tokens=256,temperature=.8,top_k=0,top_p=1.,actor_lr=1e-6,discriminator_lr=1e-6,clip=.2,beta=.001,thinking='off',native_vocabulary=True,
  checkpoint_each_complete_update=True,combined_policy_actor_D_commit=True,latest_generations=2,policy_archives=[10,20,25,30,40,50],output_global_cap=8*2**30,
  external_calls=0,prior_adapter_transferred=False,old320_gate=False)

def prepare(model,manifest,teacher,policy_manifest,out):
 out=unit_path(out);p=protocol(model,manifest,teacher,policy_manifest);q.immutable(out/'protocol.json',p)
 q.setup_native();from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(p['base'],local_files_only=True,trust_remote_code=False)
 _,tr,va,t=lib.data(manifest,teacher)
 q.immutable(out/'token-audit.json',dict(train=[dict(id=str(r['id']),prompt_tokens=len(lib.prompt(tok,r)),teacher_tokens=len(lib.pack(tok,r,t[str(r['id'])]['text'])['input_ids'])-len(lib.prompt(tok,r))) for r in tr],validation=[dict(id=str(r['id']),prompt_tokens=len(lib.prompt(tok,r))) for r in va],tokenizer_only=True,gpu_used=False,protocol_sha256=q.sha(out/'protocol.json')))
 return dict(status='PREPARED_ZERO_GPU',protocol=bind(out/'protocol.json'))
def verify(out):
 out=unit_path(out);p=q.load(out/'protocol.json')
 if protocol(p['model'],p['manifest_path'],p['teacher_manifest_path'],p['policy_manifest_path'])!=p:raise PermissionError('Frozen unit configuration changed')
 a=q.load(out/'authorization.json')
 if a.get('approved') is not True or a.get('protocol_sha256')!=q.sha(out/'protocol.json'):raise PermissionError('Root-reviewed unit authorization absent')
 return p

def context(out,stage,step):CTX.update(out=Path(out),stage=stage,step=step)
def rollout_seed(p,row,index):return int(q.digest(['overnight-gad-v1',p['model'],p['task'],CTX['stage'],CTX['step'],str(row['id']),index])[:8],16)
def rollouts(torch,tok,model,head,row,n=8,greedy=False,thinking='off',messages=None):
 if thinking!='off':raise PermissionError('Only frozen off mainline; optional on needs separate bounded protocol')
 out=CTX['out'];p=q.load(out/'protocol.json');lib.activate(model,head,'actor');model.eval()
 prefix=lib.prompt(tok,row) if messages is None else tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
 if len(prefix)+256>32768:raise PermissionError('Complete context overflow; no clipping')
 stops=[tok.convert_tokens_to_ids(x) for x in ('<|im_end|>','<|endoftext|>')];result=[]
 for index in range(n):
  target=out/'rollouts'/CTX['stage']/f"{CTX['step']:04d}"/q.digest(str(row['id']))/f'{index:02d}.json'
  identity=dict(actor_before=lib.role_hash(model,'actor'),protocol_sha256=q.sha(out/'protocol.json'),model=p['model'],task=p['task'],stage=CTX['stage'],step=CTX['step'],id=str(row['id']),group_index=index,seed=rollout_seed(p,row,index),input_ids_sha256=q.digest(prefix),greedy=greedy,temperature=None if greedy else .8,max_new_tokens=256)
  if target.exists():
   old=q.load(target)
   if old['identity']!=identity:raise PermissionError('Saved fresh rollout binding differs')
   result.append(old['response']);continue
  if target.with_suffix('.started.json').exists():raise PermissionError('Unresolved fresh rollout; no resampling')
  q.immutable(target.with_suffix('.started.json'),identity)
  torch.manual_seed(identity['seed']);ids=torch.tensor([prefix],device='cuda');start=time.monotonic()
  kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=not greedy,num_beams=1,eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)
  if not greedy:kwargs.update(temperature=.8,top_k=0,top_p=1.)
  with torch.inference_mode():generated=model.generate(**kwargs)
  emitted=generated[0,len(prefix):].tolist()
  if not emitted:raise RuntimeError('Empty generated response')
  response=dict(input_ids=prefix+emitted,prompt_length=len(prefix),response_ids=emitted,content=tok.decode(emitted,skip_special_tokens=True),raw_generated=tok.decode(emitted,skip_special_tokens=False),finish_reason='stop' if emitted[-1] in stops else 'length',group_index=index,generation_seconds=time.monotonic()-start)
  q.immutable(target,dict(identity=identity,response=response));result.append(response);del ids,generated
  atomic(out/'heartbeat.json',dict(epoch=time.time(),stage=CTX['stage'],step=CTX['step'],rollout=index+1,group=n,api_calls=0))
 return result

def tensor_bytes(value):
 if hasattr(value,'numel'):return value.numel()*value.element_size()
 if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
 if isinstance(value,(tuple,list)):return sum(map(tensor_bytes,value))
 return 0

def checkpoint(out,folder,torch,model,head,optimizers,stage,step,actual_updates=0):
 out=unit_path(out);folder=Path(folder)
 if folder.exists() or not folder.resolve().is_relative_to(out):raise PermissionError('Checkpoint overwrite/path escape')
 estimate=sum(v.numel()*v.element_size() for n,v in model.named_parameters() if 'lora_' in n)+sum(tensor_bytes(o.state_dict()) for o in optimizers.values())+64*2**20
 if size_tree(RUN)+estimate>8*2**30:raise RuntimeError('GLOBAL_OUTPUT_STAGING_8GIB_CAP')
 folder.mkdir(parents=True)
 model.save_pretrained(folder,selected_adapters=['actor','reference','discriminator'])
 torch.save(head.state_dict(),folder/'score-head.pt');torch.save({k:v.state_dict() for k,v in optimizers.items()},folder/'optimizers.pt')
 torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),folder/'rng.pt')
 q.immutable(folder/'manifest.json',dict(protocol_sha256=q.sha(out/'protocol.json'),stage=stage,step=step,actual_updates=actual_updates,
  data_order_sha256=q.digest(q.load(out/'protocol.json')['policy_ids']),combined_actor_discriminator_commit=True,
  role_hashes={r:lib.role_hash(model,r) for r in ('actor','reference','discriminator')},optimizer_hashes={k:lib.nested_hash(o.state_dict()) for k,o in optimizers.items()},files={str(f.relative_to(folder)):q.sha(f) for f in folder.rglob('*') if f.is_file()}))
 lib.verify_checkpoint(out,folder)

def retention(out):
 out=unit_path(out);work=out/'training';base=work/'checkpoints';pointer=q.load(work/'latest-checkpoint.json')
 keep={Path(pointer['path']).name}
 if pointer.get('previous'):keep.add(Path(pointer['previous']['path']).name)
 keep.update(['sft0200','W0200','D0200']+[f'policy{i:04d}' for i in (10,20,25,30,40,50)])
 if (out/'selected.json').exists():keep.add(Path(q.load(out/'selected.json')['checkpoint']).name)
 for r in (work/'pause-receipts').glob('*.json'):keep.add(Path(q.load(r)['checkpoint']).name)
 for folder in base.iterdir() if base.exists() else []:
  if not folder.is_dir() or folder.name in keep:continue
  if folder.is_symlink() or folder.resolve().parent!=base.resolve():raise PermissionError('Checkpoint path escape')
  cm=lib.verify_checkpoint(out,folder);actual={str(f.relative_to(folder)) for f in folder.rglob('*') if f.is_file()}
  if actual!=set(cm['files'])|{'manifest.json'} or any(f.is_symlink() for f in folder.rglob('*')):raise PermissionError('Unexpected checkpoint payload')
  q.immutable(work/'checkpoint-tombstones'/(folder.name+'.json'),dict(path=str(folder),manifest_sha256=q.sha(folder/'manifest.json'),manifest=cm,reason='latest2 plus frozen archives/selection/pause',kept=sorted(keep)))
  shutil.rmtree(folder)
 if size_tree(RUN)>8*2**30:raise RuntimeError('GLOBAL_OUTPUT_8GIB_CAP')
 return keep

def resume_cursor(out):
 out=unit_path(out);work=out/'training'
 if not (work/'latest-checkpoint.json').exists():
  if any(f.is_file() for f in work.rglob('*')):raise PermissionError('PARTIAL_UNSAFE_NO_COMPLETE_CHECKPOINT')
  return None,dict(stage='sft',step=0,actual_updates=0)
 pointer=q.load(work/'latest-checkpoint.json');folder=Path(pointer['path'])
 if folder.is_symlink() or folder.resolve().parent!=(work/'checkpoints').resolve():raise PermissionError('Checkpoint pointer escaped owned unit')
 if q.sha(folder/'manifest.json')!=pointer['manifest_sha256']:raise PermissionError('Pointer hash changed')
 cm=lib.verify_checkpoint(out,folder);stage,step=cm['stage'],cm['step']
 for marker in out.rglob('*.started.json'):
  if not marker.with_name(marker.name.replace('.started.json','.json')).exists():raise PermissionError('PARTIAL_UNSAFE_UNRESOLVED_TRANSACTION_NO_REPLAY')
 limits={'sft':0,'warmup':0,'policy':0}
 if stage=='sft':limits['sft']=step
 elif stage=='W':limits['sft']=200
 elif stage=='D':limits.update(sft=200,warmup=step)
 elif stage=='policy':limits.update(sft=200,warmup=200,policy=step)
 else:raise PermissionError('Unsafe cursor stage')
 if not (1<=step<= (50 if stage=='policy' else 200)):raise PermissionError('Cursor beyond fixed stage')
 for phase,n in limits.items():
  completed=sorted(int(f.stem) for f in (work/phase).glob('*.json') if not f.name.endswith('.started.json'))
  if completed!=list(range(1,n+1)):raise PermissionError('PARTIAL_UNSAFE_LOG_AHEAD_OR_BEHIND_CHECKPOINT')
 if cm.get('data_order_sha256')!=q.digest(q.load(out/'protocol.json')['policy_ids']):raise PermissionError('Sampler order changed')
 return folder,cm

def resume_check(out):
 verify(out);folder,cursor=resume_cursor(out)
 return dict(status='SAFE_RESUME',checkpoint=str(folder) if folder else None,checkpoint_manifest_sha256=q.sha(folder/'manifest.json') if folder else None,stage=cursor['stage'],step=cursor['step'],gpu_used=False)

def validation(out,folder,torch,tok,model,head,items,m):
 from run_llm_task_sweep_20261023 import score_prediction
 q.immutable(folder/'contract.json',dict(protocol_sha256=q.sha(out/'protocol.json'),actor_sha256=lib.role_hash(model,'actor')));result=[]
 for i,row in enumerate(items):
  path=folder/(q.digest(str(row['id']))+'.json')
  if path.exists():result.append(q.load(path));continue
  if path.with_suffix('.started.json').exists():raise PermissionError('Unresolved Val answer; no resampling')
  if paused():raise SavedPause('Val paused at saved answer boundary')
  q.immutable(path.with_suffix('.started.json'),dict(id=str(row['id'])));context(out,'val-'+folder.name,i)
  response=rollouts(torch,tok,model,head,row,n=1,greedy=True)[0]
  parsed=q.classify(response['raw_generated'],'off',m['phases']['validation']['scorer'],row.get('metadata',{}),response['finish_reason'])
  record=dict(id=str(row['id']),**parsed,finish_reason=response['finish_reason']);q.immutable(path,record);result.append(record)
 q.immutable(folder/'predictions-frozen.json',dict(complete=True,count=32,predictions=[dict(path=str(folder/(q.digest(str(r['id']))+'.json')),sha256=q.sha(folder/(q.digest(str(r['id']))+'.json'))) for r in items]))
 gold=q.load(q.bound(m['phases']['validation']['gold']));pairs=[score_prediction(r,dict(kind='qa',scorer=m['phases']['validation']['scorer']),gold[r['id']],items[i].get('metadata',{})) for i,r in enumerate(result)]
 value=dict(correct=sum(c for c,v in pairs),valid=sum(v for c,v in pairs),total=32,length_capped=sum(r['finish_reason']=='length' for r in result))
 q.immutable(folder/'scores.json',value);return value

def train(out,resume=False):
 p=verify(out)
 if q.load(out/'gpu-preflight.json')['status']!='PASS':raise PermissionError('Fresh functional GPU preflight required')
 work=out/'training';folder=None;cursor=dict(stage='sft',step=0,actual_updates=0)
 if resume:folder,cursor=resume_cursor(out)
 elif work.exists():raise PermissionError('Existing training requires explicit safe resume')
 else:work.mkdir()
 m,tr,va,teacher=lib.data(p['manifest_path'],p['teacher_manifest_path']);pol=policy_rows(p['policy_manifest_path'],tr,p['task'])
 torch,tok,model,head=lib.setup(out,folder);opts=lib.optimizers(torch,model,head)
 if folder:lib.restore(torch,model,head,opts,out,folder)
 actual=cursor.get('actual_updates',0)
 def boundary(stage,i):
  target=work/'checkpoints'/f'{stage}{i:04d}';checkpoint(out,target,torch,model,head,opts,stage,i,actual)
  old=q.load(work/'latest-checkpoint.json') if (work/'latest-checkpoint.json').exists() else None
  pointer=dict(path=str(target),manifest_sha256=q.sha(target/'manifest.json'))
  if old:
   previous=Path(old['path'])
   if q.sha(previous/'manifest.json')!=old['manifest_sha256']:raise PermissionError('Previous complete generation changed')
   lib.verify_checkpoint(out,previous);pointer['previous']={k:old[k] for k in ('path','manifest_sha256')}
  q.immutable(work/'commits'/f'{stage}{i:04d}.json',pointer);atomic(work/'latest-checkpoint.json',pointer);retention(out)
  atomic(out/'heartbeat.json',dict(epoch=time.time(),stage=stage,step=i,actual_optimizer_updates=actual,checkpoint=str(target),api_calls=0))
  if paused():
   q.immutable(work/'pause-receipts'/f'{stage}{i:04d}.json',dict(checkpoint=str(target),manifest_sha256=pointer['manifest_sha256'],reason='USER_STOP_SAVED_COMPLETE_BOUNDARY'))
   raise SavedPause('USER_STOP_SAVED_COMPLETE_BOUNDARY')
 def before(phase,i,row):
  if paused():raise SavedPause('USER_STOP_BEFORE_TRANSACTION')
  q.immutable(work/phase/f'{i:04d}.started.json',dict(id=str(row['id']),stage=phase,actor_before=lib.role_hash(model,'actor'),reference_before=lib.role_hash(model,'reference'),protocol_sha256=q.sha(out/'protocol.json')));context(out,phase,i)
 if cursor['stage']=='sft':
  for i,row in enumerate(tr,1):
   if i<=cursor['step']:continue
   before('sft',i,row);lib.activate(model,head,'actor');model.train();opt=opts['actor'];opt.zero_grad(set_to_none=True)
   packed=lib.pack(tok,row,teacher[str(row['id'])]['text']);ids=torch.tensor([packed['input_ids']],device='cuda')
   loss=-lib.response_log_probs(lib.hidden(model,ids),model.get_output_embeddings().weight,ids,packed['prompt_length'],temperature=1.).mean();loss.backward()
   norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1)
   if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite SFT')
   opt.step();opt.zero_grad(set_to_none=True);q.immutable(work/'sft'/f'{i:04d}.json',dict(id=str(row['id']),loss=float(loss.detach()),gradient_norm=float(norm)));del ids,loss
   boundary('sft',i)
  lib.clone_warmed(model);opts=lib.optimizers(torch,model,head);boundary('W',200);cursor=dict(stage='W',step=200)
 if cursor['stage']=='W':validation(out,work/'validation/W',torch,tok,model,head,va,m)
 reference=lib.role_hash(model,'reference')
 if cursor['stage'] in ('sft','W','D'):
  start=cursor['step'] if cursor['stage']=='D' else 0
  for i,row in enumerate(tr,1):
   if i<=start:continue
   before('warmup',i,row);fresh=rollouts(torch,tok,model,head,row,n=1)
   metric=update_d(torch,model,head,opts['discriminator'],lib.pack(tok,row,teacher[str(row['id'])]['text'])['input_ids'],fresh)
   if lib.role_hash(model,'reference')!=reference:raise RuntimeError('W reference changed during D warmup')
   q.immutable(work/'warmup'/f'{i:04d}.json',dict(id=str(row['id']),fresh_responses=fresh,**metric));boundary('D',i)
  cursor=dict(stage='policy',step=0)
 start=cursor['step'] if cursor['stage']=='policy' else 0
 if start in (25,50) and not (work/f'validation/policy{start:04d}/scores.json').exists():validation(out,work/'validation'/f'policy{start:04d}',torch,tok,model,head,va,m)
 for i,row in enumerate(pol,1):
  if i<=start:continue
  before('policy',i,row);fresh=rollouts(torch,tok,model,head,row,n=8)
  metric,rewards,advantages=lib.policy(torch,model,head,opts['actor'],fresh)
  dm=update_d(torch,model,head,opts['discriminator'],lib.pack(tok,row,teacher[str(row['id'])]['text'])['input_ids'],fresh)
  if lib.role_hash(model,'reference')!=reference:raise RuntimeError('Frozen W reference changed')
  actual+=int(metric['optimizer_updated']);q.immutable(work/'policy'/f'{i:04d}.json',dict(id=str(row['id']),fresh_responses=fresh,rewards=rewards,advantages=advantages,actor=metric,discriminator=dm))
  boundary('policy',i)
  if i in (25,50):validation(out,work/'validation'/f'policy{i:04d}',torch,tok,model,head,va,m)
 signals=lib.policy_signal_summary(work,50)
 if signals['actual_optimizer_updates']!=actual:raise RuntimeError('Actual update log mismatch')
 d_records=[q.load(work/'warmup'/f'{i:04d}.json') for i in range(1,201)]+[q.load(work/'policy'/f'{i:04d}.json')['discriminator'] for i in range(1,51)]
 q.immutable(work/'complete.json',dict(actual_discriminator_updates=sum(int(x['optimizer_updated']) for x in d_records),zero_discriminator_signal_items=sum(int(x['zero_gradient']) for x in d_records),policy_groups=50,policy_coverage='50 distinct Train200 prompts',**signals,reference_sha256=reference));return select(out)

def select(out):
 p=verify(out);work=out/'training'
 if q.load(work/'complete.json')['policy_groups']!=50:raise PermissionError('Fixed50 incomplete')
 candidates=[]
 for step in (25,50):
  value=q.load(work/f'validation/policy{step:04d}/scores.json')
  candidates.append(((-value['correct'],-value['valid'],value['length_capped'],step),step,value))
 _,step,value=min(candidates);folder=work/'checkpoints'/f'policy{step:04d}';lib.verify_checkpoint(out,folder)
 result=dict(checkpoint=str(folder),checkpoint_manifest_sha256=q.sha(folder/'manifest.json'),validation=value,step=step,protocol_sha256=q.sha(out/'protocol.json'))
 q.immutable(out/'selected.json',result);return result

def preflight(out):
 p=verify(out);m,tr,_,teacher=lib.data(p['manifest_path'],p['teacher_manifest_path']);folder=out/'disposable-preflight'
 if folder.exists():raise PermissionError('Disposable preflight already exists; no retry')
 torch,tok,model,head=lib.setup(out);opts=lib.optimizers(torch,model,head);before={r:lib.role_hash(model,r) for r in ('actor','reference','discriminator')}
 # Real teacher-response SFT proves native actor gradients even when real group advantages are zero.
 row=max(tr,key=lambda r:len(lib.pack(tok,r,teacher[str(r['id'])]['text'])['input_ids']))
 lib.activate(model,head,'actor');model.train();opts['actor'].zero_grad(set_to_none=True);packed=lib.pack(tok,row,teacher[str(row['id'])]['text']);ids=torch.tensor([packed['input_ids']],device='cuda')
 loss=-lib.response_log_probs(lib.hidden(model,ids),model.get_output_embeddings().weight,ids,packed['prompt_length'],temperature=1.).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1)
 if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)) or float(norm)<=0:raise RuntimeError('Native disposable SFT gradient failed')
 opts['actor'].step();opts['actor'].zero_grad(set_to_none=True);sft=dict(loss=float(loss.detach()),gradient_norm=float(norm),teacher_sequence_tokens=len(packed['input_ids']))
 if lib.role_hash(model,'actor')==before['actor'] or any(lib.role_hash(model,r)!=before[r] for r in ('reference','discriminator')):raise RuntimeError('Disposable SFT role isolation failed')
 del ids,loss,norm
 lib.clone_warmed(model);opts=lib.optimizers(torch,model,head);before={r:lib.role_hash(model,r) for r in before};context(out,'DISPOSABLE',0)
 row=max(tr,key=lambda r:len(lib.prompt(tok,r)));fresh=rollouts(torch,tok,model,head,row,n=8)
 dm=update_d(torch,model,head,opts['discriminator'],lib.pack(tok,row,teacher[str(row['id'])]['text'])['input_ids'],fresh)
 if any(lib.role_hash(model,r)!=before[r] for r in ('actor','reference')):raise RuntimeError('Disposable D role isolation failed')
 metric,rewards,advantages=lib.policy(torch,model,head,opts['actor'],fresh)
 if not all(__import__('math').isfinite(float(x)) for x in rewards+advantages):raise RuntimeError('Nonfinite real group reward')
 if lib.role_hash(model,'reference')!=before['reference']:raise RuntimeError('Disposable policy changed reference')
 checkpoint(out,folder,torch,model,head,opts,'DISPOSABLE',1);expected_random=torch.rand(8,device='cuda').cpu();lib.activate(model,head,'actor');model.eval()
 with torch.no_grad():live=lib.logprobs(torch,model,fresh[0]).cpu()
 peak=torch.cuda.max_memory_allocated();del model,head,opts;gc.collect();torch.cuda.empty_cache()
 torch,tok,model,head=lib.setup(out,folder);cm=lib.verify_checkpoint(out,folder);opts=lib.optimizers(torch,model,head);lib.restore(torch,model,head,opts,out,folder)
 if not torch.equal(expected_random,torch.rand(8,device='cuda').cpu()):raise RuntimeError('RNG independent reload differs')
 lib.activate(model,head,'actor');model.eval()
 with torch.no_grad():again=lib.logprobs(torch,model,fresh[0]).cpu()
 error=float((live-again).abs().max())
 if error!=0:raise RuntimeError('Native independent reload logprob differs')
 value=dict(status='PASS',disposable_optimizer_updates=1+int(dm['optimizer_updated'])+int(metric['optimizer_updated']),training_updates=0,fresh_rollouts=8,real_SFT_actor_update=True,sft=sft,discriminator=dm,actor=metric,rewards=rewards,advantages=advantages,zero_adversarial_signal=all(float(a)==0 for a in advantages),zero_signal_is_not_implementation_failure=True,optimizer_state_and_rng_exact=True,reload_max_logprob_error=error,native_vocabulary=True,peak_allocated_bytes=peak,protocol_sha256=q.sha(out/'protocol.json'))
 q.immutable(out/'gpu-preflight.json',value);q.immutable(out/'checkpoint-size-estimate.json',dict(disposable_checkpoint_bytes=size_tree(folder),anticipated_retained_full_checkpoints=12,estimated_four_unit_bytes=size_tree(folder)*48,global_cap=8*2**30,estimate_not_guarantee=True))
 return dict(status='PASS',gpu_preflight_sha256=q.sha(out/'gpu-preflight.json'))
def evaluate(out,arm,selection):
 p=verify(out)
 if arm not in ('W-off','GAD-off','base-on','W-on','GAD-on'):raise PermissionError('Mainline only off; no8192 optional-on implicit route')
 frozen=q.load(selection)
 if frozen.get('task')!=p['task'] or frozen.get('all_size_selections_frozen_before_eval') is not True:raise PermissionError('Benchmark both-size selection freeze required')
 for item in frozen['units'].values():
  for spec in item.values():q.bound(spec)
 on=arm.endswith('-on');thinking='on' if on else 'off'
 decision=q.load(RUN/'diagnostics/thinking-selection.json')['models'][p['model']] if on else None
 cap=256
 if on:
  import validate_qwen_thinking_budget_v3_20261024 as budget
  if decision.get('qualified') is not True or decision.get('arm') not in ('N','F1','F2'):raise PermissionError('Bounded thinking not qualified')
  bp=q.load(q.bound(decision['budget_protocol']))
  if bp['arms']!=budget.ARMS or bp['sampling']!=budget.PARAMS:raise PermissionError('Bounded runtime differs from frozen Dev protocol')
  cap=budget.ARMS[decision['arm']]['B']+256
 if arm.startswith('GAD'):
  chosen=q.load(out/'selected.json');check=q.load(out/'selected-reload.json')
  if check.get('status')!='PASS' or check['selected_sha256']!=q.sha(out/'selected.json'):raise PermissionError('Selected reload missing')
  folder=Path(chosen['checkpoint'])
 else:folder=out/'training/checkpoints/W0200'
 cm=lib.verify_checkpoint(out,folder);torch,tok,model,head=lib.setup(out,folder)
 if {r:lib.role_hash(model,r) for r in cm['role_hashes']}!=cm['role_hashes']:raise PermissionError('Eval reload differs')
 m=q.load(p['manifest_path']);ev=m['common_eval'];items=lib.rows(ev['queries']);target=out/'evaluation'/arm
 q.immutable(target/'contract.json',dict(protocol_sha256=q.sha(out/'protocol.json'),checkpoint_manifest_sha256=q.sha(folder/'manifest.json'),benchmark_selection=bind(selection),task=p['task'],model=p['model'],arm=arm,count=ev['count'],queries=ev['queries'],fixed_original8_messages=True,greedy=not on,max_new_tokens=cap,thinking_decision=decision,gold_read=False))
 for i,row in enumerate(items):
  path=target/'responses'/(q.digest(str(row['id']))+'.json')
  if path.exists():
   if q.load(path)['evaluation_contract_sha256']!=q.sha(target/'contract.json'):raise PermissionError('Cached Eval differs')
   continue
  if path.with_suffix('.started.json').exists():raise PermissionError('Unresolved Eval; no resampling')
  if paused():raise SavedPause('Eval stopped at saved answer boundary')
  if len(row['adapted_messages'])!=18:raise PermissionError('Original fixed8 required')
  q.immutable(path.with_suffix('.started.json'),dict(id=str(row['id']),evaluation_contract_sha256=q.sha(target/'contract.json')));context(out,'eval-'+arm,i)
  response=bounded_on(torch,tok,model,head,row,arm=='base-on',decision) if on else rollouts(torch,tok,model,head,row,n=1,greedy=True,messages=row['adapted_messages'])[0]
  parsed=q.classify(response.get('controlled_transcript',response['raw_generated']),thinking,ev['scorer'],row.get('metadata',{}),response['finish_reason'])
  if on and response['finish_reason']=='early_eos':parsed.update(reason='EARLY_EOS_NO_FINAL',valid=False,prediction=None)
  q.immutable(path,dict(id=str(row['id']),task=p['task'],model=p['model'],thinking=thinking,**parsed,evaluation_contract_sha256=q.sha(target/'contract.json'),raw_generated=response['raw_generated'],output_ids=response['response_ids'],input_tokens=response['prompt_length'],output_tokens=len(response['response_ids']),generation_seconds=response['generation_seconds'],finish_reason=response['finish_reason'],max_new_tokens=cap,decoding='bounded-on' if on else 'greedy',input_truncated=False,**({k:response[k] for k in ('controlled_transcript','forced_close','natural_close','injected_ids','phase1','phase2','actual_prefill_tokens','bounded_decision','adapter_disabled')} if on else {})))
 if len(items)!=ev['count'] or len({str(r['id']) for r in items})!=ev['count']:raise PermissionError('Fixed Eval denominator')
 q.immutable(target/'predictions-frozen.json',dict(complete=True,expected=ev['count'],contract_sha256=q.sha(target/'contract.json'),gold_read=False,predictions=[bind(target/'responses'/(q.digest(str(r['id']))+'.json')) for r in items]))
 return dict(status='FROZEN_UNSCORED',arm=arm,count=len(items))

# Only this isolated module instance is wired to the new protocol; no frozen source edit.
lib.verify=verify;lib.rollouts=rollouts;lib.checkpoint=checkpoint;lib.evaluate_validation=validation

def main():
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','train','select','reload','evaluate','resume-check'])
 ap.add_argument('--model',choices=tuple(MODELS));ap.add_argument('--manifest',type=Path);ap.add_argument('--teacher-manifest',type=Path);ap.add_argument('--policy-manifest',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--arm',choices=['W-off','GAD-off','base-on','W-on','GAD-on']);ap.add_argument('--selection',type=Path);a=ap.parse_args();out=unit_path(a.output)
 if a.action=='prepare':
  if not all((a.model,a.manifest,a.teacher_manifest)):ap.error('prepare requires model,data,teacher')
  result=prepare(a.model,a.manifest,a.teacher_manifest,a.policy_manifest or a.manifest,out)
 elif a.action=='resume-check':result=resume_check(out)
 else:
  from night_dual_20261022 import lock,cgroup_limits_verified
  import run_cityfm_repair_qwen_20261022 as guard
  if paused():raise SavedPause('Newrun owner stop; no model load')
  with lock(out/'worker.lock'),lock(guard.HISTORICAL/'locks/gpu.lock'):
   cgroup_limits_verified()
   if a.action=='train':result=train(out,a.resume)
   elif a.action=='preflight':result=preflight(out)
   elif a.action=='select':result=select(out)
   elif a.action=='reload':result=reload_both(out)
   else:
    if not a.arm or not a.selection:ap.error('evaluate requires arm and selection freeze')
    result=evaluate(out,a.arm,a.selection)
 print(json.dumps(result))




def bounded_on(torch,tok,model,head,row,base_only,decision):
 """Reuse frozen v3 phase semantics, full re-prefill, real/injected accounting; no training."""
 import validate_qwen_thinking_budget_v3_20261024 as budget
 if decision.get('qualified') is not True or decision.get('arm') not in ('N','F1','F2'):raise PermissionError('Bounded Dev gate not qualified')
 q.bound(decision['budget_protocol'])
 lib.activate(model,head,'actor');model.eval();text=tok.apply_chat_template(row['adapted_messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True);prefix=tok.encode(text,add_special_tokens=False)
 tokens=budget.token_contract(tok);control=tok.encode(budget.CONTROL,add_special_tokens=False)
 if len(prefix)+budget.ARMS[decision['arm']]['B']+len(control)+256>32768:raise PermissionError('Bounded on context overflow')
 torch.manual_seed(budget.seed_for(CTX['out'].parent.name,str(row['id'])));presence=budget.GeneratedPresence(budget.PARAMS['presence_penalty']);phases={};started=time.monotonic()
 def generate(name,prompt,limit,history,stop_close):
  path=CTX['out']/'bounded-phases'/CTX['stage']/q.digest(str(row['id']))/(name+'.json')
  if path.exists() or path.with_suffix('.started.json').exists():raise PermissionError('Unresolved logical bounded answer cannot resample or reseed phase')
  q.immutable(path.with_suffix('.started.json'),dict(id=str(row['id']),stage=CTX['stage'],phase=name,arm=decision['arm']))
  presence.begin_phase(len(prompt),history);ids=torch.tensor([prompt],device='cuda');criteria=[]
  if stop_close:criteria=[budget.NaturalThinkClose(len(prompt),tokens['close'])]
  kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=True,num_beams=1,max_new_tokens=limit,eos_token_id=tokens['eos'],pad_token_id=tok.pad_token_id,use_cache=True,temperature=budget.PARAMS['temperature'],top_p=budget.PARAMS['top_p'],top_k=budget.PARAMS['top_k'],min_p=0.,repetition_penalty=1.,logits_processor=[presence],stopping_criteria=criteria)
  start=time.monotonic()
  with torch.inference_mode(),(model.disable_adapter() if base_only else contextlib.nullcontext()):output=model.generate(**kwargs)
  emitted=output[0,len(prompt):].tolist();value=dict(name=name,ids=emitted,input_tokens=len(prompt),output_tokens=len(emitted),generation_seconds=time.monotonic()-start,full_reprefill=True)
  q.immutable(path,value);phases[name]=value;del ids,output;return value
 result=budget.two_stage(decision['arm'],prefix,control,tokens['close'],tokens['eos'],generate)
 actual=result['actual'];controlled=result['p1']['ids']+result['injected']+(result['p2']['ids'] if result['p2'] else [])
 good=result['p2'] is not None and result['s2']=='native_eos';raw=tok.decode(controlled,skip_special_tokens=False)
 return dict(raw_generated=tok.decode(actual,skip_special_tokens=False),controlled_transcript=raw,response_ids=actual,prompt_length=len(prefix),finish_reason='stop' if good else 'thinking_budget_exhausted' if result['action']=='stop_thinking_budget' else 'early_eos' if result['action']=='stop_early_eos' else 'length',generation_seconds=sum(x['generation_seconds'] for x in phases.values()),logical_seconds=time.monotonic()-started,forced_close=result['forced'],natural_close=result['natural'],injected_ids=result['injected'],phase1=result['p1'],phase2=result['p2'],actual_prefill_tokens=sum(x['input_tokens'] for x in phases.values()),bounded_decision=decision,adapter_disabled=base_only)





def reload_both(out):
 result=lib.reload(out)
 p=verify(out);folder=out/'training/checkpoints/W0200';cm=lib.verify_checkpoint(out,folder)
 torch,tok,model,head=lib.setup(out,folder)
 hashes={r:lib.role_hash(model,r) for r in cm['role_hashes']}
 if hashes!=cm['role_hashes']:raise PermissionError('Independent W reload differs')
 m,tr,_,teachers=lib.data(p['manifest_path'],p['teacher_manifest_path']);lib.activate(model,head,'actor');model.eval()
 with torch.no_grad():probe=lib.logprobs(torch,model,lib.pack(tok,tr[0],teachers[str(tr[0]['id'])]['text']))
 if not bool(torch.isfinite(probe).all()):raise RuntimeError('W reload nonfinite')
 q.immutable(out/'W-reload.json',dict(status='PASS',checkpoint_manifest_sha256=q.sha(folder/'manifest.json'),role_hashes=hashes,protocol_sha256=q.sha(out/'protocol.json')))
 return dict(result,W_reload='PASS')




def update_d(torch,model,head,optimizer,teacher_ids,records):
 """Only mathematically exact zero D gradient is a legitimate no-op; all other failures propagate."""
 before={r:lib.role_hash(model,r) for r in ('actor','reference','discriminator')};head_before=lib.nested_hash(head.state_dict());opt_before=lib.nested_hash(optimizer.state_dict())
 try:
  result=lib.update_d(torch,model,head,optimizer,teacher_ids,records)
  return dict(result,optimizer_updated=True,actual_optimizer_updated=True,optimizer_step_skipped=False,zero_gradient=False)
 except RuntimeError as error:
  if str(error)!='Invalid discriminator gradient':raise
  frame=error.__traceback__;evidence=None
  while frame is not None:
   if frame.tb_frame.f_code.co_name=='update_d' and 'losses' in frame.tb_frame.f_locals:evidence=frame.tb_frame.f_locals
   frame=frame.tb_next
  if evidence is None:raise
  norm=evidence['norm'];params=[p for p in model.parameters() if p.requires_grad]+list(head.parameters());grads=[p.grad for p in params if p.grad is not None]
  if not grads or not bool(torch.isfinite(norm)) or float(norm)!=0. or any(not bool(torch.isfinite(g).all()) or bool(torch.count_nonzero(g)!=0) for g in grads):raise
  if before!={r:lib.role_hash(model,r) for r in before} or head_before!=lib.nested_hash(head.state_dict()) or opt_before!=lib.nested_hash(optimizer.state_dict()):raise RuntimeError('Zero-D path changed model/head/optimizer') from error
  optimizer.zero_grad(set_to_none=True)
  return dict(loss=sum(evidence['losses']),gradient_norm=float(norm),optimizer_updated=False,actual_optimizer_updated=False,optimizer_step_skipped=True,zero_gradient=True,zero_adversarial_signal=True,identical_teacher_student_tokens=all(r['input_ids']==teacher_ids for r in records),reason='FINITE_EXACT_ZERO_D_GRADIENT_NO_STEP')

if __name__=='__main__':main()





