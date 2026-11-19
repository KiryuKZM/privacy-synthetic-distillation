#!/usr/bin/env python3
"""Train-only matched-condition base/W/GAD probe. Root alone launches GPU; no updates/API."""
from __future__ import annotations
import argparse,contextlib,gc,hashlib,json,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import eval_ust_sg_qwen_modes_20261024 as q
import train_overnight_native_gad_20261025 as t
RUN=ROOT/'runs/2026-10-25-overnight-onpolicy';NEW=ROOT/'runs/2026-10-25-qwen-transfer-repair';OUT=NEW/'matched-train16';MEMBERS=NEW/'data/train16-manifest.json'
TRAINER_SHA='36a4bee56127bde1b9be498f5bf8b9c465031d2826fa1285e4540d0373c0c848'
STOP=False

def bind(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=q.sha(p))
def bound(s):
 p=ROOT/s['path']
 if q.sha(p)!=s['sha256']:raise PermissionError('Probe binding changed: '+str(p))
 return p
def stopped():return STOP or (OUT/'stop.json').exists() or (OUT/'global-stop.json').exists()
def stop(*_):
 global STOP
 STOP=True
def heartbeat(**v):t.atomic(OUT/'heartbeat.json',dict(epoch=time.time(),**v))
def prepare():
 if q.sha(ROOT/'scripts/train_overnight_native_gad_20261025.py')!=TRAINER_SHA:raise PermissionError('Original trainer changed')
 membership=q.load(MEMBERS);tasks={};bindings=[bind(__file__),bind(MEMBERS),bind(NEW/'user-goal-approval.json'),bind(ROOT/'scripts/train_overnight_native_gad_20261025.py')]
 for task in ['USTBench','SGQA']:
  mp=RUN/'data'/(task+'.manifest.json');m=q.load(mp);ms=membership['tasks'][task];selected=t.lib.rows(ms['queries']);assert len(selected)==16 and len({r['group'] for r in selected})==16
  tp=RUN/'teacher'/task/'train/manifest.json';tm=q.load(tp);references={r['id']:r for r in t.lib.rows(ms['teacher_references'])};fp=q.bound(tm['fixed8_prefix']);prefix=q.load(fp)['messages']
  if len(prefix)!=17 or [x['role'] for x in prefix[1:]]!=['user','assistant']*8:raise PermissionError('Existing exact fixed8 prefix needed')
  if any(r['messages'][0]!=prefix[0] or r['messages'][-1]['content']!=r['question'] for r in selected):raise PermissionError('System/query mismatch; eight-demo-only comparison required')
  file=OUT/'inputs'/(task+'.json');q.immutable(file,dict(task=task,selected=selected,teacher={r['id']:references[r['id']] for r in selected},fixed8=prefix,source_manifest=bind(mp),teacher_manifest=bind(tp),prefix=bind(fp)))
  tasks[task]=dict(inputs=bind(file),manifest=bind(mp),train_gold=ms['gold'],scorer=ms['scorer'])
  bindings.extend([bind(mp),ms['queries'],ms['membership'],ms['teacher_references'],bind(tp),tm['records_file'],bind(fp)])
  for model_specs in ms['models'].values():bindings.extend(model_specs.values())
  for model in ['08','2b']:
   u=RUN/'training'/task/model;chosen=q.load(u/'selected.json');W=u/'training/checkpoints/W0200';G=Path(chosen['checkpoint'])
   for file0 in [u/'protocol.json',u/'W-reload.json',u/'selected.json',u/'selected-reload.json',W/'manifest.json',G/'manifest.json']:
    bindings.append(bind(file0))
   for folder in [W,G]:
    cm=q.load(folder/'manifest.json')
    for name in cm['files']:
     if name.endswith('.safetensors') or name.endswith('adapter_config.json'):bindings.append(bind(folder/name))
 protocol=dict(schema='matched-train16-probe-v1',worker=bind(__file__),trainer_sha256=TRAINER_SHA,tasks=tasks,bindings=bindings,models=['08','2b'],roles=['base','W','GAD'],conditions=['zero-shot','matched8'],seed=20261025,max_new_tokens=256,do_sample=False,native_bf16=True,context_cap=32768,expected_per_unit=96,maximum_all_units=384,no_training=True,new_api_calls=0,no_Eval_or_Final=True,all_unit_answers_freeze_before_Train_gold=True,resources=dict(memory_gib=14,cpu_threads=4,cuda_allocator_gib=9,minimum_gpu_free_gib=9,minimum_host_disk_gib=20),tinyfit_not_activated=True)
 q.immutable(OUT/'protocol.json',protocol);return dict(status='PREPARED_ZERO_GPU',protocol=bind(OUT/'protocol.json'))
def verify():
 p=q.load(OUT/'protocol.json');a=q.load(OUT/'authorization.json')
 if a.get('approved') is not True or a.get('protocol_sha256')!=q.sha(OUT/'protocol.json'):raise PermissionError('Root probe authorization required')
 bound(p['worker'])
 if p['trainer_sha256']!=q.sha(ROOT/'scripts/train_overnight_native_gad_20261025.py'):raise PermissionError('Trainer source changed')
 for b in p['bindings']:bound(b)
 return p

def packing(tok,row,teacher,condition,prefix):
 messages=row['messages'] if condition=='zero-shot' else prefix+[dict(role='user',content=row['question'])]
 prompt=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
 answer=tok.encode(teacher['text'],add_special_tokens=False);eos=tok.convert_tokens_to_ids('<|im_end|>')
 if not answer:raise PermissionError('Empty teacher target')
 if answer[-1]!=eos:answer.append(eos)
 if len(prompt)+max(256,len(answer))>32768:raise PermissionError('No prompt truncation permitted')
 return prompt,answer

def score_record(record,scorer,gold,metadata):
 from run_llm_task_sweep_20261023 import score_prediction
 return score_prediction(record,dict(kind='qa',scorer=scorer),gold,metadata)
def summarize(task,model):
 p=verify();target=OUT/task/model;data=q.load(bound(p['tasks'][task]['inputs']));frozen=q.load(target/'all-answers-frozen.json');results=[q.load(bound(s)) for s in frozen['answers']]
 expected={(role,condition,r['id']) for role in p['roles'] for condition in p['conditions'] for r in data['selected']}
 if len(results)!=96 or {(r['role'],r['condition'],r['id']) for r in results}!=expected:raise PermissionError('Full matched96 required before Train scoring')
 gold=q.load(bound(p['tasks'][task]['train_gold']));by={r['id']:r for r in data['selected']};summary={}
 for role in p['roles']:
  for condition in p['conditions']:
   rs=[r for r in results if r['role']==role and r['condition']==condition];pairs=[score_record(r,p['tasks'][task]['scorer'],gold[r['id']],by[r['id']]['metadata']) for r in rs]
   summary[role+'/'+condition]=dict(total=16,correct=sum(c for c,v in pairs),valid=sum(v for c,v in pairs),capped=sum(r['finish_reason']=='length' for r in rs),teacher_exact_text=sum(r['teacher_exact_text'] for r in rs),mean_teacher_token_NLL=sum(r['teacher_token_NLL'] for r in rs)/16,teacher_token_count=sum(r['teacher_tokens'] for r in rs),mean_output_tokens=sum(len(r['output_ids']) for r in rs)/16)
 result=dict(task=task,model=model,summary=summary,freeze=bind(target/'all-answers-frozen.json'),posthoc_Train_only=True,not_generalization=True,new_api_calls=0,training_updates=0)
 q.immutable(target/'scores.local.json',result);return result


def run(task,model_name):
 p=verify();data=q.load(bound(p['tasks'][task]['inputs']));unit=RUN/'training'/task/model_name;target=OUT/task/model_name
 if stopped():return dict(status='PAUSED_BEFORE_MODEL')
 from night_dual_20261022 import lock,cgroup_limits_verified
 import run_cityfm_repair_qwen_20261022 as guard
 with lock(OUT/'worker.lock'),lock(guard.HISTORICAL/'locks/gpu.lock'):
  caps=cgroup_limits_verified();model=None
  try:
   for roles,folder in [(['base','W'],unit/'training/checkpoints/W0200'),(['GAD'],Path(q.load(unit/'selected.json')['checkpoint']))]:
    pending=[]
    for role in roles:
     for condition in p['conditions']:
      for row in data['selected']:
       file=target/'responses'/(q.digest([role,condition,row['id']])+'.json')
       if file.exists():
        saved=q.load(file)
        if saved['protocol_sha256']!=q.sha(OUT/'protocol.json'):raise PermissionError('Saved probe binding differs')
       else:
        if file.with_suffix('.started.json').exists():raise PermissionError('Unresolved probe answer; never replay')
        pending.append((role,condition,row,file))
    if not pending:continue
    if stopped():return dict(status='PAUSED_BEFORE_MODEL')
    cm=t.lib.verify_checkpoint(unit,folder);torch,tok,model,head=t.lib.setup(unit,folder);t.lib.activate(model,head,'actor');model.eval()
    if {r:t.lib.role_hash(model,r) for r in cm['role_hashes']}!=cm['role_hashes']:raise PermissionError('Probe reload differs')
    for role,condition,row,file in pending:
     if stopped():return dict(status='PAUSED_AT_SAVED_ANSWER',task=task,model=model_name)
     teacher=data['teacher'][row['id']];prompt,answer=packing(tok,row,teacher,condition,data['fixed8']);stops=[tok.convert_tokens_to_ids(x) for x in ['<|im_end|>','<|endoftext|>']]
     identity=dict(id=row['id'],task=task,model=model_name,role=role,condition=condition,protocol_sha256=q.sha(OUT/'protocol.json'),checkpoint_manifest_sha256=q.sha(folder/'manifest.json'),prompt_ids_sha256=q.digest(prompt),teacher_text_sha256=q.digest(teacher['text']),cgroup=caps,started_epoch=time.time())
     q.immutable(file.with_suffix('.started.json'),identity);heartbeat(stage='TRAIN_ONLY_PROBE',**{k:identity[k] for k in ['task','model','role','condition','id']})
     torch.manual_seed(p['seed']);ids=torch.tensor([prompt],device='cuda');full=torch.tensor([prompt+answer],device='cuda');start=time.monotonic()
     with (model.disable_adapter() if role=='base' else contextlib.nullcontext()),torch.inference_mode():
      emitted=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=False,num_beams=1,eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)[0,len(prompt):].tolist()
      gen_seconds=time.monotonic()-start;lp=t.lib.response_log_probs(t.lib.hidden(model,full),model.get_output_embeddings().weight,full,len(prompt),temperature=1.)
      nll=float(-lp.mean());token_nll=(-lp).cpu().tolist()
     if not emitted or not __import__('math').isfinite(nll):raise RuntimeError('Empty or nonfinite probe')
     finish='stop' if emitted[-1] in stops else 'length';raw=tok.decode(emitted,skip_special_tokens=False);parsed=q.classify(raw,'off',p['tasks'][task]['scorer'],row.get('metadata',{}),finish)
     actual=emitted[:-1] if finish=='stop' else emitted;teacher_tokens=answer[:-1];value=dict(**identity,**parsed,raw_generated=raw,output_ids=emitted,input_tokens=len(prompt),output_tokens=len(emitted),input_truncated=False,finish_reason=finish,max_new_tokens=256,greedy=True,teacher_token_NLL=nll,teacher_token_NLLs=token_nll,teacher_target_ids=answer,teacher_tokens=len(answer),teacher_exact_text=tok.decode(actual,skip_special_tokens=False)==teacher['text'],teacher_exact_token_prefix=actual==teacher_tokens,generation_seconds=gen_seconds,logical_seconds=time.monotonic()-start,adapter_disabled=role=='base',peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_api_calls=0,training_updates=0)
     q.immutable(file,value);del ids,full,lp;heartbeat(stage='ANSWER_COMMITTED',task=task,model=model_name,role=role,condition=condition,id=row['id'])
    del model,head,tok;model=None;gc.collect();torch.cuda.empty_cache()
   answers=[target/'responses'/(q.digest([role,c,r['id']])+'.json') for role in p['roles'] for c in p['conditions'] for r in data['selected']]
   if not all(x.exists() for x in answers):raise PermissionError('Incomplete matched96')
   q.immutable(target/'all-answers-frozen.json',dict(complete=True,count=96,protocol_sha256=q.sha(OUT/'protocol.json'),answers=[bind(f) for f in answers],Train_gold_not_yet_read=True))
   result=summarize(task,model_name);heartbeat(stage='UNIT_COMPLETE',task=task,model=model_name);return result
  finally:
   if model is not None:del model;gc.collect()

def main():
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','summarize']);ap.add_argument('--task',choices=['USTBench','SGQA']);ap.add_argument('--model',choices=['08','2b']);a=ap.parse_args()
 if a.action=='prepare':result=prepare()
 else:
  if not a.task or not a.model:ap.error('task/model required')
  result=run(a.task,a.model) if a.action=='run' else summarize(a.task,a.model)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

