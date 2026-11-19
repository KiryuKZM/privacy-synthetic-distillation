#!/usr/bin/env python3
"""Durable UST-then-SG four-unit owner-only coordinator; no model/API import."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import eval_ust_sg_qwen_modes_20261024 as q
RUN=ROOT/'runs/2026-10-25-overnight-onpolicy';CONFIG=RUN/'execution-config.json'
TASKS=('USTBench','SGQA');MODELS=('08','2b');STOP=False
TERMINAL=('COMPLETE','TECHNICAL_STOP','PARTIAL_UNSAFE','USER_STOPPED')
def stop(*_):
 global STOP
 STOP=True

def paused():return STOP or any((RUN/n).exists() for n in ('stop.json','global-stop.json'))
def bind(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=q.sha(p))
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True)+'\n');tmp.replace(p)
def once(p,v):q.immutable(p,v)
def unit(task,model):return RUN/'training'/task/model
def tree_bytes(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())
def check_config():
 c=q.load(CONFIG)
 if not c.get('approved') or c.get('controller_sha256')!=q.sha(__file__):raise PermissionError('Root-reviewed controller/config required')
 q.bound(c['worker']);a=q.load(q.bound(c['approval']));q.bound(a['plan'])
 if not a.get('approved') or a.get('resume_old_gad_queue') is not False or a.get('max_experiment_units')!=4:raise PermissionError('New four-unit approval differs')
 if c.get('models')!=list(MODELS) or set(c['tasks'])!=set(TASKS):raise PermissionError('Fixed four model/task units')
 if (c.get('output_bytes_max'),c.get('max_top_level_jobs'),c.get('max_substages'),c.get('max_technical_repairs'))!=(8*2**30,9,13,1):raise PermissionError('Scope/cap mismatch')
 for t,s in c['tasks'].items():
  if q.load(q.bound(s['manifest']))['task']!=t:raise PermissionError('Dataset task mismatch')
 return c

def diagnostics_ready():
 path=RUN/'diagnostics/completed.json'
 if not path.exists():return False
 d=q.load(path)
 if d.get('status') not in ('COMPLETE','SKIPPED_TECHNICAL_OFF_FALLBACK'):raise PermissionError('Diagnostic technical state requires explicit fallback decision')
 selection=RUN/'diagnostics/thinking-selection.json'
 if not selection.exists():return False
 if q.load(selection).get('complete') is not True:raise PermissionError('Bounded-thinking decision must be frozen')
 return True

def teacher_ready(c,task):
 path=q.local_path(c['tasks'][task]['teacher_manifest_path'])
 if not path.exists():
  terminal=path.parent.parent/'terminal.json'
  if terminal.exists() and q.load(terminal).get('status') in ('FAILED','FAILED_MECHANICAL_REFERENCE_QUALIFICATION'):raise PermissionError('Teacher terminal failure; affected benchmark NA, no infinite wait')
  return False
 m=q.load(path)
 if m.get('task')!=task or m.get('count')!=200 or m.get('all_valid') is not True:raise PermissionError('Teacher structural validity/membership blocks unit; no drop/fill')
 q.bound(m['records_file'])
 for spec in m.get('bindings',[]):q.bound(spec)
 return True

def audit_teacher(task):
 script=ROOT/'scripts/audit_overnight_teacher_train_20261025.py'
 if q.sha(script)!='83725b126a4a47e419a05e840171193ea12a8bc12ba58d08e086a74f89e44b80':raise PermissionError('Teacher Train reporting-only audit source changed')
 result=subprocess.run([sys.executable,str(script),'--task',task],cwd=ROOT,capture_output=True,text=True)
 if result.returncode:raise PermissionError('Teacher Train structural audit failed: '+result.stderr[-500:])
 if not (RUN/'teacher'/task/'train/quality.local.json').exists():raise PermissionError('Teacher audit not complete')

def heartbeat(stage,**kw):
 value=dict(stage=stage,epoch=time.time(),local_api_calls=0,external_teacher_cost_scope='Shared parent ledger pools E-H; separate from this local coordinator',old_queues_resumed=False,**kw);atomic(RUN/'pipeline/heartbeat.json',value)
 if (RUN/'goal-state.json').exists():
  state=q.load(RUN/'goal-state.json');state.update(pipeline=value,status='PAUSED_BY_USER' if paused() else stage if stage in ('COMPLETE','PARTIAL_TECHNICAL_STOP') else 'ACTIVE_PIPELINE',ui_goal_not_required_for_execution=True);atomic(RUN/'goal-state.json',state)
def resources():
 import run_cityfm_repair_qwen_20261022 as guard
 s=guard.probe(RUN)
 return all(s.get(k,-1)>=v for k,v in dict(windows_free_gib=20,c_free_gib=20,linux_free_gib=20,gpu_free_gib=9).items()),s

def reserve(task,model,repair=False):
 path=RUN/'pipeline/trial-registry.json';r=q.load(path) if path.exists() else dict(units={},technical_repairs=[],top_level_jobs=0,optimization_substages=0)
 key=task+'/'+model
 if repair:
  if r['technical_repairs']:raise PermissionError('One global technical-repair substage exhausted')
  r['technical_repairs'].append(dict(unit=key,source='root evidence-bound repair receipt'));r['top_level_jobs']+=1;r['optimization_substages']+=1
 elif key not in r['units']:
  if len(r['units'])>=4:raise PermissionError('Four experiment unit cap')
  r['units'][key]=dict(W=True,GAD_contains_D_warmup=True,substage_count=3);r['top_level_jobs']+=2;r['optimization_substages']+=3
 if r['top_level_jobs']>9 or r['optimization_substages']>13:raise PermissionError('9 top/13 substage cap')
 atomic(path,r);return r

def stage_marker(out,stage):
 if stage=='prepare':return out/'token-audit.json'
 if stage=='preflight':return out/'gpu-preflight.json'
 if stage=='train':return out/'training/complete.json'
 if stage=='select':return out/'selected.json'
 if stage=='reload':return out/'selected-reload.json'
 return out/'evaluation'/stage.split(':',1)[1]/'predictions-frozen.json'

def run_stage(c,task,model,stage):
 out=unit(task,model);folder=RUN/'pipeline'/task/model/stage.replace(':','-');started=folder/'started.json';finished=folder/'finished.json';resume=False
 if finished.exists() and q.load(finished)['exit_code']==0:
  q.bound(q.load(finished)['completion']);return 'COMPLETE'
 if started.exists():
  grant=out/'resume-authorization.json'
  if stage!='train' or not grant.exists():return 'PARTIAL_UNSAFE'
  a=q.load(grant)
  if not(a.get('approved') is True and a.get('prior_started_sha256')==q.sha(started) and a.get('protocol_sha256')==q.sha(out/'protocol.json') and a.get('worker_sha256')==c['worker']['sha256']):return 'PARTIAL_UNSAFE'
  childfile=folder/'child.json'
  if childfile.exists() and Path('/proc',str(q.load(childfile)['pid'])).exists():return 'PARTIAL_UNSAFE'
  checked=subprocess.run([sys.executable,str(q.bound(c['worker'])),'resume-check','--output',str(out)],cwd=ROOT,capture_output=True,text=True)
  if checked.returncode or json.loads(checked.stdout.strip().splitlines()[-1]).get('status')!='SAFE_RESUME':return 'PARTIAL_UNSAFE'
  if a.get('kind')=='technical_repair':reserve(task,model,repair=True)
  elif a.get('kind')!='explicit_user_resume':return 'PARTIAL_UNSAFE'
  resume=True;folder=folder/'resume-once';started=folder/'started.json';finished=folder/'finished.json'
  if started.exists():return 'PARTIAL_UNSAFE'
 if paused():return 'USER_STOPPED'
 if stage!='prepare':
  ok,s=resources()
  if not ok:atomic(RUN/'pipeline/resource-wait.json',s);return 'WAITING_RESOURCE'
 if stage=='train' and not resume:reserve(task,model)
 action='evaluate' if stage.startswith('evaluate:') else stage
 argv=[sys.executable,'-u',str(q.bound(c['worker'])),action,'--output',str(out)]
 if stage=='prepare':argv+=['--model',model,'--manifest',str(q.bound(c['tasks'][task]['manifest'])),'--teacher-manifest',str(q.local_path(c['tasks'][task]['teacher_manifest_path']))]
 if resume:argv+=['--resume']
 if action=='evaluate':argv+=['--arm',stage.split(':',1)[1],'--selection',str(RUN/'benchmarks'/task/'model-selection-frozen.json')]
 once(started,dict(argv=argv,worker=c['worker'],config_sha256=q.sha(CONFIG),task=task,model=model,stage=stage))
 folder.mkdir(parents=True,exist_ok=True)
 with (folder/'worker.log').open('xb') as log:
  child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);once(folder/'child.json',dict(pid=child.pid,owned_pgid=child.pid,argv=argv))
  signature=None;last_progress=time.monotonic();last_hb=0.;signaled=False;stalled=False
  while child.poll() is None:
   hb=out/'heartbeat.json';sig=(hb.stat().st_mtime_ns,hb.stat().st_size) if hb.exists() else None
   if sig!=signature:signature=sig;last_progress=time.monotonic()
   if time.monotonic()-last_progress>1800 and not signaled:stalled=True;os.killpg(child.pid,signal.SIGTERM);signaled=True;last_progress=time.monotonic()
   if paused() and not signaled:os.killpg(child.pid,signal.SIGTERM);signaled=True
   if stalled and time.monotonic()-last_progress>120:os.killpg(child.pid,signal.SIGKILL);break
   if time.monotonic()-last_hb>30:
    heartbeat('WORKER_ACTIVE',task=task,model=model,child_stage=stage,pid=child.pid,paused=paused(),stalled=stalled);last_hb=time.monotonic()
    if tree_bytes(RUN)>8*2**30 and not paused():atomic(RUN/'global-stop.json',dict(reason='OUTPUT_8GIB_CAP'))
   time.sleep(.5)
  code=child.wait();marker=stage_marker(out,stage);value=dict(exit_code=code,stage=stage,task=task,model=model,stalled=stalled,user_stop=paused(),failure_log=bind(folder/'worker.log'))
  if code==0 and marker.exists():value['completion']=bind(marker)
  once(finished,value)
 if code==0 and 'completion' in value:
  if stage=='prepare':once(out/'authorization.json',dict(approved=True,protocol_sha256=q.sha(out/'protocol.json'),root_execution_config_sha256=q.sha(CONFIG)))
  return 'COMPLETE'
 return 'USER_STOPPED' if paused() else 'TECHNICAL_STOP'

def freeze_selection(task,states):
 units={};unavailable={}
 for model in MODELS:
  out=unit(task,model)
  if states[model]!='MODELS_SELECTED':unavailable[model]=states[model];continue
  selected=q.load(out/'selected.json');reload=q.load(out/'selected-reload.json')
  if reload.get('status')!='PASS' or reload.get('selected_sha256')!=q.sha(out/'selected.json'):raise PermissionError('Independent selected reload missing')
  gad=Path(selected['checkpoint'])/'manifest.json';warm=out/'training/checkpoints/W0200/manifest.json'
  wr=q.load(out/'W-reload.json')
  if wr.get('status')!='PASS' or wr['checkpoint_manifest_sha256']!=q.sha(warm):raise PermissionError('Independent W reload missing')
  units[model]=dict(protocol=bind(out/'protocol.json'),selected=bind(out/'selected.json'),reload=bind(out/'selected-reload.json'),GAD_checkpoint=bind(gad),W_checkpoint=bind(warm),W_reload=bind(out/'W-reload.json'))
 value=dict(task=task,declared_models=list(MODELS),all_size_selections_frozen_before_eval=True,units=units,unavailable=unavailable,config_sha256=q.sha(CONFIG),gold_read=False)
 once(RUN/'benchmarks'/task/'model-selection-frozen.json',value);return value

def declared_arms(model,task=None):
 if task is not None and (unit(task,model)/'optional-on-terminal.json').exists():return ['W-off','GAD-off']
 decision=q.load(RUN/'diagnostics/thinking-selection.json')['models'][model]
 return ['W-off','GAD-off']+(['base-on','W-on','GAD-on'] if decision.get('qualified') is True else [])

def freeze_benchmark(task,states):
 selection=q.load(RUN/'benchmarks'/task/'model-selection-frozen.json');specs=[]
 for model in selection['units']:
  if states[model]!='COMPLETE':raise PermissionError('All declared healthy-size answers must freeze before benchmark score')
  for arm in declared_arms(model,task):
   p=unit(task,model)/'evaluation'/arm/'predictions-frozen.json';f=q.load(p)
   if not f.get('complete'):raise PermissionError('Incomplete candidate arm')
   for entry in f['predictions']:q.bound(entry)
   specs.append(bind(p))
 value=dict(task=task,complete=True,prediction_freezes=specs,selection=bind(RUN/'benchmarks'/task/'model-selection-frozen.json'),thinking_selection=bind(RUN/'diagnostics/thinking-selection.json'),optional_on_terminal={m:bind(unit(task,m)/'optional-on-terminal.json') for m in selection['units'] if (unit(task,m)/'optional-on-terminal.json').exists()},unavailable=selection['unavailable'],all_declared_answers_before_gold=True)
 once(RUN/'benchmarks'/task/'all-predictions-frozen.json',value);return value

def comparators(c,task,evalspec):
 """Reuse only bound off scores; verify their frozen prediction receipts and query membership."""
 result={};bindings=[]
 specs=c['comparators']
 canonical={str(r['id']):r for r in [json.loads(x) for x in q.bound(evalspec['queries']).read_text().splitlines() if x.strip()]} 
 expected_ids=set(canonical)
 original=q.load(ROOT/'runs/2026-10-24-ust100-sg60-modes/local-qwen-manifest.json')
 if original['tasks'][task]['queries']['sha256']!=evalspec['queries']['sha256']:raise PermissionError('Common Eval facts differ from historical canonical queries')
 for name,spec in specs.items():
  path=q.bound(spec);value=q.load(path);bindings.append(spec)
  if name in ('08','2b'):
   fp=q.bound(value['off_freeze']);freeze=q.load(fp)
   members=[]
   for pred in freeze['predictions']:
    cached=q.load(q.bound(pred))
    if cached['task']==task and cached['thinking']=='off':
     if cached.get('max_new_tokens')!=256 or cached.get('input_truncated') is not False:raise PermissionError('Original off configuration mismatch')
     members.append(str(cached['id']))
   if len(members)!=len(expected_ids) or set(members)!=expected_ids:raise PermissionError('Original common Eval member IDs differ')
   result['original'+name]=value['scores'][task if name=='08' else task+'/off']
  elif name=='DS':
   fp=path.parent/'predictions-frozen.json'
   if q.sha(fp)!=value['freeze_sha256']:raise PermissionError('DS common-score freeze differs')
   freeze=q.load(fp)
   members={'adapted-off':[],'adapted-on-high':[]}
   for pred in freeze['predictions']:
    cached=q.load(q.bound(pred))
    if cached['task']==task and cached['arm'] in members:members[cached['arm']].append(str(cached['id']))
   packet_path=ROOT/'runs/2026-10-24-ust100-sg60-modes/execution-packet.json'
   if q.sha(packet_path)!=value['packet_sha256'] or q.sha(packet_path)!=freeze['packet_sha256']:raise PermissionError('DS frozen packet mismatch')
   packet=q.load(packet_path)
   if packet['reuse']['sha256']!=freeze['reuse_sha256']:raise PermissionError('DS reused120 receipt differs')
   reused=[json.loads(line) for line in q.bound(packet['reuse']).read_text().splitlines() if line.strip()]
   if len(reused)!=120:raise PermissionError('DS original120 scope')
   source_cache={}
   for row in reused:
    cached=q.load(q.bound(row['original_prediction']));q.bound(row['original_freeze'])
    if cached['body_sha256']!=row['body_sha256'] or cached['tag']!=row['original_tag'] or cached['identity']!=row['identity'] or q.digest(row['body'])!=row['body_sha256']:raise PermissionError('DS reused prediction/body/provider identity mismatch')
    if row['task']==task:
     if row['arm']!='adapted-off' or str(row['id']) not in canonical:raise PermissionError('DS reused off scope differs')
     spec=row['request_source'];path=q.bound(spec)
     if str(path) not in source_cache:source_cache[str(path)]=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
     source=source_cache[str(path)][row['source_line_1based']-1]
     if str(source['id'])!=str(row['id']) or source['adapted_messages']!=canonical[str(row['id'])]['adapted_messages']:raise PermissionError('DS reused Common Eval facts differ')
     members['adapted-off'].append(str(row['id']))
   if any(len(ids)!=len(expected_ids) or set(ids)!=expected_ids for ids in members.values()):raise PermissionError('DS common Eval member IDs differ')
   result['DS']=value['scores'][task+'/adapted-off'];result['DS-on-high-reference']=value['scores'][task+'/adapted-on-high']
  else:raise PermissionError('Unexpected comparator')
 # Canonical source manifests are bound by the new data-ready receipt. No different members/caps are silently re-labelled.
 common=q.load(q.bound(c['tasks'][task]['manifest']))['common_eval']
 if common!=evalspec:raise PermissionError('Common Eval scope changed')
 for name,value in result.items():
  if value['total']!=common['count']:raise PermissionError('Comparator denominator mismatch')
 return result,bindings

def score_benchmark(c,task,states):
 freeze=freeze_benchmark(task,states);m=q.load(q.bound(c['tasks'][task]['manifest']));ev=m['common_eval']
 import run_llm_task_sweep_20261023 as scorer
 # Gold is opened only here after every selected-size declared arm is immutable.
 gold=q.load(q.bound(ev['gold']));rows=[json.loads(x) for x in q.bound(ev['queries']).read_text().splitlines() if x.strip()];by={str(r['id']):r for r in rows}
 candidates={};bits={}
 selected=q.load(RUN/'benchmarks'/task/'model-selection-frozen.json')['units']
 for model in selected:
  for arm in declared_arms(model,task):
   fp=unit(task,model)/'evaluation'/arm/'predictions-frozen.json';fr=q.load(fp);items=[q.load(q.bound(s)) for s in fr['predictions']]
   if len(items)!=ev['count'] or {str(x['id']) for x in items}!=set(by):raise PermissionError('Fixed Eval membership differs')
   pairs=[scorer.score_prediction(x,dict(kind='qa',scorer=ev['scorer']),gold[str(x['id'])],by[str(x['id'])].get('metadata',{})) for x in items]
   key=model+'/'+arm;bits[key]={str(x['id']):bool(pair[0]) for x,pair in zip(items,pairs)}
   candidates[key]=dict(correct=sum(a for a,b in pairs),valid=sum(b for a,b in pairs),total=len(items),length_capped=sum(x['finish_reason'] not in ('stop',) for x in items),no_final=sum(x.get('no_final',False) for x in items),forced_close=sum(x.get('forced_close',False) for x in items),output_tokens=sum(x['output_tokens'] for x in items),generation_seconds=sum(x['generation_seconds'] for x in items),freeze=bind(fp))
 paired={}
 for model in selected:
  left=bits[model+'/GAD-off'];right=bits[model+'/W-off'];paired[model+'/GAD-vs-W']=dict(total=len(left),left_only=sum(left[k] and not right[k] for k in left),right_only=sum(right[k] and not left[k] for k in left),both_correct=sum(left[k] and right[k] for k in left),both_wrong=sum(not left[k] and not right[k] for k in left))
 baselines,bindings=comparators(c,task,ev)
 result=dict(status='COMPLETE' if len(selected)==2 else 'PARTIAL_SIZE_RESULTS',task=task,states=states,candidates=candidates,comparators=baselines,comparator_score_bindings=bindings,paired_correctness=paired,all_prediction_freeze=bind(RUN/'benchmarks'/task/'all-predictions-frozen.json'),external_teacher_cost=ledger_summary(),source=bind(Path(__file__)),scorer=bind(ROOT/'scripts/run_llm_task_sweep_20261023.py'),exposed_screen_not_sealed_final=True)
 once(RUN/'benchmarks'/task/'scores.json',result);write_report(task,result);return result

def write_report(task,result):
 def cell(x):return 'NA' if x is None else f"{x['correct']}/{x['total']} ({100*x['correct']/x['total']:.2f}%), valid {x['valid']}/{x['total']}, capped {x.get('length_capped','unrecorded')}"
 lines=[f'# {task}: DS / native Qwen / warmup / GAD',f"Status: {result['status']}",'','| Mode | Model | Original | DS | W | GAD |','|---|---|---|---|---|---|']
 for model in MODELS:
  cand=result['candidates'];base=result['comparators'];lines.append('| off | '+model+' | '+cell(base.get('original'+model))+' | '+cell(base.get('DS'))+' | '+cell(cand.get(model+'/W-off'))+' | '+cell(cand.get(model+'/GAD-off'))+' |')
  if model+'/GAD-on' in cand:lines.append('| bounded on | '+model+' | '+cell(cand.get(model+'/base-on'))+' | separately bound historical DS-on; unequal compute | '+cell(cand.get(model+'/W-on'))+' | '+cell(cand.get(model+'/GAD-on'))+' |')
  else:lines.append('\n'+model+' optional on: NOT_QUALIFIED_SKIP or explicit technical fallback; decision bound in thinking-selection.json.')
 for model in MODELS:
  out=unit(task,model)
  if (out/'training/complete.json').exists():lines+=['',model+' training: '+json.dumps(q.load(out/'training/complete.json'),sort_keys=True),model+' selected step: '+str(q.load(out/'selected.json')['step'])]
 quality=RUN/'teacher'/task/'train/quality.local.json'
 if quality.exists():lines+=['','New matched8 TeacherTrain audit (report only, no filtering): '+json.dumps({k:v for k,v in q.load(quality).items() if k in ('correct','valid','total','by_family','teacher_mode','mode','output_cap','completion_cap','status')},sort_keys=True)]
 lines+=['','External teacher accounting (shared100USD ledger; oldA-D versus newE-H, conservative reservations are not provider invoices): '+json.dumps(result['external_teacher_cost'],sort_keys=True)]
 lines+=['','Historical DS-on-high common-Eval reference: '+cell(result['comparators'].get('DS-on-high-reference'))+'; different completion budgets from new TeacherTrain, no exact-condition equivalence claim.']
 lines+=['','Train200; PolicyTrain50 distinct prompts; group8/max256; Val32 atW/25/50; selected only25/50. W controls teacher-answer SFT; W→GAD is additional-training gain, not equal-compute SFT ablation.','UST published-target ambiguity remains; SG procedural→author Eval is distribution transfer. No private-data or privacy-proof claim. Ordinary invalid/capped answers retained as wrong.','No old GAD/320 queue resumed; pipeline has no API client. Actual teacher cost follows the shared original100USD ledger/new50USD package gate.','',json.dumps(result['paired_correctness'],sort_keys=True)]
 report=ROOT/'reports/overnight-onpolicy-2026-10-25'/f'{task}-results.md';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n')
 atomic(RUN/'benchmarks'/task/'report.json',dict(path=str(report.relative_to(ROOT)),sha256=q.sha(report)))

def main():
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);argparse.ArgumentParser().parse_args()
 c=check_config();from night_dual_20261022 import lock,cgroup_limits_verified
 with lock(RUN/'pipeline/controller.lock'):
  cgroup_limits_verified()
  for task in TASKS:
   if (RUN/'benchmarks'/task/'scores.json').exists():continue
   states={m:'WAITING_PREREQUISITES' for m in MODELS};prerequisite_failure=None
   while not paused():
    try:
     if diagnostics_ready() and teacher_ready(c,task):break
    except Exception as error:prerequisite_failure=dict(type=type(error).__name__,message=str(error));break
    heartbeat('WAITING_DIAGNOSTICS_OR_TEACHER',task=task);time.sleep(15)
   if paused():break
   if prerequisite_failure:
    once(RUN/'benchmarks'/task/'prerequisite-stop.json',prerequisite_failure);continue
   try:audit_teacher(task)
   except Exception as error:
    once(RUN/'benchmarks'/task/'teacher-audit-stop.json',dict(type=type(error).__name__,message=str(error)));continue
   for model in MODELS:
    while not paused():
     waiting=False
     for stage in ('prepare','preflight','train','select','reload'):
      state=run_stage(c,task,model,stage)
      if state=='WAITING_RESOURCE':heartbeat(state,task=task,model=model);time.sleep(20);waiting=True;break
      if state!='COMPLETE':states[model]=state;break
     else:states[model]='MODELS_SELECTED'
     if not waiting:break
    if paused():break
   if paused():break
   selection=freeze_selection(task,states)
   if not selection['units']:
    once(RUN/'benchmarks'/task/'no-candidates.json',dict(task=task,states=states,status='NO_COMPLETE_UNIT'));continue
   for model in selection['units']:
    for arm in declared_arms(model,task):
     if paused():break
     while not paused():
      state=run_stage(c,task,model,'evaluate:'+arm)
      if state=='WAITING_RESOURCE':heartbeat(state,task=task,model=model);time.sleep(20);continue
      break
     if paused():break
     if state!='COMPLETE':
      if arm.endswith('-on'):
       once(unit(task,model)/'optional-on-terminal.json',dict(status='SKIPPED_TECHNICAL_OFF_FALLBACK',failed_arm=arm,reason=state,no_resampling=True,preserved_partial_answers=[bind(f) for f in (unit(task,model)/'evaluation').rglob('*.json') if '-on' in str(f)]));states[model]='COMPLETE'
      else:states[model]=state
      break
    else:states[model]='COMPLETE'
   if paused():break
   if all(states[m]=='COMPLETE' for m in selection['units']):score_benchmark(c,task,states)
   else:once(RUN/'benchmarks'/task/'partial-unscored.json',dict(states=states,reason='Declared selected-size answers incomplete; no benchmark gold read'))
  available=[q.load(RUN/'benchmarks'/t/'scores.json') for t in TASKS if (RUN/'benchmarks'/t/'scores.json').exists()]
  status='PAUSED_BY_USER' if paused() else 'COMPLETE' if len(available)==2 and all(r['status']=='COMPLETE' for r in available) else 'PARTIAL_TECHNICAL_STOP'
  report=ROOT/'reports/overnight-onpolicy-2026-10-25/final-results.md';report.parent.mkdir(parents=True,exist_ok=True)
  bodies=[(ROOT/'reports/overnight-onpolicy-2026-10-25'/f'{t}-results.md').read_text() for t in TASKS if (ROOT/'reports/overnight-onpolicy-2026-10-25'/f'{t}-results.md').exists()]
  report.write_text('\n'.join(['# Four-unit native on-policy results','','Status: '+status,'']+bodies)+'\n')
  atomic(RUN/'pipeline/completed.json',dict(status=status,epoch=time.time(),local_api_calls=0,external_teacher_cost=ledger_summary(),final_report=bind(report)));heartbeat(status)
  state=q.load(RUN/'goal-state.json');state.update(status=status,pipeline_completed=str(RUN/'pipeline/completed.json'),old_queues_resumed=False);atomic(RUN/'goal-state.json',state)






def ledger_summary():
 import sqlite3
 path=ROOT/'runs/2026-10-24-ust-sg-onpolicy/budget/teacher.sqlite'
 if not path.exists():return dict(status='LEDGER_NOT_FOUND',local_api_calls=0)
 with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5) as db:
  db.execute('PRAGMA query_only=ON')
  rows=db.execute('SELECT pool,COUNT(*),COALESCE(SUM(pico),0),COALESCE(SUM(CASE WHEN usage_known=1 THEN pico ELSE 0 END),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0),COALESCE(SUM(CASE WHEN settled=0 THEN pico ELSE 0 END),0) FROM calls GROUP BY pool').fetchall()
 pools={r[0]:dict(http=r[1],conservative_usd=r[2]/1e12,known_usage_usd=r[3]/1e12,unknown_usage=r[4],inflight=r[5],reserved_usd=r[6]/1e12) for r in rows}
 def group(names):return {k:sum(v[k] for n,v in pools.items() if n in names) for k in ('http','conservative_usd','known_usage_usd','unknown_usage','inflight','reserved_usd')}
 return dict(status='READONLY_SNAPSHOT',epoch=time.time(),ledger_path=str(path.relative_to(ROOT)),local_api_calls=0,historical_A_D=group('ABCD'),new_E_H=group('EFGH'),pools=pools,parent_global_usd=100,new_packages_hard_usd=50,scope='Teacher route shared ledger only; not all-project expenses, account balance or invoice')

if __name__=='__main__':main()

