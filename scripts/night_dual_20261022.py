#!/usr/bin/env python3
"""New-round guarded coordinator. Imports are API/GPU/credential free."""
from __future__ import annotations
import argparse,contextlib,fcntl,hashlib,json,os,signal,subprocess,sys,time,uuid,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from privacy_synth.pilot_ledger import SqliteBudget
from privacy_synth.location_classification import CLASSES,ROLES,CLASS_DEFINITIONS,CLASS_DEFINITION_SOURCES,prompt,score,qualification_gate,digest,parse_label
RUN=ROOT/'runs/2026-10-22-night-dual'
RATES={'hit':.006,'miss':.3,'output':1.2}
SPECS={
 'cityfm_dev':{'logical':96,'calls':106,'input_tokens':106*8192,'output_tokens':106*64,'usd':10},
 'cityfm_screen':{'logical':96,'calls':106,'input_tokens':106*8192,'output_tokens':106*64,'usd':10},
 'ua_dev':{'logical':90,'calls':100,'input_tokens':100*8192,'output_tokens':100*64,'usd':10},
 'ua_screen':{'logical':100,'calls':110,'input_tokens':110*8192,'output_tokens':110*64,'usd':10},
 'bird_refresh':{'logical':200,'calls':220,'input_tokens':3604480,'output_tokens':225280,'usd':10}}
RESOURCE_POLICY={'windows_start_free_gib':20,'windows_floor_free_gib':8,'gpu_start_free_gib':11,'gpu_start_util_max_percent':10,'gpu_stable_seconds':60,'cuda_target_gib':9,'cpu_threads':4,'run_memory_max_gib':14,'windows_c_free_gib':20,'new_artifacts_cap_gib':8,'heartbeat_seconds':30,'no_progress_seconds':1800,'oom_stop_count':2,'maximum_restarts':1,'post_approval_deadline_hours':12}

class StopPhase(RuntimeError):pass

def active_runtime(run):
 run=Path(run);pointer=run/'runtime-active.json'
 return read(pointer) if pointer.exists() else {'package_dir':str(run/'workpackages'),'integrity_path':str(run/'runtime-integrity.json')}

def root_authorization(run):
 """The 12h clock starts at the explicitly approved root launch, never preparation."""
 p=Path(run)/'root-goal-approval.json'
 if not p.exists():raise PermissionError('Explicit root goal/workpackage confirmation required')
 a=read(p);now=time.time()
 if a.get('root_goal_approved') is not True or not a.get('goal_id'):raise PermissionError('Root goal not approved')
 if not isinstance(a.get('approved_at_epoch'),(int,float)):raise PermissionError('Root approval has no valid start time')
 start=float(a['approved_at_epoch'])
 if not start<=now<start+12*3600:raise PermissionError('Root 12h approval expired or in future')
 acceptance=Path(run)/'runtime-acceptance.json'
 if not acceptance.exists() or read(acceptance).get('status')!='PASS' or read(acceptance).get('production_launch_permitted') is not True:raise PermissionError('Local runtime/root key acceptance is still pending')
 manifest=Path(active_runtime(run)['integrity_path'])
 if not manifest.exists() or a.get('runtime_integrity_sha256')!=sha(manifest):raise PermissionError('Root approval lacks exact runtime integrity binding')
 for path,expected in read(manifest).items():
  if not Path(path).is_file() or sha(path)!=expected:raise PermissionError('Runtime changed after approval: '+path)
 return a

def host_snapshot(run):
 """Read-only host probes. Missing probes block launch rather than assuming idle."""
 ps=Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
 script="$m=Get-CimInstance Win32_OperatingSystem; $d=Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\"; @{windows_free_gib=([double]$m.FreePhysicalMemory/1048576);c_free_gib=([double]$d.FreeSpace/1073741824)} | ConvertTo-Json -Compress"
 proc=subprocess.run([str(ps),'-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=15,check=True)
 win=json.loads(proc.stdout.strip().lstrip('\ufeff'))
 gpu=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
 lines=gpu.stdout.strip().splitlines()
 if len(lines)!=1:raise StopPhase('Exactly one configured GPU required')
 free,util=[float(x.strip()) for x in lines[0].split(',')]
 size=sum(p.stat().st_size for p in Path(run).rglob('*') if p.is_file() and not p.is_symlink())
 return dict(win,gpu_free_gib=free/1024,gpu_util_percent=util,output_gib=size/2**30,at_epoch=time.time())

def resources_ok(snapshot,startup=False,gpu=False):
 p=RESOURCE_POLICY
 checks={'windows_ram':snapshot['windows_free_gib']>=p['windows_start_free_gib' if startup else 'windows_floor_free_gib'],
         'c_disk':snapshot['c_free_gib']>=p['windows_c_free_gib'],'new_outputs':snapshot['output_gib']<p['new_artifacts_cap_gib']}
 if gpu:checks.update(gpu_free=snapshot['gpu_free_gib']>=p['gpu_start_free_gib'],gpu_idle=snapshot['gpu_util_percent']<=p['gpu_start_util_max_percent'])
 return all(checks.values()),checks

def cgroup_limits_verified():
 """Check the effective service cgroup, without modifying host settings."""
 group=next((line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')),None)
 if group is None:raise StopPhase('cgroup v2 resource scope unavailable')
 folder=Path('/sys/fs/cgroup')/group.lstrip('/')
 mem=(folder/'memory.max').read_text().strip();quota=(folder/'cpu.max').read_text().split()
 if mem=='max' or int(mem)>14*2**30:raise StopPhase('Owned service must enforce MemoryMax<=14GiB')
 if quota[0]=='max' or int(quota[0])/int(quota[1])>4:raise StopPhase('Owned service must enforce CPUQuota<=400%')
 return {'cgroup':group,'memory_max_bytes':int(mem),'cpu_quota':quota}

def supervised_worker_verified(run):
 expected=str(Path(run).resolve())
 if os.environ.get('NIGHT_DUAL_RUN')!=expected:raise PermissionError('Science workers require the bounded central supervisor')
 pid=int(os.environ.get('NIGHT_DUAL_SUPERVISOR_PID','0'));start=os.environ.get('NIGHT_DUAL_SUPERVISOR_START')
 if not start or process_start(pid)!=start:raise PermissionError('Supervisor identity missing/stale')
 if Path(f'/proc/{pid}/cgroup').read_text()!=Path('/proc/self/cgroup').read_text():raise PermissionError('Worker escaped bounded supervisor cgroup')
 return cgroup_limits_verified()

def progress_signature(paths):
 result=[]
 for base in paths:
  base=Path(base)
  files=base.rglob('*') if base.is_dir() else [base]
  for p in files:
   if p.is_file() and not p.is_symlink():
    st=p.stat();result.append((str(p),st.st_size,st.st_mtime_ns))
 return digest(sorted(result))

def stop_owned_process(worker,identity,grace=20):
 if worker.poll() is not None:return
 pid=int(identity['pid'])
 if process_start(pid)!=identity.get('starttime') or os.getpgid(pid)!=pid:raise StopPhase('Owned process identity/group changed; refuse signal')
 os.killpg(pid,signal.SIGTERM)
 try:worker.wait(timeout=grace)
 except subprocess.TimeoutExpired:
  # Process-group ownership was established above while the leader was live.
  os.killpg(pid,signal.SIGKILL);worker.wait(timeout=5)

class Supervisor:
 def __init__(self,run,approval,probe=host_snapshot,clock=time.time,sleep=time.sleep,poll=30):
  self.run=Path(run);self.approval=approval;self.probe=probe;self.clock=clock;self.sleep=sleep;self.poll=poll
  self.start=float(approval['approved_at_epoch']);self.deadline=self.start+12*3600;self.long_cutoff=self.start+11*3600;self.stop=False
 def heartbeat(self,phase,**details):
  atomic(self.run/'supervisor-heartbeat.json',dict(phase=phase,updated_epoch=self.clock(),deadline_epoch=self.deadline,**details))
 def check_time(self,new_stage=False):
  if self.stop:raise StopPhase('Owner stop requested; own workers saved/stopped')
  now=self.clock()
  if now>=self.deadline:raise StopPhase('12h deadline reached')
  if new_stage and now>=self.long_cutoff:raise StopPhase('11h cutoff: no new long stage')
 def consume_recovery(self,stage,reason):
  path=self.run/'recovery-count.json';guard=self.run/'locks/recovery-count.lock';guard.parent.mkdir(parents=True,exist_ok=True)
  with guard.open('a+') as stream:
   fcntl.flock(stream,fcntl.LOCK_EX)
   obj=read(path) if path.exists() else {'count':0,'events':[]}
   if obj['count']>=1:raise StopPhase('Global single recovery allowance exhausted')
   obj['count']+=1;obj['events'].append({'stage':stage,'reason':reason,'at_epoch':self.clock()});atomic(path,obj)
 def wait_resources(self,gpu=False,startup=False):
  stable=None
  while True:
   self.check_time(new_stage=True);snap=self.probe(self.run);ok,checks=resources_ok(snap,startup=startup,gpu=gpu)
   if ok:
    if stable is None:stable=self.clock()
    if not gpu or self.clock()-stable>=60:return snap
   else:stable=None
   self.heartbeat('WAITING_RESOURCE',snapshot=snap,checks=checks,stable_since=stable)
   self.sleep(min(self.poll,max(0,self.long_cutoff-self.clock())))
 def stage(self,name,command,progress_paths,gpu=False,completion=None,recoverable=False,recovery_check=None,yield_check=None):
  """One owned child; safe recovery only where the worker verifies saved state."""
  if completion and Path(completion).exists():return read(completion)
  self.check_time(new_stage=True);self.wait_resources(gpu=gpu)
  statepath=self.run/'stages'/(name+'.json');old=read(statepath) if statepath.exists() else {}
  recoveries=int(old.get('recoveries',0));ooms=int(old.get('oom_count',0))
  while True:
   self.check_time(new_stage=True)
   if old.get('status')=='RUNNING':
    identity=old['identity']
    if process_start(identity['pid'])==identity.get('starttime'):raise StopPhase('A prior owned stage is still alive; refusing duplicate launch')
    if not recoverable or recoveries>=1 or (recovery_check and not recovery_check()):raise StopPhase('Interrupted stage requires explicit evidence-based recovery')
    self.consume_recovery(name,'interrupted owned stage');recoveries+=1
   if recoveries and recovery_check:
    if not recovery_check():raise StopPhase('No verified recovery checkpoint; refuse restart')
    if '--resume' not in command:command=[*command,'--resume']
   folder=self.run/'logs';folder.mkdir(parents=True,exist_ok=True);logpath=folder/(name+f'-start{recoveries}.log')
   env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',NIGHT_DUAL_RUN=str(self.run.resolve()),NIGHT_DUAL_SUPERVISOR_PID=str(os.getpid()),NIGHT_DUAL_SUPERVISOR_START=str(process_start(os.getpid())))
   with logpath.open('ab') as log:
    worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    identity={'pid':worker.pid,'starttime':process_start(worker.pid)}
    state={'status':'RUNNING','command':command,'identity':identity,'started_epoch':self.clock(),'recoveries':recoveries,'oom_count':ooms}
    atomic(statepath,state);last=progress_signature(progress_paths);progress_at=self.clock();failure=None;yielding=False
    try:
     while worker.poll() is None:
      self.check_time();snap=self.probe(self.run);ok,checks=resources_ok(snap)
      if not ok:raise StopPhase('Resource floor breached: '+json.dumps(checks))
      signature=progress_signature(progress_paths)
      if signature!=last:last=signature;progress_at=self.clock()
      if not yielding and yield_check and yield_check():
       # Cooperative GPU priority handoff: GAD saves at its next complete item
       # boundary while this monitor continues deadline/resource enforcement.
       if process_start(worker.pid)!=identity['starttime']:raise StopPhase('Priority yield worker identity changed')
       os.kill(worker.pid,signal.SIGTERM);yielding=True
      self.heartbeat(name,snapshot=snap,worker=identity,last_progress_epoch=progress_at,recoveries=recoveries)
      if self.clock()-progress_at>=1800:raise StopPhase('No saved progress for30minutes')
      self.sleep(min(self.poll,max(0,self.deadline-self.clock())))
     if yielding and worker.returncode==75 and recovery_check and recovery_check():
      self.consume_recovery(name,'Cooperative priority handoff to qualified original Qwen')
      state.update(status='YIELDED_GPU',finished_epoch=self.clock(),recoveries=1);atomic(statepath,state);return state
     if worker.returncode:raise StopPhase('Owned stage exited '+str(worker.returncode))
    except BaseException as error:
     failure=str(error);stop_owned_process(worker,identity)
   tail=logpath.read_text(errors='replace')[-65536:];ooms+=int('out of memory' in tail.lower() or 'cuda_oom' in tail.lower())
   if failure:
    state.update(status='STOPPED',reason=failure,oom_count=ooms,finished_epoch=self.clock());atomic(statepath,state)
    if ooms>=2:raise StopPhase('Two OOMs: phase stopped')
    if 'No saved progress' in failure and recoverable and recoveries<1 and (not recovery_check or recovery_check()):
     self.consume_recovery(name,failure);recoveries+=1;old={};self.wait_resources(gpu=gpu);continue
    raise StopPhase(failure)
   if completion and not Path(completion).exists():raise StopPhase('Stage exited without required completion artifact: '+str(completion))
   state.update(status='COMPLETE',finished_epoch=self.clock(),oom_count=ooms);atomic(statepath,state)
   return read(completion) if completion else state

def read(p):return json.loads(Path(p).read_text())
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex);q.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(q,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ledger(run):return SqliteBudget(Path(run)/'budget/new-round.sqlite',500,{k:{x:v for x,v in s.items() if x!='logical'} for k,s in SPECS.items()},RATES)
def process_start(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
@contextlib.contextmanager
def lock(path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Owned worker/GPU lock already held')
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def require_approval(run,package):
 """Must precede credential reads, client creation, requests and reservations."""
 run=Path(run);f=Path(active_runtime(run)['package_dir'])/f'{package}.json';ap=run/'approvals'/f'{package}.json';gp=run/'root-goal-approval.json'
 if not f.exists() or not ap.exists() or not gp.exists():raise PermissionError('Separate package and root goal approval records required before science/API launch')
 w=read(f);a=read(ap);g=read(gp)
 root_authorization(run)
 if w['status']!='FROZEN_AWAITING_CONFIRMATION':raise PermissionError('Workpackage is not executable/frozen')
 if not (a.get('authorized') is True and a.get('package_sha256')==sha(f) and a.get('provider')=='DeepSeek official' and a.get('model')=='deepseek-flash' and a.get('thinking')=='disabled' and a.get('usd_cap')==10 and a.get('parent_usd_cap')==500):raise PermissionError('Exact provider/model/budget/package approval mismatch')
 if not(g.get('root_goal_approved') is True and g.get('package_hashes',{}).get(package)==sha(f) and g.get('goal_id')):raise PermissionError('Root goal approval not bound to package')
 approved_at=float(a['approved_at_epoch'])
 if not approved_at<=time.time()<approved_at+12*3600:raise PermissionError('Post-approval12h deadline expired or future approval')
 for p,expected in w['bindings'].items():
  if not Path(p).is_file() or sha(p)!=expected:raise PermissionError('Frozen source/prompt/input/code changed: '+p)
 return w,a

def prepare(run=RUN,revision=1):
 run=Path(run);out=run/('workpackages' if revision==1 else f'workpackages-v{revision}');out.mkdir(parents=True,exist_ok=True)
 prep=run/'preparation/cityfm';manifest=prep/'artifact-manifest.json'
 code=[Path(__file__),ROOT/'scripts/prepare_location_night_20261022.py',ROOT/'scripts/deepseek_ipd_interface.py',ROOT/'scripts/ipd_native_migration_20261019.py',ROOT/'src/privacy_synth/location_classification.py',ROOT/'src/privacy_synth/pilot_ledger.py',ROOT/'src/privacy_synth/round_budget.py',ROOT/'pyproject.toml',ROOT/'uv.lock']
 provider=ROOT/'reports/night-dual-2026-10-22/provider-verification.json'
 verified=read(provider)
 if verified['rates_usd_per_million_peak']!=RATES:raise RuntimeError('Official rate verification differs from budget rates')
 bindings={str(p):sha(p) for p in code+[provider]}
 tokenizer=ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json'
 bindings[str(tokenizer)]=sha(tokenizer)
 if manifest.exists():
  bindings[str(manifest)]=sha(manifest)
  for rel,expected in read(manifest).items():
   p=prep/rel
   if sha(p)!=expected:raise RuntimeError('Preparation artifact changed: '+str(p))
   bindings[str(p)]=expected
 audit=read(prep/'audit.json') if (prep/'audit.json').exists() else {}
 token_package=read(prep/'external-workpackage.proposed.json') if (prep/'external-workpackage.proposed.json').exists() else {}
 contract=run/'preparation/cityfm-contract-v2.json'
 amendment={'status':'FROZEN_INPUT_CONTRACT_RUNTIME_ACCEPTANCE_PENDING','supersedes':'Original cityfm/prompt-skeletons.json and request skeletons only; source/payload/membership files unchanged','source_manifest_sha256':sha(manifest),'class_definitions':CLASS_DEFINITIONS,'class_definition_sources':CLASS_DEFINITION_SOURCES,'variants':[prompt({},0)['system']],'adaptation':'One fixed dev prompt0,32logical calls, then one screen96; no automatic quality adaptation','input_max_tokens':8192,'output_max_tokens':64}
 if contract.exists() and read(contract)!=amendment:raise RuntimeError('Immutable CityFM contract-v2 differs')
 if not contract.exists():atomic(contract,amendment)
 bindings[str(contract)]=sha(contract)
 if audit:
  for source in audit['source'].values():
   if sha(source['path'])!=source['sha256']:raise RuntimeError('Raw source changed after preparation')
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(tokenizer))
 actual={}
 for package,role in [('cityfm_dev',ROLES[1]),('cityfm_screen',ROLES[2])]:
  rows=[]
  if (prep/role/'inputs.jsonl').exists():
   for row in loadlines(prep/role/'inputs.jsonl'):
    for variant in [0]:
     req=prompt(row['payload'],variant)
     nt=len(tok.encode('<｜begin▁of▁sentence｜>'+req['system']+'<｜User｜>'+req['user']+'<｜Assistant｜>',add_special_tokens=False).ids)
     if nt>8192:raise RuntimeError('Frozen candidate exceeds8192; no silent truncation')
     rows.append({'id':row['id'],'variant':variant,'request':req,'request_sha256':digest(req),'estimated_input_tokens':nt})
  actual[package]=rows
 package_status={}
 for name,spec in SPECS.items():
  sources_ready=name.startswith('cityfm') and audit.get('status')=='PREPARED_PENDING_MODEL_PACKAGE_APPROVAL' and token_package.get('status')=='AWAITING_EXPLICIT_MODEL_BUDGET_CONFIRMATION'
  status='FROZEN_AWAITING_CONFIRMATION' if sources_ready else 'BLOCKED_SOURCE_OR_QUOTA'
  w={'name':name,'status':status,'provider':'DeepSeek official','model':'deepseek-flash','expected_identity':'DeepSeek-V4.1-Flash-1017','thinking':'disabled','temperature':0,'spec':spec,'rates_usd_per_million':RATES,'worst_case_usd':(spec['input_tokens']*.3+spec['output_tokens']*1.2)/1e6,'parent_new_usd_cap':500,'unallocated_parent_usd':450,'bindings':dict(bindings),'input_max_tokens':16384 if name=='bird_refresh' else 8192,'output_max_tokens':1024 if name=='bird_refresh' else 64,'retry_policy':'At most one technical retry per item; no correctness retry. Unknown transport outcome stops rather than blind resend; all attempts charged.','public_proxy_exception':True,'data':'Public target physicalgeometry + sanitized neighbor categories/counts/relativegeometry; no targetname/address/use/ID/wiki/absolutecoords or gold' if name.startswith('cityfm') else 'Blocked until paired source/input manifest exists','local_alternative':'Fixed majority and logistic classifiers handle diagnostics without API.','sealed':'No CityFM sealed scoring or training','resource_policy':RESOURCE_POLICY,'implementation':'CityFM executor and bounded coordinator; UrbanAtlas source pairing blocks that alternative.'}
  if name in actual:
   rows=actual[name];requestfile=out/(name+'-requests.json')
   if requestfile.exists() and read(requestfile)!=rows:raise RuntimeError('Actual request freeze differs')
   if not requestfile.exists():atomic(requestfile,rows)
   w['bindings'][str(requestfile)]=sha(requestfile)
   w['allowed_request_sha256']={f"{r['variant']}:{r['id']}":r['request_sha256'] for r in rows}
   if name=='cityfm_dev':nominal=sum(r['estimated_input_tokens'] for r in rows)
   else:nominal=sum(max(r['estimated_input_tokens'] for r in rows if r['id']==k) for k in {r['id'] for r in rows})
   planned=len(rows)
   w['actual_tokens']={'logical_calls_planned':planned,'logical_call_allocation_max':spec['logical'],'input_without_retries_upper':nominal,'input_max_per_call':max((r['estimated_input_tokens'] for r in rows),default=0),'output_without_retries_max':planned*64,'local_tokenizer_exact_api_usage_may_differ':True}
   w['estimated_cost_no_retry_miss_usd']=(nominal*.3+planned*64*1.2)/1e6
  elif name=='bird_refresh':
   bird=run/'preparation/bird'
   cache=bird/'cache-audit.json'
   for path in [bird/'train200.json',cache,bird/'conditional-refresh-work-package.json']:
    if path.exists():w['bindings'][str(path)]=sha(path)
   if cache.exists() and read(cache).get('status')=='CACHE_MATCH':
    w.update(status='FROZEN_CONDITIONAL_CACHE_MATCH_NO_SEND',data='Same frozen200 public BIRD schema/column meanings/two sample rows/question/evidence; no gold or protected rows',local_alternative='Verified immutable200 teacher cache;0new calls required',implementation='Cache matched; refresh intentionally inactive. A later invalid cache requires a new concrete approved refresh route.')
   status=w['status']
  else:
   w['status']='BLOCKED_ASSET_PAIRING';status=w['status']
   w['data']='No paired UrbanAtlas/POI rows available; no transmission permitted'
   for path in [run/'preparation/assets/sarl-bfe0c50.zip',ROOT/'reports/night-dual-2026-10-22/assets.md']:
    if path.exists():w['bindings'][str(path)]=sha(path)
  f=out/f'{name}.json'
  if f.exists():
   if read(f)!=w:raise RuntimeError('Immutable workpackage differs; do not overwrite existing freeze')
  else:atomic(f,w)
  package_status[name]={'status':status,'sha256':sha(f),'worst_case_usd':w['worst_case_usd']}
 base=ROOT/'artifacts/hf-home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc'
 model_manifest={str(p):sha(p) for p in sorted(base.iterdir()) if p.is_file() and p.suffix in ('.json','.safetensors')}
 baseline=run/'baseline-model.json'
 if baseline.exists() and read(baseline)!=model_manifest:raise RuntimeError('Cached original checkpoint changed')
 if not baseline.exists():atomic(baseline,model_manifest)
 integrity={str(p):sha(p) for p in code+[baseline,ROOT/'scripts/bird_gad_night_20261022.py',ROOT/'src/privacy_synth/onpolicy_gad.py']}
 for p in out.glob('*.json'):integrity[str(p)]=sha(p)
 active=run/'preparation/bird/preparation-active.json'
 if active.exists():
  integrity[str(active)]=sha(active)
  for key in ('protocol_path','integrity_path','preparation_path'):
   path=Path(read(active)[key]);integrity[str(path)]=sha(path)
 manifestpath=run/('runtime-integrity.json' if revision==1 else f'runtime-integrity-v{revision}.json')
 if manifestpath.exists() and read(manifestpath)!=integrity:raise RuntimeError('Immutable runtime integrity differs')
 if not manifestpath.exists():atomic(manifestpath,integrity)
 atomic(run/'runtime-active.json',{'package_dir':str(out),'integrity_path':str(manifestpath),'integrity_sha256':sha(manifestpath),'revision':revision})
 atomic(run/'runtime-status.json',{'status':'READY_UNAPPROVED' if audit.get('status')=='PREPARED_PENDING_MODEL_PACKAGE_APPROVAL' else 'PREPARED_BLOCKED_SOURCE_OR_QUOTA','packages':package_status,'network_calls':0,'gpu_model_runs':0,'production_training':0,'requires_explicit_package_approval':True,'root_goal_record_required':True,'resource_policy':RESOURCE_POLICY,'runtime_integrity_sha256':sha(manifestpath),'runtime_limitations':['UrbanAtlas blocked on exact paired assets; BIRD cache matches so refresh stays inactive.','60second resource stability and12hour runtime are configured; seconds-long detached tests do not establish12h persistence.']})
 return package_status

class OfficialClient:
 def __init__(self,run,package):
  self.run=Path(run);self.package=package;self.w,self.approval=require_approval(run,package)
  self.ledger=ledger(run)
  # Deliberately no dotenv implicit lookup or credential access before approval.
  self.key=os.environ.get('DEEPSEEK_API_KEY')
  if not self.key:
   sys.path.insert(0,str(ROOT/'scripts'))
   import deepseek_ipd_interface as existing
   self.key=existing.load_key()
   existing.apply_proxy()
  if not self.key:raise RuntimeError('Approved package requires local DEEPSEEK_API_KEY; no alternate provider')
 def request(self,item,variant,request):
  import urllib.request,urllib.error
  require_approval(self.run,self.package)
  if request['max_tokens']!=self.w['output_max_tokens']:raise RuntimeError('Output contract mismatch')
  allowed=self.w.get('allowed_request_sha256',{})
  if allowed.get(f'{variant}:{item}')!=digest(request):raise RuntimeError('Request differs from frozen actual-data workpackage')
  from tokenizers import Tokenizer
  tok=Tokenizer.from_file(str(ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json'))
  text='<｜begin▁of▁sentence｜>'+request['system']+'<｜User｜>'+request['user']+'<｜Assistant｜>'
  nt=len(tok.encode(text,add_special_tokens=False).ids)
  if nt>self.w['input_max_tokens']:raise RuntimeError('Overlong frozen request; no truncation/task dropping')
  logical=f'{self.package}:{variant}:{item}';folder=self.run/'requests'/self.package;folder.mkdir(parents=True,exist_ok=True)
  final=folder/(digest(logical)+'.json')
  if final.exists():
   saved=read(final)
   if saved['logical_id']!=logical or saved['request_sha256']!=digest(request):raise RuntimeError('Request identity mismatch')
   return saved['text']
  for retry in range(2):
   tag=logical+f':attempt{retry}';state=folder/(digest(tag)+'.attempt.json')
   if state.exists():raise RuntimeError('Attempt already exists/uncertain: refusing double send after restart')
   call=self.ledger.reserve(self.package,tag,self.w['input_max_tokens'],request['max_tokens'])
   atomic(state,{'tag':tag,'call_id':call,'status':'RESERVED_BEFORE_SEND','request_sha256':digest(request),'started_epoch':time.time()})
   body={'model':'deepseek-flash','messages':[{'role':'system','content':request['system']},{'role':'user','content':request['user']}],'thinking':{'type':'disabled'},'temperature':0,'max_tokens':request['max_tokens'],'stream':False,'response_format':{'type':'json_object'}}
   try:
    req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
    msg=data['choices'][0]['message'];usage=data.get('usage')
    self.ledger.settle(call,usage)
    atomic(state,{'tag':tag,'call_id':call,'status':'RECEIVED','response':data,'request_sha256':digest(request)})
    if str(data.get('model','')).lower() not in ('deepseek-flash','deepseek-v4.1-flash-1017') or msg.get('reasoning_content'):raise RuntimeError('Provider model identity/thinking mismatch')
    identity={'model':data['model'],'system_fingerprint':data.get('system_fingerprint')}
    if not identity['system_fingerprint']:raise RuntimeError('Provider omitted revision fingerprint')
    identity_path=self.run/'provider-identity.json'
    if identity_path.exists() and read(identity_path)!=identity:raise RuntimeError('Provider revision changed during frozen workload')
    if not identity_path.exists():atomic(identity_path,identity)
    text=msg.get('content','')
    atomic(final,{'logical_id':logical,'text':text,'response':data,'request_sha256':digest(request)})
    return text
   except urllib.error.HTTPError as e:
    self.ledger.settle(call,None);atomic(state,{'tag':tag,'call_id':call,'status':'HTTP_ERROR_UNKNOWN_USAGE','http_status':e.code})
    if e.code in (429,500,502,503,504) and retry==0:continue
    raise
   except Exception:
    # A post-response semantic guard must never settle an already-settled call twice.
    import sqlite3
    with sqlite3.connect(self.ledger.path) as c:settled=c.execute('SELECT settled FROM calls WHERE id=?',(call,)).fetchone()[0]
    if not settled:self.ledger.settle(call,None)
    raise
  raise RuntimeError('Technical retry exhausted')

def ordered_dev(inputs,gold):
 byid={r['id']:r for r in inputs};labels={r['id']:r['class'] for r in gold};smoke=[]
 for c in CLASSES:
  ids=sorted(k for k,v in labels.items() if v==c)
  if len(ids)!=4:raise RuntimeError('Dev32 requires exact4/class')
  smoke.append(ids[0])
 return smoke+sorted(set(byid)-set(smoke))
def loadlines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def classify(run=RUN):
 run=Path(run);require_approval(run,'cityfm_dev');require_approval(run,'cityfm_screen')
 prep=run/'preparation/cityfm';audit=read(prep/'audit.json')
 if audit['status']!='PREPARED_PENDING_MODEL_PACKAGE_APPROVAL':raise RuntimeError('Source/quota audit blocks launch')
 with lock(run/'locks/classification.lock'):
  client=OfficialClient(run,'cityfm_dev');inputs=loadlines(prep/ROLES[1]/'inputs.jsonl');gold=loadlines(prep/ROLES[1]/'gold.local.jsonl')
  byid={r['id']:r['payload'] for r in inputs};labels={r['id']:r['class'] for r in gold};order=ordered_dev(inputs,gold)
  history=[]
  for variant in [0]:
   resultpath=run/'classification'/f'dev-variant{variant}.json'
   if resultpath.exists():result=read(resultpath)
   else:
    texts=[]
    for i,k in enumerate(order):
     texts.append(client.request(k,variant,prompt(byid[k],variant)))
     atomic(run/'heartbeat.json',{'phase':'cityfm_dev','variant':variant,'completed':i+1,'updated_epoch':time.time()})
     if i==7:atomic(run/'classification'/f'smoke8-variant{variant}.json',{'ids':order[:8],'scores':score([labels[x] for x in order[:8]],texts)})
    result={'variant':variant,'ids':order,'outputs':texts,'scores':score([labels[x] for x in order],texts)};atomic(resultpath,result)
   history.append(result)
   s=result['scores']
   if s['accuracy']>=.65 and s['macro_f1']>=.60 and s['valid_rate']>=.95:break
   errors=[{'id':k,'gold':labels[k],'prediction':parse_label(text),'failure_type':'INVALID_JSON_CLASS' if parse_label(text) is None else 'MISCLASSIFICATION'} for k,text in zip(result['ids'],result['outputs']) if parse_label(text)!=labels[k]]
   atomic(run/'classification/dev-error-evidence.json',{'dev_scores':s,'errors':errors,'adaptation_performed':False,'reason':'Single frozen prompt0; low dev accuracy does not authorize a new prompt or correctness retry'})
  best=max(history,key=lambda r:(r['scores']['macro_f1'],r['scores']['accuracy'],-r['variant']))
  freeze={'variant':best['variant'],'system':prompt({},best['variant'])['system'],'selection':'Single fixed prompt0; no automatic adaptation. MacroF1/accuracy reported on dev only.','dev_result_sha256':sha(run/'classification'/f"dev-variant{best['variant']}.json"),'input_contract_sha256':sha(run/'preparation/cityfm-contract-v2.json')}
  f=run/'classification/prompt-input-freeze.json'
  if f.exists() and read(f)!=freeze:raise RuntimeError('Screen prompt already frozen differently')
  atomic(f,freeze)
  screenpath=run/'classification/screen96.json'
  if screenpath.exists():return read(screenpath)
  client=OfficialClient(run,'cityfm_screen');inputs=loadlines(prep/ROLES[2]/'inputs.jsonl');gold=loadlines(prep/ROLES[2]/'gold.local.jsonl');labels={r['id']:r['class'] for r in gold};texts=[]
  if len(inputs)!=96:raise RuntimeError('Screen denominator must96')
  for i,r in enumerate(inputs):
   texts.append(client.request(r['id'],best['variant'],prompt(r['payload'],best['variant'])))
   atomic(run/'heartbeat.json',{'phase':'cityfm_screen','completed':i+1,'updated_epoch':time.time()})
  scored=score([labels[r['id']] for r in inputs],texts);baseline=read(prep/'baseline.local.json')['balanced_screen96'];passed=qualification_gate(scored,baseline['majority']['accuracy'],baseline['logistic']['accuracy'],audit['post_exclusion_shared_entity_conflicts']==0 and audit['label_mismatch_count']==0)
  result={'status':'QUALIFIED_PUBLIC_PROXY_DIAGNOSTIC' if passed else 'STOP_TEACHER_QUALIFICATION','ids':[r['id'] for r in inputs],'outputs':texts,'scores':scored,'gate_pass':passed,'freeze_sha256':sha(f),'no_sealed_use':True,'no_privacy_or_generator_acceptance':True};atomic(screenpath,result);return result

def qwen_baseline(run=RUN):
 """The original cached 2B, same frozen facts, native template, off/greedy64."""
 run=Path(run);root_authorization(run);supervised_worker_verified(run)
 for path,expected in read(run/'baseline-model.json').items():
  if sha(path)!=expected:raise StopPhase('Original model changed after freeze')
 result=read(run/'classification/screen96.json')
 if result.get('gate_pass') is not True:raise PermissionError('Qwen baseline conditional on qualified teacher')
 freeze=read(run/'classification/prompt-input-freeze.json');prep=run/'preparation/cityfm'
 import torch,gc
 from transformers import AutoTokenizer,AutoModelForImageTextToText
 import ipd_native_migration_20261019 as migration
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
 tok=AutoTokenizer.from_pretrained(migration.BASE,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(migration.BASE,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa');model.eval()
 for p in model.parameters():p.requires_grad_(False)
 folder=run/'classification/qwen-original';inputs=loadlines(prep/ROLES[2]/'inputs.jsonl');gold={r['id']:r['class'] for r in loadlines(prep/ROLES[2]/'gold.local.jsonl')};texts=[];ooms=sum(read(p).get('error')=='CUDA_OOM' for p in folder.glob('*.json'))
 if ooms>=2:raise StopPhase('Two saved OOMs: Qwen phase stopped')
 stops=[tok.convert_tokens_to_ids(x) for x in ('<|im_end|>','<|endoftext|>')]
 for i,row in enumerate(inputs):
  request=prompt(row['payload'],freeze['variant']);path=folder/(row['id']+'.json');expected=digest(request)
  if path.exists():
   record=read(path)
   if record['request_sha256']!=expected:raise StopPhase('Baseline cached request changed')
  else:
   rendered=tok.apply_chat_template([{'role':'system','content':request['system']},{'role':'user','content':request['user']}],add_generation_prompt=True,enable_thinking=False,tokenize=False)
   ids=tok(rendered,return_tensors='pt',add_special_tokens=False).to('cuda');length=ids['input_ids'].shape[1]
   try:
    with torch.inference_mode():seq=model.generate(**ids,max_new_tokens=64,do_sample=False,eos_token_id=stops,pad_token_id=tok.pad_token_id)
    out=seq[0,length:].tolist();text=tok.decode(out,skip_special_tokens=True)
    record={'id':row['id'],'request_sha256':expected,'content':text,'output_tokens':len(out),'finish_reason':'stop' if out and out[-1] in stops else 'length'}
    del seq
   except torch.OutOfMemoryError:
    ooms+=1;record={'id':row['id'],'request_sha256':expected,'content':'','finish_reason':'technical_failure','error':'CUDA_OOM'}
    gc.collect();torch.cuda.empty_cache()
   finally:del ids
   atomic(path,record)
  texts.append(record['content']);atomic(run/'heartbeat.json',{'phase':'qwen_original','completed':i+1,'updated_epoch':time.time()})
  if ooms>=2:raise StopPhase('Two OOMs: Qwen phase stopped')
 scored=score([gold[r['id']] for r in inputs],texts)
 report={'status':'COMPLETE','n':96,'scores':scored,'teacher_scores':result['scores'],'accuracy_gap_teacher_minus_qwen':result['scores']['accuracy']-scored['accuracy'],'model':'original cached Qwen3.5-2B native vocabulary','thinking':'disabled','decoding':'greedy','max_new_tokens':64,'same_facts_prompt_freeze_sha256':sha(run/'classification/prompt-input-freeze.json'),'sealed_used':False,'trained':False}
 atomic(run/'classification/qwen-original-summary.json',report);return report

def start_independent_branch(function,results,key):
 """One branch failure is recorded and cannot cancel the independent branch."""
 done=threading.Event()
 def target():
  try:results[key]=function()
  except BaseException as error:results[key]={'status':'STOPPED','reason':str(error),'exception_type':type(error).__name__}
  finally:done.set()
 thread=threading.Thread(target=target,name=key,daemon=False);thread.start();return thread,done

def coordinate(run=RUN):
 run=Path(run);approval=root_authorization(run);limits=cgroup_limits_verified()
 supervisor=Supervisor(run,approval)
 signal.signal(signal.SIGTERM,lambda *_:setattr(supervisor,'stop',True))
 signal.signal(signal.SIGINT,lambda *_:setattr(supervisor,'stop',True))
 python=str(ROOT/'.venv/bin/python');script=str(Path(__file__).resolve())
 report={'status':'RUNNING','started_epoch':time.time(),'approved_at_epoch':approval['approved_at_epoch'],'cgroup_limits':limits,'stages':{},'api_concurrency_max':1}
 with lock(run/'locks/coordinator.lock'):
  city_thread=None;city_done=threading.Event();qwen_attempted=False
  def maybe_qwen():
   nonlocal qwen_attempted
   result=report['stages'].get('cityfm',{})
   if qwen_attempted or not city_done.is_set() or result.get('gate_pass') is not True:return
   qwen_attempted=True
   try:
    with lock(run/'locks/gpu.lock'):
     report['stages']['qwen']=supervisor.stage('qwen-original',[python,script,'qwen-baseline','--run',str(run)],[run/'classification/qwen-original'],gpu=True,completion=run/'classification/qwen-original-summary.json',recoverable=True)
   except StopPhase as error:
    report['stages']['qwen']={'status':'STOPPED','reason':str(error)}
    if 'Resource floor breached' in str(error):supervisor.stop=True
  try:
   supervisor.wait_resources(startup=True)
   if read(Path(active_runtime(run)['package_dir'])/'cityfm_dev.json')['status']=='FROZEN_AWAITING_CONFIRMATION':
    require_approval(run,'cityfm_dev');require_approval(run,'cityfm_screen')
    city_thread,city_done=start_independent_branch(lambda:supervisor.stage('cityfm',[python,script,'classify','--run',str(run)],[run/'classification',run/'requests'],completion=run/'classification/screen96.json',recoverable=True),report['stages'],'cityfm')
   else:city_done.set()
   gad=run/'preparation/bird';gadscript=str(ROOT/'scripts/bird_gad_night_20261022.py')
   if not (gad/'launch-authorization.json').exists():raise PermissionError('BIRD launch authorization missing')
   stages=[('bird-preflight',['preflight'],gad/'gpu-preflight.json',True),
           ('bird-gad',['train','--trial','gad'],gad/'trials/gad/training-complete.json',True),
           ('bird-seqkd',['train','--trial','seqkd'],gad/'trials/seqkd/training-complete.json',True),
           ('bird-select',['select'],gad/'selected-models.json',False),
           ('bird-reload-gad',['reload','--trial','gad'],gad/'gad-selected-reload.json',True),
           ('bird-reload-seqkd',['reload','--trial','seqkd'],gad/'seqkd-selected-reload.json',True),
           ('bird-freeze',['freeze'],gad/'models-frozen.json',False)]
   stages += [('bird-final-'+arm,['final-eval','--eval-arm',arm],gad/'final-evaluation'/arm/'summary.json',True) for arm in ('N-0','N-R','gad','seqkd')]
   for name,args,completion,gpu in stages:
    if name!='bird-preflight':maybe_qwen()
    with lock(run/'locks/gpu.lock') if gpu else contextlib.nullcontext():
     trial=args[-1] if args[0]=='train' else None
     def recovery_ok():
      marker=gad/'trials'/trial/'interruption-checkpoint.json'
      return marker.exists() and (gad/'gpu-preflight.json').exists() and read(gad/'gpu-preflight.json').get('resume_verified') is True
     def priority_ready():
      count=read(run/'recovery-count.json')['count'] if (run/'recovery-count.json').exists() else 0
      return city_done.is_set() and report['stages'].get('cityfm',{}).get('gate_pass') is True and not qwen_attempted and count==0 and read(gad/'gpu-preflight.json').get('resume_verified') is True and not (gad/'trials'/trial/'resume-used.json').exists()
     command=[python,gadscript,*args]
     result=supervisor.stage(name,command,[gad/'trials',gad/'final-evaluation',gad/'gpu-preflight.json',gad/'worker-heartbeat.json'],gpu=gpu,completion=completion,recoverable=bool(trial),recovery_check=recovery_ok if trial else None,yield_check=priority_ready if trial else None)
    if result.get('status')=='YIELDED_GPU':
     maybe_qwen()
     with lock(run/'locks/gpu.lock'):
      result=supervisor.stage(name,[*command,'--resume'],[gad/'trials',gad/'worker-heartbeat.json'],gpu=True,completion=completion,recoverable=False,recovery_check=recovery_ok)
    if args[0]=='final-eval' and result.get('gold_technical_failures',0):
     report['stages'][name]={'status':'BLOCKED_SCORER','gold_technical_failures':result['gold_technical_failures'],'model_accuracy':'SUPPRESSED','summary_path':str(completion)}
     raise StopPhase('Final scorer has gold technical failures; model-accuracy interpretation suppressed')
    report['stages'][name]=result
   report['status']='BIRD_STAGES_COMPLETE'
  except Exception as error:
   report.update(status='STOPPED_SAFELY',reason=str(error))
   if 'Resource floor breached' in str(error) or time.time()>=supervisor.deadline:supervisor.stop=True
  finally:
   # API liveness/failure never holds up BIRD. Once BIRD is finished/stopped,
   # drain the bounded API branch and honor any qualified Qwen request.
   while city_thread and city_thread.is_alive():
    if time.time()>=supervisor.deadline:supervisor.stop=True
    city_thread.join(timeout=.25)
   maybe_qwen()
   if report['status']=='BIRD_STAGES_COMPLETE':report['status']='PARTIAL_BRANCH_FAILURE' if report['stages'].get('cityfm',{}).get('status')=='STOPPED' else 'COMPLETED_WITH_EXPLICIT_LIMITATIONS'
   report['finished_epoch']=time.time();report['new_budget']=ledger(run).summary();atomic(run/'night-summary.json',report);supervisor.heartbeat(report['status'],reason=report.get('reason'))
 return report

def launch(run=RUN):
 """Start one bounded user service. No sudo, daemon/global config or linger changes."""
 run=Path(run);root_authorization(run);require_approval(run,'cityfm_dev');require_approval(run,'cityfm_screen')
 gad=run/'preparation/bird';authorization=read(gad/'launch-authorization.json')
 active=read(gad/'preparation-active.json')
 if authorization.get('authorized') is not True or authorization.get('protocol_sha256')!=sha(active['protocol_path']) or authorization.get('integrity_sha256')!=sha(active['integrity_path']):raise PermissionError('BIRD local launch authorization missing or mismatched')
 if (run/'launch.json').exists():raise StopPhase('Launch record already exists; inspect owned service before a recovery')
 Supervisor(run,read(run/'root-goal-approval.json')).wait_resources(startup=True)
 unit='privacy-night-dual-20261022'
 command=['systemd-run','--user','--unit='+unit,'--collect','--property=MemoryMax=14G','--property=CPUQuota=400%','--property=TasksMax=128','--property=WorkingDirectory='+str(ROOT),'--property=StandardOutput=append:'+str(run/'service.log'),'--property=StandardError=append:'+str(run/'service.log'),'--setenv=PATH=/home/kiryu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib',str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'coordinate','--run',str(run)]
 subprocess.run(command,check=True,capture_output=True,text=True,timeout=15)
 record={'unit':unit,'command':command,'launched_epoch':time.time(),'deadline_epoch':read(run/'root-goal-approval.json')['approved_at_epoch']+12*3600,'persistence':'systemd user service; WSL distribution and Windows must remain running; seconds-only detached proof does not establish12h persistence'}
 atomic(run/'launch.json',record)
 record['windows_keeper']=hidden_windows_worker(['keeper','--run',str(run)])
 atomic(run/'launch.json',record);return record

def hidden_windows_worker(arguments):
 """An owned, hidden foreground wsl.exe keeps the distro referenced by Windows."""
 values=['-d','Ubuntu-24.04','--cd',str(ROOT),'--exec',str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),*map(str,arguments)]
 quote=lambda value:"'"+value.replace("'","''")+"'"
 script="$p=Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList @("+','.join(map(quote,values))+") -PassThru; @{pid=$p.Id;started=$p.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json -Compress"
 proc=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,check=True,timeout=15)
 return json.loads(proc.stdout.strip().lstrip('\ufeff'))

def keeper(run):
 run=Path(run);approval=root_authorization(run);deadline=float(approval['approved_at_epoch'])+12*3600
 while time.time()<deadline:
  summary=run/'night-summary.json'
  if summary.exists() and read(summary)['status']!='RUNNING':return
  atomic(run/'keeper-heartbeat.json',{'pid':os.getpid(),'starttime':process_start(os.getpid()),'updated_epoch':time.time(),'deadline_epoch':deadline})
  time.sleep(min(30,max(0,deadline-time.time())))

def windows_keeper_proof(folder):
 folder=Path(folder)
 if folder.exists():raise RuntimeError('Use a new evidence directory')
 folder.mkdir(parents=True);identity=hidden_windows_worker(['mock-worker','--run',str(folder)])
 deadline=time.time()+30
 while time.time()<deadline:
  p=folder/'state.json'
  if p.exists() and read(p)['status']=='COMPLETE':
   proof={'status':'PASS','windows_hidden_foreground_wsl':identity,'worker':read(p),'external_calls':0,'gpu_operations':0,'seconds_test_only':True,'twelve_hour_test_performed':False};atomic(folder/'proof.json',proof);return proof
  time.sleep(.1)
 raise RuntimeError('Hidden Windows WSL keeper proof timed out; inspect owned worker before retry')

def resource_scope_proof(folder):
 folder=Path(folder).resolve()
 if folder.exists():raise RuntimeError('Use a new scope evidence directory')
 folder.mkdir(parents=True);unit='privacy-night-scope-proof-'+uuid.uuid4().hex[:12]
 command=['systemd-run','--user','--unit='+unit,'--collect','--property=MemoryMax=14G','--property=CPUQuota=400%','--property=TasksMax=128','--property=StandardOutput=append:'+str(folder/'service.log'),'--property=StandardError=append:'+str(folder/'service.log'),str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'scope-worker','--run',str(folder)]
 subprocess.run(command,check=True,capture_output=True,text=True,timeout=15)
 deadline=time.time()+30
 while time.time()<deadline:
  p=folder/'scope-proof.json'
  if p.exists():return read(p)
  time.sleep(.1)
 raise RuntimeError('Resource-scope proof did not complete; inspect owned unit '+unit)

def resource_scope_worker(folder):
 folder=Path(folder);limits=cgroup_limits_verified();mock_worker(folder)
 atomic(folder/'scope-proof.json',{'status':'PASS','effective_cgroup_limits':limits,'worker':read(folder/'state.json'),'external_calls':0,'gpu_operations':0,'twelve_hour_test_performed':False})

def mock_worker(folder,limit=6,interval=.15):
 folder=Path(folder);state=folder/'state.json';stop=[False]
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
 with lock(folder/'worker.lock'):
  old=read(state) if state.exists() else {'ticks':0,'starts':0}
  obj={'ticks':old['ticks'],'starts':old['starts']+1,'pid':os.getpid(),'starttime':process_start(os.getpid()),'status':'RUNNING','mock_only':True}
  atomic(folder/'worker-identity.json',{'pid':obj['pid'],'starttime':obj['starttime']})
  while obj['ticks']<limit and not stop[0]:
   obj['ticks']+=1;obj['heartbeat_epoch']=time.time();atomic(state,obj);time.sleep(interval)
  obj['status']='INTERRUPTED' if stop[0] else 'COMPLETE';atomic(state,obj)

def safe_interrupt(identity):
 pid=int(identity['pid'])
 if not identity.get('starttime') or process_start(pid)!=identity['starttime']:raise RuntimeError('PID starttime changed; refuse unrelated process signal')
 os.kill(pid,signal.SIGTERM)
def mock_proof(folder):
 folder=Path(folder)
 if folder.exists():raise RuntimeError('Mock evidence must use a new directory')
 folder.mkdir(parents=True);commands=[sys.executable,str(Path(__file__).resolve()),'mock-worker','--run',str(folder)]
 log=(folder/'worker.log').open('a')
 worker=subprocess.Popen(commands,stdout=log,stderr=log,start_new_session=True)
 deadline=time.time()+15
 while time.time()<deadline:
  if (folder/'state.json').exists() and read(folder/'state.json')['ticks']>=2:break
  if worker.poll() is not None:raise RuntimeError('Mock worker failed before heartbeat')
  time.sleep(.03)
 else:raise RuntimeError('Mock heartbeat timeout')
 lock_rejected=False
 try:
  with lock(folder/'worker.lock'):pass
 except RuntimeError:lock_rejected=True
 first=read(folder/'state.json');safe_interrupt(read(folder/'worker-identity.json'));worker.wait(timeout=5);interrupted=read(folder/'state.json')
 resume=subprocess.Popen(commands,stdout=log,stderr=log,start_new_session=True);resume.wait(timeout=10);last=read(folder/'state.json');log.close()
 proof={'status':'PASS' if lock_rejected and interrupted['status']=='INTERRUPTED' and last['status']=='COMPLETE' and last['starts']==2 and last['ticks']==6 else 'FAIL','lock_exclusivity':lock_rejected,'first':first,'after_safe_interrupt':interrupted,'after_one_recovery':last,'seconds_test_only':True,'external_calls':0,'gpu_operations':0,'twelve_hour_test_performed':False}
 atomic(folder/'proof.json',proof);return proof

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','classify','qwen-baseline','coordinate','launch','keeper','host-check','mock-worker','mock-proof','windows-keeper-proof','resource-scope-proof','scope-worker']);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--revision',type=int,default=1);args=p.parse_args()
 if args.command=='prepare':result=prepare(args.run,args.revision)
 elif args.command=='classify':supervised_worker_verified(args.run);result=classify(args.run)
 elif args.command=='qwen-baseline':result=qwen_baseline(args.run)
 elif args.command=='coordinate':result=coordinate(args.run)
 elif args.command=='launch':result=launch(args.run)
 elif args.command=='keeper':keeper(args.run);return
 elif args.command=='host-check':result=host_snapshot(args.run)
 elif args.command=='mock-proof':result=mock_proof(args.run)
 elif args.command=='windows-keeper-proof':result=windows_keeper_proof(args.run)
 elif args.command=='resource-scope-proof':result=resource_scope_proof(args.run)
 elif args.command=='scope-worker':resource_scope_worker(args.run);return
 else:mock_worker(args.run);return
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
