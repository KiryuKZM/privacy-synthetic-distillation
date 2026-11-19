"""Matched W-start Qwen student-rollout KL/IPD and local-oracle CE controls.
Reuses frozen loader, checkpoint, resource guards, schedule loop and reload.
"""
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import argparse,ast,copy,fcntl,hashlib,json,os,resource,signal,subprocess,sys,time,types
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
CUDA_BEFORE=os.environ.get('CUDA_VISIBLE_DEVICES')
import cloud_qwen_synthetic_ipd_continuation_control_v2_20261101 as parent
import three_method_native_ipd_v5_20261101 as position_fix
import cloud_qwen_matched_objective_budget_20261102 as money
if CUDA_BEFORE is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=CUDA_BEFORE
assert parent.e is position_fix.e
h=parent.h;b=h.b;e=parent.e;m=parent.m;api=money.api
N=R/'runs/2026-11-02-matched-qwen-objectives-v1'
METHODS=('opd','ipd','oracle-ce');CURRENT=None;P=None;STOP=False
MODEL='qwen3.8-max';SEMANTIC=248077;NATIVE=248320
original_load=m.load;bind=m.bind;once=m.once;atomic=m.atomic

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(method):
 assert method in METHODS
 return N/'students'/method

def stopped():return STOP or (N/'STOP').exists() or (R/'runs/STOP').exists() or (CURRENT is not None and (CURRENT/'STOP').exists())
def check():
 if stopped():raise InterruptedError('Own matched objective STOP; no old namespace coupling')
def stop(*_):
 global STOP
 STOP=True

def load(p):
 p=Path(p)
 value=original_load(p)
 if p.resolve()==(N/'inputs/fifty-twice100.json').resolve():
  assert [v['slot'] for v in value]==list(range(1,101))
  return [v['row'] for v in value]
 return value

def verify(out):
 out=Path(out).resolve();p=original_load(out/'protocol.json')
 assert out==unit(p['method']).resolve() and p['unit_path']==str(out)
 g=original_load(N/'root-grant.json')
 assert g['approved'] and g['source']==bind(Path(__file__)) and bind(out/'protocol.json') in g['unit_protocols']
 for pin in g['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 assert p['lr']==1e-4 and p['maximum_optimizer_attempts']==p['scheduled_slots']==100
 assert p['lora']==dict(r=8,alpha=16,dropout=0) and p['native_vocabulary'] and not p['thinking']
 assert p['max_positions']==32 and p['support_K']==5 and p['fresh_AdamW'] and not p['restore_W_optimizer']
 assert p['inputs']==bind(R/p['inputs']['path']) and p['W_manifest']==bind(R/p['W_manifest']['path'])
 for name,info in p['base_asset_stat'].items():
  st=(Path(p['base'])/name).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==tuple(info)
 assert [v['id'] for v in load(R/p['inputs']['path'])]==p['schedule_ids']
 p.update(prior_slots=0,schedule_start=1,prior_API_calls=0,prior_observed_closed_positions=0,prior_feedback_counts={},prior_events={},prior_sources={},prior_interrupted_selected_positions=0,post50_ids=p['schedule_ids'])
 return p

def phase_bytes():
 return sum(p.stat().st_size for p in N.rglob('*') if p.is_file() and not p.is_symlink())

def setup(out,checkpoint):
 f=m.core.seeded_setup(m.native,verify);f.__globals__['formal_setup_probe']=m.setup_probe
 return f(out,checkpoint)
def answer_event_ids(tok):
 values=[tok.encode('>'+v,add_special_tokens=False) for v in 'ABCDE']
 assert all(len(v)==1 for v in values),'Actual fused answer event interface'
 return [v[0] for v in values]

def initialize(out,checkpoint):
 torch,tok,model,head=setup(out,checkpoint);torch.set_num_threads(4)
 assert len(tok)==SEMANTIC
 return torch,tok,model,head

@dataclass(frozen=True)
class StudentOnlyDecision:
 token_id:int
 source:str
 next_driver:str
 event:str

def choose(method,driver,student,teacher,student_top,teacher_top):
 if method=='opd':
  assert driver=='S' and type(student) is int and 0<=student<SEMANTIC
  return StudentOnlyDecision(student,'S','S','opd_student_rollout_no_intervention')
 assert method=='ipd'
 return b.decide_token(driver,student,teacher,student_top,teacher_top)

def selected_event(choice,mapped,teacher,tok,lookup):
 found=next((x for x in mapped if x['id']==choice.token_id),None)
 if found is not None:return found
 if teacher['id']==choice.token_id:return teacher
 assert choice.source=='S' and choice.event=='opd_student_rollout_no_intervention','Unknown chosen event must not be replaced'
 raws=[raw for raw,ids in lookup.items() if ids==[choice.token_id]]
 assert len(raws)==1,'Unqualified actual student byte event'
 raw=raws[0]
 if choice.token_id in set(tok.all_special_ids):
  label=tok.convert_ids_to_tokens(choice.token_id)
  assert label in ('<|im_end|>','<|endoftext|>') and raw==label.encode(),'Unsupported actual student control'
  kind='TURN_END' if label=='<|im_end|>' else 'SEQUENCE_END'
 else:
  assert raw.decode('utf-8',errors='strict'),'Standalone nonempty UTF8 required'
  label=None;kind='BYTES'
 return dict(id=choice.token_id,kind=kind,bytes=list(raw),provider_label=label,origin='actual_student_native_proposal_not_teacher_replacement')

def budget_for(method):return None if method=='oracle-ce' else money.ObjectiveBudget(api.LEDGER)

def compile_parent(name,ctx,change):
 source=Path(parent.__file__).read_text();fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
 fn=change(copy.deepcopy(fn));ns=dict(parent.__dict__);ns.update(ctx)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(Path(__file__))+'::frozen_parent_'+name,'exec'),ns)
 return ns[name]

AST_PROOF={}
def adapt_call(fn):
 n=0;tag=0
 for x in ast.walk(fn):
  if isinstance(x,ast.Constant) and type(x.value) is int and x.value==50:x.value=100;n+=1
  if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='tag' for y in x.targets):
   x.value=ast.parse('f"matched-v1/{p[\"method\"]}/slot{step:03d}/pos{index:02d}"',mode='eval').body;tag+=1
 assert n==tag==1,(n,tag);AST_PROOF['call_feedback']=dict(slot50to100=n,exact_new_purpose_tag=tag,transport_settle_error_body_unchanged=True)
 return fn

def adapt_trajectory(fn):
 changes=Counter()
 for x in ast.walk(fn):
  if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='b' and x.func.attr=='decide_token':
   x.func=ast.Name(id='choose',ctx=ast.Load());x.args.insert(0,ast.parse('p["method"]',mode='eval').body);changes['decision']+=1
  if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='selected' for y in x.targets):
   x.value=ast.parse('selected_event(choice,mapped,teacher,tok,lookup)',mode='eval').body;changes['selected_event']+=1
 assert changes==dict(decision=1,selected_event=1),changes
 AST_PROOF['trajectory']=dict(changes,proposal_RNG_teacher_K5_source_and_advance_unchanged=True)
 return fn

def adapt_fit(fn):
 n=0;budget=0;truth=0;flag=0
 for x in ast.walk(fn):
  if isinstance(x,ast.Constant) and type(x.value) is int and x.value==50:x.value=100;n+=1
  if isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value=='ORIGINAL_CLOUD50_ACCOUNTED_WITH_PRESERVED_PRIOR_SLOTS_RELOAD_PENDING':x.value='MATCHED100_ACCOUNTED_RELOAD_PENDING'
  if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='money' and x.func.attr=='MainlineBudget':
   x.func=ast.Name(id='budget_for',ctx=ast.Load());x.args=[ast.parse('p["method"]',mode='eval').body];budget+=1
  if isinstance(x,ast.keyword) and x.arg=='no_Test_or_gold_or_oracle_in_loss':x.value=ast.parse('p["method"]!="oracle-ce"',mode='eval').body;truth+=1
  if isinstance(x,ast.keyword) and x.arg=='has_actual_IPD_update':x.arg='has_actual_update';flag+=1
 assert n==5 and budget==truth==flag==1,(n,budget,truth,flag)
 AST_PROOF['fit']=dict(slot50to100=n,budget_constructor=budget,oracle_claim_corrected=truth,generic_update_flag=flag,optimizer_RNG_freezes_save_loop_unchanged=True)
 return fn

def adapt_reload(fn):
 n=0
 for x in ast.walk(fn):
  if isinstance(x,ast.Constant) and type(x.value) is int and x.value==50:x.value=100;n+=1
 assert n==2,n;AST_PROOF['reload']=dict(final50to100=n,exact_roles_optimizer_RNG_checks_unchanged=True)
 return fn

ANSWER_IDS=[]
def loss_diagnostics(torch,logits,union,q,mask,sources,per,reverse,forward,trace):
 with torch.no_grad():
  x=logits.float();z=x[:,:SEMANTIC].logsumexp(-1);allz=x.logsumexp(-1)
  support=x.gather(-1,union).masked_fill(~mask,-torch.inf)
  return dict(student_semantic_logZ=z.cpu().tolist(),student_teacherK5_mass=(support.logsumexp(-1)-z).exp().cpu().tolist(),teacher_returnedK5_mass=q.masked_fill(~mask,-torch.inf).logsumexp(-1).exp().cpu().tolist(),student_fixed_fused_gtA_gtE_mass=(x[:,ANSWER_IDS].logsumexp(-1)-z).exp().cpu().tolist(),fixed_answer_domain='>A..>E single native events; interpret only at compatible answer prefix, not general A-E validity mass',padded_native_tail_mass=(x[:,SEMANTIC:].logsumexp(-1)-allz).exp().cpu().tolist(),per_position_actual_loss=per.detach().cpu().tolist(),per_position_reverse=reverse.detach().cpu().tolist(),per_position_forward=forward.detach().cpu().tolist(),sources_S=sources.cpu().tolist(),support_ids=union.cpu().tolist(),gradient_scope='actual pre-parameter-clip live common-support logits; not LoRA parameter attribution',parameter_answer_format_VJP_measured=False,quality_penalty_added=False)

def instrument_KL_update():
 source=(R/'scripts/three_method_native_ipd_20261101.py').read_text();fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='update');changes=Counter();body=[]
 for stmt in fn.body:
  body.append(stmt)
  if isinstance(stmt,ast.Assign) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Name) and stmt.value.func.id=='source_split_kl':
   body+=ast.parse('live.retain_grad()\nobjective_diagnostics=loss_diagnostics(torch,logits,union,q,mask,sources,per,reverse,forward,trace)').body;changes['same_forward_diagnostics']+=1
  if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute) and isinstance(stmt.value.func.value,ast.Name) and stmt.value.func.value.id=='loss' and stmt.value.func.attr=='backward':
   body+=ast.parse('objective_diagnostics["live_logit_gradients"]=live.grad.detach().cpu().tolist()').body;changes['same_backward_gradient']+=1
  if isinstance(stmt,ast.Return):
   assert isinstance(stmt.value,ast.Call)
   stmt.value.keywords += [ast.keyword(arg='objective_diagnostics',value=ast.Name(id='objective_diagnostics',ctx=ast.Load())),ast.keyword(arg='method',value=ast.parse('trace["method"]',mode='eval').body)];changes['return_metrics']+=1
 assert changes==dict(same_forward_diagnostics=1,same_backward_gradient=1,return_metrics=1),changes
 fn.body=body;ns=dict(e.__dict__);ns.update(loss_diagnostics=loss_diagnostics,check_stop=check)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(Path(__file__))+'::same_KL_update_plus_passive_metrics','exec'),ns)
 AST_PROOF['KL_update']=dict(changes,original_loss_backward_clip_optimizer_unchanged=True)
 return ns['update']

KL_UPDATE=instrument_KL_update()

def oracle_trajectory(torch,tok,model,p,row,step,actor,budget,out,lookup):
 labels=original_load(R/p['local_oracle']['path']);letter=labels[row['id']];assert letter in 'ABCDE'
 target=tok.encode('<final>'+letter+'</final>',add_special_tokens=False)
 assert len(target)==6 and all(0<=v<SEMANTIC for v in target),'Six actual canonical text positions, no invented EOS'
 return dict(id=row['id'],method='oracle-ce',prompt_ids=row['student_prompt_ids'],actor_before=actor,oracle_target_ids=target,tokens=[dict(position=j,token_id=v,source='ORACLE') for j,v in enumerate(target)],events=dict(oracle_teacher_forcing=len(target)),sources=dict(ORACLE=len(target)),feedback_status='complete',actual_API_calls=0,API_seconds=0.,observed_positions=len(target),seconds=0.,protected_real_gold_in_loss=False,local_synthetic_oracle_in_loss=True)

def oracle_update(torch,model,head,opt,trace):
 m.lib.activate(model,head,'actor');model.train();opt.zero_grad(set_to_none=True)
 params=[p for p in model.parameters() if p.requires_grad];before=m.lib.role_hash(model,'actor');assert before==trace['actor_before']
 target=trace['oracle_target_ids'];n=len(target);ids=torch.tensor([trace['prompt_ids']+target[:-1]],device='cuda');tick=time.monotonic()
 logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n).logits[0].float();assert logits.shape==(n,NATIVE)
 loss=torch.nn.functional.cross_entropy(logits[:,:SEMANTIC],torch.tensor(target,device='cuda'))
 if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite actual fullsemantic oracleCE')
 loss.backward()
 if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):raise RuntimeError('Nonfinite oracle gradient')
 norm=torch.nn.utils.clip_grad_norm_(params,1.)
 if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite oracle gradient norm')
 check();changed=float(norm)>0
 if changed:opt.step()
 opt.zero_grad(set_to_none=True);after=m.lib.role_hash(model,'actor')
 if changed and before==after:raise RuntimeError('Oracle gradient without actual actor change')
 return dict(method='oracle-ce',optimizer_updated=changed,actor_before=before,actor_after=after,loss=float(loss.detach()),gradient_norm=float(norm),input_tokens=len(trace['prompt_ids']),target_tokens=n,sources=trace['sources'],events=trace['events'],update_seconds=time.monotonic()-tick,oracle_or_real_gold_in_loss=True,protected_real_gold_in_loss=False,local_synthetic_oracle_in_loss=True,teacher_forced_fullsemantic_not_matched_Qwen_loss=True)

def update(torch,model,head,opt,trace):
 return oracle_update(torch,model,head,opt,trace) if trace['method']=='oracle-ce' else KL_UPDATE(torch,model,head,opt,trace)

_CONTEXT=dict(P=None,MAIN=N,load=load,verify=verify,make_body=parent.make_body,money=money,choose=choose,selected_event=selected_event,budget_for=budget_for)
CALL_FEEDBACK=compile_parent('call_feedback',_CONTEXT,adapt_call)
_CONTEXT.update(call_feedback=CALL_FEEDBACK)
CLOUD_TRAJECTORY=compile_parent('trajectory',_CONTEXT,adapt_trajectory)

def trajectory(torch,tok,model,p,row,step,actor,budget,out,lookup):
 if p['method']=='oracle-ce':return oracle_trajectory(torch,tok,model,p,row,step,actor,budget,out,lookup)
 # Same fresh text cache state per route, restoring only owned ephemeral state.
 owners=[(x,getattr(x,'rope_deltas')) for x in model.modules() if hasattr(x,'rope_deltas')]
 try:
  for x,_ in owners:x.rope_deltas=None
  v=CLOUD_TRAJECTORY(torch,tok,model,p,row,step,actor,budget,out,lookup);v['method']=p['method'];return v
 finally:
  for x,old in owners:x.rope_deltas=old

_CONTEXT.update(trajectory=trajectory)
FIT=compile_parent('fit',_CONTEXT,adapt_fit)
RELOAD=compile_parent('reload',_CONTEXT,adapt_reload)

def configure(method):
 global CURRENT,P,ANSWER_IDS
 CURRENT=unit(method);P=N/'runtime'/method;P.mkdir(parents=True,exist_ok=True)
 for mod in (m,e,b,h):mod.P=P
 m.O=N;m.S=N;m.verify=verify;m.stopped=stopped;m.check_stop=check;m.phase_bytes=phase_bytes
 b.check=check;b.setup=setup;b.initialize=initialize;e.check_stop=check;e.update=update
 e.base_snapshot=e.clone(m.snapshot,P=P,check_stop=check,O=N,phase_bytes=phase_bytes)
 for fn in [CALL_FEEDBACK,CLOUD_TRAJECTORY,FIT,RELOAD]:fn.__globals__.update(P=P,verify=verify,load=load)
 # The snapshot of KL globals must use current pure guards, not prior STOP aliases.
 KL_UPDATE.__globals__.update(check_stop=check,loss_diagnostics=loss_diagnostics)
 original_initialize=initialize
 def with_answers(out,checkpoint):
  global ANSWER_IDS
  torch,tok,model,head=original_initialize(out,checkpoint)
  ANSWER_IDS=answer_event_ids(tok)
  return torch,tok,model,head
 b.initialize=with_answers
 assert e.StudentStepper is position_fix.e.StudentStepper
 assert e.base_snapshot.__globals__['P']==P and m.S==N
 return CURRENT

def action(method,mode):
 out=configure(method);p=verify(out)
 with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  once(out/(mode+'-owner.json'),dict(PID=os.getpid(),UID=os.getuid(),invocation=os.environ.get('INVOCATION_ID'),source=bind(Path(__file__)),protocol=bind(out/'protocol.json'),epoch=time.time(),method=method,mode=mode))
  try:
   (FIT if mode=='fit' else RELOAD)(out)
  except BaseException as ex:
   once(out/(mode+'-failure.json'),dict(type=type(ex).__name__,message=str(ex)[:400],epoch=time.time(),PID=os.getpid(),no_automatic_replay=True));raise

def actual_owned_event(out):
 candidates=[]
 warm=out/'warm-source.json'
 if warm.exists():
  v=original_load(warm);p=original_load(out/'protocol.json')
  if v.get('manifest')==p['W_manifest'] and v.get('roles')==p['W_roles'] and v.get('fresh_optimizer_object') is True:
   candidates.append(dict(kind='actual_model_loaded_verified_W',epoch=warm.stat().st_mtime,artifact=bind(warm)))
 live=N/'runtime'/out.name/'live-event.json'
 if live.exists():
  v=original_load(live)
  if v.get('unit')==str(out.relative_to(R)) and v.get('kind') in ('actual_teacher_HTTP_response','actual_optimizer_or_nograd_task_commit') and type(v.get('epoch')) in (int,float):
   candidates.append(dict(kind=v['kind'],epoch=v['epoch'],artifact=bind(live)))
 return max(candidates,key=lambda x:x['epoch']) if candidates else None

class OwnedAwake:
 def __init__(self,folder):self.folder=folder;self.proc=None;self.last=None
 def observe(self,event,child):
  if event is None or time.time()-event['epoch']>600:return
  check();identity=(event['artifact']['sha256'],event['epoch'])
  if identity==self.last:return
  self.folder.mkdir(parents=True,exist_ok=True)
  proc_stat=Path(f'/proc/{child.pid}/stat').read_text();birth=proc_stat.rsplit(')',1)[1].split()[19]
  atomic(self.folder/'heartbeat.json',dict(event,owned_child_PID=child.pid,owned_child_startticks=birth,no_liveness_epoch_substitution=True))
  self.last=identity
  if self.proc is not None:return
  once(self.folder/'goal.json',dict(status='ACTIVE'))
  root=r'\\wsl.localhost\Ubuntu-24.04\home\kiryu\privacy-synthetic-distillation'
  def win(p):return root+'\\'+str(p.relative_to(R)).replace('/','\\')
  cmd=['/mnt/c/Users/10950/miniconda3/python.exe','-X','utf8',root+r'\scripts\keep_ust_sg_windows_awake_20261024.py','--heartbeat',win(self.folder/'heartbeat.json'),'--goal-state',win(self.folder/'goal.json'),'--stop-file',win(N/'STOP'),'--stop-file',win(R/'runs/STOP'),'--stop-file',win(self.folder/'STOP'),'--receipt',win(self.folder/'receipt.json'),'--poll','5','--stale','600']
  self.proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  once(self.folder/'actual-launch.json',dict(command=cmd,producer_child_PID=child.pid,producer_startticks=birth,source=bind(R/'scripts/keep_ust_sg_windows_awake_20261024.py'),triggering_real_event=event,epoch=time.time(),display_required=False,power_configuration_changed=False))
 def close(self):
  if self.proc is None:return
  atomic(self.folder/'goal.json',dict(status='COMPLETE'));atomic(self.folder/'STOP',dict(reason='Owned model child closed or stopped',epoch=time.time()))
  try:self.proc.wait(timeout=12)
  except subprocess.TimeoutExpired:once(self.folder/'cleanup-pending.json',dict(reason='STOP sent; verify actual native receipt/creation before any further action',no_bare_PID_kill=True,epoch=time.time()))

def queue():
 verify(unit('opd'))
 once(N/'queue-started.json',dict(PID=os.getpid(),epoch=time.time(),source=bind(Path(__file__)),methods=METHODS))
 for method in METHODS:
  out=unit(method)
  assert not (out/'fit-started.json').exists() and not (out/'reload-started.json').exists(),'Never re-execute begun fit/reload'
  for mode in ['fit','reload']:
   check();cmd=[sys.executable,str(Path(__file__)),mode,'--method',method]
   logpath=N/'children'/f'{method}-{mode}.stdout';logpath.parent.mkdir(exist_ok=True)
   awake=OwnedAwake(N/'awake'/f'{method}-{mode}');child=None
   try:
    with logpath.open('x') as f:
     child=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
     once(logpath.with_suffix('.started.json'),dict(PID=child.pid,command=cmd,epoch=time.time()))
     while child.poll() is None:
      check()
      if mode=='fit':awake.observe(actual_owned_event(out),child)
      try:child.wait(timeout=3)
      except subprocess.TimeoutExpired:pass
     code=child.returncode
    once(logpath.with_suffix('.complete.json'),dict(exitcode=code,stdout_closed=True,epoch=time.time()))
    if code:raise RuntimeError(f'Preserved failed {method}/{mode}; no automatic re-execution')
   finally:
    awake.close()
    if child is not None and child.poll() is None:
     child.terminate()
     try:child.wait(timeout=75)
     except subprocess.TimeoutExpired:
      child.kill();child.wait(timeout=15)
      once(logpath.with_suffix('.owned-stop-forced.json'),dict(owned_Popen_child_only=True,checkpoint_state_requires_diagnosis=True,epoch=time.time()))
 once(N/'training-terminal.json',dict(status='THREE_MATCHED100_FITS_AND_INDEPENDENT_RELOADS_COMPLETE',methods={method:dict(fit=bind(unit(method)/'fit-terminal.json'),reload=bind(unit(method)/'independent-reload.json')) for method in METHODS},epoch=time.time(),no_completed_old_model_retrained=True,whole_research_goal_complete=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['queue','fit','reload','cpu-identity']);a.add_argument('--method',choices=METHODS);v=a.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 if v.action=='cpu-identity':print(json.dumps(dict(status='CPU_IMPORTED_EFFECTIVE_INTERFACE',AST=AST_PROOF,explicit_cached_positions='position_ids' in e.StudentStepper.score.__code__.co_consts if False else e.StudentStepper.score.__code__.co_filename.endswith('::one-text-position-keyword'),API=0,model_loads=0),indent=2))
 elif v.action=='queue':queue()
 else:
  assert v.method
  action(v.method,v.action)
