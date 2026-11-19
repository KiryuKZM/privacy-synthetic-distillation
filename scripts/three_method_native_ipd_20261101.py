"""Latest native Qwen IPD: actual full-domain proposals, union logits, fixed50 updates.
Frozen W/CE/GRPO sources remain untouched. No external calls or evaluation gold.
"""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import argparse,copy,fcntl,gc,hashlib,json,math,os,resource,signal,sys,time,types
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import three_method_native_student_v2_20261101 as m
from three_method_native_teacher_client_20261101 import NativeTeacher,SEMANTIC,NATIVE
from privacy_synth.ipd import decide_token,source_split_kl
O=m.O;P=O/'latest-ipd-engine-v1';Q3=O/'backend-assets-v1/q3-conversion-v1'
BASELINE_FREE=None;STARTUP_READY=None;STOP=False;LAST_GUARD=0.;LAST_FRAME=None
BINARY=R/'artifacts/three-method-llama-native-ipd-bridge-20261101'
BACKEND=R/'artifacts/llama-cpp-b12020-cuda-13.4/combined'
MODEL=R/'artifacts/Qwen3.8-27B-Q3_K_M.gguf'
TEACHER_TEMPLATE_SHA='c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
load,sha,bind,once,atomic=m.load,m.sha,m.bind,m.once,m.atomic

def stop(*_):
 global STOP;STOP=True

def check_stop():
 if STOP or m.stopped():raise InterruptedError('Owned IPD STOP; no other app controls')

def clone(fn,**overrides):
 ns=dict(fn.__globals__);ns.update(overrides)
 return types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)

base_snapshot=clone(m.snapshot,P=P,check_stop=check_stop)
def snapshot(start=False):
 global LAST_FRAME
 s=base_snapshot(start);s['combined_GPU_baseline_free_MiB']=BASELINE_FREE
 if BASELINE_FREE is not None and not start:
  used=BASELINE_FREE-s['GPU']['free_MiB'];s['combined_GPU_delta_MiB']=used
  s['checks']['combined_GPU9GiB']=used<=9216
  atomic(P/'resource-latest.json',s)
  if not s['checks']['combined_GPU9GiB']:raise BlockingIOError('Combined student/native teacher exceeded CUDA9GiB target')
 LAST_FRAME=s;return s

def preflight():
 global STARTUP_READY,BASELINE_FREE
 start=time.monotonic();stable=None
 while True:
  check_stop();now=time.monotonic()
  try:s=snapshot(True);stable=now if stable is None else stable
  except BlockingIOError:stable=None
  if stable is not None and now-stable>=20:
   STARTUP_READY=time.monotonic();BASELINE_FREE=s['GPU']['free_MiB'];return s
  if now-start>=120:raise TimeoutError('Finite IPD quiet20/Win30/GPU11 qualification')
  time.sleep(.5)

def native_guard():
 global LAST_GUARD
 check_stop()
 if time.monotonic()-LAST_GUARD>=2:snapshot();LAST_GUARD=time.monotonic()

def cooldown():
 begin=time.monotonic();s=snapshot()
 if s['GPU']['temp_C']<74:return 0.
 while s['GPU']['temp_C']>68:
  check_stop()
  if time.monotonic()-begin>=120:raise TimeoutError('IPD cooling74to68 bounded120')
  time.sleep(.5);s=snapshot()
 return time.monotonic()-begin

def setup_probe(out,checkpoint):
 s=snapshot(True) if STARTUP_READY is not None and time.monotonic()-STARTUP_READY<4 else preflight()
 return dict(windows_free_gib=s['Windows']['windows_free_gib'],c_free_gib=s['Windows']['c_free_gib'],
  linux_free_gib=__import__('shutil').disk_usage(O).free/2**30,gpu_free_gib=s['GPU']['free_MiB']/1024)

def file_stat(p):
 p=Path(p);assert p.is_file() and not p.is_symlink()
 st=p.stat();return dict(bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns,device=st.st_dev)

def dependencies():
 a=O/'student-controller-v3/terminal.json';b=Q3/'terminal.json'
 assert load(a)['status']=='ALL18_W_OFF_ON_UNITS_FIT_RELOAD_COMPLETE_IPD_AND_COMMON_TEST_STILL_PENDING'
 q=load(b);assert q['status']=='DIRECT_BF16_Q3_K_M_ARTIFACT_SHA_AND_CODEC_VERIFIED_NOT_INFERENCE_QUALIFIED'
 return [bind(a),bind(b)]

def delegate():
 g=load(P/'root-delegation.json');assert g['approved'] and g['source']==bind(__file__)
 for x in g['bindings']:assert bind(R/x['path'])==x,x['path']
 assert g['maximum_IPD_fits']==6 and g['maximum_scheduled_updates']==300 and g['API_calls']==0
 check_stop();return g

def verify(out):
 out=Path(out).resolve();p=load(out/'protocol.json')
 assert p['stage']=='ipd' and out==m.unit(p['generator'],p['seed'],'ipd').resolve()
 assert p['supervision']=='actual_latest_Qwen_native_per_token_probabilities' and not p['oracle_or_real_gold_in_loss']
 assert p['maximum_optimizer_attempts']==50 and p['lr']==1e-4 and p['lora']==dict(r=8,alpha=16,dropout=0)
 assert p['native_vocabulary'] and not p['thinking'] and p['IPD']==IPD_CONFIG
 assert bind(R/p['inputs']['path'])==p['inputs'] and len(p['post50_ids'])==len(set(p['post50_ids']))==50
 for name,info in p['base_asset_stat'].items():
  st=(Path(p['base'])/name).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==tuple(info)
 assert file_stat(MODEL)==p['teacher_artifact_stat']
 for k in ('teacher_qualification','W_manifest'):assert bind(R/p[k]['path'])==p[k]
 return p

IPD_CONFIG=dict(topK=16,common_support='top16_student_union_top16_teacher_actual_scores',temperature=1.,gamma=1.,
 student_source_KL='reverse',teacher_source_KL='forward',maximum_new_tokens=32,
 student_proposal='full248077_categorical',teacher_proposal='full248077_categorical',thinking=False)

def setup(out,checkpoint=None,qualify=False):
 f=m.core.seeded_setup(m.native,m.verify if qualify else verify);f.__globals__['formal_setup_probe']=setup_probe
 torch,tok,model,head=f(out,checkpoint);torch.set_num_threads(2)
 return torch,tok,model,head

save=clone(m.save,verify=verify,check_stop=check_stop)

def authorize(out,action):
 delegate();p=verify(out);g=load(P/'root-grant.json')
 assert g['approved'] and g['source']==bind(__file__) and action in g['actions']
 assert str(out.relative_to(R)) in g['allowed_units']
 for b in g['bindings']:assert bind(R/b['path'])==b,b['path']
 assert dependencies()==g['actual_dependencies']
 return p

class StudentStepper:
 """One real current-actor cache; exactly one new token per subsequent call."""
 def __init__(self,torch,model,device="cuda"):self.torch=torch;self.model=model;self.device=device;self.cache=None;self.prefix=[];self.calls=[]
 def score(self,prefix):
  assert prefix and all(type(i) is int and 0<=i<SEMANTIC for i in prefix)
  assert not self.prefix or (len(prefix)==len(self.prefix)+1 and prefix[:-1]==self.prefix)
  if self.cache is not None:assert self.cache.get_seq_length()==len(self.prefix)
  ids=self.torch.tensor([prefix if self.cache is None else prefix[-1:]],device=self.device)
  attention=self.torch.ones((1,len(prefix)),device=self.device,dtype=self.torch.long)
  tick=time.monotonic()
  with self.torch.no_grad():result=self.model(input_ids=ids,attention_mask=attention,past_key_values=self.cache,use_cache=True,logits_to_keep=1)
  self.cache=result.past_key_values;assert self.cache is not None and self.cache.get_seq_length()==len(prefix)
  logits=result.logits[0,-1].float();assert logits.shape==(NATIVE,) and bool(self.torch.isfinite(logits).all())
  self.prefix=list(prefix);self.calls.append(dict(prefix=len(prefix),decoded_input_tokens=ids.shape[1],seconds=time.monotonic()-tick))
  return logits
 def close(self):self.cache=None;self.prefix=[]

def trajectory(torch,model,teacher,prefix,seed,actor_hash,max_tokens=32):
 assert 1<=max_tokens<=32 and len(prefix)+max_tokens<=16384
 rng=torch.Generator(device='cuda').manual_seed(seed);cur=list(prefix);driver='S';tokens=[];stepper=StudentStepper(torch,model)
 model.eval();begin=time.monotonic();call0=len(teacher.calls)
 try:
  for pos in range(max_tokens):
   native_guard();sv=stepper.score(cur);tv=teacher.score(cur)
   semantic=sv[:SEMANTIC];sk=semantic.topk(16).indices.tolist();tk=tv['top16_ids'];support=sorted(set(sk)|set(tk))
   sp=int(torch.multinomial(semantic.softmax(-1),1,generator=rng)) if driver=='S' else None
   uniform=float(torch.rand((),device='cuda',generator=rng)) if driver=='T' or sp not in tk else None
   proposal=teacher.sample(uniform) if uniform is not None else None;tp=proposal['token_id'] if proposal is not None else None
   choice=decide_token(driver,sp,tp,set(sk),set(tk));gather=teacher.gather(support)
   actual_top=dict(zip(tk,tv['top16_raw_logits']))
   assert all(v==actual_top[k] for k,v in zip(support,gather['raw_logits']) if k in actual_top)
   idx=torch.tensor(support,device='cuda');removed=float(sv.softmax(-1)[SEMANTIC:].sum())
   tokens.append(dict(position=pos,prefix_SHA256=hashlib.sha256(json.dumps(cur).encode()).hexdigest(),driver_before=driver,
    student_proposal=sp,teacher_proposal=tp,teacher_proposal_uniform=uniform,teacher_proposal_metadata=proposal,
    **asdict(choice),student_topk_ids=sk,teacher_topk_ids=tk,union_ids=support,student_union_logits=sv[idx].cpu().tolist(),
    teacher_union_logits=gather['raw_logits'],teacher_logZ_semantic=tv['logZ_semantic'],teacher_logZ_native=tv['logZ_native'],
    removed_student_unregistered_mass=removed,removed_teacher_unregistered_mass=tv['unused_mass_native'],
    teacher_new_decoded_tokens=tv['new_decoded_tokens'],teacher_decode_seconds=tv['decode_seconds'],student_actor=actor_hash))
   cur.append(choice.token_id);driver=choice.next_driver
   if choice.token_id in (248046,248044):break
 finally:stepper.close()
 assert tokens
 return dict(prompt_ids=prefix,tokens=tokens,seed=seed,actor_before=actor_hash,seconds=time.monotonic()-begin,
  student_cached_calls=stepper.calls,teacher_native_calls=teacher.calls[call0:],events=dict(Counter(x['event'] for x in tokens)),
  sources=dict(Counter(x['source'] for x in tokens)),finish_reason='stop' if tokens[-1]['token_id'] in (248046,248044) else 'length',
  actual_raw_teacher_feedback=True,teacher_final_text_not_used=True)

def support_tensors(torch,trace,device):
 tokens=trace['tokens'];n=len(tokens);assert 1<=n<=32 and len(trace['prompt_ids'])+n<=16384;assert all(x['position']==i and x['source'] in ('S','T') and 0<=x['token_id']<SEMANTIC for i,x in enumerate(tokens));width=max(len(x['union_ids']) for x in tokens)
 union=torch.zeros((n,width),dtype=torch.long,device=device);mask=torch.zeros((n,width),dtype=torch.bool,device=device)
 teacher=torch.zeros((n,width),device=device);cached=torch.zeros((n,width),device=device)
 for i,x in enumerate(tokens):
  ids=x['union_ids'];assert ids==sorted(set(ids)) and 16<=len(ids)<=32 and all(0<=k<SEMANTIC for k in ids)
  assert len(x['teacher_union_logits'])==len(x['student_union_logits'])==len(ids)
  k=len(ids);union[i,:k]=torch.tensor(ids,device=device);mask[i,:k]=True
  teacher[i,:k]=torch.tensor(x['teacher_union_logits'],device=device);cached[i,:k]=torch.tensor(x['student_union_logits'],device=device)
 sources=torch.tensor([x['source']=='S' for x in tokens],device=device,dtype=torch.bool)
 return union,mask,teacher,cached,sources

def update(torch,model,head,opt,trace):
 m.lib.activate(model,head,'actor');model.train();opt.zero_grad(set_to_none=True)
 params=[p for p in model.parameters() if p.requires_grad];before=m.lib.role_hash(model,'actor');n=len(trace['tokens'])
 assert before==trace['actor_before'];union,mask,q,cached,sources=support_tensors(torch,trace,'cuda')
 ids=torch.tensor([trace['prompt_ids']+[x['token_id'] for x in trace['tokens'][:-1]]],device='cuda');tick=time.monotonic()
 logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n).logits[0].float()
 assert logits.shape==(n,NATIVE),'One exact next-token prediction for every trace position'
 live=logits.gather(-1,union);loss,per,reverse,forward=source_split_kl(live,q,mask,sources)
 if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite actual IPD loss')
 replay=float((live.detach()-cached).abs().masked_fill(~mask,0).max());loss.backward()
 if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):raise RuntimeError('Nonfinite actual IPD gradient')
 norm=torch.nn.utils.clip_grad_norm_(params,1.)
 if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite IPD gradient norm')
 check_stop();changed=float(norm)>0
 if changed:opt.step()
 opt.zero_grad(set_to_none=True);after=m.lib.role_hash(model,'actor')
 if changed and before==after:raise RuntimeError('Nonzero gradient without actual actor change')
 return dict(optimizer_updated=changed,actor_before=before,actor_after=after,loss=float(loss.detach()),gradient_norm=float(norm),
  loss_reverse_mean=float(reverse.mean().detach()),loss_forward_mean=float(forward.mean().detach()),
  cached_vs_gradient_replay_logits_max_abs=replay,input_tokens=len(trace['prompt_ids']),target_tokens=n,
  sources=trace['sources'],events=trace['events'],update_seconds=time.monotonic()-tick,oracle_or_real_gold_in_loss=False)

def bound_torch_pool(torch):
 # Reserve the measured native-teacher/context footprint inside the original combined9GiB.
 frame=snapshot();reserved=torch.cuda.memory_reserved();used=max(0.,(BASELINE_FREE-frame['GPU']['free_MiB'])*2**20)
 non_torch=max(0.,used-reserved);budget=int(9*2**30-non_torch)
 if budget<=reserved:raise BlockingIOError('Measured combined9GiB has no extra Torch allocation margin')
 torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
 return dict(torch_pool_cap_bytes=budget,measured_non_torch_GPU_bytes=non_torch,resource_frame=frame)

def rng_seed(gen,seed,step,identifier):
 return int.from_bytes(hashlib.sha256(f'latest-native-IPD-v1/{gen}/{seed}/{step}/{identifier}'.encode()).digest()[:8],'big')%(2**63-1)

def teacher_open(folder,qualification=False):
 profile=load(P/'profile-proposal.json' if qualification else P/'qualified-profile.json')
 assert profile['ngl']==20 and profile['threads']==2
 return NativeTeacher(BINARY,MODEL,BACKEND,folder/'native-teacher.stderr.txt',profile['ngl'],threads=2,guard=native_guard,timeout=900,qualification=qualification)

def warm_start(out,p):
 w=m.unit(p['generator'],p['seed'],'warm');term=load(w/'fit-terminal.json');folder=Path(term['checkpoint'])
 assert load(w/'independent-reload.json')['status']=='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS'
 cm=m.lib.verify_checkpoint(w,folder);assert bind(folder/'manifest.json')==p['W_manifest']
 torch,tok,model,head=setup(out,folder)
 assert {r:m.lib.role_hash(model,r) for r in cm['role_hashes']}==cm['role_hashes']
 rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
 once(out/'warm-source.json',dict(manifest=p['W_manifest'],actor=m.lib.role_hash(model,'actor'),new_optimizer=True,W_not_retrained=True))
 return torch,tok,model,head

def fit(out):
 p=authorize(out,'fit');data=load(R/p['inputs']['path']);rows=m.ordered(data,p['post50_ids'],'ipd')
 assert not (out/'fit-started.json').exists(),'No unknown started replay or implicit optimizer resume'
 preflight();once(out/'fit-started.json',dict(epoch=time.time(),source=bind(__file__),scheduled=50))
 stage_begin=time.monotonic();tick=stage_begin;torch,tok,model,head=warm_start(out,p);m.lib.activate(model,head,'actor');opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
 assert not opt.state;initial={r:m.lib.role_hash(model,r) for r in ('actor','reference','discriminator')}
 base_versions={k:x._version for k,x in model.named_parameters() if 'lora_' not in k};setup_seconds=time.monotonic()-tick
 actual=committed=0;initial_cp_begin=time.monotonic();folder,_=save(out,0,0,torch,model,head,opt);initial_checkpoint_seconds=time.monotonic()-initial_cp_begin;fit_begin=time.monotonic();compute=pace=cool=checkpoint_seconds=0.;events=Counter();sources=Counter()
 teacher_load_begin=time.monotonic()
 try:
  with teacher_open(out) as teacher:
   assert teacher.template_SHA256==TEACHER_TEMPLATE_SHA
   teacher_stat=file_stat(MODEL);teacher_load_seconds=time.monotonic()-teacher_load_begin;pool=bound_torch_pool(torch);once(out/'actual-combined-GPU-pool.json',pool)
   for step,row in enumerate(rows,1):
    check_stop()
    if time.monotonic()-fit_begin>14400:raise TimeoutError('Bounded actual IPD50 wall4h')
    cool+=cooldown();tick=time.monotonic();time.sleep(.5);pace+=time.monotonic()-tick
    actor=m.lib.role_hash(model,'actor');once(out/'updates'/f'{step:04d}.started.json',dict(step=step,id=row['id'],actor_before=actor))
    prefix=tok.encode(tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
    tick=time.monotonic();trace=trajectory(torch,model,teacher,prefix,rng_seed(p['generator'],p['seed'],step,row['id']),actor)
    once(out/'traces'/f'{step:04d}.json',dict(id=row['id'],**trace));gc.collect()
    bound_torch_pool(torch);metric=update(torch,model,head,opt,trace);torch.cuda.synchronize();metric['compute_seconds']=time.monotonic()-tick;compute+=metric['compute_seconds']
    assert all(m.lib.role_hash(model,r)==initial[r] for r in ('reference','discriminator'))
    assert all(x._version==base_versions[k] for k,x in model.named_parameters() if k in base_versions)
    assert file_stat(MODEL)==teacher_stat
    actual+=int(metric['optimizer_updated']);events.update(metric['events']);sources.update(metric['sources']);metric.update(step=step,id=row['id'],sampling_seconds=trace['seconds'],checkpoint_seconds=0.)
    if step%5==0 or step==50 or STOP:
     tick=time.monotonic();folder,_=save(out,step,actual,torch,model,head,opt);metric['checkpoint_seconds']=time.monotonic()-tick;checkpoint_seconds+=metric['checkpoint_seconds']
    once(out/'updates'/f'{step:04d}.json',metric);committed=step;snapshot()
    atomic(P/'progress.json',dict(unit=str(out.relative_to(R)),scheduled_committed=step,scheduled_total=50,actual_updates=actual,events=dict(events),sources=dict(sources),epoch=time.time(),kind='actual_native_IPD_task_commit'))
    print('ACTUAL_LATEST_NATIVE_IPD',p['generator'],p['seed'],step,'OPTIMIZER',actual,flush=True)
 except InterruptedError:
  if not (out/'checkpoints'/f'step{committed:04d}').exists():save(out,committed,actual,torch,model,head,opt)
  once(out/'paused.json',dict(last_committed=committed,actual_updates=actual,unknown_started_preserved=True));return
 assert actual>0,'All50 tasks closed but no genuine IPD optimizer update; do not claim IPD completion'
 once(out/'fit-terminal.json',dict(status='FIXED_IPD50_COMPLETE_RELOAD_PENDING',scheduled=50,actual_updates=actual,checkpoint=str(folder),manifest=bind(folder/'manifest.json'),final_actor=m.lib.role_hash(model,'actor'),
  setup_seconds=setup_seconds,initial_checkpoint_seconds=initial_checkpoint_seconds,stage_wall_seconds=time.monotonic()-stage_begin,teacher_load_seconds=teacher_load_seconds,fit_wall_seconds=time.monotonic()-fit_begin,compute_seconds=compute,pacing_seconds=pace,cooling_seconds=cool,checkpoint_seconds=checkpoint_seconds,
  events=dict(events),sources=dict(sources),teacher_qualification=p['teacher_qualification'],teacher_artifact_unchanged=True,CUDA_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
  CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),CPU_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,API_calls=0,no_Dev_selection=True))

def reload(out):
 p=authorize(out,'reload');t=load(out/'fit-terminal.json');assert t['status']=='FIXED_IPD50_COMPLETE_RELOAD_PENDING'
 assert not (out/'reload-started.json').exists(),'No unknown reload replay'
 folder=Path(t['checkpoint']);preflight();once(out/'reload-started.json',dict(epoch=time.time(),manifest=bind(folder/'manifest.json')))
 tick=time.monotonic();torch,tok,model,head=setup(out,folder);m.lib.activate(model,head,'actor');opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
 cm=m.lib.restore(torch,model,head,dict(actor=opt),out,folder);rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
 assert torch.equal(torch.get_rng_state(),rng['cpu']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda']))
 assert cm['actual_updates']==t['actual_updates'] and cm['role_hashes']['actor']==t['final_actor'] and cm['step']==50
 once(out/'independent-reload.json',dict(status='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS',manifest=bind(folder/'manifest.json'),actual_updates=cm['actual_updates'],step=50,reload_seconds=time.monotonic()-tick,no_new_optimizer_updates=True))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['fit','reload']);a.add_argument('--generator',choices=m.GENS,required=True);a.add_argument('--seed',type=int,choices=m.SEEDS,required=True);v=a.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 out=m.unit(v.generator,v.seed,'ipd')
 try:
  with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(fit if v.action=='fit' else reload)(out)
 except BaseException as e:
  once(out/(v.action+'-failure.json'),dict(error_type=type(e).__name__,message=str(e),epoch=time.time()));raise