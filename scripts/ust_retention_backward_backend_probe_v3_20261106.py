"""Eight fixed no-update full-model backward probes; prepare is CPU-only.

Same Train item x INIT/source50 x actual FLA/installed reference chunk x2 repeats.
Only process-local chunk dispatch and authorized allocator cap differ from setup.
"""
from pathlib import Path
import argparse,ast,collections,contextlib,copy,fcntl,functools,hashlib,importlib.util,inspect,json,os,signal,subprocess,sys,time
R=Path(__file__).resolve().parents[1];N=R/'runs/2026-11-06-ust-retention-information-v1';O=N/'backward-backend-probe-v3'
PREVIOUS=R/'runs/2026-11-06-ust-autonomous-repair-v1';SOURCE=PREVIOUS/'position-order-replicates-v1/2026110602/synthetic-balanced200'
OLD=R/'runs/2026-10-25-qwen-transfer-repair/seqkd200/USTBench/2b';INIT=OLD/'work/checkpoints/step0000'
LR=N/'learning-rate-control-v1';CONT=N/'continuation-control-v1';HELPER=R/'scripts/ust_retention_forward_probe_v2_20261106.py'
STOP=False;TOTAL=8;NATIVE=248320;SEMANTIC=248077
SCOPE=('No-update implementation diagnostic on one fixed understanding training item and two fixed parameter states. '
 'Cross-backend forward rounding also changes downstream gradients; differences alone do not prove a backward bug or retention cause. '
 'Two within-backend repeats are a limited numerical-noise reference, not a general determinism guarantee. No task utility or heldout test.')
def load(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return dict(path=str(p.relative_to(R)) if p.is_relative_to(R) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def tensor_save(torch,p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:torch.save(x,f)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def check():
 if STOP or (N/'STOP').exists() or (O/'STOP').exists():raise InterruptedError('Own backward diagnostic STOP')
def stop(*_):
 global STOP;STOP=True
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def helper():return module(HELPER,'backward_same_setup_helpers')
def machinery():
 h=helper();h.O=O;h.verify=verify;h.check=check;h.live=live;return h.machinery()
def units():
 return {f'{checkpoint}-{backend}-repeat{repeat}':dict(checkpoint=checkpoint,backend=backend,repeat=repeat) for checkpoint in ('INIT','source50') for repeat in (1,2) for backend in ('fla','reference')}
def checkpoint_specs():return {'INIT':dict(source=OLD,folder=INIT,step=0),'source50':dict(source=SOURCE,folder=SOURCE/'checkpoints/step0050',step=50)}
def reference_function(q):
 original=q.torch_chunk_gated_delta_rule;ref=inspect.unwrap(original)
 assert ref is not original and ref.__name__=='torch_chunk_gated_delta_rule'
 assert Path(inspect.getfile(ref)).resolve()==Path(q.__file__).resolve()
 assert inspect.unwrap(q.Qwen3_5GatedDeltaNet.forward).__globals__['torch_chunk_gated_delta_rule'] is original
 closure=inspect.getclosurevars(original).nonlocals
 assert closure.get('is_new_implementation') is True,'Actual baseline must dispatch beyond the Torch reference'
 impl=closure['implementation'];paths=[];seen=set()
 def visit(fn):
  if not inspect.isfunction(fn) or id(fn) in seen:return
  seen.add(id(fn));paths.append(dict(name=fn.__qualname__,file=fn.__code__.co_filename,firstline=fn.__code__.co_firstlineno))
  for v in inspect.getclosurevars(fn).nonlocals.values():
   if inspect.isfunction(v):visit(v)
  if hasattr(fn,'__wrapped__'):visit(fn.__wrapped__)
 visit(impl);assert any('/fla/' in x['file'] and 'gated_delta_rule/chunk.py' in x['file'] for x in paths)
 return ref,dict(original_symbol_name=original.__name__,reference_source=bind(q.__file__),reference_firstline=ref.__code__.co_firstlineno,baseline_implementation_chain=paths)
def callable_identity(fn):
 raw=inspect.unwrap(fn)
 return dict(name=fn.__qualname__,code_source=bind(fn.__code__.co_filename),firstline=fn.__code__.co_firstlineno,
  unwrapped_name=raw.__qualname__,unwrapped_source=bind(raw.__code__.co_filename),unwrapped_firstline=raw.__code__.co_firstlineno,wrapped_is_reference=fn is raw)
def numerical_flags(torch):
 return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
  cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
  float32_matmul_precision=torch.get_float32_matmul_precision(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
  SDPA_enabled={name:getattr(torch.backends.cuda,name+'_sdp_enabled')() for name in ['flash','mem_efficient','math','cudnn']},
  environment={key:os.environ.get(key) for key in ['CUBLAS_WORKSPACE_CONFIG','CUBLASLT_WORKSPACE_SIZE','NVIDIA_TF32_OVERRIDE','TORCH_BLAS_PREFER_CUBLASLT','FLA_CACHE_MODE','FLA_DISABLE_BACKEND_DISPATCH','TRITON_INTERPRET','CUDA_LAUNCH_BLOCKING']})
@contextlib.contextmanager
def chunk_dispatch(q,backend,torch,expected_length):
 assert backend in ('fla','reference');original=q.torch_chunk_gated_delta_rule;ref,proof=reference_function(q);chosen=original if backend=='fla' else ref;calls=[]
 proof=dict(proof,selected_backend=backend,selected_callable=callable_identity(chosen),original_callable=callable_identity(original),reference_callable=callable_identity(ref))
 @functools.wraps(chosen)
 def observe(*args,**kwargs):
  assert len(args)>=3;query,key,value=args[:3]
  assert query.shape[0]==1 and query.shape[1]==expected_length and key.shape[:2]==query.shape[:2] and value.shape[:2]==query.shape[:2]
  assert kwargs.get('initial_state') is None and kwargs.get('output_final_state') is False and kwargs.get('cu_seqlens') is None and kwargs.get('use_qk_l2norm_in_kernel') is True
  calls.append(dict(query_shape=list(query.shape),query_dtype=str(query.dtype),key_dtype=str(key.dtype),value_dtype=str(value.dtype),grad_enabled=torch.is_grad_enabled()))
  return chosen(*args,**kwargs)
 q.torch_chunk_gated_delta_rule=observe
 assert inspect.unwrap(q.Qwen3_5GatedDeltaNet.forward).__globals__['torch_chunk_gated_delta_rule'] is observe
 try:yield calls,proof
 finally:
  assert q.torch_chunk_gated_delta_rule is observe
  q.torch_chunk_gated_delta_rule=original
  assert inspect.unwrap(q.Qwen3_5GatedDeltaNet.forward).__globals__['torch_chunk_gated_delta_rule'] is original

def cap_setup_ast(worker_source):
 node=copy.deepcopy(next(x for x in ast.parse(Path(worker_source).read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='setup'));original=copy.deepcopy(node)
 expected='12*2**30/torch.cuda.get_device_properties(0).total_memory';found=[]
 for x in ast.walk(node):
  if isinstance(x,ast.Constant) and x.value==expected:found.append(x);x.value='0.8'
 assert len(found)==1
 reverse=copy.deepcopy(node)
 for x in ast.walk(reverse):
  if isinstance(x,ast.Constant) and x.value=='0.8':x.value=expected
 assert ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False)
 return ast.fix_missing_locations(node),dict(source=bind(worker_source),changed_only='setup AST allocator expression12GiB/total ->0.8',reverse_exact_original_AST=True)
def allocator_fraction_proof(fraction,total):
 assert total>0 and 0<=fraction<=1
 tolerance=max(1e-12,1./total);error=abs(fraction-.8)
 assert error<=tolerance,'Allocator readback differs by more than one-byte fraction granularity'
 return dict(requested_fraction=.8,reported_fraction=fraction,absolute_error=error,tolerance=tolerance,error_bytes=error*total,tolerance_bytes=tolerance*total)
def install_cap_setup(w):
 node,proof=cap_setup_ast(w.__file__);ns=dict(w.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),str(w.__file__)+'::authorized80percentallocator','exec'),ns);w.setup=ns['setup'];return proof

def prepare():
 assert not O.exists(),'Immutable namespace';check();os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('HF_HUB_OFFLINE','1')
 h=helper();source_rows=load(SOURCE/'training.local.json');row=source_rows[0];assert row['metadata']['ability']=='st_understanding'
 update=load(SOURCE/'updates/0001.json');assert row['id']==update['id'] and update['input_tokens']==6004 and update['target_tokens']==7
 prefix=load(CONT/'fixed8.local.json');p0=load(SOURCE/'protocol.json');sys.path[:0]=[str(R/'scripts'),str(R/'src')]
 from transformers import AutoTokenizer
 from transformers.models.qwen3_5 import modeling_qwen3_5 as q
 from privacy_synth.ust_task_contract import certify_native_ranking_reference
 import probe_overnight_train_fit_20261025 as packing
 import torch
 assert not torch.cuda.is_initialized();tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False)
 prompt,target=packing.packing(tok,row,dict(text=row['target']),'matched8',prefix);assert prompt==h.eval_prompt(tok,prefix,row['question']) and len(prompt)==6004 and len(target)==7 and len(prompt)+len(target)<=16384
 answer=row['target'][8];assert row['target']=='<Answer>'+answer+'</Answer>' and certify_native_ranking_reference(row['question'],answer)['trusted_gold']
 family,decision=h.target_family(tok);assert target==family[answer]
 _,implementation=reference_function(q);pins=[bind(__file__),bind(R/'scripts/ust_retention_backward_backend_probe_v2_20261106.py'),bind(R/'scripts/ust_retention_backward_backend_probe_20261106.py'),bind(R/'tests/test_ust_retention_backward_backend_probe_v3.py'),bind(HELPER),bind(SOURCE/'training.local.json'),bind(SOURCE/'updates/0001.json'),bind(CONT/'fixed8.local.json'),bind(N/'budget-continuation.json')];specs={}
 for key,s in checkpoint_specs().items():
  cm,pp=h.checkpoint_pins(s['source'],s['folder'],s['step']);pins+=pp;specs[key]=dict(source=str(s['source'].relative_to(R)),folder=str(s['folder'].relative_to(R)),step=s['step'],roles=cm['role_hashes'],optimizer=cm['optimizer_hashes']['actor'],manifest=bind(s['folder']/'manifest.json'))
 O.mkdir(parents=True);save(O/'query.local.json',dict(source_id=row['id'],question=row['question'],target=row['target'],metadata=row['metadata'],answer=answer,prompt_ids=prompt,target_ids=target,source_step=1));save(O/'fixed8.local.json',prefix);save(O/'implementation-before-GPU.json',implementation)
 m,w,_=machinery();node,proof=cap_setup_ast(w.__file__);save(O/'allocator-AST-proof.json',proof)
 with (O/'allocator-setup-source.py').open('x') as f:f.write(ast.unparse(node)+'\n')
 for name in ('query.local.json','fixed8.local.json','implementation-before-GPU.json','allocator-AST-proof.json','allocator-setup-source.py'):pins.append(bind(O/name))
 for mod in tuple(sys.modules.values()):
  path=getattr(mod,'__file__',None)
  if path:
   path=Path(path).resolve()
   if path.is_file() and path.is_relative_to(R) and path.suffix=='.py' and '.venv' not in path.parts and 'vendor' not in path.parts:pins.append(bind(path))
 import fla
 for path in Path(fla.__file__).resolve().parent.rglob('*.py'):pins.append(bind(path))
 for rel in ('models/qwen3_5/modeling_qwen3_5.py','integrations/hub_kernels.py','integrations/sdpa_attention.py','modeling_layers.py','cache_utils.py'):
  candidates=list((R/'.venv/lib').glob('python*/site-packages/transformers/'+rel));assert len(candidates)==1;pins.append(bind(candidates[0]))
 for item in implementation['baseline_implementation_chain']:pins.append(bind(item['file']))
 for rel in ('utils/checkpoint.py','nn/functional.py','nn/attention/__init__.py','backends/cuda/__init__.py','backends/cudnn/__init__.py'):
  pins.append(bind(Path(torch.__file__).resolve().parent/rel))
 pins += [bind(Path(torch.__file__)),bind(R/'pyproject.toml'),bind(R/'uv.lock')]+[bind(p) for p in Path(p0['base']).iterdir() if p.is_file()]
 pins=list({x['path']:x for x in pins}.values());assert not torch.cuda.is_initialized()
 save(O/'protocol.json',dict(source=bind(__file__),bindings=pins,scope=SCOPE,units=units(),checkpoints=specs,unit_count=8,source_queries=1,prompt_tokens=6004,target_tokens=7,total_new_updates=0,API=0,
  semantic_vocabulary=SEMANTIC,native_vocabulary=NATIVE,decision_position=decision,candidate_target_ids=family,allocator_fraction=.8,unit_wall_limit_seconds=180,batch_wall_limit_seconds=1200,
  unchanged_other_numerical_settings=True,no_SDPA_backend_override=True,no_grad_clipping=True,no_optimizer_step=True,all8_close_before_comparison=True,zero_gradient_norm_cosine='NA',dependencies=[str(LR.relative_to(R)),str(CONT.relative_to(R))],Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False))
 protocols=[bind(O/'protocol.json')]
 for key,u in units().items():
  spec=specs[u['checkpoint']];p=dict(p0);p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],model_id=key,steps=0,training=0,fixed_checkpoint=spec['step'],source_checkpoint=spec,
   total_raw=0,method='no-update-backward-implementation-diagnostic',scope=SCOPE,evaluation_use=SCOPE,ability_order=[],checkpoint_steps=[],protected_real_training_gold_in_loss=False,synthetic_reference_in_loss=False,teacher_API=0,Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False)
  save(O/key/'protocol.json',p);protocols.append(bind(O/key/'protocol.json'))
 save(O/'execution-bindings.json',dict(source=bind(__file__),protocols=protocols));verify();print('CPU_PREPARED8no-updatebackwardunits;GPU0/API0',flush=True)
def verify(unit=None):
 check();e=load(O/'execution-bindings.json');assert e['source']==bind(__file__)
 for pin in e['protocols']:assert bind(R/pin['path'])==pin,pin['path']
 p=load((Path(unit) if unit is not None else O)/'protocol.json');assert p['source']==bind(__file__)
 for pin in p['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 if unit is not None:assert Path(unit).parent==O and Path(unit).name in units() and p['steps']==p['training']==0
 return p

def gate(require_dependencies=True):
 verify();a=load(O/'launch-authorization.json');assert a['status']=='AUTHORIZED_FIXED_BACKWARD_PROBE' and a['protocol']==bind(O/'protocol.json')
 pin=a['independent_review'];assert (R/pin['path']).resolve().is_relative_to(N.resolve()) and bind(R/pin['path'])==pin and str(load(R/pin['path'])['status']).startswith('PASS')
 if require_dependencies:
  for folder in (LR,CONT):
   t=load(folder/'terminal.json');assert t['status']=='COMPLETE' and bind(R/t['results']['path'])==t['results']
 return a

def run_unit(key):
 assert key in units();gate();unit=O/key;p=verify(unit);spec=p['source_checkpoint'];s=units()[key];m,w,_=machinery();assert install_cap_setup(w)==load(O/'allocator-AST-proof.json');h=helper();qrow=load(O/'query.local.json');tick=time.monotonic()
 save(unit/'started.json',dict(PID=os.getpid(),epoch=time.time(),new_updates=0,backend=s['backend']))
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);folder=R/spec['folder'];torch,tok,model,head=w.initialize(unit,folder);w.m.lib.activate(model,head,'actor');model.train()
  from transformers.models.qwen3_5 import modeling_qwen3_5 as q
  params=[(name,v) for name,v in model.named_parameters() if v.requires_grad];assert len(params)==48 and sum(p.numel() for _,p in params)==737280 and all('lora_' in n and '.actor.' in n and p.dtype==torch.float32 for n,p in params) and not any(p.requires_grad for p in head.parameters())
  opt=torch.optim.AdamW([v for _,v in params],lr=1e-4,weight_decay=0,foreach=False);cm=w.m.lib.restore(torch,model,head,dict(actor=opt),R/spec['source'],folder)
  roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};oh=w.m.lib.nested_hash(opt.state_dict());rng=lambda:dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all());rh=w.m.lib.nested_hash(rng());hh=w.m.lib.nested_hash(head.state_dict())
  flags=numerical_flags(torch)
  assert roles==spec['roles'] and oh==spec['optimizer'] and cm['actual_updates']==spec['step'] and rh==w.m.lib.nested_hash(torch.load(folder/'rng.pt',map_location='cpu',weights_only=True))
  assert all(float(x.p)==0 for x in model.modules() if isinstance(x,torch.nn.Dropout));assert model.is_gradient_checkpointing
  assert h.eval_prompt(tok,load(O/'fixed8.local.json'),qrow['question'])==qrow['prompt_ids'];h.rope_check(model)
  total=torch.cuda.get_device_properties(0).total_memory;fraction=torch.cuda.get_per_process_memory_fraction();allocator_readback=allocator_fraction_proof(fraction,total)
  save(unit/'reload.json',dict(status='EXACT_ROLES_ADAM_TORCH_RNG_NO_UPDATE',roles=roles,optimizer=oh,RNG=rh,head=hh,allocator_fraction=fraction,allocator_readback=allocator_readback,total_GPU_bytes=total,parameter_names=[n for n,_ in params],torch_version=torch.__version__,CUDA_build=torch.version.cuda,numerical_flags=flags,gradient_checkpointing=True,base_dtype=str(model.get_base_model().dtype),adapter_dtypes=sorted({str(v.dtype) for _,v in params}),actual_linear_dispatch=reference_function(q)[1],attn_implementation=model.get_base_model().config.text_config._attn_implementation))
  model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);prompt=qrow['prompt_ids'];target=qrow['target_ids'];ids=torch.tensor([prompt+target[:-1]],device='cuda');gold=torch.tensor(target,device='cuda');torch.cuda.reset_peak_memory_stats();compute_start=time.monotonic()
  with chunk_dispatch(q,s['backend'],torch,len(ids[0])) as (calls,dispatch):
   out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(target));logits=out.logits[0].float();assert out.past_key_values is None and tuple(logits.shape)==(7,NATIVE)
   cpu_logits=logits.detach().cpu().clone();tensor_save(torch,unit/'logits.pt',cpu_logits);assert bool(torch.isfinite(logits).all())
   loss=torch.nn.functional.cross_entropy(logits[:,:SEMANTIC],gold);assert bool(torch.isfinite(loss));forward_calls=len(calls);assert forward_calls==18
   save(unit/'forward.json',dict(loss=float(loss.detach()),linear_calls=forward_calls,logits=bind(unit/'logits.pt'),dispatch=dispatch))
   loss.backward();grads={n:v.grad.detach().cpu().clone() for n,v in params if v.grad is not None};tensor_save(torch,unit/'gradients.pt',grads)
   assert set(grads)=={n for n,_ in params} and all(bool(torch.isfinite(g).all()) for g in grads.values())
   assert all(v.grad is None for n,v in model.named_parameters() if not v.requires_grad)
  assert len(calls)>=forward_calls and numerical_flags(torch)==flags;h.rope_check(model);torch.cuda.synchronize();compute_seconds=time.monotonic()-compute_start
  save(unit/'diagnostic.json',dict(status='NO_UPDATE_FORWARD_BACKWARD_COMPLETE',loss=float(loss.detach()),backend=s['backend'],checkpoint=s['checkpoint'],repeat=s['repeat'],linear_forward_calls=forward_calls,linear_calls_through_backward=len(calls),calls=calls,
   gradients=[dict(name=n,shape=list(g.shape),dtype=str(g.dtype),finite=True,L2=float(g.double().norm()),max_abs=float(g.abs().max()),zero_norm=bool(g.count_nonzero()==0)) for n,g in grads.items()],
   roles_before=roles,optimizer_before=oh,RNG_before=rh,head_before=hh,compute_seconds=compute_seconds,elapsed_seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocator_fraction=fraction,total_GPU_bytes=total,
   native_target_position_logits=bind(unit/'logits.pt'),all48_gradients=bind(unit/'gradients.pt'),optimizer_calls=0,clipping_calls=0))
  model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
  assert roles=={r:w.m.lib.role_hash(model,r) for r in roles} and oh==w.m.lib.nested_hash(opt.state_dict()) and rh==w.m.lib.nested_hash(rng()) and hh==w.m.lib.nested_hash(head.state_dict()) and all(v.grad is None for v in model.parameters())
  verify(unit);save(unit/'terminal.json',dict(status='COMPLETE_NO_UPDATE_BACKWARD',roles=roles,optimizer=oh,RNG=rh,head=hh,roles_unchanged=True,optimizer_unchanged=True,RNG_unchanged=True,head_unchanged=True,dispatch_restored=True,new_updates=0,API=0,
   outputs=[bind(unit/name) for name in ['reload.json','forward.json','diagnostic.json','logits.pt','gradients.pt']],epoch=time.time()));print('CLOSED',key,round(time.monotonic()-tick,1),flush=True)

def differences(torch,a,b):
 assert a.shape==b.shape;aa=a.detach().double().reshape(-1);bb=b.detach().double().reshape(-1);delta=bb-aa;na=float(aa.norm());nb=float(bb.norm());nd=float(delta.norm())
 return dict(equal_bytes=bool(torch.equal(a,b)),finite_a=bool(torch.isfinite(aa).all()),finite_b=bool(torch.isfinite(bb).all()),L2_a=na,L2_b=nb,L2_difference=nd,max_abs_difference=float(delta.abs().max()),mean_abs_difference=float(delta.abs().mean()),
  relative_L2_to_a=nd/na if na>0 else None,relative_L2_to_max=nd/max(na,nb) if max(na,nb)>0 else None,cosine=float(torch.dot(aa,bb))/(na*nb) if na>0 and nb>0 else None,zero_norm_a=na==0,zero_norm_b=nb==0)
def score():
 verify();pins=[];terms={}
 for key in units():
  unit=O/key;t=load(unit/'terminal.json');assert t['status']=='COMPLETE_NO_UPDATE_BACKWARD' and all(t[k] for k in ['roles_unchanged','optimizer_unchanged','RNG_unchanged','head_unchanged','dispatch_restored']) and t['new_updates']==t['API']==0
  for pin in t['outputs']:assert bind(R/pin['path'])==pin
  terms[key]=t;pins+=[bind(unit/'terminal.json')]+t['outputs']
 save(O/'all8-closed-before-comparison.json',dict(bindings=pins,epoch=time.time()));import torch
 data={key:dict(logits=torch.load(O/key/'logits.pt',map_location='cpu',weights_only=True),grads=torch.load(O/key/'gradients.pt',map_location='cpu',weights_only=True),diagnostic=load(O/key/'diagnostic.json')) for key in units()}
 comparisons={}
 pairs=[]
 for checkpoint in ('INIT','source50'):
  pairs += [(f'{checkpoint}-{backend}-repeat1',f'{checkpoint}-{backend}-repeat2','within_backend_repeat') for backend in ('fla','reference')]
  pairs += [(f'{checkpoint}-fla-repeat{repeat}',f'{checkpoint}-reference-repeat{repeat}','paired_backend') for repeat in (1,2)]
 for first,second,kind in pairs:
  a=data[first];b=data[second];assert set(a['grads'])==set(b['grads']) and len(a['grads'])==48
  comparisons[first+'__'+second]=dict(kind=kind,loss_a=a['diagnostic']['loss'],loss_b=b['diagnostic']['loss'],loss_difference=b['diagnostic']['loss']-a['diagnostic']['loss'],
   all_logits=differences(torch,a['logits'],b['logits']),per_target_position=[dict(position=i,native_argmax_a=int(a['logits'][i].argmax()),native_argmax_b=int(b['logits'][i].argmax()),**differences(torch,a['logits'][i],b['logits'][i])) for i in range(7)],
   pooled_gradients=differences(torch,torch.cat([a['grads'][n].flatten() for n in sorted(a['grads'])]),torch.cat([b['grads'][n].flatten() for n in sorted(b['grads'])])),
   per_parameter=[dict(name=n,**differences(torch,a['grads'][n],b['grads'][n])) for n in sorted(a['grads'])])
 assert all(bind(R/p['path'])==p for p in pins);verify()
 if (O/'active-batch-started.json').exists():assert time.time()<=load(O/'active-batch-started.json')['deadline_epoch'],'Active batch including comparison exceeded1200seconds'
 save(O/'results.json',dict(status='COMPLETE_FIXED8_NO_UPDATE_BACKWARD',scope=SCOPE,comparisons=comparisons,unit_count=8,optimizer_calls=0,API=0,interpretation='Numerical implementation comparison, not proof that any gradient difference caused retention failure; reference forward precision differs.'))
 save(O/'terminal.json',dict(status='COMPLETE',results=bind(O/'results.json'),epoch=time.time(),updates=0,API=0));live(stage='COMPLETE');print('COMPLETE8backwarddiagnostic',flush=True)
def controller():
 gate(require_dependencies=False);save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time()));wait_deadline=time.monotonic()+21600
 while not all((p/'terminal.json').exists() for p in (LR,CONT)):
  check();assert not any(list(p.glob('failure-*.json')) for p in (LR,CONT)),'Required study failed; no automatic probe'
  if time.monotonic()>wait_deadline:raise TimeoutError('Dependency wait exceeded21600seconds')
  live(stage='WAITING_FOR_LR_AND_CONTINUATION');time.sleep(5)
 gate();save(O/'dependencies-closed.json',dict(dependencies=[dict(terminal=bind(p/'terminal.json'),results=load(p/'terminal.json')['results']) for p in (LR,CONT)],epoch=time.time()))
 deadline=time.monotonic()+1200;save(O/'active-batch-started.json',dict(epoch=time.time(),deadline_epoch=time.time()+1200,hard_unit_seconds=180,total_seconds=1200))
 for key in units():
  check();remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('Active batch exceeded1200seconds')
  live(stage='NO_UPDATE_BACKWARD',unit=key,remaining_seconds=remaining)
  with (O/(key+'.log')).open('x') as f:subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)),'run-unit','--unit',key],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=min(180,remaining))
 if time.monotonic()>deadline:raise TimeoutError('Active batch exceeded1200seconds after final unit')
 remaining=deadline-time.monotonic()
 if remaining<=0:raise TimeoutError('No remaining batch time for CPU comparison')
 with (O/'score.log').open('x') as f:subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)),'score'],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','run-unit','score','controller']);p.add_argument('--unit',choices=units());args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if args.action=='run-unit':run_unit(args.unit)
  else:globals()[args.action]()
 except BaseException as e:
  if O.exists():(O/('failure-'+str(time.time_ns())+'.json')).write_text(json.dumps(dict(type=type(e).__name__,error=str(e),epoch=time.time(),no_automatic_retry=True,updates=0,API=0)))
  raise
