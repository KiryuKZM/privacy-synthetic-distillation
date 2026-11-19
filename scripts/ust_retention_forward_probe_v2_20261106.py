"""Frozen 4 x 18 seen-training forward/cache diagnosis; zero updates and API.

Teacher-forced candidate scores are conditional diagnostics, not generated
accuracy. No Common, Audit, Confirm or Final questions are read.
"""
from pathlib import Path
import argparse,ast,collections,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
B=R/'runs/2026-11-06-ust-autonomous-repair-v1'
N=R/'runs/2026-11-06-ust-retention-information-v1';O=N/'retention-forward-probe-v4'
REP=B/'position-order-replicates-v1/2026110602';OWN=B/'own-training-probe-v1'
CAUSAL=R/'scripts/ust_causal_training_20261106.py';DECODER=R/'scripts/three_method_common_eval_20261101.py'
SEMANTIC=248077;NATIVE=248320;COUNT=18;TOTAL=72;STOP=False
MODES=('train_full','eval_full','eval_cached')
SCOPE='Posthoc retention diagnosis on fixed18 training sources seen by update50; two own representations and two checkpoints; no heldout utility, no candidate scoring as free generation, no training or API.'
def load(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return dict(path=str(p.relative_to(R)) if p.is_relative_to(R) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def check():
 if STOP or (O/'STOP').exists() or (N/'STOP').exists():raise InterruptedError('Retention probe STOP')
def stop(*_):
 global STOP;STOP=True
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def units():return {f'{kind}-step{step:04d}':dict(kind=kind,step=step,source=REP/f'synthetic-{kind}200') for kind in ('identity','balanced') for step in (50,200)}
def checkpoint_pins(root,folder,step):
 m=load(folder/'manifest.json');assert m['actual_updates']==m['step']==step
 assert m['protocol_sha256']==bind(root/'protocol.json')['sha256']
 assert set(m['role_hashes'])=={'actor','reference','discriminator'}
 assert {'actor/adapter_model.safetensors','reference/adapter_model.safetensors','discriminator/adapter_model.safetensors','score-head.pt','optimizers.pt','rng.pt'}<=set(m['files'])
 pins=[bind(folder/'manifest.json'),bind(root/'protocol.json')]
 for name,digest in m['files'].items():
  path=(folder/name).resolve();assert path.is_relative_to(folder.resolve());pin=bind(path);assert pin['sha256']==digest;pins.append(pin)
 return m,pins

def machinery():
 c=module(CAUSAL,'retention_same_worker');c.O=O;c.B=N;c.verify=verify;c.check=check;c.live=live;return c.machinery()
def decoder():
 nodes=[x for x in ast.parse(DECODER.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in ('hf_greedy','context_failure')];assert len(nodes)==2
 ns=dict(time=time,CAP=128,CONTEXT=16384);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(DECODER)+'::UNCHANGED_DECODER','exec'),ns);return ns['hf_greedy']
def eval_prompt(tok,prefix,question):return tok.encode(tok.apply_chat_template(prefix+[dict(role='user',content=question)],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
def target_family(tok):
 eos=tok.convert_tokens_to_ids('<|im_end|>');assert eos==248046
 family={a:tok.encode('<Answer>'+a+'</Answer>',add_special_tokens=False)+[eos] for a in 'ABCD'}
 assert {len(x) for x in family.values()}=={7}
 varying=[i for i in range(7) if len({x[i] for x in family.values()})>1];assert varying==[2]
 assert all(tok.decode(x,skip_special_tokens=False)=='<Answer>'+a+'</Answer><|im_end|>' for a,x in family.items())
 return family,varying[0]

def prepare():
 assert not O.exists(),'Immutable output namespace already exists';check()
 p0=load(REP/'synthetic-balanced200/protocol.json');prefix=load(B/'position-training-v2/fixed8.local.json')
 sys.path[:0]=[str(R/'scripts'),str(R/'src')]
 import probe_overnight_train_fit_20261025 as packing
 from privacy_synth.ust_task_contract import certify_native_ranking_reference
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);assert len(tok)==SEMANTIC
 family,decision=target_family(tok);pins=[bind(__file__),bind(R/'tests/test_ust_retention_forward_probe_v2.py'),bind(R/'scripts/ust_retention_forward_probe_20261106.py'),bind(CAUSAL),bind(DECODER),bind(Path(packing.__file__)),bind(N/'budget-continuation.json'),bind(R/'src/privacy_synth/ust_task_contract.py')]
 frozen={};prepared={};source_ids=None
 for key,u in units().items():
  source=u['source'];step=u['step'];folder=source/'checkpoints'/f'step{step:04d}';cm,cp=checkpoint_pins(source,folder,step);pins+=cp
  training=load(source/'training.local.json');assert len(training)==200
  chosen=sorted([(i,row) for i,row in enumerate(training,1) if i<=50 and row['metadata']['ability']=='st_understanding'],key=lambda x:x[1]['id']);assert len(chosen)==COUNT
  ids=[row['id'] for _,row in chosen]
  if source_ids is None:source_ids=ids
  assert ids==source_ids
  if u['kind']=='identity' and step==200:
   oldbase=source;oldqueries=load(REP/'queries.local.json');oldpairs={q['source_id']:(i,q) for i,q in enumerate(oldqueries) if q['variant']=='train-original'};pins.append(bind(REP/'queries.local.json'))
  else:
   oldbase=OWN/f"{u['kind']}-order2026110602-step{step:04d}";oldqueries=load(oldbase/'queries.local.json');oldpairs={q['source_id']:(i,q) for i,q in enumerate(oldqueries)};pins.append(bind(oldbase/'queries.local.json'))
  assert len(oldpairs)==68
  oldterm=load(oldbase/'evaluation-terminal.json');assert oldterm['roles_unchanged'] and oldterm['optimizer_unchanged'];pins.append(bind(oldbase/'evaluation-terminal.json'))
  rows=[];proof=[]
  for first,row in chosen:
   answer=row['target'][8];assert row['target']=='<Answer>'+answer+'</Answer>' and answer in 'ABCD'
   cert=certify_native_ranking_reference(row['question'],answer);assert cert['trusted_gold']
   prompt,target=packing.packing(tok,row,dict(text=row['target']),'matched8',prefix)
   assert prompt==eval_prompt(tok,prefix,row['question']) and target==family[answer] and len(prompt)+128<=16384
   oi,oq=oldpairs[row['id']];assert oq['question']==row['question'],'Must be exact actual training prompt, not silently canonicalized'
   oldrawpath=oldbase/'raw'/f'{oi:04d}.json';oldraw=load(oldrawpath);assert oldraw['id']==oq['id'] and oldraw['prompt_tokens']==len(prompt)
   assert oldraw['response_ids'][:decision]==target[:decision], 'Prior free output must reach the same decision prefix'
   updatepath=source/'updates'/f'{first:04d}.json';update=load(updatepath);assert update['id']==row['id'] and update['input_tokens']==len(prompt) and update['target_tokens']==len(target)
   # Independently preserve trace fields rather than assuming names or meanings.
   rows.append(dict(id=row['id']+'::retention-own',source_id=row['id'],question=row['question'],metadata=row['metadata'],target=row['target'],answer=answer,
    first_training_step=first,seen_by_50=True,prompt_ids=prompt,target_ids=target,old_raw=bind(oldrawpath),old_id=oq['id'],source_update=bind(updatepath)))
   proof.append(dict(source_id=row['id'],first_training_step=first,prompt_tokens=len(prompt),target_tokens=len(target),train_eval_token_equal=True,old_eval_exact_question_equal=True,source_update_token_counts_equal=True,old_free_decision_prefix_equal=True,certificate=cert))
   pins.extend([bind(oldrawpath),bind(updatepath)])
  prepared[key]=(rows,proof)
  frozen[key]=dict(source_root=str(source.relative_to(R)),checkpoint=str(folder.relative_to(R)),step=step,kind=u['kind'],role_hashes=cm['role_hashes'],optimizer_hashes=cm['optimizer_hashes'])
  pins.append(bind(source/'training.local.json'))
 O.mkdir(parents=True)
 save(O/'fixed8.local.json',prefix);pins.append(bind(O/'fixed8.local.json'))
 for key,(rows,proof) in prepared.items():
  save(O/key/'queries.local.json',rows);save(O/key/'qualification.local.json',proof)
  pins.extend([bind(O/key/'queries.local.json'),bind(O/key/'qualification.local.json')])
 machinery()
 import eval_ust_sg_qwen_modes_20261024,run_llm_task_sweep_20261023,build_llm_task_sweep_qa_20261023
 # Freeze local execution modules, actual installed model/cache/generation source,
 # environment locks and all base assets before GPU execution.
 for mod in tuple(sys.modules.values()):
  path=getattr(mod,'__file__',None)
  if path:
   path=Path(path).resolve()
   if path.is_file() and path.is_relative_to(R) and path.suffix=='.py' and '.venv' not in path.parts and 'vendor' not in path.parts:pins.append(bind(path))
 for rel in ('models/qwen3_5/modeling_qwen3_5.py','modeling_layers.py','cache_utils.py','generation/utils.py'):
  matches=list((R/'.venv/lib').glob('python*/site-packages/transformers/'+rel));assert len(matches)==1;pins.append(bind(matches[0]))
 pins += [bind(R/'pyproject.toml'),bind(R/'uv.lock')]+[bind(p) for p in Path(p0['base']).iterdir() if p.is_file()]
 pins=list({x['path']:x for x in pins}.values())
 save(O/'protocol.json',dict(source=bind(__file__),bindings=pins,units=frozen,queries_each=COUNT,total_raw=TOTAL,source_ids=source_ids,source_cases=18,
  scope=SCOPE,modes=MODES,training=0,API=0,candidate_target_ids=family,decision_position=decision,semantic_vocabulary=SEMANTIC,native_vocabulary=NATIVE,
  candidate_scores_are_free_generation=False,train_cached_not_run='gradient checkpoint wrapper disables caching in training mode; no config intervention',
  cache_path='prompt prefill predicts target0; feed only previous target token for each next position with full-length attention mask; separate fresh cache per case',
  all_raw_closed_before_score=True,text_rope_state='Each independent case starts None; actual generate pure-text zero [1,1] int64 state recorded and cleared after raw save; any nonzero state rejected',actual_generate_crosscheck='Read-only forward hook captures first3 candidate logits and native top1 from the same existing free generate call; compare decision only if same emitted prefix',prior_failed_attempt=dict(namespace='retention-forward-probe-v3',new_updates=0,completed_teacherforced_cases=1,executed_free_generations=1,free_raw_closed=0),Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False))
 for key,spec in frozen.items():
  p=dict(p0);p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],model_id=key,source_checkpoint=spec,steps=0,training=0,queries_each=COUNT,total_raw=TOTAL,
   method='retention-forward-diagnostic-only',fixed_checkpoint=spec['step'],teacher_API=0,scope=SCOPE,evaluation_use=SCOPE,ability_order=[],checkpoint_steps=[],
   protected_real_training_gold_in_loss=False,synthetic_reference_in_loss=False,Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False)
  save(O/key/'protocol.json',p)
 save(O/'execution-bindings.json',dict(source=bind(__file__),protocols=[bind(O/'protocol.json')]+[bind(O/key/'protocol.json') for key in frozen]))
 verify();print('FROZEN 4 x 18 = 72; CPU only; 0 updates/API',flush=True)

def verify(unit=None):
 check();e=load(O/'execution-bindings.json');assert e['source']==bind(__file__)
 for pin in e['protocols']:assert bind(R/pin['path'])==pin,pin['path']
 p=load((Path(unit) if unit is not None else O)/'protocol.json');assert p['source']==bind(__file__) and p['queries_each']==COUNT and p['total_raw']==TOTAL
 for pin in p['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 if unit is not None:assert Path(unit).parent==O and Path(unit).name in units() and p['steps']==p['training']==0
 return p

def rng_state(torch):return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def rope_check(model):
 base=model.get_base_model();assert hasattr(base.model,'rope_deltas')
 assert base.model.rope_deltas is None,'Unexpected multimodal state in text-only probe'
def text_rope_snapshot(model):
 value=model.get_base_model().model.rope_deltas
 if value is None:return None
 return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.detach().cpu().tolist())
def reset_text_rope(model):
 before=text_rope_snapshot(model)
 if before is not None:
  assert before==dict(shape=[1,1],dtype='torch.int64',values=[[0]]),'Unexpected nonzero/invalid text rope state'
 model.get_base_model().model.rope_deltas=None
 return dict(before=before,after=None)
def free_decode_observed(torch,tok,model,prompt,decode,family,decision):
 """Observe existing free generation without changing its tensors or outputs."""
 choice_ids=[family[a][decision] for a in 'ABCD'];captured=[]
 def observe(module,args,kwargs,result):
  if len(captured)>decision:return
  z=result.logits[0,-1].detach();positions=kwargs.get('position_ids')
  captured.append(dict(candidate_logits=z[choice_ids].clone(),native_top1=z.argmax().clone(),
   position_shape=list(positions.shape) if positions is not None else None,
   last_positions=positions[..., -1:].detach().clone() if positions is not None else None,
   input_tokens=int(kwargs['input_ids'].shape[1]) if kwargs.get('input_ids') is not None else None))
 handle=model.get_base_model().register_forward_hook(observe,with_kwargs=True)
 try:raw=decode(torch,tok,model,prompt)
 finally:handle.remove()
 assert len(captured)==min(decision+1,len(raw['response_ids']))
 observed=[]
 for j,x in enumerate(captured):
  values=x['candidate_logits'].float().cpu();assert bool(torch.isfinite(values).all())
  observed.append(dict(position=j,candidate_logits=dict(zip('ABCD',map(float,values))),native_top1=int(x['native_top1'].cpu()),
   position_shape=x['position_shape'],last_positions=x['last_positions'].cpu().tolist() if x['last_positions'] is not None else None,input_tokens=x['input_tokens']))
 return raw,observed
def forward_full(torch,model,prompt,target,training):
 model.train(training);rope_check(model)
 ids=torch.tensor([prompt+target[:-1]],device='cuda')
 with torch.no_grad():out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(target))
 assert out.past_key_values is None;logits=out.logits[0].detach().float().cpu();assert tuple(logits.shape)==(len(target),NATIVE)
 rope_check(model);return logits,dict(mode='train' if training else 'eval',use_cache=False,input_tokens=len(ids[0]),target_tokens=len(target))
def forward_cached(torch,model,prompt,target):
 model.eval();rope_check(model);cache=None;all_logits=[];steps=[]
 with torch.no_grad():
  for j in range(len(target)):
   tokens=prompt if j==0 else [target[j-1]];total=len(prompt)+j
   ids=torch.tensor([tokens],device='cuda');mask=torch.ones((1,total),device='cuda',dtype=ids.dtype)
   out=model(input_ids=ids,attention_mask=mask,use_cache=True,past_key_values=cache,logits_to_keep=1)
   cache=out.past_key_values;assert cache is not None and cache.get_seq_length()==total
   logits=out.logits[0,-1].detach().float().cpu();assert tuple(logits.shape)==(NATIVE,);all_logits.append(logits)
   steps.append(dict(target_position=j,input_tokens=len(tokens),attention_mask_tokens=total,cache_tokens=int(cache.get_seq_length()),expected_linear_attention_path='full_chunk_prefill' if j==0 else 'single_token_recurrent'))
 rope_check(model);return torch.stack(all_logits),dict(mode='eval',use_cache=True,steps=steps)
def logits_summary(torch,logits,target,family,decision):
 assert torch.isfinite(logits).all();s=logits[:,:SEMANTIC];z=torch.logsumexp(s,dim=-1);zn=torch.logsumexp(logits,dim=-1)
 idx=torch.arange(len(target));gold=torch.tensor(target);goldlogit=logits[idx,gold];p=s.softmax(-1)
 grad=p.clone();grad[idx,gold]-=1;grad/=len(target)
 pos=[]
 for j in range(len(target)):
  pos.append(dict(position=j,target_id=target[j],target_logit=float(goldlogit[j]),semantic_nll=float(z[j]-goldlogit[j]),native_nll=float(zn[j]-goldlogit[j]),
   semantic_top1=int(s[j].argmax()),native_top1=int(logits[j].argmax()),padding_tail_mass=float(torch.exp(torch.logsumexp(logits[j,SEMANTIC:],0)-zn[j])),
   mean_CE_logit_gradient_L1=float(grad[j].abs().sum()),mean_CE_logit_gradient_L2=float(grad[j].norm())))
 choice_ids=[family[a][decision] for a in 'ABCD'];cl=logits[decision,choice_ids];probs=cl.softmax(0);answer='ABCD'[choice_ids.index(target[decision])];correct='ABCD'.index(answer)
 wrong=[i for i in range(4) if i!=correct]
 return dict(mean_semantic_CE=float((z-goldlogit).mean()),per_position=pos,decision=dict(position=decision,correct_answer=answer,conditional_top1='ABCD'[int(cl.argmax())],
  logits=dict(zip('ABCD',map(float,cl))),conditional_four_answer_probabilities=dict(zip('ABCD',map(float,probs))),correct_margin=float(cl[correct]-cl[wrong].max()),
  four_answer_semantic_mass=float(torch.exp(torch.logsumexp(cl,0)-z[decision])),semantic_top1=int(s[decision].argmax()),native_top1=int(logits[decision].argmax())),
  gradient_scope='Analytic derivative of mean semantic CE with respect to logits only; not measured LoRA parameter gradients or attribution.')
def compare_logits(torch,a,b):
 d=a-b
 return dict(per_position=[dict(position=j,max_abs=float(d[j].abs().max()),mean_abs=float(d[j].abs().mean()),rms=float(d[j].square().mean().sqrt()),native_top1_equal=bool(a[j].argmax()==b[j].argmax()),semantic_top1_equal=bool(a[j,:SEMANTIC].argmax()==b[j,:SEMANTIC].argmax())) for j in range(len(a))])

def evaluate(key):
 assert key in units();unit=O/key;p=verify(unit);protocol=load(O/'protocol.json');s=p['source_checkpoint'];m,w,_=machinery();decode=decoder()
 source=R/s['source_root'];folder=R/s['checkpoint'];save(unit/'evaluation-started.json',dict(PID=os.getpid(),epoch=time.time(),new_updates=0))
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch,tok,model,head=w.initialize(unit,folder);w.m.lib.activate(model,head,'actor')
  opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
  cm=w.m.lib.restore(torch,model,head,dict(actor=opt),source,folder)
  roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};oh=w.m.lib.nested_hash(opt.state_dict());rh=w.m.lib.nested_hash(rng_state(torch))
  assert roles==s['role_hashes']==cm['role_hashes'] and oh==s['optimizer_hashes']['actor'] and cm['actual_updates']==s['step']
  assert rh==w.m.lib.nested_hash(torch.load(folder/'rng.pt',map_location='cpu',weights_only=True))
  assert all(float(x.p)==0 for x in model.modules() if isinstance(x,torch.nn.Dropout));rope_check(model)
  names={id(v):name for name,v in model.named_parameters()};trainable=[name for name,v in model.named_parameters() if v.requires_grad]
  optimizer_names=[[names[id(v)] for v in group['params']] for group in opt.param_groups]
  assert len(trainable)==48 and all('lora_' in name and '.actor.' in name for name in trainable)
  assert [name for group in optimizer_names for name in group]==trainable and not any(v.requires_grad for v in head.parameters())
  family,decision=target_family(tok);assert family==protocol['candidate_target_ids'] and decision==protocol['decision_position']
  save(unit/'reload.json',dict(status='EXACT_ROLES_OPTIMIZER_TORCH_RNG',roles=roles,optimizer=oh,torch_rng=rh,source_manifest=bind(folder/'manifest.json'),actual_source_updates=s['step'],new_updates=0,
   torch_version=torch.__version__,gradient_checkpointing=model.is_gradient_checkpointing,all_dropout_zero=True,trainable_parameter_names=trainable,optimizer_parameter_names=optimizer_names,head_requires_grad=False,attn_implementation=model.get_base_model().config.text_config._attn_implementation))
  prefix=load(O/'fixed8.local.json');rows=load(unit/'queries.local.json');start=time.monotonic()
  for i,q in enumerate(rows):
   check();save(unit/'raw'/f'{i:04d}.started.json',dict(id=q['id'],epoch=time.time()));rope_start=reset_text_rope(model);assert eval_prompt(tok,prefix,q['question'])==q['prompt_ids']
   prompt=q['prompt_ids'];target=q['target_ids'];logits={};traces={}
   logits['train_full'],traces['train_full']=forward_full(torch,model,prompt,target,True)
   logits['eval_full'],traces['eval_full']=forward_full(torch,model,prompt,target,False)
   logits['eval_cached'],traces['eval_cached']=forward_cached(torch,model,prompt,target)
   diagnostic=dict(id=q['id'],source_id=q['source_id'],text_rope_at_case_start=rope_start,first_training_step=q['first_training_step'],branches={k:dict(trace=traces[k],**logits_summary(torch,v,target,family,decision)) for k,v in logits.items()},
    comparisons={'train_full_vs_eval_full':compare_logits(torch,logits['train_full'],logits['eval_full']),'eval_full_vs_eval_cached':compare_logits(torch,logits['eval_full'],logits['eval_cached'])})
   model.eval();rope_check(model);raw,observed=free_decode_observed(torch,tok,model,prompt,decode,family,decision)
   # Preserve real raw before checking the generated ephemeral position state.
   after_generate=text_rope_snapshot(model)
   save(unit/'raw'/f'{i:04d}.json',dict(id=q['id'],source_id=q['source_id'],cap=128,text_rope_after_generation=after_generate,observed_generation_prefix=observed,**raw))
   diagnostic['text_rope_after_generation_reset']=reset_text_rope(model)
   same_prefix=raw['response_ids'][:decision]==target[:decision] and len(raw['response_ids'])>decision
   comparison=dict(same_emitted_decision_prefix=same_prefix,actual_generation_inference_mode=True,manual_cached_no_grad_mode=True,observed_generation_prefix=observed)
   if same_prefix:
    actual=observed[decision]['candidate_logits'];manual=diagnostic['branches']['eval_cached']['decision']['logits']
    delta={a:actual[a]-manual[a] for a in 'ABCD'}
    comparison.update(actual_candidate_logits=actual,manual_cached_candidate_logits=manual,actual_minus_manual=delta,max_abs_choice_logit_difference=max(abs(x) for x in delta.values()),
     actual_conditional_top1=max('ABCD',key=lambda a:actual[a]),manual_conditional_top1=max('ABCD',key=lambda a:manual[a]),
     actual_native_top1=observed[decision]['native_top1'],manual_native_top1=diagnostic['branches']['eval_cached']['decision']['native_top1'])
   diagnostic['actual_generation_vs_manual_cached']=comparison
   save(unit/'diagnostics'/f'{i:04d}.json',diagnostic);del logits
   assert all(v.grad is None for v in model.parameters()),'Probe must not create gradients'
   assert w.m.lib.nested_hash(rng_state(torch))==rh,'Deterministic no-update inference changed restored torch RNG'
   live(stage='DIAGNOSIS',model=key,closed=i+1,total=COUNT);print('CASE',key,i+1,COUNT,round(time.monotonic()-start,1),flush=True)
  assert roles=={r:w.m.lib.role_hash(model,r) for r in roles} and oh==w.m.lib.nested_hash(opt.state_dict())
  assert rh==w.m.lib.nested_hash(rng_state(torch));verify(unit)
  save(unit/'evaluation-terminal.json',dict(status='ALL18_DIAGNOSTICS_AND_RAW_CLOSED',raw=COUNT,diagnostics=COUNT,new_updates=0,API=0,roles=roles,optimizer=oh,torch_rng=rh,roles_unchanged=True,optimizer_unchanged=True,torch_rng_unchanged=True))

def score():
 p=verify();pins=[]
 for key in units():
  unit=O/key;t=load(unit/'evaluation-terminal.json');assert t['status']=='ALL18_DIAGNOSTICS_AND_RAW_CLOSED' and t['raw']==t['diagnostics']==COUNT and t['new_updates']==t['API']==0
  assert t['roles_unchanged'] and t['optimizer_unchanged'] and t['torch_rng_unchanged']
  assert len(list((unit/'raw').glob('[0-9][0-9][0-9][0-9].json')))==COUNT
  for i,q in enumerate(load(unit/'queries.local.json')):
   raw=load(unit/'raw'/f'{i:04d}.json');d=load(unit/'diagnostics'/f'{i:04d}.json');assert raw['id']==q['id']==d['id']==load(unit/'raw'/f'{i:04d}.started.json')['id']
   pins.extend([bind(unit/'raw'/f'{i:04d}.json'),bind(unit/'diagnostics'/f'{i:04d}.json')])
 assert len(pins)==2*TOTAL;save(O/'all72-raw-diagnostics-frozen-before-score.json',dict(bindings=pins,epoch=time.time()))
 sys.path[:0]=[str(R/'scripts'),str(R/'src')];import eval_ust_sg_qwen_modes_20261024 as parser
 result={}
 for key in units():
  unit=O/key;grades=[]
  for i,q in enumerate(load(unit/'queries.local.json')):
   raw=load(unit/'raw'/f'{i:04d}.json');old=load(R/q['old_raw']['path']);assert bind(R/q['old_raw']['path'])==q['old_raw'];d=load(unit/'diagnostics'/f'{i:04d}.json')
   g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);og=parser.classify(old['raw_generated'],'off','ust',q['metadata'],old['finish_reason'])
   grades.append(dict(id=q['id'],source_id=q['source_id'],answer=q['answer'],prediction=g['prediction'],valid=bool(g['valid']),correct=bool(g['valid'] and g['prediction']==q['answer']),capped=raw['finish_reason']=='length',old_prediction=og['prediction'],old_correct=bool(og['valid'] and og['prediction']==q['answer']),new_free_decision_prefix_equal=raw['response_ids'][:p['decision_position']]==q['target_ids'][:p['decision_position']],raw_equal_old=raw['raw_generated']==old['raw_generated'],response_ids_equal_old=raw['response_ids']==old['response_ids'],
    actual_generation_vs_manual_cached=d['actual_generation_vs_manual_cached'],conditional_predictions={mode:d['branches'][mode]['decision']['conditional_top1'] for mode in MODES}))
  save(unit/'graded.local.json',grades)
  result[key]=dict(n=COUNT,free_correct=sum(x['correct'] for x in grades),free_valid=sum(x['valid'] for x in grades),capped=sum(x['capped'] for x in grades),old_free_correct=sum(x['old_correct'] for x in grades),raw_equal_old=sum(x['raw_equal_old'] for x in grades),response_ids_equal_old=sum(x['response_ids_equal_old'] for x in grades),free_predictions=dict(collections.Counter(str(x['prediction']) for x in grades)),
   teacherforced_conditional_correct={mode:sum(x['conditional_predictions'][mode]==x['answer'] for x in grades) for mode in MODES},
   actual_generation_comparable=sum(x['actual_generation_vs_manual_cached']['same_emitted_decision_prefix'] for x in grades),actual_generate_manual_choice_disagreements=sum(x['actual_generation_vs_manual_cached']['actual_conditional_top1']!=x['actual_generation_vs_manual_cached']['manual_conditional_top1'] for x in grades if x['actual_generation_vs_manual_cached']['same_emitted_decision_prefix']),actual_generate_manual_max_abs_choice_difference=max((x['actual_generation_vs_manual_cached']['max_abs_choice_logit_difference'] for x in grades if x['actual_generation_vs_manual_cached']['same_emitted_decision_prefix']),default=None),conditional_mode_flips=sum(x['conditional_predictions']['train_full']!=x['conditional_predictions']['eval_full'] for x in grades),conditional_cache_flips=sum(x['conditional_predictions']['eval_full']!=x['conditional_predictions']['eval_cached'] for x in grades))
 save(O/'results.json',dict(status='COMPLETE',scores=result,source_cases=18,total_raw=72,total_new_updates=0,API=0,scope=SCOPE))
 save(O/'terminal.json',dict(status='COMPLETE',results=bind(O/'results.json'),epoch=time.time()));live(stage='COMPLETE');print(json.dumps(result),flush=True)
def controller():
 verify();save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time()))
 for key in units():
  with (O/(key+'.log')).open('x') as f:subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)),'evaluate','--unit',key],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1500)
 score()
if __name__=='__main__':
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','evaluate','score','controller','verify'));p.add_argument('--unit',choices=units());a=p.parse_args()
 try:
  if a.action=='evaluate':evaluate(a.unit)
  elif a.action=='verify':verify(O/a.unit if a.unit else None);print('VERIFY PASS')
  else:globals()[a.action]()
 except BaseException as e:
  if O.exists():save(O/('failure-'+str(time.time_ns())+'.json'),dict(type=type(e).__name__,error=str(e),epoch=time.time()))
  raise
