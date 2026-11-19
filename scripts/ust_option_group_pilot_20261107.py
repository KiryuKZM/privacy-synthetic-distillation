"""Matched three-group, two-pass batch4 understanding pilot; GPU separately gated.

New explicit weak-order instruction is used for all272 variants. FreshINIT308,
3*(own68 at68 + variants272/Dev36 at136) =1436 new raw outputs before scoring.
"""
from pathlib import Path
import argparse,collections,copy,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time
R=Path(__file__).resolve().parents[1];N=R/'runs/2026-11-06-ust-retention-information-v1';O=N/'option-cycle-pilot-v1';DATA=N/'option-cycle-grouping-weak-v3'
PREVIOUS=R/'runs/2026-11-06-ust-autonomous-repair-v1';SOURCE=PREVIOUS/'position-training-v2/synthetic-balanced200'
OLD=R/'runs/2026-10-25-qwen-transfer-repair/seqkd200/USTBench/2b';INIT=OLD/'work/checkpoints/step0000'
LR=N/'learning-rate-control-v1';CONT=N/'continuation-control-v1';BACKWARD=N/'backward-backend-probe-v3';DETERMINISTIC=N/'deterministic-backward-probe-v2'
DEPENDENCIES=(LR,CONT,BACKWARD,DETERMINISTIC)
HELPER=R/'scripts/ust_retention_forward_probe_v2_20261106.py';BATCH=R/'scripts/ust_option_group_batch_update_20261107.py';ADAPTER=R/'scripts/ust_option_cycle_weak_data_v2_20261106.py'
ARMS=('same-source-four','random-four','distinct-source-balanced-four');TOTAL=1436;STOP=False
NUMERICAL_PROFILE='strict-deterministic-existing-FLA-v1'
NUMERICAL_DECISION=N/'option-pilot-numerical-profile-decision.json'
NUMERICAL_HELPER=R/'scripts/ust_retention_backward_backend_probe_v3_20261106.py'
DATA_REVIEW=N/'independent-audit/option-cycle-weak-v3-review.json'
SCOPE=('Matched two-pass grouping diagnostic on explicitly clarified displayed-value non-increasing ranking. '
 'Same272 examples,136Adam calls and544microexamples perarm; only frozen batch membership/order differs. '
 'Different training exposure/task mix/wording/numerical profile versus oldmixed200 means cross-study changes cannot be attributed to grouping. '
 'Own68/272 are68 repeated source tasks, Dev36 repeats12 sources, no independent final/replacement or outlier utility claim. '
 'Old weak-order verifier assumptions and old results remain unchanged; new text gets a freshINIT baseline.')
def load(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(2**20),b''):h.update(z)
 return dict(path=str(p.relative_to(R)) if p.is_relative_to(R) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def check():
 if STOP or (N/'STOP').exists() or (O/'STOP').exists():raise InterruptedError('Own option-group pilot STOP')
def stop(*_):
 global STOP;STOP=True
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def helper():return module(HELPER,'option_group_same_helpers')
def machinery():
 h=helper();h.O=O;h.verify=verify;h.check=check;h.live=live;return h.machinery()
def eval_units():
 units={'freshINIT':dict(arm=None,step=0,count=308)}
 for arm in ARMS:
  for step,count in ((68,68),(136,308)):units[f'{arm}-step{step:04d}']=dict(arm=arm,step=step,count=count)
 assert sum(x['count'] for x in units.values())==TOTAL;return units

def numerical_decision():
 decision=load(NUMERICAL_DECISION)
 assert decision['status']=='APPROVED_FOR_FIXED_OPTION_GROUP_PILOT' and decision['paired_arms_same_profile'] and decision['fresh_INIT_and_all_evaluations_same_profile']
 assert decision['required_launcher_environment']==dict(CUBLAS_WORKSPACE_CONFIG=':4096:8')
 assert decision['required_flags']==dict(deterministic_algorithms=True,deterministic_warn_only=False,cudnn_deterministic=True)
 assert decision['no_SDPA_backend_override'] and decision['no_precision_change'] and decision['learning_rate']==1e-4
 for key in ('closed_qualification','baseline_provenance_correction','unchanged_setting_reference'):
  pin=decision[key];assert bind(R/pin['path'])==pin
 assert str(load(R/decision['closed_qualification']['path'])['status']).startswith('PASS')
 return decision

def numerical_helpers():return module(NUMERICAL_HELPER,'option_group_numeric_helpers')
def apply_numerical_profile():
 decision=numerical_decision()
 import torch
 assert not torch.cuda.is_initialized(),'Numerical profile must be selected before CUDA initialization'
 os.environ.update(decision['required_launcher_environment'])
 torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.deterministic=True
 flags=numerical_helpers().numerical_flags(torch)
 reference=load(R/decision['unchanged_setting_reference']['path'])
 assert flags==reference['numerical_flags'],'No silent backend/precision/environment change'
 return flags

def actual_numerical_profile(torch,model):
 decision=numerical_decision();reference=load(R/decision['unchanged_setting_reference']['path']);nh=numerical_helpers()
 from transformers.models.qwen3_5 import modeling_qwen3_5 as q
 actual=dict(numerical_flags=nh.numerical_flags(torch),gradient_checkpointing=bool(model.is_gradient_checkpointing),
  base_dtype=str(model.get_base_model().dtype),adapter_dtypes=sorted({str(v.dtype) for v in model.parameters() if v.requires_grad}),
  attn_implementation=model.get_base_model().config.text_config._attn_implementation,actual_linear_dispatch=nh.reference_function(q)[1])
 assert all(actual[key]==reference[key] for key in actual),'Runtime differs from the approved fixed FLA profile'
 return dict(profile=NUMERICAL_PROFILE,decision=bind(NUMERICAL_DECISION),torch_version=torch.__version__,CUDA_build=torch.version.cuda,**actual)

def prepare():
 decision=numerical_decision()
 assert not O.exists(),'Immutable new namespace';check();dp=load(DATA/'data-protocol.json');variants=load(DATA/'variants.local.json')
 assert str(load(DATA_REVIEW)['status']).startswith('PASS')
 import torch
 assert not torch.cuda.is_initialized()
 assert dp['status']=='ALL_THREE_GROUPINGS_FROZEN_CPU_ONLY' and dp['variant_count']==272 and dp['source_questions']==68
 term=load(DATA/'terminal.json');assert term['status']=='COMPLETE_CPU_DATA_ONLY' and term['data_protocol']==bind(DATA/'data-protocol.json')
 for pin in dp['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 assert len(variants)==272 and len({v['id'] for v in variants})==272 and len({v['source_id'] for v in variants})==68
 vmap={v['id']:v for v in variants};source_groups={sid:[v for v in variants if v['source_id']==sid] for sid in {v['source_id'] for v in variants}}
 assert all(len(g)==4 and {v['variant_index'] for v in g}==set(range(4)) and {v['answer'] for v in g}==set('ABCD') for g in source_groups.values())
 adapter=module(ADAPTER,'option_group_fixed_weak_adapter');h=helper();cm,cpins=h.checkpoint_pins(OLD,INIT,0);p0=load(SOURCE/'protocol.json');prefix=load(PREVIOUS/'position-training-v2/fixed8.local.json')
 assert prefix==load(DATA/'fixed8.local.json')==load(LR/'fixed8.local.json'),'No silent fixed8 prefix changes'
 sys.path[:0]=[str(R/'scripts'),str(R/'src')]
 import probe_overnight_train_fit_20261025 as packing
 from privacy_synth.ust_task_contract import certify_native_ranking_reference
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);family,_=h.target_family(tok)
 full=[];answers={};tokens={};quals=[]
 for v in variants:
  assert v['metadata']['ability']=='st_understanding' and v['target']=='<Answer>'+v['answer']+'</Answer>'
  cert=adapter.certify_explicit_weak_reference(v['question'],v['answer']);assert cert['trusted_gold']
  prompt,target=packing.packing(tok,v,dict(text=v['target']),'matched8',prefix);assert prompt==h.eval_prompt(tok,prefix,v['question']) and target==family[v['answer']] and len(prompt)+128<=16384
  q=dict(id=v['id'],source_id=v['source_id'],question=v['question'],metadata=v['metadata'],variant='training-option-cycle',variant_index=v['variant_index'],old_to_new_letter=v['old_to_new_letter'],source_index=v['source_index'],cohort='new-explicit-weak-understanding-train',cap=128,metric=cert['metric'])
  full.append(q);answers[q['id']]=dict(answer=v['answer'],status='explicit displayed weak-order uniquely compatible option');tokens[q['id']]=dict(prompt_ids=prompt,target_ids=target)
  quals.append(dict(id=q['id'],certificate=cert,pack_prompt_equals_eval=True,prompt_tokens=len(prompt),target_tokens=7))
 own=[q for q in full if q['variant_index']==0];assert len(own)==68
 oldqs=load(LR/'fresh-init0/queries.local.json');oldaa=load(LR/'fresh-init0/answers.local.json')
 dev=[q for q in oldqs if q['variant'] in ('canonical','both','numeric-counterfactual')];assert len(dev)==36 and len({q['source_id'] for q in dev})==12
 for q0 in dev:
  q=copy.deepcopy(q0);answer=oldaa[q['id']]['answer'];cert=certify_native_ranking_reference(q['question'],answer);assert cert['trusted_gold'] and q['metadata']['ability']=='st_understanding'
  ids=h.eval_prompt(tok,prefix,q['question']);assert len(ids)+128<=16384;q['metric']=cert['metric'];full.append(q);answers[q['id']]=dict(answer=answer,status='legacy displayed weak-order compatible Dev reference; old question unchanged')
  tokens[q['id']]=dict(prompt_ids=ids,target_ids=family[answer]);quals.append(dict(id=q['id'],certificate=cert,legacy_Dev_semantic_boundary=True,prompt_tokens=len(ids)))
 assert len(full)==len(answers)==308 and len({q['id'] for q in full})==308
 batches={}
 for arm in ARMS:
  ss=load(DATA/arm/'batches.local.json');assert len(ss)==68 and all(len(batch)==4 and len(set(batch))==4 for batch in ss)
  assert collections.Counter(i for batch in ss for i in batch)==collections.Counter(vmap.keys())
  for batch in ss:
   vv=[vmap[i] for i in batch]
   if arm=='same-source-four':assert len({v['source_id'] for v in vv})==1 and {v['answer'] for v in vv}==set('ABCD')
   if arm=='distinct-source-balanced-four':assert len({v['source_id'] for v in vv})==4 and {v['answer'] for v in vv}==set('ABCD')
  batches[arm]=ss
 O.mkdir(parents=True);save(O/'variants.local.json',variants);save(O/'queries-full.local.json',full);save(O/'queries-own68.local.json',own);save(O/'answers.local.json',answers);save(O/'tokens.local.json',tokens);save(O/'fixed8.local.json',prefix);save(O/'qualification.local.json',quals)
 pins=[bind(__file__),bind(BATCH),bind(ADAPTER),bind(HELPER),bind(h.CAUSAL),bind(h.DECODER),bind(NUMERICAL_HELPER),bind(NUMERICAL_DECISION),bind(DATA_REVIEW),bind(SOURCE/'protocol.json'),bind(PREVIOUS/'position-training-v2/fixed8.local.json'),bind(R/'tests/test_ust_option_group_batch_update.py'),bind(DATA/'data-protocol.json'),bind(DATA/'variants.local.json'),bind(N/'budget-continuation.json'),bind(LR/'fresh-init0/queries.local.json'),bind(LR/'fresh-init0/answers.local.json')]+cpins
 for key in ('closed_qualification','baseline_provenance_correction','unchanged_setting_reference'):pins.append(decision[key])
 for folder in DEPENDENCIES:
  terminal=load(folder/'terminal.json');assert terminal['status']=='COMPLETE' and bind(R/terminal['results']['path'])==terminal['results'];pins.extend([bind(folder/'terminal.json'),bind(folder/'protocol.json'),terminal['results']])
 for arm,ss in batches.items():
  schedule=[dict(step=epoch*68+i+1,epoch=epoch+1,batch_index=i,variant_ids=batch) for epoch in range(2) for i,batch in enumerate(ss)]
  assert len(schedule)==136 and [r['variant_ids'] for r in schedule[:68]]==[r['variant_ids'] for r in schedule[68:]]
  save(O/arm/'schedule.local.json',schedule);pins.extend([bind(DATA/arm/'batches.local.json'),bind(O/arm/'schedule.local.json')])
 for name in ('variants.local.json','queries-full.local.json','queries-own68.local.json','answers.local.json','tokens.local.json','fixed8.local.json','qualification.local.json'):pins.append(bind(O/name))
 # Preserve the data adapter's own qualification outputs and all current files.
 for p in DATA.rglob('*'):
  if p.is_file():pins.append(bind(p))
 machinery();import eval_ust_sg_qwen_modes_20261024,run_llm_task_sweep_20261023,build_llm_task_sweep_qa_20261023
 for m in tuple(sys.modules.values()):
  path=getattr(m,'__file__',None)
  if path:
   path=Path(path).resolve()
   if path.is_file() and path.is_relative_to(R) and path.suffix=='.py' and '.venv' not in path.parts and 'vendor' not in path.parts:pins.append(bind(path))
 for rel in ('models/qwen3_5/modeling_qwen3_5.py','integrations/sdpa_attention.py','integrations/hub_kernels.py','modeling_layers.py','cache_utils.py','generation/utils.py'):
  paths=list((R/'.venv/lib').glob('python*/site-packages/transformers/'+rel));assert len(paths)==1;pins.append(bind(paths[0]))
 from transformers.models.qwen3_5 import modeling_qwen3_5 as q
 numerical=module(NUMERICAL_HELPER,'option_group_numerical_helpers');_,implementation=numerical.reference_function(q)
 save(O/'actual-backend-before-GPU.json',implementation);pins.append(bind(O/'actual-backend-before-GPU.json'))
 import fla
 pins.extend(bind(path) for path in Path(fla.__file__).resolve().parent.rglob('*.py'))
 for item in implementation['baseline_implementation_chain']:pins.append(bind(item['file']))
 for rel in ('utils/checkpoint.py','nn/functional.py','nn/attention/__init__.py','backends/cuda/__init__.py','backends/cudnn/__init__.py','optim/adamw.py','optim/adam.py','optim/optimizer.py'):
  pins.append(bind(Path(torch.__file__).resolve().parent/rel))
 pins.append(bind(torch.__file__));assert not torch.cuda.is_initialized()
 pins += [bind(p) for p in Path(p0['base']).iterdir() if p.is_file()]+[bind(R/'pyproject.toml'),bind(R/'uv.lock')];pins=list({x['path']:x for x in pins}.values())
 save(O/'protocol.json',dict(source=bind(__file__),bindings=pins,scope=SCOPE,arms=ARMS,total_raw=TOTAL,new_models=3,new_optimizer_calls=408,micro_forward_backward_calls=1632,epochs=2,batches_per_epoch=68,per_arm_optimizer_calls=136,per_arm_microexamples=544,
  variants=272,source_cases=68,Dev_source_cases=12,Dev_variants=36,freshINIT_new_raw=308,eval_units=eval_units(),saved_steps=[34,68,102,136],evaluated_steps=[68,136],no_old_INIT_raw_reused=True,only_grouping_differs_between_arms=True,
  source_INIT=bind(INIT/'manifest.json'),numerical_profile=NUMERICAL_PROFILE,numerical_decision=bind(NUMERICAL_DECISION),API=0,Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False,backend_gate='all earlier LR/continuation/backend results closed plus explicit engineering-approved root authorization'))
 protocols=[bind(O/'protocol.json')]
 for arm in ARMS:
  p=dict(p0);p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],arm=arm,steps=136,lr=1e-4,fixed_checkpoint=136,seed=20261024,method='explicit-weak-understanding-batch4-CE',ability_order=['st_understanding']*136,checkpoint_steps=[34,68,102,136],
   scope=SCOPE,evaluation_use=SCOPE,total_raw=TOTAL,queries_each=376,protected_real_training_gold_in_loss=False,synthetic_reference_in_loss=True,uncertain_planning_reflection_references=False,Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False,teacher_API=0)
  save(O/arm/'protocol.json',p);protocols.append(bind(O/arm/'protocol.json'))
 for key,u in eval_units().items():
  p=dict(p0);p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],model_id=key,steps=0,training=0,fixed_checkpoint=u['step'],scope=SCOPE,evaluation_use=SCOPE,total_raw=TOTAL,queries_each=u['count'],method='new-text-fixed-checkpoint-greedy-probe',ability_order=[],checkpoint_steps=[],
   protected_real_training_gold_in_loss=False,synthetic_reference_in_loss=False,Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False,teacher_API=0)
  save(O/key/'protocol.json',p);protocols.append(bind(O/key/'protocol.json'))
 save(O/'execution-bindings.json',dict(source=bind(__file__),protocols=protocols));verify();print('CPU_PREPARED3x136Adam1632microFB1436newraw;GPU_NOT_STARTED',flush=True)

def verify(unit=None):
 check();e=load(O/'execution-bindings.json');assert e['source']==bind(__file__)
 for pin in e['protocols']:assert bind(R/pin['path'])==pin,pin['path']
 p=load((Path(unit) if unit is not None else O)/'protocol.json');assert p['source']==bind(__file__) and p['total_raw']==TOTAL
 for pin in p['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 if unit is not None:assert Path(unit).parent==O and Path(unit).name in set(ARMS)|set(eval_units())
 return p

def gate(require_dependencies=True):
 verify();a=load(O/'launch-authorization.json');assert a['status']=='AUTHORIZED_FIXED_OPTION_GROUP_PILOT' and a['protocol']==bind(O/'protocol.json') and a['engineering_gate_pass'] is True
 assert a['backend_review']==numerical_decision()['closed_qualification']
 for key in ('independent_review','backend_review'):
  pin=a[key];assert (R/pin['path']).resolve().is_relative_to(N.resolve()) and bind(R/pin['path'])==pin and str(load(R/pin['path'])['status']).startswith('PASS')
 if require_dependencies:
  for folder in DEPENDENCIES:
   t=load(folder/'terminal.json');assert t['status']=='COMPLETE' and bind(R/t['results']['path'])==t['results']
 return a

def fit(arm):
 assert arm in ARMS;gate();unit=O/arm;verify(unit);flags=apply_numerical_profile();m,w,_=machinery();batch=module(BATCH,'option_group_batch4');h=helper()
 save(unit/'fit-started.json',dict(PID=os.getpid(),epoch=time.time(),new_updates=136,microexamples=544))
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch,tok,model,head=w.initialize(unit,INIT);w.m.lib.activate(model,head,'actor');params=[v for v in model.parameters() if v.requires_grad]
  names={id(v):name for name,v in model.named_parameters()};assert len(params)==48 and all('.actor.' in names[id(v)] and 'lora_' in names[id(v)] and v.dtype==torch.float32 for v in params)
  opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0,foreach=False);cm=w.m.lib.restore(torch,model,head,dict(actor=opt),OLD,INIT);assert not opt.state and cm['actual_updates']==0
  roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};rng=lambda:dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all());rh=w.m.lib.nested_hash(rng())
  assert roles==cm['role_hashes'] and w.m.lib.nested_hash(opt.state_dict())==cm['optimizer_hashes']['actor'] and all(g['lr']==1e-4 for g in opt.param_groups)
  profile=actual_numerical_profile(torch,model);assert profile['numerical_flags']==flags
  assert rh==w.m.lib.nested_hash(torch.load(INIT/'rng.pt',map_location='cpu',weights_only=True)) and all(float(x.p)==0 for x in model.modules() if isinstance(x,torch.nn.Dropout))
  save(unit/'initialization-verified.json',dict(status='EXACT_INIT_ROLES_EMPTY_ADAM_RNG',roles=roles,optimizer=cm['optimizer_hashes']['actor'],RNG=rh,parameter_names=[names[id(v)] for v in params],numerical_profile=profile))
  tokens=load(O/'tokens.local.json');variants={v['id']:v for v in load(O/'variants.local.json')};prefix=load(O/'fixed8.local.json');start=time.monotonic();parameter_change_batches=0
  for row in load(unit/'schedule.local.json'):
   step=row['step'];check();save(unit/'updates'/f'{step:04d}.started.json',dict(step=step,epoch=row['epoch'],variant_ids=row['variant_ids'],time=time.time()));h.rope_check(model)
   traces=[]
   for vid in row['variant_ids']:
    ids=h.eval_prompt(tok,prefix,variants[vid]['question']);assert ids==tokens[vid]['prompt_ids'];traces.append(dict(id=vid,prompt_ids=ids,oracle_target_ids=tokens[vid]['target_ids']))
   assert batch.optimizer_steps(opt,params)==[step-1]*48
   result=batch.batch4_update(torch,model,head,opt,traces,w.m.lib,check);parameter_change_batches+=int(result['parameter_changed']);assert result['state_steps_after']==[step]*48 and rh==w.m.lib.nested_hash(rng())
   result.update(step=step,epoch=row['epoch'],batch_index=row['batch_index'],variant_ids=row['variant_ids'],source_ids=[variants[x]['source_id'] for x in row['variant_ids']],answers=[variants[x]['answer'] for x in row['variant_ids']],RNG=rh)
   save(unit/'updates'/f'{step:04d}.json',result)
   if step%17==0:live(stage='TRAINING',arm=arm,step=step,total=136);print('FIT',arm,step,round(time.monotonic()-start,1),flush=True)
   if step in (34,68,102,136):
    folder=unit/'checkpoints'/f'step{step:04d}';w.m.native.tiny.SAVE_CHECKPOINT(unit,folder,torch,model,head,dict(actor=opt),'OPTION_GROUP_BATCH4',step,actual_updates=step);w.m.lib.verify_checkpoint(unit,folder)
  assert profile==actual_numerical_profile(torch,model)
  final={r:w.m.lib.role_hash(model,r) for r in roles};assert all(final[r]==roles[r] for r in roles if r!='actor')
  save(unit/'fit-terminal.json',dict(status='FIXED136_COMPLETE',actual_updates=136,optimizer_calls=136,parameter_change_batches=parameter_change_batches,actual_updates_semantics='optimizer calls including zero aggregate gradient steps',microexamples=544,roles=final,optimizer=w.m.lib.nested_hash(opt.state_dict()),RNG=rh,seconds=time.monotonic()-start,checkpoint=bind(unit/'checkpoints/step0136/manifest.json'),API=0))
def freeze_models():
 gate();h=helper();specs={};pins=[]
 for arm in ARMS:
  t=load(O/arm/'fit-terminal.json');assert t['status']=='FIXED136_COMPLETE' and t['actual_updates']==136 and t['microexamples']==544;pins.extend([bind(O/arm/'fit-terminal.json'),bind(O/arm/'initialization-verified.json')])
 for key,u in eval_units().items():
  source=O/u['arm'] if u['arm'] else OLD;folder=source/'checkpoints'/f"step{u['step']:04d}" if u['arm'] else INIT;cm,pp=h.checkpoint_pins(source,folder,u['step']);pins+=pp
  specs[key]=dict(source=str(source.relative_to(R)),folder=str(folder.relative_to(R)),step=u['step'],roles=cm['role_hashes'],optimizer=cm['optimizer_hashes']['actor'])
 save(O/'models-frozen.json',dict(status='ALL3FITS_AND7_CHECKPOINTS_FROZEN',specs=specs,bindings=list({x['path']:x for x in pins}.values()),epoch=time.time()))
def frozen_models():
 v=load(O/'models-frozen.json');assert v['status']=='ALL3FITS_AND7_CHECKPOINTS_FROZEN' and set(v['specs'])==set(eval_units())
 for pin in v['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 return v['specs']
def evaluate(key):
 assert key in eval_units();gate();unit=O/key;p=verify(unit);spec=frozen_models()[key];flags=apply_numerical_profile();m,w,_=machinery();h=helper();decode=h.decoder();save(unit/'evaluation-started.json',dict(PID=os.getpid(),epoch=time.time(),model=spec))
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);folder=R/spec['folder'];torch,tok,model,head=w.initialize(unit,folder);w.m.lib.activate(model,head,'actor');model.eval();opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
  cm=w.m.lib.restore(torch,model,head,dict(actor=opt),R/spec['source'],folder);roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};oh=w.m.lib.nested_hash(opt.state_dict());rng=lambda:dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all());rh=w.m.lib.nested_hash(rng())
  assert roles==spec['roles'] and oh==spec['optimizer'] and cm['actual_updates']==spec['step'] and rh==w.m.lib.nested_hash(torch.load(folder/'rng.pt',map_location='cpu',weights_only=True))
  profile=actual_numerical_profile(torch,model);assert profile['numerical_flags']==flags
  save(unit/'reload.json',dict(status='EXACT_ROLES_ADAM_RNG',source=spec,roles=roles,optimizer=oh,RNG=rh,new_updates=0,numerical_profile=profile))
  qq=load(O/('queries-own68.local.json' if spec['step']==68 else 'queries-full.local.json'));tokens=load(O/'tokens.local.json');prefix=load(O/'fixed8.local.json');assert len(qq)==p['queries_each']
  for i,q in enumerate(qq):
   check();save(unit/'raw'/f'{i:04d}.started.json',dict(id=q['id'],epoch=time.time()));h.reset_text_rope(model);ids=h.eval_prompt(tok,prefix,q['question']);assert ids==tokens[q['id']]['prompt_ids'];raw=decode(torch,tok,model,ids)
   save(unit/'raw'/f'{i:04d}.json',dict(id=q['id'],source_id=q['source_id'],cap=128,variant=q['variant'],**raw));h.reset_text_rope(model)
   if (i+1)%34==0:live(stage='EVALUATION',unit=key,closed=i+1,total=len(qq))
  assert profile==actual_numerical_profile(torch,model)
  assert roles=={r:w.m.lib.role_hash(model,r) for r in roles} and oh==w.m.lib.nested_hash(opt.state_dict()) and rh==w.m.lib.nested_hash(rng());verify(unit)
  save(unit/'evaluation-terminal.json',dict(status='ALL_FIXED_RAW_CLOSED',raw=len(qq),roles=roles,optimizer=oh,RNG=rh,roles_unchanged=True,optimizer_unchanged=True,RNG_unchanged=True,API=0))
def summary(rows):return dict(n=len(rows),source_groups=len({x['source_id'] for x in rows}),correct=sum(x['correct'] for x in rows),valid=sum(x['valid'] for x in rows),capped=sum(x['capped'] for x in rows),predictions=dict(collections.Counter(str(x['prediction']) for x in rows)))
def paired(a,b):
 a={x['id']:x for x in a};b={x['id']:x for x in b};assert set(a)==set(b) and all(x['answer']==b[k]['answer'] for k,x in a.items())
 return dict(n=len(a),before_correct=sum(x['correct'] for x in a.values()),after_correct=sum(x['correct'] for x in b.values()),retained=sum(x['correct'] and b[k]['correct'] for k,x in a.items()),lost=sum(x['correct'] and not b[k]['correct'] for k,x in a.items()),gained=sum(not x['correct'] and b[k]['correct'] for k,x in a.items()))
def score():
 verify();specs=frozen_models();pins=[]
 for key,u in eval_units().items():
  unit=O/key;t=load(unit/'evaluation-terminal.json');assert t['status']=='ALL_FIXED_RAW_CLOSED' and t['raw']==u['count'] and t['roles']==specs[key]['roles'] and t['optimizer']==specs[key]['optimizer'] and t['roles_unchanged'] and t['optimizer_unchanged'] and t['RNG_unchanged']
  assert len(list((unit/'raw').glob('[0-9][0-9][0-9][0-9].json')))==u['count'];pins += [bind(unit/'raw'/f'{i:04d}.json') for i in range(u['count'])]
 assert len(pins)==TOTAL;save(O/'all1436-raw-frozen-before-score.json',dict(raw=pins,epoch=time.time()));save(O/'score-started.json',dict(epoch=time.time()));sys.path[:0]=[str(R/'scripts'),str(R/'src')];import eval_ust_sg_qwen_modes_20261024 as parser
 labels=load(O/'answers.local.json');grades={};scores={};behavior={};qfull=load(O/'queries-full.local.json');qmap={q['id']:q for q in qfull}
 for key,u in eval_units().items():
  qq=load(O/('queries-own68.local.json' if u['step']==68 else 'queries-full.local.json'));rr=[]
  for i,q in enumerate(qq):
   raw=load(O/key/'raw'/f'{i:04d}.json');assert raw['id']==q['id'] and raw['source_id']==q['source_id'] and raw['cap']==128 and raw['variant']==q['variant'];g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);answer=labels[q['id']]['answer']
   rr.append(dict(id=q['id'],source_id=q['source_id'],variant=q['variant'],variant_index=q.get('variant_index'),metric=q['metric'],answer=answer,prediction=g['prediction'],valid=bool(g['valid']),correct=bool(g['valid'] and g['prediction']==answer),capped=raw['finish_reason']=='length'))
  save(O/key/'graded.local.json',rr);grades[key]=rr;train=[x for x in rr if x['variant']=='training-option-cycle'];own=[x for x in train if x['variant_index']==0]
  scores[key]=dict(step=u['step'],identity_own68=summary(own),training_variants=summary(train),training_by_metric={metric:summary([x for x in train if x['metric']==metric]) for metric in sorted({x['metric'] for x in train})},Dev={v:summary([x for x in rr if x['variant']==v]) for v in ('canonical','both','numeric-counterfactual')} if u['step']!=68 else {})
  if u['step']!=68:
   bysource={sid:[x for x in train if x['source_id']==sid] for sid in {x['source_id'] for x in train}};assert len(bysource)==68 and all(len(g)==4 for g in bysource.values())
   equiv=0
   for group in bysource.values():
    baseline=next(x for x in group if x['variant_index']==0)
    equiv+=bool(baseline['valid'] and all(x['valid'] and qmap[x['id']]['old_to_new_letter'].get(baseline['prediction'])==x['prediction'] for x in group))
   can={x['source_id']:x for x in rr if x['variant']=='canonical'};joint={x['source_id']:x for x in rr if x['variant']=='both'};cf={x['source_id']:x for x in rr if x['variant']=='numeric-counterfactual'};maps={q['source_id']:q['old_to_new_letter'] for q in qq if q['variant']=='both'};assert set(can)==set(joint)==set(cf) and len(can)==12
   behavior[key]=dict(training_source_groups=68,all4_correct=sum(all(x['correct'] for x in g) for g in bysource.values()),all4_valid=sum(all(x['valid'] for x in g) for g in bysource.values()),all4_prediction_equivariant=equiv,
    correct_count_per_source_histogram=dict(collections.Counter(sum(x['correct'] for x in g) for g in bysource.values())),Dev_sources=12,Dev_joint_equivariant=sum(x['valid'] and joint[k]['valid'] and maps[k].get(x['prediction'])==joint[k]['prediction'] for k,x in can.items()),Dev_numeric_prediction_changed=sum(x['valid'] and cf[k]['valid'] and x['prediction']!=cf[k]['prediction'] for k,x in can.items()))
 retention={};contrasts={};init=grades['freshINIT']
 for arm in ARMS:
  a=grades[arm+'-step0068'];b=grades[arm+'-step0136'];bown=[x for x in b if x['variant']=='training-option-cycle' and x['variant_index']==0]
  retention[arm]=paired(a,bown);contrasts[arm+'__INIT']={v:paired([x for x in init if x['variant']==v],[x for x in b if x['variant']==v]) for v in ('training-option-cycle','canonical','both','numeric-counterfactual')}
 for arm in ARMS[1:]:
  a=grades['same-source-four-step0136'];b=grades[arm+'-step0136'];contrasts['same-source-four__'+arm]={v:paired([x for x in a if x['variant']==v],[x for x in b if x['variant']==v]) for v in ('training-option-cycle','canonical','both','numeric-counterfactual')}
 assert all(bind(R/p['path'])==p for p in pins);verify();save(O/'results.json',dict(status='COMPLETE_FIXED_OPTION_GROUP_PILOT',scope=SCOPE,scores=scores,retention68_to136=retention,paired_contrasts=contrasts,behavior=behavior,new_raw=TOTAL,new_optimizer_calls=408,micro_forward_backward_calls=1632,API=0,all_groups_reported=True,new_INIT_baseline=True,no_old_INIT_raw_reused=True))
 save(O/'terminal.json',dict(status='COMPLETE',results=bind(O/'results.json'),epoch=time.time()));live(stage='COMPLETE');print('COMPLETE3group-pilot',flush=True)
def controller():
 gate(require_dependencies=False);os.environ.update(numerical_decision()['required_launcher_environment']);save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time()));deadline=time.monotonic()+21600
 while not all((p/'terminal.json').exists() for p in DEPENDENCIES):
  check();assert not any(list(p.glob('failure-*.json')) for p in DEPENDENCIES),'Earlier diagnosis failed; no automatic pilot'
  if time.monotonic()>deadline:raise TimeoutError('Dependency wait exceeded21600seconds')
  live(stage='WAITING_FOR_REQUIRED_DIAGNOSTICS');time.sleep(5)
 gate();save(O/'dependencies-closed.json',dict(dependencies=[dict(terminal=bind(p/'terminal.json'),results=load(p/'terminal.json')['results']) for p in DEPENDENCIES],epoch=time.time()))
 for arm in ARMS:
  with (O/(arm+'-fit.log')).open('x') as f:subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)),'fit','--unit',arm],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
 freeze_models()
 for key in eval_units():
  with (O/(key+'-evaluate.log')).open('x') as f:subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)),'evaluate','--unit',key],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
 score()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','fit','freeze_models','evaluate','score','controller']);p.add_argument('--unit',choices=[*ARMS,*eval_units()]);a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if a.action in ('fit','evaluate'):globals()[a.action](a.unit)
  else:globals()[a.action]()
 except BaseException as e:
  if O.exists():(O/('failure-'+str(time.time_ns())+'.json')).write_text(json.dumps(dict(type=type(e).__name__,error=str(e),epoch=time.time(),no_automatic_retry=True)))
  raise
