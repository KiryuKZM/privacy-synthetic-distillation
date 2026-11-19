"""Fixed real/old/new generator understanding pilot; original Dev155 primary."""
from pathlib import Path
import argparse,collections,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_option_group_pilot_v2_20261107 as core
O=R/'runs/2026-11-07-ust-joint-generator-repair-v1/student-utility-v1'
ARMS=('real68','old_shared_calibrated68','conditional_calibrated68');UNITS=('freshINIT',)+ARMS;TOTAL=1444;STOP=False
load=core.load;save=core.save;bind=core.bind
SCOPE='Fixed understanding-only replacement pilot, common renderer/options/quotas/resources. Original Dev155 primary; joint155 auxiliary, own68 resubstitution. Exposed development, single order, no final or whole-UST equivalence claim. Planning/reflection not trained.'
def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own joint-generator study STOP')
def stop(*_):
 global STOP;STOP=True

def units():return {key:dict(arm=None if key=='freshINIT' else key,step=0 if key=='freshINIT' else 136,count=310 if key=='freshINIT' else 378) for key in UNITS}
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def verify(unit=None):
 check();e=load(O/'execution-bindings.json');assert e['source']==bind(__file__)
 for pin in e['protocols']:assert bind(R/pin['path'])==pin,pin['path']
 if unit is not None:assert Path(unit).parent==O and Path(unit).name in UNITS
 p=load((Path(unit) if unit else O)/'protocol.json');assert p['source']==bind(__file__) and p['total_raw']==TOTAL
 for pin in p['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 return p

def gate(**_):
 verify();a=load(O/'launch-authorization.json');assert a['protocol']==bind(O/'protocol.json') and a['status']=='AUTHORIZED_LOCAL_FIXED_PILOT'
 assert a['preflight']==bind(O/'preflight.json') and load(O/'preflight.json')['status']=='PASS_CPU_ONLY'
 return a

def configure():
 core.O=O;core.ARMS=ARMS;core.TOTAL=TOTAL;core.SCOPE=SCOPE;core.check=check;core.live=live;core.verify=verify;core.gate=gate;core.eval_units=units
configure()

def prepare():
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
 assert not (O/'protocol.json').exists();plan=load(O/'study-plan.json');summary=load(O/'data-summary.json');assert summary['status']=='ALL_DATA_FROZEN'
 for path,digest in summary['source_bindings'].items():assert bind(R/path)['sha256']==digest,path
 assert summary['source_script_sha256']==bind(R/'scripts/ust_joint_generator_student_data_20261107.py')['sha256']
 decision=core.numerical_decision();h=core.helper();_,cp=h.checkpoint_pins(core.OLD,core.INIT,0)
 source=R/'runs/2026-11-06-ust-autonomous-repair-v1/position-training-v2/real-canonical200/protocol.json';p0=load(source)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);family,_=h.target_family(tok);prefix=load(O/'fixed8.local.json');tokens={};variants=[];schedule0=None;quals=[]
 import probe_overnight_train_fit_20261025 as packing
 import ust_option_cycle_weak_data_v2_20261106 as weak
 for arm in ARMS:
  vv=load(O/arm/'variants.local.json');ss=load(O/arm/'schedule.local.json');variants+=vv;assert len(vv)==272 and len(ss)==136
  assert collections.Counter(v['metadata']['metric'] for v in vv)==dict(distance=168,charging_time=56,waiting_time=48)
  mapping={v['id']:v for v in vv};assert collections.Counter(x for row in ss for x in row['variant_ids'])==collections.Counter({v['id']:2 for v in vv})
  structure=[[x.split('/',1)[1] for x in row['variant_ids']] for row in ss]
  if schedule0 is None:schedule0=structure
  assert structure==schedule0
  for slot in range(68):assert {v['answer'] for v in vv if v['source_slot']==slot}==set('ABCD')
  for v in vv:
   cert=weak.certify_explicit_weak_reference(v['question'],v['answer']);ind=weak.independent_numeric(v['question']);assert cert['trusted_gold'] and ind['compatible_answers']==[v['answer']]
   prompt,target=packing.packing(tok,v,dict(text=v['target']),'matched8',prefix);assert prompt==h.eval_prompt(tok,prefix,v['question']) and target==family[v['answer']] and len(prompt)+128<=16384
   tokens[v['id']]=dict(prompt_ids=prompt,target_ids=target);quals.append(dict(id=v['id'],prompt_tokens=len(prompt),target_tokens=len(target),independent_correct=True))
 labels=load(O/'answers.local.json');dev0=None
 for key,u in units().items():
  qq=load(O/key/'queries.local.json');assert len(qq)==u['count'];dev=[q for q in qq if q['variant']!='own']
  if dev0 is None:dev0=dev
  assert dev==dev0 and len(dev)==310
  for q in qq:
   ids=h.eval_prompt(tok,prefix,q['question']);assert len(ids)+128<=16384
   record=dict(prompt_ids=ids,target_ids=family[labels[q['id']]['answer']])
   if q['id'] in tokens:assert tokens[q['id']]==record
   tokens[q['id']]=record
 save(O/'variants.local.json',variants);save(O/'tokens.local.json',tokens);save(O/'token-qualification.local.json',quals)
 pins=[bind(__file__),bind(source),bind(core.__file__),bind(core.BATCH),bind(core.HELPER),bind(core.NUMERICAL_HELPER),bind(core.NUMERICAL_DECISION),bind(h.CAUSAL),bind(h.DECODER),bind(R/'tests/test_ust_option_group_batch_update.py')]+cp
 for k in ('closed_qualification','baseline_provenance_correction','unchanged_setting_reference'):pins.append(decision[k])
 pins.extend(bind(R/p) for p in summary['source_bindings']);pins += [bind(p) for p in O.rglob('*.json')]
 core.machinery();import eval_ust_sg_qwen_modes_20261024,run_llm_task_sweep_20261023,build_llm_task_sweep_qa_20261023
 for mod in tuple(sys.modules.values()):
  path=getattr(mod,'__file__',None)
  if path:
   path=Path(path).resolve()
   if path.is_file() and path.is_relative_to(R) and path.suffix=='.py' and '.venv' not in path.parts and 'vendor' not in path.parts:pins.append(bind(path))
 import torch,fla
 from transformers.models.qwen3_5 import modeling_qwen3_5 as q
 _,implementation=core.numerical_helpers().reference_function(q);save(O/'backend-before-GPU.json',implementation);pins.append(bind(O/'backend-before-GPU.json'))
 pins += [bind(p) for p in Path(fla.__file__).resolve().parent.rglob('*.py')]
 for item in implementation['baseline_implementation_chain']:pins.append(bind(item['file']))
 for rel in ('models/qwen3_5/modeling_qwen3_5.py','integrations/sdpa_attention.py','integrations/hub_kernels.py','modeling_layers.py','cache_utils.py','generation/utils.py'):
  paths=list((R/'.venv/lib').glob('python*/site-packages/transformers/'+rel));assert len(paths)==1;pins.append(bind(paths[0]))
 for rel in ('utils/checkpoint.py','nn/functional.py','nn/attention/__init__.py','backends/cuda/__init__.py','backends/cudnn/__init__.py','optim/adamw.py','optim/adam.py','optim/optimizer.py'):pins.append(bind(Path(torch.__file__).resolve().parent/rel))
 pins += [bind(torch.__file__),bind(R/'pyproject.toml'),bind(R/'uv.lock')]+[bind(p) for p in Path(p0['base']).iterdir() if p.is_file()]
 assert not torch.cuda.is_initialized();pins=list({x['path']:x for x in pins}.values())
 save(O/'protocol.json',dict(source=bind(__file__),bindings=pins,scope=SCOPE,total_raw=TOTAL,arms=ARMS,eval_units=units(),source_INIT=bind(core.INIT/'manifest.json'),steps_per_fit=136,microexamples_per_fit=544,numerical_profile=core.NUMERICAL_PROFILE,API=0,Common_Audit_Confirm_Final_access=False,study_plan=bind(O/'study-plan.json')))
 protocols=[bind(O/'protocol.json')]
 for key,u in units().items():
  p=dict(p0);p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],arm=key,steps=u['step'],lr=1e-4,seed=20261024,fixed_checkpoint=u['step'],method='matched-generator-understanding-batch4-CE' if u['step'] else 'fresh-INIT-greedy',ability_order=['st_understanding']*u['step'],checkpoint_steps=[34,68,102,136] if u['step'] else [],scope=SCOPE,total_raw=TOTAL,queries_each=u['count'],protected_real_training_gold_in_loss=key=='real68',synthetic_reference_in_loss=key in ARMS[1:],teacher_API=0,Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False)
  save(O/key/'protocol.json',p);protocols.append(bind(O/key/'protocol.json'))
 save(O/'execution-bindings.json',dict(source=bind(__file__),protocols=protocols));verify()
 save(O/'preflight.json',dict(status='PASS_CPU_ONLY',all816training_views_independently_numeric_checked=True,identical_source_slot_schedule=True,context_max=max(len(v['prompt_ids']) for v in tokens.values()),target_tokens=7,Dev_source_cases=155,Dev_ambiguous=summary['Dev']['ambiguous'],CUDA_initialized=False,original_Dev_unchanged=True,API=0))
 print('PREPARED',TOTAL,'raw;3x136updates;GPU_NOT_STARTED',flush=True)

def freeze_models():
 gate();h=core.helper();specs={};pins=[]
 for key,u in units().items():
  source=O/key if u['step'] else core.OLD;folder=source/'checkpoints/step0136' if u['step'] else core.INIT
  if u['step']:
   t=load(source/'fit-terminal.json');assert t['status']=='FIXED136_COMPLETE' and t['actual_updates']==136 and t['microexamples']==544;pins.extend([bind(source/'fit-terminal.json'),bind(source/'initialization-verified.json')])
  cm,pp=h.checkpoint_pins(source,folder,u['step']);pins+=pp;specs[key]=dict(source=str(source.relative_to(R)),folder=str(folder.relative_to(R)),step=u['step'],roles=cm['role_hashes'],optimizer=cm['optimizer_hashes']['actor'])
 save(O/'models-frozen.json',dict(status='ALL3FITS_AND4_ENDPOINTS_FROZEN',specs=specs,bindings=list({x['path']:x for x in pins}.values()),epoch=time.time()))
def frozen_models():
 v=load(O/'models-frozen.json');assert v['status']=='ALL3FITS_AND4_ENDPOINTS_FROZEN' and set(v['specs'])==set(UNITS)
 for pin in v['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 return v['specs']

def evaluate(key):
 gate();unit=O/key;p=verify(unit);spec=frozen_models()[key];flags=core.apply_numerical_profile();m,w,_=core.machinery();h=core.helper();decode=h.decoder();save(unit/'evaluation-started.json',dict(PID=os.getpid(),epoch=time.time(),model=spec))
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);folder=R/spec['folder'];torch,tok,model,head=w.initialize(unit,folder);w.m.lib.activate(model,head,'actor');model.eval();opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
  cm=w.m.lib.restore(torch,model,head,dict(actor=opt),R/spec['source'],folder);roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};oh=w.m.lib.nested_hash(opt.state_dict());rng=lambda:dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all());rh=w.m.lib.nested_hash(rng())
  assert roles==spec['roles'] and oh==spec['optimizer'] and cm['actual_updates']==spec['step'] and rh==w.m.lib.nested_hash(torch.load(folder/'rng.pt',map_location='cpu',weights_only=True))
  profile=core.actual_numerical_profile(torch,model);assert profile['numerical_flags']==flags
  save(unit/'reload.json',dict(status='EXACT_ROLES_ADAM_RNG',source=spec,roles=roles,optimizer=oh,RNG=rh,new_updates=0,numerical_profile=profile))
  qq=load(unit/'queries.local.json');tokens=load(O/'tokens.local.json');prefix=load(O/'fixed8.local.json');assert len(qq)==p['queries_each']
  for i,q in enumerate(qq):
   check();save(unit/'raw'/f'{i:04d}.started.json',dict(id=q['id'],epoch=time.time()));h.reset_text_rope(model);ids=h.eval_prompt(tok,prefix,q['question']);assert ids==tokens[q['id']]['prompt_ids'];raw=decode(torch,tok,model,ids)
   save(unit/'raw'/f'{i:04d}.json',dict(id=q['id'],source_id=q['source_id'],cap=128,variant=q['variant'],**raw));h.reset_text_rope(model)
   if (i+1)%34==0:live(stage='EVALUATION',unit=key,closed=i+1,total=len(qq));print('EVAL',key,i+1,flush=True)
  assert profile==core.actual_numerical_profile(torch,model)
  assert roles=={r:w.m.lib.role_hash(model,r) for r in roles} and oh==w.m.lib.nested_hash(opt.state_dict()) and rh==w.m.lib.nested_hash(rng());verify(unit)
  save(unit/'evaluation-terminal.json',dict(status='ALL_FIXED_RAW_CLOSED',raw=len(qq),roles=roles,optimizer=oh,RNG=rh,roles_unchanged=True,optimizer_unchanged=True,RNG_unchanged=True,API=0))

def summary(rows):
 return dict(n=len(rows),numeric_groups=len({x['group'] for x in rows}),correct=sum(x['correct'] for x in rows),weak_compatible=sum(x['weak_compatible'] for x in rows),valid=sum(x['valid'] for x in rows),capped=sum(x['capped'] for x in rows),predictions=dict(collections.Counter(str(x['prediction']) for x in rows)))
def score():
 verify();specs=frozen_models();pins=[]
 for key,u in units().items():
  t=load(O/key/'evaluation-terminal.json');assert t['status']=='ALL_FIXED_RAW_CLOSED' and t['raw']==u['count'] and t['roles']==specs[key]['roles'] and t['roles_unchanged'] and t['optimizer_unchanged'] and t['RNG_unchanged']
  assert len(list((O/key/'raw').glob('[0-9][0-9][0-9][0-9].json')))==u['count'];pins += [bind(O/key/'raw'/f'{i:04d}.json') for i in range(u['count'])]
 assert len(pins)==TOTAL;save(O/'all1444-raw-frozen-before-score.json',dict(raw=pins,epoch=time.time()))
 import eval_ust_sg_qwen_modes_20261024 as parser
 answers=load(O/'answers.local.json');grades={};scores={};behavior={}
 for key in UNITS:
  qq=load(O/key/'queries.local.json');rows=[]
  for i,q in enumerate(qq):
   raw=load(O/key/'raw'/f'{i:04d}.json');assert raw['id']==q['id'] and raw['source_id']==q['source_id'] and raw['variant']==q['variant'] and raw['cap']==128
   g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);a=answers[q['id']]
   rows.append(dict(id=q['id'],source_id=q['source_id'],group=q['group'],variant=q['variant'],metric=q['metric'],flags=q['flags'],answer=a['answer'],compatible=a['compatible'],prediction=g['prediction'],valid=bool(g['valid']),correct=bool(g['valid'] and g['prediction']==a['answer']),weak_compatible=bool(g['valid'] and g['prediction'] in a['compatible']),capped=raw['finish_reason']=='length'))
  save(O/key/'graded.local.json',rows);grades[key]=rows
  scores[key]={v:dict(overall=summary([x for x in rows if x['variant']==v]),by_metric={f:summary([x for x in rows if x['variant']==v and x['metric']==f]) for f in ('distance','charging_time','waiting_time')},subgroups={f:summary([x for x in rows if x['variant']==v and x['flags'].get(f)]) for f in ('any_tail','upper_tail','lower_tail','time_NaN','statistical_outlier_candidate')}) for v in ('original','joint','own')}
  can={x['source_id']:x for x in rows if x['variant']=='original'};joint={x['source_id']:x for x in rows if x['variant']=='joint'};maps={q['source_id']:q['old_to_new_letter'] for q in qq if q['variant']=='joint'}
  behavior[key]=dict(n=155,both_correct=sum(x['correct'] and joint[k]['correct'] for k,x in can.items()),equivariant=sum(x['valid'] and joint[k]['valid'] and maps[k].get(x['prediction'])==joint[k]['prediction'] for k,x in can.items()))
 contrasts={}
 for before,after in [('freshINIT',a) for a in ARMS]+[('real68',a) for a in ARMS[1:]]+[('old_shared_calibrated68','conditional_calibrated68')]:
  contrasts[before+'__'+after]={v:core.paired([x for x in grades[before] if x['variant']==v],[x for x in grades[after] if x['variant']==v]) for v in ('original','joint')}
 assert all(bind(R/p['path'])==p for p in pins);verify()
 save(O/'results.json',dict(status='COMPLETE_FIXED_MATCHED_GENERATOR_PILOT',scope=SCOPE,scores=scores,behavior=behavior,paired_contrasts=contrasts,new_raw=TOTAL,updates=408,microexamples=1632,API=0))
 save(O/'terminal.json',dict(status='COMPLETE',results=bind(O/'results.json'),epoch=time.time()));live(stage='COMPLETE');print('COMPLETE',flush=True)

def controller():
 gate();os.environ.update(core.numerical_decision()['required_launcher_environment']);os.environ['HF_HUB_OFFLINE']='1';plan=load(O/'study-plan.json');deadline=time.monotonic()+plan['maximum_controller_seconds']
 save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time(),max_seconds=plan['maximum_controller_seconds'],independent_session=os.getsid(0)==os.getpid()))
 def run(action,key,limit):
  check();remaining=deadline-time.monotonic();assert remaining>0
  with (O/(key+'-'+action+'.log')).open('x') as log:
   child=subprocess.Popen([str(R/'.venv/bin/python'),str(Path(__file__).resolve()),action,'--unit',key],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   started=time.monotonic();save(O/(key+'-'+action+'-process.json'),dict(PID=child.pid,epoch=time.time(),limit_seconds=min(limit,remaining)))
   try:
    while child.poll() is None:
     check()
     if time.monotonic()-started>min(limit,remaining):raise TimeoutError(key+' '+action+' limit')
     time.sleep(2)
    assert child.returncode==0,(key,action,child.returncode)
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 for arm in ARMS:run('fit',arm,plan['per_fit_seconds'])
 freeze_models()
 for key in UNITS:run('evaluate',key,plan['per_evaluation_seconds'])
 score()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','fit','freeze_models','evaluate','score','controller']);p.add_argument('--unit',choices=UNITS);a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if a.action=='fit':core.fit(a.unit)
  elif a.action=='evaluate':evaluate(a.unit)
  else:globals()[a.action]()
 except BaseException as e:
  save(O/('failure-'+str(time.time_ns())+'.json'),dict(type=type(e).__name__,error=str(e),epoch=time.time(),no_automatic_retry=True));raise
