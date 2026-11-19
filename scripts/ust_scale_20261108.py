"""Frozen source-count scaling, independent cloud controller; no Final access."""
from pathlib import Path
import argparse,collections,copy,fcntl,hashlib,json,os,random,signal,subprocess,sys,time,traceback
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_missingness_data_20261107 as d
import ust_missingness_delivery_20261107 as audit
import ust_joint_generator_student_run_20261107 as base
core=base.core
O=R/'runs/2026-11-08-ust-scale-v1'
load=core.load;save=core.save;bind=core.bind
SIZES=(68,272,1088);SEEDS=(831,832,833);KINDS=('real','simple','old','current')
ARMS=tuple(f'{k}_n{n}_s{seed}' for n in SIZES for seed in SEEDS for k in KINDS)
INIT=core.INIT;OLD=core.OLD;BATCH=core.BATCH
module=core.module;apply_numerical_profile=core.apply_numerical_profile;actual_numerical_profile=core.actual_numerical_profile
STOP=False

def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own scale STOP')
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def verify(unit=None):
 check();p=load(O/'protocol.json')
 for b in p['bindings']:assert bind(R/b['path'])==b,b['path']
 return load(Path(unit)/'protocol.json') if unit else p

def gate(**kw):return verify()
def helper():return core.helper()
def machinery():return core.machinery()
def configure():
 core.O=O;core.check=check;core.live=live;core.verify=verify;core.gate=gate;core.ARMS=ARMS
 base.O=O;base.check=check;base.live=live;base.verify=verify;base.gate=gate;base.frozen_models=frozen_models

def prepare():
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
 assert not (O/'protocol.json').exists()
 gp=load(d.G/'protocol.json');fit=load(R/gp['fit']['path'])
 model=d.ResidualRepair(fit,'conditional_charge_and_wait');saved=load(d.G/'conditional_charge_and_wait/fitted-parameters.local.json')
 assert np.array_equal(model.beta_charge,saved['charge_coefficients']) and np.array_equal(model.beta_wait,saved['wait_coefficients']) and np.array_equal(model.root,saved['root'])
 cal=d.Calibrated.__new__(d.Calibrated);cal.model=model;cal.tables=[(np.array(x['log_probe']),np.array(x['Fit_positive_log_inverse'])) for x in load(d.G/'conditional_calibrated/tables.local.json')]
 current,_=cal.sample(32768,2026110809)
 # Original neural generator's already-saved independent check draws, with its frozen marginal map.
 folder=d.old.D/'full-schema-shared-calibrated-v2';raw=np.load(folder/'independent-check.npy');raw=raw.astype(float)
 tables=load(folder/'calibration-tables.local.json')
 for j,t in enumerate(tables):
  v=np.array(t['generated_CDF']);q=np.array(t['Fit_inverse']);take=raw[:,33+j-12]==0 if 12<=j<20 else np.ones(len(raw),bool)
  u=(np.searchsorted(v,raw[take,j],side='left')+np.searchsorted(v,raw[take,j],side='right'))/(2*len(v));raw[take,j]=q[np.minimum((u*len(q)).astype(int),len(q)-1)]
  if 12<=j<20:raw[~take,j]=0
 old=d.old.gen.base.Codec(fit,True).decode(raw)
 rng=np.random.default_rng(2026110810);simple=[]
 for i in range(32768):
  obs=[[round(float(rng.uniform(0,1)),3),round(float(rng.uniform(0,100)),2),round(float(rng.uniform(0,1000)),2),round(float(rng.uniform(0,24)),2),round(float(rng.uniform(0,24)),2)] for _ in range(4)]
  if i%4==0:obs[(i//4)%4][3:]=[None,None]
  simple.append(dict(index=i,observations=obs,failures=[]))
 pools=dict(current=current,old=old,simple=simple)
 excluded=set()
 # Only already-exposed local development/diagnostic files, never the sealed source.
 ep=[d.S/'data/Dev310-queries.local.json',d.S/'data/Select-queries.local.json',d.S/'data/Confirm-queries.local.json',R/'runs/2026-11-08-ust-teacher-expanded-v1/synthetic-selected.local.json']
 for name in ('fresh-data','fresh-data-v2'):ep.append(R/'runs/2026-11-08-ust-student-instability-v1'/name/'queries.local.json')
 for p in ep:
  for q in load(p):
   obs,_=audit.sections(q['question']);excluded.add(d.sig([[None if obs[i][f] is None else float(obs[i][f]) for f in d.old.FIELDS] for i in range(1,5)]))
 real=load(d.S/'data/real_rule/source68.local.json');assert len(real)==68
 for x in real:excluded.add(d.sig(x['obs']))
 goals=[x['metric'] for x in real];assert collections.Counter(goals)==dict(distance=42,charging_time=14,waiting_time=12)
 missing_slots={m:[i for i,v in enumerate(goals) if v==m][:4] for m in ('waiting_time','charging_time')}
 datasets={};counts={}
 for kind,pool in pools.items():
  save(O/'data'/f'{kind}-pool.local.json',pool)
  candidates=collections.defaultdict(list)
  for row in pool:
   if row['failures'] or d.sig(row['observations']) in excluded:continue
   obs=row['observations']
   for metric in ('distance','waiting_time','charging_time'):
    count=sum(v[d.old.FIELDS.index(metric)] is None for v in obs)
    if count<=1:candidates[(metric,count)].append(row)
  for rows in candidates.values():rows.sort(key=lambda x:d.hashkey('scale',kind,x['index']))
  chosen=[];used=set();pos=collections.defaultdict(int)
  for slot in range(1088):
   metric=goals[slot%68];missing=int(slot%68 in missing_slots.get(metric,[]));bucket=(metric,missing);rows=candidates[bucket]
   while pos[bucket]<len(rows):
    row=rows[pos[bucket]];pos[bucket]+=1;sig=d.sig(row['observations'])
    if sig in used or d.old.options(row['observations'],metric,slot) is None:continue
    used.add(sig);chosen.append(dict(obs=row['observations'],metric=metric,world=f'scale/{kind}/{row["index"]}',pool=kind));break
   else:raise RuntimeError(f'Frozen {kind} pool insufficient stratum {missing}, slot {slot}; no automatic refill')
  datasets[kind]=chosen;counts[kind]=dict(pool=len(pool),eligible={str(k):len(v) for k,v in candidates.items()},selected=len(chosen),target_missing=128)
 datasets['real']=real
 prefix=load(d.S/'data/fixed8.local.json');save(O/'fixed8.local.json',prefix)
 qq=load(d.S/'data/Dev310-queries.local.json');aa=load(d.S/'data/Dev310-answers.local.json')
 save(O/'answers.local.json',aa)
 from transformers import AutoTokenizer
 p0=load(d.P/'real68/protocol.json');tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);h=helper();family,_=h.target_family(tok)
 tokens={};variants=[];specs={}
 for n in SIZES:
  for kind in KINDS:
   rows=datasets[kind][:n];vv=[]
   for slot,item in enumerate(rows):
    q,answer,_=d.native(item['obs'],item['metric'],slot)
    for offset in range(4):
     letters='ABCD';tr=d.old.transform.transform(q,{i:i for i in range(1,5)},letters[offset:]+letters[:offset]);gold=tr['old_to_new_letter'][answer];question=d.clarify(tr['question'],'rule');d.certify(question,gold,True)
     vid=f'{kind}_n{n}/{slot}/{offset}';v=dict(id=vid,source_id=item['world'],source_slot=slot,variant_index=offset,question=question,answer=gold,target='<Answer>'+gold+'</Answer>',metadata=dict(ability='st_understanding',metric=item['metric']),metric=item['metric']);vv.append(v)
     ids=h.eval_prompt(tok,prefix,question);assert len(ids)+128<=16384;tokens[vid]=dict(prompt_ids=ids,target_ids=family[gold])
   assert collections.Counter(v['answer'] for v in vv)=={a:len(rows) for a in 'ABCD'}
   variants+=vv;save(O/'data'/f'{kind}_n{n}-variants.local.json',vv)
   for seed in SEEDS:
    arm=f'{kind}_n{n}_s{seed}';order=list(range(len(vv)));random.Random(seed).shuffle(order)
    repeats=2*n//len(rows);schedule=[dict(step=e*len(rows)+i+1,epoch=e+1,batch_index=i,variant_ids=[vv[j]['id'] for j in order[4*i:4*i+4]]) for e in range(repeats) for i in range(len(rows))]
    assert len(schedule)==2*n and all(len(x['variant_ids'])==4 for x in schedule)
    save(O/arm/'schedule.local.json',schedule);save(O/arm/'queries.local.json',qq)
    p={k:p0[k] for k in ('base','model','seed','lr','lora','loss','optimizer_restored')};p.update(steps=2*n,checkpoint_steps=[2*n],ability_order=['st_understanding']*(2*n),queries_each=310,source=bind(__file__),Final_access=False,teacher_API=0)
    save(O/arm/'protocol.json',p);specs[arm]=dict(step=2*n,kind=kind,unique_sources=len(rows),order_seed=seed)
 for q in qq:
  ids=h.eval_prompt(tok,prefix,q['question']);tokens[q['id']]=dict(prompt_ids=ids,target_ids=family[aa[q['id']]['answer']])
 save(O/'freshINIT/protocol.json',dict(p0,steps=0,queries_each=310,source=bind(__file__)));save(O/'freshINIT/queries.local.json',qq)
 specs['freshINIT']=dict(step=0,kind='INIT',unique_sources=0)
 save(O/'variants.local.json',variants);save(O/'tokens.local.json',tokens);save(O/'specs.json',specs)
 # Include prior frozen software/model/backend dependencies, and every new immutable input.
 parent=load(d.P/'protocol.json');pins=parent['bindings']+[bind(__file__),bind(d.__file__),bind(d.old.__file__)]+[bind(p) for p in ep]
 pins += [bind(p) for p in O.rglob('*.json') if p.name not in ('launch.json','live.json')]
 save(O/'protocol.json',dict(bindings=list({x['path']:x for x in pins}.values()),sizes=SIZES,seeds=SEEDS,fits=36,total_updates=sum(x['step'] for x in specs.values()),total_raw=37*310,counts=counts,API=0,Final_access=False,common_real_demonstrations=8,real_unique_targets=68,reference_labelled_SFT=True,maximum_seconds=172800,old_pool='frozen original neural independent calibration-check draws; no new neural fit',simple='arbitrary fixed public-schema ranges, no Fit statistics; not physical simulator',replicates='training orders; one numeric pool per generator',inference='exposed Dev only, no final acceptance'))
 verify();save(O/'preflight.json',dict(status='PASS_ALL_DATA_AND_BINDINGS',fits=36,raw=11470,API=0));print('PREPARED',counts,flush=True)

def frozen_models():
 specs=load(O/'specs.json');out={};h=helper()
 for key,s in specs.items():
  if s['step'] and not (O/key/'fit-terminal.json').exists():continue
  source=O/key if s['step'] else OLD;folder=source/'checkpoints'/f'step{s["step"]:04d}' if s['step'] else INIT
  cm,_=h.checkpoint_pins(source,folder,s['step']);out[key]=dict(source=str(source.relative_to(R)),folder=str(folder.relative_to(R)),step=s['step'],roles=cm['role_hashes'],optimizer=cm['optimizer_hashes']['actor'])
 return out

def fit(arm):
 assert arm in ARMS;gate();unit=O/arm;verify(unit);flags=apply_numerical_profile();m,w,_=machinery();batch=module(BATCH,'option_group_batch4');h=helper()
 steps=load(unit/'protocol.json')['steps']
 save(unit/'fit-started.json',dict(PID=os.getpid(),epoch=time.time(),new_updates=steps,microexamples=steps*4))
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
   if step%17==0:live(stage='TRAINING',arm=arm,step=step,total=steps);print('FIT',arm,step,round(time.monotonic()-start,1),flush=True)
   if step in sorted(set([steps//4,steps//2,3*steps//4,steps])):
    folder=unit/'checkpoints'/f'step{step:04d}';w.m.native.tiny.SAVE_CHECKPOINT(unit,folder,torch,model,head,dict(actor=opt),'OPTION_GROUP_BATCH4',step,actual_updates=step);w.m.lib.verify_checkpoint(unit,folder)
  assert profile==actual_numerical_profile(torch,model)
  final={r:w.m.lib.role_hash(model,r) for r in roles};assert all(final[r]==roles[r] for r in roles if r!='actor')
  save(unit/'fit-terminal.json',dict(status='FIXED_SCALE_COMPLETE',actual_updates=steps,optimizer_calls=steps,parameter_change_batches=parameter_change_batches,actual_updates_semantics='optimizer calls including zero aggregate gradient steps',microexamples=steps*4,roles=final,optimizer=w.m.lib.nested_hash(opt.state_dict()),RNG=rh,seconds=time.monotonic()-start,checkpoint=bind(unit/'checkpoints'/f'step{steps:04d}'/'manifest.json'),API=0))

def score():
 import eval_ust_sg_qwen_modes_20261024 as parser
 import ust_joint_generator_student_closed_audit_20261107 as independent
 pred=independent.independent();answers=load(O/'answers.local.json');summary={}
 for key in load(O/'specs.json'):
  assert load(O/key/'evaluation-terminal.json')['raw']==310
  grades=[]
  for i,q in enumerate(load(O/key/'queries.local.json')):
   raw=load(O/key/'raw'/f'{i:04d}.json');assert raw['id']==q['id'];v=pred(raw['raw_generated'],raw['finish_reason']);g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);assert v==g['prediction'];a=answers[q['id']]
   grades.append(dict(id=q['id'],group=q['group'],variant=q['variant'],metric=q['metric'],flags=q['flags'],prediction=v,correct=v==a['answer'],compatible=v in a['compatible'],valid=bool(g['valid']),capped=raw['finish_reason']=='length'))
  save(O/key/'graded.local.json',grades)
  def stats(rr):return dict(n=len(rr),correct=sum(r['correct'] for r in rr),compatible=sum(r['compatible'] for r in rr),valid=sum(r['valid'] for r in rr),capped=sum(r['capped'] for r in rr))
  summary[key]={view:dict(overall=stats([r for r in grades if r['variant']==view]),by_metric={m:stats([r for r in grades if r['variant']==view and r['metric']==m]) for m in ('distance','waiting_time','charging_time')},subgroups={f:stats([r for r in grades if r['variant']==view and r['flags'].get(f)]) for f in ('any_tail','time_NaN','statistical_outlier_candidate')}) for view in ('original','joint')}
 save(O/'results.json',dict(status='COMPLETE_ALL36_FITS_AND11470_OUTPUTS',scores=summary,Final_access=False,API=0));save(O/'terminal.json',dict(status='COMPLETE',epoch=time.time(),results=bind(O/'results.json')))

def controller():
 configure();save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time()));deadline=time.monotonic()+172800
 def call(action,key=None):
  check();name=action+('-'+key if key else '');receipt=O/'processes'/f'{name}.json';log=O/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
  cmd=[str(R/'.venv/bin/python'),str(Path(__file__).resolve()),action]+([key] if key else []);env=dict(os.environ,HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');limit=min(14400 if action=='fit' else 7200,deadline-time.monotonic());assert limit>0
  with log.open('x') as f:
   child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);save(receipt,dict(PID=child.pid,action=action,unit=key,epoch=time.time()));start=time.monotonic()
   try:
    while child.poll() is None:
     check()
     if time.monotonic()-start>limit:raise TimeoutError(name)
     time.sleep(2)
    if child.returncode:raise RuntimeError(f'{name}:exit {child.returncode}')
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 try:
  if not (O/'protocol.json').exists():call('prepare')
  verify()
  call('evaluate','freshINIT')
  for key in ARMS:call('fit',key);call('evaluate',key)
  call('score')
 except BaseException as e:
  save(O/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','fit','evaluate','score','controller']);a.add_argument('unit',nargs='?');args=a.parse_args();configure()
 def stop(*_):
  global STOP;STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 if args.action=='fit':fit(args.unit)
 elif args.action=='evaluate':base.evaluate(args.unit)
 else:globals()[args.action]()
