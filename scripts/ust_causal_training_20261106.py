"""Three frozen local training diagnostics; no external service or hidden final data."""
from pathlib import Path
import argparse,ast,collections,fcntl,hashlib,importlib.util,json,os,re,sqlite3,subprocess,sys,time
R=Path(__file__).resolve().parents[1];B=R/'runs/2026-11-06-ust-autonomous-repair-v1';O=B/'causal-training'
PAIR=R/'scripts/ust_real_synthetic_paired_ce_20261105.py';OLD=R/'runs/2026-10-25-qwen-transfer-repair/seqkd200/USTBench/2b';INIT=OLD/'work/checkpoints/step0000'
SOURCE=R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1/synthetic-reference-ce'
ARMS={'A-original200-replay':200,'B-understanding-real-template200':200,'C-understanding-only68':68}
STOP=False

def load(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check():
 if STOP or (B/'STOP').exists() or (O/'STOP').exists():raise InterruptedError('Own causal diagnostic STOP')
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def verify(unit=None):
 p=load((Path(unit) if unit else O)/'protocol.json');assert p['source']==bind(__file__)
 for x in p['bindings']:assert bind(R/x['path'])==x,x['path']
 check();return p

def prepare():
 assert not (O/'protocol.json').exists()
 data=load(O/'data-protocol.json')
 # Data-side certificates are retained; verify all files independently here.
 t=module(R/'scripts/ust_dev_semantic_transforms_20261106.py','ust_causal_transforms')
 sys.path.insert(0,str(R/'src'));from privacy_synth.ust_task_contract import certify_native_ranking_reference
 original=load(SOURCE/'training200.local.json');a=load(O/'A-original200-replay/training.local.json');b=load(O/'B-understanding-real-template200/training.local.json');c=load(O/'C-understanding-only68/training.local.json')
 assert a==original and c==[x for x in original if x['metadata']['ability']=='st_understanding'] and len(c)==68
 gate=[]
 for x,y in zip(a,b):
  assert x['id']==y['id'] and x['target']==y['target'] and x['metadata']==y['metadata']
  if x['metadata']['ability']!='st_understanding':assert x==y
  else:
   xx=t.sections(x['question']);yy=t.sections(y['question']);assert xx['Data']==yy['Data'] and xx['Question']==yy['Question']
 for arm in ARMS:
  rows=load(O/arm/'training.local.json');assert len(rows)==ARMS[arm]
  for x in rows:
   if x['metadata']['ability']=='st_understanding':
    answer=re.search(r'<Answer>([ABCD])</Answer>',x['target']).group(1);result=certify_native_ranking_reference(x['question'],answer)
    assert result['trusted_gold'],(arm,x['id'],result);gate.append(dict(arm=arm,id=x['id'],**result))
 save(O/'native-gate-training.local.json',gate)
 prior=R/'runs/2026-11-06-ust-task-validity-diagnostic-v1/inference-v2';oldq=load(prior/'queries.local.json');oldans=load(prior/'answers.local.json')
 queries=[];answers={}
 for q in oldq:
  if q['variant'] not in ('canonical','both','numeric-counterfactual'):continue
  v=dict(q,cap=128,cohort='RepairDev');queries.append(v);answers[v['id']]=oldans[v['id']]
 assert len(queries)==76
 for x,y in zip(a,b):
  if x['metadata']['ability']!='st_understanding':continue
  for variant,row in [('train-original',x),('train-native',y)]:
   qid=x['id']+'::'+variant;queries.append(dict(id=qid,source_id=x['id'],question=t.canonical(row['question']),metadata=row['metadata'],variant=variant,cohort='synthetic_train',cap=128))
   answers[qid]=dict(answer=re.search(r'<Answer>([ABCD])</Answer>',row['target']).group(1),provenance='independently verified visible native ranking including explicit NaN rule')
 assert len(queries)==212 and len(answers)==212 and len({q['id'] for q in queries})==212
 save(O/'queries.local.json',queries);save(O/'answers.local.json',answers)
 save(O/'fixed8.local.json',load(SOURCE.parent/'fixed8.local.json'))
 sources=[PAIR,R/'scripts/ust_causal_training_prepare_20261106.py',R/'scripts/ust_task_validity_diagnostic_v2_20261106.py',R/'scripts/ust_dev_semantic_transforms_20261106.py',R/'scripts/probe_overnight_train_fit_20261025.py',R/'scripts/matched_qwen_objective_control_20261102.py',R/'scripts/three_method_common_eval_20261101.py',R/'scripts/eval_ust_sg_qwen_modes_20261024.py',R/'src/privacy_synth/ust_task_contract.py',R/'runs/2026-11-05-ust-environment-generator-ipd-v1/full-schema-mainline-v3/gpu_student_worker.py']
 pins=[bind(p) for p in sources]+[bind(O/f) for f in ['data-protocol.json','queries.local.json','answers.local.json','fixed8.local.json','native-gate-training.local.json']]+[bind(INIT/'manifest.json'),bind(SOURCE/'protocol.json'),bind(SOURCE/'training200.local.json')]
 pins += [bind(O/arm/'training.local.json') for arm in ARMS]
 save(O/'protocol.json',dict(source=bind(__file__),bindings=pins,arms=ARMS,queries_each=212,total_raw=636,initialization=bind(INIT/'manifest.json'),API=0,new_weights=0,scope='controlled local CE mechanism diagnosis; mixed target references uncertain; training resubstitution distinct from exposed Dev; no independent replacement test',resource_note='same understanding exposures, C omits132other updates; C is not equal total-compute comparison'))
 for arm,n in ARMS.items():
  p=load(SOURCE/'protocol.json');p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],arm=arm,steps=n,protected_real_training_gold_in_loss=False,synthetic_reference_in_loss=True,uncertain_planning_reflection_references=arm!='C-understanding-only68')
  save(O/arm/'protocol.json',p)
 print('FROZEN3arms468updates636raw',flush=True)

def machinery():
 m=module(PAIR,'ust_causal_pair');m.O=O;m.verify=verify;m.check=check;w,update=m.machinery();return m,w,update

def fit(arm):
 unit=O/arm;p=verify(unit);m,w,update=machinery();sys.path.insert(0,str(R/'scripts'));import probe_overnight_train_fit_20261025 as packing
 save(unit/'fit-started.json',dict(PID=os.getpid(),epoch=time.time()))
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch,tok,model,head=w.initialize(unit,INIT);w.m.lib.activate(model,head,'actor')
  opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
  cm=w.m.lib.restore(torch,model,head,dict(actor=opt),OLD,INIT);assert not opt.state and cm['actual_updates']==0
  roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};assert roles==cm['role_hashes']
  save(unit/'initialization-verified.json',dict(roles=roles,optimizer=w.m.lib.nested_hash(opt.state_dict()),RNG_restored=True))
  rows=load(unit/'training.local.json');prefix=load(O/'fixed8.local.json');actual=0;start=time.monotonic();checkpoints={min(50,len(rows)),100,150,len(rows)}
  for step,row in enumerate(rows,1):
   check();save(unit/'updates'/f'{step:04d}.started.json',dict(id=row['id'],epoch=time.time()))
   prompt,target=packing.packing(tok,row,dict(text=row['target']),'matched8',prefix);assert len(prompt)+len(target)<=16384
   trace=dict(prompt_ids=prompt,oracle_target_ids=target,actor_before=w.m.lib.role_hash(model,'actor'),sources={'LOCAL_REFERENCE':len(target)},events={'teacher_forcing':len(target)})
   result=update(torch,model,head,opt,trace);actual+=int(result['optimizer_updated'])
   result.update(id=row['id'],step=step,method='local-reference-ce',teacher_API=0,reference_scope='visible_verified_understanding' if row['metadata']['ability']=='st_understanding' else 'legacy_uncertain_target_not_trusted_gold')
   save(unit/'updates'/f'{step:04d}.json',result)
   if step%25==0 or step==len(rows):live(stage='TRAINING',arm=arm,step=step,total=len(rows));print('FIT',arm,step,round(time.monotonic()-start,1),flush=True)
   if step in checkpoints:
    folder=unit/'checkpoints'/f'step{step:04d}';w.m.native.tiny.SAVE_CHECKPOINT(unit,folder,torch,model,head,dict(actor=opt),'UST_CAUSAL_CE',step,actual_updates=actual);w.m.lib.verify_checkpoint(unit,folder)
  assert actual==len(rows)
  final={r:w.m.lib.role_hash(model,r) for r in roles};assert all(final[r]==roles[r] for r in roles if r!='actor') and final['actor']!=roles['actor']
  oldfinal=load(SOURCE/'fit-terminal.json')['roles']
  save(unit/'fit-terminal.json',dict(status='FIXED_ENDPOINT_COMPLETE',actual_updates=actual,seconds=time.monotonic()-start,roles=final,checkpoint=bind(unit/'checkpoints'/f'step{len(rows):04d}'/'manifest.json'),teacher_API=0,original_endpoint_roles_equal=final==oldfinal if arm=='A-original200-replay' else None))

def evaluate(arm):
 unit=O/arm;verify(unit);m,w,_=machinery();n=ARMS[arm];assert load(unit/'fit-terminal.json')['actual_updates']==n
 src=R/'scripts/three_method_common_eval_20261101.py';nodes=[x for x in ast.parse(src.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in ['hf_greedy','context_failure']];ns=dict(time=time,CAP=128,CONTEXT=16384);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(src)+'::UNCHANGED_DECODER','exec'),ns)
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);folder=unit/'checkpoints'/f'step{n:04d}';torch,tok,model,head=w.initialize(unit,folder);w.m.lib.activate(model,head,'actor');model.eval()
  opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False);cm=w.m.lib.restore(torch,model,head,dict(actor=opt),unit,folder);roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};assert roles==cm['role_hashes']==load(unit/'fit-terminal.json')['roles'];oh=w.m.lib.nested_hash(opt.state_dict())
  save(unit/'independent-reload.json',dict(status='EXACT_ROLES_OPTIMIZER_RNG_RELOAD',roles=roles,optimizer=oh,checkpoint=bind(folder/'manifest.json')))
  rows=load(O/'queries.local.json');prefix=load(O/'fixed8.local.json')
  for i,row in enumerate(rows):
   check();save(unit/'raw'/f'{i:04d}.started.json',dict(id=row['id'],epoch=time.time()))
   ids=tok.encode(tok.apply_chat_template(prefix+[dict(role='user',content=row['question'])],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
   raw=ns['hf_greedy'](torch,tok,model,ids);save(unit/'raw'/f'{i:04d}.json',dict(id=row['id'],source_id=row['source_id'],variant=row['variant'],cap=128,**raw))
   if (i+1)%32==0:live(stage='EVALUATION',arm=arm,closed=i+1,total=212);print('EVAL',arm,i+1,212,flush=True)
  assert roles=={r:w.m.lib.role_hash(model,r) for r in roles} and oh==w.m.lib.nested_hash(opt.state_dict())
  save(unit/'evaluation-terminal.json',dict(status='ALL212_RAW_CLOSED',raw=212,roles_unchanged=True,optimizer_unchanged=True,API=0))

def score():
 verify();pins=[]
 for arm in ARMS:
  assert load(O/arm/'evaluation-terminal.json')['raw']==212;pins += [bind(O/arm/'raw'/f'{i:04d}.json') for i in range(212)]
 save(O/'all-raw-frozen-before-score.json',dict(raw=pins,epoch=time.time()))
 sys.path[:0]=[str(R/'scripts'),str(R/'src')];import eval_ust_sg_qwen_modes_20261024 as parser
 rows=load(O/'queries.local.json');labels=load(O/'answers.local.json');scores={}
 for arm in ARMS:
  graded=[];scores[arm]={}
  for i,q in enumerate(rows):
   raw=load(O/arm/'raw'/f'{i:04d}.json');assert raw['id']==q['id'];g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);answer=labels[q['id']]['answer']
   graded.append(dict(id=q['id'],source_id=q['source_id'],cohort=q['cohort'],variant=q['variant'],ability=q['metadata']['ability'],answer=answer,prediction=g['prediction'],valid=bool(g['valid']),correct=bool(g['valid'] and g['prediction']==answer),capped=raw['finish_reason']=='length'))
  save(O/arm/'graded.local.json',graded)
  for v in sorted({q['variant'] for q in rows}):
   rr=[g for g in graded if g['variant']==v];s=dict(total=len(rr),correct=sum(g['correct'] for g in rr),valid=sum(g['valid'] for g in rr),capped=sum(g['capped'] for g in rr),predictions=dict(collections.Counter(g['prediction'] for g in rr)),abilities={})
   for ability in sorted({g['ability'] for g in rr}):
    aa=[g for g in rr if g['ability']==ability];s['abilities'][ability]=dict(total=len(aa),correct=sum(g['correct'] for g in aa),valid=sum(g['valid'] for g in aa))
   scores[arm][v]=s
 budget=load(B/'budget-scope-initial.json');c=sqlite3.connect('file:'+str(R/budget['historical_SQL'])+'?mode=ro',uri=True);sha=hashlib.sha256('\n'.join(c.iterdump()).encode()).hexdigest();c.close();assert sha==budget['historical_dump_sha256']
 save(O/'results.json',dict(status='COMPLETE',scores=scores,updates=ARMS,total_updates=sum(ARMS.values()),total_raw=636,API=0,budget_SQL_unchanged=True,original_replay_roles_equal=load(O/'A-original200-replay/fit-terminal.json')['original_endpoint_roles_equal'],scope=load(O/'protocol.json')['scope']))
 save(O/'terminal.json',dict(status='COMPLETE',results=bind(O/'results.json'),epoch=time.time()));live(stage='COMPLETE');print(json.dumps(scores),flush=True)

def controller():
 verify();save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time()))
 for arm in ARMS:
  for action in ['fit','evaluate']:
   with (O/(arm+'-'+action+'.log')).open('x') as f:subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)),action,'--arm',arm],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
 score()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','evaluate','score','controller']);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
 try:
  if a.action in ['fit','evaluate']:globals()[a.action](a.arm)
  else:globals()[a.action]()
 except BaseException as e:
  if O.exists():(O/('failure-'+str(time.time_ns())+'.json')).write_text(json.dumps(dict(type=type(e).__name__,error=str(e),epoch=time.time())))
  raise
