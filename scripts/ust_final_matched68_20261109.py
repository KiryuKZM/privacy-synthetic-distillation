"""Authorized supplemental Final inference on nine pre-existing matched68 models.
No fitting, selection, prompt change, API, or new Final construction.
"""
from pathlib import Path
import ast, json, os, signal, subprocess, sys, time, traceback
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_final_acceptance_v3_20261109 as f
PREVIOUS=f.O;P=f.P;O=R/'runs/2026-11-09-ust-final-matched68-v1'
NAMES=[f'{k}_n68_s{seed}' for seed in (831,832,833) for k in ('real','old','current')]
load=f.load;save=f.save;bind=f.bind;STOP=False

def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own matched68 STOP')
def plancheck():
 check();p=load(O/'plan.json');assert p['source']==bind(__file__) and p['models']==NAMES and p['authorized_supplemental_Final']
 return p
def verify(unit=None):
 plancheck()
 for b in load(O/'protocol.json')['bindings']:assert bind(R/b['path'])==b,b['path']
 return load(Path(unit)/'protocol.json') if unit else load(O/'protocol.json')
def frozen_models():
 d=load(O/'models-frozen.json')
 for b in d['bindings']:assert bind(R/b['path'])==b,b['path']
 assert set(d['models'])==set(NAMES)
 return d['models']
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def configure():
 f.O=O;f.NAMES=NAMES;f.plancheck=plancheck;f.check=check;f.verify=verify;f.frozen_models=frozen_models;f.live=live;f.configure()
def prepare():
 assert not (O/'protocol.json').exists();O.mkdir(exist_ok=True)
 save(O/'plan.json',dict(source=bind(__file__),models=NAMES,authorized_supplemental_Final=True,user_instruction='那就用现有68题的9个模型补测Final，真实、旧生成器、新生成器三个顺序都报，不调参不重训',unique_sources_each=68,training_updates_each=136,orders=[831,832,833],extra_training=0,API=0,primary='All1154 original Final understanding questions, original author gold; exact existing joint2308 views',scope='POI understanding only; Final already exposed to researcher; these models trained before first Final exposure. Supplemental comparison, no fresh-test or noninferiority proof.',selection='All9fixed136-step68-source endpoints, no best-seed/checkpoint selection',all_raw_before_score=True,maximum_controller_seconds=21600,epoch=time.time()))
 specs=load(P/'specs.json');h=f.s.helper();models={};modelpins=[];initial=None
 for name in NAMES:
  spec=specs[name];assert spec['step']==136 and spec['unique_sources']==68
  unit=P/name;fit=load(unit/'fit-terminal.json');assert fit['actual_updates']==136 and fit['microexamples']==544
  init=load(unit/'initialization-verified.json')
  if initial is None:initial=init
  assert init==initial
  folder=unit/'checkpoints/step0136';cm,cp=h.checkpoint_pins(unit,folder,136);modelpins+=cp+[bind(unit/'fit-terminal.json'),bind(unit/'initialization-verified.json'),bind(unit/'schedule.local.json')]
  models[name]=dict(source=str(unit.relative_to(R)),folder=str(folder.relative_to(R)),step=136,roles=cm['role_hashes'],optimizer=cm['optimizer_hashes']['actor'])
 save(O/'models-frozen.json',dict(models=models,bindings=modelpins,epoch=time.time(),pre_existing_models=True))
 # Reuse exact previously qualified Final inputs; no additional dataset/label construction.
 for filename in ['fixed8.local.json','tokens.local.json','answers.local.json','qualification.local.json']:
  source=PREVIOUS/filename;dest=O/filename;dest.write_bytes(source.read_bytes());assert bind(source)['sha256']==bind(dest)['sha256']
 qq=load(PREVIOUS/'freshINIT/queries.local.json');assert len(qq)==2308
 scope=load(PREVIOUS/'scope.json');scope.update(models=NAMES,total_raw=2308*9,supplemental_matched_count=True,extra_training=0,Final_previously_exposed=True,source_counts_each=68,optimizer_updates_each=136)
 save(O/'scope.json',scope)
 for name in NAMES:
  save(O/name/'queries.local.json',qq)
  p=load(PREVIOUS/'freshINIT/protocol.json');p.update(arm=name,steps=0,training=0,Final_access=True,Final_authorized=True,source=bind(__file__),fixed_checkpoint=136,queries_each=2308,scope=scope)
  save(O/name/'protocol.json',p)
 pins=load(PREVIOUS/'protocol.json')['bindings']+[bind(__file__),bind(P/'specs.json'),bind(PREVIOUS/'freshINIT/queries.local.json')]
 pins += [bind(p) for p in O.rglob('*.json')]
 save(O/'protocol.json',dict(bindings=list({b['path']:b for b in pins}.values()),no_training=True,API=0,total_raw=20772,source=bind(__file__),Final_supplement_authorized=True))
 verify();frozen_models();save(O/'preflight.json',dict(status='PASS_ALL9_MODELS_FROZEN_AND_IDENTICAL_FINAL_INPUTS',models=9,raw=20772,API=0,extra_training=0,epoch=time.time()))
 print('PREPARED9_MODELS_20772_OUTPUTS',flush=True)
def score():
 verify()
 text=Path(f.__file__).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='score');src=ast.get_source_segment(text,node)
 assert src.count('_n1088_s')==2 and src.count("('real','old','simple')")==1
 src=src.replace('_n1088_s','_n68_s').replace("('real','old','simple')","('real','old')").replace('COMPLETE_ONE_FINAL_ACCEPTANCE','COMPLETE_SUPPLEMENTAL_MATCHED68_FINAL')
 ns=dict(vars(f));exec(compile(src,str(Path(__file__))+'::same_final_scorer_matched68','exec'),ns);ns['score']()
 live(stage='COMPLETE',closed=20772,total=20772)
def controller():
 verify();save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time()));deadline=time.monotonic()+plancheck()['maximum_controller_seconds']
 for action,name in [('evaluate',x) for x in NAMES]+[('score',None)]:
  check();label=action+('-'+name if name else '');log=O/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
  args=[str(R/'.venv/bin/python'),str(Path(__file__).resolve()),action]+([name] if name else [])
  with log.open('x') as out:
   child=subprocess.Popen(args,cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0' if action=='evaluate' else ''),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   save(O/'processes'/f'{label}.json',dict(PID=child.pid,epoch=time.time(),action=action,unit=name));start=time.monotonic()
   try:
    while child.poll() is None:
     check();assert time.monotonic()<deadline and time.monotonic()-start<14400;time.sleep(2)
    assert child.returncode==0,f'{label}:exit{child.returncode}'
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
if __name__=='__main__':
 def stop(*_):
  global STOP;STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);configure()
 try:
  if sys.argv[1]=='evaluate':f.s.base.evaluate(sys.argv[2])
  else:globals()[sys.argv[1]]()
 except BaseException as e:
  save(O/f'failure-{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise
