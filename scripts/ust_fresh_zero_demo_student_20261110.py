"""Local-only frozen nine-model evaluation, zero demonstrations, no fitting."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_scale_20261108 as s
import eval_ust_sg_qwen_modes_20261024 as parser
O=R/'runs/2026-11-10-ust-fresh100-zero-demo-v1'
PREV=R/'runs/2026-11-09-ust-final-matched68-v1';NAMES=[f'{k}_n68_s{seed}' for seed in (831,832,833) for k in ('real','old','current')]
load=s.load;save=s.save;bind=s.bind;STOP=False

def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own zero-demo evaluation STOP')
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def verify(unit=None):
 check();p=load(O/'student-protocol.json')
 for b in p['bindings']:assert bind(R/b['path'])==b,b['path']
 return load(Path(unit)/'protocol.json') if unit else p
def frozen_models():
 x=load(O/'student-models-frozen.json')
 for b in x['bindings']:assert bind(R/b['path'])==b,b['path']
 assert set(x['models'])==set(NAMES)
 return x['models']
def configure():
 for m in [s.core,s.base]:
  m.O=O;m.verify=verify;m.check=check;m.live=live;m.gate=lambda **_:verify()
 s.base.frozen_models=frozen_models

def prepare():
 assert not (O/'student-protocol.json').exists()
 plan=load(O/'study-plan.json');assert plan['cohort_choice_confirmed'] and plan['no_retraining'] and plan['student_models']==NAMES
 rows=load(O/'queries.local.json');answers=load(O/'answers.local.json');prefix=load(O/'zero-demo-prefix.local.json')
 original=load(PREV/'fixed8.local.json');assert prefix==original[:1] and prefix[0]['role']=='system' and len(original)==17
 save(O/'fixed8.local.json',prefix) # Legacy evaluator filename; contents contain zero examples.
 modelsource=load(PREV/'models-frozen.json');assert set(modelsource['models'])==set(NAMES)
 for b in modelsource['bindings']:assert bind(R/b['path'])==b
 save(O/'student-models-frozen.json',dict(modelsource,freeze_epoch=time.time(),prior_freeze=bind(PREV/'models-frozen.json'),new_training=0))
 from transformers import AutoTokenizer
 p0=load(PREV/'real_n68_s831/protocol.json');tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);h=s.helper();family,_=h.target_family(tok);tokens={};ids=set()
 from privacy_synth.ust_task_contract import certify_native_ranking_reference
 for q in rows:
  assert q['id'] not in ids;ids.add(q['id']);assert q['variant']=='original'
  a=answers[q['id']];c=certify_native_ranking_reference(q['question'],a['answer']);assert c['trusted_gold'] and c['compatible_answers']==[a['answer']]
  inp=h.eval_prompt(tok,prefix,q['question']);assert len(inp)+128<=16384
  tokens[q['id']]=dict(prompt_ids=inp,target_ids=family[a['answer']])
 save(O/'tokens.local.json',tokens)
 for name in NAMES:
  save(O/name/'queries.local.json',rows);save(O/name/'protocol.json',dict(p0,arm=name,source=bind(__file__),steps=0,training=0,queries_each=len(rows),Final_access=False,Final_authorized=False,demonstrations=0,source_model_training_demonstrations=8))
 pins=load(s.O/'protocol.json')['bindings']+modelsource['bindings']+[bind(__file__),bind(PREV/'models-frozen.json')]
 pins += [bind(O/x) for x in ['study-plan.json','queries.local.json','answers.local.json','zero-demo-prefix.local.json','fixed8.local.json','student-models-frozen.json','tokens.local.json']]
 pins += [bind(O/name/x) for name in NAMES for x in ['queries.local.json','protocol.json']]
 save(O/'student-protocol.json',dict(bindings=list({b['path']:b for b in pins}.values()),source=bind(__file__),units=NAMES,questions_each=len(rows),total_raw=len(rows)*9,API=0,new_training=0,demonstrations=0,all_student_raw_before_score=True,maximum_controller_seconds=7200))
 verify();save(O/'student-preflight.json',dict(status='PASS_FROZEN9_MODELS_ZERO_DEMOS',questions=len(rows),groups=len({x['group'] for x in rows}),input_token_min=min(len(x['prompt_ids']) for x in tokens.values()),input_token_max=max(len(x['prompt_ids']) for x in tokens.values()),API=0,extra_training=0))
 print('PREPARED',len(rows)*9,flush=True)

def score():
 import ust_joint_generator_student_closed_audit_20261107 as independent
 verify();a=load(O/'answers.local.json');parse=independent.independent();scores={};pins=[]
 for name in NAMES:
  qq=load(O/name/'queries.local.json');term=load(O/name/'evaluation-terminal.json');assert term['raw']==len(qq) and all(term[k] for k in ['roles_unchanged','optimizer_unchanged','RNG_unchanged'])
  pins += [bind(O/name/'raw'/f'{i:04d}.json') for i in range(len(qq))]
 save(O/'all-student-raw-closed-before-score.json',dict(bindings=pins,epoch=time.time()))
 def stats(rr):return dict(n=len(rr),groups=len({x['group'] for x in rr}),correct=sum(x['correct'] for x in rr),valid=sum(x['valid'] for x in rr),capped=sum(x['capped'] for x in rr))
 for name in NAMES:
  rr=[]
  for i,q in enumerate(load(O/name/'queries.local.json')):
   raw=load(O/name/'raw'/f'{i:04d}.json');assert raw['id']==q['id'];pred=parse(raw['raw_generated'],raw['finish_reason']);g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);assert pred==g['prediction']
   rr.append(dict(id=q['id'],group=q['group'],metric=q['metric'],prediction=pred,correct=bool(g['valid'] and pred==a[q['id']]['answer']),valid=bool(g['valid']),capped=raw['finish_reason']=='length'))
  save(O/name/'graded.local.json',rr);scores[name]=dict(overall=stats(rr),metrics={m:stats([x for x in rr if x['metric']==m]) for m in ['distance','charging_time','waiting_time']})
 save(O/'student-results.json',dict(status='COMPLETE_NINE_FROZEN_STUDENTS_ZERO_DEMOS',scores=scores,API=0,new_training=0,scope=load(O/'study-plan.json')))
 save(O/'student-terminal.json',dict(status='COMPLETE',epoch=time.time(),results=bind(O/'student-results.json')));live(stage='COMPLETE')
def controller():
 verify();save(O/'student-controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time()));deadline=time.monotonic()+7200
 for action,name in [('evaluate',n) for n in NAMES]+[('score',None)]:
  check();label=action+('-'+name if name else '');log=O/'student-logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
  with log.open('x') as out:
   cmd=[str(R/'.venv/bin/python'),__file__,action]+([name] if name else [])
   child=subprocess.Popen(cmd,cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0' if action=='evaluate' else ''),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);save(O/'student-processes'/f'{label}.json',dict(PID=child.pid,epoch=time.time()));start=time.monotonic()
   try:
    while child.poll() is None:
     check();assert time.monotonic()<deadline and time.monotonic()-start<1800;time.sleep(2)
    assert child.returncode==0,f'{label}:exit{child.returncode}'
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
if __name__=='__main__':
 def stop(*_):
  global STOP;STOP=True
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);configure()
 try:
  if sys.argv[1]=='evaluate':s.base.evaluate(sys.argv[2])
  else:globals()[sys.argv[1]]()
 except BaseException as e:
  save(O/f'student-failure-{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise
