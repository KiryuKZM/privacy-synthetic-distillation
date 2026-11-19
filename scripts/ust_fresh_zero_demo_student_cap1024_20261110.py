"""Repeat frozen zero-demo students with actual decoder CAP1024 only.
All original CAP128 data/results remain immutable. No training or API.
"""
from pathlib import Path
import ast,contextlib,difflib,hashlib,json,marshal,os,signal,subprocess,sys,time,traceback,types
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_fresh_zero_demo_student_20261110 as z
PARENT=z.O;O=PARENT/'student-cap1024-v1';NAMES=z.NAMES;load=z.load;save=z.save;bind=z.bind;STOP=False

def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own student CAP1024 STOP')
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def verify(unit=None):
 check();p=load(O/'student-protocol.json');assert p['source']==bind(__file__) and p['actual_decoder_cap']==1024
 for b in p['bindings']:assert bind(R/b['path'])==b,b['path']
 return load(Path(unit)/'protocol.json') if unit else p
def frozen_models():
 d=load(O/'student-models-frozen.json')
 for b in d['bindings']:assert bind(R/b['path'])==b,b['path']
 assert set(d['models'])==set(NAMES)
 return d['models']
def configure():
 z.O=O;z.check=check;z.verify=verify;z.frozen_models=frozen_models;z.live=live;z.configure()
def decoder1024():
 h=z.s.core.helper();decode=h.decoder();assert decode.__globals__['CAP']==128 and decode.__globals__['CONTEXT']==16384
 code=marshal.dumps(decode.__code__);decode.__globals__['CAP']=1024
 assert decode.__globals__['CAP']==1024 and marshal.dumps(decode.__code__)==code
 return decode

def changed_evaluator():
 path=Path(z.s.base.__file__);text=path.read_text();node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='evaluate');before=ast.get_source_segment(text,node)
 assert before.count('decode=h.decoder()')==1 and before.count('cap=128')==1
 after=before.replace('decode=h.decoder()','decode=decoder1024()').replace('cap=128','cap=1024')
 assert after.replace('decode=decoder1024()','decode=h.decoder()').replace('cap=1024','cap=128')==before
 return path,before,after

def cpu_cap_contract():
 import numpy as np
 calls=[]
 class Hook:
  def remove(self):pass
 class Model:
  def register_forward_pre_hook(self,*args,**kw):return Hook()
  def register_forward_hook(self,*args,**kw):return Hook()
  def generate(self,**kw):
   calls.append({k:v for k,v in kw.items() if k not in ['input_ids','attention_mask']});assert kw['max_new_tokens']==1024 and kw['do_sample'] is False
   return np.concatenate([kw['input_ids'],np.array([[7]])],axis=1)
 class Tok:
  pad_token_id=0
  def convert_tokens_to_ids(self,x):return 7 if x=='<|im_end|>' else 8
  def decode(self,x,**kw):return '<|im_end|>'
 fake=types.SimpleNamespace(tensor=lambda x,**kw:np.array(x),ones_like=np.ones_like,inference_mode=contextlib.nullcontext,cuda=types.SimpleNamespace(synchronize=lambda:None))
 d=decoder1024();out=d(fake,Tok(),Model(),[1,2]);assert out['response_ids']==[7] and out['finish_reason']=='stop'
 blocked=d(fake,Tok(),Model(),[1]*16000);assert blocked['status']=='CONTEXT_OVERFLOW_NO_DECODE' and blocked['maximum_new_tokens']==1024 and len(calls)==1
 return dict(status='PASS_ACTUAL_MODEL_GENERATE_ARGUMENT_CAPTURE_CPU_ONLY',calls=calls,context_guard_uses1024=True,new_model_inference=0,new_GPU=0)

def prepare():
 assert not (O/'student-protocol.json').exists();O.mkdir(exist_ok=True)
 assert load(PARENT/'student-terminal.json')['status']=='COMPLETE'
 save(O/'authorization.json',dict(user_request='学生也统一1024，同72题、9个现有2B模型、无8道示例、不重训',authorized_by_parent_confirmed_human_instruction=True,API=0,new_training=0,epoch=time.time()))
 for filename in ['queries.local.json','answers.local.json','zero-demo-prefix.local.json','fixed8.local.json','tokens.local.json','student-models-frozen.json']:
  src=PARENT/filename;dst=O/filename
  with dst.open('xb') as f:f.write(src.read_bytes())
  assert bind(src)['sha256']==bind(dst)['sha256']
 plan=load(PARENT/'study-plan.json');plan.update(student_decoding='Existing greedy off1024; actual hf_greedy CAP1024',parent_study=bind(PARENT/'study-plan.json'),teacher_external_authorization='Not used by student-only repeat',actual_student_output_cap=1024,previous_student_output_cap=128,no_retraining=True)
 save(O/'study-plan.json',plan)
 queries=load(O/'queries.local.json');prefix=load(O/'fixed8.local.json');tokens=load(O/'tokens.local.json');assert len(queries)==72 and len(prefix)==1 and prefix[0]['role']=='system' and len({q['group'] for q in queries})==50
 assert all(len(tokens[q['id']]['prompt_ids'])+1024<=16384 for q in queries)
 for name in NAMES:
  save(O/name/'queries.local.json',queries);p=load(PARENT/name/'protocol.json');p.update(source=bind(__file__),maximum_new_tokens=1024,actual_decoder_cap=1024,training=0,steps=0)
  save(O/name/'protocol.json',p)
 path,before,after=changed_evaluator()
 (O/'original-evaluator.py.txt').write_text(before);(O/'cap1024-evaluator.py.txt').write_text(after)
 (O/'exact-evaluator-diff.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='unchanged_previous_evaluate',tofile='cap1024_evaluate')))
 save(O/'decoder-change.json',dict(original_evaluator=bind(path),original_helper=bind(z.s.core.helper().__file__),decoder_source=bind(z.s.core.helper().DECODER),actual_change='Only decoder globals CAP128->1024; hf_greedy/context_failure function code and numerical settings unchanged. Evaluator replaces decoder factory and cap metadata only.',original_cap=128,new_cap=1024,context=16384,qualifier='CPU stub proves actual model.generate(max_new_tokens=1024), not numerical equivalence; actual648 outputs will be compared token-for-token after all9close.'))
 save(O/'cpu-decoder-cap-contract.json',cpu_cap_contract())
 # Freeze original raw outputs too, so paired comparisons cannot silently use changed originals.
 priorpins=[bind(PARENT/name/'raw'/f'{i:04d}.json') for name in NAMES for i in range(72)]
 save(O/'prior128-raw-bindings.json',dict(bindings=priorpins,total=648))
 pins=load(PARENT/'student-protocol.json')['bindings']+[bind(__file__),bind(PARENT/'student-terminal.json'),bind(PARENT/'student-results.json')]+priorpins
 pins += [bind(p) for p in O.rglob('*') if p.is_file()]
 save(O/'student-protocol.json',dict(source=bind(__file__),bindings=list({b['path']:b for b in pins}.values()),actual_decoder_cap=1024,previous_decoder_cap=128,units=NAMES,questions_each=72,total_raw=648,API=0,new_training=0,demonstrations=0,all_student_raw_before_score=True,maximum_controller_seconds=10800))
 verify();frozen_models();save(O/'student-preflight.json',dict(status='PASS_FROZEN9_CAP1024_REAL_DECODER_ARGUMENT_QUALIFIED',models=9,questions=72,groups=50,planned_raw=648,actual_output_cap=1024,identical_inputs_tokens_models=True,new_GPU=0,new_training=0,API=0,epoch=time.time()))
 print('PREPARED9_CAP1024_648OUTPUTS_NO_GPU_STARTED',flush=True)

def evaluate(name):
 verify();d=decoder1024();assert d.__globals__['CAP']==1024
 save(O/name/'decoder-runtime.json',dict(actual_CAP=d.__globals__['CAP'],CONTEXT=d.__globals__['CONTEXT'],function_code_sha256=hashlib.sha256(marshal.dumps(d.__code__)).hexdigest(),source=bind(__file__),epoch=time.time()))
 path,before,after=changed_evaluator();ns=dict(vars(z.s.base),decoder1024=decoder1024);exec(compile(after,str(path)+'::EXACT_CAP1024_CHANGE','exec'),ns);ns['evaluate'](name)

def score():
 # Original scorer requires ALL9 raw closure before it reads any model scores.
 z.score();comparisons={};different=[]
 for name in NAMES:
  counters=dict(n=72,identical_response_ids=0,identical_raw_text=0,identical_finish_reason=0,original_truncated=0,new_truncated=0)
  for i in range(72):
   old=load(PARENT/name/'raw'/f'{i:04d}.json');new=load(O/name/'raw'/f'{i:04d}.json')
   assert old['id']==new['id'] and old['prompt_tokens']==new['prompt_tokens'] and old['cap']==128 and new['cap']==1024
   counters['identical_response_ids']+=old['response_ids']==new['response_ids'];counters['identical_raw_text']+=old['raw_generated']==new['raw_generated'];counters['identical_finish_reason']+=old['finish_reason']==new['finish_reason'];counters['original_truncated']+=old['finish_reason']=='length';counters['new_truncated']+=new['finish_reason']=='length'
   if old['response_ids']!=new['response_ids'] or old['finish_reason']!=new['finish_reason']:different.append(dict(model=name,index=i,id=new['id'],old=bind(PARENT/name/'raw'/f'{i:04d}.json'),new=bind(O/name/'raw'/f'{i:04d}.json')))
  comparisons[name]=counters
 save(O/'cap128-vs1024-token-comparison.json',dict(status='COMPLETE_ALL648_PAIRED_OUTPUTS',models=comparisons,changed_outputs=different,all648_token_identical=all(x['identical_response_ids']==72 for x in comparisons.values()),original_results_preserved=True,epoch=time.time()))
 save(O/'cap1024-terminal.json',dict(status='COMPLETE_STUDENT_REINFERENCE_AND_PAIRED_TOKEN_CHECK',results=bind(O/'student-results.json'),comparison=bind(O/'cap128-vs1024-token-comparison.json'),API=0,new_training=0,epoch=time.time()));live(stage='COMPLETE')

def controller():
 verify();save(O/'student-controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time()));deadline=time.monotonic()+10800
 for action,name in [('evaluate',n) for n in NAMES]+[('score',None)]:
  check();label=action+('-'+name if name else '');log=O/'student-logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
  with log.open('x') as out:
   cmd=[str(R/'.venv/bin/python'),__file__,action]+([name] if name else []);child=subprocess.Popen(cmd,cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0' if action=='evaluate' else ''),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   save(O/'student-processes'/f'{label}.json',dict(PID=child.pid,epoch=time.time()));start=time.monotonic()
   try:
    while child.poll() is None:
     check();assert time.monotonic()<deadline and time.monotonic()-start<2400;time.sleep(2)
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
  if sys.argv[1]=='evaluate':evaluate(sys.argv[2])
  else:globals()[sys.argv[1]]()
 except BaseException as e:save(O/f'student-failure-{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise
