"""Unmodified official task methods with local inference/transport and journal adapters."""
from pathlib import Path
import ast,collections,copy,hashlib,inspect,json,os,re,sys,time,traceback
import numpy as np,regex
from tqdm import tqdm
R=Path(__file__).resolve().parents[1];O=R/'runs/2026-11-11-ust-original-baselines-v1';S=O/'source'
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
MAXOUT=8192

def load(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def pin(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return dict(path=str(p.relative_to(R)) if p.is_relative_to(R) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check():
 if (O/'STOP').exists():raise InterruptedError('Own original baseline STOP')
def verify():
 p=load(O/'execution.json')
 for x in p['code']+p['inputs']:assert pin(R/x['path'])==x,x['path']
 return p

def official_class():
 source=S/'utils/language_model.py';tree=ast.parse(source.read_text());cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LLM')
 ns=dict(copy=copy,np=np,re=re,regex=regex,tqdm=tqdm,json=json,time=time,load_json=load,markdown_code_pattern=r'```(?:\w+)?\n([\s\S]*?)\n```')
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cl],type_ignores=[])),str(source)+'::UNCHANGED_CLASS','exec'),ns)
 return ns['LLM']

def agent(backend,task_info=None):
 Base=official_class()
 class Adapter(Base):
  def initialize_llm(self,llm_path,top_k,top_p,temperature,max_tokens):
   assert (top_k,top_p,temperature,max_tokens)==(50,1.0,.1,8192)
   return None,None,dict(top_k=top_k,top_p=top_p,temperature=temperature,max_tokens=max_tokens),False
  def inference(self,query,system_prompt=None):return self.batch_inference([query],system_prompt)[0]
  def batch_inference(self,queries,system_prompt=None):
   check();phase=next((x.function for x in inspect.stack() if x.function in ['evaluate','decision_making','self_reflection','update_memory']),'unknown');assert phase!='unknown'
   messages=[[dict(role='system',content=system_prompt if system_prompt is not None else self.system_prompt),dict(role='user',content=q)] for q in queries]
   return backend.chat(messages,phase)
 return Adapter('local/Qwen/Qwen3.5-2B' if backend.kind=='student' else 'DeepSeek/official/deepseek-flash',batch_size=4,task_info=task_info,use_reflection=True)

def runtime_dir(unit):
 runtime=unit/'runtime';runtime.mkdir(parents=True,exist_ok=True)
 for name,target in [('prompts',S/'prompts'),('UST_tasks/road_planning/Data',R/'runs/2026-11-10-ust-original-env-audit-v1/source/UST_tasks/road_planning/Data')]:
  p=runtime/name;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():p.symlink_to(target,target_is_directory=True)
 os.chdir(runtime)
 return runtime

def run_online(backend,unit):
 # Original source import only remaps legacy public pickle class names; CUDA visibility restored before model loading.
 cv=os.environ.get('CUDA_VISIBLE_DEVICES','0')
 import ust_original_env_audit_20261110 as native
 os.environ['CUDA_VISIBLE_DEVICES']=cv;runtime_dir(unit)
 e=native.RoadPlanningEnvironment(native.Config('demo','CapeTown1',1,False,str(unit/'native-logs')))
 llm=agent(backend,load(S/'UST_tasks/road_planning/Data/task_info.json'));history=[];game_start=True
 backend.part='online';backend.unit=unit
 save(unit/'online/started.json',dict(epoch=time.time(),map='CapeTown1',reflection=True,memory_size=3))
 while not e.game_over:
  check();assert e.step_count<38
  e.step(llm,game_start,history);game_start=False
  rec=history[-1];decision=rec['answer'];selected=int(e.env._action_history[-1]);legal_before=[int(x) for x in re.findall(r'^road (\d+):',rec['data_text'],re.M)]
  match=re.fullmatch(r'road[ _](\d+)',str(decision).lower());requested=int(match[1]) if match else None
  save(unit/'online/steps'/f'{e.step_count:03}.json',dict(step=e.step_count,history=copy.deepcopy(rec),memory=copy.deepcopy(llm.memory),action=selected,model_requested_action=requested,format_or_illegal_fallback=requested is None or requested!=selected,after=native.fingerprint(e.env),epoch=time.time()))
  atomic(unit/'live.json',dict(part='online',closed_steps=e.step_count,expected_steps=38,epoch=time.time()))
 metrics=e.get_metrics();assert len(history)==38
 save(unit/'online/terminal.json',dict(status='COMPLETE_ORIGINAL_ONLINE',steps=38,metrics=dict(zip(['number_of_road_segments','average_distance','sum_of_costs'],map(float,metrics))),fallbacks=sum(load(p)['format_or_illegal_fallback'] for p in (unit/'online/steps').glob('*.json')),epoch=time.time()))
 print('ONLINE_COMPLETE',backend.kind,metrics,flush=True)

def run_qa(backend,unit):
 runtime_dir(unit);backend.part='qa';backend.unit=unit;llm=agent(backend)
 data=load(R/'runs/2026-11-07-ust-benchmark-anatomy-v1/public-release/road_planning__planning_QA.json');assert len(data)==3000
 results=[]
 for start in range(0,len(data),100):
  check();chunk=data[start:start+100];answers,acc,st=llm.evaluate(copy.deepcopy(chunk),'planning')
  save(unit/'qa/chunks'/f'{start:04}.json',dict(start=start,count=len(chunk),official_accuracy=acc,answers=answers,epoch=time.time()))
  if hasattr(backend,'rng'):backend.rng(unit/'qa/rng'/f'{start:04}.pt')
  results+=answers;atomic(unit/'live.json',dict(part='qa',closed=len(results),total=3000,correct_so_far=sum(x['is_correct'] for x in results),epoch=time.time()));print('QA',backend.kind,len(results),'/',3000,flush=True)
 score=dict(n=3000,correct=sum(x['is_correct'] for x in results),accuracy=sum(x['is_correct'] for x in results)/3000,invalid=sum(x['decision'] is None for x in results),by_stage={k:dict(n=sum(x['stage']==k for x in results),correct=sum(x['stage']==k and x['is_correct'] for x in results)) for k in sorted({x['stage'] for x in results})})
 save(unit/'qa/terminal.json',dict(status='COMPLETE_ORIGINAL_QA',**score,epoch=time.time()))

class Student:
 kind='student'
 def __init__(self,unit):self.unit=unit;self.part='online';self.serial=0;self.model=None
 def initialize(self):
  import eval_ust_sg_qwen_modes_20261024 as config
  config.setup_native()
  import torch
  from transformers import AutoTokenizer,AutoModelForImageTextToText
  p=load(O/'base-assets.json');base=R/p['base']
  for x in p['files']:assert pin(base/x['name'])['sha256']==x['sha256']
  assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>30*2**30
  torch.set_num_threads(4);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
  self.torch=torch;self.tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False,padding_side='left')
  self.model=AutoModelForImageTextToText.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
  assert not hasattr(self.model,'peft_config');self.model.requires_grad_(False)
  save(self.unit/'model-loaded.json',dict(status='UNMODIFIED_BASE_NO_ADAPTER',base_assets=pin(O/'base-assets.json'),torch=torch.__version__,parameters=sum(x.numel() for x in self.model.parameters()),requires_grad=any(x.requires_grad for x in self.model.parameters()),GPU=torch.cuda.get_device_name(),epoch=time.time()))
 def rng(self,p):p.parent.mkdir(parents=True,exist_ok=True);self.torch.save(dict(cpu=self.torch.get_rng_state(),cuda=self.torch.cuda.get_rng_state_all()),p)
 def chat(self,messages,phase):
  if self.model is None:self.initialize()
  torch=self.torch;tok=self.tok;model=self.model;check()
  texts=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
  ids=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda');assert ids['input_ids'].shape[1]+8192<=32768,'Context ceiling; no silent clipping'
  ids_start=[]
  for m in messages:
   serial=self.serial;self.serial+=1;ids_start.append(serial);save(self.unit/'raw'/f'{serial:05}.started.json',dict(serial=serial,part=self.part,phase=phase,messages=m,epoch=time.time()))
  model.model.rope_deltas=None;started=time.monotonic();stops=[tok.convert_tokens_to_ids(x) for x in ('<|im_end|>','<|endoftext|>')]
  with torch.inference_mode():out=model.generate(**ids,max_new_tokens=8192,do_sample=True,temperature=.1,top_k=50,top_p=1.,num_beams=1,eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)
  torch.cuda.synchronize();elapsed=time.monotonic()-started;responses=[]
  for i,serial in enumerate(ids_start):
   tokens=out[i,ids['input_ids'].shape[1]:].tolist();stop=next((j for j,t in enumerate(tokens) if t in stops),None)
   if stop is not None:tokens=tokens[:stop+1]
   text=tok.decode(tokens,skip_special_tokens=True);responses.append(text)
   save(self.unit/'raw'/f'{serial:05}.json',dict(serial=serial,part=self.part,phase=phase,content=text,raw_generated=tok.decode(tokens,skip_special_tokens=False),response_ids=tokens,prompt_tokens=int(ids['attention_mask'][i].sum()),output_tokens=len(tokens),finish_reason='stop' if stop is not None else 'length',batch_seconds=elapsed,batch_size=len(messages),epoch=time.time()))
  model.model.rope_deltas=None;return responses

def student_main():
 import fcntl
 unit=O/'student-base2B';unit.mkdir(exist_ok=True);verify()
 with (R/'runs/2026-11-05-ust-environment-generator-ipd-v1/full-schema-mainline-v3/gpu.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);b=Student(unit);run_online(b,unit);run_qa(b,unit)
 save(unit/'terminal.json',dict(status='COMPLETE_ORIGINAL_BASELINE',epoch=time.time()))

if __name__=='__main__':
 try:student_main()
 except BaseException as e:
  save(O/'student-base2B'/f'failure-{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
