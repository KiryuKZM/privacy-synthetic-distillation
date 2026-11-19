"""Fixed33-endpoint SAME1569 local Test: inference-only, raw-first and once scoring.
No external API, training, checkpoint selection, protected-data transport or app control.
"""
from pathlib import Path
import argparse,fcntl,gc,hashlib,json,os,re,resource,signal,subprocess,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from three_method_native_ipd_v4_20261101 import e
import prepare_three_method_common_test_v4_20261101 as qb
from privacy_synth.choice_answer import parse_choice,strict_format_valid
O=e.O;P=O/'common-eval-v1';TEST=qb.b.P;COUNT=1569;CAP=32;CONTEXT=16384
STUDENT_DEP=O/'student-controller-v5/terminal.json';IPD_DEP=O/'latest-ipd-controller-v4/terminal.json'
CHILD=None;STOP=False;AWAKE=None;AWAKE_DIR=None;AWAKE_SERIAL=0;OUTPUT_USED=None
load,bind=e.load,e.bind

def stop(*_):
 global STOP;STOP=True;e.stop();e.m.stop()

def check():
 if STOP or (P/'STOP').exists():raise InterruptedError('Owned common-evaluation STOP')
 e.check_stop()

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def output_bytes():
 global OUTPUT_USED
 if OUTPUT_USED is None:OUTPUT_USED=sum(x.stat().st_size for x in P.rglob('*') if x.is_file()) if P.exists() else 0
 return OUTPUT_USED

def write_record(path,value,exclusive):
 global OUTPUT_USED
 path=Path(path);size=len(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode())
 old=path.stat().st_size if path.exists() else 0
 if exclusive and old:raise FileExistsError('Immutable record exists: '+str(path))
 assert output_bytes()+size-old<3*2**30-16*2**20,'Own3GiB cap with16MiB helper/temp reserve'
 (e.once if exclusive else e.atomic)(path,value)
 OUTPUT_USED+=path.stat().st_size-old

def once(path,value):return write_record(path,value,True)
def atomic(path,value):return write_record(path,value,False)

def guard():
 check();s=e.snapshot()
 if output_bytes()>=3*2**30:raise BlockingIOError('Own whole common-eval3GiB output boundary')
 return s

def dependency_records():
 a=load(STUDENT_DEP);b=load(IPD_DEP)
 assert a['status']=='ALL18_W_OFF_ON_UNITS_FIT_RELOAD_COMPLETE_IPD_AND_COMMON_TEST_STILL_PENDING' and len(a['units'])==18
 assert b['status']=='ALL6_LATEST_NATIVE_QWEN_IPD50_ACTUAL_UPDATES_AND_INDEPENDENT_RELOAD_COMPLETE' and len(b['results'])==6
 # Includes all24 real manifests, independent reloads and actual same-teacher qualification.
 endpoints=qb.b.actual_endpoints_ready();assert len(endpoints)==24
 return [bind(STUDENT_DEP),bind(IPD_DEP)],endpoints

def validate_questions(qs):
 assert len(qs)==COUNT and len({q['id'] for q in qs})==COUNT
 assert [q['ordinal'] for q in qs]==list(range(COUNT))
 assert all(q['messages']==[dict(role='system',content=q['system']),dict(role='user',content=q['prompt'])] for q in qs)
 assert not any('answer' in q or 'gold' in q for q in qs),'No gold in inference query'
 assert len({q['room_ordinal'] for q in qs})==523
 return qs

def verify_old(endpoint):
 for name,st in endpoint['base_stat'].items():
  p=Path(endpoint['base'])/name;s=p.stat()
  assert dict(bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,device=s.st_dev,resolved_path=str(p.resolve()))==st
 for item in endpoint.get('files',[]):assert bind(R/item['path'])==item
 return endpoint

def prepare():
 assert not (P/'root-delegation.json').exists()
 proof=load(P/'CPU-proof.json');assert proof['source']==bind(Path(__file__)) and proof['status']=='COMMON_EVAL_CPU_CONTROL_TESTS_PASS'
 pins=[Path(__file__),R/'tests/test_three_method_common_eval_20261101.py',P/'CPU-proof.json',R/'scripts/prepare_three_method_common_test_v4_20261101.py',R/'scripts/prepare_three_method_common_test_20261101.py',
  TEST/'proposal.json',TEST/'current-dependency-root.json',TEST/'old-endpoints8-frozen.json',e.P/'root-delegation.json',e.P/'profile-proposal.json',R/'src/privacy_synth/choice_answer.py',
  R/'scripts/three_method_native_ipd_v4_20261101.py',R/'scripts/three_method_native_ipd_20261101.py',R/'scripts/three_method_native_teacher_client_20261101.py',
  R/'scripts/eval_ust_sg_qwen_modes_20261024.py',R/'scripts/keep_ust_sg_windows_awake_20261024.py',R/'runs/2026-10-31-gpu-app-close-permission-revoked.json']
 once(P/'root-delegation.json',dict(approved=True,actor='Root',source=bind(Path(__file__)),bindings=[bind(x) for x in pins],
  authority='Explicit new-generator three-method user goal: same questions, all original/old/new students and actual latest qualified Qwen teacher; continuous in-scope execution within original500USD',
  activation='Actual18 W/off/on and6 latest-native IPD fit+independent reload success, actual same Q3 teacher probability/gradient profile, then existing fixed1569 builder; all33 settings and all51777 raw before once gold read',
  unit=UNIT,endpoints=33,questions_each=COUNT,planned_raw=33*COUNT,rooms=523,greedy=True,thinking=False,context_cap=CONTEXT,maximum_new_tokens=CAP,
  no_truncation_refill_or_answer_selection=True,no_optimizers_or_training=True,no_unknown_started_replay=True,no_other_app_closure=True,optional_own_idle_cache='only exact current unit and procs self after new exit0 child, GPUlock, hard14GiB/noSwap unchanged; MemoryHigh512MiB then restore original in finally within120s; no shared/global/system/app control or synthetic progress',
  original500USD_total_nonadditional=True,API_calls=0,maximum_endpoint_seconds=86400,maximum_controller_seconds=7*86400,
  resources='singleGPUlock/sharedCPU4host14GiBnoSwap/CUDA9/startWin30run20/GPU11util10quiet20/temp80/disk20/ownwholeoutput3GiB',actual_Test_payload_read=False))
 print('COMMON33x1569_ROOT_DELEGATION_BOUND_WAIT_ACTUAL24_RELOADS_NO_TEST_READ',flush=True)

def authorize():
 g=load(P/'root-delegation.json');assert g['approved'] and g['source']==bind(Path(__file__))
 for b in g['bindings']:assert bind(R/b['path'])==b,b['path']
 check();return g

def activate():
 authorize();deps,new=dependency_records();t=load(TEST/'terminal.json')
 assert t['status']=='SAME1569_LOCAL_QUERIES_AND_SEPARATE_FACT_GOLD_FROZEN_NO_DECODE_OR_SCORE' and t['questions']==COUNT
 assert bind(R/t['queries']['path'])==t['queries']
 qs=validate_questions(rows(R/t['queries']['path']))
 olds=load(TEST/'old-endpoints8-frozen.json')['endpoints'];assert len(olds)==8
 endpoints=[dict(verify_old(x),engine='HF_native',origin='old_original') for x in olds]
 for x in new:
  u=R/x['unit'];term=load(u/'fit-terminal.json');proto=load(u/'protocol.json');folder=Path(term['checkpoint'])
  assert bind(folder/'manifest.json')==x['manifest'] and bind(u/'independent-reload.json')==x['reload']
  endpoints.append(dict(id=f"new-{x['generator']}-s{x['seed']}-{x['stage']}",kind='new_trained_actor',engine='HF_native',origin='new',
   base=proto['base'],adapter_directory=str(folder/'actor'),manifest=x['manifest'],reload=x['reload'],fit_terminal=bind(u/'fit-terminal.json'),protocol=bind(u/'protocol.json'),
   actor_role_hash=term['final_actor'],base_asset_stat=proto['base_asset_stat'],generator=x['generator'],seed=x['seed'],stage=x['stage'],size='2B'))
 qual=e.P/'probability-qualification-v1/terminal.json';q=load(qual);profile=e.P/'qualified-profile.json'
 assert q['status']=='ACTUAL_LATEST_QWEN_NATIVE_PROBABILITY_CODEC_CACHE_AND_COMBINED_MEMORY_PASS' and q['artifact']['stat']==e.file_stat(e.MODEL)
 endpoints.append(dict(id='teacher-Qwen3.8-27B-Q3_K_M-native-off',kind='same_actual_IPD_teacher',engine='native_teacher',origin='teacher',
  model_artifact=q['artifact'],qualified_profile=bind(profile),actual_probability_qualification=bind(qual),profile=load(profile),size='27B_Q3_K_M'))
 assert len(endpoints)==len({x['id'] for x in endpoints})==33
 for x in endpoints:
  x.update(context_cap=CONTEXT,maximum_new_tokens=CAP,greedy=True,thinking=False,questions=t['queries'],question_count=COUNT)
  once(P/'endpoints'/x['id']/'settings.json',x)
 settings=[bind(P/'endpoints'/x['id']/'settings.json') for x in endpoints]
 once(P/'all33-settings-frozen.json',dict(status='ALL33_ACTUAL_ENDPOINT_SETTINGS_AND_SAME1569_QUERIES_FROZEN_BEFORE_FIRST_DECODE',
  actual_dependencies=deps,test_terminal=bind(TEST/'terminal.json'),settings=settings,questions=t['queries'],expected_answers=33*COUNT,API_calls=0,epoch=time.time()))
 return settings

def endpoint_settings(identifier):
 authorize();frozen=load(P/'all33-settings-frozen.json');assert frozen['expected_answers']==33*COUNT
 for b in frozen['actual_dependencies']:assert bind(R/b['path'])==b
 spec=next(x for x in frozen['settings'] if Path(x['path']).parent.name==identifier)
 assert bind(R/spec['path'])==spec;v=load(R/spec['path']);assert v['id']==identifier
 assert bind(R/v['questions']['path'])==v['questions'];return v

def request_paths(out,index):
 assert type(index) is int and 0<=index<COUNT
 raw=out/'raw'/f'{index:04d}.json';return raw,raw.with_suffix('.started.json')

def request_state(out,index):
 raw,start=request_paths(out,index)
 if raw.exists():
  value=load(raw);assert start.exists() and value['request']==load(start) and value['ordinal']==index
  return 'closed'
 if start.exists():return 'unknown'
 return 'unstarted'

def context_failure(prefix):
 return dict(status='CONTEXT_OVERFLOW_NO_DECODE',prompt_tokens=len(prefix),maximum_new_tokens=CAP,context_cap=CONTEXT,
  content='',raw_generated='',response_ids=[],finish_reason='context_overflow',generated_tokens=0,request_wall_seconds=0.,no_clipping_or_refill=True)

def native_greedy(teacher,tok,prefix):
 if len(prefix)+CAP>CONTEXT:return context_failure(prefix)
 begin=time.monotonic();cur=list(prefix);emitted=[];calls=[];stops={tok.convert_tokens_to_ids(x) for x in ('<|im_end|>','<|endoftext|>')}
 for _ in range(CAP):
  check();value=teacher.score(cur);maximum=max(value['top16_raw_logits'])
  token=min(i for i,z in zip(value['top16_ids'],value['top16_raw_logits']) if z==maximum)
  emitted.append(token);calls.append({k:value[k] for k in ('new_decoded_tokens','decode_seconds')})
  if token in stops:break
  cur.append(token)
 return dict(status='GENERATED',content=tok.decode(emitted,skip_special_tokens=True),raw_generated=tok.decode(emitted,skip_special_tokens=False),response_ids=emitted,
  prompt_tokens=len(prefix),generated_tokens=len(emitted),finish_reason='stop' if emitted[-1] in stops else 'length',request_wall_seconds=time.monotonic()-begin,
  native_prefill_seconds=calls[0]['decode_seconds'],native_decode_seconds=sum(x['decode_seconds'] for x in calls[1:]),native_prefill_tokens=calls[0]['new_decoded_tokens'],native_decode_steps=len(calls)-1,
  measured_native_calls=calls,CPU_RPC_and_sampling_overhead_in_request_wall=True,model_temperature='greedy_argmax_semantic_domain',logit_domain=248077)

def hf_greedy(torch,tok,model,prefix):
 if len(prefix)+CAP>CONTEXT:return context_failure(prefix)
 stops=[tok.convert_tokens_to_ids(x) for x in ('<|im_end|>','<|endoftext|>')];events=[];active=[]
 def before(module,args,kwargs):
  start=torch.cuda.Event(enable_timing=True);start.record();active.append(start)
 def after(module,args,kwargs,result):
  end=torch.cuda.Event(enable_timing=True);end.record();events.append((active.pop(),end))
 forward_target=model.get_base_model() if hasattr(model,'peft_config') else model
 hooks=[forward_target.register_forward_pre_hook(before,with_kwargs=True),forward_target.register_forward_hook(after,with_kwargs=True)]
 begin=time.monotonic()
 try:
  ids=torch.tensor([prefix],device='cuda')
  with torch.inference_mode():value=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=CAP,do_sample=False,num_beams=1,eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)
  torch.cuda.synchronize();wall=time.monotonic()-begin;emitted=value[0,len(prefix):].tolist();assert 0<len(emitted)<=CAP
  measured=[a.elapsed_time(b)/1000 for a,b in events]
  timing_matches=len(measured)==len(emitted)  # Valid generated answers are retained even if backend bypasses hooks.
  return dict(status='GENERATED',content=tok.decode(emitted,skip_special_tokens=True),raw_generated=tok.decode(emitted,skip_special_tokens=False),response_ids=emitted,prompt_tokens=len(prefix),generated_tokens=len(emitted),
   finish_reason='stop' if emitted[-1] in stops else 'length',request_wall_seconds=wall,GPU_prefill_forward_seconds=measured[0] if timing_matches else None,GPU_decode_forward_seconds=sum(measured[1:]) if timing_matches else None,GPU_decode_forward_steps=len(measured)-1 if timing_matches else None,forward_timing_complete=timing_matches,
   forward_GPU_seconds=measured,measurement='CUDA events around actual forward hooks; no per-token synchronization, synchronized end-of-request. Includes hooks; excludes guards/pacing/writes. GPU-forward time is not full application latency.',logit_domain=248320)
 finally:
  for h in hooks:h.remove()

def load_hf(s):
 import eval_ust_sg_qwen_modes_20261024 as config
 config.setup_native()
 import torch
 from transformers import AutoTokenizer,AutoModelForImageTextToText
 from peft import PeftModel
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
 if 'base_stat' in s:verify_old(s)
 else:
  for name,info in s['base_asset_stat'].items():
   st=(Path(s['base'])/name).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==tuple(info)
  for key in ('manifest','reload','fit_terminal','protocol'):assert bind(R/s[key]['path'])==s[key]
 tok=AutoTokenizer.from_pretrained(s['base'],local_files_only=True,trust_remote_code=False)
 model=AutoModelForImageTextToText.from_pretrained(s['base'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
 if s['adapter_directory']:
  model=PeftModel.from_pretrained(model,s['adapter_directory'],adapter_name='actor',is_trainable=False)
  assert e.m.lib.role_hash(model,'actor')==s['actor_role_hash']
 for p in model.parameters():p.requires_grad_(False)
 model.eval();model.config.use_cache=True
 return torch,tok,model

def closure(out,settings):
 qs=validate_questions(rows(R/settings['questions']['path']));raws=[]
 for index,q in enumerate(qs):
  assert request_state(out,index)=='closed',f'Question {index} is unstarted/unknown'
  p,_=request_paths(out,index);v=load(p)
  assert v['id']==q['id'] and v['query_SHA256']==digest(q) and v['request']['settings']==bind(out/'settings.json')
  assert v['ordinal']==index and v['result']['status'] in ('GENERATED','CONTEXT_OVERFLOW_NO_DECODE')
  raws.append(bind(p))
 return raws

def run_endpoint(identifier):
 s=endpoint_settings(identifier);out=P/'endpoints'/identifier
 assert not (out/'model-started.json').exists(),'No implicit completed or unknown model replay'
 qs=validate_questions(rows(R/s['questions']['path']));assert all(request_state(out,i)=='unstarted' for i in range(COUNT))
 e.preflight();guard();once(out/'model-started.json',dict(settings=bind(out/'settings.json'),epoch=time.time(),source=bind(Path(__file__))))
 begin=time.monotonic();model=None;teacher=None
 try:
  if s['engine']=='HF_native':torch,tok,model=load_hf(s)
  else:
   from transformers import AutoTokenizer
   assert bind(R/s['qualified_profile']['path'])==s['qualified_profile'] and s['profile']==load(e.P/'qualified-profile.json') and s['model_artifact']['stat']==e.file_stat(e.MODEL)
   tok=AutoTokenizer.from_pretrained(load(e.m.unit('seed01',e.m.SEEDS[0],'warm')/'protocol.json')['base'],local_files_only=True,trust_remote_code=False)
   teacher=e.teacher_open(out);assert teacher.template_SHA256==e.TEACHER_TEMPLATE_SHA
  setup=time.monotonic()-begin;once(out/'actual-model-loaded.json',dict(setup_seconds=setup,settings=bind(out/'settings.json'),after_load_resource=guard(),epoch=time.time()))
  for index,q in enumerate(qs):
   check()
   if time.monotonic()-begin>86400:raise TimeoutError('Bounded24h endpoint; retain closedraw, no unknown replay')
   cooldown=e.cooldown();tick=time.monotonic();time.sleep(.5);pace=time.monotonic()-tick;frame=guard()
   text=tok.apply_chat_template(q['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   prefix=tok.encode(text,add_special_tokens=False)
   assert prefix and min(prefix)>=0
   request=dict(settings=bind(out/'settings.json'),id=q['id'],ordinal=index,query_SHA256=digest(q),prefix_SHA256=digest(prefix),prompt_tokens=len(prefix),epoch=time.time())
   raw,start=request_paths(out,index);once(start,request)
   result=hf_greedy(torch,tok,model,prefix) if model is not None else native_greedy(teacher,tok,prefix)
   once(raw,dict(id=q['id'],ordinal=index,query_SHA256=digest(q),request=request,result=result,guard_before=frame,cooling_seconds=cooldown,pacing_seconds=pace))
   atomic(P/'progress.json',dict(kind='actual_same_Test_answer_commit',endpoint=identifier,committed=index+1,total=COUNT,epoch=time.time()))
   if index%50==49:print('ACTUAL_COMMON_TEST',identifier,index+1,'/',COUNT,flush=True)
  evidence=closure(out,s)
  peak=dict(CPU_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,last_actual_resource=guard())
  if model is not None:peak.update(CUDA_allocated_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved())
  once(out/'terminal.json',dict(status='ALL1569_SAME_TEST_RAW_FROZEN_NO_GOLD_READ',settings=bind(out/'settings.json'),raw=evidence,count=COUNT,setup_seconds=setup,endpoint_wall_seconds=time.monotonic()-begin,resources=peak,API_calls=0,optimizer_updates=0,gold_read=False,epoch=time.time()))
 finally:
  if teacher is not None:teacher.close()
  if model is not None:del model;gc.collect();torch.cuda.empty_cache()

def all_raw_closed():
 frozen=load(P/'all33-settings-frozen.json');terms=[]
 assert len(frozen['settings'])==33
 for item in frozen['settings']:
  assert bind(R/item['path'])==item;s=load(R/item['path']);out=(R/item['path']).parent;t=load(out/'terminal.json')
  assert t['status']=='ALL1569_SAME_TEST_RAW_FROZEN_NO_GOLD_READ' and t['settings']==item and len(t['raw'])==COUNT and t['count']==COUNT
  assert closure(out,s)==t['raw'];terms.append(bind(out/'terminal.json'))
 return terms

def score():
 authorize();terminals=all_raw_closed();assert not (P/'scoring.started.json').exists(),'Unknown scoring is never replayed'
 once(P/'scoring.started.json',dict(actual_all51777_raw_closed=terminals,epoch=time.time(),source=bind(Path(__file__))))
 # First interpretation/read of sealed answer content by this inference/scoring system.
 t=load(TEST/'terminal.json');assert bind(R/t['gold']['path'])==t['gold'];gold=rows(R/t['gold']['path']);qs=validate_questions(rows(R/t['queries']['path']))
 assert len(gold)==COUNT and [x['id'] for x in gold]==[x['id'] for x in qs] and all(x['answer'] in ('A','B','C','D','E') for x in gold)
 summary=[]
 for setting in load(P/'all33-settings-frozen.json')['settings']:
  s=load(R/setting['path']);out=(R/setting['path']).parent;values=[load(request_paths(out,i)[0]) for i in range(COUNT)];correct=valid=strict=strictcorrect=truncated=overflow=0;families={};tokens=inputs=0;wall=pace=cool=0.
  for q,g,v in zip(qs,gold,values):
   assert v['id']==q['id']==g['id'];r=v['result'];pred=parse_choice(r['content']);ok=pred==g['answer'];fmt=strict_format_valid(r['content'])
   correct+=ok;valid+=pred is not None;strict+=fmt;strictcorrect+=fmt and ok;truncated+=r['finish_reason']=='length';overflow+=r['status']=='CONTEXT_OVERFLOW_NO_DECODE'
   a=families.setdefault(q['family'],dict(correct=0,valid=0,total=0));a['correct']+=ok;a['valid']+=pred is not None;a['total']+=1
   tokens+=r['generated_tokens'];inputs+=r['prompt_tokens'];wall+=r['request_wall_seconds'];pace+=v['pacing_seconds'];cool+=v['cooling_seconds']
  summary.append(dict(id=s['id'],origin=s['origin'],kind=s['kind'],size=s['size'],generator=s.get('generator'),seed=s.get('seed'),stage=s.get('stage'),correct=correct,total=COUNT,accuracy=correct/COUNT,semantic_valid=valid,strict_format_valid=strict,strict_format_correct=strictcorrect,
   truncated=truncated,context_overflow=overflow,families=families,input_tokens=inputs,output_tokens=tokens,requests_wall_seconds=wall,output_tokens_per_request_wall_second=tokens/wall if wall else None,pacing_seconds=pace,cooling_seconds=cool,terminal=bind(out/'terminal.json')))
 once(P/'scores.json',dict(status='ALL33x1569_ONCE_SCORED',rows=summary,raw_answers=33*COUNT,questions=COUNT,rooms=523,once=True,API_calls=0,privacy_proof=False,causal_generator_claim=False,epoch=time.time()))
 groups=[]
 for gen in e.m.GENS:
  for stage in ('warm','off','on','ipd'):
   selected=[x for x in summary if x['generator']==gen and x['stage']==stage]
   assert len(selected)==3
   accuracies=[x['accuracy'] for x in selected]
   groups.append(dict(generator=gen,stage=stage,seeds=[x['seed'] for x in selected],correct=[x['correct'] for x in selected],total_per_seed=COUNT,mean_accuracy=sum(accuracies)/3,minimum_accuracy=min(accuracies),maximum_accuracy=max(accuracies),no_bestseed=True))
 once(P/'three-seed-groups.json',dict(groups=groups,variation='three initialization seeds; both generator seeds reported separately; no uncertainty/cause/privacy claim'))
 header='|端点|正确/1569|有效/1569|严格格式/1569|截断|超长失败|请求秒|输出token/请求秒|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 table=''.join(f"|{x['id']}|{x['correct']}/1569 ({100*x['accuracy']:.2f}%)|{x['semantic_valid']}/1569|{x['strict_format_valid']}/1569|{x['truncated']}|{x['context_overflow']}|{x['requests_wall_seconds']:.2f}|{x['output_tokens_per_request_wall_second'] or 0:.2f}|\n" for x in summary)
 report=R/'reports/new-generator-three-method-same-Test-final-20261101.md'
 with report.open('x') as f:f.write('# 同一冻结空间题集：原始、旧端点、新生成器三法与实际Qwen教师\n\n523房间×3任务=1569题；33端点全部51777条原始回答冻结后一次评分。thinking off/greedy/原生分词/max32/16k；超长与无效均保留分母。\n\n'+header+table+'\n\n|生成器|分支|三seed正确/各1569|均值|范围|\n|---|---|---|---:|---|\n'+''.join(f"|{x['generator']}|{x['stage']}|{x['correct']}|{100*x['mean_accuracy']:.2f}%|{100*x['minimum_accuracy']:.2f}–{100*x['maximum_accuracy']:.2f}%|\n" for x in groups)+'\n请求速率含真实整题生成，另有原始prefill/decode计时；不是训练速率。旧2B时间场景桥接、旧0.8B与新2B、DS和Qwen教师/数据/算力不同，不能单独归因生成器或当隐私证明。历史其他题集的分数单独保留。\n')
 once(P/'terminal.json',dict(status='ALL33x1569_COMMON_TEST_ONCE_SCORED_REPORT_WRITTEN',scores=bind(P/'scores.json'),report=bind(report),all_raw_terminal=terminals,API_calls=0,epoch=time.time()))

def awake_begin():
 global AWAKE,AWAKE_DIR,AWAKE_SERIAL
 if AWAKE is not None and AWAKE.poll() is None:return
 if not (P/'progress.json').exists() or load(P/'progress.json').get('kind')!='actual_same_Test_answer_commit':return
 AWAKE_SERIAL+=1;AWAKE_DIR=P/'awake'/f'{AWAKE_SERIAL:04d}';AWAKE_DIR.mkdir(parents=True,exist_ok=False)
 atomic(AWAKE_DIR/'goal.json',dict(status='ACTIVE'))
 root=r'\\wsl.localhost\Ubuntu-24.04\home\kiryu\privacy-synthetic-distillation'
 def win(p):return root+'\\'+str(p.relative_to(R)).replace('/','\\')
 AWAKE=subprocess.Popen(['/mnt/c/Users/10950/miniconda3/python.exe','-X','utf8',root+r'\scripts\keep_ust_sg_windows_awake_20261024.py','--heartbeat',win(P/'progress.json'),'--goal-state',win(AWAKE_DIR/'goal.json'),'--stop-file',win(O/'STOP'),'--stop-file',win(P/'STOP'),'--stop-file',win(AWAKE_DIR/'STOP'),'--receipt',win(AWAKE_DIR/'receipt.json'),'--poll','5','--stale','600'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def awake_end():
 global AWAKE,AWAKE_DIR
 if AWAKE_DIR is not None:
  atomic(AWAKE_DIR/'goal.json',dict(status='COMPLETE'));atomic(AWAKE_DIR/'STOP',dict(reason='Actual inference child closed/stopped'))
 if AWAKE is not None:
  try:AWAKE.wait(timeout=12)
  except subprocess.TimeoutExpired:atomic(P/'awake-cleanup-pending.json',dict(owner_receipt=str(AWAKE_DIR/'receipt.json'),no_PID_only_kill=True,epoch=time.time()))
 AWAKE=None;AWAKE_DIR=None

UNIT='codex-three-method-common-test-eval-20261101-v1.service'

def own_boundary_cache(closed_key,proof_path):
 """Optional bounded OWN idle-cache soft pressure after a real child exit.
 Never treats optional cleanup or a single Win sample as model startup success.
 """
 proof=load(proof_path);assert proof['exit_code']==0 and CHILD is None
 folder=P/'cache-boundaries'/closed_key
 if (folder/'receipt.json').exists():raise PermissionError('No repeated own-boundary action')
 assert os.environ.get('INVOCATION_ID'),'Actual current own service identity required'
 name=next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
 cg=Path('/sys/fs/cgroup')/name.lstrip('/');assert cg.name==UNIT and os.getuid()!=0
 inode=cg.stat().st_ino;pid=os.getpid()
 def identity():
  assert cg.stat().st_ino==inode and set(map(int,(cg/'cgroup.procs').read_text().split()))=={pid}
  assert int((cg/'memory.max').read_text())==14*2**30 and (cg/'memory.swap.max').read_text().strip()=='0' and (cg/'memory.swap.current').read_text().strip()=='0'
  return dict((k,int(v)) for k,v in (x.split() for x in (cg/'memory.stat').read_text().splitlines()))
 identity();initial=e.snapshot();stat=identity()
 if initial['Windows']['windows_free_gib']>=30 or stat['file_mapped'] or stat['file']<128*2**20:
  once(folder/'receipt.json',dict(status='OPTIONAL_CACHE_NO_WRITE',control_writes=0,actual_closed_child=bind(proof_path),initial=initial,stat=stat,no_model_progress=True));return
 with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:
   once(folder/'receipt.json',dict(status='OPTIONAL_CACHE_NO_WRITE_GPU_LOCK_BUSY',control_writes=0,actual_closed_child=bind(proof_path)));return
  identity();old=(cg/'memory.high').read_text().strip();once(folder/'reserved.json',dict(cgroup=name,inode=inode,PID=pid,invocation=os.environ.get('INVOCATION_ID'),source=bind(Path(__file__)),closed_child=bind(proof_path),soft_bytes=512*2**20,hard14GiB_Swap0_unchanged=True,maximum_seconds=120,original_high=old,stat_before=stat,initial=initial))
  tick=time.monotonic();samples=[];error=None;writes=0
  try:
   identity();(cg/'memory.high').write_text(str(512*2**20));writes+=1
   while time.monotonic()-tick<120:
    check();current=identity();frame=e.snapshot();samples.append(dict(epoch=time.time(),Win=frame['Windows']['windows_free_gib'],current_bytes=int((cg/'memory.current').read_text()),file_bytes=current['file']))
    if frame['Windows']['windows_free_gib']>=30:break
    time.sleep(1)
  except BaseException as ex:
   error=dict(type=type(ex).__name__,message=str(ex));raise
  finally:
   identity();(cg/'memory.high').write_text(old);writes+=1
   once(folder/'receipt.json',dict(status='OWN_IDLE_SOFT_CACHE_BOUNDARY_OBSERVED_NOT_STARTUP_QUALIFICATION',control_writes=writes,error=error,seconds=time.monotonic()-tick,samples=samples,stat_after=identity(),original_high_restored=(cg/'memory.high').read_text().strip()==old,no_strict_exclusive_clean_page_or_immediate_host_return_claim=True,no_model_progress=True))

def child(action,identifier=None):
 global CHILD,OUTPUT_USED
 key=action+('-'+identifier if identifier else '');out=P/'children';cmd=[sys.executable,str(Path(__file__)),action]
 if identifier:cmd+=['--endpoint',identifier]
 before=load(P/'progress.json') if (P/'progress.json').exists() else None
 CHILD=subprocess.Popen(cmd,cwd=R);once(out/(key+'.started.json'),dict(pid=CHILD.pid,arguments=cmd,epoch=time.time()));tick=time.monotonic()
 try:
  while CHILD.poll() is None:
   check()
   if time.monotonic()-tick>90000:raise TimeoutError('Bounded actual evaluation child')
   if identifier and (P/'progress.json').exists():
    event=load(P/'progress.json')
    if event!=before and event.get('endpoint')==identifier:awake_begin();before=event
   time.sleep(1)
  code=CHILD.returncode;OUTPUT_USED=None;once(out/(key+'.json'),dict(exit_code=code,seconds=time.monotonic()-tick,epoch=time.time()))
  if code:raise RuntimeError('Actual child failed; preserve closed/unknown raw; no implicit replay: '+key)
 finally:
  if CHILD.poll() is None:
   CHILD.terminate()
   try:CHILD.wait(timeout=30)
   except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait(timeout=5)
  CHILD=None;OUTPUT_USED=None;awake_end()
 if identifier:own_boundary_cache(key,out/(key+'.json'))

def run():
 authorize();once(P/'controller-started.json',dict(invocation=os.environ.get('INVOCATION_ID'),source=bind(Path(__file__)),epoch=time.time()));begin=time.monotonic()
 while not (IPD_DEP.exists() and STUDENT_DEP.exists()):
  check()
  for path in (IPD_DEP.parent,STUDENT_DEP.parent):
   if (path/'failure.json').exists():raise RuntimeError('Actual current dependency failed: '+str(path.relative_to(R)))
   if (path/'paused.json').exists():raise InterruptedError('Actual dependency stopped; preserve')
  if time.monotonic()-begin>7*86400:raise TimeoutError('Bounded7d prerequisite window')
  atomic(P/'controller-progress.json',dict(status='WAIT_ACTUAL24_FIT_RELOADS_AND_SAME_TEACHER_QUALIFICATION',model_answers=0,API_calls=0,epoch=time.time()));time.sleep(10)
 dependency_records()
 # This invokes the previously CPU-proven sealed builder only after genuine training.
 g=load(TEST/'current-dependency-root.json');assert g['source']==bind(Path(qb.__file__))
 for b in g['bindings']:assert bind(R/b['path'])==b
 assert not (TEST/'started.json').exists() and not (TEST/'terminal.json').exists(),'No unknown sealed builder replay'
 qb.b.freeze();settings=activate()
 for item in settings:
  identifier=Path(item['path']).parent.name;child('endpoint',identifier)
 child('score')
 assert load(P/'terminal.json')['status']=='ALL33x1569_COMMON_TEST_ONCE_SCORED_REPORT_WRITTEN'
 print('ALL33_SAME1569_COMPLETE_ONCE_SCORED_REPORT_READY',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','endpoint','score']);p.add_argument('--endpoint');v=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if v.action=='prepare':prepare()
  elif v.action=='run':run()
  elif v.action=='score':score()
  else:
   assert v.endpoint
   with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_endpoint(v.endpoint)
 except BaseException as ex:
  path=P/('controller-failure.json' if v.action=='run' else 'failure-'+v.action+(('-'+v.endpoint) if v.endpoint else '')+'.json')
  if not path.exists():once(path,dict(type=type(ex).__name__,message=str(ex),epoch=time.time(),unknown_started_never_replayed=True))
  raise