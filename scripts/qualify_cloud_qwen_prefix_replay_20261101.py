"""Two-call public Max direct/replayed boundary qualification, no project records/GPU."""
from pathlib import Path
import argparse,ast,hashlib,json,math,os,re,sqlite3,ssl,sys,time,urllib.error,urllib.request
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import qwen_max_real100_20261030 as old
P=R/'runs/2026-11-01-cloud-qwen-prefix-replay-v1';UNIT='codex-cloud-qwen-prefix-replay-20261101-v1.service'
MODEL='qwen3.8-max';USER='Continue this list using English number words separated by commas.';PREFIX='one, two,'
MAX_CALLS=2;IN_RESERVE=1048576;TOL=1e-4;SCALE=10**12
load=lambda p:json.loads(p.read_text())
once=old.put

def bind(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def position(response,i):
 assert response['status']=='RESPONDED' and response['model_returned'] in ('qwen3.8-max','qwen3.8-max-0902','qwen3.8-max-2026-09-02') and not response['thinking_content_present']
 content=response['generic_response']['choices'][0]['logprobs']['content'];point=content[i];top=point['top_logprobs'];assert len(top)==5
 for x in [point,*top]:
  assert isinstance(x['token'],str) and isinstance(x['bytes'],list) and x['bytes'] and all(type(b) is int and 0<=b<=255 for b in x['bytes'])
  assert type(x['logprob']) in (int,float) and math.isfinite(x['logprob']) and x['logprob']<=1e-6
 assert len({tuple(x['bytes']) for x in top})==5
 return point

def first_text(response):
 point=position(response,0);position(response,1)
 assert not point['token'].startswith('<|'),'Control event not literal text'
 text=bytes(point['bytes']).decode('utf-8',errors='strict');assert text and text.encode()==bytes(point['bytes']) and point['token']==text
 allpoints=response['generic_response']['choices'][0]['logprobs']['content'];assert len(allpoints)==2
 actual=response['generic_response']['choices'][0]['message']['content'];assert actual==b''.join(bytes(x['bytes']) for x in allpoints).decode('utf-8',errors='strict')
 return text

def body(index,first=None):
 assert index in (0,1)
 prefix=PREFIX if index==0 else PREFIX+first_text(first)
 return dict(model=MODEL,max_tokens=2 if index==0 else 1,enable_thinking=False,stream=False,temperature=1,top_p=1,top_k=None,presence_penalty=0,repetition_penalty=1,logprobs=True,top_logprobs=5,messages=[dict(role='user',content=USER),dict(role='assistant',content=prefix,partial=True)])

def validate(value,index,first=None):
 assert value==body(index,first) and len(json.dumps(value).encode())<2048,'Only exact fixed generic fixture and actual first returned event may leave'
 return True

def compare(first,second):
 first_text(first);a=position(first,1);b=position(second,0)
 assert first['model_returned']==second['model_returned'],'No mixed returned model identity'
 aa={tuple(x['bytes']):x for x in a['top_logprobs']};bb={tuple(x['bytes']):x for x in b['top_logprobs']}
 same=set(aa)==set(bb);same_tokens=same and all(aa[k]['token']==bb[k]['token'] for k in aa)
 delta=max(abs(aa[k]['logprob']-bb[k]['logprob']) for k in aa) if same else None
 return dict(status='OBSERVED_BOUNDARY_REPLAY_PASS' if same_tokens and delta<=TOL else 'OBSERVED_BOUNDARY_REPLAY_MISMATCH',same_complete_top5_byte_set=same,same_top5_labels=same_tokens,maximum_absolute_logprob_difference=delta,frozen_absolute_tolerance=TOL,comparison_positions=1,full_hosted_tokenizer_or_future_prefix_equivalence=False,sampling_equality_not_required=True)

def authorize():
 g=load(P/'root-decision.json');assert g['approved'] and g['source']==bind(Path(__file__)) and g['maximum_HTTP']==2
 for b in g['bindings']:assert bind(R/b['path'])==b
 assert not (P/'STOP').exists()
 return g

def prepare():
 proof=load(P/'CPU-proof.json');assert proof['source']==bind(Path(__file__)) and proof['tests_passed']==9
 with sqlite3.connect(old.LEDGER.resolve().as_uri()+'?mode=ro',uri=True) as db:
  db.execute('BEGIN');cfg=load_config=json.loads(db.execute('SELECT config FROM settings').fetchone()[0]);assert cfg['global_pico']==500*SCALE
  calls,spent=db.execute("SELECT COUNT(*),COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose!='historical-smoke'").fetchone();worst=old.cost(IN_RESERVE,2)+old.cost(IN_RESERVE,1)
  combined=db.execute('SELECT COALESCE(SUM(pico),0) FROM calls').fetchone()[0]+db.execute('SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls').fetchone()[0]
  assert calls+2<=111 and spent+worst<=10*SCALE and combined+worst<=500*SCALE
  actual=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name IN ('sgqa_global500_qwen_insert','sgqa_global500_legacy_insert')").fetchall());expected={}
  for node in ast.walk(ast.parse(Path(old.__file__).read_text())):
   if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.lstrip().startswith('CREATE TRIGGER IF NOT EXISTS sgqa_global500_'):expected[re.search(r'EXISTS\s+(sgqa_global500_\w+)',node.value).group(1)]=node.value
  norm=lambda q:re.sub(r'\s+','',q.upper()).replace('IFNOTEXISTS','').rstrip(';');assert len(actual)==len(expected)==2 and all(norm(actual[n])==norm(expected[n]) for n in expected)
 once(P/'protocol.json',dict(fixed_direct_body=body(0),replay_rule='append exact first observed UTF8 event bytes to same assistant partial prefix; all5 teacher scores retained; no refills or thirdcall',absolute_logprob_tolerance=TOL,maximum_HTTP=2,maximum_output_tokens=[2,1],input_reserve_per_call=IN_RESERVE,no_project_records_stats_synthetic_Test_or_GPU=True))
 once(P/'budget-before.json',dict(original_total_USD500_nonadditional=True,prior_accounted_USD=combined/SCALE,prior_qwen_nonhistorical_calls=calls,old111HTTP_10USD_caps_unchanged=True,new_HTTP_cap2=True,maximum_new_reserved_USD=worst/SCALE,peak_cumulative_reserved_USD=(combined+worst)/SCALE,actual_input_not_estimated=True,price_USD_per_M=dict(input=1.65,output=4.951),price_checked_20261101='https://www.alibabacloud.com/help/en/model-studio/model-pricing',invoice=False,epoch=time.time()))
 pins=[Path(__file__),P/'CPU-proof.json',P/'protocol.json',P/'budget-before.json',R/'scripts/qwen_max_real100_20261030.py',R/'reports/cloud-qwen-ipd-prefix-event-qualification-20261101.md']
 once(P/'root-decision.json',dict(approved=True,actor='Root',source=bind(Path(__file__)),bindings=[bind(x) for x in pins],provider='Already authorized DashScope China endpoint',model=MODEL,maximum_HTTP=2,maximum_new_reserved_USD=worst/SCALE,data='Only hardcoded public English numbers fixture and actual first cloud token; no project fields',authority='Active latest-cloud-Qwen genuineIPD goal and delegated in-scope decisions within original500; this is bounded finite-prefix qualification, not a goal replacement',cheaper_alternative='Flash/local model cannot verify the requested latest Max hosted prefix contract',no_retries_unknown_replay_or_GPU=True,original500_cumulative=True,epoch=time.time()))
 print('PREFIX_REPLAY_TWO_CALLS_ROOT_BOUND',flush=True)

def call(index):
 assert index in (0,1);authorize();out=P/f'call-{index}.json';start=P/f'call-{index}.started.json';assert not out.exists() and not start.exists(),'No unknown or completed API replay'
 first=load(P/'call-0.json') if index else None;value=body(index,first);validate(value,index,first);once(P/f'body-{index}.json',value)
 secret=old.key();raw=json.dumps(value,separators=(',',':')).encode();h=hashlib.sha256(raw).hexdigest();budget=old.MaxBudget(old.LEDGER);row=budget.reserve('cloud-max-prefix-replay-20261101-'+str(index),'ipd-public-prefix-replay',MODEL,h,IN_RESERVE,value['max_tokens'])
 once(start,dict(index=index,ledger_id=row,body_sha256=h,epoch=time.time(),one_HTTP_only=True));tick=time.monotonic();usage=None;result=dict(index=index,requested=MODEL,ledger_id=row,body_sha256=h)
 try:
  req=urllib.request.Request(old.URL,data=raw,method='POST',headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'});opener=urllib.request.build_opener(old.NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
  with opener.open(req,timeout=60) as response:answer=json.loads(response.read(2**20));result['http_status']=response.status
  u=answer.get('usage',{})
  if all(type(u.get(k)) is int for k in ('prompt_tokens','completion_tokens')):usage={k:u[k] for k in ('prompt_tokens','completion_tokens')}
  returned=answer.get('model');assert returned in ('qwen3.8-max','qwen3.8-max-0902','qwen3.8-max-2026-09-02'),'Unexpected model'
  reasoning=bool((answer.get('choices') or [{}])[0].get('message',{}).get('reasoning_content'));result.update(status='THINKING_OFF_VIOLATED' if reasoning else 'RESPONDED',model_returned=returned,thinking_content_present=reasoning,system_fingerprint=answer.get('system_fingerprint'),usage=usage,generic_response=answer)
 except urllib.error.HTTPError as ex:result.update(status='HTTP_ERROR',http_status=ex.code,raw_error_not_logged=True)
 except Exception as ex:result.update(status='MODEL_IDENTITY_CHANGED' if isinstance(ex,AssertionError) else 'CALL_OUTCOME_UNKNOWN',error_type=type(ex).__name__,no_retry=True)
 once(P/f'call-{index}.observed.json',dict(result,observed_before_budget_settlement=True,epoch=time.time()));budget.settle(row,usage);result.update(seconds=time.monotonic()-tick,epoch=time.time(),usage_known=usage is not None);once(out,result);old.atomic(P/'budget-live.json',budget.summary());print(json.dumps({k:result.get(k) for k in ('index','status','http_status','model_returned','usage')}),flush=True);return result

def run():
 name=next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));assert Path(name).name==UNIT and os.environ.get('INVOCATION_ID') and os.getuid()!=0
 authorize();once(P/'started.json',dict(source=bind(Path(__file__)),invocation=os.environ['INVOCATION_ID'],epoch=time.time()));begin=time.monotonic();first=call(0);results=[bind(P/'call-0.json')]
 try:first_text(first)
 except (AssertionError,KeyError,IndexError,UnicodeError,TypeError) as ex:assessment=dict(status='DIRECT_RESPONSE_NOT_ELIGIBLE_FOR_REPLAY',reason_type=type(ex).__name__,no_requery=True)
 else:
  second=call(1);results.append(bind(P/'call-1.json'))
  try:assessment=compare(first,second)
  except (AssertionError,KeyError,IndexError,UnicodeError,TypeError) as ex:assessment=dict(status='REPLAY_RESPONSE_UNQUALIFIED',reason_type=type(ex).__name__,no_requery=True)
 once(P/'terminal.json',dict(status=assessment['status'],assessment=assessment,calls=len(results),results=results,seconds=time.monotonic()-begin,GPU_calls=0,model_loads=0,updates=0,project_data_sent=False,no_global_tokenizer_or_multi_step_IPD_training_claim=True,whole_goal_complete=False,epoch=time.time()));print('PREFIX_REPLAY_TERMINAL',assessment['status'],flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);x=a.parse_args()
 try:(prepare if x.action=='prepare' else run)()
 except BaseException as ex:
  f=P/(x.action+'-failure.json')
  if not f.exists():once(f,dict(type=type(ex).__name__,message='Failure preserved; no credentials/rawHTTPerror',epoch=time.time(),no_automatic_retry=True))
  raise
