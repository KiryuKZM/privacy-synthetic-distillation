"""Exactly one user-approved recovery; original API batch remains immutable."""
from pathlib import Path
import json,os,signal,sqlite3,ssl,subprocess,sys,time,traceback,urllib.request
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import ust_scale_teacher_20261108 as t
O=t.S/'single-recovery-v1';PURPOSE='ust-scale-teacher-one-recovery-20261108-v1';TAG=PURPOSE+'/0570';STOP=False

def stop(*_):
 global STOP;STOP=True

def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own single-recovery STOP')

def one():
 proposal=t.load(t.S/'truncation-review-v1/single-recovery-PENDING.local.json');a=t.load(O/'authorization.json');assert a['authorized'] and a['proposal']==t.pin(t.S/'truncation-review-v1/single-recovery-PENDING.local.json')
 body=proposal['body'];original=t.load(t.S/'requests.local.json')[570]['body'];assert body==dict(original,max_tokens=1024);assert proposal['calls']==1
 wire=t.old.wire(dict(body=body));import hashlib
 digest=hashlib.sha256(wire).hexdigest();i=proposal['input_reserve'];cap=i*12+1024*36;assert cap==68604
 assert not (O/'started.json').exists();token=t.old.api.key()
 with sqlite3.connect(t.old.LEDGER,timeout=30) as c:
  c.execute('BEGIN IMMEDIATE');assert not c.execute('SELECT 1 FROM qwen_max_calls WHERE tag=? OR purpose=?',(TAG,PURPOSE)).fetchone(),'No replay of single authorized attempt'
  total=c.execute('SELECT coalesce(sum(q.inp*c.input_micro+q.out*c.output_micro),0) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task=?',(t.TASK,)).fetchone()[0];assert total+cap<=500000000
  c.execute('INSERT INTO ust_research_cny_contracts VALUES(?,?,?,?,?)',(PURPOSE,t.TASK,12,36,500000000))
  cid=c.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico,settled,usage_known) VALUES(?,?,?,?,?,?,?,0,0)',(TAG,PURPOSE,t.old.MODEL,digest,i,1024,t.old.api.cost(i,1024))).lastrowid
 t.save(O/'started.json',dict(PID=os.getpid(),epoch=time.time(),ledger_id=cid,body_sha=digest,reserve_CNY=cap/1e6));usage=None
 try:
  request=urllib.request.Request(t.old.api.URL,data=wire,method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),t.old.api.NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
  with opener.open(request,timeout=75) as response:provider=json.loads(response.read(2**20))
  t.save(O/'provider-response.local.json',provider)
  u=provider.get('usage',{})
  if all(type(u.get(k)) is int for k in ('prompt_tokens','completion_tokens')):
   assert 0<=u['prompt_tokens']<=i and 0<=u['completion_tokens']<=1024;usage=u
  assert provider['model'] in (t.old.MODEL,'qwen3.8-max-2026-09-02','qwen3.8-max')
  choice=provider['choices'][0];assert not choice['message'].get('reasoning_content');text=choice['message'].get('content','');finish=choice['finish_reason']
  pred=t.study.audit.parse_function()(text,finish)
  import eval_ust_sg_qwen_modes_20261024 as parser
  g=parser.classify(text,'off','ust',dict(ability='st_understanding'),finish);assert g['prediction']==pred
  t.save(O/'result.json',dict(raw_generated=text,finish_reason=finish,prediction=pred,valid=bool(g['valid']),usage=usage,original_index=570,model_returned=provider['model']))
 finally:
  with sqlite3.connect(t.old.LEDGER,timeout=30) as c:
   c.execute('BEGIN IMMEDIATE');assert c.execute('SELECT settled FROM qwen_max_calls WHERE id=?',(cid,)).fetchone()==(0,)
   if usage is None:c.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=?',(cid,))
   else:c.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=?',(usage['prompt_tokens'],usage['completion_tokens'],t.old.api.cost(usage['prompt_tokens'],usage['completion_tokens']),cid))
 t.save(O/'budget.json',t.budget());assert g['valid'],'Single authorized recovery did not yield a valid answer; no retry'
 labels=t.load(t.S/'labels.local.json');assert not labels[570]['valid'];labels[570]=dict(labels[570],prediction=pred,valid=True,recovery_result=t.pin(O/'result.json'))
 t.save(O/'labels.local.json',labels);t.save(O/'label-summary.json',dict(n=1088,valid=sum(x['valid'] for x in labels),correct=sum(x['prediction']==x['oracle'] for x in labels),original_batch_valid=1087,recovery_calls=1,budget=t.budget(),initial_truncation_retained=True))
 print(json.dumps(t.load(O/'label-summary.json')),flush=True)

def prepare_students():
 labels=t.load(O/'labels.local.json');assert all(x['valid'] for x in labels)
 original_S=t.S;t.S=O
 try:t.build_students(labels)
 finally:t.S=original_S
 # Validate every final teacher target, option rotation and paired schedule before GPU.
 vv=t.load(t.T/'variants.local.json');old={r['id']:r for r in t.load(t.P/'data/current_n1088-variants.local.json')};tokens=t.load(t.T/'tokens.local.json');base_tokens=t.load(t.P/'tokens.local.json');different=[]
 for v in vv:
  a=old[v['id']];assert v['question']==a['question'];assert tokens[v['id']]['prompt_ids']==base_tokens[v['id']]['prompt_ids']
  expected='ABCD'[('ABCD'.index(labels[v['source_slot']]['prediction'])-v['variant_index'])%4];assert v['answer']==expected
  if tokens[v['id']]['target_ids']!=base_tokens[v['id']]['target_ids']:different.append(v['id'])
 for seed in t.study.SEEDS:
  assert t.load(t.T/f'teacher_n1088_s{seed}'/'schedule.local.json')==t.load(t.P/f'current_n1088_s{seed}'/'schedule.local.json')
 t.save(O/'training-comparison.json',dict(views=4352,different_target_count=len(different),different_ids=different,same_prompts_schedules_INIT=True,three_repeated_fits_are_reproducibility_checks_if_all_targets_equal=True))

def controller():
 check();assert (O/'training-comparison.json').exists();t.save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time()));deadline=time.monotonic()+172800
 while not (t.P/'terminal.json').exists():
  check();assert not (t.P/'failure.json').exists(),'Main suite failed; no overlapping GPU launch';assert time.monotonic()<deadline;time.sleep(10)
 for seed in t.study.SEEDS:
  arm=f'teacher_n1088_s{seed}'
  for action in ('fit','evaluate'):
   check()
   with (O/f'{action}-{seed}.log').open('x') as out:
    child=subprocess.Popen([str(R/'.venv/bin/python'),str(R/'scripts/ust_scale_teacher_20261108.py'),action,arm],cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    t.save(O/f'{action}-{seed}-process.json',dict(PID=child.pid,epoch=time.time()));start=time.monotonic()
    try:
     while child.poll() is None:
      check();assert time.monotonic()-start<14400;time.sleep(2)
     assert child.returncode==0
    finally:
     if child.poll() is None:
      os.killpg(child.pid,signal.SIGTERM)
      try:child.wait(timeout=30)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 t.teacher_score();t.save(O/'terminal.json',dict(status='COMPLETE_ONE_RECOVERY_AND3_STUDENT_FITS',budget=t.budget(),epoch=time.time()));t.save(t.S/'recovery-terminal.json',dict(status='COMPLETE_WITH_EXPLICIT_SINGLE_RECOVERY',recovery=t.pin(O/'terminal.json'),original_failure_preserved=True))

if __name__=='__main__':
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  t.save(O/f'failure-{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise
