"""Bounded original-teacher synthetic labels; subsequent matched student fits."""
from pathlib import Path
import collections,concurrent.futures,copy,hashlib,json,os,signal,sqlite3,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_scale_20261108 as study
import ust_teacher_expanded_api_20261108 as api
old=api.old
S=R/'runs/2026-11-08-ust-scale-teacher-v1';T=S/'students';P=study.O
PURPOSE='ust-scale-synthetic-teacher-20261108-v1';CAP=100;TASK=old.TASK
load=study.load;save=study.save;pin=study.bind

def budget(path=old.LEDGER):
 with sqlite3.connect(path,timeout=30) as c:
  total=c.execute('SELECT coalesce(sum(q.inp*c.input_micro+q.out*c.output_micro),0) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task=?',(TASK,)).fetchone()[0]
  cost,n,unknown=c.execute('SELECT coalesce(sum(inp*12+out*36),0),count(*),coalesce(sum(1-usage_known),0) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
 return dict(cumulative_CNY=total/1e6,remaining_CNY=500-total/1e6,batch_CNY=cost/1e6,calls=n,unknown=unknown)
def verify():
 p=load(S/'protocol.json')
 a=load(S/'api-user-authorization.json');assert a['authorized'] is True and a['protocol']==pin(S/'protocol.json') and a['calls']==1088
 for b in p['bindings']:assert pin(R/b['path'])==b
 assert not (S/'STOP').exists()
 return p

def reserve(path,row):
 # Serialized cumulative reservation BEFORE HTTP, including unresolved earlier attempts.
 assert row==ALLOW[row['id']]
 with sqlite3.connect(path,timeout=30) as c:
  c.execute('BEGIN IMMEDIATE');i=row['input_reserve'];cost=i*12+512*36
  total=c.execute('SELECT coalesce(sum(q.inp*c.input_micro+q.out*c.output_micro),0) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task=?',(TASK,)).fetchone()[0]
  phase,n=c.execute('SELECT coalesce(sum(inp*12+out*36),0),count(*) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
  assert total+cost<=500000000 and phase+cost<=CAP*1000000 and n<1088,'Cumulative budget/count cap'
  assert not c.execute('SELECT 1 FROM qwen_max_calls WHERE tag=?',(row['id'],)).fetchone(),'No repeat'
  return c.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico,settled,usage_known) VALUES(?,?,?,?,?,?,?,0,0)',(row['id'],PURPOSE,old.MODEL,hashlib.sha256(old.wire(row)).hexdigest(),i,512,old.api.cost(i,512))).lastrowid

def settle(path,call,usage):
 with sqlite3.connect(path,timeout=30) as c:
  c.execute('BEGIN IMMEDIATE');r=c.execute('SELECT inp,out,settled FROM qwen_max_calls WHERE id=? AND purpose=?',(call,PURPOSE)).fetchone();assert r and r[2]==0
  if usage is None:c.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=?',(call,))
  else:
   i,o=usage['prompt_tokens'],usage['completion_tokens'];assert type(i)is int and type(o)is int and 0<=i<=r[0] and 0<=o<=r[1]
   c.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=?',(i,o,old.api.cost(i,o),call))

def prepare():
 study.verify();vv=[v for v in load(P/'data/current_n1088-variants.local.json') if v['variant_index']==0];assert len(vv)==1088
 gp=load(study.d.G/'protocol.json');fit=load(R/gp['fit']['path']);private={study.d.sig(x['observations']) for x in fit};rows=[]
 for i,v in enumerate(vv):
  obs,_=study.audit.sections(v['question']);sig=study.d.sig([[None if obs[j][f] is None else float(obs[j][f]) for f in study.d.old.FIELDS] for j in range(1,5)]);assert sig not in private,'Exact protected Fit scene clone excluded from API'
  body=dict(model=old.MODEL,temperature=0,enable_thinking=False,max_tokens=512,stream=False,messages=[dict(role='system',content='Solve the supplied urban spatiotemporal task from its data, objective and feedback. Follow the stated output format.'),dict(role='user',content=v['question'])])
  rows.append(dict(id=f'scaleteacher20261108/{i:04d}',source_id=v['source_id'],source_slot=v['source_slot'],cohort='synthetic_training1088',view='canonical',body=body,input_reserve=sum(len(x['content'].encode()) for x in body['messages'])+256,oracle=v['answer'],metric=v['metric']))
 bound=sum(r['input_reserve']*12+512*36 for r in rows)/1e6;assert bound<CAP and budget()['cumulative_CNY']+bound<500
 save(S/'requests.local.json',rows);save(S/'protocol.json',dict(bindings=[pin(__file__),pin(P/'protocol.json'),pin(P/'data/current_n1088-variants.local.json'),pin(S/'requests.local.json'),pin(api.__file__),pin(old.__file__),pin(old.api.__file__)],provider='Alibaba Cloud original configured endpoint',model=old.MODEL,calls=1088,maximum_input_tokens=sum(r['input_reserve'] for r in rows),maximum_output_tokens=1088*512,conservative_max_CNY=bound,batch_cap_CNY=100,task_cap_CNY=500,prior=budget(),external_data='Only newly generated1088 canonical training questions; no real demonstrations, protected rows, labels or Fit statistics',cheaper_alternative='Existing exact-rule local labels, already in main scaling control',price_source='https://help.aliyun.com/zh/model-studio/qwen3-8-max',price_verified='2026-11-08',authorization='User: 做 大规模实验 api budget 500; existing original API provider/model/synthetic data scope retained',Final_access=False,retries=0,student_policy='Use teacher letter even when wrong; rotate consistently across options; no oracle correction; invalid response stops teacher student phase; same current1088 baseline settings'))
 print(json.dumps(dict(calls=1088,bound_CNY=bound,prior=budget()),ensure_ascii=False),flush=True)

def run():
 global ALLOW
 verify();rows=load(S/'requests.local.json');ALLOW={r['id']:r for r in rows};assert budget()['calls']==0,'No automatic replay'
 with sqlite3.connect(old.LEDGER,timeout=30) as c:
  c.execute('BEGIN IMMEDIATE');c.execute('INSERT INTO ust_research_cny_contracts VALUES(?,?,?,?,?)',(PURPOSE,TASK,12,36,500000000))
 # Existing HTTP transport prohibits redirects, uses original credential, records raw provider responses.
 api.S=S;api.PURPOSE=PURPOSE;api.reserve=reserve;api.settle=settle
 token=old.api.key();save(S/'started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time(),budget=budget()));deadline=time.monotonic()+7200
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  pending={};nexti=0
  while nexti<len(rows) or pending:
   api.check();assert time.monotonic()<deadline
   while nexti<len(rows) and len(pending)<4:
    pending[ex.submit(api.one,nexti,rows[nexti],token,deadline)]=nexti;nexti+=1
   done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    idx=pending.pop(f);f.result();raw=load(S/'raw'/f'{idx:03d}.json');assert raw['status']=='RESPONDED','No failure retries'
   old.atomic(S/'live.json',dict(stage='TEACHER_LABELS',submitted=nexti,pending=len(pending),budget=budget(),epoch=time.time()))
 save(S/'raw-frozen.json',[pin(S/'raw'/f'{i:03d}.json') for i in range(1088)])
 parse=study.audit.parse_function();labels=[]
 for i,row in enumerate(rows):
  raw=load(S/'raw'/f'{i:03d}.json');pred=parse(raw['raw_generated'],raw['finish_reason']);labels.append(dict(source_slot=row['source_slot'],oracle=row['oracle'],prediction=pred,valid=pred in 'ABCD' if isinstance(pred,str) else False))
 save(S/'labels.local.json',labels);save(S/'label-results.json',dict(total=1088,valid=sum(x['valid'] for x in labels),correct=sum(x['prediction']==x['oracle'] for x in labels),budget=budget()))
 assert all(x['valid'] for x in labels),'Invalid teacher labels; stop before training, no selection or oracle repair'
 build_students(labels)
 # Queue GPU only after own main suite releases it. No interference with other workloads.
 until=time.monotonic()+172800
 while not (P/'terminal.json').exists():
  api.check();assert not (P/'failure.json').exists(),'Main suite failed; preserve teacher labels';assert time.monotonic()<until;time.sleep(10)
 for seed in study.SEEDS:
  arm=f'teacher_n1088_s{seed}'
  for action in ('fit','evaluate'):
   with (S/f'{action}-{seed}.log').open('x') as out:
    child=subprocess.Popen([str(R/'.venv/bin/python'),__file__,action,arm],cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    save(S/f'{action}-{seed}-process.json',dict(PID=child.pid,epoch=time.time()))
    try:
     start=time.monotonic()
     while child.poll() is None:
      api.check();assert time.monotonic()-start<14400;time.sleep(2)
     assert child.returncode==0
    finally:
     if child.poll() is None:
      os.killpg(child.pid,signal.SIGTERM)
      try:child.wait(timeout=30)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 teacher_score();save(S/'terminal.json',dict(status='COMPLETE1088_TEACHER_LABELS_AND3_STUDENT_FITS',budget=budget(),epoch=time.time()))

def build_students(labels):
 prefix='current_n1088/';vv=[v for v in load(P/'variants.local.json') if v['id'].startswith(prefix)];tokens=load(P/'tokens.local.json');keep={v['id'] for v in vv}|{q['id'] for q in load(P/'freshINIT/queries.local.json')};tokens={k:v for k,v in tokens.items() if k in keep}
 # All answer token sequences differ only in the certified single decision-token position.
 families={v['answer']:tokens[v['id']]['target_ids'] for v in vv}
 for v in vv:
  pred=labels[v['source_slot']]['prediction'];gold='ABCD'[('ABCD'.index(pred)-v['variant_index'])%4];v['answer']=gold;v['target']='<Answer>'+gold+'</Answer>';tokens[v['id']]['target_ids']=families[gold]
 save(T/'variants.local.json',vv);save(T/'tokens.local.json',tokens)
 for name in ('fixed8.local.json','answers.local.json'):save(T/name,load(P/name))
 specs={}
 for seed in study.SEEDS:
  source=P/f'current_n1088_s{seed}';arm=f'teacher_n1088_s{seed}'
  for name in ('protocol.json','schedule.local.json','queries.local.json'):save(T/arm/name,load(source/name))
  specs[arm]=dict(step=2176,kind='teacher_current',order_seed=seed,unique_sources=1088)
 save(T/'specs.json',specs);pins=[pin(p) for p in T.rglob('*.json')]+[pin(S/'labels.local.json'),pin(P/'protocol.json'),pin(__file__),pin(study.__file__)]
 save(T/'protocol.json',dict(bindings=pins,Final_access=False,teacher_labels=True,original_wrong_labels_kept=True))
 save(S/'student-preparation.json',dict(fits=3,updates=6528,raw=930,API_cost=budget(),same_questions_schedules_INIT_as_main=True))

def configure_students():
 study.O=T;study.ARMS=tuple(load(T/'specs.json'));study.configure()
def teacher_score():
 configure_students()
 import eval_ust_sg_qwen_modes_20261024 as parser
 parse=study.audit.parse_function();answers=load(T/'answers.local.json');scores={}
 for arm in load(T/'specs.json'):
  assert load(T/arm/'evaluation-terminal.json')['raw']==310
  grades=[]
  for i,q in enumerate(load(T/arm/'queries.local.json')):
   raw=load(T/arm/'raw'/f'{i:04d}.json');assert raw['id']==q['id'];pred=parse(raw['raw_generated'],raw['finish_reason']);g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);assert pred==g['prediction']
   grades.append(dict(id=q['id'],group=q['group'],variant=q['variant'],metric=q['metric'],flags=q['flags'],prediction=pred,correct=pred==answers[q['id']]['answer'],valid=bool(g['valid']),capped=raw['finish_reason']=='length'))
  save(T/arm/'graded.local.json',grades)
  scores[arm]={view:dict(n=155,correct=sum(r['correct'] for r in grades if r['variant']==view),valid=sum(r['valid'] for r in grades if r['variant']==view)) for view in ('original','joint')}
 save(T/'results.json',dict(status='COMPLETE3_TEACHER_LABELLED_STUDENTS',raw=930,scores=scores,Final_access=False))

if __name__=='__main__':
 signal.signal(signal.SIGTERM,api.halt);signal.signal(signal.SIGINT,api.halt)
 try:
  action=sys.argv[1]
  if action in ('fit','evaluate'):
   configure_students();study.fit(sys.argv[2]) if action=='fit' else study.base.evaluate(sys.argv[2])
  else:globals()[action]()
 except BaseException as e:
  save(S/f'failure-{time.time_ns()}.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise
