"""Approved90+90 paired teacher experiment, bounded four-way cloud worker."""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,json,os,re,signal,sqlite3,ssl,sys,threading,time,urllib.error,urllib.request
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_teacher_api_compare_20261108 as old
import ust_missingness_delivery_20261107 as audit
S=R/'runs/2026-11-08-ust-teacher-expanded-v1';LEDGER=old.LEDGER
PURPOSE='ust-teacher-expanded-20261108-v1';TASK=old.TASK;MODEL=old.MODEL;PREFIX='teachermore20261108/'
CNY=old.CNY;USD=old.USD;STOP=threading.Event();CONCURRENCY=4
load=old.load;save=old.save;pin=old.pin;atomic=old.atomic;wire=old.wire;digest=old.digest

def halt(*_):STOP.set()
def check():
 if STOP.is_set() or (S/'STOP').exists():raise InterruptedError('Own expanded teacher STOP')
def verify():
 check();p=load(S/'api-protocol.json');assert p['source']==pin(__file__)
 for b in p['bindings']:assert pin(R/b['path'])==b,b['path']
 a=load(S/'api-user-authorization.json');assert a['authorized'] and a['public_original_exception']==90 and a['calls']==360
 return p

def declared_map_check(a,b,m):
 va,oa=audit.sections(a);vb,ob=audit.sections(b);lookup={tuple(row[f] for f in audit.FIELDS):i for i,row in va.items()};assert len(lookup)==4
 back={i:lookup[tuple(row[f] for f in audit.FIELDS)] for i,row in vb.items()};assert set(back.values())=={1,2,3,4} and set(m)==set(m.values())==set('ABCD')
 for letter,seq in oa.items():assert seq==tuple(back[x] for x in ob[m[letter]])
 return True

def prepare():
 a=load(S/'api-user-authorization.json');assert a['authorized'];pr=load(S/'preparation-review.json');assert pr['source']==pin(R/'scripts/ust_teacher_expansion_prepare_20261108.py')
 for b in pr['bindings']+pr['API_packets']:assert pin(R/b['path'])==b
 rows=load(S/'requests-PENDING.local.json');answers=load(S/'answers.local.json');release=load(S/'original-release-PENDING.local.json');assert len(rows)==360 and len(release)==90
 qs={r['id']:r['question'] for r in release};groups={r['group'] for r in release};assert len(groups)==90
 covered=collections.Counter();paired=collections.defaultdict(dict)
 for r in rows:
  b=r['body'];assert b['model']==MODEL and b['max_tokens']==512 and b['temperature']==0 and b['enable_thinking'] is False and b['stream'] is False
  assert [x['role'] for x in b['messages']]==['system','user'];q=b['messages'][1]['content'];assert r['input_reserve']==sum(len(m['content'].encode()) for m in b['messages'])+256
  obs,opts=audit.sections(q);val={k:float('inf') if row[r['metric']] is None else row[r['metric']] for k,row in obs.items()};correct=[a for a,seq in opts.items() if all(val[i]>=val[j] for i,j in zip(seq,seq[1:]))];assert correct==[answers[r['id']]]
  if r['cohort']=='public_original90_PENDING' and r['view']=='original':assert q==qs[r['source_id']]
  covered[(r['cohort'],r['view'],tuple(r['stratum']))]+=1;paired[(r['cohort'],r['pair'])][r['view']]=r
 option_properties={};declared=load(S/'effective-maps.local.json')
 for (cohort,index),pair in paired.items():
  a,b=pair['original'],pair['joint'];m=declared[cohort+'/'+f'{index:03d}'];assert declared_map_check(a['body']['messages'][1]['content'],b['body']['messages'][1]['content'],m) and m[answers[a['id']]]==answers[b['id']]
  _,opts=audit.sections(a['body']['messages'][1]['content']);option_properties[cohort+'/'+f'{index:03d}']=dict(distinct_options=len(set(opts.values())),duplicate_incorrect_options=len(set(opts.values()))<4)
 save(S/'candidate-option-properties.local.json',option_properties)
 for view in ('original','joint'):
  assert {k[2]:v for k,v in covered.items() if k[:2]==('public_original90_PENDING',view)}=={k[2]:v for k,v in covered.items() if k[:2]==('matched_synthetic90',view)}
 assert pr['combined_conservative_CNY']<=20
 save(S/'public-proxy-release.json',dict(status='USER_AUTHORIZED_NEW_PUBLIC_PROXY_EXCEPTION',authorization=pin(S/'api-user-authorization.json'),members=[dict(id=r['id'],group=r['group'],question_sha256=hashlib.sha256(r['question'].encode()).hexdigest()) for r in release],original_source='Previously locally protected unused Screen pool',excludes='All other role pools and recorded Common/train/fit/dev/audit/confirm memberships; Final inputs untouched',instruction='These90/groups must henceforth be treated as externally exposed public diagnostic material, never private or independent final acceptance; teacher labels not used for student training.'))
 save(S/'api-protocol.json',dict(source=pin(__file__),bindings=pr['bindings']+pr['API_packets']+[pin(S/n) for n in ['preparation-review.json','api-user-authorization.json','public-proxy-release.json','candidate-option-properties.local.json']]+[pin(old.__file__),pin(old.api.__file__),pin(audit.__file__),pin(audit.PARSER)],purpose=PURPOSE,model=MODEL,total_requests=360,concurrency=4,maximum_seconds=3600,call_timeout_seconds=75,retries=0,prior_comparison_CNY=pr['previous_research_CNY'],new_bound_CNY=pr['new_conservative_max_CNY'],combined_bound_CNY=pr['combined_conservative_CNY'],comparison_family_cap_CNY=20,task_cap_CNY=500,whole360_before_score=True,no_new_generator_or_student_training=True,Final_access=False))
 save(S/'api-input-qualification.json',dict(status='PASS360_EXACT_DECIMAL_ORIGINAL_COPY_MATCHED_STRATA_AND_JOINT_INVERSE',source=pin(__file__),original_source_count=90,original_numeric_groups=90,synthetic_source_count=90,pairs=180,all_unique_reference=True,duplicate_distractor_originals=sum(x['duplicate_incorrect_options'] for k,x in option_properties.items() if k.startswith('public_original')),price={'CNY_per_M_input':12,'CNY_per_M_output':36,'source':'https://help.aliyun.com/zh/model-studio/qwen3-8-max'},new_GPU=0,new_API=0));print('PREPARED360',flush=True)

def sqls():
 fam=[old.VLA,old.UST,old.OLDQUALITY,old.PURPOSE,PURPOSE];ust=fam[1:];family=','.join(repr(x) for x in fam);ustsql=','.join(repr(x) for x in ust);compare=','.join(repr(x) for x in (old.PURPOSE,PURPOSE))
 out=[f"""CREATE TRIGGER ustteachermore26_insert BEFORE INSERT ON qwen_max_calls WHEN NEW.purpose='{PURPOSE}' OR NEW.tag GLOB '{PREFIX}*' BEGIN SELECT CASE WHEN
 NEW.purpose!='{PURPOSE}' OR NEW.model!='{MODEL}' OR NEW.out!=512 OR NEW.settled!=0 OR NEW.usage_known!=0 OR NEW.pico!=NEW.inp*1650000+NEW.out*4951000 OR
 NOT EXISTS(SELECT 1 FROM ustteachermore26_allowlist WHERE tag=NEW.tag AND body_sha=NEW.body_sha AND input_reserve=NEW.inp) OR
 (SELECT count(*) FROM qwen_max_calls WHERE purpose='{PURPOSE}')>=360 OR
 (SELECT coalesce(sum(inp*12+out*36),0) FROM qwen_max_calls WHERE purpose IN({compare}))+NEW.inp*12+NEW.out*36>{20*CNY} OR
 (SELECT coalesce(sum(q.inp*c.input_micro+q.out*c.output_micro),0) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task='{TASK}')+NEW.inp*12+NEW.out*36>{500*CNY} OR
 (SELECT count(*) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task='{TASK}' AND q.settled=0)>={CONCURRENCY} OR
 (SELECT coalesce(sum(pico),0) FROM qwen_max_calls WHERE purpose IN({family}))+NEW.pico>{100*USD} OR
 (SELECT coalesce(sum(pico),0) FROM qwen_max_calls WHERE purpose IN({ustsql}))+NEW.pico>{60*USD}
 THEN RAISE(ABORT,'ExpandedPayloadCountCombined20Task500Inflight4OrPriorUSD') END; END"""]
 # Reuse the prior immutable-settlement/history semantics with a new namespace.
 for sql in old.sqls():
  if 'ustteacher26_update ' in sql or 'ustteacher26_delete ' in sql or 'ustteacher26_allow_' in sql:
   out.append(sql.replace('ustteacher26','ustteachermore26').replace(old.PURPOSE,PURPOSE))
 return out

def install(path):
 verify();rows=load(S/'requests-PENDING.local.json');allow=sorted((r['id'],hashlib.sha256(wire(r)).hexdigest(),r['input_reserve']) for r in rows)
 with sqlite3.connect(path,timeout=30) as db:
  db.execute('BEGIN IMMEDIATE');prior=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'"));tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")};assert 'ust_research_cny_contracts' in tables
  if 'ustteachermore26_allowlist' not in tables:
   db.execute('CREATE TABLE ustteachermore26_allowlist(tag TEXT PRIMARY KEY,body_sha TEXT UNIQUE NOT NULL,input_reserve INTEGER NOT NULL)');db.executemany('INSERT INTO ustteachermore26_allowlist VALUES(?,?,?)',allow)
  assert sorted(db.execute('SELECT * FROM ustteachermore26_allowlist'))==allow
  c=(PURPOSE,TASK,12,36,500*CNY);existing=db.execute('SELECT * FROM ust_research_cny_contracts WHERE purpose=?',(PURPOSE,)).fetchone()
  if existing is None:db.execute('INSERT INTO ust_research_cny_contracts VALUES(?,?,?,?,?)',c)
  else:assert existing==c
  for sql in sqls():
   name=re.search('CREATE TRIGGER (\\w+)',sql)[1]
   if name in prior:assert prior[name]==sql
   else:db.execute(sql)
  assert all(db.execute('SELECT sql FROM sqlite_master WHERE name=?',(k,)).fetchone()==(v,) for k,v in prior.items())

def reserve(path,row):
 with sqlite3.connect(path,timeout=30) as db:
  db.execute('BEGIN IMMEDIATE')
  for sql in sqls():
   name=re.search('CREATE TRIGGER (\\w+)',sql)[1];assert db.execute('SELECT sql FROM sqlite_master WHERE name=?',(name,)).fetchone()==(sql,)
  i=row['input_reserve'];return db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico,settled,usage_known) VALUES(?,?,?,?,?,?,?,0,0)',(row['id'],PURPOSE,row['body']['model'],hashlib.sha256(wire(row)).hexdigest(),i,512,old.api.cost(i,512))).lastrowid

def settle(path,call,usage):
 with sqlite3.connect(path,timeout=30) as db:
  db.execute('BEGIN IMMEDIATE')
  if usage is None:db.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=? AND purpose=?',(call,PURPOSE))
  else:
   i,o=usage['prompt_tokens'],usage['completion_tokens'];assert type(i)is int and type(o)is int
   db.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=? AND purpose=?',(i,o,old.api.cost(i,o),call,PURPOSE))

def budget(path=LEDGER):
 with sqlite3.connect(path,timeout=30) as db:
  n,c,ip,unk,u=db.execute('SELECT count(*),coalesce(sum(inp*12+out*36),0),coalesce(sum(1-settled),0),coalesce(sum(1-usage_known),0),coalesce(sum(pico),0) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
  combined=db.execute('SELECT coalesce(sum(inp*12+out*36),0) FROM qwen_max_calls WHERE purpose IN(?,?)',(old.PURPOSE,PURPOSE)).fetchone()[0]
  task=db.execute('SELECT coalesce(sum(q.inp*c.input_micro+q.out*c.output_micro),0) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task=?',(TASK,)).fetchone()[0]
  hist=db.execute('SELECT (SELECT coalesce(sum(pico),0) FROM calls)+(SELECT coalesce(sum(pico),0) FROM qwen_max_calls)').fetchone()[0]
 return dict(HTTP_attempts=n,new_phase_CNY=c/CNY,comparison_cumulative_CNY=combined/CNY,comparison_cap_CNY=20,cumulative_task_CNY=task/CNY,task_cap_CNY=500,remaining_task_CNY=500-task/CNY,inflight=ip,unknown_or_unsettled=unk,new_phase_USD=u/USD,historical_all_accounted_USD=hist/USD,accounting='Posted no-cache token rates; unresolved responses retain full reserves, not reconciled invoice',epoch=time.time())

def tests():
 verify();folder=S/'budget-tests';folder.mkdir();rows=load(S/'requests-PENDING.local.json');passed=[]
 def clone(name):
  p=folder/(name+'.sqlite')
  with sqlite3.connect('file:'+str(LEDGER)+'?mode=ro',uri=True) as src,sqlite3.connect(p) as dst:src.backup(dst)
  install(p);return p
 def no(label,fn):
  try:fn()
  except sqlite3.IntegrityError:passed.append(label);return
  raise AssertionError('Expected SQL refusal '+label)
 p=clone('standard');ids=[reserve(p,r) for r in rows[:4]];no('fifth simultaneous reservation',lambda:reserve(p,rows[4]))
 for c in ids:settle(p,c,dict(prompt_tokens=1000,completion_tokens=20))
 assert abs(budget(p)['new_phase_CNY']-.05088)<1e-10;passed.append('four concurrent reservations and known settlements')
 no('duplicate',lambda:reserve(p,rows[0]));no('second settlement',lambda:settle(p,ids[0],None))
 with sqlite3.connect(p) as db:
  no('delete prior call',lambda:db.execute('DELETE FROM qwen_max_calls WHERE id=?',(ids[0],)));no('release alteration',lambda:db.execute('DELETE FROM ustteachermore26_allowlist'))
 bad=json.loads(json.dumps(rows[5]));bad['body']['messages'][1]['content']='tampered';no('payload alteration',lambda:reserve(p,bad))
 bad=json.loads(json.dumps(rows[5]));bad['input_reserve']+=1;no('reservation alteration',lambda:reserve(p,bad))
 p=clone('unknown');c=reserve(p,rows[0]);settle(p,c,None);assert abs(budget(p)['new_phase_CNY']-(rows[0]['input_reserve']*12+512*36)/1e6)<1e-10;passed.append('full reserve for unknown')
 p=clone('overusage');c=reserve(p,rows[0]);no('usage over reserve',lambda:settle(p,c,dict(prompt_tokens=rows[0]['input_reserve']+1,completion_tokens=1)))
 for label,purpose,inp,pico,n in [('combined20',old.PURPOSE,1700000,0,1),('task500','FICTION_OTHER_TASK',42000000,0,1),('legacy100',old.VLA,0,101*USD,1),('legacy60',old.UST,0,61*USD,1),('legacy500','FICTION_OLD',0,501*USD,1),('360count',PURPOSE,0,0,360)]:
  p=clone(label)
  with sqlite3.connect(p) as db:
   original=list(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'"))
   for name,_ in original:db.execute('DROP TRIGGER '+name)
   if label=='task500':db.execute('INSERT INTO ust_research_cny_contracts VALUES(?,?,?,?,?)',(purpose,TASK,12,36,500*CNY))
   for i in range(n):db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico,settled,usage_known) VALUES(?,?,?,?,?,0,?,1,1)',('FICTION_ONLY_'+str(i),purpose,MODEL,'FICTITIOUS',inp,pico))
   for _,sql in original:db.execute(sql)
  no(label,lambda:reserve(p,rows[0]))
 save(S/'budget-qualification.json',dict(status='PASS_CPU_COPIES_ONLY',source=pin(__file__),cases=passed,new_API=0));print('PASS_BUDGET',len(passed),flush=True)

def one(index,row,token,deadline):
 check();out=S/'raw'/f'{index:03d}.json';mark=out.with_suffix('.started.json');assert not out.exists() and not mark.exists(),'No automatic replay'
 raw=wire(row);remaining=deadline-time.monotonic();assert remaining>0;call=reserve(LEDGER,row);save(mark,dict(id=row['id'],ledger_id=call,epoch=time.time(),body_sha=hashlib.sha256(raw).hexdigest(),public_original=row['cohort']=='public_original90_PENDING'))
 usage=None;t=time.monotonic();result=dict(id=row['id'],cohort=row['cohort'],source_id=row['source_id'],view=row['view'],ledger_id=call,body_sha=hashlib.sha256(raw).hexdigest())
 try:
  request=urllib.request.Request(old.api.URL,data=raw,method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'});opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),old.api.NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
  with opener.open(request,timeout=min(75,remaining)) as response:provider=json.loads(response.read(2**20))
  save(S/'provider-responses'/f'{index:03d}.local.json',provider);u=provider.get('usage',{})
  if all(type(u.get(k))is int for k in ('prompt_tokens','completion_tokens')):usage={k:u[k] for k in ('prompt_tokens','completion_tokens')}
  msg=provider['choices'][0]['message'];assert provider['model'] in (MODEL,'qwen3.8-max-2026-09-02','qwen3.8-max') and not msg.get('reasoning_content')
  result.update(status='RESPONDED',raw_generated=msg.get('content',''),finish_reason=provider['choices'][0]['finish_reason'],model_returned=provider['model'],usage=usage)
 except BaseException as e:result.update(status='UNKNOWN_OR_FAILURE',error_type=type(e).__name__,http_status=e.code if isinstance(e,urllib.error.HTTPError) else None)
 settle(LEDGER,call,usage);result['seconds']=time.monotonic()-t;save(out,result)
 if result['status']!='RESPONDED' or usage is None:STOP.set();raise RuntimeError('Paid response failure/unknown; retained cost; no retry')
 return index

def run():
 verify();assert load(S/'budget-qualification.json')['source']==pin(__file__)
 with sqlite3.connect(LEDGER) as db:
  before=dict(maxid=db.execute('select max(id) from qwen_max_calls').fetchone()[0],qwen_hash=digest(list(db.execute('select * from qwen_max_calls order by id'))),legacy_hash=digest(list(db.execute('select * from calls order by id'))),triggers=list(db.execute("select name,sql from sqlite_master where type='trigger'")))
 save(S/'budget-before.json',before);install(LEDGER);token=old.api.key();rows=load(S/'requests-PENDING.local.json');save(S/'started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time(),independent_session=os.getsid(0)==os.getpid(),concurrency=CONCURRENCY,source=pin(__file__)))
 save(R/'runs/ust-public-proxy-release-registry-20261108-expanded.json',dict(release=pin(S/'public-proxy-release.json'),authorization=pin(S/'api-user-authorization.json'),scope='90 original Screen numeric groups; teacher diagnostic only; treat as publicly exposed from this dispatch',Final_access=False))
 deadline=time.monotonic()+3600;nextidx=0;finished=[];errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
  futures={}
  while nextidx<len(rows) or futures:
   if time.monotonic()>=deadline or (S/'STOP').exists():STOP.set()
   while nextidx<len(rows) and len(futures)<CONCURRENCY and not STOP.is_set():
    verify();f=pool.submit(one,nextidx,rows[nextidx],token,deadline);futures[f]=nextidx;nextidx+=1
   if not futures:break
   done,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    i=futures.pop(f)
    try:finished.append(f.result())
    except BaseException as e:errors.append(dict(index=i,type=type(e).__name__));STOP.set()
   if done:
    b=budget();atomic(S/'live.json',dict(completed=len(finished),dispatched=nextidx,total=360,inflight=len(futures),epoch=time.time(),stage='CALLING' if not errors else 'DRAINING_AFTER_ERROR',budget=b));atomic(S/'budget-live.json',b);atomic(R/'runs/ust-research-api-budget-cny-current.json',dict(**b,source=str(S.relative_to(R)),ledger=str(LEDGER.relative_to(R))));print('CLOSED',len(finished),'DISPATCHED',nextidx,flush=True)
 if errors or len(finished)!=360:raise RuntimeError('Stopped incomplete; no replay; '+str(errors))
 save(S/'all360-raw-frozen.json',dict(epoch=time.time(),raw=[pin(S/'raw'/f'{i:03d}.json') for i in range(360)],all_closed_before_score=True));score()

def stat(rows):return dict(n=len(rows),correct=sum(x['correct'] for x in rows),valid=sum(x['valid'] for x in rows),capped=sum(x['capped'] for x in rows))
def score():
 verify();rawpins=load(S/'all360-raw-frozen.json')['raw'];assert len(rawpins)==360
 for b in rawpins:assert pin(R/b['path'])==b
 import eval_ust_sg_qwen_modes_20261024 as parser
 independent=audit.parse_function();qs=load(S/'requests-PENDING.local.json');aa=load(S/'answers.local.json');grades=[];mistakes=[]
 for i,q in enumerate(qs):
  r=load(S/'raw'/f'{i:03d}.json');assert r['id']==q['id'] and r['body_sha']==hashlib.sha256(wire(q)).hexdigest();g=parser.classify(r['raw_generated'],'off','ust',dict(ability='st_understanding'),r['finish_reason']);pred=independent(r['raw_generated'],r['finish_reason']);assert pred==g['prediction'] and (pred in ('A','B','C','D'))==bool(g['valid'])
  vals,opts=audit.sections(q['body']['messages'][1]['content']);vv={k:float('inf') if x[q['metric']] is None else x[q['metric']] for k,x in vals.items()};correct=[a for a,seq in opts.items() if all(vv[x]>=vv[y] for x,y in zip(seq,seq[1:]))];assert correct==[aa[q['id']]]
  row=dict(id=q['id'],source_id=q['source_id'],group=q['group'],cohort=q['cohort'],metric=q['metric'],view=q['view'],pair=q['pair'],stratum=q['stratum'],answer=aa[q['id']],prediction=pred,valid=bool(g['valid']),correct=bool(g['valid'] and pred==aa[q['id']]),capped=r['finish_reason']=='length');grades.append(row)
  if not row['correct']:mistakes.append(dict(query=q,reference=aa[q['id']],raw=r,grade=row))
 scores={};equiv={};maps=load(S/'effective-maps.local.json');byid={r['id']:r for r in grades}
 for cohort in ('public_original90_PENDING','matched_synthetic90'):
  scores[cohort]={}
  for view in ('original','joint'):
   rows=[r for r in grades if r['cohort']==cohort and r['view']==view];total=stat(rows)
   scores[cohort][view]=dict(all=total,descriptive_wilson95=old.wilson(total['correct'],total['n']),metric={m:stat([r for r in rows if r['metric']==m]) for m in ('distance','charging_time','waiting_time')},subgroups={name:stat([r for r in rows if r['stratum'][ix]]) for name,ix in [('target_NaN',1),('displayed_tie',2),('target_tail',3)]})
  orig=[r for r in grades if r['cohort']==cohort and r['view']=='original'];equiv[cohort]=dict(n=90,both_correct=sum(r['correct'] and byid[r['id'].rsplit('/',1)[0]+'/joint']['correct'] for r in orig),equivariant=sum(r['valid'] and byid[r['id'].rsplit('/',1)[0]+'/joint']['valid'] and maps[cohort+'/'+f'{r["pair"]:03d}'].get(r['prediction'])==byid[r['id'].rsplit('/',1)[0]+'/joint']['prediction'] for r in orig))
 distinct={};op=load(S/'candidate-option-properties.local.json');keep={i for i in range(90) if not op['public_original90_PENDING/'+f'{i:03d}']['duplicate_incorrect_options']};assert len(keep)==79
 for c in scores:distinct[c]={v:stat([r for r in grades if r['cohort']==c and r['view']==v and r['pair'] in keep]) for v in ('original','joint')}
 for c in equiv:
  same=0
  for row in [r for r in grades if r['cohort']==c and r['view']=='original']:
   j=byid[row['id'].rsplit('/',1)[0]+'/joint'];m=maps[c+'/'+f'{row["pair"]:03d}'];inv={v:k for k,v in m.items()};q=next(x for x in qs if x['id']==row['id']);_,opts=audit.sections(q['body']['messages'][1]['content'])
   same+=bool(row['valid'] and j['valid'] and opts[row['prediction']]==opts[inv[j['prediction']]])
  equiv[c]['selected_option_semantic_consistency']=same
 before=load(S/'budget-before.json')
 with sqlite3.connect(LEDGER) as db:
  assert digest(list(db.execute('select * from qwen_max_calls where id<=? order by id',(before['maxid'],))))==before['qwen_hash']
  assert digest(list(db.execute('select * from calls order by id')))==before['legacy_hash']
  assert all(db.execute('select sql from sqlite_master where name=?',(n,)).fetchone()==(s,) for n,s in before['triggers'])
 b=budget();assert b['HTTP_attempts']==360 and b['inflight']==b['unknown_or_unsettled']==0 and b['comparison_cumulative_CNY']<=20
 save(S/'graded.local.json',grades);save(S/'wrong-cases.local.json',mistakes);save(S/'results.json',dict(status='COMPLETE360_INDEPENDENT_PARSE_AND_EXACT_DECIMAL',scores=scores,equivariance=equiv,distinct_distractor_pairs79=distinct,wrong_outputs=len(mistakes),budget=b,old_ledger_records_and_guards_unchanged=True,original_public_exception_count=90,Final_access=False,initial7_not_pooled=True,all_model_outputs_reported=True,new_GPU=0))
 report=R/'reports/ust-teacher-expanded-comparison-20261108.md';lines=['# 扩大教师对照：90道原题与90道匹配合成题','','同一Qwen3.8-Max0902教师、零示例、关闭思考、温度0、输出最多512token。原版及同步位置/选项重排共360次调用全部闭合后统一评分；独立解析及精确数值答案验证通过。','','| 数据 | 原版 | 重排 | 两视图均正确 | 变换一致 |','|---|---:|---:|---:|---:|']
 for c,x in scores.items():lines.append(f'| {"公开原题90" if c=="public_original90_PENDING" else "匹配合成90"} | {x["original"]["all"]["correct"]}/90 | {x["joint"]["all"]["correct"]}/90 | {equiv[c]["both_correct"]}/90 | {equiv[c]["equivariant"]}/90 |')
 lines+=['','两侧各距离30、充电时间30、等待时间30，逐样本配齐目标缺失、并列值、Fit阈值数值尾部标记。原题题面不改，来自以前未登记用于训练/拟合/评测的Screen候选池，每题一个不同数值依赖组；只纳入参考答案能唯一核对的题，排除的歧义题另存。它是按题型均衡的可判定子集，不能当全benchmark自然分布准确率。合成题取自既有固定生成器2000候选池，不改参数、不重新生成，且与既有训练/Select/Confirm数值场景不重复。','','| 数据/视图 | 距离 | 充电时间 | 等待时间 | 缺失 | 并列 | 数值尾部 |','|---|---:|---:|---:|---:|---:|---:|']
 for c,x in scores.items():
  for view,v in x.items():
   ss=[v['metric'][m] for m in ('distance','charging_time','waiting_time')]+[v['subgroups'][f] for f in ('target_NaN','displayed_tie','target_tail')];lines.append('| '+c+'/'+view+' | '+' | '.join(f'{z["correct"]}/{z["n"]}' for z in ss)+' |')
 lines+=['','本轮90道原题经用户明确批准，作为新增公开代理例外外发。今后不能再把这些题/组视为私有或独立最终验收材料；教师标签不用于学生训练。其他真实Train、GeneratorFit、Dev、Common、Audit、Confirm及Final不涉及。此例外不是“没有原始数据外发”的证据。','','90道原题中11道有重复的错误选项，但正确参考仍唯一。保留这11题，不改原题；results.json另报双方对应的79对所有选项均不同的子集。严格字母变换一致性与选项实际内容一致性分别统计。', '', '90原题与90合成题为新样本，旧7/68结果保留且不混入本轮分母。每题的两个视图不是两个独立场景；数值组分离也不证明物理场景独立或无预训练污染。即使答对率接近也不能自动证明统计等价、覆盖保真或学生训练信号可替代。缺失和尾部小组按实际数量报告，真实异常真值未确认，不能宣称outlier保留。','','价格按官方无缓存原价保守记账：本轮'+f'{b["new_phase_CNY"]:.6f}'+'元，两轮教师比较/当前研究累计'+f'{b["cumulative_task_CNY"]:.6f}'+'元，500元剩余'+f'{b["remaining_task_CNY"]:.6f}'+'元；两轮比较共同20元上限。零失败重试、无未结算调用。没有使用GPU。','','完整请求、外发授权、数据资格、预算、原始回答和逐题评分位于 '+str(S)+' 。','']
 with report.open('x') as f:f.write('\n'.join(lines))
 save(S/'terminal.json',dict(status='COMPLETE',epoch=time.time(),results=pin(S/'results.json'),report=pin(report),budget=b));atomic(S/'live.json',dict(stage='COMPLETE',completed=360,total=360,epoch=time.time(),budget=b));print('COMPLETE360',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','tests','run','score']);a=p.parse_args();signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
 try:globals()[a.action]()
 except BaseException as e:
  save(S/('failure-'+str(time.time_ns())+'.json'),dict(action=a.action,type=type(e).__name__,error=str(e)[:500],epoch=time.time(),no_replay=True));raise
