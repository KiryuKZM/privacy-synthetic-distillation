"""Authorized150-query public/synthetic teacher probe with cumulative caps."""
from pathlib import Path
import argparse,collections,copy,fcntl,hashlib,json,os,re,signal,sqlite3,ssl,sys,time,urllib.error,urllib.request
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import qwen_max_real100_20261030 as api
import ust_missingness_delivery_20261107 as analysis
import ust_missingness_data_20261107 as data
S=R/'runs/2026-11-08-ust-teacher-comparison-v1';LEDGER=api.LEDGER
PURPOSE='ust-teacher-comparison-20261108-v1';TASK='ust-autonomous-research-500CNY-20261106';PREFIX='teacher20261108/'
MODEL='qwen3.8-max-0902';INP=8192;OUT=512;CNY=10**6;USD=10**12;RESERVE=api.cost(INP,OUT);STOP=False
VLA='generator-tail-outlier-ipd-20261104-v1';UST='ust-environment-generator-ipd-20261105-v1';OLDQUALITY='ust-teacher-domain-quality-20261105-v1'
load=analysis.load;pin=analysis.pin;save=analysis.save

def wire(row):return json.dumps(row['body'],ensure_ascii=False,separators=(',',':')).encode()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def stopped(*_):
 global STOP;STOP=True

def verify():
 p=load(S/'api-protocol.json');assert p['source']==pin(__file__)
 for x in p['bindings']:assert pin(R/x['path'])==x,x['path']
 assert load(S/'api-user-authorization.json')['authorized'] is True
 if STOP or (S/'STOP').exists():raise InterruptedError('Own teacher STOP')
 return p

def prepare():
 review=load(S/'preparation-review.json');rows=load(S/'public-proxy-API-PENDING.local.json');gold=load(S/'public-proxy-answers.local.json');assert len(rows)==150
 h=R/'runs/2026-11-05-ust-teacher-domain-quality-v1';old=load(h/'requests.local.json');public={x['messages'][1]['content'] for x in old if x['cohort']=='public_original' and x['ability']=='st_understanding'}
 current={x['question'] for x in load(data.S/'data/coverage8_native/source68.local.json')}
 assert len(public)==7 and len(current)==68
 features={};pairs=collections.defaultdict(dict);threshold=load(R/'runs/2026-11-05-ust-environment-generator-ipd-v1/tail-outlier-utility-v2/thresholds.local.json')['positive_field_q01_q99'];rawseen=set()
 for row in rows:
  b=row['body'];assert b['model']==MODEL and b['temperature']==0 and b['enable_thinking'] is False and b['max_tokens']==OUT and b['stream'] is False
  assert len(b['messages'])==2 and [x['role'] for x in b['messages']]==['system','user']
  assert b['messages'][0]['content']=='Solve the supplied urban spatiotemporal task from its data, objective and feedback. Follow the stated output format.'
  q=b['messages'][1]['content'];assert row['conservative_input_token_bound']<=INP
  if row['view']=='original':assert q in (public if row['cohort']=='public_original7' else current)
  cert=data.certify(q,gold[row['id']]);assert cert['metric']==row['metric']
  obs,opts=analysis.sections(q);values=[obs[i][row['metric']] for i in range(1,5)];j=analysis.FIELDS.index(row['metric']);lo,hi=threshold[j]
  features[row['id']]=dict(target_NaN=any(v is None for v in values),displayed_tie=cert['displayed_ties'],target_tail=any(v is not None and v>0 and (float(v)<lo or float(v)>hi) for v in values))
  group=row['id'].rsplit('/',1)[0];pairs[group][row['view']]=row
  assert len(wire(row))<=32768 and hashlib.sha256(wire(row)).hexdigest() not in rawseen;rawseen.add(hashlib.sha256(wire(row)).hexdigest())
 maps={}
 for k,v in pairs.items():
  a,b=v['original'],v['joint'];m=analysis.effective_map(a['body']['messages'][1]['content'],b['body']['messages'][1]['content']);assert m[gold[a['id']]]==gold[b['id']];maps[k]=m
 save(S/'api-features.local.json',features);save(S/'api-effective-maps.local.json',maps)
 save(S/'api-protocol.json',dict(source=pin(__file__),bindings=[pin(S/n) for n in ['public-proxy-API-PENDING.local.json','public-proxy-answers.local.json','api-features.local.json','api-effective-maps.local.json','preparation-review.json','api-user-authorization.json']]+[pin(api.__file__),pin(analysis.__file__),pin(data.__file__),pin(analysis.PARSER)],
  purpose=PURPOSE,task=TASK,provider=api.URL,model=MODEL,calls=150,zero_shot=True,thinking=False,temperature=0,max_output=OUT,input_reserve=INP,task_CNY_cap=500,phase_CNY_cap=20,prior_task_CNY=0,price_CNY=[12,36],price_USD=[1.65,4.951],price_source=review['original_teacher_option']['teacher_prices']['source'],retries=0,deadline_seconds=3600,original_public_sources=7,current_synthetic_sources=68,Final_access=False,protected_original_queries_sent=0,all_outputs_before_scoring=True))
 print('PREPARED150',flush=True)

def sqls():
 family=','.join(repr(x) for x in (VLA,UST,OLDQUALITY,PURPOSE));ust=','.join(repr(x) for x in (UST,OLDQUALITY,PURPOSE))
 return [
 f"""CREATE TRIGGER ustteacher26_insert BEFORE INSERT ON qwen_max_calls WHEN NEW.purpose='{PURPOSE}' OR NEW.tag GLOB '{PREFIX}*' BEGIN SELECT CASE WHEN
 NEW.purpose!='{PURPOSE}' OR NEW.model!='{MODEL}' OR NEW.inp!={INP} OR NEW.out!={OUT} OR NEW.pico!={RESERVE} OR NEW.settled!=0 OR NEW.usage_known!=0 OR
 NOT EXISTS(SELECT 1 FROM ustteacher26_allowlist WHERE tag=NEW.tag AND body_sha=NEW.body_sha) OR
 (SELECT count(*) FROM qwen_max_calls WHERE purpose='{PURPOSE}')>=150 OR
 (SELECT coalesce(sum(inp*12+out*36),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+{INP*12+OUT*36}>{20*CNY} OR
 (SELECT coalesce(sum(q.inp*c.input_micro+q.out*c.output_micro),0) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task='{TASK}')+{INP*12+OUT*36}>{500*CNY} OR
 EXISTS(SELECT 1 FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task='{TASK}' AND q.settled=0) OR
 (SELECT coalesce(sum(pico),0) FROM qwen_max_calls WHERE purpose IN({family}))+NEW.pico>{100*USD} OR
 (SELECT coalesce(sum(pico),0) FROM qwen_max_calls WHERE purpose IN({ust}))+NEW.pico>{60*USD}
 THEN RAISE(ABORT,'TeacherPayloadCountPhase20Task500OrPriorBudget') END; END""",
 f"""CREATE TRIGGER ustteacher26_update BEFORE UPDATE ON qwen_max_calls WHEN OLD.purpose='{PURPOSE}' OR NEW.purpose='{PURPOSE}' BEGIN SELECT CASE WHEN
 NEW.id!=OLD.id OR NEW.tag!=OLD.tag OR NEW.purpose!=OLD.purpose OR NEW.model!=OLD.model OR NEW.body_sha!=OLD.body_sha OR OLD.settled!=0 OR NEW.settled!=1 OR NEW.usage_known NOT IN(0,1) OR
 typeof(NEW.inp)!='integer' OR typeof(NEW.out)!='integer' OR typeof(NEW.pico)!='integer' OR NEW.inp<0 OR NEW.inp>OLD.inp OR NEW.out<0 OR NEW.out>OLD.out OR NEW.pico<0 OR NEW.pico>OLD.pico OR
 (NEW.usage_known=1 AND NEW.pico!=NEW.inp*1650000+NEW.out*4951000) OR
 (NEW.usage_known=0 AND (NEW.inp!=OLD.inp OR NEW.out!=OLD.out OR NEW.pico!=OLD.pico)) THEN RAISE(ABORT,'TeacherImmutableOneSettlement') END; END""",
 f"CREATE TRIGGER ustteacher26_delete BEFORE DELETE ON qwen_max_calls WHEN OLD.purpose='{PURPOSE}' BEGIN SELECT RAISE(ABORT,'ImmutableTeacherHistory'); END",
 ]+[f"CREATE TRIGGER ustteacher26_allow_{event.lower()} BEFORE {event} ON ustteacher26_allowlist BEGIN SELECT RAISE(ABORT,'FrozenTeacherAllowlist'); END" for event in ('INSERT','UPDATE','DELETE')]+[
 "CREATE TRIGGER ustresearch26_contract_update BEFORE UPDATE ON ust_research_cny_contracts BEGIN SELECT RAISE(ABORT,'ImmutableCNYContract'); END",
 "CREATE TRIGGER ustresearch26_contract_delete BEFORE DELETE ON ust_research_cny_contracts BEGIN SELECT RAISE(ABORT,'ImmutableCNYContract'); END"]

def install(path):
 verify();rows=load(S/'public-proxy-API-PENDING.local.json')
 with sqlite3.connect(path,timeout=15) as db:
  db.execute('BEGIN IMMEDIATE');before=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'"));assert json.loads(db.execute('select config from settings').fetchone()[0])['global_pico']==500*USD
  tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
  if 'ustteacher26_allowlist' not in tables:
   db.execute('CREATE TABLE ustteacher26_allowlist(tag TEXT PRIMARY KEY,body_sha TEXT UNIQUE NOT NULL)');db.executemany('INSERT INTO ustteacher26_allowlist VALUES(?,?)',[(r['id'],hashlib.sha256(wire(r)).hexdigest()) for r in rows])
  assert sorted(db.execute('SELECT * FROM ustteacher26_allowlist'))==sorted((r['id'],hashlib.sha256(wire(r)).hexdigest()) for r in rows)
  if 'ust_research_cny_contracts' not in tables:db.execute('CREATE TABLE ust_research_cny_contracts(purpose TEXT PRIMARY KEY,task TEXT NOT NULL,input_micro INTEGER NOT NULL,output_micro INTEGER NOT NULL,task_cap_micro INTEGER NOT NULL)')
  contract=(PURPOSE,TASK,12,36,500*CNY)
  old=db.execute('SELECT * FROM ust_research_cny_contracts WHERE purpose=?',(PURPOSE,)).fetchone()
  if old is None:db.execute('INSERT INTO ust_research_cny_contracts VALUES(?,?,?,?,?)',contract)
  else:assert old==contract
  for sql in sqls():
   name=re.search('CREATE TRIGGER (\\w+)',sql)[1]
   if name in before:assert before[name]==sql
   else:db.execute(sql)
  assert all(db.execute('SELECT sql FROM sqlite_master WHERE name=?',(n,)).fetchone()==(sql,) for n,sql in before.items())

def reserve(path,row):
 with sqlite3.connect(path,timeout=15) as db:
  db.execute('BEGIN IMMEDIATE')
  for sql in sqls():
   name=re.search('CREATE TRIGGER (\\w+)',sql)[1];assert db.execute('SELECT sql FROM sqlite_master WHERE name=?',(name,)).fetchone()==(sql,)
  return db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico,settled,usage_known) VALUES(?,?,?,?,?,?,?,0,0)',(row['id'],PURPOSE,row['body']['model'],hashlib.sha256(wire(row)).hexdigest(),INP,OUT,RESERVE)).lastrowid

def settle(path,call,usage):
 with sqlite3.connect(path,timeout=15) as db:
  db.execute('BEGIN IMMEDIATE')
  if usage is None:db.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=? AND purpose=?',(call,PURPOSE))
  else:
   i,o=usage['prompt_tokens'],usage['completion_tokens'];assert type(i)is int and type(o)is int
   db.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=? AND purpose=?',(i,o,api.cost(i,o),call,PURPOSE))

def budget(path=LEDGER):
 with sqlite3.connect(path) as db:
  n,amount,inflight,unknown,usd=db.execute('SELECT count(*),coalesce(sum(inp*12+out*36),0),coalesce(sum(1-settled),0),coalesce(sum(1-usage_known),0),coalesce(sum(pico),0) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
  task=db.execute('SELECT coalesce(sum(q.inp*c.input_micro+q.out*c.output_micro),0) FROM qwen_max_calls q JOIN ust_research_cny_contracts c ON q.purpose=c.purpose WHERE c.task=?',(TASK,)).fetchone()[0]
  hist=db.execute('SELECT (SELECT coalesce(sum(pico),0) FROM calls)+(SELECT coalesce(sum(pico),0) FROM qwen_max_calls)').fetchone()[0]
 return dict(HTTP_attempts=n,phase_CNY=amount/CNY,phase_cap_CNY=20,cumulative_task_CNY=task/CNY,task_cap_CNY=500,remaining_task_CNY=(500*CNY-task)/CNY,inflight=inflight,unknown_or_unsettled=unknown,phase_USD=usd/USD,historical_all_accounted_USD=hist/USD,cost='conservative posted token rates, cache discounts ignored; unresolved usage retains full reservation; not invoice reconciliation',epoch=time.time())

def tests():
 verify();folder=S/'api-budget-tests';folder.mkdir();rows=load(S/'public-proxy-API-PENDING.local.json');passed=[]
 def clone(name):
  p=folder/(name+'.sqlite')
  with sqlite3.connect('file:'+str(LEDGER)+'?mode=ro',uri=True) as src,sqlite3.connect(p) as dst:src.backup(dst)
  install(p);return p
 def refuses(label,fn):
  try:fn()
  except sqlite3.IntegrityError:passed.append(label);return
  raise AssertionError('Expected SQL refusal: '+label)
 p=clone('standard');c=reserve(p,rows[0]);refuses('one in-flight',lambda:reserve(p,rows[1]));settle(p,c,dict(prompt_tokens=1000,completion_tokens=20));assert budget(p)['phase_CNY']==.01272;passed.append('valid dual currency settlement')
 refuses('duplicate',lambda:reserve(p,rows[0]));refuses('second settlement',lambda:settle(p,c,None))
 with sqlite3.connect(p) as db:
  refuses('history deletion',lambda:db.execute('DELETE FROM qwen_max_calls WHERE id=?',(c,)))
  refuses('allowlist update',lambda:db.execute('DELETE FROM ustteacher26_allowlist'))
  refuses('contract update',lambda:db.execute('UPDATE ust_research_cny_contracts SET task_cap_micro=1'))
 bad=copy.deepcopy(rows[1]);bad['body']['messages'][1]['content']='different';refuses('payload alteration',lambda:reserve(p,bad))
 bad=copy.deepcopy(rows[1]);bad['body']['model']='other';refuses('model alteration',lambda:reserve(p,bad))
 p=clone('unknown');c=reserve(p,rows[0]);settle(p,c,None);assert budget(p)['phase_CNY']==.116736;passed.append('unknown retains full reserve')
 p=clone('overusage');c=reserve(p,rows[0]);refuses('usage beyond reserve',lambda:settle(p,c,dict(prompt_tokens=INP+1,completion_tokens=1)))
 for label,purpose,inp,out,pico,n in [('phase20',PURPOSE,1700000,0,0,1),('task500','fixture_task',42000000,0,0,1),('legacy100',VLA,0,0,101*USD,1),('legacy60',UST,0,0,61*USD,1),('legacy500','fixture_old',0,0,501*USD,1),('calls150',PURPOSE,0,0,0,150)]:
  p=clone(label)
  with sqlite3.connect(p) as db:
   old=list(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'"))
   for name,_ in old:db.execute('DROP TRIGGER '+name)
   if label=='task500':db.execute('INSERT INTO ust_research_cny_contracts VALUES(?,?,?,?,?)',(purpose,TASK,12,36,500*CNY))
   for i in range(n):db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico,settled,usage_known) VALUES(?,?,?,?,?,?,?,1,1)',('FICTION_ONLY_'+str(i),purpose,MODEL,'FICTITIOUS',inp,out,pico))
   for _,sql in old:db.execute(sql)
  refuses(label,lambda:reserve(p,rows[0]))
 save(S/'api-budget-qualification.json',dict(status='PASS_CPU_ONLY_NO_CANONICAL_MUTATION',source=pin(__file__),cases=passed,new_API=0));print('PASS_BUDGET',len(passed),flush=True)

def run():
 verify();assert load(S/'api-budget-qualification.json')['source']==pin(__file__);rows=load(S/'public-proxy-API-PENDING.local.json')
 with sqlite3.connect(LEDGER) as db:
  before=dict(qwen_max_id=db.execute('SELECT max(id) FROM qwen_max_calls').fetchone()[0],qwen_hash=digest(list(db.execute('SELECT * FROM qwen_max_calls ORDER BY id'))),legacy_hash=digest(list(db.execute('SELECT * FROM calls ORDER BY id'))),triggers=list(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'")))
 save(S/'api-budget-before.json',before);install(LEDGER)
 token=api.key();save(S/'api-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time(),independent_session=os.getsid(0)==os.getpid(),source=pin(__file__),new_GPU=False))
 deadline=time.monotonic()+3600
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),api.NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
 with (S/'api.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for i,row in enumerate(rows):
   verify();assert time.monotonic()<deadline;output=S/'api-raw'/f'{i:03d}.json';started=output.with_suffix('.started.json');assert not output.exists() and not started.exists(),'No unknown replay'
   raw=wire(row);call=reserve(LEDGER,row);save(started,dict(id=row['id'],ledger_id=call,epoch=time.time(),body_sha=hashlib.sha256(raw).hexdigest()))
   usage=None;tick=time.monotonic();result=dict(id=row['id'],cohort=row['cohort'],metric=row['metric'],view=row['view'],ledger_id=call,body_sha=hashlib.sha256(raw).hexdigest())
   try:
    request=urllib.request.Request(api.URL,data=raw,method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with opener.open(request,timeout=75) as response:provider=json.loads(response.read(2**20))
    save(S/'api-provider-responses'/f'{i:03d}.local.json',provider)
    u=provider.get('usage',{})
    if all(type(u.get(k))is int for k in ('prompt_tokens','completion_tokens')):usage={k:u[k] for k in ('prompt_tokens','completion_tokens')}
    msg=provider['choices'][0]['message'];assert provider['model'] in (MODEL,'qwen3.8-max-2026-09-02','qwen3.8-max') and not msg.get('reasoning_content')
    result.update(status='RESPONDED',raw_generated=msg.get('content',''),finish_reason=provider['choices'][0]['finish_reason'],model_returned=provider['model'],usage=usage)
   except BaseException as error:
    result.update(status='UNKNOWN_OR_FAILURE',error_type=type(error).__name__,http_status=error.code if isinstance(error,urllib.error.HTTPError) else None)
   settle(LEDGER,call,usage);result['seconds']=time.monotonic()-tick;save(output,result);b=budget();atomic(S/'api-budget-live.json',b);atomic(R/'runs/ust-research-api-budget-cny-current.json',dict(**b,source=str(S.relative_to(R)),ledger=str(LEDGER.relative_to(R))))
   atomic(S/'api-live.json',dict(completed=i+1,total=150,epoch=time.time(),phase='CALLING',budget=b));print('CLOSED',i+1,150,result['status'],flush=True)
   if result['status']!='RESPONDED' or usage is None:raise RuntimeError('Call failed/unknown; full reserve retained, no retry')
 save(S/'api-all-outputs-frozen.json',dict(raw=[pin(S/'api-raw'/f'{i:03d}.json') for i in range(150)],epoch=time.time(),all_closed_before_scoring=True));score()

def wilson(c,n):
 if not n:return None
 z=1.95996398454;p=c/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
 return [max(0,center-half),min(1,center+half)]
def stats(rows):return dict(n=len(rows),correct=sum(x['correct'] for x in rows),valid=sum(x['valid'] for x in rows),capped=sum(x['capped'] for x in rows))
def score():
 verify();closure=load(S/'api-all-outputs-frozen.json');assert len(closure['raw'])==150
 for x in closure['raw']:assert pin(R/x['path'])==x
 import eval_ust_sg_qwen_modes_20261024 as parser
 independent=analysis.parse_function();rows=load(S/'public-proxy-API-PENDING.local.json');gold=load(S/'public-proxy-answers.local.json');features=load(S/'api-features.local.json');grades=[]
 for i,q in enumerate(rows):
  raw=load(S/'api-raw'/f'{i:03d}.json');assert raw['id']==q['id'] and raw['status']=='RESPONDED';g=parser.classify(raw['raw_generated'],'off','ust',dict(ability='st_understanding'),raw['finish_reason']);v=independent(raw['raw_generated'],raw['finish_reason']);assert v==g['prediction'] and (v in ('A','B','C','D'))==bool(g['valid'])
  grades.append(dict(id=q['id'],source_id=q['id'].rsplit('/',1)[0],cohort=q['cohort'],metric=q['metric'],view=q['view'],answer=gold[q['id']],prediction=v,correct=bool(g['valid'] and v==gold[q['id']]),valid=bool(g['valid']),capped=raw['finish_reason']=='length',**features[q['id']]))
 scores={};equiv={};maps=load(S/'api-effective-maps.local.json');byid={x['id']:x for x in grades}
 for cohort in ('public_original7','current_synthetic68'):
  scores[cohort]={}
  for view in ('original','joint'):
   rr=[x for x in grades if x['cohort']==cohort and x['view']==view];total=stats(rr)
   scores[cohort][view]=dict(all=total,descriptive_wilson95=wilson(total['correct'],total['n']),metric={m:stats([x for x in rr if x['metric']==m]) for m in ('distance','charging_time','waiting_time')},subgroups={f:stats([x for x in rr if x[f]]) for f in ('target_NaN','target_tail','displayed_tie')},finite=stats([x for x in rr if not x['target_NaN']]))
  original=[x for x in grades if x['cohort']==cohort and x['view']=='original'];equiv[cohort]=dict(n=len(original),both_correct=sum(x['correct'] and byid[x['source_id']+'/joint']['correct'] for x in original),equivariant=sum(x['valid'] and byid[x['source_id']+'/joint']['valid'] and maps[x['source_id']].get(x['prediction'])==byid[x['source_id']+'/joint']['prediction'] for x in original))
 before=load(S/'api-budget-before.json')
 with sqlite3.connect(LEDGER) as db:
  assert digest(list(db.execute('SELECT * FROM qwen_max_calls WHERE id<=? ORDER BY id',(before['qwen_max_id'],))))==before['qwen_hash']
  assert digest(list(db.execute('SELECT * FROM calls ORDER BY id')))==before['legacy_hash']
  assert all(db.execute('SELECT sql FROM sqlite_master WHERE name=?',(name,)).fetchone()==(sql,) for name,sql in before['triggers'])
 b=budget();assert b['HTTP_attempts']==150 and b['inflight']==b['unknown_or_unsettled']==0 and b['phase_CNY']<=20
 save(S/'api-grades.local.json',grades);result=dict(status='COMPLETE_ALL150_INDEPENDENT_PARSE',scores=scores,equivariance=equiv,budget=b,old_ledger_history_unchanged=True,old_guards_unchanged=True,protected_originals_sent=0,Final_access=False,new_GPU=0,model=MODEL,thinking=False,zero_shot=True,previous_teacher_7of7_not_current_generator=True,original_public_n=7,synthetic_n=68,no_equivalence_claim=True)
 save(S/'api-results.json',result)
 lines=['# 原API教师：公开原题与当前合成题小样本对照','',f'全部150次调用闭合。教师固定{MODEL}、不启用思考、温度0、输出上限512、相同零示例任务指令。仅7道此前明确公开的理解原题与当前生成器68道题，以及各自同步位置/选项重排；未发送真实Train/Dev或Final。使用零示例避免公开原题中包含既有示例而泄漏答案。参考来自本地数值规则独立核验，题面不附答案。','', '| 数据 | 原题正确 | 重排正确 | 两视图均正确 | 位置变换一致 |','|---|---:|---:|---:|---:|']
 for cohort,x in scores.items():
  aa=x['original']['all'];bb=x['joint']['all'];ee=equiv[cohort];lines.append(f'| {cohort} | {aa["correct"]}/{aa["n"]} | {bb["correct"]}/{bb["n"]} | {ee["both_correct"]}/{ee["n"]} | {ee["equivariant"]}/{ee["n"]} |')
 lines+=['','| 数据/视图 | 距离 | 充电时间 | 等待时间 | 缺失目标 | 数值目标尾部 | 显示并列 |','|---|---:|---:|---:|---:|---:|---:|']
 for cohort,x in scores.items():
  for view,v in x.items():
   ss=[v['metric'][m] for m in ('distance','charging_time','waiting_time')]+[v['subgroups'][f] for f in ('target_NaN','target_tail','displayed_tie')];lines.append('| '+cohort+'/'+view+' | '+' | '.join(f'{t["correct"]}/{t["n"]}' for t in ss)+' |')
 lines+=['','两侧题型/缺失/尾部比例不同，不能把整体正确率差直接解释为生成器问题。原题只有7道且是公开校准题，不是完整真实开发集或独立最终验收；7/7的Wilson95%区间仍约为65%–100%，不足以证明相等。重排视图不算新增独立场景。当前生成68题曾用于学生训练；本次是教师直接答题诊断，不能证明学生充分吸收训练信号、IPD收益或合成替代。原规划/反思标签不可信问题不在本轮范围。所有无效/截断输出留在分母。','',f'API费用按官方无缓存原价保守记录：本轮{b["phase_CNY"]:.6f}元，研究累计{b["cumulative_task_CNY"]:.6f}元，500元剩余{b["remaining_task_CNY"]:.6f}元；本轮上限20元。人民币与原美元账同时保留，原历史支出与限制未改。没有新GPU推理或训练。价格来源：https://help.aliyun.com/zh/model-studio/qwen3-8-max 。','',f'所有配置、请求、原始输出、参考、独立评分、预算与失败记录位于{S}。API报告为api-results.json；此处的小样本结果不能套用到本地未执行的838题方案。','']
 report=R/'reports/ust-teacher-api-comparison-20261108.md'
 with report.open('x') as f:f.write('\n'.join(lines))
 save(S/'api-terminal.json',dict(status='COMPLETE',epoch=time.time(),results=pin(S/'api-results.json'),report=pin(report),budget=b));atomic(S/'api-live.json',dict(phase='COMPLETE',completed=150,total=150,epoch=time.time(),budget=b));print('COMPLETE150',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','tests','run','score']);a=p.parse_args();signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 try:globals()[a.action]()
 except BaseException as e:
  save(S/('api-failure-'+str(time.time_ns())+'.json'),dict(type=type(e).__name__,error=str(e)[:500],epoch=time.time(),action=a.action,no_retries=True));raise
