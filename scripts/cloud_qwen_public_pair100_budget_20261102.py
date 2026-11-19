"""Purpose adapter extending the existing canonical Qwen ledger; no reset of old pools."""
import hashlib,json,re,sqlite3,time
import cloud_qwen_direct_teacher_budget_20261102 as previous
api=previous.api;SCALE=10**12
PURPOSE='public-direct-partial-pair100-20261102-v1';MODEL='qwen3.8-max';TRIGGER='sgqa_public_pair100_20261102_insert'
CONTRACT=dict(purpose=PURPOSE,model=MODEL,max_calls=200,max_input_tokens=3000000,max_output_tokens=3300,max_pico=5*SCALE,input_reserve_each=1048576,output_reserve_by_mode=dict(direct=32,partial=1),single_inflight=True,original_global_pico=500*SCALE)
def tag_mode(tag):
 m=re.fullmatch(r'pair100-v1/([0-9]{3})/(direct|partial)',tag)
 assert m and int(m[1])<100,'Only frozen100x2 tags'
 return m[2]
def trigger():
 # CASE requires full context and actual arm output reservations, never arbitrary cheap initial rows.
 c=CONTRACT
 return f"""CREATE TRIGGER IF NOT EXISTS {TRIGGER} BEFORE INSERT ON qwen_max_calls
 WHEN NEW.purpose='{PURPOSE}' AND (
 NEW.model!='{MODEL}' OR NEW.inp!=1048576 OR NEW.settled!=0 OR NEW.usage_known!=0 OR
 NOT ((NEW.tag GLOB 'pair100-v1/[0-9][0-9][0-9]/direct' AND CAST(substr(NEW.tag,12,3) AS INTEGER)<100 AND NEW.out=32 AND NEW.pico={api.cost(1048576,32)}) OR
      (NEW.tag GLOB 'pair100-v1/[0-9][0-9][0-9]/partial' AND CAST(substr(NEW.tag,12,3) AS INTEGER)<100 AND NEW.out=1 AND NEW.pico={api.cost(1048576,1)})) OR
 (SELECT COUNT(*) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+1>{c['max_calls']} OR
 (SELECT COALESCE(SUM(inp),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.inp>{c['max_input_tokens']} OR
 (SELECT COALESCE(SUM(out),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.out>{c['max_output_tokens']} OR
 (SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.pico>{c['max_pico']} OR
 EXISTS(SELECT 1 FROM qwen_max_calls WHERE purpose='{PURPOSE}' AND settled=0))
 BEGIN SELECT RAISE(ABORT,'PublicPair100CapOrInflight'); END"""
def old_snapshot(db):
 result={}
 for table in ('settings','calls','qwen_max_calls','qwen_ipd_pool_contracts','qwen_direct_teacher_pool_contracts'):
  if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
   if table.endswith('_contracts'):result[table]=hashlib.sha256(b'[]').hexdigest()
   continue
  haspurpose=table in ('qwen_max_calls','qwen_ipd_pool_contracts','qwen_direct_teacher_pool_contracts')
  sql='SELECT * FROM '+table+(' WHERE purpose!=?' if haspurpose else '')
  rows=db.execute(sql,(PURPOSE,) if haspurpose else ()).fetchall()
  result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 result['old_triggers']=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name!=?",(TRIGGER,)))
 return result
class PairBudget(previous.DirectBudget):
 def __init__(self,path):
  self.path=path
  with sqlite3.connect(path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');previous.MainlineBudget.check_global(db);before=old_snapshot(db)
   db.execute('CREATE TABLE IF NOT EXISTS qwen_direct_teacher_pool_contracts(purpose TEXT PRIMARY KEY NOT NULL,config TEXT NOT NULL)')
   row=db.execute('SELECT config FROM qwen_direct_teacher_pool_contracts WHERE purpose=?',(PURPOSE,)).fetchone()
   if row:assert json.loads(row[0])==CONTRACT,'No pool reset'
   else:db.execute('INSERT INTO qwen_direct_teacher_pool_contracts VALUES(?,?)',(PURPOSE,json.dumps(CONTRACT,sort_keys=True)))
   db.execute(trigger());self.check(db);assert old_snapshot(db)==before
 def check(self,db):
  previous.MainlineBudget.check_global(db)
  row=db.execute('SELECT config FROM qwen_direct_teacher_pool_contracts WHERE purpose=?',(PURPOSE,)).fetchone();assert row and json.loads(row[0])==CONTRACT
  row=db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",(TRIGGER,)).fetchone();assert row and previous.norm(row[0])==previous.norm(trigger())
 def check_historical_pools(self,db):
  registered={}
  for table in ('qwen_ipd_pool_contracts','qwen_direct_teacher_pool_contracts'):
   if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
    for purpose,config in db.execute('SELECT purpose,config FROM '+table):
     c=json.loads(config);assert c['purpose']==purpose
     if purpose in registered:assert registered[purpose]==c
     registered[purpose]=c
  for purpose,c in registered.items():
   n,i,o,p,unknown_inflight=db.execute('SELECT COUNT(*),COALESCE(SUM(inp),0),COALESCE(SUM(out),0),COALESCE(SUM(pico),0),COALESCE(SUM(1-settled),0) FROM qwen_max_calls WHERE purpose=?',(purpose,)).fetchone()
   assert n<=c['max_calls'] and i<=c['max_input_tokens'] and o<=c['max_output_tokens'] and p<=c['max_pico'] and unknown_inflight<=1,'Historical registered cap violated'
  rows=db.execute("SELECT purpose,pico FROM qwen_max_calls WHERE purpose!='historical-smoke'").fetchall();public=[p for k,p in rows if k not in registered]
  assert len(public)<=111 and sum(public)<=10*SCALE,'Original public package111/10 remains'
 def reserve(self,tag,purpose,model,body_sha,inp=1048576,out=None):
  mode=tag_mode(tag);cap=CONTRACT['output_reserve_by_mode'][mode]
  assert purpose==PURPOSE and model==MODEL and type(inp) is int and inp==1048576 and (out is None or type(out) is int and out==cap) and re.fullmatch('[0-9a-f]{64}',body_sha)
  with sqlite3.connect(self.path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');self.check(db);self.check_historical_pools(db)
   return db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico) VALUES(?,?,?,?,?,?,?)',(tag,PURPOSE,MODEL,body_sha,1048576,cap,api.cost(1048576,cap))).lastrowid
 def settle(self,call,usage):
  with sqlite3.connect(self.path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');self.check(db)
   row=db.execute('SELECT tag,purpose,model,inp,out,pico,settled FROM qwen_max_calls WHERE id=?',(call,)).fetchone();assert row and row[1:3]==(PURPOSE,MODEL),'Foreign row'
   cap=CONTRACT['output_reserve_by_mode'][tag_mode(row[0])];assert row[3:]==(1048576,cap,api.cost(1048576,cap),0),'Already settled or altered reservation'
   if usage is None:db.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=?',(call,));return
   i,o=usage['prompt_tokens'],usage['completion_tokens'];assert type(i) is int and type(o) is int and 0<=i<=1048576 and 0<=o<=cap
   amount=api.cost(i,o);assert amount<=row[5]
   db.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=?',(i,o,amount,call))
 def summary(self):
  with sqlite3.connect(f'file:{self.path}?mode=ro',uri=True,timeout=10) as db:
   db.execute('BEGIN');self.check(db)
   rows={t:db.execute('SELECT COUNT(*),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM '+t).fetchone() for t in ('calls','qwen_max_calls')}
   v=db.execute('SELECT COUNT(*),COALESCE(SUM(inp),0),COALESCE(SUM(out),0),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
   return dict(epoch=time.time(),original_tables=rows,cumulative_conservative_USD=sum(x[1] for x in rows.values())/SCALE,phase=dict(HTTP=v[0],input_accounted=v[1],output_accounted=v[2],conservative_USD=v[3]/SCALE,unknown=v[4],inflight=v[5]),contract=CONTRACT,invoice_reconciled=False)
