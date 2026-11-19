"""New directteacher purpose only; original cumulative ledger and historical pools unchanged."""
import hashlib,json,re,sqlite3
import qwen_max_real100_20261030 as api
from cloud_qwen_ipd_mainline_budget_20261101 import norm,MainlineBudget
PURPOSE='direct-teacher-same-Test-20261102-v1'
MODEL='qwen3.8-max';SCALE=10**12
CONTRACT=dict(purpose=PURPOSE,model=MODEL,max_calls=1569,max_input_tokens=11000000,max_output_tokens=50208,max_pico=20*SCALE,input_reserve_each=1048576,output_reserve_each=32,single_inflight=True,original_global_pico=500*SCALE)
TRIGGER='sgqa_direct_teacher_20261102_insert'
def valid_tag(tag):
 m=re.fullmatch(r'direct-teacher-20261102/(\d{4})',tag)
 return bool(m and 0<=int(m[1])<1569)
def trigger():
 c=CONTRACT
 return f"""CREATE TRIGGER IF NOT EXISTS {TRIGGER} BEFORE INSERT ON qwen_max_calls
 WHEN NEW.purpose='{PURPOSE}' AND (
 NEW.model!='{MODEL}' OR NEW.inp!=1048576 OR NEW.out!=32 OR NEW.pico!={api.cost(1048576,32)} OR NEW.settled!=0 OR NEW.usage_known!=0 OR
 (SELECT COUNT(*) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+1>{c['max_calls']} OR
 (SELECT COALESCE(SUM(inp),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.inp>{c['max_input_tokens']} OR
 (SELECT COALESCE(SUM(out),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.out>{c['max_output_tokens']} OR
 (SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.pico>{c['max_pico']} OR
 EXISTS(SELECT 1 FROM qwen_max_calls WHERE purpose='{PURPOSE}' AND settled=0))
 BEGIN SELECT RAISE(ABORT,'DirectTeacherPhaseCapOrInflight'); END"""
def old_snapshot(db):
 names=('settings','calls','qwen_max_calls','qwen_ipd_pool_contracts')
 out={}
 for name in names:
  if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(name,)).fetchone():
   sql='SELECT * FROM '+name+(' WHERE purpose!=?' if name=='qwen_max_calls' else '')
   rows=db.execute(sql,(PURPOSE,) if name=='qwen_max_calls' else ()).fetchall()
   out[name]=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 out['old_triggers']=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name!=?",(TRIGGER,)).fetchall())
 return out
class DirectBudget(api.MaxBudget):
 def __init__(self,path):
  self.path=path
  with sqlite3.connect(path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');MainlineBudget.check_global(db);before=old_snapshot(db)
   db.execute('CREATE TABLE IF NOT EXISTS qwen_direct_teacher_pool_contracts(purpose TEXT PRIMARY KEY NOT NULL,config TEXT NOT NULL)')
   row=db.execute('SELECT config FROM qwen_direct_teacher_pool_contracts WHERE purpose=?',(PURPOSE,)).fetchone()
   if row:assert json.loads(row[0])==CONTRACT,'No reset or contract drift'
   else:db.execute('INSERT INTO qwen_direct_teacher_pool_contracts VALUES(?,?)',(PURPOSE,json.dumps(CONTRACT,sort_keys=True)))
   db.execute(trigger());self.check(db);assert old_snapshot(db)==before,'Old ledger/settings/triggers changed'
 def check(self,db):
  MainlineBudget.check_global(db)
  row=db.execute('SELECT config FROM qwen_direct_teacher_pool_contracts WHERE purpose=?',(PURPOSE,)).fetchone()
  assert row and json.loads(row[0])==CONTRACT
  row=db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",(TRIGGER,)).fetchone()
  assert row and norm(row[0])==norm(trigger()),'Exact new purpose guard changed'
 def reserve(self,tag,purpose,model,body_sha,inp=1048576,out=32):
  assert valid_tag(tag) and purpose==PURPOSE and model==MODEL
  assert type(inp) is int and type(out) is int and (inp,out)==(1048576,32) and re.fullmatch('[0-9a-f]{64}',body_sha)
  amount=api.cost(inp,out)
  with sqlite3.connect(self.path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');self.check(db)
   # Prior public package and synthetic-IPD package retain their own historical limits.
   n,usd=db.execute("SELECT COUNT(*),COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose NOT IN (?,?, 'historical-smoke')",(PURPOSE,'ipd-cloud-new-generator-mainline-20261101-v1')).fetchone()
   assert n<=111 and usd<=10*SCALE,'Historical package changed'
   return db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico) VALUES(?,?,?,?,?,?,?)',(tag,PURPOSE,MODEL,body_sha,inp,out,amount)).lastrowid
 def settle(self,call,usage):
  with sqlite3.connect(self.path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');self.check(db)
   row=db.execute('SELECT inp,out,pico,settled,purpose,model,tag FROM qwen_max_calls WHERE id=?',(call,)).fetchone()
   assert row and row[4]==PURPOSE and row[5]==MODEL and valid_tag(row[6]),'Foreign row'
   assert row[3]==0 and row[:3]==(1048576,32,api.cost(1048576,32)),'Already settled or altered reserve'
   if usage is None:db.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=?',(call,));return
   i,o=usage['prompt_tokens'],usage['completion_tokens']
   assert type(i) is int and type(o) is int and 0<=i<=1048576 and 0<=o<=32
   charge=api.cost(i,o);assert charge<=row[2]
   db.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=?',(i,o,charge,call))
 def summary(self):
  with sqlite3.connect(f'file:{self.path}?mode=ro',uri=True,timeout=10) as db:
   db.execute('BEGIN');self.check(db)
   old=db.execute('SELECT COUNT(*),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM calls').fetchone()
   new=db.execute('SELECT COUNT(*),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM qwen_max_calls').fetchone()
   phase=db.execute('SELECT COUNT(*),COALESCE(SUM(inp),0),COALESCE(SUM(out),0),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
   return dict(epoch=__import__('time').time(),legacy=old,all_Qwen=new,phase=dict(calls=phase[0],input_accounted=phase[1],output_accounted=phase[2],conservative_USD=phase[3]/SCALE,unknown=phase[4],inflight=phase[5]),cumulative_conservative_USD=(old[1]+new[1])/SCALE,cumulative_cap_USD=500,contract=CONTRACT,invoice_reconciled=False)
