"""Purpose-scoped extension of original MaxBudget, SAME cumulative SQLite table/500 guard."""
from pathlib import Path
import ast,hashlib,json,re,sqlite3,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import qwen_max_real100_20261030 as api
PURPOSE='ipd-cloud-new-generator-mainline-20261101-v1';MODEL='qwen3.8-max';SCALE=10**12
CONTRACT=dict(purpose=PURPOSE,model=MODEL,max_calls=9600,max_input_tokens=240000000,max_output_tokens=9600,max_pico=400*SCALE,input_reserve_each=1048576,output_reserve_each=1,single_inflight=True,original_global_pico=500*SCALE)

def norm(sql):
 # Preserve every byte of SQL string literals, including purpose case and spaces.
 parts=re.split(r"('(?:''|[^'])*')",sql)
 return ''.join(part if i%2 else re.sub(r'\s+','',part).replace('IFNOTEXISTS','') for i,part in enumerate(parts)).rstrip(';')
def expected_global_triggers():
 out={}
 for n in ast.walk(ast.parse(Path(api.__file__).read_text())):
  if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.lstrip().startswith('CREATE TRIGGER IF NOT EXISTS sgqa_global500_'):out[re.search(r'EXISTS\s+(sgqa_global500_\w+)',n.value).group(1)]=n.value
 assert len(out)==2;return out

def pool_trigger(contract):
 assert contract==CONTRACT
 return f"""CREATE TRIGGER IF NOT EXISTS sgqa_qwen_ipd_mainline_v1_insert BEFORE INSERT ON qwen_max_calls
 WHEN NEW.purpose='{PURPOSE}' AND (
 NEW.model!='{MODEL}' OR NEW.inp!={CONTRACT['input_reserve_each']} OR NEW.out!=1 OR NEW.pico!={api.cost(CONTRACT['input_reserve_each'],1)} OR NEW.settled!=0 OR NEW.usage_known!=0 OR
 (SELECT COUNT(*) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+1>{CONTRACT['max_calls']} OR
 (SELECT COALESCE(SUM(inp),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.inp>{CONTRACT['max_input_tokens']} OR
 (SELECT COALESCE(SUM(out),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.out>{CONTRACT['max_output_tokens']} OR
 (SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.pico>{CONTRACT['max_pico']} OR
 EXISTS(SELECT 1 FROM qwen_max_calls WHERE purpose='{PURPOSE}' AND settled=0))
 BEGIN SELECT RAISE(ABORT,'CloudIPDMainlinePoolCapOrInflight'); END"""

class MainlineBudget(api.MaxBudget):
 def __init__(self,path):
  super().__init__(path)
  with sqlite3.connect(self.path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');self.check_global(db)
   db.execute('CREATE TABLE IF NOT EXISTS qwen_ipd_pool_contracts(purpose TEXT PRIMARY KEY NOT NULL,config TEXT NOT NULL)');old=db.execute('SELECT config FROM qwen_ipd_pool_contracts WHERE purpose=?',(PURPOSE,)).fetchone()
   if old:assert json.loads(old[0])==CONTRACT,'Pool contract changed, do not reset'
   else:db.execute('INSERT INTO qwen_ipd_pool_contracts(purpose,config) VALUES(?,?)',(PURPOSE,json.dumps(CONTRACT,sort_keys=True)))
   db.execute(pool_trigger(CONTRACT));actual=db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='sgqa_qwen_ipd_mainline_v1_insert'").fetchone();assert actual and norm(actual[0])==norm(pool_trigger(CONTRACT)),'Pool trigger changed'
 @staticmethod
 def check_global(db):
  actual=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name IN ('sgqa_global500_qwen_insert','sgqa_global500_legacy_insert')").fetchall());expected=expected_global_triggers();assert set(actual)==set(expected) and all(norm(actual[k])==norm(expected[k]) for k in expected),'Original exact cumulative500 triggers changed'
  assert json.loads(db.execute('SELECT config FROM settings').fetchone()[0])['global_pico']==500*SCALE
 def reserve(self,tag,purpose,model,body_sha,inp=1048576,out=1):
  assert purpose==PURPOSE and model==MODEL and re.fullmatch(r'cloud-main-v1/seed0[12]/202610275[123]/step(00[1-9]|0[1-4][0-9]|050)/task[0-9a-f]{12}/pos(0[0-9]|[12][0-9]|3[01])',tag),'Only frozen mainline tags'
  assert type(inp) is int and inp==CONTRACT['input_reserve_each'] and out==1 and re.fullmatch('[0-9a-f]{64}',body_sha)
  amount=api.cost(inp,out)
  with sqlite3.connect(self.path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');self.check_global(db);assert json.loads(db.execute('SELECT config FROM qwen_ipd_pool_contracts WHERE purpose=?',(PURPOSE,)).fetchone()[0])==CONTRACT
   # Earlier public STAR/probe package retains its historical111/10 caps; no old rows/settings are rewritten.
   oldn,oldusd=db.execute("SELECT COUNT(*),COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose NOT IN (?, 'historical-smoke')",(PURPOSE,)).fetchone();assert oldn<=111 and oldusd<=10*SCALE,'Original public package cap violated'
   n,i,o,spent,inflight=db.execute('SELECT COUNT(*),COALESCE(SUM(inp),0),COALESCE(SUM(out),0),COALESCE(SUM(pico),0),COALESCE(SUM(1-settled),0) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
   total=db.execute('SELECT COALESCE(SUM(pico),0) FROM calls').fetchone()[0]+db.execute('SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls').fetchone()[0]
   assert n+1<=CONTRACT['max_calls'] and i+inp<=CONTRACT['max_input_tokens'] and o+out<=CONTRACT['max_output_tokens'] and spent+amount<=CONTRACT['max_pico'] and total+amount<=500*SCALE and inflight==0,'Pool/global/token/call/inflight cap'
   return db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico) VALUES(?,?,?,?,?,?,?)',(tag,purpose,model,body_sha,inp,out,amount)).lastrowid
 def settle(self,call,usage):
  with sqlite3.connect(self.path,timeout=10) as db:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT inp,out,pico,settled,purpose,model,tag FROM qwen_max_calls WHERE id=?',(call,)).fetchone()
   assert row and row[4]==PURPOSE and row[5]==MODEL and re.fullmatch(r'cloud-main-v1/seed0[12]/202610275[123]/step(00[1-9]|0[1-4][0-9]|050)/task[0-9a-f]{12}/pos(0[0-9]|[12][0-9]|3[01])',row[6]),'Foreign row cannot be settled by mainline'
   inp,out,reserved,settled=row[:4]
   if settled:raise PermissionError('Callalreadysettled')
   assert (inp,out,reserved)==(CONTRACT['input_reserve_each'],1,api.cost(CONTRACT['input_reserve_each'],1))
   if usage is None:db.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=?',(call,));return
   i,o=usage['prompt_tokens'],usage['completion_tokens'];assert type(i) is int and type(o) is int
   charge=api.cost(i,o)
   if min(i,o)<0 or i>inp or o>out or charge>reserved:raise PermissionError('Usageexceededreservation')
   db.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=?',(i,o,charge,call))
 def mainline_summary(self):
  total=self.summary()
  with sqlite3.connect(self.path,timeout=10) as db:
   row=db.execute('SELECT COUNT(*),COALESCE(SUM(inp),0),COALESCE(SUM(out),0),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM qwen_max_calls WHERE purpose=?',(PURPOSE,)).fetchone()
  return dict(total=total,pool=dict(attempts=row[0],input_tokens=row[1],output_tokens=row[2],conservative_USD=row[3]/SCALE,unknown=row[4],inflight=row[5]),contract=CONTRACT,all_original_rows_kept=True)
