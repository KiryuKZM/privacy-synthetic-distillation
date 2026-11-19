"""Thin new-purpose contract using frozen canonical PairBudget operations, no old pool reset."""
from pathlib import Path
import hashlib,re,types
import cloud_qwen_public_pair100_budget_20261102 as parent
assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest()=='f8d349f70b9b191b0b4e608bbb7b56535eab01969b32b3cc346d3a0a135ab333'
api=parent.api;previous=parent.previous;SCALE=parent.SCALE
PURPOSE='matched-qwen-objectives-public50x2-20261102-v1';MODEL=parent.MODEL;TRIGGER='sgqa_matched_objectives_20261102_insert'
CONTRACT=dict(purpose=PURPOSE,model=MODEL,max_calls=6400,max_input_tokens=12000000,max_output_tokens=6400,max_pico=20*SCALE,input_reserve_each=1048576,output_reserve_by_mode=dict(opd=1,ipd=1),single_inflight=True,original_global_pico=500*SCALE)
assert api.cost(CONTRACT['max_input_tokens'],CONTRACT['max_output_tokens'])<=CONTRACT['max_pico']
def tag_mode(tag):
 m=re.fullmatch(r'matched-v1/(opd|ipd)/slot([0-9]{3})/pos([0-9]{2})',tag)
 assert m and 1<=int(m[2])<=100 and 0<=int(m[3])<32,'Only exact two100slot/32position schedules'
 return m[1]
def trigger():
 c=CONTRACT
 return f"""CREATE TRIGGER IF NOT EXISTS {TRIGGER} BEFORE INSERT ON qwen_max_calls
 WHEN NEW.purpose='{PURPOSE}' AND (
 NEW.model!='{MODEL}' OR NEW.inp!=1048576 OR NEW.out!=1 OR NEW.pico!={api.cost(1048576,1)} OR NEW.settled!=0 OR NEW.usage_known!=0 OR
 NOT (NEW.tag GLOB 'matched-v1/[oi]pd/slot[0-9][0-9][0-9]/pos[0-9][0-9]' AND CAST(substr(NEW.tag,20,3) AS INTEGER) BETWEEN 1 AND 100 AND CAST(substr(NEW.tag,27,2) AS INTEGER) BETWEEN 0 AND 31) OR
 (SELECT COUNT(*) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+1>{c['max_calls']} OR
 (SELECT COALESCE(SUM(inp),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.inp>{c['max_input_tokens']} OR
 (SELECT COALESCE(SUM(out),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.out>{c['max_output_tokens']} OR
 (SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose='{PURPOSE}')+NEW.pico>{c['max_pico']} OR
 EXISTS(SELECT 1 FROM qwen_max_calls WHERE purpose='{PURPOSE}' AND settled=0))
 BEGIN SELECT RAISE(ABORT,'MatchedObjectiveCapOrInflight'); END"""
# Bind exact frozen method code to this purpose; do not mutate imported module globals.
context=dict(parent.__dict__);context.update(__name__=__name__,PURPOSE=PURPOSE,MODEL=MODEL,TRIGGER=TRIGGER,CONTRACT=CONTRACT,tag_mode=tag_mode,trigger=trigger)
def reuse(fn):
 value=types.FunctionType(fn.__code__,context,fn.__name__,fn.__defaults__,fn.__closure__);value.__kwdefaults__=fn.__kwdefaults__;return value
old_snapshot=reuse(parent.old_snapshot);context['old_snapshot']=old_snapshot
class ObjectiveBudget(parent.PairBudget):pass
for name in ('__init__','check','check_historical_pools','reserve','settle','summary'):
 setattr(ObjectiveBudget,name,reuse(getattr(parent.PairBudget,name)))
ObjectiveBudget.mainline_summary=ObjectiveBudget.summary
