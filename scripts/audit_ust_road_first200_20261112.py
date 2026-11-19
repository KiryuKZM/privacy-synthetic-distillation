"""Offline audit of frozen first200: no model calls, no changes to running evaluation."""
from pathlib import Path
import sys,json,collections,re,copy,contextlib,io
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import ust_original_baseline_runtime_20261111 as rt
O=rt.O;U=O/'student-qa-parallel-v1';D=O/'first200-output-audit-20261112'
def parse(text):
 blocks=re.findall(r'```(?:\w+)?\n([\s\S]*?)\n```',text)
 try:
  value=json.loads(blocks[-1] if blocks else text)
  return value['answer'],value['summary'],None
 except (ValueError,KeyError,TypeError,IndexError) as exc:return None,None,type(exc).__name__
def main():
 D.mkdir(exist_ok=False)
 chunks=[U/'qa/chunks'/f'{n:04}.json' for n in (0,100)]
 rows=sum([rt.load(p)['answers'] for p in chunks],[]);lookup={x['question']:i for i,x in enumerate(rows)}
 assert len(rows)==len(lookup)==200
 attempts=collections.defaultdict(list);pins=[rt.pin(p) for p in chunks]
 for p in sorted((U/'raw').glob('*.started.json')):
  request=rt.load(p);i=lookup.get(request['messages'][1]['content'])
  if i is None:continue
  path=p.with_name(p.name.replace('.started',''));output=rt.load(path)
  attempts[i].append(dict(path=str(path.relative_to(R)),request=request,output=output));pins += [rt.pin(p),rt.pin(path)]
 assert len(attempts)==200
 # Replay the actual official evaluator with only archived strings, in original 100/16 batching.
 class Archived:
  kind='student'
  def __init__(self):self.pos=collections.Counter()
  def chat(self,messages,phase):
   assert phase=='evaluate';out=[]
   for m in messages:
    i=lookup[m[1]['content']];a=attempts[i][self.pos[i]];assert a['request']['messages']==m
    self.pos[i]+=1;out.append(a['output']['content'])
   return out
 rt.runtime_dir(D)
 fake=Archived();llm=rt.agent(fake);llm.batch_size=16;replayed=[]
 with (D/'official-replay.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  for start in (0,100):
   inputs=[{k:v for k,v in x.items() if k not in ('reasoning','decision','is_correct','summary')} for x in rows[start:start+100]]
   replayed+=llm.evaluate(copy.deepcopy(inputs),'planning')[0]
 for i,(old,new) in enumerate(zip(rows,replayed)):
  assert all(old.get(k)==new.get(k) for k in ('decision','is_correct','reasoning','summary'))
  assert fake.pos[i]==len(attempts[i])
 per=[];rescues=[]
 for i,row in enumerate(rows):
  aa=attempts[i];accepted=[];meta=[]
  for a in aa:
   x=a['output'];answer,summary,error=parse(x['content'])
   if not error:accepted.append(answer)
   meta.append(dict(serial=x['serial'],path=a['path'],tokens=x['output_tokens'],finish=x['finish_reason'],parse_error=error,answer=answer))
  assert (accepted[0] if accepted else None)==row['decision']
  assert row['is_correct']==(row['decision']==row['answer'])
  # Search any embedded JSON object, independently of official fences; retain only A-D answers.
  if row['decision'] is None:
   for a in aa:
    text=a['output']['content']
    for match in re.finditer(r'\{',text):
     try:value,_=json.JSONDecoder().raw_decode(text[match.start():])
     except ValueError:continue
     if isinstance(value,dict) and value.get('answer') in ('A','B','C','D'):rescues.append(dict(index=i,serial=a['output']['serial'],answer=value['answer']))
  per.append(dict(source_index=i,stage=row['stage'],reference=row['answer'],decision=row['decision'],category='invalid' if row['decision'] is None else 'correct' if row['is_correct'] else 'reference_mismatch',attempts=meta))
 invalid=[x for x in per if x['category']=='invalid'];raw=[a['output'] for aa in attempts.values() for a in aa]
 assert all(len(x['attempts'])==3 and all(a['finish']=='length' and a['tokens']==8192 for a in x['attempts']) for x in invalid)
 summary=dict(n=200,counts=dict(collections.Counter(x['category'] for x in per)),valid_accuracy=19/69,official_accuracy=19/200,attempts=len(raw),finish_counts=dict(collections.Counter(x['finish_reason'] for x in raw)),output_tokens=sum(x['output_tokens'] for x in raw),capped_output_tokens=sum(x['output_tokens'] for x in raw if x['finish_reason']=='length'),all131_invalid_exhaust_three_8192_attempts=True,embedded_JSON_rescues=rescues,official_replay='ALL_200_MATCH_ALL_ATTEMPTS_CONSUMED',independent_parser='ALL_200_MATCH',by_stage={s:dict(collections.Counter(x['category'] for x in per if x['stage']==s)) for s in sorted({x['stage'] for x in per})},limitations=['Reference mismatch does not independently certify author label optimality','Conditional accuracy among valid outputs is selected subset, not recovered full200 accuracy','Underlying cause of repetition not isolated: model/prompt/decoding/backend remain hypotheses','No GPU/API calls; running experiment unchanged'])
 rt.save(D/'summary.json',summary);rt.save(D/'per-question.json',per)
 examples=[]
 for category in ('invalid','reference_mismatch','correct'):
  row=next(x for x in per if x['category']==category);i=row['source_index']
  examples.append(dict(**row,question=rows[i]['question'],raw_outputs=[dict(serial=a['output']['serial'],content=a['output']['content'],finish=a['output']['finish_reason'],tokens=a['output']['output_tokens']) for a in attempts[i]]))
 rt.save(D/'examples.json',examples)
 text=[]
 for e in examples:
  text += [f"Source index {e['source_index']} (one-based question {e['source_index']+1}); {e['category']}; reference {e['reference']}; parsed {e['decision']}",e['question']]
  for x in e['raw_outputs']:text += [f"Attempt serial {x['serial']}; {x['finish']}; {x['tokens']} tokens",x['content']]
 (D/'examples.txt').write_text('\n\n'.join(text))
 rt.save(D/'manifest.json',dict(code=rt.pin(__file__),inputs=pins,parser=rt.pin(rt.S/'utils/language_model.py')))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
