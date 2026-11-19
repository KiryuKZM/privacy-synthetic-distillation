"""Paired same72/same9 students: replace original8 with fixed synthetic8."""
from pathlib import Path
import ast,json,os,signal,subprocess,sys,time,traceback
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_fresh_zero_demo_student_cap1024_20261110 as c
P=c.PARENT;ZERO=P/'student-cap1024-v1';O=P/'student-synth8-cap1024-v1';FINAL=R/'runs/2026-11-09-ust-final-matched68-v1'
REAL8=P/'student-restore8-cap1024-v1';SCALE=R/'runs/2026-11-08-ust-scale-v1'
NAMES=c.NAMES;load=c.load;save=c.save;bind=c.bind;STOP=False

def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own synthetic8 STOP')
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def verify(unit=None):
 check();p=load(O/'student-protocol.json');assert p['source']==bind(__file__)
 for b in p['bindings']:assert bind(R/b['path'])==b,b['path']
 return load(Path(unit)/'protocol.json') if unit else p
def frozen_models():
 d=load(O/'student-models-frozen.json')
 for b in d['bindings']:assert bind(R/b['path'])==b,b['path']
 assert set(d['models'])==set(NAMES)
 return d['models']
def configure():
 c.O=O;c.check=check;c.live=live;c.verify=verify;c.frozen_models=frozen_models;c.configure()
def select_examples(original):
 import collections
 d=c.z.s.d;audit=c.z.s.audit
 source=SCALE/'data/current_n68-variants.local.json';rows=load(source)
 # Fixed prior to any new model output. Two missing targets, one finite tied target.
 slots=[('distance',False,False,'A'),('charging_time',True,False,'B'),('waiting_time',True,None,'C'),('distance',False,False,'D'),('charging_time',False,False,'A'),('waiting_time',False,True,'B'),('distance',False,False,'C'),('charging_time',False,False,'D')]
 def signature(question):
  obs,_=audit.sections(question)
  return d.sig([[None if obs[i][f] is None else float(obs[i][f]) for f in d.old.FIELDS] for i in range(1,5)])
 excluded={signature(q['question']) for q in load(ZERO/'queries.local.json')}
 used=set();chosen=[];certs=[]
 for metric,missing,tie,letter in slots:
  for row in sorted(rows,key=lambda r:(r['source_slot'],r['variant_index'])):
   if row['source_id'] in used or row['metric']!=metric or row['answer']!=letter:continue
   cert=d.certify(row['question'],row['answer'],True)
   if cert['target_NaN']!=missing or (tie is not None and cert['displayed_ties']!=tie):continue
   assert signature(row['question']) not in excluded,'Synthetic demo collides with frozen72; fail closed, no outcome-based refill'
   used.add(row['source_id']);chosen.append(row);certs.append(dict(id=row['id'],**cert));break
  else:raise RuntimeError('Insufficient frozen candidates for '+str((metric,missing,tie,letter)))
 assert len(used)==8 and collections.Counter(r['answer'] for r in chosen)==dict(A=2,B=2,C=2,D=2)
 prefix=list(original)
 for row in chosen:prefix += [dict(role='user',content=row['question']),dict(role='assistant',content=row['target'])]
 assert len(prefix)==17 and prefix[0]==load(REAL8/'fixed8.local.json')[0]
 save(O/'selected-synthetic8.local.json',chosen)
 save(O/'example-qualification.json',dict(source=bind(source),selection_slots=slots,proofs=certs,distinct_sources=8,no_numeric_group_overlap_with72=True,correct_position_counts=dict(collections.Counter(r['answer'] for r in chosen)),metric_counts=dict(collections.Counter(r['metric'] for r in chosen)),API=0))
 return prefix

def prepare():
 assert not (O/'student-protocol.json').exists();O.mkdir(exist_ok=True)
 assert load(ZERO/'cap1024-terminal.json')['status']=='COMPLETE_STUDENT_REINFERENCE_AND_PAIRED_TOKEN_CHECK'
 for fn in ['queries.local.json','answers.local.json','student-models-frozen.json']:
  (O/fn).write_bytes((ZERO/fn).read_bytes());assert bind(O/fn)['sha256']==bind(ZERO/fn)['sha256']
 original=load(ZERO/'fixed8.local.json');realprefix=load(REAL8/'fixed8.local.json')
 assert len(realprefix)==17 and original==realprefix[:1]
 prefix=select_examples(original)
 save(O/'fixed8.local.json',prefix)
 plan=dict(user_request='同样72题，将8道真实示例替换为8道合成示例，其他设置不变',models=NAMES,queries=72,numeric_groups=50,training_sources=68,training_updates=136,new_training=0,API=0,output_cap=1024,demonstrations=8,only_intervention='Replace original8 example package by the same fixed current-generator8 understanding examples for all9 models; unchanged system/query/gold/models/decoder',paired_real8=bind(REAL8/'student-results.json'),paired_zero=bind(ZERO/'student-results.json'),primary='All9 model accuracies and paired synthetic8 minus real8 gains/losses; fixed3-order means and conditional50-group bootstrap; no formal equivalence margin',secondary='Synthetic8 minus zero-demo gains/losses',selection='Current_n68 source variants; eight distinct sources, earliest source_slot matching preregistered metric/missing/tie slots, answer letters ABCDABCD using existing rotated variants; no model-output selection',limits=['Original8 mixed abilities versus synthetic8 understanding-only: package replacement, not isolated provenance effect.','Current generator demonstrations shared by all9 arms; some are in current training68, so this is practical reuse, not unseen-example generalization.','All9 old training runs used original real8 prompt context; inference replacement does not erase this historical exposure.','Generator fitted local real data; no formal privacy proof.','Same72 are now exposed diagnostic data, not new independent acceptance.'],all_raw_before_score=True)
 save(O/'study-plan.json',plan)
 from transformers import AutoTokenizer
 p0=load(ZERO/NAMES[0]/'protocol.json');tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);h=c.z.s.helper();q=load(O/'queries.local.json');prior_tokens=load(ZERO/'tokens.local.json');tokens={}
 for r in q:
  assert h.eval_prompt(tok,original,r['question'])==prior_tokens[r['id']]['prompt_ids']
  ids=h.eval_prompt(tok,prefix,r['question']);assert len(ids)+1024<=16384
  tokens[r['id']]=dict(prompt_ids=ids,target_ids=prior_tokens[r['id']]['target_ids'])
 save(O/'tokens.local.json',tokens)
 for name in NAMES:
  save(O/name/'queries.local.json',q);p0=load(ZERO/name/'protocol.json');p0.update(source=bind(__file__),demonstrations=8,maximum_new_tokens=1024,actual_decoder_cap=1024,training=0,steps=0);save(O/name/'protocol.json',p0)
 save(O/'cpu-decoder-contract.json',c.cpu_cap_contract())
 pins=load(ZERO/'student-protocol.json')['bindings']+[bind(__file__),bind(c.__file__),bind(ZERO/'cap1024-terminal.json'),bind(ZERO/'student-results.json'),bind(REAL8/'fixed8.local.json'),bind(REAL8/'student-results.json'),bind(SCALE/'data/current_n68-variants.local.json'),bind(c.z.s.d.__file__)]
 pins += [bind(REAL8/name/'graded.local.json') for name in NAMES]+[bind(ZERO/name/'graded.local.json') for name in NAMES]+[bind(p) for p in O.rglob('*.json')]
 save(O/'student-protocol.json',dict(source=bind(__file__),bindings=list({x['path']:x for x in pins}.values()),models=NAMES,actual_decoder_cap=1024,demonstrations=8,API=0,new_training=0,total_raw=648))
 verify();frozen_models();save(O/'student-preflight.json',dict(status='PASS_ONLY_REPLACE_EXAMPLE_PACKAGE_SYNTHETIC8',same9_models=True,same72_queries_gold=True,same_system=True,same1024_decoder=True,same_synthetic8_for_all9=True,synthetic_labels_independently_qualified=True,input_token_min=min(len(x['prompt_ids']) for x in tokens.values()),input_token_max=max(len(x['prompt_ids']) for x in tokens.values()),API=0,new_training=0));print('PREPARED_SYNTHETIC8_648',flush=True)

def score():
 source=Path(c.z.__file__).read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='score');fn=ast.get_source_segment(source,node)
 assert fn.count('COMPLETE_NINE_FROZEN_STUDENTS_ZERO_DEMOS')==1
 fn=fn.replace('COMPLETE_NINE_FROZEN_STUDENTS_ZERO_DEMOS','COMPLETE_NINE_FROZEN_STUDENTS_SYNTHETIC8')
 ns=dict(vars(c.z));exec(compile(fn,str(Path(__file__))+'::same_scorer_condition_label','exec'),ns);ns['score']()
 allscores={key:load(path/'student-results.json') for key,path in [('synthetic8',O),('real8',REAL8),('zero',ZERO)]}
 models={};arrays={}
 for name in NAMES:
  grades={key:load(path/name/'graded.local.json') for key,path in [('synthetic8',O),('real8',REAL8),('zero',ZERO)]}
  assert all([x['id'] for x in grades[k]]==[x['id'] for x in grades['synthetic8']] for k in grades)
  models[name]={k:allscores[k]['scores'][name]['overall'] for k in allscores}
  arrays[name]={}
  for ref in ['real8','zero']:
   a=grades['synthetic8'];b=grades[ref];gain=sum(x['correct'] and not y['correct'] for x,y in zip(a,b));loss=sum(y['correct'] and not x['correct'] for x,y in zip(a,b))
   arrays[name][ref]=[int(x['correct'])-int(y['correct']) for x,y in zip(a,b)]
   models[name]['synthetic8_minus_'+ref]=dict(gained=gain,lost=loss,delta_correct=gain-loss,delta_pp=100*(gain-loss)/72)
 q=load(O/'queries.local.json');groups=sorted({x['group'] for x in q});sizes=np.array([sum(x['group']==g for x in q) for g in groups]);draw=np.random.default_rng(2026111028).integers(len(groups),size=(5000,len(groups)));means={}
 for kind in ['real','old','current']:
  names=[f'{kind}_n68_s{s}' for s in [831,832,833]]
  means[kind]={k+'_percent':sum(models[n][k]['correct'] for n in names)/216*100 for k in allscores}
  for ref in ['real8','zero']:
   delta=np.mean([arrays[n][ref] for n in names],axis=0);sums=np.array([sum(delta[i] for i,x in enumerate(q) if x['group']==g) for g in groups]);ci=np.quantile(100*sums[draw].sum(axis=1)/sizes[draw].sum(axis=1),[.025,.975])
   means[kind]['synthetic8_minus_'+ref]=dict(delta_pp=float(100*delta.mean()),conditional_group95_ci=ci.tolist())
 save(O/'paired-results.json',dict(status='COMPLETE_SYNTHETIC8_PAIRED_ALL9',models=models,means=means,questions=72,numeric_groups=50,new_training=0,API=0,limits=load(O/'study-plan.json')['limits']))
 save(O/'synth8-terminal.json',dict(status='COMPLETE',epoch=time.time(),results=bind(O/'paired-results.json')));live(stage='COMPLETE');print(json.dumps(means),flush=True)

def controller():
 verify();save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time()));deadline=time.monotonic()+10800
 for action,name in [('evaluate',n) for n in NAMES]+[('score',None)]:
  check();label=action+('-'+name if name else '');log=O/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
  with log.open('x') as out:
   args=[str(R/'.venv/bin/python'),__file__,action]+([name] if name else []);child=subprocess.Popen(args,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if action=='evaluate' else '',HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);save(O/'processes'/f'{label}.json',dict(PID=child.pid,epoch=time.time()));start=time.monotonic()
   try:
    while child.poll() is None:
     check();assert time.monotonic()<deadline and time.monotonic()-start<2400;time.sleep(2)
    assert child.returncode==0,f'{label}:exit{child.returncode}'
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
if __name__=='__main__':
 def stop(*_):
  global STOP;STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);configure()
 try:
  if sys.argv[1]=='evaluate':c.evaluate(sys.argv[2])
  else:globals()[sys.argv[1]]()
 except BaseException as e:save(O/f'failure-{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise
