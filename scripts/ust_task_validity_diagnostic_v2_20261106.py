"""Frozen Dev diagnostics on four existing UST checkpoints; no training or API."""
from pathlib import Path
import argparse,ast,collections,copy,fcntl,hashlib,importlib.util,itertools,json,os,re,sqlite3,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
B=R/'runs/2026-11-06-ust-task-validity-diagnostic-v1'
O=B/'inference-v2'
PAIR=R/'scripts/ust_real_synthetic_paired_ce_20261105.py'
TRANS=R/'scripts/ust_dev_semantic_transforms_20261106.py'
DEV=R/'runs/2026-10-25-qwen-transfer-repair/data/USTBench.repair-dev32.requests.jsonl'
MODELS={
 'real-gold':R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1/real-reference-ce',
 'learned-synthetic':R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1/synthetic-reference-ce',
 'source-one-step':R/'runs/2026-11-05-ust-author-paper-reimplementation-v1/student-control/source-myopic-ce',
 'source-H5MC':R/'runs/2026-11-05-ust-author-paper-reimplementation-v1/student-control/source-h5mc-ce'}
VARIANTS=['original','canonical','letters','scene','both','canonical-cap128']
TARGETS=('enhancing overall station coverage','minimizing overall travel time','reducing overall waiting time','reducing overall charging time')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check():
 if (O/'STOP').exists():raise InterruptedError('Own task-validity diagnostic STOP')
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')
def derangement(values,key):
 pp=[p for p in itertools.permutations(values) if all(a!=b for a,b in zip(p,values))]
 return min(pp,key=lambda p:hashlib.sha256((key+repr(p)).encode()).hexdigest())
def numerical_cf(t,row,official):
 question=t.canonical(row['question']);oldanswer=t.understanding_answers(question)
 if oldanswer!=[official]:return None,dict(status='ORIGINAL_NOT_UNIQUE_OR_GOLD_MISMATCH',attempts=0)
 parts=t.sections(question);intro=t._options(parts['Question'])[0]
 metric='distance' if 'average distance' in intro else 'waiting_time' if 'average waiting time' in intro else 'charging_time'
 blocks=t._data_blocks(parts['Data']);pattern=r'^(- '+metric+r':\s*)([^\n]+)$'
 values={i:re.search(pattern,block,re.M).group(2) for i,block in blocks.items()}
 perms=sorted(itertools.permutations(range(1,5)),key=lambda p:hashlib.sha256(('UST-dev-counterfactual:'+row['id']+repr(p)).encode()).hexdigest())
 for attempt,permutation in enumerate(perms,1):
  new=dict(parts);newblocks={i:re.sub(pattern,lambda m:m.group(1)+values[permutation[i-1]],block,flags=re.M) for i,block in blocks.items()}
  new['Data']='\n\n'.join('location '+str(i)+':\n'+newblocks[i] for i in range(1,5));text=t.render_sections(new);answers=t.understanding_answers(text)
  if answers is None or len(answers)!=1 or answers==oldanswer:continue
  assert all(new[k]==parts[k] for k in parts if k!='Data')
  for i in blocks:assert re.sub(pattern,'',blocks[i],flags=re.M)==re.sub(pattern,'',newblocks[i],flags=re.M)
  assert sorted(values.values())==sorted(re.search(pattern,b,re.M).group(2) for b in newblocks.values())
  return dict(question=text,answer=answers[0]),dict(status='QUALIFIED',attempts=attempt,metric=metric,value_source_order=permutation,old_answer=oldanswer[0],new_answer=answers[0],only_target_field_values_changed=True,artificial_arithmetic_only=True,physical_realizability_claimed=False)
 return None,dict(status='NO_UNIQUE_CHANGED_ANSWER_IN_24_PERMUTATIONS',attempts=24)
def prepare():
 O.mkdir(exist_ok=True);assert not (O/'protocol.json').exists()
 t=module(TRANS,'ust_diag_transform');rows=[json.loads(l) for l in DEV.open()];assert len(rows)==32
 proof=load('/tmp/ust-dev-transform-qualification-20261106.json');assert proof['status']=='PASS' and proof['source_sha256']==bind(TRANS)['sha256'] and proof['Dev32_source_sha256']==bind(DEV)['sha256']
 save(O/'transform-qualification.json',proof)
 goldpath=DEV.with_name('USTBench.repair-dev32.gold.local.json');gold=load(goldpath)
 prefixpath=R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1/fixed8.local.json';save(O/'fixed8.local.json',load(prefixpath))
 queries=[];labels={};cf_details=[];contracts=[]
 identity={i:i for i in range(1,5)}
 for row in rows:
  lm=dict(zip(range(1,5),derangement(tuple(range(1,5)),'UST-dev-location:'+row['id'])));op=derangement(tuple('ABCD'),'UST-dev-options:'+row['id'])
  for variant in VARIANTS:
   loc=lm if variant in ['scene','both'] else identity;order=op if variant in ['letters','both'] else tuple('ABCD');tr=t.transform(row['question'],loc,order)
   text=row['question'] if variant=='original' else tr['question'];qid=row['id']+'::'+variant
   queries.append(dict(id=qid,source_id=row['id'],group=row['group'],metadata=row['metadata'],variant=variant,cap=128 if variant=='canonical-cap128' else 32,question=text,location_map=loc,old_to_new_letter=tr['old_to_new_letter']))
   labels[qid]=dict(answer=tr['old_to_new_letter'][gold[row['id']]],provenance='published RepairDev gold with mechanically preserved option identity')
  parts=t.sections(row['question']);fields=[set(re.findall(r'^- ([a-z_]+):',block,re.M)) for block in t._data_blocks(parts['Data']).values()]
  contracts.append(dict(id=row['id'],ability=row['metadata']['ability'],group=row['group'],sections=list(parts),data_fields=[sorted(s) for s in fields],schema_mentions_travel_time='travel_time:' in parts['Data Schema'],data_has_travel_time=all('travel_time' in s for s in fields),visible_budget_field=bool(re.search(r'\bbudget\b',parts['Data'],re.I)),reflection_single_selected_action_feedback='Environment Changes' in parts,understanding_independent_answers=t.understanding_answers(row['question'])))
  if row['metadata']['ability']=='st_understanding':
   cf,detail=numerical_cf(t,row,gold[row['id']]);cf_details.append(dict(id=row['id'],**detail))
   if cf is not None:
    qid=row['id']+'::numeric-counterfactual';queries.append(dict(id=qid,source_id=row['id'],group=row['group'],metadata=row['metadata'],variant='numeric-counterfactual',cap=32,question=cf['question'],location_map=identity,old_to_new_letter=dict(zip('ABCD','ABCD'))))
    labels[qid]=dict(answer=cf['answer'],provenance='independent Decimal solver on changed displayed target-metric values; arithmetic-only diagnostic')
 assert len(queries)<=204 and len(queries)==len(labels)==len({q['id'] for q in queries})
 save(O/'queries.local.json',queries);save(O/'answers.local.json',labels);save(O/'counterfactual-qualification.local.json',cf_details);save(O/'question-contracts.local.json',contracts)
 isolation=R/'runs/2026-11-06-dataenvgym-ust-transfer-v1/metadata-isolation.json';save(O/'prior-metadata-isolation.json',load(isolation))
 audit=B/'h5-exact-audit/attempt2/results.json';assert audit.exists(),'CPU oracle audit must finish before freezing this batch'
 pins=[bind(Path(__file__)),bind(TRANS),bind(PAIR),bind(DEV),bind(goldpath),bind(prefixpath),bind(isolation),bind(audit)]
 pins +=[bind(O/f) for f in ['queries.local.json','answers.local.json','fixed8.local.json','transform-qualification.json','counterfactual-qualification.local.json','question-contracts.local.json']]
 pins +=[bind(R/'scripts/three_method_common_eval_20261101.py'),bind(R/'runs/2026-11-05-ust-environment-generator-ipd-v1/full-schema-mainline-v3/gpu_student_worker.py')]
 for source in MODELS.values():pins.extend([bind(source/'protocol.json'),bind(source/'checkpoints/step0200/manifest.json')])
 save(O/'protocol.json',dict(source=bind(Path(__file__)),bindings=pins,status='FROZEN_BEFORE_GPU_INFERENCE',models={k:str(v.relative_to(R)) for k,v in MODELS.items()},queries_each=len(queries),total_raw=4*len(queries),variants=dict(collections.Counter(q['variant'] for q in queries)),training_updates=0,teacher_calls=0,new_weights=0,Final_Common_Audit_Confirm_read=False,role='existing RepairDev local exploratory diagnosis; no new held-out confirmation',decoder='unchanged greedy 16k decoder; CAP32 except explicit CAP128 same-canonical-prompt control',transform='qualified full crossproduct cases and inverse checks; canonical whitespace control; single-action feedback unchanged',counterfactuals='understanding-only arithmetic; no physical scene claim',H5_scope='exact miniature replay of own source-physics label rule; hidden budget interventions not original-episode reachability proof'))
 for name,source in MODELS.items():
  p=load(source/'protocol.json');p.update(source=bind(Path(__file__)),bindings=pins+[bind(O/'protocol.json')],source_model_unit=str(source.relative_to(R)),inference_only=True,optimizer_updates=0)
  save(O/name/'protocol.json',p)
 before=load(R/'runs/2026-11-06-dataenvgym-ust-balanced-repair-v1/budget-before.json');conn=sqlite3.connect('file:'+str(R/before['path'])+'?mode=ro',uri=True);assert hashlib.sha256('\n'.join(conn.iterdump()).encode()).hexdigest()==before['dump_sha256'];conn.close();save(O/'budget-before.json',before)
 print('FROZEN',len(queries),'questions each,',4*len(queries),'total; training0/API0',flush=True)
def verify(unit=None):
 p=load((Path(unit) if unit else O)/'protocol.json');assert p['source']==bind(Path(__file__))
 for pin in p['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 check();return p
def worker():
 m=module(PAIR,'ust_diag_legacy_pair');m.O=O;m.check=check;m.verify=verify;w,_=m.machinery();return m,w
def evaluate(name):
 unit=O/name;verify(unit);source=MODELS[name];folder=source/'checkpoints/step0200';m,w=worker()
 src=R/'scripts/three_method_common_eval_20261101.py';nodes=[n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['hf_greedy','context_failure']];ns=dict(time=time,CAP=32,CONTEXT=16384);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(src)+'::UNCHANGED_DIAGNOSTIC_DECODER','exec'),ns)
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(unit/'evaluation-started.json',dict(PID=os.getpid(),epoch=time.time()))
  torch,tok,model,head=w.initialize(unit,folder);w.m.lib.activate(model,head,'actor');model.eval()
  opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
  cm=w.m.lib.restore(torch,model,head,dict(actor=opt),source,folder);roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};assert roles==cm['role_hashes']==load(source/'fit-terminal.json')['roles'];opthash=w.m.lib.nested_hash(opt.state_dict())
  save(unit/'independent-reload.json',dict(status='EXACT_SOURCE_ROLES_OPTIMIZER_RNG_RELOAD',source_manifest=bind(folder/'manifest.json'),roles=roles,optimizer=opthash))
  rows=load(O/'queries.local.json');prefix=load(O/'fixed8.local.json');start=time.monotonic()
  for i,row in enumerate(rows):
   check();save(unit/'raw'/f'{i:04d}.started.json',dict(id=row['id'],epoch=time.time()))
   messages=prefix+[dict(role='user',content=row['question'])];ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
   ns['CAP']=row['cap'];value=ns['hf_greedy'](torch,tok,model,ids);save(unit/'raw'/f'{i:04d}.json',dict(id=row['id'],source_id=row['source_id'],variant=row['variant'],cap=row['cap'],**value))
   if (i+1)%32==0:live(stage='GPU_INFERENCE',model=name,closed=i+1,total=len(rows));print('DECODE',name,i+1,len(rows),round(time.monotonic()-start,1),flush=True)
  assert roles=={r:w.m.lib.role_hash(model,r) for r in roles};assert opthash==w.m.lib.nested_hash(opt.state_dict())
  save(unit/'terminal.json',dict(status='ALL_RAW_CLOSED_UNCHANGED_MODEL',raw=len(rows),seconds=time.monotonic()-start,roles_unchanged=True,optimizer_unchanged=True,training_updates=0,API=0))
def summarize(graded):
 out={}
 for variant in sorted({r['variant'] for r in graded}):
  rr=[r for r in graded if r['variant']==variant]
  out[variant]=dict(total=len(rr),correct=sum(r['correct'] for r in rr),valid=sum(r['valid'] for r in rr),capped=sum(r['capped'] for r in rr),abilities={a:dict(total=sum(r['ability']==a for r in rr),correct=sum(r['correct'] for r in rr if r['ability']==a),valid=sum(r['valid'] for r in rr if r['ability']==a),capped=sum(r['capped'] for r in rr if r['ability']==a)) for a in ['st_understanding','planning','reflection']})
 return out
def score():
 p=verify();assert all(load(O/n/'terminal.json')['raw']==p['queries_each'] for n in MODELS)
 save(O/'all-model-raw-frozen-before-score.json',dict(models=list(MODELS),total_raw=p['total_raw'],epoch=time.time()))
 sys.path[:0]=[str(R/'scripts'),str(R/'src')];import eval_ust_sg_qwen_modes_20261024 as q
 t=module(TRANS,'ust_diag_score_transform');rows=load(O/'queries.local.json');labels=load(O/'answers.local.json');scores={};pairs={};allgraded={};baselines={}
 for name in MODELS:
  graded=[];raws={}
  for i,row in enumerate(rows):
   raw=load(O/name/'raw'/f'{i:04d}.json');assert raw['id']==row['id'];parsed=q.classify(raw['raw_generated'],'off','ust',row['metadata'],raw['finish_reason']);raws[row['id']]=raw
   graded.append(dict(id=row['id'],source_id=row['source_id'],variant=row['variant'],ability=row['metadata']['ability'],prediction=parsed['prediction'],answer=labels[row['id']]['answer'],correct=bool(parsed['valid'] and parsed['prediction']==labels[row['id']]['answer']),valid=bool(parsed['valid']),capped=raw['finish_reason']=='length'))
  save(O/name/'graded.local.json',graded);scores[name]=summarize(graded);allgraded[name]=graded;lookup={r['id']:r for r in graded};base={r['source_id']:r for r in graded if r['variant']=='canonical'};pairs[name]={}
  for variant in ['original','letters','scene','both','canonical-cap128','numeric-counterfactual']:
   selected=[r for r in rows if r['variant']==variant];events=[]
   for row in selected:
    a=base[row['source_id']];b=lookup[row['id']];valid=a['valid'] and b['valid'];expected=row['old_to_new_letter'].get(a['prediction']) if a['valid'] else None
    event=dict(source_id=row['source_id'],ability=b['ability'],base_correct=a['correct'],variant_correct=b['correct'],both_valid=bool(valid),semantically_consistent=bool(valid and b['prediction']==expected) if variant!='numeric-counterfactual' else None,answer_changed=bool(valid and a['prediction']!=b['prediction']))
    if variant=='canonical-cap128':
     r32=raws[a['id']];r128=raws[b['id']];event['raw_prefix_equal']=r32['response_ids']==r128['response_ids'][:len(r32['response_ids'])];event['invalid_to_valid']=not a['valid'] and b['valid']
    events.append(event)
   save(O/name/(variant+'-paired.local.json'),events)
   pairs[name][variant]=dict(n=len(events),base_correct=sum(e['base_correct'] for e in events),variant_correct=sum(e['variant_correct'] for e in events),gained=sum(not e['base_correct'] and e['variant_correct'] for e in events),lost=sum(e['base_correct'] and not e['variant_correct'] for e in events),both_valid=sum(e['both_valid'] for e in events),semantically_consistent=sum(e['semantically_consistent'] is True for e in events),answer_changed=sum(e['answer_changed'] for e in events))
   if variant=='canonical-cap128':pairs[name][variant].update(raw_prefix_agree=sum(e['raw_prefix_equal'] for e in events),invalid_to_valid=sum(e['invalid_to_valid'] for e in events))
 # Two separately frozen shortcuts: target->fixedletter, target->fixedlocation.
 for variant in VARIANTS:
  rr=[r for r in rows if r['variant']==variant and r['metadata']['ability']!='st_understanding'];letter_correct=location_correct=recognized=0
  for row in rr:
   text=t.sections(row['question'])['Question'];goal=next((i for i,s in enumerate(TARGETS) if s in text),None)
   if goal is None:continue
   recognized+=1;expected=labels[row['id']]['answer'];letter_correct+=('ABCD'[goal]==expected);options=t._options(text)[1]
   candidates=[letter for letter,body in options.items() if re.fullmatch(r'location '+str(goal+1),body,re.I)]
   assert len(candidates)==1;location_correct+=(candidates[0]==expected)
  baselines[variant]=dict(nonunderstanding_total=len(rr),recognized=recognized,goal_to_fixed_letter_correct=letter_correct,goal_to_fixed_location_correct=location_correct)
 budget=load(O/'budget-before.json');conn=sqlite3.connect('file:'+str(R/budget['path'])+'?mode=ro',uri=True);assert hashlib.sha256('\n'.join(conn.iterdump()).encode()).hexdigest()==budget['dump_sha256'];conn.close()
 save(O/'results.json',dict(status='COMPLETE',scores=scores,paired=pairs,goal_only_baselines=baselines,queries_each=p['queries_each'],total_raw=p['total_raw'],external_API=0,training_updates=0,budget_SQL_unchanged=True,all_original_model_roles_unchanged=True,Final_Common_Audit_Confirm_read=False,scope='RepairDev exploratory diagnostics on four frozen models, not independent final performance; arithmetic counterfactuals not physical scenes',H5_exact_audit=bind(B/'h5-exact-audit/attempt2/results.json')))
 report=['# UST题目有效性与位置规律诊断','','四个模型均复用已冻结的200步检查点；本轮训练更新0、外部API调用0。使用RepairDev32开发题，未读取Common100/Audit32/Final/Confirm。下表为答对题数，含无效／截断输出在分母内。','','| 模型 | 原文32 | 规范化32 | 仅选项32 | 位置及展示顺序32 | 联合重排32 | 同题128token32 | 数值反事实 |','| --- | --- | --- | --- | --- | --- | --- | --- |']
 for name in MODELS:
  report.append('| '+name+' | '+' | '.join(str(scores[name][v]['correct'])+'/'+str(scores[name][v]['total']) for v in VARIANTS+['numeric-counterfactual'] if v in scores[name])+' |')
 report+=['','## 目标类型捷径检查','',json.dumps(baselines,ensure_ascii=False),'','## 解释边界','','只重排选项仍可能保留目标到位置编号的规律；联合重排同时改变位置编号和选项字母，并同步Data展示顺序、Analysis、Decision、Reason及选项引用。原文与规范化对照只改变空白。','同题32/128输出上限单列，逐条检查生成token前缀一致性；这是截断诊断，不替换旧结果。','数值反事实仅改理解题的目标字段数值分配，题目目标和原选项不变，独立Decimal规则得到唯一新答案；不声称城市物理可实现。','所有结果都是开发诊断，不用于声称独立测试性能或完整原论文方法失败。标签不充分、模型缺少相关训练以及位置规律依赖仍需分别解释。','H5精确小场景审计见h5-exact-audit/attempt2/results.json；隐藏预算干预不代表原官方轨迹可达，也不是官方gold错误证据。']
 (O/'delivery-report.md').write_text('\n'.join(report)+'\n');save(O/'terminal.json',dict(status='COMPLETE',results=bind(O/'results.json'),report=bind(O/'delivery-report.md'),training_updates=0,external_API=0,epoch=time.time()));live(stage='COMPLETE')
 print('DIAGNOSTIC COMPLETE',json.dumps(scores),flush=True)
def controller():
 verify();save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time()))
 for name in MODELS:
  live(stage='LOADING_FROZEN_MODEL',model=name)
  with (O/('decode-'+name+'.log')).open('x') as f:subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)),'evaluate','--model',name],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 score()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate','score','controller']);p.add_argument('--model',choices=MODELS);a=p.parse_args()
 try:
  if a.action=='evaluate':evaluate(a.model)
  else:globals()[a.action]()
 except BaseException as ex:
  if O.exists():save(O/(a.action+'-failure-'+str(time.time_ns())+'.json'),dict(type=type(ex).__name__,message=str(ex)[:1200],epoch=time.time(),no_automatic_replay=True))
  raise
