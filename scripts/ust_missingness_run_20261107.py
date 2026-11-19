"""Bounded cloud pipeline: missingness factorial, qualification, three orders."""
from pathlib import Path
import argparse,collections,copy,hashlib,json,math,os,random,signal,subprocess,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_missingness_data_20261107 as data
import ust_joint_generator_student_run_20261107 as base
INDEPENDENT_SOURCE=R/'scripts/ust_joint_generator_student_closed_audit_20261107.py'
INDEPENDENT_ORIGINAL=R/'runs/2026-11-06-ust-retention-information-v1/independent-audit/real-own-runtime-and-closed-audit-v2.py'
S=data.S;PREVIOUS=data.P;O=None;PHASE=None;STOP=False
load=base.load;save=base.save;bind=base.bind

def check():
 if STOP or (S/'STOP').exists() or (O is not None and (O/'STOP').exists()):raise InterruptedError('Own missingness pipeline STOP')
def stop(*_):
 global STOP;STOP=True

def live(**kw):
 x=dict(PID=os.getpid(),epoch=time.time(),phase=PHASE,**kw);p=S/'live.tmp';p.write_text(json.dumps(x));p.replace(S/'live.json')

def specs(phase):
 p=load(S/'study-plan.json');out={}
 if phase=='diagnostic':
  for name in ('coverage0_native','coverage8_native','coverage0_rule','coverage8_rule'):out[name]=dict(dataset=name,order_seed=p['diagnostic']['order_seed'],step=136)
 elif phase=='fallback':out['coverage16_rule']=dict(dataset='coverage16_rule',order_seed=p['diagnostic']['order_seed'],step=136)
 else:
  decision=load(S/'recipe-decision.json');assert decision['qualified'];chosen=decision['selected'];representation='rule' if chosen.endswith('_rule') else 'native'
  for seed in p['comparison']['order_seeds']:
   out[f'real_order{seed}']=dict(dataset='real_'+representation,order_seed=seed,step=136)
   out[f'synthetic_order{seed}']=dict(dataset=chosen,order_seed=seed,step=136)
 if phase!='fallback':out={'freshINIT':dict(dataset=None,order_seed=None,step=0),**out}
 return out

def verify(unit=None):
 check();receipt=load(O/'execution-bindings.json');assert receipt['source']==bind(__file__)
 for pin in receipt['protocols']:assert bind(R/pin['path'])==pin,pin['path']
 if unit is not None:assert Path(unit).parent==O and Path(unit).name in specs(PHASE)
 p=load((Path(unit) if unit else O)/'protocol.json');assert p['source']==bind(__file__)
 for pin in p['bindings']:assert bind(R/pin['path'])==pin,pin['path']
 return p

def gate(**_):
 verify();a=load(S/'launch-authorization.json');assert a['source']==bind(__file__) and a['data_review']==bind(S/'data-review.json') and a['study_plan']==bind(S/'study-plan.json') and a['status']=='AUTHORIZED_USER_FIXED_GENERATOR_STUDY'
 return a

def frozen_models():
 v=load(O/'models-frozen.json');assert v['status']=='ALL_FIXED_ENDPOINTS_FROZEN' and set(v['specs'])==set(specs(PHASE))
 for pin in v['bindings']:assert bind(R/pin['path'])==pin
 return v['specs']

def configure(phase):
 global O,PHASE;PHASE=phase;O=S/phase;sp=specs(phase)
 base.O=O;base.ARMS=tuple(k for k,v in sp.items() if v['step']);base.UNITS=tuple(sp);base.TOTAL=sum(634 if v['step'] else 256 for v in sp.values());base.verify=verify;base.gate=gate;base.check=check;base.live=live;base.frozen_models=frozen_models
 base.core.O=O;base.core.ARMS=base.ARMS;base.core.TOTAL=base.TOTAL;base.core.verify=verify;base.core.gate=gate;base.core.check=check;base.core.live=live


def prepare(phase):
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';configure(phase)
 if (O/'protocol.json').exists():verify();print('ALREADY_PREPARED',phase);return
 assert load(S/'data-review.json')['status']=='PASS_ALL_FROZEN_DATA'
 parent=load(PREVIOUS/'protocol.json')
 for pin in parent['bindings']:assert bind(R/pin['path'])==pin
 p0=load(PREVIOUS/'real68/protocol.json');prefix=load(S/'data/fixed8.local.json');h=base.core.helper();_,init_pins=h.checkpoint_pins(base.core.OLD,base.core.INIT,0)
 from transformers import AutoTokenizer
 import probe_overnight_train_fit_20261025 as packing
 tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);family,_=h.target_family(tok)
 cohort='Confirm' if phase=='comparison' else 'Select';diag=load(S/'data'/f'{cohort}-queries.local.json');answers=load(S/'data'/f'{cohort}-answers.local.json');dev=load(S/'data/Dev310-queries.local.json');answers.update(load(S/'data/Dev310-answers.local.json'))
 dev=[dict(q,cohort='Dev',family='benchmark',spelling='native',view=q['variant']) for q in dev]
 assert len(diag)==256 and len(dev)==310;sp=specs(phase);allvariants=[];tokens={};counts={};schedules={};queries={};shared_schedule={}
 for key,spec in sp.items():
  if spec['step']:
   vv=copy.deepcopy(load(S/'data'/spec['dataset']/'variants.local.json'))
   for v in vv:v['id']=f"{key}/{v['source_slot']:03d}/{v['variant_index']}"
   allvariants+=vv;own=[]
   for v in vv:
    cert=data.certify(v['question'],v['answer'],True);prompt,target=packing.packing(tok,v,dict(text=v['target']),'matched8',prefix)
    assert prompt==h.eval_prompt(tok,prefix,v['question']) and target==family[v['answer']] and len(prompt)+128<=16384
    tokens[v['id']]=dict(prompt_ids=prompt,target_ids=target)
    if v['variant_index']==0:
     own.append(dict(id=v['id'],source_id=v['source_id'],group=v['source_id'],question=v['question'],metadata=v['metadata'],metric=v['metric'],variant='own',cohort='Own',family='own',spelling='native',view='identity',flags={},old_to_new_letter={a:a for a in 'ABCD'}));answers[v['id']]=dict(answer=v['answer'],compatible=[v['answer']])
   slots=list(range(272));random.Random(spec['order_seed']).shuffle(slots);ordered=[vv[i]['id'] for i in slots];structure=[(vv[i]['source_slot'],vv[i]['variant_index']) for i in slots]
   if spec['order_seed'] in shared_schedule:assert structure==shared_schedule[spec['order_seed']]
   else:shared_schedule[spec['order_seed']]=structure
   schedule=[dict(step=e*68+b+1,epoch=e+1,batch_index=b,variant_ids=ordered[b*4:b*4+4]) for e in range(2) for b in range(68)]
   assert len(schedule)==136 and collections.Counter(v for x in schedule for v in x['variant_ids'])==collections.Counter({v['id']:2 for v in vv})
   save(O/key/'schedule.local.json',schedule);save(O/key/'variants.local.json',vv);schedules[key]=schedule;queries[key]=dev+diag+own
  else:queries[key]=diag
  for q in queries[key]:
   ids=h.eval_prompt(tok,prefix,q['question']);assert len(ids)+128<=16384;record=dict(prompt_ids=ids,target_ids=family[answers[q['id']]['answer']])
   if q['id'] in tokens:assert tokens[q['id']]==record
   tokens[q['id']]=record
  counts[key]=len(queries[key]);assert counts[key]==(634 if spec['step'] else 256);save(O/key/'queries.local.json',queries[key])
 if phase=='comparison':assert len({str(x) for x in shared_schedule.values()})==3
 save(O/'variants.local.json',allvariants);save(O/'tokens.local.json',tokens);save(O/'answers.local.json',answers);save(O/'fixed8.local.json',prefix)
 pins=parent['bindings']+init_pins+[bind(__file__),bind(data.__file__),bind(INDEPENDENT_SOURCE),bind(INDEPENDENT_ORIGINAL),bind(S/'study-plan.json'),bind(S/'data-summary.json'),bind(S/'data-review.json'),bind(PREVIOUS/'protocol.json'),bind(PREVIOUS/'freshINIT/graded.local.json'),bind(PREVIOUS/'freshINIT/queries.local.json')]
 pins+=[bind(p) for p in (S/'data').rglob('*.json')]+[bind(p) for p in O.rglob('*.json')]
 if phase=='comparison':pins.append(bind(S/'recipe-decision.json'))
 pins=list({x['path']:x for x in pins}.values());save(O/'protocol.json',dict(source=bind(__file__),bindings=pins,phase=phase,units=sp,total_raw=sum(counts.values()),counts=counts,diagnostic_cohort=cohort,original_Dev_unchanged=True,own_train_is_resubstitution=True,generator_fixed=True,API=0,Final_access=False,choice_uses_Select_not_Dev_or_Confirm=True))
 pp=[bind(O/'protocol.json')]
 for key,spec in sp.items():
  p={k:p0[k] for k in ('base','model','seed','lr','lora','loss','optimizer_restored')};p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],method='fixed-generator-missingness-batch4-understanding' if spec['step'] else 'fresh-INIT-diagnostic',steps=spec['step'],queries_each=counts[key],total_raw=sum(counts.values()),arm=key,dataset=spec['dataset'],order_seed=spec['order_seed'],fixed_checkpoint=spec['step'],ability_order=['st_understanding']*spec['step'],checkpoint_steps=[34,68,102,136] if spec['step'] else [],protected_real_training_gold_in_loss=bool(spec['dataset'] and spec['dataset'].startswith('real_')),synthetic_reference_in_loss=bool(spec['dataset'] and not spec['dataset'].startswith('real_')),uncertain_planning_reflection_references=False,training_label_status='Only uniquely certified visible weak-order understanding targets',teacher_API=0,Final_access=False)
  save(O/key/'protocol.json',p);pp.append(bind(O/key/'protocol.json'))
 save(O/'execution-bindings.json',dict(source=bind(__file__),protocols=pp));verify()
 import torch
 assert not torch.cuda.is_initialized();save(O/'preflight.json',dict(status='PASS_CPU_ONLY',phase=phase,total_raw=sum(counts.values()),fits=len(schedules),updates=len(schedules)*136,max_prompt_tokens=max(len(v['prompt_ids']) for v in tokens.values()),target_tokens=7,CUDA_initialized=False));print('PREPARED',phase,sum(counts.values()),flush=True)


def freeze_models():
 gate();sp=specs(PHASE);models={};pins=[];h=base.core.helper()
 for key,spec in sp.items():
  source=O/key if spec['step'] else base.core.OLD;folder=source/'checkpoints/step0136' if spec['step'] else base.core.INIT
  if spec['step']:
   t=load(source/'fit-terminal.json');assert t['actual_updates']==136 and t['microexamples']==544 and t['status']=='FIXED136_COMPLETE';pins.extend([bind(source/'fit-terminal.json'),bind(source/'initialization-verified.json')])
  cm,cp=h.checkpoint_pins(source,folder,spec['step']);pins+=cp;models[key]=dict(source=str(source.relative_to(R)),folder=str(folder.relative_to(R)),step=spec['step'],roles=cm['role_hashes'],optimizer=cm['optimizer_hashes']['actor'])
 save(O/'models-frozen.json',dict(status='ALL_FIXED_ENDPOINTS_FROZEN',specs=models,bindings=list({x['path']:x for x in pins}.values()),epoch=time.time()))

def stats(rows):
 return dict(n=len(rows),source_worlds=len({r['source_id'] for r in rows}),numeric_groups=len({r['group'] for r in rows}),correct=sum(x['correct'] for x in rows),compatible=sum(x['compatible_correct'] for x in rows),valid=sum(x['valid'] for x in rows),capped=sum(x['capped'] for x in rows))

def score():
 verify();sp=specs(PHASE);p=load(O/'protocol.json');models=frozen_models();pins=[]
 for key in sp:
  t=load(O/key/'evaluation-terminal.json');assert t['status']=='ALL_FIXED_RAW_CLOSED' and t['raw']==p['counts'][key] and t['roles']==models[key]['roles'] and t['roles_unchanged'] and t['optimizer_unchanged'] and t['RNG_unchanged'];files=list((O/key/'raw').glob('[0-9][0-9][0-9][0-9].json'));assert len(files)==p['counts'][key];pins += [bind(O/key/'raw'/f'{i:04d}.json') for i in range(len(files))]
 assert len(pins)==p['total_raw'];save(O/'all-raw-frozen-before-score.json',dict(epoch=time.time(),raw=pins))
 import eval_ust_sg_qwen_modes_20261024 as parser
 import ust_joint_generator_student_closed_audit_20261107 as independent
 pred=independent.independent();aa=load(O/'answers.local.json');tokens=load(O/'tokens.local.json');summaries={};allgrades={};request_seconds=0
 for key in sp:
  rows=[]
  for i,q in enumerate(load(O/key/'queries.local.json')):
   raw=load(O/key/'raw'/f'{i:04d}.json');assert raw['id']==q['id'] and raw['source_id']==q['source_id'] and raw['prompt_tokens']==len(tokens[q['id']]['prompt_ids']) and raw['cap']==128
   v=pred(raw['raw_generated'],raw['finish_reason']);g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);assert v==g['prediction'] and (v in ('A','B','C','D'))==bool(g['valid']);a=aa[q['id']]
   rows.append(dict(id=q['id'],source_id=q['source_id'],group=q['group'],cohort=q['cohort'],family=q['family'],spelling=q['spelling'],view=q['view'],variant=q['variant'],metric=q['metric'],flags=q['flags'],answer=a['answer'],compatible_answers=a['compatible'],prediction=v,valid=bool(g['valid']),correct=bool(g['valid'] and v==a['answer']),compatible_correct=bool(g['valid'] and v in a['compatible']),capped=raw['finish_reason']=='length'));request_seconds+=raw['request_wall_seconds']
  save(O/key/'graded.local.json',rows);allgrades[key]=rows;cohort=p['diagnostic_cohort'];dg=[x for x in rows if x['cohort']==cohort];dv=[x for x in rows if x['cohort']=='Dev'];own=[x for x in rows if x['cohort']=='Own']
  summaries[key]=dict(diagnostic=dict(all=stats(dg),missing=stats([x for x in dg if x['family']=='missing']),native_bare=stats([x for x in dg if x['family']=='missing' and x['spelling']=='bare']),finite=stats([x for x in dg if x['family']!='missing']),by_family={f:stats([x for x in dg if x['family']==f]) for f in ('missing','finite_twin','ordinary')},by_spelling={s:stats([x for x in dg if x['family']=='missing' and x['spelling']==s]) for s in ('bare','unit','spaced')},by_view={v:stats([x for x in dg if x['view']==v]) for v in ('original','joint')}),Dev={v:dict(overall=stats([x for x in dv if x['view']==v]),by_metric={m:stats([x for x in dv if x['view']==v and x['metric']==m]) for m in ('distance','waiting_time','charging_time')},subgroups={f:stats([x for x in dv if x['view']==v and x['flags'].get(f)]) for f in ('any_tail','time_NaN','statistical_outlier_candidate')}) for v in ('original','joint')},own=stats(own))
 # Independent saved-state/schedule audit on CPU after all model output closure.
 import torch
 assert not torch.cuda.is_initialized();fits={};original=load(base.core.INIT/'manifest.json');h=base.core.helper()
 for key,spec in sp.items():
  if not spec['step']:continue
  unit=O/key;start=load(unit/'initialization-verified.json');end=load(unit/'fit-terminal.json');assert start['roles']==original['role_hashes'] and start['optimizer']==original['optimizer_hashes']['actor'] and all(end['roles'][k]==original['role_hashes'][k] for k in ('reference','discriminator'))
  for row in load(unit/'schedule.local.json'):
   up=load(unit/'updates'/f"{row['step']:04d}.json");assert up['variant_ids']==row['variant_ids'] and up['state_steps_before']==[row['step']-1]*48 and up['state_steps_after']==[row['step']]*48 and up['optimizer_calls']==1 and up['micro_forward_backward_calls']==4 and up['RNG']==start['RNG'] and math.isfinite(up['loss'])
  for step in (34,68,102,136):
   folder=unit/'checkpoints'/f'step{step:04d}';h.checkpoint_pins(unit,folder,step);opt=torch.load(folder/'optimizers.pt',map_location='cpu',weights_only=True)['actor'];assert len(opt['state'])==48 and all(int(x['step'].item())==step for x in opt['state'].values())
  fits[key]=dict(updates=136,microexamples=544,fit_seconds=end['seconds'])
 for pin in pins:assert bind(R/pin['path'])==pin
 extra={}
 if PHASE=='comparison':
  pairs={}
  for seed in load(S/'study-plan.json')['comparison']['order_seeds']:
   a=allgrades[f'real_order{seed}'];b=allgrades[f'synthetic_order{seed}'];pairs[str(seed)]={}
   for cohort in ('Dev','Confirm'):
    for view in ('original','joint'):
     aa0={x['id']:x for x in a if x['cohort']==cohort and x['view']==view};bb={x['id']:x for x in b if x['cohort']==cohort and x['view']==view};assert set(aa0)==set(bb)
     pairs[str(seed)][cohort+'_'+view]=dict(n=len(aa0),real_correct=sum(x['correct'] for x in aa0.values()),synthetic_correct=sum(x['correct'] for x in bb.values()),real_only=sum(x['correct'] and not bb[k]['correct'] for k,x in aa0.items()),synthetic_only=sum(not x['correct'] and bb[k]['correct'] for k,x in aa0.items()))
  extra['paired_orders']=pairs
 save(O/'results.json',dict(status='COMPLETE_ALL_FIXED_OUTPUTS_INDEPENDENT_PARSE_AND_STATE_AUDIT',phase=PHASE,summaries=summaries,fits=fits,new_raw=p['total_raw'],request_wall_seconds=request_seconds,API=0,Final_access=False,training_order_not_initialization_seed=True,all_models_reported=True,**extra));save(O/'terminal.json',dict(status='COMPLETE',epoch=time.time(),results=bind(O/'results.json')));live(stage='PHASE_COMPLETE',raw=p['total_raw']);print('COMPLETE',PHASE,flush=True)


def qualifies(summary,control):
 c=load(S/'study-plan.json')['choice'];d=summary['diagnostic'];b=control['diagnostic'];tests=dict(missing=d['missing']['correct']>=c['minimum_missing_correct'],bare=d['native_bare']['correct']>=c['minimum_native_bare_correct'],finite=d['finite']['correct']>=c['minimum_finite_correct'],finite_loss=b['finite']['correct']-d['finite']['correct']<=c['maximum_finite_loss_vs_same_representation_control'])
 assert d['missing']['n']==144 and d['native_bare']['n']==48 and d['finite']['n']==112 and b['finite']['n']==112
 return dict(qualified=all(tests.values()),tests=tests,missing_correct=d['missing']['correct'],bare_correct=d['native_bare']['correct'],finite_correct=d['finite']['correct'],control_finite_correct=b['finite']['correct'])

def choose(after_fallback=False):
 diag=load(S/'diagnostic/results.json');ss=diag['summaries'];decisions={};chosen=None
 for key in load(S/'study-plan.json')['choice']['candidates_in_order']:
  control='coverage0_'+key.rsplit('_',1)[1];decisions[key]=qualifies(ss[key],ss[control])
  if chosen is None and decisions[key]['qualified']:chosen=key
 if after_fallback:
  assert chosen is None;fb=load(S/'fallback/results.json');decisions['coverage16_rule']=qualifies(fb['summaries']['coverage16_rule'],ss['coverage0_rule'])
  if decisions['coverage16_rule']['qualified']:chosen='coverage16_rule'
 receipt=dict(epoch=time.time(),selected=chosen,qualified=chosen is not None,decisions=decisions,criterion_source=bind(S/'study-plan.json'),selection_uses_Select_only=True,Dev_scores_not_used=True,Confirm_scores_not_available=True,diagnostic_results=bind(S/'diagnostic/results.json'))
 if after_fallback:receipt['fallback_results']=bind(S/'fallback/results.json')
 save(S/('recipe-decision.json' if chosen is not None or after_fallback else 'primary-repair-not-qualified.json'),receipt);return chosen


def pipeline():
 global PHASE,O
 a=load(S/'launch-authorization.json');assert a['source']==bind(__file__) and a['data_review']==bind(S/'data-review.json') and a['study_plan']==bind(S/'study-plan.json');plan=load(S/'study-plan.json');deadline=time.monotonic()+plan['runtime']['maximum_seconds'];save(S/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time(),maximum_seconds=plan['runtime']['maximum_seconds'],independent_session=os.getsid(0)==os.getpid()));os.environ.update(base.core.numerical_decision()['required_launcher_environment']);os.environ['HF_HUB_OFFLINE']='1'
 def call(action,phase,key=None):
  check();remaining=deadline-time.monotonic();assert remaining>0
  folder=S/phase;folder.mkdir(exist_ok=True);label=action+('-'+key if key else '');log=folder/(label+'.log');receipt=folder/(label+'-process.json');limit=plan['runtime']['per_fit_seconds'] if action=='fit' else plan['runtime']['per_eval_seconds'] if action=='evaluate' else 600
  args=[str(R/'.venv/bin/python'),str(Path(__file__).resolve()),action,'--phase',phase]+(['--unit',key] if key else []);env=dict(os.environ)
  if action not in ('fit','evaluate'):env['CUDA_VISIBLE_DEVICES']=''
  with log.open('x') as f:
   child=subprocess.Popen(args,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);save(receipt,dict(PID=child.pid,epoch=time.time(),action=action,phase=phase,unit=key,maximum_seconds=min(limit,remaining)));started=time.monotonic()
   try:
    while child.poll() is None:
     check()
     if time.monotonic()-started>min(limit,remaining):raise TimeoutError(label+' bound')
     time.sleep(2)
    if child.returncode:raise RuntimeError(f'{phase}/{label} exited {child.returncode}')
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 def phase_run(phase):
  configure(phase);live(stage='STARTING_PHASE');folder=S/phase
  if not (folder/'protocol.json').exists():call('prepare',phase)
  verify();sp=specs(phase)
  for key,spec in sp.items():
   if spec['step']:call('fit',phase,key)
  call('freeze',phase)
  for key in sp:call('evaluate',phase,key)
  call('score',phase)
 phase_run('diagnostic');chosen=choose()
 if chosen is None:phase_run('fallback');chosen=choose(True)
 if chosen is None:
  save(S/'terminal.json',dict(status='REPAIR_NOT_QUALIFIED_COMPARISON_NOT_STARTED',epoch=time.time(),decision=bind(S/'recipe-decision.json'),API=0));live(stage='REPAIR_NOT_QUALIFIED');return
 phase_run('comparison');save(S/'terminal.json',dict(status='COMPLETE_REPAIR_AND_THREE_PAIRED_ORDERS',epoch=time.time(),decision=bind(S/'recipe-decision.json'),comparison=bind(S/'comparison/results.json'),API=0));live(stage='COMPLETE');print('PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','freeze','evaluate','score','pipeline','verify']);p.add_argument('--phase',choices=['diagnostic','fallback','comparison']);p.add_argument('--unit');a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if a.action=='pipeline':pipeline()
  else:
   configure(a.phase)
   if a.action=='prepare':prepare(a.phase)
   elif a.action=='fit':base.core.fit(a.unit)
   elif a.action=='freeze':freeze_models()
   elif a.action=='evaluate':base.evaluate(a.unit)
   else:globals()[a.action]()
 except BaseException as e:
  save(S/('failure-'+str(time.time_ns())+'.json'),dict(type=type(e).__name__,error=str(e),epoch=time.time(),phase=PHASE,PID=os.getpid(),no_unknown_partial_replay=True));raise
