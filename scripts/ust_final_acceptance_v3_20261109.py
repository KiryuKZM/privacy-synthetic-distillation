"""Once-only Final understanding acceptance, gated on completed study and frozen models."""
from pathlib import Path
import collections,hashlib,json,os,signal,subprocess,sys,time,traceback
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_scale_20261108 as s
import eval_ust_sg_qwen_modes_20261024 as parser
P=s.O;O=R/'runs/2026-11-09-ust-final-acceptance-v3';load=s.load;save=s.save;bind=s.bind
WAIT=R/'runs/2026-11-09-ust-efficient-curves-v1/curves'
NAMES=['freshINIT']+[f'{k}_n272_s{seed}' for seed in s.SEEDS for k in s.KINDS]
STOP=False

def check():
 if STOP or (O/'STOP').exists():raise InterruptedError('Own Final STOP')
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(epoch=time.time(),PID=os.getpid(),**kw)));p.replace(O/'live.json')
def plancheck():
 check();p=load(O/'plan.json');assert p['source']==bind(__file__) and p['authorized_once_after_main_complete']
 assert p['models']==NAMES
 for b in p['bindings']:assert bind(R/b['path'])==b
 return p
def verify(unit=None):
 plancheck();p=load(O/'protocol.json')
 for b in p['bindings']:assert bind(R/b['path'])==b
 return load(Path(unit)/'protocol.json') if unit else p
def frozen_models():
 x=load(O/'models-frozen-before-Final.json')
 for b in x['bindings']:assert bind(R/b['path'])==b
 return x['models']
def configure():
 s.core.O=O;s.core.verify=verify;s.core.check=check;s.core.live=live;s.core.gate=lambda **kw:verify()
 s.base.O=O;s.base.verify=verify;s.base.check=check;s.base.live=live;s.base.gate=lambda **kw:verify();s.base.frozen_models=frozen_models

def freeze():
 plancheck();assert load(WAIT/'terminal.json')['status']=='COMPLETE';spec=s.load(P/'specs.json');h=s.helper();models={};pins=[]
 for name in NAMES:
  step=spec[name]['step'];source=P/name if step else s.OLD;folder=source/'checkpoints'/f'step{step:04d}' if step else s.INIT
  if step:
   end=load(source/'fit-terminal.json');assert end['actual_updates']==544;assert load(source/'evaluation-terminal.json')['raw']==310
  cm,cp=h.checkpoint_pins(source,folder,step);pins+=cp;models[name]=dict(source=str(source.relative_to(R)),folder=str(folder.relative_to(R)),step=step,roles=cm['role_hashes'],optimizer=cm['optimizer_hashes']['actor'])
 save(O/'models-frozen-before-Final.json',dict(epoch=time.time(),models=models,bindings=pins,main_completion=bind(WAIT/'terminal.json'),source=bind(__file__),Final_contents_read=False));configure()

def prepare():
 plancheck();assert load(WAIT/'terminal.json')['status']=='COMPLETE';models=frozen_models();assert set(models)==set(NAMES)
 # This is the first Final read. No code above opens Final membership or raw inputs.
 save(O/'Final-access-started.json',dict(epoch=time.time(),frozen=bind(O/'models-frozen-before-Final.json'),authorization=bind(O/'plan.json')))
 roles=load(s.d.old.B/'roles.json')['roles'];wanted=set(roles['FinalSealed']['members']);assert len(wanted)==3670
 records=[];parents=[];seen={};sourcepins=[]
 def root(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 import re
 for ability in ('st_understanding','planning','reflection'):
  path=s.d.old.B.parent/'sources'/f'ust-data-poi_placement__{ability}_QA.json';sourcepins.append(bind(path))
  for index,row in enumerate(load(path)):
   q=row['question'];ident=f'poi_placement:{ability}:{index}';i=len(records);parents.append(i)
   data=q.split('## Data\n\n',1)[1].split('\n\n## ',1)[0];vectors=[b.strip() for b in re.split(r'location \d+:\n',data)[1:]];assert len(vectors)==4
   records.append(dict(id=ident,ability=ability,question=q,gold=str(row['answer']),vectors=vectors))
   for v in vectors:
    if v in seen:parents[root(i)]=root(seen[v])
    else:seen[v]=i
 groups=collections.defaultdict(list)
 for i,r in enumerate(records):groups[root(i)].append(r['id'])
 groupid={k:'g:'+min(v) for k,v in groups.items()};final=[dict(r,group=groupid[root(i)]) for i,r in enumerate(records) if r['id'] in wanted];assert len(final)==3670
 assert {r['group'] for r in final}==set(roles['FinalSealed']['groups'])
 rows=sorted([r for r in final if r['ability']=='st_understanding'],key=lambda r:r['id']);assert rows
 prefix=load(P/'fixed8.local.json');p0=load(P/'freshINIT/protocol.json');from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(p0['base'],local_files_only=True,trust_remote_code=False);h=s.helper();family,_=h.target_family(tok)
 thresholds=load(s.d.old.D/'tail-outlier-utility-v2/thresholds.local.json');qs=[];answers={};proof=[];tokens={}
 for r in rows:
  q=r['question'];gold=r['gold'];assert gold in 'ABCD' and len(gold)==1
  # Keep original primary labels and every understanding row; qualify semantics separately.
  cert=s.d.old.certify_native_ranking_reference(q,gold)
  obs,opts=s.audit.sections(q);metric=cert['metric'];values=[obs[i][metric] for i in range(1,5)];finite=[v for v in values if v is not None];lo,hi=thresholds['positive_field_q01_q99'][s.d.old.FIELDS.index(metric)]
  flags=dict(target_tail=any(v is not None and v>0 and (float(v)<lo or float(v)>hi) for v in values),target_NaN=any(v is None for v in values),target_tie=len(set(finite))<len(finite),reference_trusted=bool(cert['trusted_gold']))
  loc,order=s.d.old.cm.derangements('final-once20261109/'+r['id']);tr=s.d.old.transform.transform(q,loc,order)
  inv=s.d.old.transform.transform(tr['question'],{v:k for k,v in loc.items()},[tr['old_to_new_letter'][a] for a in 'ABCD']);assert inv['question']==s.d.old.transform.canonical(q)
  proof.append(dict(id=r['id'],certificate=cert,inverse_exact=True))
  for variant,question,mapping in [('original',q,{a:a for a in 'ABCD'}),('joint',tr['question'],tr['old_to_new_letter'])]:
   ans=mapping[gold];qid=r['id']+'::Final-'+variant;ids=h.eval_prompt(tok,prefix,question);assert len(ids)+128<=16384
   qs.append(dict(id=qid,source_id=r['id'],group=r['group'],question=question,variant=variant,metric=metric,flags=flags,metadata=dict(ability='st_understanding')));answers[qid]=dict(answer=ans,compatible=[mapping[a] for a in cert['compatible_answers']]);tokens[qid]=dict(prompt_ids=ids,target_ids=family[ans])
 save(O/'fixed8.local.json',prefix);save(O/'tokens.local.json',tokens);save(O/'answers.local.json',answers);save(O/'qualification.local.json',proof)
 save(O/'scope.json',dict(Final_total=3670,evaluated_understanding=len(rows),numeric_groups=len({r['group'] for r in rows}),other_abilities_not_model_evaluated=3670-len(rows),all_Final_inputs_mechanically_scanned_for_original_group_reconstruction=True,ability_counts=dict(collections.Counter(r['ability'] for r in final)),metric_counts=dict(collections.Counter(q['metric'] for q in qs if q['variant']=='original')),subgroups={k:sum(q['flags'][k] for q in qs if q['variant']=='original') for k in flags},models=len(NAMES),total_raw=len(qs)*len(NAMES),not_whole_UST=True))
 for name in NAMES:
  save(O/name/'queries.local.json',qs);save(O/name/'protocol.json',dict(p0,steps=0,queries_each=len(qs),Final_authorized=True,training=0))
 pins=sourcepins+[bind(__file__),bind(O/'plan.json'),bind(O/'models-frozen-before-Final.json')]+[bind(p) for p in O.rglob('*.json') if p.name not in ('controller-started.json','launch.json','progress.json') and 'progress' not in p.parts]
 save(O/'protocol.json',dict(bindings=list({b['path']:b for b in pins}.values()),Final_once=True,no_training=True,API=0,total_raw=len(qs)*len(NAMES)));verify();save(O/'preflight.json',dict(status='PASS_FINAL_PREPARED_AFTER_MODELS_FROZEN',questions=len(rows),outputs=len(qs)*len(NAMES)))

def score():
 import ust_joint_generator_student_closed_audit_20261107 as independent
 verify();a=load(O/'answers.local.json');parse=independent.independent();scores={};grades={};rawpins=[]
 for name in NAMES:
  qq=load(O/name/'queries.local.json');term=load(O/name/'evaluation-terminal.json');assert term['raw']==len(qq) and term['roles_unchanged'] and term['optimizer_unchanged'] and term['RNG_unchanged']
  rawpins.extend(bind(O/name/'raw'/f'{i:04d}.json') for i in range(len(qq)))
 save(O/'all-raw-closed-before-score.json',dict(bindings=rawpins,epoch=time.time()))
 def stats(rr):return dict(n=len(rr),groups=len({r['group'] for r in rr}),correct=sum(r['correct'] for r in rr),compatible=sum(r['compatible'] for r in rr),valid=sum(r['valid'] for r in rr),capped=sum(r['capped'] for r in rr))
 for name in NAMES:
  rr=[]
  for i,q in enumerate(load(O/name/'queries.local.json')):
   raw=load(O/name/'raw'/f'{i:04d}.json');assert raw['id']==q['id'];pred=parse(raw['raw_generated'],raw['finish_reason']);g=parser.classify(raw['raw_generated'],'off','ust',q['metadata'],raw['finish_reason']);assert pred==g['prediction']
   rr.append(dict(id=q['id'],group=q['group'],variant=q['variant'],metric=q['metric'],flags=q['flags'],prediction=pred,correct=bool(g['valid'] and pred==a[q['id']]['answer']),compatible=bool(g['valid'] and pred in a[q['id']]['compatible']),valid=bool(g['valid']),capped=raw['finish_reason']=='length'))
  save(O/name/'graded.local.json',rr);grades[name]=rr;scores[name]={v:dict(overall=stats([x for x in rr if x['variant']==v]),metrics={m:stats([x for x in rr if x['variant']==v and x['metric']==m]) for m in ('distance','waiting_time','charging_time')},subgroups={f:stats([x for x in rr if x['variant']==v and x['flags'][f]]) for f in ('target_tail','target_NaN','target_tie')}) for v in ('original','joint')}
 contrasts={};rng=np.random.default_rng(2026110913)
 for other in ('real','old','simple'):
  for view in ('original','joint'):
   av=[[x for x in grades[f'current_n1088_s{seed}'] if x['variant']==view] for seed in s.SEEDS];bv=[[x for x in grades[f'{other}_n1088_s{seed}'] if x['variant']==view] for seed in s.SEEDS]
   for x,y in zip(av,bv):assert [r['id'] for r in x]==[r['id'] for r in y]
   delta=np.array([[int(x['correct'])-int(y['correct']) for x,y in zip(aa,bb)] for aa,bb in zip(av,bv)]);mean=delta.mean(axis=0);group=[x['group'] for x in av[0]];gg=sorted(set(group));sums=np.array([sum(mean[i] for i,g in enumerate(group) if g==v) for v in gg]);counts=np.array([group.count(v) for v in gg]);draw=rng.integers(len(gg),size=(5000,len(gg)));ci=100*np.quantile(sums[draw].sum(axis=1)/counts[draw].sum(axis=1),[.025,.975]);contrasts[other+'/'+view]=dict(per_order=delta.sum(axis=1).tolist(),mean_pp=float(100*mean.mean()),conditional_cluster95_pp=ci.tolist())
 save(O/'results.json',dict(status='COMPLETE_ONE_FINAL_ACCEPTANCE',scores=scores,contrasts=contrasts,scope=load(O/'scope.json'),API=0,Final_now_exposed=True,not_noninferiority_proof=True));save(O/'terminal.json',dict(status='COMPLETE',epoch=time.time(),results=bind(O/'results.json')))

def controller():
 plancheck();save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time()));deadline=time.monotonic()+172800
 while not (WAIT/'terminal.json').exists():
  check();assert not list(WAIT.glob('failure-*.json')),'Curve failure; Final stays sealed';assert time.monotonic()<deadline;time.sleep(10)
 def call(action,key=None):
  check();args=[str(R/'.venv/bin/python'),__file__,action]+([key] if key else []);name=action+('-'+key if key else '');folder=O/'logs';folder.mkdir(exist_ok=True)
  with (folder/(name+'.log')).open('x') as out:
   child=subprocess.Popen(args,cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',**({'CUDA_VISIBLE_DEVICES':''} if action!='evaluate' else {})),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);save(O/'processes'/(name+'.json'),dict(PID=child.pid,epoch=time.time()));start=time.monotonic()
   try:
    while child.poll() is None:
     check();assert time.monotonic()-start<14400 and time.monotonic()<deadline;time.sleep(2)
    assert child.returncode==0,f'{name} exit {child.returncode}'
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 call('freeze');call('prepare')
 for name in NAMES:call('evaluate',name)
 call('score')
if __name__=='__main__':
 def stop(*_):
  global STOP;STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);configure()
 try:
  action=sys.argv[1]
  if action=='evaluate':s.base.evaluate(sys.argv[2])
  else:globals()[action]()
 except BaseException as e:save(O/f'failure-{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),epoch=time.time()));raise
