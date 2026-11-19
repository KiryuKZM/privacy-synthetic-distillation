"""Once-only, fixed-model Common100 descriptive evaluation; never model selection.

Prepare is gated on ALL twelve completed training endpoints and hashes their
payloads before opening Common questions. Published gold is opened only after
all 2700 new raw outputs close. No fit, teacher call, or protected final role.
"""
from pathlib import Path
import argparse,ast,collections,fcntl,hashlib,importlib.util,itertools,json,os,re,signal,sqlite3,subprocess,sys,time

R=Path(__file__).resolve().parents[1]
B=R/'runs/2026-11-06-ust-autonomous-repair-v1';O=B/'common100-descriptive-v1'
POSITION=B/'position-training-v2';CAUSAL=B/'causal-training';REPLICATION=B/'position-order-replicates-v1'
OLD=R/'runs/2026-10-25-qwen-transfer-repair/seqkd200/USTBench/2b';INIT=OLD/'work/checkpoints/step0000'
COMMON=R/'runs/2026-10-24-ust100-sg60-modes/data-freeze'
TAIL=R/'runs/2026-11-05-ust-environment-generator-ipd-v1/tail-outlier-utility-v2'
TRANSFORM_CODE=R/'scripts/ust_dev_semantic_transforms_20261106.py'
REFERENCE_CODE=R/'scripts/ust_position_balanced_training_prepare_20261106.py'
ADAPTER_CODE=R/'scripts/ust_output_format_adapter_20261106.py'
ADAPTER_SHA='a836c4c23df6b0377de99248c9a6a1c1d6138e07ee1db6fb4d0859662079e06c'
CAUSAL_CODE=R/'scripts/ust_causal_training_20261106.py'
DECODER_CODE=R/'scripts/three_method_common_eval_20261101.py'
ORDER_SEEDS=(2026110602,2026110603)
ORDERS=('original_order',*(str(x) for x in ORDER_SEEDS))
ARMS=('real-canonical200','real-balanced200','synthetic-identity200','synthetic-balanced200')
ABILITIES=('st_understanding','planning','reflection');VARIANTS=('original','canonical','joint')
FLAGS=('upper_tail','lower_tail','any_tail','time_NaN','time_NaN_mismatch','statistical_outlier_candidate','Fit_unseen_joint_bin','Fit_rare_seen_joint_bin')
EXPECTED_FLAGS=dict(upper_tail=17,lower_tail=9,any_tail=21,time_NaN=1,time_NaN_mismatch=0,statistical_outlier_candidate=0,Fit_unseen_joint_bin=21,Fit_rare_seen_joint_bin=100)
MODEL_IDS=('fresh-init0','order0-real-canonical200','order0-synthetic-identity200',
           'order0-real-balanced200','order0-synthetic-balanced200',
           *(f'order{s}-{a}' for s in ORDER_SEEDS for a in ('real-balanced200','synthetic-balanced200')))
COUNT=300;TOTAL=2700;STOP=False
SCOPE=('Already exposed Common100 descriptive assessment of a prospectively fixed nine-model union. '
       'No candidate, checkpoint, recipe, parameter or subgroup selection from these results. '
       'Published planning/reflection references remain uncertain; no independent final acceptance, '
       'synthetic-replacement equivalence, physical-scene independence or verified-outlier claim.')

def load(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(pin):
 p=R/pin['path'];actual=bind(p)
 assert all(actual[k]==v for k,v in pin.items()),pin['path']
 return p
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def check():
 if STOP or (B/'STOP').exists() or (O/'STOP').exists():raise InterruptedError('Own Common descriptive STOP')
def stop(*_):
 global STOP;STOP=True
def live(**kw):
 p=O/'live.tmp';p.write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time(),**kw)));p.replace(O/'live.json')

def source_units():
 units={('original_order',a):(CAUSAL/'A-original200-replay' if a=='synthetic-identity200' else POSITION/a) for a in ARMS}
 units.update({(str(s),a):REPLICATION/str(s)/a for s in ORDER_SEEDS for a in ARMS})
 assert len(units)==12
 return units

def checkpoint_pins(root,folder,updates):
 """Verify actual payload bytes, not merely the small manifest's existence."""
 m=load(folder/'manifest.json');assert m['actual_updates']==m['step']==updates
 assert m['protocol_sha256']==bind(root/'protocol.json')['sha256']
 assert set(m['role_hashes'])=={'actor','reference','discriminator'}
 required={'actor/adapter_model.safetensors','reference/adapter_model.safetensors',
           'discriminator/adapter_model.safetensors','score-head.pt','optimizers.pt','rng.pt'}
 assert required<=set(m['files'])
 pins=[bind(folder/'manifest.json'),bind(root/'protocol.json')]
 for name,digest in m['files'].items():
  path=(folder/name).resolve();assert path.is_relative_to(folder.resolve()),'Unsafe checkpoint member'
  pin=bind(path);assert pin['sha256']==digest,name;pins.append(pin)
 return m,pins

def freeze_models():
 """This must finish before ANY Common question/gold/flags file is opened."""
 check();pins=[]
 for root in (CAUSAL,POSITION,REPLICATION):
  t=load(root/'terminal.json');assert t['status']=='COMPLETE',str(root)
  checked(t['results']);pins.extend([bind(root/'terminal.json'),t['results'],bind(root/'protocol.json')])
 rep=load(REPLICATION/'results.json')
 assert rep['training_order_seeds']==list(ORDER_SEEDS) and rep['total_new_updates']==1600 and rep['total_new_raw']==1696
 units=source_units();records={};model_specs={}
 for (order,arm),root in units.items():
  f=load(root/'fit-terminal.json');e=load(root/'evaluation-terminal.json')
  assert f['status']=='FIXED_ENDPOINT_COMPLETE' and f['actual_updates']==200
  assert e['status']=='ALL212_RAW_CLOSED' and e['raw']==212 and e['roles_unchanged'] and e['optimizer_unchanged']
  folder=root/'checkpoints/step0200';m,more=checkpoint_pins(root,folder,200)
  assert f['checkpoint']==bind(folder/'manifest.json') and f['roles']==m['role_hashes']
  pins.extend(more+[bind(root/'fit-terminal.json'),bind(root/'evaluation-terminal.json')])
  key=('order0' if order=='original_order' else 'order'+order)+'-'+arm
  records[key]=dict(order=order,arm=arm,source_root=str(root.relative_to(R)),checkpoint=str(folder.relative_to(R)),
    manifest=bind(folder/'manifest.json'),actual_updates=200,role_hashes=m['role_hashes'],optimizer_hashes=m['optimizer_hashes'])
 assert len(records)==12
 m,more=checkpoint_pins(OLD,INIT,0);pins+=more
 initial=dict(order='none',arm='fresh-init0',source_root=str(OLD.relative_to(R)),checkpoint=str(INIT.relative_to(R)),
   manifest=bind(INIT/'manifest.json'),actual_updates=0,role_hashes=m['role_hashes'],optimizer_hashes=m['optimizer_hashes'])
 for key in MODEL_IDS:model_specs[key]=initial if key=='fresh-init0' else records[key]
 assert len(model_specs)==9
 return dict(status='ALL12_TRAINED_ENDPOINTS_AND_INIT_PAYLOADS_HASH_FROZEN',epoch=time.time(),
   trained_endpoints=records,models=model_specs,bindings=list({p['path']:p for p in pins}.values()),
   model_selection='exact prior nine-model union, not selected from replication or Common outcomes')

def read_common_requests(pin):
 # Kept as an explicit seam so CPU tests can prove freeze failures never read it.
 return [json.loads(line) for line in checked(pin).read_text().splitlines()]

def derangements(qid):
 digest=hashlib.sha256(('UST-common-20261106:'+qid).encode()).digest()
 locs=[x for x in itertools.permutations(range(1,5)) if all(a!=b for a,b in zip(x,range(1,5)))]
 opts=[x for x in itertools.permutations('ABCD') if all(a!=b for a,b in zip(x,'ABCD'))]
 lp=locs[int.from_bytes(digest[:16],'big')%len(locs)];op=opts[int.from_bytes(digest[16:],'big')%len(opts)]
 return dict(zip(range(1,5),lp)),op

def build_queries(rows):
 """Transforms and displayed-number relations are checked without source gold."""
 t=module(TRANSFORM_CODE,'ust_common_pure_transforms');guard=module(REFERENCE_CODE,'ust_common_reference_grammar');sys.path.insert(0,str(R/'src'))
 from privacy_synth.ust_task_contract import certify_native_ranking_reference
 queries=[];proof=[]
 for row in rows:
  qid=row['id'];raw=row['adapted_messages'][-1]
  assert raw['role']=='user' and isinstance(raw['content'],str)
  original=raw['content'];source_row=dict(question=original,metadata=row['metadata'])
  refs=guard.understood_references(source_row);canonical=t.canonical(original)
  assert guard.understood_references(dict(source_row,question=canonical))==refs
  assert re.sub(r'\s+','',original)==re.sub(r'\s+','',canonical),'Canonical must be whitespace only'
  loc,order=derangements(qid);joint=t.transform(original,loc,order)
  checks=t.validate_pair(original,joint,loc,order);assert checks['status']=='PASS'
  newrefs=guard.understood_references(dict(source_row,question=joint['question']))
  assert newrefs['analysis_refs']==[[title,loc[n],value] for title,n,value in refs['analysis_refs']]
  assert newrefs['reason_location_refs']==[loc[n] for n in refs['reason_location_refs']]
  assert newrefs['reason_letter_refs']==[joint['old_to_new_letter'][a] for a in refs['reason_letter_refs']]
  if row['metadata']['ability']=='reflection':
   assert t.sections(original)['Environment Changes']==t.sections(joint['question'])['Environment Changes']
  inv={v:k for k,v in loc.items()};rev=t.transform(joint['question'],inv,[joint['old_to_new_letter'][a] for a in 'ABCD'])
  assert rev['question']==canonical,'Full inverse must restore canonical text'
  numerical=None
  if row['metadata']['ability']=='st_understanding':
   before=certify_native_ranking_reference(original,None);canon=certify_native_ranking_reference(canonical,None);after=certify_native_ranking_reference(joint['question'],None)
   assert before['compatible_answers']==canon['compatible_answers']
   assert set(after['compatible_answers'])=={joint['old_to_new_letter'][a] for a in before['compatible_answers']}
   independent=t.understanding_answers(original)
   if independent is not None:assert set(independent)==set(before['compatible_answers'])
   numerical=dict(status='DISPLAYED_RELATION_PRESERVED_WITHOUT_SOURCE_GOLD',compatible_answer_count=len(before['compatible_answers']),
     missing_values=before['missing_values'],nan_policy=before['nan_policy'],independent_Decimal_checked=independent is not None)
  proof.append(dict(source_id=qid,location_map=loc,option_order=order,old_to_new_letter=joint['old_to_new_letter'],
    checks=checks,references=refs,all_supported_references_mapped=True,feedback_unchanged=True,
    inverse_equal=True,canonical_whitespace_only=True,numerical=numerical))
  for variant,question,mapping in [('original',original,{a:a for a in 'ABCD'}),('canonical',canonical,{a:a for a in 'ABCD'}),('joint',joint['question'],joint['old_to_new_letter'])]:
   queries.append(dict(id=qid+'::'+variant,source_id=qid,group=row['group'],question=question,
     metadata={'ability':row['metadata']['ability']},variant=variant,cohort='exposed_Common100',cap=128,old_to_new_letter=mapping))
 return queries,proof

def frozen_flags(tags,rows):
 selected=[x for x in tags if x['cohort']=='historical_Common100'];lookup={x['id']:x for x in selected}
 assert len(selected)==len(lookup)==len(rows)==100
 assert set(lookup)=={x['id'] for x in rows}
 for row in rows:
  x=lookup[row['id']];assert x['group']==row['group'] and x['ability']==row['metadata']['ability']
  assert set(x['flags'])==set(FLAGS) and all(type(v) is bool for v in x['flags'].values())
 assert {f:sum(x['flags'][f] for x in selected) for f in FLAGS}==EXPECTED_FLAGS
 assert len({x['group'] for x in selected})==99
 return selected

def machinery():
 c=module(CAUSAL_CODE,'ust_common_same_initialization');c.O=O;c.B=B;c.verify=verify;c.check=check;c.live=live
 assert c.OLD==OLD and c.INIT==INIT
 return c.machinery()

def prepare():
 assert not (O/'protocol.json').exists() and not (O/'models-frozen-before-common.json').exists(),'No implicit reprepare/overwrite'
 frozen=freeze_models()  # Deliberately first: never read Common while a model can still change.
 assert bind(ADAPTER_CODE)['sha256']==ADAPTER_SHA,'Prospectively frozen output adapter changed'
 base_protocol=load(POSITION/'real-canonical200/protocol.json');base=Path(base_protocol['base'])
 assets=[bind(p) for p in sorted(base.iterdir()) if p.is_file()]
 frozen['base_assets']=assets
 O.mkdir(parents=True,exist_ok=True);save(O/'models-frozen-before-common.json',frozen)
 manifest=load(COMMON/'manifest.json');qpin=manifest['USTBench_requests'];gpin=manifest['USTBench_gold']
 assert qpin['path']==str((COMMON/'USTBench.requests.jsonl').relative_to(R))
 assert gpin['path']==str((COMMON/'USTBench.screen100.gold.local.json').relative_to(R))
 save(O/'common-question-read-started.json',dict(epoch=time.time(),models_frozen=bind(O/'models-frozen-before-common.json'),gold_opened=False))
 rows=read_common_requests(qpin)
 assert len(rows)==len({r['id'] for r in rows})==100 and len({r['group'] for r in rows})==99
 assert collections.Counter(r['metadata']['ability'] for r in rows)==dict(st_understanding=34,planning=33,reflection=33)
 queries,proof=build_queries(rows);assert len(queries)==COUNT and len({q['id'] for q in queries})==COUNT
 flags=frozen_flags(load(TAIL/'input-flags.local.json'),rows)
 save(O/'queries.local.json',queries);save(O/'transform-qualification.local.json',proof);save(O/'source-flags.local.json',flags)
 prefix=(POSITION/'fixed8.local.json').read_bytes();assert prefix==(CAUSAL/'fixed8.local.json').read_bytes()==(REPLICATION/'fixed8.local.json').read_bytes()
 with (O/'fixed8.local.json').open('xb') as f:f.write(prefix)
 m,w,_=machinery()  # Imports only; no model initialization or calls.
 sys.path[:0]=[str(R/'scripts'),str(R/'src')]
 import eval_ust_sg_qwen_modes_20261024,run_llm_task_sweep_20261023,build_llm_task_sweep_qa_20261023
 module(ADAPTER_CODE,'ust_common_frozen_adapter')
 runtime=[Path(__file__),TRANSFORM_CODE,REFERENCE_CODE,ADAPTER_CODE,CAUSAL_CODE,DECODER_CODE,R/'scripts/train_ust_sg_native_gad_20261024.py',R/'pyproject.toml',R/'uv.lock']
 for mod in tuple(sys.modules.values()):
  path=getattr(mod,'__file__',None)
  if path:
   p=Path(path).resolve()
   if p.is_relative_to(R) and p.suffix=='.py' and '.venv' not in p.parts and 'vendor' not in p.parts:runtime.append(p)
 # No Common gold pin is verified/read here, including indirectly through bindings.
 pins=frozen['bindings']+[bind(p) for p in set(runtime)]+assets
 pins+=[bind(p) for p in (COMMON/'manifest.json',TAIL/'protocol.json',TAIL/'thresholds.local.json',TAIL/'input-flags.local.json')]
 pins+=[bind(O/n) for n in ('models-frozen-before-common.json','queries.local.json','transform-qualification.local.json','source-flags.local.json','fixed8.local.json')]+[bind(checked(qpin))]
 pins=list({p['path']:p for p in pins}.values());assert all(p['path']!=gpin['path'] for p in pins)
 save(O/'protocol.json',dict(source=bind(__file__),bindings=pins,models=frozen['models'],all12_endpoints_frozen=True,
   models_each=9,queries_each=COUNT,total_raw=TOTAL,source_cases=100,source_groups=99,variants=VARIANTS,
   expected_gold_binding_not_opened=gpin,original_questions=qpin,original_flags=bind(TAIL/'input-flags.local.json'),
   fixed8=bind(O/'fixed8.local.json'),cap=128,context_cap=16384,thinking=False,greedy=True,training=0,API=0,new_weights=0,
   transformation_seed='SHA256(UTF8(UST-common-20261106:+qid)); first/last16bytes choose lexicographic location/option derangements modulo9',
   source_gold_mapping='mechanical old_to_new_letter only; no physics re-solve or replacement',
   scope=SCOPE,Common_use='descriptive_only_never_future_parameter_selection',Audit_access=False,Confirm_access=False,Final_access=False,
   adapter=bind(ADAPTER_CODE),score_modes=['strict_primary','normalized_interface_secondary'],outlier_status='NA_NO_CASES',
   no_best_order_selection=True,no_extra_candidates=True,base_assets=assets))
 for key,spec in frozen['models'].items():
  p=dict(base_protocol);p.update(source=bind(__file__),bindings=pins+[bind(O/'protocol.json')],model_id=key,
    source_checkpoint=spec,steps=0,fixed_checkpoint=spec['actual_updates'],method='inference-only',training=0,
    queries_each=COUNT,total_raw=TOTAL,scope=SCOPE,Common_access=True,Audit_access=False,Confirm_access=False,Final_access=False,teacher_API=0,
    ability_order=[],checkpoint_steps=[],evaluation_use=SCOPE,protected_real_training_gold_in_loss=False,
    synthetic_reference_in_loss=False,training_label_status='no new training',
    optimizer_restored='frozen source optimizer for integrity verification only; never updated')
  save(O/key/'protocol.json',p)
 save(O/'execution-bindings.json',dict(source=bind(__file__),protocols=[bind(O/'protocol.json')]+[bind(O/key/'protocol.json') for key in MODEL_IDS]))
 verify();print('FROZEN9models300queries2700raw; source gold unopened',flush=True)

def verify(unit=None):
 check();closure=load(O/'execution-bindings.json');assert closure['source']==bind(__file__)
 for p in closure['protocols']:checked(p)
 p=load((Path(unit) if unit else O)/'protocol.json');assert p['source']==bind(__file__)
 assert p['queries_each']==COUNT and p['total_raw']==TOTAL
 for pin in p['bindings']:checked(pin)
 if unit is not None:assert Path(unit).name in MODEL_IDS and Path(unit).parent==O and p['steps']==p['training']==0
 return p

def decoder():
 nodes=[x for x in ast.parse(DECODER_CODE.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in ('hf_greedy','context_failure')]
 assert len(nodes)==2
 ns=dict(time=time,CAP=128,CONTEXT=16384)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(DECODER_CODE)+'::UNCHANGED_DECODER','exec'),ns)
 return ns['hf_greedy']

def evaluate(model_id):
 assert model_id in MODEL_IDS;unit=O/model_id;p=verify(unit);spec=p['source_checkpoint'];m,w,_=machinery();decode=decoder()
 source=R/spec['source_root'];folder=R/spec['checkpoint']
 save(unit/'evaluation-started.json',dict(epoch=time.time(),PID=os.getpid(),source_checkpoint=spec,training_updates=0))
 with (m.S/'gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch,tok,model,head=w.initialize(unit,folder);w.m.lib.activate(model,head,'actor');model.eval()
  opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
  cm=w.m.lib.restore(torch,model,head,dict(actor=opt),source,folder)
  roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']};oh=w.m.lib.nested_hash(opt.state_dict())
  assert roles==cm['role_hashes']==spec['role_hashes'] and oh==spec['optimizer_hashes']['actor']
  assert cm['actual_updates']==spec['actual_updates']
  if model_id=='fresh-init0':assert source==OLD and folder==INIT and not opt.state and cm['actual_updates']==0
  save(unit/'independent-reload.json',dict(status='EXACT_SOURCE_ROLES_OPTIMIZER_RNG_RELOAD',roles=roles,optimizer=oh,checkpoint=bind(folder/'manifest.json'),source_root=str(source.relative_to(R)),new_training_updates=0))
  queries=load(O/'queries.local.json');prefix=load(O/'fixed8.local.json')
  assert len(queries)==COUNT
  for i,q in enumerate(queries):
   check();save(unit/'raw'/f'{i:04d}.started.json',dict(id=q['id'],epoch=time.time()))
   ids=tok.encode(tok.apply_chat_template(prefix+[dict(role='user',content=q['question'])],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
   raw=decode(torch,tok,model,ids)
   save(unit/'raw'/f'{i:04d}.json',dict(id=q['id'],source_id=q['source_id'],variant=q['variant'],cap=128,**raw))
   if (i+1)%50==0:live(stage='EVALUATION',model=model_id,closed=i+1,total=COUNT);print('EVAL',model_id,i+1,COUNT,flush=True)
  assert roles=={r:w.m.lib.role_hash(model,r) for r in roles} and oh==w.m.lib.nested_hash(opt.state_dict())
  if model_id=='fresh-init0':assert not opt.state
  verify(unit)
  save(unit/'evaluation-terminal.json',dict(status='ALL300_RAW_CLOSED',raw=COUNT,roles_unchanged=True,optimizer_unchanged=True,
    role_hashes=roles,optimizer_hash=oh,source_checkpoint=spec['manifest'],new_training_updates=0,API=0))

def close_raw():
 queries=load(O/'queries.local.json');assert len(queries)==COUNT;pins=[]
 for model_id in MODEL_IDS:
  unit=O/model_id;t=load(unit/'evaluation-terminal.json');spec=load(O/'protocol.json')['models'][model_id]
  assert t['status']=='ALL300_RAW_CLOSED' and t['raw']==COUNT and t['roles_unchanged'] and t['optimizer_unchanged']
  assert t['new_training_updates']==t['API']==0 and t['role_hashes']==spec['role_hashes'] and t['source_checkpoint']==spec['manifest']
  assert t['optimizer_hash']==spec['optimizer_hashes']['actor']
  assert len(list((unit/'raw').glob('[0-9][0-9][0-9][0-9].json')))==COUNT
  for i,q in enumerate(queries):
   path=unit/'raw'/f'{i:04d}.json';raw=load(path);started=load(path.with_suffix('.started.json'))
   assert raw['id']==started['id']==q['id'] and raw['source_id']==q['source_id'] and raw['variant']==q['variant'] and raw['cap']==128
   assert isinstance(raw['raw_generated'],str) and raw['finish_reason'] in ('stop','length','context_overflow')
   pins.append(bind(path))
 assert len(pins)==TOTAL
 return pins

def aggregate(rows,mode):
 n=len(rows)
 return dict(n=n,source_cases=len({r['source_id'] for r in rows}),source_groups=len({r['group'] for r in rows}),
   correct=sum(r[mode]['correct'] for r in rows),valid=sum(r[mode]['valid'] for r in rows),
   capped=sum(r['capped'] for r in rows),converted=sum(r['adapter']['converted'] for r in rows),
   status='MEASURED_DESCRIPTIVE' if n else 'NA_NO_CASES')

def summaries(rows):
 result={}
 for mode in ('strict','normalized'):
  result[mode]={}
  for variant in VARIANTS:
   rr=[x for x in rows if x['variant']==variant];assert len(rr)==100
   result[mode][variant]={}
   for flag in ('all',*FLAGS):
    take=[x for x in rr if flag=='all' or x['flags'][flag]]
    result[mode][variant][flag]=dict(**aggregate(take,mode),abilities={a:aggregate([x for x in take if x['ability']==a],mode) for a in ABILITIES})
 return result

def equivariance(rows,queries):
 lookup={(r['source_id'],r['variant']):r for r in rows};out={}
 for mode in ('strict','normalized'):
  out[mode]={}
  for variant in ('original','joint'):
   pairs=[]
   for q in queries:
    if q['variant']!=variant:continue
    a=lookup[q['source_id'],'canonical'];b=lookup[q['source_id'],variant];valid=a[mode]['valid'] and b[mode]['valid']
    assert q['old_to_new_letter'][a['answer']]==b['answer']
    pairs.append(dict(source_id=q['source_id'],ability=q['metadata']['ability'],flags=b['flags'],
      both_valid=valid,equivariant=bool(valid and q['old_to_new_letter'].get(a[mode]['prediction'])==b[mode]['prediction']),
      both_correct=a[mode]['correct'] and b[mode]['correct']))
   def count(pp):return dict(n=len(pp),both_valid=sum(x['both_valid'] for x in pp),prediction_equivariant=sum(x['equivariant'] for x in pp),both_correct=sum(x['both_correct'] for x in pp),status='MEASURED_DESCRIPTIVE' if pp else 'NA_NO_CASES')
   out[mode][variant+'_versus_canonical']={f:dict(**count([x for x in pairs if f=='all' or x['flags'][f]]),
     abilities={a:count([x for x in pairs if (f=='all' or x['flags'][f]) and x['ability']==a]) for a in ABILITIES}) for f in ('all',*FLAGS)}
 return out

def order_contrasts(all_rows):
 result=[]
 for mode in ('strict','normalized'):
  for variant in VARIANTS:
   for flag in ('all','any_tail','statistical_outlier_candidate'):
    for ability in ('all',*ABILITIES):
     records=[]
     for order in ORDERS:
      pre='order0' if order=='original_order' else 'order'+order
      real=all_rows[pre+'-real-balanced200'];synth=all_rows[pre+'-synthetic-balanced200']
      assert [x['id'] for x in real]==[x['id'] for x in synth]
      pairs=[(a,b) for a,b in zip(real,synth) if a['variant']==variant and (flag=='all' or a['flags'][flag]) and (ability=='all' or a['ability']==ability)]
      n=len(pairs);rc=sum(a[mode]['correct'] for a,b in pairs);sc=sum(b[mode]['correct'] for a,b in pairs)
      records.append(dict(order=order,n=n,real_correct=rc,synthetic_correct=sc,delta_correct=sc-rc,
        synthetic_wins=sum(not a[mode]['correct'] and b[mode]['correct'] for a,b in pairs),
        real_wins=sum(a[mode]['correct'] and not b[mode]['correct'] for a,b in pairs)))
     deltas=[x['delta_correct'] for x in records]
     result.append(dict(mode=mode,variant=variant,subgroup=flag,ability=ability,orders=records,
       source_cases_per_order=records[0]['n'],delta_range=[min(deltas),max(deltas)] if records[0]['n'] else None,
       mean_delta_correct=sum(deltas)/3 if records[0]['n'] else None,status='MEASURED_DESCRIPTIVE' if records[0]['n'] else 'NA_NO_CASES',
       repeated_cases_not_independent_samples=True,no_best_order_selection=True))
 return result

def score():
 p=verify();assert not (O/'score-started.json').exists(),'Once-only scoring; never automatic replay'
 rawpins=close_raw();save(O/'all-raw-frozen-before-gold.json',dict(raw=rawpins,total=TOTAL,epoch=time.time()))
 save(O/'score-started.json',dict(epoch=time.time(),protocol=bind(O/'protocol.json'),all_raw=bind(O/'all-raw-frozen-before-gold.json')))
 # First and sole source-reference read, after complete raw closure.
 pin=p['expected_gold_binding_not_opened'];data=(R/pin['path']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==pin['sha256'];gold=json.loads(data)
 save(O/'gold-read-receipt.json',dict(epoch=time.time(),path=pin['path'],bytes=len(data),sha256=pin['sha256'],raw_closed=TOTAL))
 queries=load(O/'queries.local.json');flags={x['id']:x for x in load(O/'source-flags.local.json')}
 assert set(gold)==set(flags)=={q['source_id'] for q in queries} and all(isinstance(x,str) and x in 'ABCD' and len(x)==1 for x in gold.values())
 sys.path[:0]=[str(R/'scripts'),str(R/'src')];import eval_ust_sg_qwen_modes_20261024 as parser
 from privacy_synth.ust_task_contract import certify_native_ranking_reference
 adapter=module(ADAPTER_CODE,'ust_common_unchanged_output_adapter');all_rows={};scores={};positions={}
 consistency=[]
 for q in queries:
  if q['variant']=='original' and q['metadata']['ability']=='st_understanding':
   cert=certify_native_ranking_reference(q['question'],gold[q['source_id']]);consistency.append(dict(source_id=q['source_id'],**cert))
 save(O/'published-understanding-reference-check.local.json',consistency)
 for model_id in MODEL_IDS:
  rows=[]
  for i,q in enumerate(queries):
   raw=load(O/model_id/'raw'/f'{i:04d}.json');answer=q['old_to_new_letter'][gold[q['source_id']]]
   a=adapter.normalize_output(raw['raw_generated'],q['metadata']['ability'],raw['finish_reason']);grades={}
   for mode,text in [('strict',raw['raw_generated']),('normalized',a['text'])]:
    g=parser.classify(text,'off','ust',q['metadata'],raw['finish_reason'])
    grades[mode]=dict(prediction=g['prediction'],valid=bool(g['valid']),correct=bool(g['valid'] and g['prediction']==answer),reason=g.get('reason'))
   if not a['converted']:assert grades['strict']==grades['normalized']
   rows.append(dict(id=q['id'],source_id=q['source_id'],group=q['group'],variant=q['variant'],ability=q['metadata']['ability'],
     flags=flags[q['source_id']]['flags'],answer=answer,capped=raw['finish_reason']=='length',adapter=a,**grades))
  save(O/model_id/'graded.local.json',rows);all_rows[model_id]=rows;scores[model_id]=summaries(rows);positions[model_id]=equivariance(rows,queries)
 budget=load(B/'budget-scope-initial.json');db=sqlite3.connect('file:'+str(R/budget['historical_SQL'])+'?mode=ro',uri=True)
 digest=hashlib.sha256('\n'.join(db.iterdump()).encode()).hexdigest();db.close();assert digest==budget['historical_dump_sha256']
 verify();[checked(pin) for pin in rawpins]
 result=dict(status='COMPLETE_DESCRIPTIVE_NOT_SELECTION',models=list(MODEL_IDS),scores=scores,position_equivariance=positions,
   all_three_balanced_orders=order_contrasts(all_rows),new_raw=TOTAL,source_cases=100,source_groups=99,variants_are_repeated_cases=True,
   understanding_reference_check=dict(n=len(consistency),trusted=sum(x['trusted_gold'] for x in consistency),original_gold_unchanged=True),
   outlier='NA_NO_CASES; cannot reliably estimate true outlier utility',training=0,API=0,budget_SQL_unchanged=True,scope=SCOPE,
   strict_is_primary=True,normalized_is_format_only_no_reasoning_claim=True,no_best_order_selection=True)
 save(O/'results.json',result);save(O/'terminal.json',dict(status='COMPLETE',results=bind(O/'results.json'),epoch=time.time(),training=0,API=0));live(stage='COMPLETE')
 print('COMMON DESCRIPTIVE COMPLETE9models2700raw; no selection/final claim',flush=True)

def wait_for_replication(timeout=7200):
 deadline=time.monotonic()+timeout
 while not (REPLICATION/'terminal.json').exists():
  check()
  assert not list(REPLICATION.glob('failure-*.json')),'Source replication failed; no implicit recovery'
  if time.monotonic()>=deadline:raise TimeoutError('Waiting for source replication completion exceeded bound')
  live(stage='WAITING_FOR_REPLICATION_COMPLETE');time.sleep(5)
 assert load(REPLICATION/'terminal.json')['status']=='COMPLETE'

def controller():
 O.mkdir(parents=True,exist_ok=True)
 save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time(),Common_opened=False))
 wait_for_replication()
 if not (O/'protocol.json').exists():prepare()
 verify()
 for model_id in MODEL_IDS:
  check()
  with (O/(model_id+'-evaluate.log')).open('x') as f:
   subprocess.run([str(R/'.venv/bin/python'),str(Path(__file__)), 'evaluate','--model',model_id],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
 score()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','evaluate','score','controller'));parser.add_argument('--model',choices=MODEL_IDS);args=parser.parse_args()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if args.action=='evaluate':evaluate(args.model)
  else:globals()[args.action]()
 except BaseException as ex:
  if O.exists():save(O/('failure-'+str(time.time_ns())+'.json'),dict(type=type(ex).__name__,error=str(ex),epoch=time.time()))
  raise
