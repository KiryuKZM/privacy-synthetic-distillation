#!/usr/bin/env python3
"""Formal SGQA student adapter. CPU preparation is separate from exact Root-granted GPU execution.
Reuses frozen native single-row CE, optimizer, packing, checkpoint serializer and loader.
Never imports the teacher collector, credentials, Holdout queries or any gold.
"""
from __future__ import annotations
import argparse, ast, collections, copy, gc, hashlib, json, os, shutil, signal, subprocess, sys, time, types, weakref
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/2026-10-27-generator-formal'
PREP=RUN/'preparation/student-executor-v1'
OUT=RUN/'students-v1'
BODIES=RUN/'teacher-bodies-cpu-v1'
TEACHER=RUN/'teacher-execution-v2'
SELECT=RUN/'cpu-public-proxy-v2/selection.json'
PROTOCOL=RUN/'preparation/formal-study-protocol.json'
GPU_LOCK=ROOT/'runs/2026-10-22-night-dual/locks/gpu.lock'
SEEDS=(2026102751,2026102752,2026102753)
MILESTONES=[0,200,500,1000]
OLD_HASHES={
 'scripts/tinyfit_qwen_train16_20261025.py':'9c41ec8e5e66e4929784c6e8a17151c0132afb8f310559501b27a1f557cea280',
 'scripts/train_sg_seqkd200_20261025.py':'a0962e8e58a6fb7657606c826000f8a622ea9766b3ccf9ce68ea08a36b257c36',
 'scripts/train_ust_sg_native_gad_20261024.py':'fd04244b6d288dd3b6412a87eaca46f319d7dfe950ae50346959b419b8c8ce16',
 'scripts/probe_overnight_train_fit_20261025.py':'ad1033f625d4b65275b784ed8583d169c6b0ebd280793b44963babb132f12bcb',
 'scripts/train_overnight_native_gad_20261025.py':'36a4bee56127bde1b9be498f5bf8b9c465031d2826fa1285e4540d0373c0c848'}
SELECT_SHA='4fd89bd90d576de583fa3482f624fb658d8c4ae5c97826de873cb4bcd523a757'
BODY_SHA='2b6001682e695ca59cb8fdd926b6d7da595a5763a4c52cf7adbde5ae7ed16859'
STOP=False

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve().relative_to(ROOT)),sha256=sha(p))
def local(p):
 p=Path(p)
 if p.is_absolute() or '..' in p.parts:raise PermissionError('Project-relative binding required')
 raw=ROOT/p;r=raw.resolve()
 if not r.is_relative_to(ROOT) or any(x.is_symlink() for x in [raw,*raw.parents] if x.is_relative_to(ROOT)):raise PermissionError('Project binding escape')
 return r

def bound(s):
 p=local(s['path'])
 if sha(p)!=s['sha256']:raise PermissionError('Frozen bytes changed: '+s['path'])
 return p

def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 with temp.open('x',encoding='utf-8') as f:json.dump(x,f,sort_keys=True,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def once(p,x):
 if Path(p).exists():
  if load(p)!=x:raise PermissionError('Immutable record changed: '+str(p))
 else:atomic(p,x)

def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def old_sources():
 for p,h in OLD_HASHES.items():
  if sha(ROOT/p)!=h:raise PermissionError('Frozen old source changed: '+p)
 sys.path.insert(0,str(ROOT/'scripts'))
 import train_sg_seqkd200_20261025 as native
 return native

def matrix():
 return [dict(unit=f'SGQA-{size}-{route}-N{n}-s{seed}',size=size,route=route,N=n,seed=seed,updates=1000,presentations=1000,role='main' if n==1000 else 'secondary_nested_scale')
  for n in (1000,200) for size in (('08','2b') if n==1000 else ('08',)) for route in ('markov','control') for seed in SEEDS]

def check_matrix():
 z=load(PROTOCOL)
 if z['total_student_fits']!=18 or z['optimization_seeds']!=list(SEEDS) or z['student_recipe']['updates']!=1000 or z['student_recipe']['lora_r']!=8 or z['student_recipe']['lora_alpha']!=16 or z['student_recipe']['learning_rate']!=1e-4:raise PermissionError('Formal matrix/recipe drift')
 if sha(SELECT)!=SELECT_SHA or sha(BODIES/'manifest.json')!=BODY_SHA:raise PermissionError('Frozen selection/body manifest changed')
 return matrix()

def clean_row(identifier,question,system,family):
 if not isinstance(question,str) or not question.strip():raise PermissionError('Full actual input required')
 return dict(id=identifier,question=question,messages=[system,dict(role='user',content=question)],metadata=dict(family_index=family))

def final_text(ref,identifier):
 # Parsed-invalid and wrong answers are deliberately retained. Technical failures are forbidden.
 if ref.get('id')!=identifier or ref.get('actual_teacher_only') is not True or ref.get('finish_reason')!='stop' or not isinstance(ref.get('text'),str) or not ref['text'].strip() or ref.get('mode')!='on-high':raise PermissionError('Actual nonempty completed final text required; no local/gold fallback')
 return dict(id=identifier,text=ref['text'],finish_reason='stop',valid=ref.get('valid'),actual_teacher_only=True,mode='on-high')

def route_data(route):
 if route not in ('markov','control'):raise PermissionError('Frozen two routes only')
 check_matrix();manifest=load(BODIES/'manifest.json');stage=route+'-trainpool';qm=load(TEACHER/(route+'-qual')/'qualification-score.json')
 if qm.get('total')!=100 or qm.get('correct',0)<79 or qm.get('valid',0)<95 or qm.get('qualified') is not True:raise PermissionError('Route qualification79correct/95valid/100 required')
 for b in qm.get('bindings',[]):bound(b)
 request_path=BODIES/(stage+'.requests.jsonl');member_path=BODIES/(stage+'.members.jsonl')
 if sha(request_path)!=manifest['packages'][stage]['request_sha256'] or sha(member_path)!=manifest['packages'][stage]['members_sha256']:raise PermissionError('Frozen bodies/member bytes changed')
 requests=rows(request_path);members=rows(member_path);by_member={r['id']:r for r in members}
 folder=TEACHER/stage;rf=load(folder/'references-frozen.json');pf=load(bound(rf['predictions']))
 if rf.get('count')!=1000 or rf.get('all1000_retained') is not True or rf.get('no_teacher_filtering') is not True:raise PermissionError('All1000 actual references mandatory')
 for b in pf['predictions']:bound(b)
 refs=rows(bound(rf['records']));by={r['id']:r for r in refs}
 if len(requests)!=1000 or len(members)!=1000 or len(refs)!=1000 or len(by)!=1000 or [r['id'] for r in refs]!=[r['id'] for r in requests]:raise PermissionError('No filtering/refill/duplicate references')
 selected=[];teacher={};prefix=requests[0]['body']['messages'][:-1]
 if len(prefix)!=17 or [r['role'] for r in prefix]!=['system']+['user','assistant']*8:raise PermissionError('Same fixed8 required')
 for r in requests:
  identifier=r['id'];ref=by[identifier];raw=load(bound(ref['source_result']));m=by_member[identifier]
  if digest(r['body'])!=r['body_sha256'] or r['body']['messages'][:-1]!=prefix or raw.get('body_sha256')!=r['body_sha256'] or any(raw.get(k)!=ref.get(k) for k in ('text','valid','finish_reason','prediction','mode')):raise PermissionError('Actual body-exact final response provenance mismatch')
  teacher[identifier]=final_text(ref,identifier)
  selected.append(clean_row(identifier,r['body']['messages'][-1]['content'],prefix[0],m['family']))
 nested=load(SELECT)['nested_N200'][route];nested_sources={r['id'] for r in nested}
 # Teacher public IDs differ from raw ordinal IDs; map through frozen request source.
 nested_ids=[r['id'] for r in requests if f"SGFormal-{route}-trainpool-{int(Path(r['source']).stem):05d}" in nested_sources]
 if len(nested_ids)!=200 or len(set(nested_ids))!=200 or collections.Counter(r['metadata']['family_index'] for r in selected if r['id'] in set(nested_ids))!={f:20 for f in range(10)}:raise PermissionError('Exact nested per-family200 required')
 return dict(selected=selected,teacher=teacher,fixed8=prefix,nested_ids=nested_ids),[bind(SELECT),bind(BODIES/'manifest.json'),bind(request_path),bind(member_path),bind(folder/'references-frozen.json'),rf['records'],rf['predictions'],bind(TEACHER/(route+'-qual')/'qualification-score.json'),*pf['predictions'],*[r['source_result'] for r in refs]]

def order_row(rows_,step):
 if len(rows_) not in (200,1000) or not 1<=step<=1000:raise PermissionError('Fixed N200x5/N1000x1 order only')
 return rows_[(step-1)%len(rows_)]

def specialized_nodes(n):
 tree=ast.parse((ROOT/'scripts/tinyfit_qwen_train16_20261025.py').read_text(encoding='utf-8'))
 node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run');before=copy.deepcopy(node)
 class Transform(ast.NodeTransformer):
  ranges=epochs=0
  def visit_Call(self,x):
   self.generic_visit(x)
   if isinstance(x.func,ast.Name) and x.func.id=='range' and len(x.args)==2 and isinstance(x.args[1],ast.Constant) and x.args[1].value==65:x.args[1]=ast.Constant(1001);self.ranges+=1
   return x
  def visit_keyword(self,x):
   self.generic_visit(x)
   if x.arg=='epoch' and ast.dump(x.value)==ast.dump(ast.parse('1+(step-1)//16',mode='eval').body):x.value=ast.parse('1+(step-1)//'+str(n),mode='eval').body;self.epochs+=1
   return x
 tr=Transform();node=tr.visit(node)
 if (tr.ranges,tr.epochs)!=(1,1):raise PermissionError('Expected exactly loop-bound/epoch adaptations')
 return before,ast.fix_missing_locations(node)

def seeded_setup(native,verify):
 node=next(x for x in ast.parse((ROOT/'scripts/train_ust_sg_native_gad_20261024.py').read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef) and x.name=='setup')
 count=0;probes=0
 for x in ast.walk(node):
  if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='guard' and x.func.attr=='probe':x.func=ast.Name(id='formal_setup_probe',ctx=ast.Load());x.args.append(ast.Name(id='checkpoint',ctx=ast.Load()));probes+=1
  if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ('manual_seed','manual_seed_all'):
   if len(x.args)!=1 or not isinstance(x.args[0],ast.Constant) or x.args[0].value!=20261024:raise PermissionError('Frozen seed call changed')
   x.args[0]=ast.parse("p['seed']",mode='eval').body;count+=1
 if (count,probes)!=(2,1):raise PermissionError('Exactly two seed calls and one resource callback adaptation required')
 ns=dict(native.t.lib.setup.__globals__,verify=verify,formal_setup_probe=formal_setup_probe)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/'scripts/train_ust_sg_native_gad_20261024.py'),'exec'),ns)
 return ns['setup']

def owned(out):
 raw=Path(out).absolute();p=raw.resolve()
 if raw!=p or p.is_symlink() or p.parent!=OUT or p.name not in {u['unit'] for u in matrix()}:raise PermissionError('Exact18-unit owned output required')
 return p

def verify(out,grant=True):
 out=owned(out);p=load(out/'protocol.json');u=next(u for u in check_matrix() if u['unit']==out.name)
 if p['unit']!=u or p['unit_path']!=str(out) or p['maximum_optimizer_attempts']!=1000 or p['milestones']!=MILESTONES or p['seed']!=u['seed'] or p['lr']!=1e-4:raise PermissionError('Frozen student recipe mismatch')
 if (out/'OOM_STOP.json').exists() or (RUN/'students-OOM-STOP.json').exists():raise PermissionError('Two OOM phase stop')
 for s in p['bindings']:bound(s)
 d=load(bound(p['inputs']));ids=[r['id'] for r in d['selected']]
 if len(ids)!=u['N'] or len(set(ids))!=u['N'] or p['training_order']!=ids*(1000//u['N']):raise PermissionError('Exact full corpus/presentation order required')
 for r in d['selected']:final_text(d['teacher'][r['id']],r['id'])
 if grant:
  g=load(out/'authorization.json')
  if g.get('approved') is not True or g.get('protocol_sha256')!=sha(out/'protocol.json') or g.get('unit')!=out.name or g.get('actions')!=['fit-fixed1000'] or g.get('source_sha256')!=sha(__file__):raise PermissionError('Exact Root GPU unit authorization absent')
 return p

def prepare(route):
 check_matrix();native=old_sources();data,bindings=route_data(route)
 # Only cached tokenizers; no model/weight loading or inference.
 native.q.setup_native();from transformers import AutoTokenizer
 for n in (1000,200):
  d=copy.deepcopy(data);d.pop('nested_ids')
  if n==200:d['selected']=[r for r in d['selected'] if r['id'] in set(data['nested_ids'])];d['teacher']={r['id']:d['teacher'][r['id']] for r in d['selected']}
  dest=OUT/'inputs'/f'{route}-N{n}.json';once(dest,d)
  for size in (('08','2b') if n==1000 else ('08',)):
   original=ROOT/'runs/2026-10-25-overnight-onpolicy/training/SGQA'/size/'protocol.json';base=load(original);tok=AutoTokenizer.from_pretrained(base['base'],local_files_only=True,trust_remote_code=False)
   audits=[]
   for row in d['selected']:
    pre,target=native.probe.packing(tok,row,d['teacher'][row['id']],'matched8',d['fixed8']);audits.append(dict(id=row['id'],prompt_tokens=len(pre),target_tokens=len(target)))
   for u in matrix():
    if (u['route'],u['size'],u['N'])!=(route,size,n):continue
    out=OUT/u['unit'];once(out/'token-audit.json',dict(rows=audits,tokenizer_only=True,model_loads=0,output_targets='raw final teacher text including invalid; native EOS'))
    bs=bindings+[bind(dest),bind(original),bind(PROTOCOL),bind(__file__),bind(out/'token-audit.json')]+[bind(ROOT/p) for p in OLD_HASHES]
    p=dict(schema='SG-formal-native-SeqKD-fixed1000-v1',unit=u,unit_path=str(out),model=size,task='SGQA',seed=u['seed'],lr=1e-4,condition='matched8',base=base['base'],base_assets=base['base_assets'],runtime=base['runtime'],inputs=bind(dest),bindings=bs,maximum_optimizer_attempts=1000,milestones=MILESTONES,training_order=[r['id'] for r in d['selected']]*(1000//n),policy_ids=[r['id'] for r in d['selected']],teacher_final_only=True,no_teacher_filtering=True,gold_in_loss=False,thinking='off',native_vocabulary=True,initialization='fresh_original_independent',lora=dict(r=8,alpha=16,dropout=0),optimizer='AdamW',weight_decay=0,clip_norm=1,checkpoint_every=5,full_generations=2,actor_milestones=[200,500,1000],selection='fixed final1000; no Dev/bestseed',new_api_calls=0,Holdout_read=False)
    once(out/'protocol.json',p);verify(out,False)
 return dict(status='PREPARED_TOKENIZER_ONLY_NO_GPU',route=route,units=[u['unit'] for u in matrix() if u['route']==route])

def output_size():return sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
def resource_checks(s,start=False):
 c=dict(winram=s.get('windows_free_gib',-1)>=(24 if start else 20),c_disk=s.get('c_free_gib',-1)>=20,linux_disk=s.get('linux_free_gib',-1)>=20,phase_output=s.get('new_outputs_gib',99)<3,cgroup_cpu=s.get('cpu_threads')==4,cgroup_host=s.get('host_memory_max_gib',99)<=14)
 if start:c.update(gpu_free=s.get('gpu_free_gib',-1)>=11,gpu_quiet=0<=s.get('gpu_util_percent',101)<=10)
 return c

def runtime_snapshot():
 # Parent Root runtime supplies Windows measurements (native read-only probe), bounded45s.
 p=RUN/'runtime/resources.json';s=load(p);s=dict(s.get('snapshot',s))
 if not 0<=time.time()-s.get('epoch',0)<=45:raise PermissionError('Fresh formal Windows/resource receipt required')
 z=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip().splitlines()
 if len(z)!=1:raise PermissionError('Exactly one GPU required')
 free,util=map(float,z[0].split(','))
 s.update(new_outputs_gib=output_size()/2**30,linux_free_gib=shutil.disk_usage(RUN).free/2**30,gpu_free_gib=free/1024,gpu_util_percent=util)
 return s

def formal_setup_probe(out,checkpoint):
 # Called inside unchanged TRAIN global GPU flock, before any weight load.
 began=time.monotonic();stable=None
 while True:
  if STOP or (RUN/'STOP').exists() or (RUN/'students-OOM-STOP.json').exists() or (out/'stop.json').exists():raise PermissionError('Pause before model load')
  s=runtime_snapshot();checks=resource_checks(s,start=checkpoint is None)
  if checkpoint is not None:
   if not all(checks.values()):raise RuntimeError('Reload resource floor')
   return s
  now=time.monotonic();stable=(now if stable is None else stable) if all(checks.values()) else None
  if stable is not None and now-stable>=20:return s
  if now-began>=60:raise BlockingIOError('Locked quiet20s/WinRAM24/GPU11 launch gate not met')
  time.sleep(2)

def namespace(out):
 native=old_sources();u=load(out/'protocol.json')['unit'];live={};ns=dict(native.tiny.__dict__)
 lib=types.SimpleNamespace(**native.t.lib.__dict__);tn=types.SimpleNamespace(**native.t.__dict__);tn.lib=lib
 def rebind(f,g):return types.FunctionType(f.__code__,g,f.__name__,f.__defaults__)
 lib.setup=seeded_setup(native,verify)
 # Exact original serializer/restore, isolated from GAD monkeypatches.
 saver=rebind(native.tiny.SAVE_CHECKPOINT,dict(native.tiny.SAVE_CHECKPOINT.__globals__,unit_path=owned,RUN=RUN,lib=lib))
 stop=lambda:STOP or any(p.exists() for p in [RUN/'STOP',RUN/'stop.json',RUN/'root-stop.json',RUN/'students-OOM-STOP.json',out/'stop.json',out/'OOM_STOP.json'])
 def cursor(out):
  if native.tiny.unresolved(out):raise PermissionError('Unknown started transaction; no replay')
  ptr=load(out/'work/latest.json');folder=Path(ptr['path'])
  if folder.is_symlink() or folder.resolve().parent!=(out/'work/checkpoints').resolve() or sha(folder/'manifest.json')!=ptr['manifest_sha256']:raise PermissionError('Checkpoint pointer mismatch')
  cm=lib.verify_checkpoint(out,folder);step=cm['step'];logs=[int(f.stem) for f in sorted((out/'work/updates').glob('*.json')) if '.started.' not in f.name]
  if cm['stage']!='FORMAL_SEQKD1000' or not 0<=step<=1000 or logs!=list(range(1,step+1)):raise PermissionError('Worklog ahead/behind checkpoint; no replay')
  return folder,step
 def delete_full(out,folder):
  base=out/'work/checkpoints'
  if folder.is_symlink() or folder.resolve().parent!=base.resolve():raise PermissionError('Retention path escape')
  cm=lib.verify_checkpoint(out,folder);files={str(f.relative_to(folder)) for f in folder.rglob('*') if f.is_file()}
  if files!=set(cm['files'])|{'manifest.json'} or any(f.is_symlink() for f in folder.rglob('*')):raise PermissionError('Unexpected checkpoint payload')
  once(out/'work/tombstones'/(folder.name+'.json'),dict(manifest=cm,manifest_sha256=sha(folder/'manifest.json'),reason='latest2 full; actor milestones retained'))
  shutil.rmtree(folder)
 def retain(out):
  generations=sorted((out/'work/checkpoints').iterdir())
  for folder in generations[:-2]:delete_full(out,folder)
 def save_boundary(out,step,torch,model,head,opt,metric=None):
  if not all(resource_checks(runtime_snapshot()).values()):raise RuntimeError('Formal runtime resource floor/cap')
  folder=out/'work/checkpoints'/f'step{step:04d}';save=step%5==0 or step in MILESTONES or stop()
  if save:
   estimate=sum(v.numel()*v.element_size() for name,v in model.named_parameters() if 'lora_' in name)+sum(v.numel()*v.element_size() for st in opt.state.values() for v in st.values() if hasattr(v,'numel'))+64*2**20
   if output_size()+estimate>3*2**30:raise RuntimeError('WHOLE_NEW_PHASE_CHECKPOINT_STAGING_3GIB_CAP')
   old=[load(f) for f in (out/'work/updates').glob('*.json') if '.started.' not in f.name];actual=sum(r['optimizer_updated'] for r in old)+int(metric['optimizer_updated'] if metric else 0)
   saver(out,folder,torch,model,head,dict(actor=opt),'FORMAL_SEQKD1000',step,actual_updates=actual);lib.verify_checkpoint(out,folder)
   if step in (200,500,1000):
    dest=out/'milestones'/f'step{step:04d}'
    if dest.exists():raise PermissionError('Milestone already exists')
    if output_size()+sum(p.stat().st_size for p in (folder/'actor').rglob('*') if p.is_file())>3*2**30:raise RuntimeError('Milestone whole-phase3GiB cap')
    shutil.copytree(folder/'actor',dest/'actor');once(dest/'manifest.json',dict(step=step,actor_sha256=lib.role_hash(model,'actor'),source_manifest_sha256=sha(folder/'manifest.json'),files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))
  if metric is not None:once(out/'work/updates'/f'{step:04d}.json',metric)
  if save:atomic(out/'work/latest.json',dict(path=str(folder),manifest_sha256=sha(folder/'manifest.json')));retain(out)
  live.update(step=step,torch=torch,model=weakref.ref(model),head=weakref.ref(head),opt=weakref.ref(opt))
  return folder if save else Path(load(out/'work/latest.json')['path'])
 def stopped(out):
  if stop() and live.get('step',0)>0:
   ptr=load(out/'work/latest.json');cm=lib.verify_checkpoint(out,Path(ptr['path']))
   if cm['step']<live['step']:save_boundary(out,live['step'],live['torch'],*[live[k]() for k in ('model','head','opt')])
  return stop()
 def reload_saved(out,p,folder):
  # Existing independent loader checks parameter and optimizer hashes; additionally prove restored RNG equality.
  f=rebind(native.tiny.reload_saved,dict(native.tiny.reload_saved.__globals__,t=tn));torch,tok,model,head,opt=f(out,p,folder)
  rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
  if not torch.equal(torch.get_rng_state(),rng['cpu']) or len(torch.cuda.get_rng_state_all())!=len(rng['cuda']) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda'])):raise PermissionError('Independent RNG restore differs')
  return torch,tok,model,head,opt
 def evaluate_milestone(out,p,step,*_):
  once(out/'measurements'/f'step{step:04d}'/'checkpoint-only.json',dict(step=step,evaluation=False,selection='fixed1000',Holdout_read=False));return True
 def summarize(out):
  p=verify(out);folder,step=cursor(out);logs=[load(f) for f in sorted((out/'work/updates').glob('*.json')) if '.started.' not in f.name]
  if step!=1000 or len(logs)!=1000 or [r['id'] for r in logs]!=p['training_order'] or sum(r['optimizer_updated'] for r in logs)!=1000 or not (out/'measurements/step1000/independent-reload.json').exists():raise PermissionError('Require1000 actual completed updates and independent final reload; no fake completion')
  z=dict(status='TRAINING_COMPLETE_FIXED1000',unit=p['unit'],presentations=1000,actual_optimizer_updates=1000,selected_step=1000,endpoint=bind(out/'milestones/step1000/manifest.json'),checkpoint_manifest_sha256=sha(folder/'manifest.json'),student_metrics='NOT_MEASURED',Holdout_read=False,API_calls=0,final_reload=bind(out/'measurements/step1000/independent-reload.json'))
  once(out/'training-complete.json',z);return z
 ns.update(owned=owned,verify=verify,bound=bound,bind=bind,stopped=stopped,cursor=cursor,order_row=order_row,save_boundary=save_boundary,reload_saved=reload_saved,evaluate_milestone=evaluate_milestone,summarize=summarize,t=tn,MILESTONES=MILESTONES)
 _,node=specialized_nodes(u['N']);exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'scripts/tinyfit_qwen_train16_20261025.py'),'exec'),ns)
 ns['delete_full']=delete_full;ns['retain']=retain
 return ns

def run(unit,resume=False):
 out=owned(OUT/unit);verify(out)
 if os.environ.get('FORMAL_OFFLINE_CHILD')!='1' or os.environ.get('HF_HUB_OFFLINE')!='1':raise PermissionError('Reviewed offline/cgroup parent required')
 # TRAIN itself owns the exact historical global GPU lock and verifies cgroup limits.
 # Parent must hold a bounded20s quiet receipt fresh at child launch; no lock bypass.
 s=runtime_snapshot()
 if not all(resource_checks(s,True).values()) or s.get('gpu_quiet_seconds',0)<20:raise PermissionError('WinRAM24/GPU11/quiet20sec launch gate')
 ns=namespace(out)
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 try:return ns['run'](out,resume=resume)
 except Exception as e:
  oom=type(e).__name__=='OutOfMemoryError' or 'out of memory' in str(e).lower()
  once(out/'failures'/f'{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),OOM=oom,unknown_started=old_sources().tiny.unresolved(out),no_auto_retry=True))
  if sum(load(f).get('OOM',False) for f in OUT.glob('SGQA-*/failures/*.json'))>=2:once(RUN/'students-OOM-STOP.json',dict(reason='TWO_OOM_STOP_FORMAL_STUDENT_PHASE'))
  if sum(load(f).get('OOM',False) for f in (out/'failures').glob('*.json'))>=2:once(out/'OOM_STOP.json',dict(reason='TWO_OOM_AFFECTED_UNIT_STOP'))
  raise

def request_stop(*_):
 global STOP
 STOP=True

def queue_plan():
 units=check_matrix();old_sources();PREP.mkdir(parents=True,exist_ok=True)
 z=dict(schema='formal-student18-queue-v1',status='PREPARED_NOT_TRAINING_AUTHORIZATION',units=units,bindings=[bind(PROTOCOL),bind(SELECT),bind(BODIES/'manifest.json'),bind(__file__)]+[bind(ROOT/p) for p in OLD_HASHES],total_fits=18,size_counts=dict(collections.Counter(u['size'] for u in units)),optimization_seeds=list(SEEDS),independent_generator_draws=1,shared_teacher_routes=2,teacher_cache='exact same1000 body cache shared across size/seed;200 nested',actual_optimizer_updates=0,API_calls=0,GPU=False,Holdout_read=False,next='wait actual teacher freeze; prepare route; source review; exact Root unit grant; controlled GPU child')
 once(PREP/'queue-plan.json',z);return z

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','prepare','verify','run','resume-check']);p.add_argument('--route',choices=['markov','control']);p.add_argument('--unit',choices=[u['unit'] for u in matrix()]);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.action=='plan':z=queue_plan()
 elif a.action=='prepare':z=prepare(a.route)
 elif a.action=='verify':z=verify(OUT/a.unit,False)
 elif a.action=='resume-check':out=owned(OUT/a.unit);verify(out);folder,step=namespace(out)['cursor'](out);z=dict(status='SAFE_COMPLETE_BOUNDARY',step=step,checkpoint=bind(folder/'manifest.json'),GPU=False)
 else:z=run(a.unit,a.resume)
 print(json.dumps(z,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
