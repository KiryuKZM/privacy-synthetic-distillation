"""New matched 2B W/CE/actual-current-actor GRPO, using frozen native primitives.
IPD is deliberately unavailable until the actual latest Qwen probability backend qualifies.
No external-model transport or protected evaluation data is imported here.
"""
from pathlib import Path
import argparse,ast,contextlib,copy,fcntl,gc,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,types
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
O=R/'runs/2026-10-31-new-generator-three-method-comparison';S=O/'students-v2';P=O/'student-engine-v2'
TEACHER=O/'teacher-training-view-v2';PLAN=O/'experiment-plan.v5.json'
SEEDS=(2026102751,2026102752,2026102753);GENS=('seed01','seed02');STAGES=('warm','off','on')
STOP=False;STARTUP_READY=None
from privacy_synth.choice_answer import parse_choice
from spatial_task_packing_v3_20261031 import packing
import sg_formal_student_executor_20261027 as core
native=core.old_sources();lib=native.t.lib

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind(p):
    p=Path(p);return dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def once(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)
def stop(*_):
    global STOP;STOP=True
def stopped():return STOP or (O/'STOP').exists() or (R/'runs/STOP').exists()
def check_stop():
    if stopped():raise InterruptedError('User owned STOP')
def unit(gen,seed,stage):
    if gen not in GENS or seed not in SEEDS or stage not in (*STAGES,'ipd'):raise PermissionError('Exact frozen unit only')
    return S/gen/str(seed)/stage

def ordered(data,ids,stage):
    rows=data['selected'];by={x['id']:x for x in rows}
    assert len(rows)==288 and len(by)==288
    if stage=='warm':return rows
    assert stage in ('off','on','ipd') and len(ids)==len(set(ids))==50 and set(ids)<=by.keys()
    return [by[k] for k in ids]

def verify(out):
    out=Path(out).resolve();p=load(out/'protocol.json')
    assert out==unit(p['generator'],p['seed'],p['stage']).resolve()
    assert p['lora']==dict(r=8,alpha=16,dropout=0) and p['native_vocabulary'] and not p['thinking']
    assert p['lr']==(1e-5 if p['stage']=='on' else 1e-4)
    assert p['maximum_optimizer_attempts']==(288 if p['stage']=='warm' else 50)
    assert p['supervision']=='actual_DS_final_only' and not p['oracle_or_real_gold_in_loss']
    assert bind(R/p['inputs']['path'])==p['inputs']
    for name,info in p['base_asset_stat'].items():
        st=(Path(p['base'])/name).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==tuple(info)
    return p

def phase_bytes():return sum(p.stat().st_size for p in S.rglob('*') if p.is_file()) if S.exists() else 0

def snapshot(start=False):
    check_stop();script=r'\\wsl.localhost\Ubuntu-24.04\home\kiryu\privacy-synthetic-distillation\scripts\three_method_synthetic_cpu_v4_20261101.py'
    z=subprocess.run(['/mnt/c/Users/10950/miniconda3/python.exe','-X','utf8',script,'windows-probe'],capture_output=True,text=True,timeout=10)
    assert z.returncode==0,z.stderr[:100];w=json.loads(z.stdout)
    gpu=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
    free,util,temp=map(float,gpu.stdout.strip().split(','))
    name=next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    cg=Path('/sys/fs/cgroup')/name.lstrip('/');shared=next(x for x in cg.parents if x.name=='sg-formal-20261027.slice')
    quota,period=(cg/'cpu.max').read_text().split();assert quota!='max' and int(quota)/int(period)<=4
    for x in (cg,shared):assert int((x/'memory.max').read_text())<=14*2**30 and (x/'memory.swap.max').read_text().strip()=='0'
    cgdata=dict(current=int((cg/'memory.current').read_text()),peak=int((cg/'memory.peak').read_text()),swap=int((cg/'memory.swap.current').read_text()))
    checks=dict(Win=w['windows_free_gib']>=(30 if start else 20),disk=min(w['c_free_gib'],shutil.disk_usage(O).free/2**30)>=20,
        output=phase_bytes()<3*2**30,noSwap=cgdata['swap']==0,temperature=temp<80)
    if start:checks.update(GPU_free=free>=11264,GPU_quiet=0<=util<=10)
    sample=dict(epoch=time.time(),Windows=w,GPU=dict(free_MiB=free,util=util,temp_C=temp),cgroup=cgdata,checks=checks,start=start)
    atomic(P/'resource-latest.json',sample)
    if not all(checks.values()):raise BlockingIOError('Measured resource boundary: '+json.dumps(checks))
    return sample

def preflight():
    begin=time.monotonic();stable=None
    while True:
        check_stop();now=time.monotonic()
        try:s=snapshot(True);stable=now if stable is None else stable
        except BlockingIOError:stable=None
        if stable is not None and now-stable>=20:
            global STARTUP_READY
            STARTUP_READY=time.monotonic()
            return s
        if now-begin>=120:raise TimeoutError('Finite actual GPU/host preflight, no model replay')
        time.sleep(.5)

def setup_probe(out,checkpoint):
    # Reuse an actual just-completed20s qualification, then remeasure before weights.
    # This avoids performing the same quiet20 wait twice in a single locked load.
    s=snapshot(True) if STARTUP_READY is not None and time.monotonic()-STARTUP_READY<3 else preflight()
    return dict(windows_free_gib=s['Windows']['windows_free_gib'],c_free_gib=s['Windows']['c_free_gib'],
        linux_free_gib=shutil.disk_usage(O).free/2**30,gpu_free_gib=s['GPU']['free_MiB']/1024)

def setup(out,checkpoint=None):
    f=core.seeded_setup(native,verify);f.__globals__['formal_setup_probe']=setup_probe
    return f(out,checkpoint)

def cooldown():
    tick=time.monotonic();s=snapshot()
    if s['GPU']['temp_C']<74:return 0.
    while s['GPU']['temp_C']>68:
        if time.monotonic()-tick>=120:raise TimeoutError('Bounded 74to68 cooling')
        time.sleep(.5);s=snapshot()
    return time.monotonic()-tick

def policy_function():
    path=R/'scripts/train_sg_teacher_label_grpo_20261025.py';node=next(x for x in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(x,ast.FunctionDef) and x.name=='policy_update')
    ns=dict(t=types.SimpleNamespace(lib=lib));exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::exact-policy-update','exec'),ns)
    return ns['policy_update']

def ce_step(torch,tok,model,head,opt,row,target):
    before=lib.role_hash(model,'actor')
    if target['text'] is None:
        return dict(optimizer_updated=False,missing_teacher_final=True,actor_before=before,actor_after=before,input_tokens=0,target_tokens=0,loss=None,gradient_norm=0.)
    assert isinstance(target['text'],str) and target.get('available',True)
    prefix,answer=packing(tok,row['messages'],target['text'])
    if len(prefix)+len(answer)>16384:raise RuntimeError('No context clipping or task replacement')
    lib.activate(model,head,'actor');model.train();opt.zero_grad(set_to_none=True)
    ids=torch.tensor([prefix+answer],device='cuda')
    loss=-lib.response_log_probs(lib.hidden(model,ids),model.get_output_embeddings().weight,ids,len(prefix),temperature=1.).mean()
    loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite CE loss/gradient')
    updated=float(norm)>0
    if updated:opt.step()
    opt.zero_grad(set_to_none=True);after=lib.role_hash(model,'actor')
    if updated and before==after:raise RuntimeError('Nonzero CE gradient without actor change')
    return dict(optimizer_updated=updated,missing_teacher_final=False,actor_before=before,actor_after=after,
        input_tokens=len(prefix),target_tokens=len(answer),loss=float(loss.detach()),gradient_norm=float(norm))

def on_step(out,step,torch,tok,model,head,opt,row,target):
    records=[];sample_tick=time.monotonic();before=lib.role_hash(model,'actor')
    for i in range(8):
        check_stop();r=lib.rollouts(torch,tok,model,head,row,n=1,greedy=False,thinking='off',messages=row['messages'])[0]
        r['group_index']=i;records.append(r)
    torch.cuda.synchronize();sample_seconds=time.monotonic()-sample_tick
    if lib.role_hash(model,'actor')!=before:raise RuntimeError('Sampling mutated actor')
    teacher=parse_choice(target['text']) if target['text'] is not None else None
    predictions=[parse_choice(x['content']) for x in records]
    rewards=[float(teacher is not None and pred==teacher) for pred in predictions]
    once(out/'rollouts'/f'{step:04d}.json',dict(id=row['id'],actor_before=before,records=records,
        teacher_reference=target['source_result'],teacher_choice=teacher,rewards=rewards,missing_teacher_final=teacher is None,
        fresh_current_actor=True,temperature=.8,output_cap=256,no_oracle_reward=True,sampling_seconds=sample_seconds))
    metric,saved=policy_function()(torch,model,head,opt,records,rewards,
        before_update=lambda v:once(out/'policy-probabilities'/f'{step:04d}.json',dict(id=row['id'],**v)))
    metric.update(actor_before=before,actor_after=lib.role_hash(model,'actor'),sampling_seconds=sample_seconds,
        input_tokens=sum(r['prompt_length'] for r in records),target_tokens=sum(len(r['response_ids']) for r in records),
        rollout_count=8,teacher_missing=teacher is None,gold_or_oracle_in_reward=False)
    return metric

def save(out,step,actual,torch,model,head,opt):
    estimate=sum(p.numel()*p.element_size() for n,p in model.named_parameters() if 'lora_' in n)+64*2**20
    estimate+=sum(x.numel()*x.element_size() for state in opt.state.values() for x in state.values() if hasattr(x,'numel'))
    if phase_bytes()+estimate>3*2**30:raise RuntimeError('Whole student output3GiB before save')
    folder=out/'checkpoints'/f'step{step:04d}'
    native.tiny.SAVE_CHECKPOINT(out,folder,torch,model,head,dict(actor=opt),'NEWGEN_'+verify(out)['stage'].upper(),step,actual_updates=actual)
    manifest=lib.verify_checkpoint(out,folder)
    once(out/'checkpoint-pointers'/f'{step:04d}.json',dict(folder=str(folder),manifest=bind(folder/'manifest.json')))
    folders=sorted((out/'checkpoints').glob('step*'))
    for old in folders[:-2]:
        if old.is_symlink() or old.resolve().parent!=(out/'checkpoints').resolve():raise PermissionError('Retention path escape')
        cm=lib.verify_checkpoint(out,old);files={str(p.relative_to(old)) for p in old.rglob('*') if p.is_file()}
        if files!=set(cm['files'])|{'manifest.json'} or any(p.is_symlink() for p in old.rglob('*')):raise PermissionError('Unexpected payload')
        once(out/'checkpoint-tombstones'/(old.name+'.json'),dict(manifest=cm,source_manifest_SHA=sha(old/'manifest.json'),reason='latest2 verified full only'))
        shutil.rmtree(old)
    return folder,manifest

def authorize(out,action):
    p=verify(out);g=load(P/'root-grant.json');protocol=load(P/'protocol.json')
    assert g['approved'] and g['source']==bind(__file__) and g['protocol']==bind(P/'protocol.json')
    assert str(out.relative_to(R)) in g['allowed_units'] and action in g['actions']
    for b in protocol['bindings']:assert bind(R/b['path'])==b,b['path']
    check_stop();return p

def model_start(out,p):
    if p['stage']=='warm':return setup(out)
    w=unit(p['generator'],p['seed'],'warm');terminal=load(w/'fit-terminal.json')
    assert load(w/'independent-reload.json')['status']=='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS'
    folder=Path(terminal['checkpoint']);cm=lib.verify_checkpoint(w,folder)
    torch,tok,model,head=setup(out,folder)
    assert {r:lib.role_hash(model,r) for r in cm['role_hashes']}==cm['role_hashes']
    # Branch source is verified against W's protocol. Never restore W optimizer under a different branch protocol.
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    if p['stage']=='on':
        state={k:v.detach().clone() for k,v in get_peft_model_state_dict(model,adapter_name='actor').items()}
        set_peft_model_state_dict(model,state,adapter_name='reference')
        assert lib.role_hash(model,'reference')==lib.role_hash(model,'actor')
    rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    once(out/'warm-source.json',dict(source_manifest=bind(folder/'manifest.json'),W_actor=cm['role_hashes']['actor'],
        branch_actor=lib.role_hash(model,'actor'),new_optimizer=True,reference_W_clone=p['stage']=='on',W_not_retrained=True))
    return torch,tok,model,head

def fit(out):
    p=authorize(out,'fit');data=load(R/p['inputs']['path']);rows=ordered(data,p['post50_ids'],p['stage'])
    if (out/'fit-started.json').exists():raise PermissionError('No implicit optimizer or unknown stage replay')
    preflight();once(out/'fit-started.json',dict(epoch=time.time(),source=bind(__file__),scheduled=len(rows)))
    start=time.monotonic();torch,tok,model,head=model_start(out,p);lib.activate(model,head,'actor');setup_seconds=time.monotonic()-start
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=p['lr'],weight_decay=0,foreach=False)
    assert not opt.state
    initial={r:lib.role_hash(model,r) for r in ('actor','reference','discriminator')};once(out/'initial-role-hashes.json',dict(role_hashes=initial,seed=p['seed']))
    actual=0;folder,_=save(out,0,0,torch,model,head,opt);begin=time.monotonic();compute=pace=cool=checkpoint_seconds=0.
    last_committed=0
    try:
        for step,row in enumerate(rows,1):
            if stopped():
                if not (out/'checkpoints'/f'step{step-1:04d}').exists():save(out,step-1,actual,torch,model,head,opt)
                once(out/'paused.json',dict(complete_scheduled=step-1,actual_updates=actual));return
            if time.monotonic()-begin>(7200 if p['stage']=='on' else 3600):raise TimeoutError('Bounded fixed worker wall')
            cool+=cooldown();tick=time.monotonic();time.sleep(.5);pace+=time.monotonic()-tick
            file=out/'updates'/f'{step:04d}.json';once(file.with_suffix('.started.json'),dict(step=step,id=row['id'],actor_before=lib.role_hash(model,'actor')))
            tick=time.monotonic();metric=(on_step(out,step,torch,tok,model,head,opt,row,data['teacher'][row['id']]) if p['stage']=='on'
                else ce_step(torch,tok,model,head,opt,row,data['teacher'][row['id']]))
            torch.cuda.synchronize();metric['compute_seconds']=time.monotonic()-tick;compute+=metric['compute_seconds']
            assert all(lib.role_hash(model,r)==initial[r] for r in ('reference','discriminator'))
            actual+=int(metric['optimizer_updated']);metric.update(step=step,id=row['id'],oracle_or_real_gold_in_loss=False,scheduled_not_necessarily_optimizer_update=True)
            metric['checkpoint_seconds']=0.
            if step%5==0 or step==len(rows) or stopped():
                tick=time.monotonic();folder,_=save(out,step,actual,torch,model,head,opt);metric['checkpoint_seconds']=time.monotonic()-tick;checkpoint_seconds+=metric['checkpoint_seconds']
            once(file,metric);last_committed=step;atomic(P/'progress.json',dict(unit=str(out.relative_to(R)),scheduled_committed=step,scheduled_total=len(rows),actual_updates=actual,epoch=time.time(),kind='real_optimizer_task_commit'))
            if step%25==0:print('ACTUAL_STUDENT_TASK',out.name,step,'OPTIMIZER',actual,flush=True)
    except InterruptedError:
        if not (out/'checkpoints'/f'step{last_committed:04d}').exists():save(out,last_committed,actual,torch,model,head,opt)
        once(out/'paused.json',dict(complete_scheduled=last_committed,actual_updates=actual,partial_started_preserved=True))
        return
    once(out/'fit-terminal.json',dict(status='FIXED_SCHEDULE_COMPLETE_RELOAD_PENDING',scheduled=len(rows),actual_updates=actual,
        checkpoint=str(folder),manifest=bind(folder/'manifest.json'),final_actor=lib.role_hash(model,'actor'),
        setup_seconds=setup_seconds,fit_wall_seconds=time.monotonic()-begin,compute_seconds=compute,pacing_seconds=pace,
        cooling_seconds=cool,checkpoint_seconds=checkpoint_seconds,CUDA_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
        CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),CPU_RSS_peak_KiB=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
        no_Dev_selection=True,API_calls=0))

def reload(out):
    p=authorize(out,'reload');terminal=load(out/'fit-terminal.json');folder=Path(terminal['checkpoint'])
    assert terminal['status']=='FIXED_SCHEDULE_COMPLETE_RELOAD_PENDING'
    if (out/'reload-started.json').exists():raise PermissionError('No unknown reload replay')
    preflight();once(out/'reload-started.json',dict(epoch=time.time(),manifest=bind(folder/'manifest.json')))
    tick=time.monotonic();torch,tok,model,head=setup(out,folder);lib.activate(model,head,'actor')
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=p['lr'],weight_decay=0,foreach=False)
    cm=lib.restore(torch,model,head,dict(actor=opt),out,folder)
    rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
    assert torch.equal(torch.get_rng_state(),rng['cpu']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda']))
    assert cm['actual_updates']==terminal['actual_updates'] and cm['role_hashes']['actor']==terminal['final_actor']
    once(out/'independent-reload.json',dict(status='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS',manifest=bind(folder/'manifest.json'),
        step=cm['step'],actual_updates=cm['actual_updates'],reload_seconds=time.monotonic()-tick,no_new_optimizer_updates=True))

def prepare():
    assert load(TEACHER/'terminal.json')['status']=='TRAINING_VIEW576_FINAL_AVAILABILITY_FROZEN_NO_NEW_SCORE'
    P.mkdir(exist_ok=True);assert not (P/'protocol.json').exists(),'No reprepare frozen engine'
    base=load(R/'runs/2026-10-25-overnight-onpolicy/training/SGQA/2b/protocol.json')
    ids=load(O/'teacher-labels-v1/post50-task-IDs-frozen.json');units=[]
    stats={name:[(Path(base['base'])/name).stat().st_size,(Path(base['base'])/name).stat().st_ino,(Path(base['base'])/name).stat().st_mtime_ns] for name in base['base_assets']}
    for gen in GENS:
        corpus=TEACHER/('train-corpus-'+gen+'.local.json');data=load(corpus);assert len(data['selected'])==288
        for seed in SEEDS:
            for stage in STAGES:
                out=unit(gen,seed,stage);out.mkdir(parents=True,exist_ok=False)
                p=dict(generator=gen,seed=seed,stage=stage,unit_path=str(out),base=base['base'],base_assets=base['base_assets'],base_asset_stat=stats,
                    runtime=base['runtime'],inputs=bind(corpus),post50_ids=ids[gen],lr=1e-5 if stage=='on' else 1e-4,
                    maximum_optimizer_attempts=288 if stage=='warm' else 50,lora=dict(r=8,alpha=16,dropout=0),native_vocabulary=True,thinking=False,
                    supervision='actual_DS_final_only',oracle_or_real_gold_in_loss=False,checkpoint_every=5,latest2full=True,
                    fixed_endpoint_not_Dev_selected=True,new_API_calls=0)
                once(out/'protocol.json',p);units.append(out)
    paths=[Path(core.__file__),*[R/p for p in core.OLD_HASHES],R/'scripts/bird_gad_night_20261022.py',R/'scripts/train_sg_teacher_label_grpo_20261025.py',
        R/'src/privacy_synth/onpolicy_gad.py',R/'src/privacy_synth/choice_answer.py',R/'scripts/spatial_task_packing_20261031.py',
        R/'scripts/spatial_task_packing_v3_20261031.py',R/'scripts/three_method_synthetic_cpu_v4_20261101.py',PLAN,
        TEACHER/'terminal.json',TEACHER/'all-raw-frozen.json',O/'teacher-labels-v1/post50-task-IDs-frozen.json',
        *[TEACHER/('train-corpus-'+g+'.local.json') for g in GENS],*[u/'protocol.json' for u in units]]
    once(P/'protocol.json',dict(source=bind(__file__),bindings=[bind(p) for p in paths],units=[str(u.relative_to(R)) for u in units],
        status='W_CE_ACTUAL_GRPO_ENGINE_PREPARED_NO_GPU_GRANT',no_new_generator_fit=True,latest_IPD_requires_actual_backend_qualification=True,
        original_student_update_cap_including_future_IPD=2628,resources='singleGPU/CPU4host14noSwap/CUDA9/Win30run20/quiet20/free11/temp80/disk20/output3GiB'))
    print('ACTUAL18_W_CE_GRPO_UNIT_PROTOCOLS_FROZEN_GPU_NOT_STARTED')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','fit','reload']);parser.add_argument('--generator',choices=GENS);parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--stage',choices=STAGES);args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.action=='prepare':prepare()
    else:
        out=unit(args.generator,args.seed,args.stage)
        try:
            # The lock belongs to this worker, never an unrelated application.
            with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                (fit if args.action=='fit' else reload)(out)
        except BaseException as exc:
            once(out/(args.action+'-failure.json'),dict(error_type=type(exc).__name__,message=str(exc),epoch=time.time()));raise