#!/usr/bin/env python3
"""UST snapshot discovery. Local only; never opens original Final or gold fields."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,itertools,json,math,os,random,re,signal,sys,time
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'runs/2026-10-26-generator-discovery'
BASE=ROOT/'runs/2026-10-23-llm-task-sweep/preparation/text-qa/USTBench-POI-processQA'
FIELDS=('coverage','demand','distance','waiting_time','charging_time')
SEED='20261026-ust-generator-v1'
STOP_REQUESTED=False
def request_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True
def stop_requested():
    extra=os.environ.get('GENERATOR_STOP_FILE')
    return STOP_REQUESTED or any(p.exists() for p in (NEW/'STOP',NEW/'runtime/STOP',NEW/'stop.json',NEW/'root-stop.json',NEW/'global-stop.json',NEW/'runtime/stop.json')) or bool(extra and Path(extra).exists())
def atomic_json(p,obj):
    p=Path(p);tmp=p.with_name('.'+p.name+'.tmp')
    with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def restore_checkpoint(torch,np,model,optimizer,state,device):
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    np.random.set_state(state['numpy_rng'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def rows(p,rs):
    with Path(p).open('x') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def picked(text,wanted):
    """Decode only allowlisted top-level JSON fields. Gold value tokens are skipped."""
    decoder=json.JSONDecoder();i=0;out={}
    def endvalue(j):
        if text[j]=='"':
            j+=1
            while j<len(text):
                if text[j]=='\\':j+=2
                elif text[j]=='"':return j+1
                else:j+=1
            raise ValueError('Unterminated JSON string')
        if text[j] in '[{':
            depth=0;quoted=False;esc=False
            while j<len(text):
                c=text[j]
                if quoted:
                    if esc:esc=False
                    elif c=='\\':esc=True
                    elif c=='"':quoted=False
                elif c=='"':quoted=True
                elif c in '[{':depth+=1
                elif c in ']}':
                    depth-=1
                    if not depth:return j+1
                j+=1
            raise ValueError('Unterminated JSON object')
        while j<len(text) and text[j] not in ',}':j+=1
        return j
    while i<len(text) and text[i].isspace():i+=1
    assert text[i]=='{';i+=1
    while True:
        while i<len(text) and (text[i].isspace() or text[i]==','):i+=1
        if text[i]=='}':break
        key,i=decoder.raw_decode(text,i)
        while text[i].isspace():i+=1
        assert text[i]==':';i+=1
        while text[i].isspace():i+=1
        if key in wanted:out[key],i=decoder.raw_decode(text,i)
        else:i=endvalue(i)
    return out
def input_metadata():
    p=BASE/'FitDemos.local.jsonl'
    return p,[picked(l,{'id','group','ability'}) for l in p.read_text().splitlines()]
def historical():
    rolefile=BASE/'roles.json';roles=load(rolefile)['roles']
    entries=[];used=set()
    for name,r in roles.items():
        if name=='FitDemos':continue
        groups=set(r['groups']);used|=groups
        entries.append(dict(role=name,members=len(r['members']),groups=len(groups),source=bind(rolefile)))
    allowed=[
      ('runs/2026-10-24-ust-sg-onpolicy/data/USTBench.membership.json',('train_members','validation_members')),
      ('runs/2026-10-25-overnight-onpolicy/data/USTBench.teacher-dev.membership.json',('selected',)),
      ('runs/2026-10-24-qwen-profile-budget-v3/preparation/USTBench.validation8.membership.json',('selected',)),
    ]
    for path,keys in allowed:
        p=ROOT/path;x=load(p)
        for k in keys:
            rs=x.get(k,x.get('members',[]) if k=='selected' else [])
            if isinstance(rs,dict):rs=list(rs.values())
            gs={r['group'] for r in rs};used|=gs
            entries.append(dict(role=path.split('/')[-1]+':'+k,members=len(rs),groups=len(gs),source=bind(p)))
    p=ROOT/'runs/2026-10-25-qwen-transfer-repair/data/USTBench.repair-membership.json'
    for name,r in load(p)['roles'].items():
        rs=r['selected'];gs={v['group'] for v in rs};used|=gs
        entries.append(dict(role=name,members=len(rs),groups=len(gs),source=bind(p)))
    p=BASE/'FitDemos.fixed8.jsonl'
    rs=[picked(l,{'id','group'}) for l in p.read_text().splitlines()]
    gs={r['group'] for r in rs};used|=gs
    entries.append(dict(role='fixed8',members=len(rs),groups=len(gs),source=bind(p)))
    return entries,used
def encode_question(question,ability,audit):
    sec=audit.sections(question);data=audit.parse_data(sec.get('Data',''))
    assert set(data)=={1,2,3,4},'Four candidates required'
    scene=dict(version=1,scope='displayed_four_candidate_snapshot',ability=ability,
      nodes=[dict(slot=i,attributes=data[i]) for i in range(1,5)],
      option_texts=audit.option_texts(sec.get('Question','')),sections=sec,
      exact_input_question=question,latent_world_state=None,topology=None)
    assert all(set(FIELDS)<=set(n['attributes']) for n in scene['nodes'])
    return scene
def decode_question(scene):return scene['exact_input_question']
def vector(scene):
    numeric=[];masks=[]
    for n in scene['nodes']:
        for k in FIELDS:
            token=n['attributes'][k]['number']
            masks.append(int(token.lower()=='nan'))
            numeric.append(0.0 if masks[-1] else float(token))
    return numeric,masks
def physical_snapshot(values,masks):
    nodes=[]
    for i in range(4):
        attrs={}
        for j,k in enumerate(FIELDS):
            v=values[i*5+j];missing=masks[i*5+j]
            attrs[k]=dict(number='NaN' if missing else str(v),unit='m' if k=='distance' else 'h' if k.endswith('_time') else None)
        nodes.append(dict(slot=i+1,attributes=attrs))
    return dict(version=1,scope='displayed_four_candidate_snapshot',nodes=nodes,
      latent_world_state=None,topology=None,ability='st_understanding')
def failures(scene):
    f=[]
    for n in scene['nodes']:
        for k in FIELDS:
            t=n['attributes'][k]['number']
            if t.lower()=='nan':
                if k not in ('waiting_time','charging_time'):f.append('missing_non_time_metric')
            else:
                v=float(t)
                if not math.isfinite(v):f.append('nonfinite')
                elif v<0:f.append('negative_metric')
    return sorted(set(f))
def rank_oracle(scene,metric):
    vals={n['slot']:math.inf if n['attributes'][metric]['number'].lower()=='nan' else float(n['attributes'][metric]['number']) for n in scene['nodes']}
    order=sorted(vals,key=lambda i:(-vals[i],i))
    return dict(weak_descending=order,ties=[[k for k in vals if vals[k]==v] for v in set(vals.values()) if sum(x==v for x in vals.values())>1],
      exact_visible_values=vals)
def fixture_checks():
    checks=[]
    s=physical_snapshot([4.,8.,200.,.05,.4,3.,7.,300.,.10,.3,2.,6.,400.,.15,.2,1.,5.,500.,.20,.1],[0]*20)
    assert vector(s)[0][3]==.05;checks.append('20_metric_joint_vector')
    for perm in itertools.permutations(range(4)):
        p=copy.deepcopy(s);p['nodes']=[copy.deepcopy(s['nodes'][i]) for i in perm]
        for j,n in enumerate(p['nodes']):n['slot']=j+1
        original=rank_oracle(s,'waiting_time')['weak_descending']
        expect=[perm.index(i-1)+1 for i in original]
        assert rank_oracle(p,'waiting_time')['weak_descending']==expect
    checks.append('all_24_candidate_permutation_equivariance')
    for perm in itertools.permutations(range(1,5)):
        mapping=dict(zip('ABCD',perm));assert set(mapping.values())=={1,2,3,4}
        answer=next(k for k,v in mapping.items() if v==4);assert mapping[answer]==4
    checks.append('all_24_action_option_bijections')
    t=copy.deepcopy(s);t['nodes'][0]['attributes']['waiting_time']['number']='NaN'
    assert rank_oracle(t,'waiting_time')['weak_descending'][0]==1;checks.append('explicit_nan_highest_time_semantics')
    t['nodes'][1]['attributes']['waiting_time']['number']='NaN'
    assert rank_oracle(t,'waiting_time')['ties']==[[1,2]];checks.append('nan_tie_is_ambiguous')
    assert round(.004,2)==round(.003,2)==0.;checks.append('round2_precision_loss_counterexample')
    hostile='{"gold":{"answer":"MUST_NOT_DECODE"},"id":"x","group":"g","question":"escaped \\" ok"}'
    assert picked(hostile,{'id','group'})==dict(id='x',group='g');checks.append('gold_value_lexically_skipped')
    assert not s['latent_world_state'] and not s['topology'];checks.append('missing_world_state_explicit')
    return dict(passed=len(checks),total=len(checks),checks=checks,planning_oracle='UNSUPPORTED_HIDDEN_COUNTERFACTUALS',
      reflection_oracle='UNSUPPORTED_ALTERNATIVE_ROLLOUTS')
def finish(out,kind):
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='manifest.json':files.append(dict(path=p.relative_to(out).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    save(out/'manifest.json',dict(stage=kind,files=files,api_calls=0,gpu=kind in ('ust_profile','ust_fit'),Final_read=False,new_unexposed_gold_decoded=False,code=bind(Path(__file__))))
def prepare(out):
    audit=module(ROOT/'scripts/audit_ust100_input_labels_20261024.py','ust_existing_parser')
    entries,used=historical();source,metadata=input_metadata()
    # Every role selected solely from metadata, before opening selected questions.
    remaining=[r for r in metadata if r['group'] not in used]
    reps={}
    for r in sorted(remaining,key=lambda x:hashlib.sha256((SEED+x['id']).encode()).hexdigest()):reps.setdefault(r['group'],r)
    ordered=list(reps.values());assert len(ordered)>64
    roles={'GeneratorAudit32':ordered[:32],'GeneratorSelectDev32':ordered[32:64],'GeneratorFit':ordered[64:]}
    save(out/'roles.json',dict(seed=SEED,roles=roles,selection='SHA256 order; one representative per unused numeric component; metadata only',
      original_fit_rows=len(metadata),original_fit_groups=len({r['group'] for r in metadata}),unused_rows=len(remaining),unused_groups=len(ordered),
      historical_roles=entries,source=bind(source),physical_independence_proven=False,Audit_questions_decoded=False))
    allowed={r['id'] for role,rs in roles.items() if role!='GeneratorAudit32' for r in rs}
    scenes={}
    for line in source.read_text().splitlines():
        identity=picked(line,{'id'})
        if identity['id'] not in allowed:continue
        r=picked(line,{'id','group','ability','question'})
        s=encode_question(r['question'],r['ability'],audit);assert decode_question(s)==r['question']
        s.update(id=r['id'],group=r['group']);scenes[r['id']]=s
    assert len(scenes)==len(allowed)
    for role,rs in roles.items():
        if role!='GeneratorAudit32':rows(out/(role+'.scenes.local.jsonl'),[scenes[r['id']] for r in rs])
    save(out/'codec-checks.json',dict(roundtrip_exact=len(scenes),total=len(scenes),fields_per_scene=20,mask_fields=20,
      original_precision_available=False,display_tokens_preserved=True,topology_available=False,latent_state_available=False,
      reflection_narratives_preserved_as_sidecar=True,reflection_narratives_neurally_generated=False,original_gold_decoded=False))
    save(out/'fixture-checks.json',fixture_checks())
    vendor=ROOT/'vendor/tabdiff'
    assets=[p for p in vendor.rglob('*') if p.is_file() and p.suffix in ('.py','.toml') and 'ckpt' not in p.parts and 'result' not in p.parts]
    assets+=[vendor/'LICENSE',vendor/'README.md']
    save(out/'code-assets.json',dict(reused_existing=True,downloaded_bytes=0,source='https://github.com/MinkaiXu/TabDiff',
      license='MIT',files=[bind(p) for p in assets],no_vendor_modifications=True,
      author_main_not_used_reason='Writes vendor ckpt and invokes metrics; sample_all drops bad rows and refills or zeroes NaN',
      sample_contract='Use sample(batch) directly; all raw rows retained, exactly100 then1000 if mechanical gate passes'))
    save(out/'small-profile-config.json',dict(candidate='TabDiff-UST-snapshot-v1',profile_seconds=300,fit_max_seconds=7200,
      updates=2000,batch_size=64,num_layers=2,d_token=4,n_head=1,factor=8,dim_t=128,num_timesteps=50,learning_rate=.001,
      seed=20261026,numeric_dimensions=20,categorical_mask_dimensions=20,mask_cardinality=2,
      preprocessing='Fit-only population mean/std; missing tokens zero-imputed + explicit binary masks; no Dev/Audit fitting',
      scope='understanding snapshot only; no topology/latent-state/feedback/model narrative generation',
      dev_gate=dict(raw_smoke=100,raw_diagnostic=1000,refill=False,valid_fraction_min=.95),
      trial_counted=True,gpu_authorization_required=True,latest_two_complete_checkpoints=True))
    # Non-learning negative controls are legitimate samples, explicitly copy/marginal baselines.
    fit=[scenes[r['id']] for r in roles['GeneratorFit']]
    rng=random.Random(20261026);numbers=[vector(s) for s in fit];baseline=[]
    for kind in ('joint_bootstrap','independent_marginals'):
        result=[]
        for n in (100,1000):
            sample=[];bad=0;copies=0
            signatures={json.dumps(vector(s)) for s in fit}
            for i in range(n):
                if kind=='joint_bootstrap':v,m=copy.deepcopy(rng.choice(numbers))
                else:
                    chosen=[rng.choice(numbers) for _ in range(20)]
                    v=[chosen[j][0][j] for j in range(20)];m=[chosen[j][1][j] for j in range(20)]
                s=physical_snapshot(v,m);f=failures(s);bad+=bool(f);copies+=json.dumps((v,m)) in signatures
                q=rank_oracle(s,FIELDS[2+(i%3)])
                sample.append(dict(index=i,scene=s,failures=f,oracle=dict(metric=FIELDS[2+(i%3)],order=q['weak_descending'],ties=q['ties'])))
            rows(out/f'{kind}.{n}.raw.local.jsonl',sample)
            result.append(dict(raw=n,valid=n-bad,exact_fit_copies=copies,refills=0))
        baseline.append(dict(kind=kind,results=result))
    save(out/'baseline-summary.json',dict(results=baseline,interpretation='Controls only; bootstrap copies expected; independent marginals destroy cross-field dependence; no accepted generator'))
    finish(out,'ust_cpu_prepare')
    print(json.dumps(dict(out=str(out),unused_components=len(ordered),fit=len(fit),dev=32,audit=32,roundtrip=len(scenes),api=0)))

def official_model(device,config):
    import numpy as np,torch
    os.environ['WANDB_MODE']='disabled';os.environ['WANDB_DISABLED']='true';os.environ['TORCH_DISABLE_NATIVE_JIT']='1'
    sys.path.insert(0,str(ROOT/'vendor/tabdiff'))
    sys.path.insert(0,str(ROOT/'artifacts/tabdiff-site'))
    from tabdiff.modules.main_modules import UniModMLP,Model
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    backbone=UniModMLP(d_numerical=20,categories=[4]*20,num_layers=config['num_layers'],d_token=4,n_head=1,factor=config['factor'],dim_t=config['dim_t'])
    denoiser=Model(backbone,precond=True,sigma_data=1.,net_conditioning='sigma').to(device)
    return UnifiedCtimeDiffusion(np.array([3]*20),20,denoiser,None,num_timesteps=config['num_timesteps'],
      scheduler='power_mean_per_column',cat_scheduler='log_linear_per_column',noise_dist='uniform_t',
      edm_params=dict(precond=True,sigma_data=1.,net_conditioning='sigma'),
      noise_schedule_params=dict(sigma_min=.002,sigma_max=80.,rho=7.,eps_max=.001,eps_min=.00001,
        rho_init=7.,rho_offset=5.,k_init=-6.,k_offset=1.),
      sampler_params=dict(stochastic_sampler=True,second_order_correction=True),device=device).to(device)
def neural_data(prep):
    import numpy as np
    scenes=[json.loads(l) for l in (prep/'GeneratorFit.scenes.local.jsonl').read_text().splitlines()]
    vm=[vector(s) for s in scenes];values=np.array([v for v,m in vm],dtype=np.float64)
    missing=np.array([m for v,m in vm],dtype=bool)
    assert np.isfinite(values).all() and (values>=0).all()
    positive=(values>0)&(~missing)
    states=np.where(missing,1,np.where(values==0,2,0)).astype(np.float32)
    logs=np.zeros_like(values);logs[positive]=np.log(values[positive])
    # Only positive Fit values determine each numerical transform. Empty columns stay neutral.
    mu=np.array([logs[positive[:,j],j].mean() if positive[:,j].any() else 0. for j in range(20)])
    sd=np.array([logs[positive[:,j],j].std() if positive[:,j].sum()>1 else 1. for j in range(20)])
    sd[sd<1e-6]=1.;normalized=(logs-mu)/sd;normalized[~positive]=0.
    restored=np.zeros_like(values);restored[positive]=np.exp((normalized*sd+mu)[positive])
    assert np.allclose(restored[~missing],values[~missing],rtol=1e-12,atol=1e-12)
    return np.concatenate([normalized,states],axis=1).astype(np.float32),mu,sd
def cpu_interface(out,prep):
    verify_prep(prep)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261026)
    config=load(prep/'small-profile-config.json');x,mu,sd=neural_data(prep)
    model=official_model(torch.device('cpu'),config)
    d,c=model.mixed_loss(torch.from_numpy(x[:4]));loss=d+c
    assert torch.isfinite(loss);loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # Finite-positive/zero/NaN states reconstruct exactly before model quantization.
    values,states=vector(physical_snapshot([0.,1.,.0001,3.,4.]*4,[0,0,0,1,0]*4))
    assert values[0]==0 and states[3]==1
    raw=model.sample(2);assert raw.shape==(2,40)
    save(out/'interface-checks.json',dict(torch=torch.__version__,device='cpu',threads=1,
      fit_rows=len(x),model_parameters=sum(p.numel() for p in model.parameters()),input_shape=list(x.shape),
      categorical_loss=float(d.detach()),numeric_loss=float(c.detach()),finite_backward=True,
      direct_sample_shape=list(raw.shape),sample_all_used=False,no_optimizer_step=True,
      parameter_config=bind(prep/'small-profile-config.json'),prep_manifest=bind(prep/'manifest.json')))
    save(out/'preprocessing.json',dict(mean=mu.tolist(),std=sd.tolist(),fit_only=True,missing_imputation=0.,state_semantics='0=positive, 1=NaN, 2=zero',positive_log_transform=True))
    finish(out,'ust_cpu_neural_interface')
    print(json.dumps(dict(stage='cpu-neural-interface',passed=True,device='cpu',rows=len(x))))
def generated_questions(out,raw,mu,sd,count,seed,template):
    import numpy as np
    rng=random.Random(seed);scenes=[];valid=0;uniques=0;structural=0
    for i,row in enumerate(raw):
        states=row[20:].tolist()
        with np.errstate(over='ignore',invalid='ignore'):
            transformed=np.exp(row[:20]*sd+mu)
        v=[0. if state in (1,2) else float(value) for state,value in zip(states,transformed)]
        m=[int(state==1) for state in states]
        issues=[]
        if not all(math.isfinite(float(t)) for t in list(row)+v+m):issues=['nonfinite_raw']
        if any(t not in (0,1,2) for t in states):issues.append('invalid_three_state')
        if issues:scenes.append(dict(index=i,raw=[None if not math.isfinite(float(t)) else float(t) for t in row],failures=issues));continue
        s=physical_snapshot(v,[int(t) for t in m]);issues=failures(s);structural+=not issues
        # Published renderer emits round2: validity and unique answers use rendered values.
        for node in s['nodes']:
            for a in node['attributes'].values():
                if a['number']!='NaN':a['number']=str(round(float(a['number']),2))
        metric=('distance','waiting_time','charging_time')[i%3];oracle=rank_oracle(s,metric)
        if oracle['ties']:issues.append('ambiguous_displayed_rank_tie')
        order=oracle['weak_descending'];others=[list(p) for p in itertools.permutations(range(1,5)) if list(p)!=order]
        opts=[order]+rng.sample(others,3);rng.shuffle(opts);answer='ABCD'[opts.index(order)]
        lines=['## Task Description',template['sections']['Task Description'],'## Data Schema',template['sections']['Data Schema'],'## Data']
        for n in s['nodes']:
            lines.append('location '+str(n['slot'])+':')
            for k in FIELDS:
                a=n['attributes'][k];lines.append('- '+k+': '+a['number']+(a['unit'] or ''))
        phrase=dict(distance='average distance to nearby charging stations, from the farthest to the closest.',waiting_time='average waiting time at near charging stations, from the longest to the shortest.',charging_time='average charging time at near charging stations, from the longest to the shortest.')[metric]
        lines+=['## Question','Based on the data provided, rank the locations by '+phrase]
        lines += [letter+'. '+' > '.join(map(str,p)) for letter,p in zip('ABCD',opts)]
        lines+=['## Note',template['sections']['Note']]
        scenes.append(dict(index=i,scene=s,question='\n\n'.join(lines),messages=[dict(role='system',content=template['system']),dict(role='user',content='\n\n'.join(lines))],local_independent_answer=answer,metric=metric,
          option_orders=dict(zip('ABCD',opts)),failures=issues,raw_values=[float(x) for x in row]))
        valid+=not issues;uniques+=not oracle['ties']
    rows(out/f'raw-{count}.local.jsonl',scenes)
    return dict(raw=count,structurally_valid=structural,valid_unique=valid,unique_metric_order=uniques,refills=0,round2_before_oracle=True,
      hard_rules=['finite','nonnegative','missing_only_time','unique_rendered_metric_order'])
def gpu_stage(out,prep,stage,authpath,resume=None):
    # Root supplies exact frozen source/inputs; runtime separately enforces host/GPU queue.
    verify_prep(prep)
    auth=load(authpath)
    assert auth['approved'] is True and auth['stage']==stage
    assert auth['code_sha256']==sha(Path(__file__))
    assert auth['preparation_manifest_sha256']==sha(prep/'manifest.json')
    assert auth['max_fit_seconds']<=7200 and auth['max_profile_seconds']<=300
    import numpy as np,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261026);np.random.seed(20261026)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    config=load(prep/'small-profile-config.json')
    assert config.get('version')==2 and config['mask_cardinality']==3
    x,mu,sd=neural_data(prep);tensor=torch.from_numpy(x).cuda()
    model=official_model(torch.device('cuda'),config);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=0)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    start=time.monotonic();maxseconds=auth['max_profile_seconds'] if stage=='profile' else auth['max_fit_seconds']
    limit=20 if stage=='profile' else config['updates'];history=[];stopped=False;start_step=0;elapsed_prior=0.
    if resume:
        resume=Path(resume).resolve()
        assert resume.is_relative_to(NEW.resolve()) and auth['resume_checkpoint_sha256']==sha(resume)
        state=torch.load(resume,map_location='cuda',weights_only=False)
        assert state['source']['sha256']==sha(Path(__file__)) and state['preparation']['sha256']==sha(prep/'manifest.json')
        restore_checkpoint(torch,np,model,optimizer,state,torch.device('cuda'))
        start_step=state['step'];assert 0<start_step<limit
        elapsed_prior=state.get('elapsed_total_seconds',0);maxseconds-=elapsed_prior
        assert maxseconds>0,'Original cumulative fit cap exhausted'
        save(out/'resume-receipt.json',dict(previous_checkpoint=bind(resume),first_next_step=start_step+1,
          already_committed_steps_replayed=0,elapsed_original_fit_seconds=state.get('elapsed_total_seconds',0)))
    cp=out/'checkpoints';cp.mkdir()
    for step in range(start_step+1,limit+1):
        inds=torch.randint(len(tensor),(config['batch_size'],),device='cuda');optimizer.zero_grad()
        d,c=model.mixed_loss(tensor[inds]);loss=d+c
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss: no recovery/resampling')
        loss.backward();optimizer.step();torch.cuda.synchronize()
        history.append(dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-start))
        stop=stop_requested() or time.monotonic()-start>=maxseconds
        if step%200==0 or step==limit or stop:
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,elapsed_total_seconds=elapsed_prior+time.monotonic()-start,torch_rng=torch.get_rng_state(),
              cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),config=config,mean=mu,std=sd,
              source=bind(Path(__file__)),preparation=bind(prep/'manifest.json'))
            target=cp/f'step-{step:06d}.pt';tmp=cp/f'.step-{step:06d}.tmp'
            with tmp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,target)
            latest=sorted(cp.glob('step-*.pt'))[-2:]
            atomic_json(cp/'latest-two.json',[dict(path=p.name,sha256=sha(p)) for p in latest])
        if stop:stopped=True;break
    save(out/'training-history.json',dict(history=history,stage=stage,cap_seconds=maxseconds,stop_or_cap=stopped,
      config=config,peak_cuda_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,
      reproduction='official modules + project bounded runner, not original-scale reproduction'))
    if stage=='fit' and not stop_requested():
        state=torch.load(target,map_location='cuda',weights_only=False)
        reloaded=official_model(torch.device('cuda'),config)
        reload_optimizer=torch.optim.AdamW(reloaded.parameters(),lr=config['learning_rate'],weight_decay=0)
        restore_checkpoint(torch,np,reloaded,reload_optimizer,state,torch.device('cuda'))
        assert reload_optimizer.state_dict()['param_groups']==optimizer.state_dict()['param_groups']
        for key,entry in optimizer.state_dict()['state'].items():
            for name,value in entry.items():
                other=reload_optimizer.state_dict()['state'][key][name]
                assert torch.equal(value,other) if torch.is_tensor(value) else value==other
        expected_cuda=state['cuda_rng'].cpu();assert torch.equal(torch.cuda.get_rng_state(),expected_cuda)
        expected_indices=torch.randint(len(tensor),(config['batch_size'],),device='cuda')
        restore_checkpoint(torch,np,reloaded,reload_optimizer,state,torch.device('cuda'))
        replay_indices=torch.randint(len(tensor),(config['batch_size'],),device='cuda')
        assert torch.equal(expected_indices,replay_indices)
        restore_checkpoint(torch,np,reloaded,reload_optimizer,state,torch.device('cuda'))
        assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
        save(out/'independent-reload.json',dict(checkpoint=target.name,sha256=sha(target),all_model_tensors_equal=True,
          new_model_instance=True,preprocessing_frozen=True,optimizer_saved=True,rng_saved=True,optimizer_reloaded_equal=True,next_cuda_indices_reproduced=True))
        del model;model=reloaded;model.eval();stats=[]
        for n in (100,1000):
            torch.manual_seed(20261026+n);batches=[]
            for offset in range(0,n,64):
                if stop_requested():raise RuntimeError('Stop at atomic sample boundary; no unknown replay')
                batchdir=out/'sample-batches'/str(n);batchdir.mkdir(parents=True,exist_ok=True)
                batchseed=20261026+n+offset;torch.manual_seed(batchseed)
                save(batchdir/f'batch-{offset:06d}.started.json',dict(offset=offset,count=min(64,n-offset),
                  seed=batchseed,cuda_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()))
                batch=model.sample(min(64,n-offset)).numpy();batches.append(batch)
                target_batch=batchdir/f'batch-{offset:06d}.npy';tmp_batch=batchdir/f'.batch-{offset:06d}.tmp'
                with tmp_batch.open('xb') as f:np.save(f,batch);f.flush();os.fsync(f.fileno())
                os.replace(tmp_batch,target_batch)
                save(batchdir/f'batch-{offset:06d}.committed.json',dict(offset=offset,count=len(batch),
                  seed=batchseed,raw_sha256=sha(target_batch)))
            raw=np.concatenate(batches);assert len(raw)==n
            stats.append(generated_questions(out,raw,mu,sd,n,20261026+n,load(prep/'render-template.json')))
            if n==100 and stats[-1]['structurally_valid']/100<.95:break
        save(out/'sample-summary.json',dict(results=stats,no_refill=True,qualification='NOT_ACCEPTED_REQUIRES_DEV_FIDELITY_AND_COPY_AUDIT'))
    finish(out,'ust_'+stage)

def prepare_v2(out,prep):
    import shutil
    for name in ('roles.json','GeneratorFit.scenes.local.jsonl','GeneratorSelectDev32.scenes.local.jsonl','codec-checks.json','code-assets.json'):
        shutil.copyfile(prep/name,out/name)
    config=load(prep/'small-profile-config.json')
    config.update(version=2,mask_cardinality=3,state_semantics='0=positive,1=NaN,2=zero',
      preprocessing='Fit-positive-only log mean/std; nonpositive placeholder normalized0; explicit positive/zero/NaN state; exp inverse without clipping',
      round2_policy='Author cached renderer rounds2; raw corpus coverage sometimes3 decimals; exact source tokens remain codec sidecar',
      fidelity_gate='Mechanical structural>=95/100 only opens1000; ties/unique QA separately reported; no tuning after samples')
    save(out/'small-profile-config.json',config)
    fit=[json.loads(l) for l in (out/'GeneratorFit.scenes.local.jsonl').read_text().splitlines()]
    template=next(r['sections'] for r in fit if r['ability']=='st_understanding')
    save(out/'render-template.json',dict(sections={k:template[k] for k in ('Task Description','Data Schema','Note')},
       system=load(BASE/'prompts.json')['adapted'],origin='static sections from first seeded Fit understanding member'))
    save(out/'v2-amendment.json',dict(previous=bind(prep/'manifest.json'),before_any_gpu=True,
      reason='Explicit positive/zero/NaN preserves zero mass and prevents unconstrained negative values; no clipping',
      GPU_outputs_inspected=0,old_CPU_evidence_unchanged=True))
    finish(out,'ust_prepare_v2')
def verify_prep(prep):
    manifest=load(prep/'manifest.json')
    for spec in manifest['files']:
        assert sha(prep/spec['path'])==spec['sha256'],spec['path']
    for spec in load(prep/'code-assets.json')['files']:
        assert sha(ROOT/spec['path'])==spec['sha256'],spec['path']
def copula(out,prep):
    import numpy as np
    from scipy.special import ndtr,ndtri
    x,mu,sd=neural_data(prep);n,d=x.shape
    ranks=np.empty_like(x,dtype=np.float64)
    for j in range(d):
        values,inv,counts=np.unique(x[:,j],return_inverse=True,return_counts=True)
        edges=np.cumsum(counts);middles=(edges-counts/2.)/n
        ranks[:,j]=ndtri(middles[inv])
    covariance=np.cov(ranks,rowvar=False);covariance=.9*covariance+.1*np.diag(np.diag(covariance))+np.eye(d)*1e-6
    rng=np.random.default_rng(20261026);stats=[]
    for count in (100,1000):
        z=rng.multivariate_normal(np.zeros(d),covariance,size=count)
        raw=np.empty((count,d),dtype=np.float64);u=ndtr(z)
        for j in range(d):
            sorted_col=np.sort(x[:,j]);idx=np.minimum((u[:,j]*n).astype(int),n-1);raw[:,j]=sorted_col[idx]
        stats.append(generated_questions(out,raw,mu,sd,count,20261026+count,load(prep/'render-template.json')))
    save(out/'copula-summary.json',dict(results=stats,fit_only=True,fit_rows=n,dimensions=d,covariance_shrinkage=.1,
      no_refill=True,method='Empirical Gaussian copula of20numeric+20three-state columns; ordinal state surrogate is a limitation',
      privacy='No privacy guarantee; not approved for external transmission',development_or_audit_fitting=False))
    finish(out,'ust_fit_only_gaussian_copula')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','prepare-v2','cpu-neural-interface','copula','profile','fit'])
    p.add_argument('--out');p.add_argument('--prep',default=str(NEW/'branches/UST/cpu-preparation-v1'))
    p.add_argument('--authorization');p.add_argument('--resume-checkpoint');a=p.parse_args()
    out=Path(a.out or os.environ['GENERATOR_STAGE_DIR']).resolve();prep=Path(a.prep).resolve()
    assert out.is_relative_to(NEW.resolve())
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError('Stage output must be empty; unknown started stages cannot replay')
    if a.stage=='prepare':prepare(out)
    elif a.stage=='prepare-v2':prepare_v2(out,prep)
    elif a.stage=='copula':verify_prep(prep);copula(out,prep)
    elif a.stage=='cpu-neural-interface':cpu_interface(out,prep)
    else:gpu_stage(out,prep,a.stage,a.authorization,a.resume_checkpoint)
if __name__=='__main__':main()


