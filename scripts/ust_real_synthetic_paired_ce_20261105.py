"""Fixed local real/synthetic reference-CE control; no external requests.

Reuses the original checkpoint, packing, CE update, serializer and decoder.
Exposed evaluation is descriptive. Synthetic gains are not physical H5 gold.
"""
from pathlib import Path
import argparse, ast, collections, fcntl, hashlib, json, os, signal, subprocess, sys, time

R = Path(__file__).resolve().parents[1]
D = R/'runs/2026-11-05-ust-environment-generator-ipd-v1'
S = D/'full-schema-mainline-v3'
O = R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1'
OLD = R/'runs/2026-10-25-qwen-transfer-repair/seqkd200/USTBench/2b'
INIT = OLD/'work/checkpoints/step0000'
ARMS = ('real-reference-ce', 'synthetic-reference-ce')
sys.path[:0] = [str(S), str(R/'scripts'), str(R/'src')]

def load(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve()
    return dict(path=str(p.relative_to(R)), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def check():
    if (O/'STOP').exists(): raise InterruptedError('Own paired run STOP')
def verified(pin):
    p=R/pin['path']; assert bind(p)==pin, pin['path']; return p
def verify(out):
    p=load(Path(out)/'protocol.json')
    assert p['source']==bind(__file__)
    for pin in p['bindings']: verified(pin)
    assert p['steps']==200 and p['lr']==1e-4
    return p

def prepare():
    assert not O.exists(), 'New immutable namespace required'
    O.mkdir()
    oldp=load(OLD/'protocol.json'); inp=load(OLD/'inputs.json')
    gold=load(R/'runs/2026-10-24-ust-sg-onpolicy/data/USTBench.train.gold.local.json')
    real=inp['selected']; counts=collections.Counter(x['metadata']['ability'] for x in real)
    assert counts==dict(st_understanding=68,planning=66,reflection=66)
    candidates=[json.loads(line) for line in (S/'shared-cal-v2.raw-questions.local.jsonl').open()]
    pools={}
    for ability,n in counts.items():
        pool=[x for x in candidates if x['role']=='train' and x['eligible'] and x['metadata']['ability']==ability]
        pools[ability]=iter(sorted(pool,key=lambda x:hashlib.sha256(('full-UST-20261104:'+x['id']).encode()).hexdigest())[:n])
    synth=[next(pools[r['metadata']['ability']]) for r in real]
    assert len({r['id'] for r in synth})==200
    save(O/'fixed8.local.json',inp['fixed8'])
    for arm,rows in zip(ARMS,(real,synth)):
        packed=[]
        for row in rows:
            ability=row['metadata']['ability']
            answer=gold[row['id']] if arm==ARMS[0] else row['local_understanding_answer'] if ability=='st_understanding' else row['local_synthetic_gain_answer']
            assert answer in 'ABCD'
            target='<Answer>'+answer+'</Answer>' if ability=='st_understanding' else json.dumps(dict(answer=answer,summary='Matches the requested goal.'),separators=(',',':'))
            question=row['question'] if arm==ARMS[0] else row['messages'][-1]['content']
            packed.append(dict(id=row['id'],question=question,metadata=row['metadata'],target=target,
                               label_provenance='published real training gold' if arm==ARMS[0] else row['target_provenance']))
        unit=O/arm; save(unit/'training200.local.json',packed)
        pins=[bind(INIT/'manifest.json'),bind(OLD/'protocol.json'),bind(O/'fixed8.local.json'),bind(unit/'training200.local.json'),
              bind(S/'gpu_student_worker.py'),bind(R/'scripts/matched_qwen_objective_control_20261102.py'),
              bind(R/'scripts/probe_overnight_train_fit_20261025.py'),bind(R/'scripts/three_method_common_eval_20261101.py')]
        save(unit/'protocol.json',dict(source=bind(__file__),arm=arm,base=oldp['base'],seed=20261024,steps=200,lr=1e-4,
             lora=oldp['lora'],bindings=pins,loss='response-only fullsemantic CE including native EOS',
             protected_real_training_gold_in_loss=arm==ARMS[0],synthetic_reference_in_loss=arm==ARMS[1],teacher_API=0,
             fixed_checkpoint=200,ability_order=[r['metadata']['ability'] for r in real],
             evaluation_use='Already exposed Common100 and Audit32, descriptive only; no selection',Final_access=False))
    save(O/'protocol.json',dict(source=bind(__file__),arms=list(ARMS),initialization=bind(INIT/'manifest.json'),
         selection='Existing calibrated generator fixed before any new results',teacher_API=0,physical_H5_recovered=False))
    print('PREPARED two matched200 arms; zero API',flush=True)

def machinery():
    import gpu_student_worker as w
    w.O=O; w.P=None; w.verify=verify; w.check=check
    source=R/'scripts/matched_qwen_objective_control_20261102.py'
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='oracle_update')
    ns=dict(w.c.__dict__,check=check)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source)+'::unchanged_fullsemantic_CE','exec'),ns)
    return w,ns['oracle_update']

def fit(arm):
    unit=O/arm; p=verify(unit); check()
    save(unit/'fit-started.json',dict(PID=os.getpid(),epoch=time.time(),protocol=bind(unit/'protocol.json')))
    w,update=machinery()
    import probe_overnight_train_fit_20261025 as packing
    with (S/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch,tok,model,head=w.initialize(unit,INIT)
        w.m.lib.activate(model,head,'actor')
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
        cm=w.m.lib.restore(torch,model,head,dict(actor=opt),OLD,INIT)
        assert not opt.state and cm['actual_updates']==0
        roles={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']}; assert roles==cm['role_hashes']
        save(unit/'initialization-verified.json',dict(roles=roles,optimizer=w.m.lib.nested_hash(opt.state_dict()),RNG_restored=True))
        prefix=load(O/'fixed8.local.json'); rows=load(unit/'training200.local.json'); actual=0; tick=time.monotonic()
        for step,row in enumerate(rows,1):
            check()
            save(unit/'updates'/f'{step:04d}.started.json',dict(id=row['id'],epoch=time.time()))
            prompt,target=packing.packing(tok,row,dict(text=row['target']),'matched8',prefix)
            assert len(prompt)+len(target)<=16384
            actor=w.m.lib.role_hash(model,'actor')
            trace=dict(prompt_ids=prompt,oracle_target_ids=target,actor_before=actor,
                       sources={'LOCAL_REFERENCE':len(target)},events={'teacher_forcing':len(target)})
            result=update(torch,model,head,opt,trace)
            result.update(method='reference-ce',id=row['id'],step=step,
                          protected_real_gold_in_loss=arm==ARMS[0],local_synthetic_oracle_in_loss=arm==ARMS[1],teacher_API=0)
            actual+=int(result['optimizer_updated']); save(unit/'updates'/f'{step:04d}.json',result)
            if step%25==0: print('FIT',arm,step,'seconds',round(time.monotonic()-tick,1),flush=True)
            if step in (100,200):
                folder=unit/'checkpoints'/f'step{step:04d}'
                w.m.native.tiny.SAVE_CHECKPOINT(unit,folder,torch,model,head,dict(actor=opt),'PAIRED_REFERENCE_CE',step,actual_updates=actual)
                w.m.lib.verify_checkpoint(unit,folder)
        assert actual==200
        final={r:w.m.lib.role_hash(model,r) for r in roles}
        assert all(final[r]==roles[r] for r in roles if r!='actor') and final['actor']!=roles['actor']
        save(unit/'fit-terminal.json',dict(status='FIXED200_COMPLETE',actual_updates=actual,seconds=time.monotonic()-tick,roles=final,
                                         checkpoint=bind(unit/'checkpoints/step0200/manifest.json'),teacher_API=0))

def questions():
    plan=load(S/'evaluation-plan.json'); out=[]
    for line in (R/plan['original_Common100']['queries']['path']).open():
        row=json.loads(line); out.append(dict(id=row['id'],cohort='Common100',metadata=row['metadata'],messages=row['adapted_messages']))
    for row in load(D/'final-joint-audit32-v4/questions32.local.json'):
        out.append(dict(**row,cohort='exposed_Audit32'))
    assert len(out)==132 and len({r['id'] for r in out})==132
    return out

def evaluate(arm):
    unit=O/arm; p=verify(unit); assert load(unit/'fit-terminal.json')['actual_updates']==200
    w,_=machinery()
    source=R/'scripts/three_method_common_eval_20261101.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('hf_greedy','context_failure')]
    ns=dict(time=time,CAP=32,CONTEXT=16384); exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),ns)
    with (S/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(unit/'evaluation-started.json',dict(PID=os.getpid(),epoch=time.time()))
        folder=unit/'checkpoints/step0200'; torch,tok,model,head=w.initialize(unit,folder)
        w.m.lib.activate(model,head,'actor'); model.eval()
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False)
        cm=w.m.lib.restore(torch,model,head,dict(actor=opt),unit,folder)
        assert cm['role_hashes']==load(unit/'fit-terminal.json')['roles']
        save(unit/'independent-reload.json',dict(status='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS',manifest=bind(folder/'manifest.json')))
        for i,row in enumerate(questions()):
            check(); save(unit/'raw'/f'{i:04d}.started.json',dict(id=row['id'],epoch=time.time()))
            prefix=tok.encode(tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
            value=ns['hf_greedy'](torch,tok,model,prefix)
            save(unit/'raw'/f'{i:04d}.json',dict(id=row['id'],cohort=row['cohort'],metadata=row['metadata'],**value))
            if i%25==0: print('EVAL',arm,i,132,flush=True)
        assert cm['role_hashes']=={r:w.m.lib.role_hash(model,r) for r in cm['role_hashes']}
        save(unit/'evaluation-terminal.json',dict(status='ALL132_RAW_CLOSED',raw=132,API=0))

def score():
    import eval_ust_sg_qwen_modes_20261024 as q
    import generator_discovery_ust_20261026 as g
    rows=questions()
    for arm in ARMS: assert load(O/arm/'evaluation-terminal.json')['raw']==132
    save(O/'all264-new-raw-frozen.json',dict(arms=list(ARMS),raw=264,epoch=time.time()))
    plan=load(S/'evaluation-plan.json'); gold=load(R/plan['original_Common100']['gold']['path'])
    wanted={r['id'] for r in rows if r['cohort']=='exposed_Audit32'}
    pool=R/load(D/'final-joint-audit32-v4/protocol.json')['source_pool']['path']
    for line in pool.open():
        identity=g.picked(line,{'id'})
        if identity['id'] in wanted:
            row=g.picked(line,{'id','gold'}); gold[row['id']]=row['gold']
    assert all(r['id'] in gold for r in rows)
    results={}; flags={}
    for arm in ARMS:
        graded=[]
        for i,row in enumerate(rows):
            raw=load(O/arm/'raw'/f'{i:04d}.json'); assert raw['id']==row['id']
            parsed=q.classify(raw['raw_generated'],'off','ust',row['metadata'],raw['finish_reason'])
            graded.append(dict(id=row['id'],cohort=row['cohort'],ability=row['metadata']['ability'],
                               correct=bool(parsed['valid'] and parsed['prediction']==gold[row['id']]),valid=bool(parsed['valid']),length_capped=raw['finish_reason']=='length'))
        save(O/arm/'graded.local.json',graded); flags[arm]=graded; results[arm]={}
        for cohort in ('Common100','exposed_Audit32'):
            a=[r for r in graded if r['cohort']==cohort]
            results[arm][cohort]=dict(correct=sum(r['correct'] for r in a),total=len(a),valid=sum(r['valid'] for r in a),
                 length_capped=sum(r['length_capped'] for r in a),abilities={k:dict(correct=sum(r['correct'] for r in a if r['ability']==k),total=sum(r['ability']==k for r in a)) for k in ('st_understanding','planning','reflection')})
    paired={}
    for cohort in ('Common100','exposed_Audit32'):
        pairs=[(a,b) for a,b in zip(flags[ARMS[0]],flags[ARMS[1]]) if a['cohort']==cohort]
        paired[cohort]=dict(synthetic_wins=sum(not a['correct'] and b['correct'] for a,b in pairs),real_wins=sum(a['correct'] and not b['correct'] for a,b in pairs),both_correct=sum(a['correct'] and b['correct'] for a,b in pairs),both_wrong=sum(not a['correct'] and not b['correct'] for a,b in pairs))
    save(O/'results.json',dict(status='COMPLETE',scores=results,paired=paired,API=0,diagnostic='local reference CE, not teacher distillation',evaluation='exposed descriptive',physical_H5_recovered=False))
    print(json.dumps(results),flush=True)

def controller():
    save(O/'controller-started.json',dict(PID=os.getpid(),PPID=os.getppid(),epoch=time.time()))
    for action in ('fit','evaluate'):
        for arm in ARMS:
            cmd=[str(R/'.venv/bin/python'),str(Path(__file__)),action,'--arm',arm]
            with (O/(action+'-'+arm+'.log')).open('x') as f:
                subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
    score(); save(O/'terminal.json',dict(status='COMPLETE',epoch=time.time(),API=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('prepare','fit','evaluate','score','controller')); parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    try:
        if args.action in ('fit','evaluate'): globals()[args.action](args.arm)
        else: globals()[args.action]()
    except BaseException as ex:
        if O.exists():
            p=O/(args.action+'-'+str(args.arm)+'-failure-'+str(time.time_ns())+'.json')
            save(p,dict(type=type(ex).__name__,message=str(ex)[:700],epoch=time.time(),no_automatic_restart=True))
        raise
