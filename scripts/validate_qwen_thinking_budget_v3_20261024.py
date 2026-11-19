#!/usr/bin/env python3
"""Dev-only Qwen budget calibration; root alone schedules GPU; no imports of models at import."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import eval_ust_sg_qwen_modes_20261024 as q
RUN=ROOT/'runs/2026-10-24-qwen-profile-budget-v3'
OUT=RUN/'budget'
MANIFEST=RUN/'preparation/calibration-manifest.json'
MANIFEST_SHA='98435b7897bbb4c0b18bdd14e4fbc6590aa499a199c690b2369292c9532d315f'
APPROVAL_SHA='4b16667d9768fca6f5c2d32419c872720b32dd4b8b45a4e548b7ee3649f09ff0'
MODELS={'08':ROOT/'artifacts/hf-home/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17','2b':q.DEFAULT_MODEL}
TASKS=('USTBench','SGQA')
ARMS={'N':dict(B=1024,A=256,force=False),'F1':dict(B=1024,A=256,force=True),'F2':dict(B=2048,A=256,force=True),'OFF':dict(B=0,A=256,force=False)}
PARAMS=dict(temperature=1.,top_p=.95,top_k=20,min_p=0.,repetition_penalty=1.,presence_penalty=1.5)
CONTROL='\n</think>\n\n'
GOLD={'USTBench':dict(path='runs/2026-10-23-llm-task-sweep/canonical/USTBench/dev.gold.local.json',sha256='1a4697ef0d3745b948edc8091ab84adecb8e365ab61e9812e9e5853b5ef17cde'),'SGQA':dict(path='runs/2026-10-23-llm-task-sweep/canonical/SGQA/dev.gold.local.json',sha256='c92f787823f8198923e8349758199b5c2f632f0c2df3e8decf996d148ec9e4fa')}
STOP=False

def bind(path):
    p=Path(path).resolve()
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=q.sha(p))
def seed_for(task,rid):return int(hashlib.sha256(('qwen-budget-v3|'+task+'|'+str(rid)).encode()).hexdigest()[:8],16)
def key(model,arm,task,rid):return '|'.join((model,arm,task,str(rid)))
def answer_path(item,output=OUT):return Path(output)/'answers'/item['model']/item['arm']/item['task']/(q.digest(item['id'])+'.json')
def pending_state(path):
    p=Path(path)
    if p.exists():return 'saved'
    if p.with_suffix('.started.json').exists():raise PermissionError('UNKNOWN_STARTED_NO_REPLAY: '+str(p))
    return 'pending'
def phase_stop(ids,budget,close,eos,stop_on_close):
    if not ids or len(ids)>budget:raise RuntimeError('Invalid generated phase count')
    positions=[i for i,x in enumerate(ids) if x in eos]
    if positions:
        if positions!=[len(ids)-1]:raise RuntimeError('Continued after true EOS')
        if stop_on_close and close in ids[:-1]:raise RuntimeError('Missed natural close criterion')
        return 'native_eos'
    if stop_on_close and close in ids:
        if ids[-1]!=close or ids.count(close)!=1:raise RuntimeError('Continued after natural close')
        return 'natural_close'
    if len(ids)!=budget:raise RuntimeError('Unexplained early termination')
    return 'budget_exhausted'
def action_for(arm,reason):
    if reason=='native_eos':return 'stop_early_eos'
    if reason=='natural_close':return 'final_natural'
    if reason!='budget_exhausted':raise ValueError('Unknown phase termination')
    return 'final_forced' if ARMS[arm]['force'] else 'stop_thinking_budget'
def presence_reference(prompt_length,history,ids):return set(history)|set(ids[prompt_length:])
class GeneratedPresence:
    """Full-vocabulary GPU mask, before temperature; real history only across phases."""
    def __init__(self,penalty=1.5):self.penalty=penalty;self.begin_phase(0,[])
    def begin_phase(self,prompt_length,history):self.cursor=prompt_length;self.history=list(history);self.seen=None
    def __call__(self,input_ids,scores):
        import torch
        if input_ids.shape[0]!=1:raise ValueError('batch1 required')
        if self.seen is None:
            self.seen=torch.zeros(scores.shape,dtype=torch.bool,device=scores.device)
            if self.history:self.seen.scatter_(1,torch.tensor([self.history],dtype=torch.long,device=scores.device),True)
        n=input_ids.shape[-1]
        if n<self.cursor:raise RuntimeError('Presence phase boundary changed')
        if n>self.cursor:self.seen.scatter_(1,input_ids[:,self.cursor:n].to(scores.device),True);self.cursor=n
        return scores-self.penalty*self.seen.to(dtype=scores.dtype)
class NaturalThinkClose:
    def __init__(self,prompt_length,closing_id):self.prompt_length=prompt_length;self.closing_id=closing_id
    def __call__(self,input_ids,scores,**kwargs):
        if input_ids.shape[0]!=1:raise ValueError('batch1 required')
        return (input_ids[:,-1]==self.closing_id)&(input_ids.shape[-1]>self.prompt_length)
def two_stage(arm,prompt,control,close,eos,generate):
    """Pure coordinator. Callback generates real tokens; never inserts EOS."""
    if arm not in ARMS:raise PermissionError('Undeclared arm')
    cfg=ARMS[arm]
    if arm=='OFF':
        p2=generate('off',list(prompt),cfg['A'],[],False)
        s2=phase_stop(p2['ids'],cfg['A'],close,eos,False)
        return dict(p1=None,p2=p2,s1=None,s2=s2,action='off',injected=[],forced=False,natural=False,truncated=False,actual=list(p2['ids']))
    p1=generate('thinking',list(prompt),cfg['B'],[],True)
    s1=phase_stop(p1['ids'],cfg['B'],close,eos,True)
    action=action_for(arm,s1);injected=list(control) if action=='final_forced' else []
    p2=None;s2=None
    if action.startswith('final_'):
        # Re-prefill full prefix. No partial GDN/KV state is reused.
        p2=generate('final',list(prompt)+p1['ids']+injected,cfg['A'],list(p1['ids']),False)
        s2=phase_stop(p2['ids'],cfg['A'],close,eos,False)
    return dict(p1=p1,p2=p2,s1=s1,s2=s2,action=action,injected=injected,forced=bool(injected),natural=s1=='natural_close',truncated=s1=='budget_exhausted',actual=p1['ids']+(p2['ids'] if p2 else []))
def load_rows():
    if q.sha(MANIFEST)!=MANIFEST_SHA:raise PermissionError('Calibration manifest changed')
    m=q.load(MANIFEST);q.bound(m['source_manifest']);rows={}
    for task in TASKS:
        spec=m['tasks'][task];values=[json.loads(s) for s in q.bound(spec['queries']).read_text(encoding='utf-8').splitlines() if s.strip()]
        if spec['count']!=4 or len(values)!=4 or len({str(x['id']) for x in values})!=4:raise PermissionError('Four Dev members/task required')
        for row,selected in zip(values,spec['selected']):
            if q.digest(row)!=selected['row_sha256'] or str(row['id'])!=selected['id'] or row['group']!=selected['group']:raise PermissionError('Dev selection changed')
            msg=row['adapted_messages']
            if len(msg)!=18 or msg[0]['role']!='system' or msg[-1]['role']!='user' or [x['role'] for x in msg[1:-1]]!=['user','assistant']*8:raise PermissionError('Fixed8 messages changed')
            if any(not isinstance(x['content'],str) for x in msg) or {'gold','answer','gold_label','feedbacks'}&row.keys():raise PermissionError('Unexpected calibration label field')
        rows[task]=values
    return rows
def plan(rows):return [dict(model=m,arm=a,task=t,id=str(r['id']),group=r['group']) for m in MODELS for a in ARMS for t in TASKS for r in rows[t]]
def is_smoke(item,rows):return item['model']=='08' and item['arm'] in ('N','F1') and item['id']==str(rows[item['task']][0]['id'])
def rendered(tokenizer,row,on):
    text=tokenizer.apply_chat_template(row['adapted_messages'],tokenize=False,add_generation_prompt=True,enable_thinking=on)
    expected='<think>\n' if on else '<think>\n\n</think>\n\n'
    if not text.endswith(expected):raise PermissionError('Native template tail differs')
    return text,tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
def token_contract(tokenizer):
    close=tokenizer.convert_tokens_to_ids('</think>');eos=[tokenizer.convert_tokens_to_ids(t) for t in ('<|im_end|>','<|endoftext|>')]
    control=tokenizer(CONTROL,add_special_tokens=False)['input_ids']
    if close!=248069 or eos!=[248046,248044] or control.count(close)!=1 or any(x in eos for x in control):raise PermissionError('Pinned native IDs/control differ')
    return dict(close=close,eos=eos,control=control)
def validate_off_cache(value,model,task,rid,audit):
    expected=audit['inputs'][key(model,'off',task,rid)];e=value.get('cache_identity',{})
    if (value.get('id')!=str(rid) or value.get('task')!=task or value.get('thinking')!='off' or value.get('decoding')!='greedy' or value.get('max_new_tokens')!=256 or e.get('model_revision')!=audit['revision'] or e.get('model_assets')!=audit['assets'] or e.get('input_ids_sha256')!=expected['input_ids_sha256'] or e.get('messages_sha256')!=expected['messages_sha256'] or value.get('error') or not isinstance(value.get('output_ids'),list)):
        raise PermissionError('Off cache lacks exact complete input/model/config binding; compute new off control')
def prepare(approval,off_cache=None):
    if q.sha(approval)!=APPROVAL_SHA:raise PermissionError('Exact v3 approval required')
    rows=load_rows();q.setup_native()
    from transformers import AutoTokenizer
    audit={}
    for model,path in MODELS.items():
        tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False);tokens=token_contract(tok);entries={}
        for task in TASKS:
            for row in rows[task]:
                for on in (False,True):
                    text,ids=rendered(tok,row,on)
                    if len(ids)+2048+256+len(tokens['control'])>32768:raise PermissionError('Context truncation forbidden')
                    entries[key(model,'on' if on else 'off',task,row['id'])]=dict(rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),input_ids_sha256=q.digest(ids),messages_sha256=q.digest(row['adapted_messages']),input_tokens=len(ids))
        audit[model]=dict(path=str(path),revision=path.name,assets=q.model_assets(path),tokens=tokens,inputs=entries)
    cached=q.load(off_cache) if off_cache else {}
    allowed={key(x['model'],'OFF',x['task'],x['id']) for x in plan(rows) if x['arm']=='OFF'}
    if set(cached)-allowed:raise PermissionError('Undeclared off cache identity')
    for k,spec in cached.items():
        m,_,t,rid=k.split('|',3);validate_off_cache(q.load(q.bound(spec)),m,t,rid,audit[m])
    p=dict(schema='qwen-thinking-budget-v3-dev-only',worker=bind(__file__),tests=bind(ROOT/'tests/test_qwen_thinking_budget_v3_20261024.py'),approval=bind(approval),manifest=bind(MANIFEST),runtime=q.runtime_bindings(),models=audit,arms=ARMS,sampling=PARAMS,control=CONTROL,expected=64,on48_off16=True,plan=plan(rows),smoke=[x for x in plan(rows) if is_smoke(x,rows)],off_cache=cached,off_cache_manifest=bind(off_cache) if off_cache else None,gold=GOLD,all64_before_gold=True,no_Eval_or_Final=True,automatic_full_eval_resume=False,seed_once=True,real_generated_presence_history_continues=True,injection_separate=True,cache_reuse_between_phases=False,phase2_full_reprefill_charged=True,final_native_eos_required=True,resource_limits=dict(memory_gib=14,cpu_threads=4,cuda_allocator_gib=9,batch_size=1),context_cap=32768,new_api_calls=0,calibration_is_fresh=False,limitation='Exposed tiny Dev adaptation diagnostic; SG four old graphs already exposed; no independent generalization claim.')
    q.immutable(OUT/'protocol.json',p)
    return dict(status='PREPARED_ZERO_GPU',protocol=bind(OUT/'protocol.json'),expected=64)
def verify(launch=True):
    p=q.load(OUT/'protocol.json')
    for name in ('worker','tests','approval','manifest'):q.bound(p[name])
    if p['approval']['sha256']!=APPROVAL_SHA or p['arms']!=ARMS or p['sampling']!=PARAMS or p['control']!=CONTROL or p['expected']!=64 or p['plan']!=plan(load_rows()) or p['gold']!=GOLD:raise PermissionError('Frozen scientific scope changed')
    if p['runtime']!=q.runtime_bindings():raise PermissionError('Runtime/helper changed')
    if p.get('off_cache_manifest'):q.bound(p['off_cache_manifest'])
    for s in p['off_cache'].values():q.bound(s)
    if launch:
        a=q.load(OUT/'launch-authorization.json')
        if a.get('approved') is not True or a.get('protocol_sha256')!=q.sha(OUT/'protocol.json') or a.get('approval_sha256')!=APPROVAL_SHA:raise PermissionError('Root-reviewed launch authorization required')
    return p
def stopped():return STOP or any((RUN/n).exists() for n in ('stop.json','root-stop.json','global-stop.json'))
def unresolved(output=OUT):
    bad=[p for p in (Path(output)/'answers').rglob('*.started.json') if not p.with_name(p.name.replace('.started.json','.json')).exists()]
    if bad:raise PermissionError('UNKNOWN_STARTED_NO_REPLAY: '+str(bad[0]))
def check_record(value,item,protocol_sha):
    if value.get('key')!=key(item['model'],item['arm'],item['task'],item['id']) or value.get('protocol_sha256')!=protocol_sha:raise PermissionError('Saved answer binding differs')
def make_record(result,tok,item,p,prompt_length,seconds):
    p1,p2=result['p1'],result['p2'];actual=result['actual'];transcript=(p1['ids'] if p1 else [])+result['injected']+(p2['ids'] if p2 else [])
    good=result['s2']=='native_eos'
    finish='stop' if good else 'early_eos' if result['action']=='stop_early_eos' else 'thinking_budget_exhausted' if result['action']=='stop_thinking_budget' else 'length'
    raw=tok.decode(actual,skip_special_tokens=False);controlled=tok.decode(transcript,skip_special_tokens=False)
    parsed=q.classify(controlled,'off' if item['arm']=='OFF' else 'on','ust' if item['task']=='USTBench' else 'sgqa',item.get('metadata',{}),finish)
    if finish=='thinking_budget_exhausted':parsed['reason']='THINKING_BUDGET_EXHAUSTED'
    elif finish=='early_eos':parsed['reason']='EARLY_NATIVE_EOS_WITHOUT_THINK_CLOSE'
    phases=[x for x in (p1,p2) if x]
    return dict(parsed,id=item['id'],group=item['group'],task=item['task'],model=item['model'],revision=p['models'][item['model']]['revision'],arm=item['arm'],thinking='off' if item['arm']=='OFF' else 'on',key=key(item['model'],item['arm'],item['task'],item['id']),protocol_sha256=q.sha(OUT/'protocol.json'),seed=seed_for(item['task'],item['id']),raw_generated=raw,output_ids=actual,controlled_transcript=controlled,phase1=p1,phase2=p2,phase1_stop=result['s1'],phase2_stop=result['s2'],action=result['action'],injected_ids=result['injected'],injected_text=CONTROL if result['forced'] else '',injected_tokens=len(result['injected']),forced_close=result['forced'],natural_close=result['natural'],reasoning_truncated=result['truncated'],output_tokens=len(actual),input_tokens=prompt_length,actual_prefill_tokens=sum(x['input_tokens'] for x in phases),generation_seconds=sum(x['generation_seconds'] for x in phases),logical_seconds=seconds,finish_reason=finish,native_eos=good or result['action']=='stop_early_eos',final_native_eos=good,input_truncated=False,output_truncated=finish in ('length','thinking_budget_exhausted'),decoding='greedy' if item['arm']=='OFF' else 'native-sampling-budget-v3',max_new_tokens=256 if item['arm']=='OFF' else ARMS[item['arm']]['B']+256,forced_protocol_scoring=result['forced'],technical_failure=False,cache_reuse_between_phases=False)
def run(model_name,scope):
    p=verify();rows=load_rows();protocol_sha=q.sha(OUT/'protocol.json')
    if scope=='smoke' and model_name!='08':raise PermissionError('Smoke is exactly08 two questions crossed N,F1')
    if scope=='all':
        receipt=q.load(OUT/'smoke-checked.json')
        if receipt.get('status')!='PASS' or receipt.get('protocol_sha256')!=protocol_sha:raise PermissionError('Mechanical smoke check required')
        for s in receipt['predictions']:q.bound(s)
    selected=[x for x in p['plan'] if x['model']==model_name and (scope=='all' or is_smoke(x,rows))]
    unresolved()
    for item in selected:
        if pending_state(answer_path(item))=='saved':check_record(q.load(answer_path(item)),item,protocol_sha)
    if all(answer_path(item).exists() for item in selected):return dict(status='CACHED_SCOPE_COMPLETE',model=model_name,scope=scope,newly_generated=0)
    if stopped():return dict(status='STOPPED_BEFORE_MODEL_LOAD')
    from night_dual_20261022 import lock,cgroup_limits_verified
    import run_cityfm_repair_qwen_20261022 as guard
    with lock(OUT/'worker.lock'),lock(guard.HISTORICAL/'locks/gpu.lock'):
        cgroup=cgroup_limits_verified();snapshot=guard.probe(OUT)
        if not all(snapshot.get(k,-1)>=20 for k in ('windows_free_gib','c_free_gib','linux_free_gib')) or snapshot.get('gpu_free_gib',-1)<9:raise RuntimeError('Immediate RAM/disk20GiB/GPU9GiB floor unmet')
        path=MODELS[model_name]
        if q.model_assets(path)!=p['models'][model_name]['assets']:raise PermissionError('Native asset bytes changed')
        q.setup_native()
        import torch,transformers
        torch.set_num_threads(4)
        if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
        torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
        tok=transformers.AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False);tokens=token_contract(tok)
        if tokens!=p['models'][model_name]['tokens']:raise PermissionError('Tokenizer contract changed')
        cfg=q.load(path/'config.json');cls=transformers.AutoModelForImageTextToText if cfg.get('model_type') in ('qwen3_5','qwen3_5_moe') else transformers.AutoModelForCausalLM
        model=cls.from_pretrained(str(path),dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True,trust_remote_code=False)
        if any('lora' in n.lower() for n,_ in model.named_parameters()):raise PermissionError('Adapter forbidden')
        model.eval();model.requires_grad_(False)
        signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True));signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True))
        count=0
        for item in selected:
            answer=answer_path(item)
            if pending_state(answer)=='saved':check_record(q.load(answer),item,protocol_sha);continue
            if stopped():return dict(status='STOPPED_AT_LOGICAL_ANSWER_BOUNDARY',saved=count)
            snapshot=guard.probe(OUT)
            if not all(snapshot.get(k,-1)>=20 for k in ('windows_free_gib','c_free_gib','linux_free_gib')):raise RuntimeError('Host resource floor breached')
            row=next(r for r in rows[item['task']] if str(r['id'])==item['id']);text,ids=rendered(tok,row,item['arm']!='OFF')
            expected=p['models'][model_name]['inputs'][key(model_name,'off' if item['arm']=='OFF' else 'on',item['task'],item['id'])]
            if q.digest(ids)!=expected['input_ids_sha256'] or hashlib.sha256(text.encode()).hexdigest()!=expected['rendered_sha256']:raise PermissionError('Native rendering changed')
            cache_key=key(model_name,'OFF',item['task'],item['id'])
            if item['arm']=='OFF' and cache_key in p['off_cache']:
                cached=q.load(q.bound(p['off_cache'][cache_key]));validate_off_cache(cached,model_name,item['task'],item['id'],p['models'][model_name])
                q.immutable(answer,dict(cached,key=cache_key,arm='OFF',model=model_name,protocol_sha256=protocol_sha,reused_cache=p['off_cache'][cache_key]));continue
            q.immutable(answer.with_suffix('.started.json'),dict(key=key(model_name,item['arm'],item['task'],item['id']),protocol_sha256=protocol_sha,seed=seed_for(item['task'],item['id']),started_epoch=time.time(),cgroup=cgroup))
            # Exactly one RNG initialization per logical answer; neither phase re-seeds.
            torch.manual_seed(seed_for(item['task'],item['id']))
            presence=GeneratedPresence(PARAMS['presence_penalty']);phase_values={};logical_start=time.monotonic()
            def generate(name,prefix,budget,history,stop_close):
                presence.begin_phase(len(prefix),history)
                inputs=torch.tensor([prefix],dtype=torch.long,device='cuda');attention=torch.ones_like(inputs)
                kwargs=dict(input_ids=inputs,attention_mask=attention,do_sample=item['arm']!='OFF',num_beams=1,max_new_tokens=budget,eos_token_id=tokens['eos'],pad_token_id=tok.pad_token_id,use_cache=True)
                if item['arm']!='OFF':
                    kwargs.update({k:v for k,v in PARAMS.items() if k!='presence_penalty'});kwargs['logits_processor']=transformers.LogitsProcessorList([presence])
                if stop_close:kwargs['stopping_criteria']=transformers.StoppingCriteriaList([NaturalThinkClose(len(prefix),tokens['close'])])
                torch.cuda.synchronize();started=time.monotonic()
                with torch.inference_mode():generated=model.generate(**kwargs)
                torch.cuda.synchronize();elapsed=time.monotonic()-started;emitted=generated[0,len(prefix):].tolist()
                value=dict(name=name,ids=emitted,input_tokens=len(prefix),output_tokens=len(emitted),real_history_tokens=len(history),generation_seconds=elapsed,full_reprefill=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                phase_values[name]=value;q.immutable(answer.with_suffix('.'+name+'-phase.json'),dict(value,key=key(model_name,item['arm'],item['task'],item['id']),protocol_sha256=protocol_sha))
                del generated,inputs,attention
                return value
            try:
                result=two_stage(item['arm'],ids,tokens['control'],tokens['close'],tokens['eos'],generate)
                q.immutable(answer,make_record(result,tok,dict(item,metadata=row.get('metadata',{})),p,len(ids),time.monotonic()-logical_start));count+=1
            except torch.OutOfMemoryError:
                q.immutable(answer,dict(id=item['id'],task=item['task'],model=model_name,arm=item['arm'],key=key(model_name,item['arm'],item['task'],item['id']),protocol_sha256=protocol_sha,valid=False,prediction=None,text='',no_final=True,error='CUDA_OOM',finish_reason='technical_failure',phases_completed=phase_values,logical_seconds=time.monotonic()-logical_start))
                raise RuntimeError('CUDA_OOM_SAVED_INVALID_STOP_NO_RETRY')
        return dict(status='SCOPE_COMPLETE',model=model_name,scope=scope,newly_generated=count)
def smoke_check():
    p=verify(False);unresolved();specs=[]
    for item in p['smoke']:
        path=answer_path(item);v=q.load(path);check_record(v,item,q.sha(OUT/'protocol.json'))
        if v.get('error'):raise RuntimeError('Smoke technical failure')
        if v['forced_close'] and not(item['arm']=='F1' and v['phase1_stop']=='budget_exhausted'):raise RuntimeError('Illegal forced close')
        if v['phase1_stop']=='native_eos' and v['phase2'] is not None:raise RuntimeError('Continued after true EOS')
        if v['output_tokens']>ARMS[item['arm']]['B']+256:raise RuntimeError('Smoke budget exceeded')
        specs.append(bind(path))
    value=dict(status='PASS',mechanical_only=True,gold_read=False,expected=4,protocol_sha256=q.sha(OUT/'protocol.json'),predictions=specs,ordinary_invalid_answers_are_not_technical_failures=True)
    q.immutable(OUT/'smoke-checked.json',value);return value
def freeze():
    p=verify(False);unresolved();expected=set();specs=[]
    for item in p['plan']:
        path=answer_path(item);check_record(q.load(path),item,q.sha(OUT/'protocol.json'));expected.add(path);specs.append(bind(path))
    actual={x for x in (OUT/'answers').rglob('*.json') if not x.name.endswith('.started.json') and not x.name.endswith('-phase.json')}
    if actual!=expected or len(specs)!=64:raise PermissionError('Fixed64 denominator/extra answer mismatch')
    value=dict(complete=True,expected=64,on48_off16=True,protocol=bind(OUT/'protocol.json'),predictions=specs,gold_read=False)
    q.immutable(OUT/'all64-predictions-frozen.json',value);return value
def score():
    # Only this function opens gold, and only after every declared answer freezes.
    frozen=freeze()
    for s in frozen['predictions']:q.bound(s)
    from run_llm_task_sweep_20261023 import score_prediction
    p=q.load(OUT/'protocol.json');rows=load_rows();gold={t:q.load(q.bound(p['gold'][t])) for t in TASKS};results={};per_id=[]
    for model in MODELS:
        for arm in ARMS:
            for task in TASKS:
                s=dict(total=4,correct=0,valid=0,forced=0,natural_close=0,final_native_eos=0,thinking_budget_exhausted=0,final_budget_exhausted=0,early_eos=0,output_tokens=0,injected_tokens=0,generation_seconds=0.,phase2_seconds=0.)
                for row in rows[task]:
                    v=q.load(answer_path(dict(model=model,arm=arm,task=task,id=str(row['id']))))
                    correct,valid=score_prediction(v,dict(kind='qa',scorer='ust' if task=='USTBench' else 'sgqa'),gold[task][str(row['id'])],row.get('metadata',{}))
                    s['correct']+=bool(correct);s['valid']+=bool(valid);s['forced']+=bool(v.get('forced_close'));s['natural_close']+=bool(v.get('natural_close'));s['final_native_eos']+=bool(v.get('final_native_eos'))
                    s['thinking_budget_exhausted']+=v.get('phase1_stop')=='budget_exhausted';s['final_budget_exhausted']+=v.get('phase2_stop')=='budget_exhausted';s['early_eos']+=v.get('action')=='stop_early_eos'
                    for k in ('output_tokens','injected_tokens','generation_seconds'):s[k]+=v.get(k,0)
                    s['phase2_seconds']+=(v.get('phase2') or {}).get('generation_seconds',0.)
                    per_id.append(dict(model=model,arm=arm,task=task,id=str(row['id']),correct=bool(correct),valid=bool(valid),prediction=v.get('prediction')))
                results[model+'/'+arm+'/'+task]=s
    value=dict(status='COMPLETE_DEV_DIAGNOSTIC',scores=results,per_id=per_id,all64_freeze=bind(OUT/'all64-predictions-frozen.json'),gold_bindings=GOLD,scorer=bind(ROOT/'scripts/run_llm_task_sweep_20261023.py'),calibration_is_not_fresh_validation=True,no_Eval_or_Final=True,automatic_full_eval_resume=False)
    q.immutable(OUT/'scores.local.json',value);return value
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','smoke-check','freeze','score'));p.add_argument('--approval',type=Path,default=RUN/'user-approval.json');p.add_argument('--off-cache',type=Path);p.add_argument('--model',choices=tuple(MODELS));p.add_argument('--scope',choices=('smoke','all'),default='smoke');a=p.parse_args()
    if a.action=='prepare':result=prepare(a.approval,a.off_cache)
    elif a.action=='run':
        if not a.model:p.error('run requires --model')
        result=run(a.model,a.scope)
    elif a.action=='smoke-check':result=smoke_check()
    elif a.action=='freeze':result=freeze()
    else:result=score()
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()


