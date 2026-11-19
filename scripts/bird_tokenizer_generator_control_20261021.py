"""Matched native/migrated tokenizer x teacher-context diagnostic (not IPD).

Only the teacher command uses the network, and requires approval.json bound to
the frozen workload. All gold scoring stays local. Historical runs are read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import ipd_native_migration_20261019 as migration
from rescore_bird_student_inplace import load_bird
read,dump,sha,now=migration.read,migration.dump,migration.sha,migration.now
RUN=ROOT/'runs/2026-10-21-bird-tokenizer-generator-control'
OLD=ROOT/'runs/2026-10-21-bird-ipd-recovered-comparison'
RAW=ROOT/'data/raw/candidate-qualification/bird'
SEED=20261021
ARMS=['N-G','N-R','M-G','M-R']
RATES={'hit':.006,'miss':.30,'output':1.20}

def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def frozen(path,value):
    if path.exists():
        if read(path)!=value:raise RuntimeError('Immutable artifact differs: '+str(path))
    else:dump(path,value)

def event(phase,**kwargs):migration.state(RUN,phase,**kwargs)

def historical_spend():
    path=ROOT/'runs/2026-10-20-bird-six-arm-ipd/api-budget.sqlite'
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as c:
        n,pico,inflight=c.execute('SELECT COUNT(*),SUM(pico),SUM(1-settled) FROM calls').fetchone()
    if inflight:raise RuntimeError('Historical API workload still active')
    return dict(path=str(path),attempts=n,conservative_usd=pico/1e12,inflight=inflight)

def prepare():
    if (RUN/'protocol.json').exists():
        print('ALREADY_PREPARED',flush=True);return
    RUN.mkdir(exist_ok=True)
    for d in ['inputs','responses','requests','scores','arms','code']:(RUN/d).mkdir(exist_ok=True)
    rows=[]
    for source,arm in [('G','C2'),('R','D')]:
        path=OLD/'proposals/full200'/(arm+'-items.json')
        for row in read(path):
            rows.append(dict(id=row['id']+'-'+source,task_id=row['id'],source=source,
                source_index=row['source_index'],db_id=row['db_id'],user=row['user'],
                prompt_sha256=digest(row['user']),estimated_prompt_tokens=len(row['prompt_ids'])))
    # Interleave sources; every task is unique within each source.
    rows.sort(key=lambda r:(r['source_index'],r['source']))
    assert len(rows)==400 and len({r['id'] for r in rows})==400
    grouped={s:{r['source_index']:r for r in rows if r['source']==s} for s in ['G','R']}
    assert set(grouped['G'])==set(grouped['R']) and len(grouped['G'])==200
    for index in grouped['G']:
        g,r=[grouped[s][index] for s in ['G','R']]
        # Outside the declared sample view, the entire semantic input must match.
        def without_samples(row):
            prefix,content=row['user'].split('\n\n',1);value=json.loads(content)
            value.pop('sample_rows',None)
            return prefix,value
        assert without_samples(g)==without_samples(r)
    frozen(RUN/'inputs/teacher-items.json',rows)
    # Exclude all previously manifested public-dev exposures, including aliases
    # by normalized DB/question/evidence fingerprint, without reading scores.
    import prepare_bird_ipd_comparison_20261019 as selection
    dev=read(RAW/'dev.json');devkeys={selection.fingerprint(r):r['question_id'] for r in dev}
    by_id={r['question_id']:r for r in dev};excluded=set();evidence=[]
    names={'eval-items.json','frozen-items.json','eval300-identities.json','selection.json',
        'items.json','full-items.json','selection-items.json','audit-items.json'}
    for folder,dirs,files in os.walk(ROOT/'runs'):
        # Requests, trajectories and checkpoints contain no additional selection
        # identities and can be very large; retain manifests at their run roots.
        dirs[:]=[d for d in dirs if d not in {'api-requests','requests','responses','trajectories','checkpoints','snapshots','code-snapshot','code','completion-audits'}]
        if Path(folder).is_relative_to(RUN):dirs[:]=[];continue
        for name in files:
            if name not in names:continue
            p=Path(folder)/name
            if p.stat().st_size>30_000_000:raise RuntimeError('Manifest requires explicit audit: '+str(p))
            obj=read(p);found=set()
            def visit(v):
                if isinstance(v,dict):
                    db=v.get('db_id',v.get('source_db_id'))
                    if db:
                        for key in ['question_id','source_question_id','source_id']:
                            q=v.get(key)
                            if isinstance(q,int) and q in by_id and by_id[q]['db_id']==db:found.add(q)
                        if 'question' in v:
                            fp=selection.fingerprint(dict(db_id=db,question=v['question'],evidence=v.get('evidence',v.get('expert_evidence',''))))
                            if fp in devkeys:found.add(devkeys[fp])
                    for k,x in v.items():
                        if k not in {'prompt_ids','input_ids'}:visit(x)
                elif isinstance(v,list):
                    for x in v:visit(x)
            visit(obj)
            if found:excluded.update(found);evidence.append(dict(path=str(p),sha256=sha(p),dev_ids=sorted(found)))
    assert len(excluded)>=500
    excluded_fp={selection.fingerprint(by_id[i]) for i in excluded}
    pool=[];seen=set()
    for i,row in enumerate(dev):
        fp=selection.fingerprint(row)
        if row['question_id'] in excluded or fp in excluded_fp or fp in seen:continue
        seen.add(fp);pool.append(dict(source_index=i,question_id=row['question_id'],db_id=row['db_id'],
            difficulty=row['difficulty'],question_evidence_sha256=fp))
    if len(pool)<200:raise RuntimeError('Insufficient untouched public dev pool')
    selected,_=selection.stratified(pool,200,lambda r:(r['db_id'],r['difficulty']),SEED)
    frozen(RUN/'inputs/eval200-identities.json',selected)
    frozen(RUN/'historical-exposure-audit.json',dict(excluded_count=len(excluded),excluded_ids=sorted(excluded),
        remaining_pool=len(pool),manifests=evidence,selected_overlap=0,selection_uses_scores=False))
    train=read(RAW/'train.json');gold={}
    # Reuse actual historical public training DB paths, never invent an extraction.
    for row in rows:
        item=train[row['source_index']]
        assert item['db_id']==row['db_id']
        config=read(ROOT/'runs/2026-10-15-105305-bird-generator-pilot/db'/row['db_id']/'frozen_config.json')
        database=Path(config['sqlite'])
        if not database.is_file():raise RuntimeError('Missing historical training DB '+str(database))
        gold[row['id']]=dict(db_id=row['db_id'],db_path=str(database),sql=item['SQL'])
    frozen(RUN/'teacher-gold.local.json',gold)
    old=historical_spend()
    package=dict(provider='DeepSeek official',model='deepseek-flash',thinking='disabled',
        initial_calls=400,max_http_attempts=440,max_technical_retries=40,max_retries_per_item=1,
        estimated_input_tokens=sum(r['estimated_prompt_tokens'] for r in rows),
        estimated_output_tokens=[23002,100000],max_output_tokens_per_call=1024,
        max_input_tokens=12000000,max_output_tokens=450560,max_usd=5,parent_max_usd=200,
        peak_rates_per_million=RATES,estimated_usd_no_cache=[.71,.81],
        request_data='Public BIRD training questions/evidence/schema/column descriptions and two C2 synthetic or real public fit sample rows per table; no gold SQL/results or protected data',
        cheaper_alternative='Flash is designated cheap teacher; local Qwen cannot measure DS teacher-source sensitivity',
        input_sha256=sha(RUN/'inputs/teacher-items.json'),historical_budget=old,
        combined_conservative_ceiling=old['conservative_usd']+5,
        approval_required=True,workers=8,temperature=0,no_quality_retries=True)
    frozen(RUN/'api-work-package.json',package)
    binding=read(OLD/'baseline-binding.json')
    protocol=dict(status='AUTHORIZED_LOCAL_EXECUTION_API_PENDING',owner_instruction='行 做 尽可能加速。',
        method='teacher-text sequence-KD diagnostic; SFT on teacher answers; NOT IPD or B recovery attempt4',
        seed=SEED,arms=ARMS,baselines=['N-0','M-0'],tasks_per_arm=200,updates_per_arm=200,
        eval_denominator=200,eval_ids_sha256=sha(RUN/'inputs/eval200-identities.json'),
        source_sha256=sha(RUN/'inputs/teacher-items.json'),source_checkpoint_N=str(migration.BASE),
        source_checkpoint_M=binding['checkpoint'],B_status='NOT_RECOVERED_BEST_AVAILABLE',
        baseline_binding=binding,rank=8,alpha=16,dropout=0,lora_targets='same q/k/v/o as historical IPD',
        optimizer='AdamW',lr=1e-4,weight_decay=0,gradient_clip=1,
        loss='answer+actualEOS cross entropy sum divided by UTF8 teacher-answer bytes+1 EOS unit',
        epochs=1,order='same deterministic source_index ascending for all arms',
        max_new_tokens=1024,train_prompt='whole prompt, no length clipping or task dropping',
        cpu_threads=4,cuda_fraction=.8,host_memory_max='28G',gpu_workers=1,
        freeze_all_models_before_eval=True,analysis='paired source effects and interaction; DB-cluster bootstrap; single-seed exploratory',
        no_further_B_repair=True,api_work_package_sha256=sha(RUN/'api-work-package.json'))
    frozen(RUN/'protocol.json',protocol)
    dump(RUN/'local-preparation-complete.json',dict(at=now(),status='PASS',teacher_items=400,
        new_eval_items=200,excluded_historical_dev=len(excluded),external_calls=0,GPU_inference=0))
    print(json.dumps(package,indent=2),flush=True)

def body(row):
    return dict(model='deepseek-flash',messages=[dict(role='user',content=row['user'])],
        thinking=dict(type='disabled'),temperature=0,max_tokens=1024,stream=False)

def prepare_eval():
    """Only builds inputs; never runs a model or scores heldout gold."""
    from transformers import AutoTokenizer
    qwen=AutoTokenizer.from_pretrained(migration.BASE,local_files_only=True)
    ds=migration.native_tokenizer();formatter=migration.formatter()
    contexts={};paths={}
    oldgold=read(OLD/'eval-gold-final.local.json')
    for row in read(OLD/'eval-items.json'):
        prefix,content=row['user'].split('\n\n',1)
        contexts[row['db_id']]=(prefix,json.loads(content))
        paths[row['db_id']]=oldgold[row['id']]['db_path']
    dev=read(RAW/'dev.json');items=[];gold={}
    for identity in read(RUN/'inputs/eval200-identities.json'):
        row=dev[identity['source_index']];prefix,context=contexts[row['db_id']]
        context=dict(context,question=row['question'],expert_evidence=row['evidence'])
        user=prefix+'\n\n'+json.dumps(context,ensure_ascii=False,indent=2)
        item=dict(identity,id=f"bird-control-{row['question_id']:04d}",user=user,prompt_sha256=digest(user))
        for state,tok in [('N',qwen),('M',ds)]:
            text=render(user,state,tok,formatter)
            item['prompt_tokens_'+state]=len(tok.encode(text,add_special_tokens=False))
        items.append(item)
        gold[item['id']]=dict(db_id=row['db_id'],db_path=paths[row['db_id']],sql=row['SQL'])
    frozen(RUN/'inputs/eval-items.json',items)
    frozen(RUN/'eval-gold.local.json',gold)
    dump(RUN/'evaluation-preparation.json',dict(at=now(),count=len(items),
        max_prompt_N=max(x['prompt_tokens_N'] for x in items),max_prompt_M=max(x['prompt_tokens_M'] for x in items),
        all_semantic_contexts_reused_from_existing_public_database_views=True,model_inference=0))

def render(user,state,tok,formatter=None):
    if state=='N':return tok.apply_chat_template([dict(role='user',content=user)],add_generation_prompt=True,enable_thinking=False,tokenize=False)
    return (formatter or migration.formatter()).encode_messages([dict(role='user',content=user),
        dict(role='assistant',content='',wo_eos=True)],thinking_mode='chat')

def supervised_ids(user,answer,state,tok):
    prompt=render(user,state,tok)
    suffix='<|im_end|>' if state=='N' else '<｜end▁of▁sentence｜>'
    prompt_ids=tok.encode(prompt,add_special_tokens=False)
    ids=tok.encode(prompt+answer+suffix,add_special_tokens=False)
    if ids[:len(prompt_ids)]!=prompt_ids:raise RuntimeError('Tokenization crosses assistant answer boundary')
    eos=tok.convert_tokens_to_ids(suffix)
    if not ids or ids[-1]!=eos or len(ids)<=len(prompt_ids)+1:raise RuntimeError('Missing actual answer/EOS')
    if tok.decode(ids,skip_special_tokens=False)!=prompt+answer+suffix:raise RuntimeError('Teacher text does not roundtrip')
    labels=[-100]*len(prompt_ids)+ids[len(prompt_ids):]
    return dict(input_ids=ids,labels=labels,prompt_length=len(prompt_ids),
        answer_bytes=len(answer.encode('utf-8')),supervised_tokens=len(ids)-len(prompt_ids),eos_id=eos,
        rendered_sha256=digest(prompt),teacher_text_sha256=digest(answer))

def ledger():
    from privacy_synth.pilot_ledger import SqliteBudget
    from decimal import Decimal
    spec=read(RUN/'api-work-package.json')
    # Separate immutable-workload subledger, explicitly attached to the existing
    # parent budget; no old settings, unknown reservations or calls are reset.
    old=historical_spend()
    if old!=spec['historical_budget']:raise RuntimeError('Parent ledger changed; reconcile before new workload')
    route=read(RUN/'budget-route-amendment.json')
    if route.get('parent_max_usd')!=200 or route.get('explicit_user_correction')!='封顶200dollar不是5daollor':
        raise RuntimeError('Explicit parent-budget route amendment required')
    remaining=Decimal('200')-Decimal(str(old['conservative_usd']))
    if remaining<=0:raise RuntimeError('Parent cap exhausted')
    return SqliteBudget(RUN/'api-budget.sqlite',remaining,{'teacher':dict(calls=440,
        input_tokens=12000000,output_tokens=450560,usd=float(remaining))},RATES)

def teacher():
    spec=read(RUN/'api-work-package.json');approval=read(RUN/'approval.json')
    if approval.get('authorized') is not True or not approval.get('user_confirmation') or approval.get('work_package_sha256')!=sha(RUN/'api-work-package.json'):
        raise RuntimeError('Explicit approval for frozen workload required')
    if sha(RUN/'inputs/teacher-items.json')!=spec['input_sha256']:raise RuntimeError('Teacher inputs changed')
    import httpx
    import deepseek_ipd_interface as api
    book=ledger()
    if any(v['inflight'] for v in book.summary().values()):raise RuntimeError('Uncertain in-flight request; reconcile instead of resend')
    api.apply_proxy()
    stop=threading.Event();guard=threading.Lock()
    with httpx.Client(timeout=httpx.Timeout(connect=30,read=120,write=60,pool=30),
            limits=httpx.Limits(max_connections=8,max_keepalive_connections=8)) as client:
        page=client.get('https://api-docs.deepseek.com/quick_start/pricing/',follow_redirects=True);page.raise_for_status()
        rates=api.parse_rates(page.text)
        if (rates['cache_hit_input_peak'],rates['cache_miss_input_peak'],rates['output_peak'])!=(.006,.3,1.2):
            raise RuntimeError('Pricing changed; re-estimate before calls')
        frozen(RUN/'price-verification.json',dict(rates=rates,source='https://api-docs.deepseek.com/quick_start/pricing/'))
        key=api.load_key()
        def one(row):
            path=RUN/'responses/teacher'/(row['id']+'.json')
            if path.exists():return
            payload_body=body(row)
            for attempt in [1,2]:
                if stop.is_set():raise RuntimeError('Teacher package stopped')
                # Only 40 retries globally; ledger enforces total440 too.
                if attempt==2:
                    with guard:
                        with sqlite3.connect(book.path) as c:n=c.execute("SELECT COUNT(*) FROM calls WHERE tag LIKE '%-try2'").fetchone()[0]
                        if n>=40:break
                        call=book.reserve('teacher',row['id']+'-try2',len(json.dumps(payload_body,ensure_ascii=False).encode())+256,1024)
                else:call=book.reserve('teacher',row['id']+'-try1',len(json.dumps(payload_body,ensure_ascii=False).encode())+256,1024)
                began=time.monotonic()
                try:
                    response=client.post('https://api.deepseek.com/chat/completions',json=payload_body,
                        headers={'Authorization':'Bearer '+key})
                    status=response.status_code
                    try:payload=response.json()
                    except ValueError:payload=dict(error=api.redact(response.text[:2000]))
                except Exception as exc:status=0;payload=dict(error=api.redact(type(exc).__name__+': '+str(exc)))
                record=dict(call_id=call,id=row['id'],at=now(),request=payload_body,http_status=status,payload=payload,
                    elapsed_seconds=time.monotonic()-began)
                with gzip.open(RUN/'requests'/f'{call:06d}.json.gz','xb') as f:f.write(json.dumps(record,ensure_ascii=False).encode())
                book.settle(call,payload.get('usage'))
                if status==200:
                    choice=payload['choices'][0];message=choice['message']
                    if payload.get('model')!='deepseek-flash' or message.get('reasoning_content'):
                        stop.set();raise RuntimeError('Teacher model/thinking changed')
                    with guard:
                        identity=RUN/'teacher-identity.json';fp=payload.get('system_fingerprint')
                        if identity.exists() and read(identity)['fingerprint']!=fp:
                            stop.set();raise RuntimeError('Teacher fingerprint drift')
                        if not identity.exists():dump(identity,dict(model=payload['model'],fingerprint=fp))
                    content=message.get('content','')
                    dump(path,dict(id=row['id'],task_id=row['task_id'],source=row['source'],db_id=row['db_id'],
                        content=content,teacher_text_sha256=digest(content),finish_reason=choice['finish_reason'],
                        http_status=status,prompt_sha256=row['prompt_sha256'],call_id=call,usage=payload.get('usage'),fingerprint=fp))
                    break
                if not api.technical(status):stop.set();raise RuntimeError('Nonretryable HTTP '+str(status))
            else:pass
            if not path.exists():raise RuntimeError('PARTIAL teacher answer collection: '+row['id'])
            with guard:
                count=len(list((RUN/'responses/teacher').glob('*.json')))
                if count%20==0:event('TEACHER_COLLECTION',written=count,total=400,budget=book.summary())
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for _ in pool.map(one,read(RUN/'inputs/teacher-items.json')):pass
        except BaseException:stop.set();raise
        finally:dump(RUN/'api-budget-summary.json',dict(new=book.summary(),historical=historical_spend()))
    responses=[read(RUN/'responses/teacher'/(r['id']+'.json')) for r in read(RUN/'inputs/teacher-items.json')]
    complete=len(responses)==400 and all(r['http_status']==200 and r['finish_reason']=='stop' and r['content'] for r in responses)
    dump(RUN/'teacher-collection-complete.json',dict(status='COMPLETE' if complete else 'PARTIAL',
        count=len(responses),finish_reasons=dict(Counter(r['finish_reason'] for r in responses)),
        teacher_manifest={r['id']:r['teacher_text_sha256'] for r in responses},at=now()))
    if not complete:raise RuntimeError('Incomplete/length-limited teacher targets; do not silently train partial four-arm')

def pack():
    if read(RUN/'teacher-collection-complete.json')['status']!='COMPLETE':raise RuntimeError('Need400 complete teacher targets')
    from transformers import AutoTokenizer
    tokenizers={'N':AutoTokenizer.from_pretrained(migration.BASE,local_files_only=True),'M':migration.native_tokenizer()}
    rows=read(RUN/'inputs/teacher-items.json');stats={}
    for state,tok in tokenizers.items():
        for source in ['G','R']:
            packed=[]
            for row in rows:
                if row['source']!=source:continue
                response=read(RUN/'responses/teacher'/(row['id']+'.json'))
                if response['prompt_sha256']!=row['prompt_sha256'] or digest(response['content'])!=response['teacher_text_sha256']:
                    raise RuntimeError('Teacher target provenance mismatch')
                value=supervised_ids(row['user'],response['content'],state,tok)
                packed.append(dict(id=row['id'],task_id=row['task_id'],source_index=row['source_index'],db_id=row['db_id'],**value))
            arm=state+'-'+source
            frozen(RUN/'inputs'/(arm+'-packed.json'),packed)
            stats[arm]=dict(examples=len(packed),total_tokens=sum(len(r['input_ids']) for r in packed),
                supervised_tokens=sum(r['supervised_tokens'] for r in packed),answer_bytes=sum(r['answer_bytes'] for r in packed),
                max_length=max(len(r['input_ids']) for r in packed),sha256=sha(RUN/'inputs'/(arm+'-packed.json')))
    for source in ['G','R']:
        n=read(RUN/'inputs'/('N-'+source+'-packed.json'));m=read(RUN/'inputs'/('M-'+source+'-packed.json'))
        assert [(r['task_id'],r['teacher_text_sha256'],r['answer_bytes']) for r in n]==[(r['task_id'],r['teacher_text_sha256'],r['answer_bytes']) for r in m]
    frozen(RUN/'packed-manifest.json',stats)
    print(json.dumps(stats,indent=2),flush=True)

def gpu_setup():
    import torch
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8)
    return torch

def base_for(state):return Path(read(RUN/'protocol.json')['source_checkpoint_'+state])

def load_student(state,adapter=None):
    import torch
    from transformers import AutoModelForImageTextToText
    from peft import LoraConfig,get_peft_model,PeftModel
    import preflight_qwen35_lora as pf
    model=AutoModelForImageTextToText.from_pretrained(base_for(state),local_files_only=True,
        dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
    if adapter is False:
        for p in model.parameters():p.requires_grad_(False)
        return model
    if adapter is not None:return PeftModel.from_pretrained(model,Path(adapter)/'student',adapter_name='student',is_trainable=False)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    targets=pf._text_lora_modules(model)
    return get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,
        target_modules=targets,bias='none',task_type='CAUSAL_LM'),adapter_name='student')

def adapter_hash(model):
    from peft import get_peft_model_state_dict
    import torch
    h=hashlib.sha256()
    for key,value in sorted(get_peft_model_state_dict(model,adapter_name='student').items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def preflight():
    """Local training-side checks, no update and no heldout inference."""
    import torch
    from transformers import AutoTokenizer
    torch=gpu_setup();result={}
    # Verify actual gradients of the memory-bounded loss against dense CE.
    torch.manual_seed(SEED)
    h=torch.randn(1,9,7,requires_grad=True);w=torch.randn(31,7,requires_grad=True)
    ids=torch.randint(0,31,(1,9));labels=ids.clone();labels[:,:5]=-100
    chunk=migration.chunked_clm_loss(h,w,ids,chunk_tokens=2,labels=labels)*4/19
    grads=torch.autograd.grad(chunk,(h,w),retain_graph=True)
    dense=torch.nn.functional.cross_entropy(torch.nn.functional.linear(h[:,:-1],w).reshape(-1,31),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/19
    dense_grads=torch.autograd.grad(dense,(h,w))
    assert torch.allclose(chunk,dense,atol=1e-6)
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(grads,dense_grads))
    result['loss_and_gradients_dense_equivalence']=True
    rows=read(RUN/'inputs/teacher-items.json')
    for state in ['N','M']:
        tok=AutoTokenizer.from_pretrained(base_for(state),local_files_only=True) if state=='N' else migration.native_tokenizer()
        packed=[supervised_ids(r['user'],'{"sql":"SELECT 1;"}',state,tok) for r in rows]
        sample=max(packed,key=lambda r:len(r['input_ids']))
        model=load_student(state)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False;model.train()
        ids=torch.tensor([sample['input_ids']],device='cuda');labels=torch.tensor([sample['labels']],device='cuda')
        hidden=model.get_base_model().model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        loss=migration.chunked_clm_loss(hidden,model.get_output_embeddings().weight,ids,128,labels)*sample['supervised_tokens']/(sample['answer_bytes']+1)
        loss.backward()
        params=[p for p in model.parameters() if p.requires_grad]
        grad=float(torch.nn.utils.clip_grad_norm_(params,1))
        if not torch.isfinite(loss) or not 0<grad<float('inf'):raise RuntimeError('Real full-prompt backward failed')
        if model.get_input_embeddings().weight.grad is not None:raise RuntimeError('Embedding should remain frozen')
        result[state]=dict(full_prompt_tokens=len(sample['input_ids']),loss=float(loss.detach()),
            grad_norm=grad,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
            initial_adapter_sha256=adapter_hash(model),trainable_parameters=sum(p.numel() for p in params),
            prompt_masked=True,actual_eos=True,optimizer_updates=0)
        del model,params,hidden,loss,ids,labels;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    assert result['N']['initial_adapter_sha256']==result['M']['initial_adapter_sha256']
    dump(RUN/'local-gpu-preflight.json',dict(status='PASS',at=now(),**result))
    print(json.dumps(result,indent=2),flush=True)

def train(arm):
    if arm not in ARMS:raise ValueError(arm)
    if read(RUN/'local-gpu-preflight.json')['status']!='PASS':raise RuntimeError('GPU preflight required')
    manifest=read(RUN/'packed-manifest.json');path=RUN/'inputs'/(arm+'-packed.json')
    if sha(path)!=manifest[arm]['sha256']:raise RuntimeError('Packed training data changed')
    if (RUN/'models-frozen.json').exists():raise RuntimeError('Cannot train after heldout evaluation release')
    out=RUN/'arms'/arm
    if out.exists():raise RuntimeError('Existing arm; do not restart/resend automatically')
    out.mkdir(parents=True)
    torch=gpu_setup();model=load_student(arm[0]);initial=adapter_hash(model)
    if initial!=read(RUN/'local-gpu-preflight.json')[arm[0]]['initial_adapter_sha256']:raise RuntimeError('Independent fresh adapter differs')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    if any('lora_' not in n for n,_ in named):raise RuntimeError('Nonadapter parameter trainable')
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0,foreach=False)
    rows=read(path);assert len(rows)==200 and len({r['task_id'] for r in rows})==200
    start=time.monotonic()
    for i,row in enumerate(rows,1):
        began=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
        ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor([row['labels']],device='cuda')
        before=[p.detach().clone() for p in params]
        hidden=model.get_base_model().model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        loss=migration.chunked_clm_loss(hidden,model.get_output_embeddings().weight,ids,128,labels)
        loss=loss*row['supervised_tokens']/(row['answer_bytes']+1)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1)
        if not torch.isfinite(norm) or float(norm)<=0:raise RuntimeError('Invalid gradient')
        optimizer.step()
        delta=sum(float((p.detach()-b).float().square().sum()) for p,b in zip(params,before))**.5
        if not 0<delta<float('inf'):raise RuntimeError('No finite actual parameter update')
        rec=dict(update=i,id=row['id'],task_id=row['task_id'],teacher_text_sha256=row['teacher_text_sha256'],
            input_tokens=len(row['input_ids']),supervised_tokens=row['supervised_tokens'],answer_bytes=row['answer_bytes'],
            loss_nats_per_byte_plus_eos=float(loss.detach()),gradient_norm=float(norm),parameter_delta_L2=delta,
            seconds=time.monotonic()-began,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,at=now())
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        del before,hidden,loss,ids,labels
        if i%10==0:event('STUDENT_TRAINING',arm=arm,update=i,total=200,seconds=time.monotonic()-start,**{k:rec[k] for k in ['peak_cuda_mib','loss_nats_per_byte_plus_eos']})
        if i%50==0:
            model.save_pretrained(out/'student-adapter')
            tmp=out/'optimizer.pt.tmp';torch.save(dict(update=i,optimizer=optimizer.state_dict()),tmp);tmp.replace(out/'optimizer.pt')
    model.eval();model.gradient_checkpointing_disable()
    probe=rows[0]['input_ids']
    with torch.inference_mode():
        ids=torch.tensor([probe],device='cuda')
        check=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits.detach().float().cpu()
    torch.save(dict(input_ids=probe,logits=check),out/'reload-probe.pt')
    final=adapter_hash(model)
    states={int(s['step']) for s in optimizer.state.values()}
    assert states=={200} and final!=initial
    dump(out/'training-complete.json',dict(status='TRAINING_COMPLETE',method='sequence-KD, not IPD',
        arm=arm,updates=200,tasks=200,initial_adapter_sha256=initial,final_adapter_sha256=final,
        optimizer_states=len(optimizer.state),optimizer_steps=list(states),at=now(),
        packed_sha256=sha(path),seconds=time.monotonic()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20))

def reload_check(arm):
    torch=gpu_setup();out=RUN/'arms'/arm
    model=load_student(arm[0],out/'student-adapter');model.eval()
    saved=torch.load(out/'reload-probe.pt',map_location='cpu',weights_only=True)
    ids=torch.tensor([saved['input_ids']],device='cuda')
    with torch.inference_mode():live=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits.detach().float().cpu()
    difference=float((live-saved['logits']).abs().max())
    if difference!=0 or adapter_hash(model)!=read(out/'training-complete.json')['final_adapter_sha256']:
        raise RuntimeError('Independent checkpoint reload mismatch')
    dump(out/'reload-check.json',dict(status='PASS',max_logit_difference=difference,at=now()))

def freeze_models():
    manifest={}
    for arm in ARMS:
        folder=RUN/'arms'/arm
        if read(folder/'training-complete.json')['updates']!=200 or read(folder/'reload-check.json')['status']!='PASS':
            raise RuntimeError('All four trained/reloaded models required before evaluation')
        adapter=folder/'student-adapter'
        manifest[arm]={str(p.relative_to(adapter)):sha(p) for p in adapter.rglob('*') if p.is_file()}
        if 'student/adapter_model.safetensors' not in manifest[arm]:raise RuntimeError('Missing named student adapter')
    frozen(RUN/'models-frozen.json',dict(at=now(),arms=manifest,heldout_opened_before_freeze=False))

def evaluate(arm):
    frozen_models=read(RUN/'models-frozen.json')
    if arm not in ARMS+['N-0','M-0']:raise ValueError(arm)
    if arm in ARMS:
        adapter=RUN/'arms'/arm/'student-adapter'
        if {str(p.relative_to(adapter)):sha(p) for p in adapter.rglob('*') if p.is_file()}!=frozen_models['arms'][arm]:
            raise RuntimeError('Final model changed after freeze')
    torch=gpu_setup()
    from transformers import AutoTokenizer
    state=arm[0];tok=AutoTokenizer.from_pretrained(base_for(state),local_files_only=True) if state=='N' else migration.native_tokenizer()
    tok.padding_side='left';model=load_student(state,False if arm.endswith('-0') else RUN/'arms'/arm/'student-adapter')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    rows=read(RUN/'inputs/eval-items.json');rows.sort(key=lambda r:(r['prompt_tokens_'+state],r['id']))
    rendered={r['id']:render(r['user'],state,tok) for r in rows}
    batches=[];batch=[]
    for row in rows:
        if row['prompt_tokens_'+state]>8192:
            if batch:batches.append(batch);batch=[]
            batches.append([row])
        else:
            batch.append(row)
            if len(batch)==4:batches.append(batch);batch=[]
    if batch:batches.append(batch)
    stops=[tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')] if state=='N' else [tok.eos_token_id]
    out=RUN/'responses'/arm;out.mkdir(exist_ok=True)
    began=time.monotonic()
    def generate(batch,singleton_retry=False):
        pending=[r for r in batch if not (out/(r['id']+'.json')).exists()]
        if not pending:return
        inp=seq=None
        try:
            inp=tok([rendered[r['id']] for r in pending],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
            length=inp['input_ids'].shape[1]
            with torch.inference_mode():
                seq=model.generate(**inp,max_new_tokens=1024,do_sample=False,eos_token_id=stops,
                    pad_token_id=tok.pad_token_id,cache_implementation='static',max_cache_len=((length+1279)//256)*256)
            for row,value in zip(pending,seq):
                ids=value[length:].tolist();end=next((i+1 for i,t in enumerate(ids) if t in stops),len(ids));ids=ids[:end]
                dump(out/(row['id']+'.json'),dict(id=row['id'],arm=arm,db_id=row['db_id'],content=tok.decode(ids,skip_special_tokens=True),
                    finish_reason='stop' if ids and ids[-1] in stops else 'length',output_tokens=len(ids),
                    prompt_sha256=row['prompt_sha256'],rendered_sha256=digest(rendered[row['id']]),at=now()))
        except torch.OutOfMemoryError:
            del inp,seq;inp=seq=None;gc.collect();torch.cuda.empty_cache()
            if len(pending)>1 and not singleton_retry:
                for r in pending:generate([r],True)
            else:
                for row in pending:dump(out/(row['id']+'.json'),dict(id=row['id'],arm=arm,content='',finish_reason='technical_failure',error='CUDA_OOM'))
        finally:del inp,seq
    for batch in batches:
        generate(batch);count=len(list(out.glob('*.json')))
        event('STUDENT_EVALUATING',arm=arm,written=count,total=200,seconds=time.monotonic()-began)
    dump(RUN/(arm+'-generation-complete.json'),dict(status='COMPLETE',count=len(list(out.glob('*.json'))),
        seconds=time.monotonic()-began,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20))

def score(kind):
    """Torch-free CLI; bounded independent SQLite workers, gold stays local."""
    bird=load_bird();original=bird.execute_summary
    bird.execute_summary=lambda path,sql,**kwargs:original(path,sql,timeout=30,max_rows=50000)
    if kind=='teacher':
        rows=read(RUN/'inputs/teacher-items.json');gold=read(RUN/'teacher-gold.local.json');folder=RUN/'responses/teacher'
    else:
        read(RUN/'models-frozen.json')
        rows=read(RUN/'inputs/eval-items.json');gold=read(RUN/'eval-gold.local.json');folder=RUN/'responses'/kind
    cache=RUN/('teacher-gold-cache.local.json' if kind=='teacher' else 'eval-gold-cache.local.json')
    expected=read(cache) if cache.exists() else {}
    queries={digest(g['db_path']+'\n'+g['sql']):g for g in gold.values()}
    def execute(pair):
        key,g=pair
        return key,bird.execute_summary(g['db_path'],g['sql'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        for key,result in pool.map(execute,[(k,g) for k,g in queries.items() if k not in expected]):
            expected[key]=result;dump(cache,expected)
    def one(row):
        g=gold[row['id']];value=expected[digest(g['db_path']+'\n'+g['sql'])]
        path=folder/(row['id']+'.json');response=read(path) if path.exists() else {}
        if value['status']=='ok':
            metric=bird.score_response(response.get('content',''),dict(g,expected=value.get('summary')),
                finish_reason=response.get('finish_reason','technical_failure'))
        else:metric=dict(official_ex=False,gold_technical_failure=True,gold_execution_status=value['status'])
        return dict(id=row['id'],db_id=row['db_id'],source=row.get('source'),finish_reason=response.get('finish_reason'),
            response_present=path.exists(),**metric)
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,rows))
    dump(RUN/'scores'/(kind+'-per-item.json'),results)
    summary=dict(denominator=len(rows),correct=sum(bool(r.get('official_ex')) for r in results),
        parse_ok=sum(bool(r.get('parse_ok')) for r in results),execution_ok=sum(bool(r.get('execution_ok')) for r in results),
        gold_technical_failures=sum(bool(r.get('gold_technical_failure')) for r in results),
        truncated=sum(r['finish_reason']=='length' for r in results))
    if kind=='teacher':summary['by_source']={s:dict(denominator=200,correct=sum(r['source']==s and bool(r.get('official_ex')) for r in results)) for s in ['G','R']}
    dump(RUN/'scores'/(kind+'-summary.json'),summary);print(json.dumps(summary),flush=True)

def wait_gpu():
    ready=None
    while True:
        out=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        free,util=map(int,out.strip().split(','))
        if free>=10000 and util<=10:
            ready=ready or time.monotonic()
            if time.monotonic()-ready>=10:return
        else:ready=None
        event('WAITING_GPU_IDLE',free_mib=free,utilization_percent=util);time.sleep(5)

def verify_integrity(create=False):
    """Bind inputs, code and cached base weights without reading any scores."""
    files=[RUN/'protocol.json',RUN/'api-work-package.json',RUN/'budget-route-amendment.json',RUN/'inputs/teacher-items.json',
        RUN/'inputs/eval-items.json',RUN/'inputs/eval200-identities.json',RUN/'teacher-gold.local.json',
        RUN/'eval-gold.local.json']
    files += [ROOT/p for p in read(RUN/'code-manifest.json')['files']]
    for state in ['N','M']:
        files += [p for p in base_for(state).iterdir() if p.is_file() and (p.suffix in {'.json','.safetensors','.py'})]
    actual={str(p):sha(p) for p in files}
    path=RUN/'integrity-manifest.json'
    if create and not path.exists():frozen(path,actual)
    elif read(path)!=actual:raise RuntimeError('Frozen source, base checkpoint or input changed')
    return actual

def child(command,arm=None):
    args=[sys.executable,__file__,command]+(['--arm',arm] if arm else [])
    label=command+('-'+arm if arm else '')
    with (RUN/(label+'.stdout.log')).open('a') as stdout,(RUN/(label+'.stderr.log')).open('a') as stderr:
        subprocess.run(args,check=True,stdout=stdout,stderr=stderr,cwd=ROOT)

def coordinate():
    import fcntl
    lock=(RUN/'controller.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    event('CONTROL_STARTED',method='sequence-KD diagnostic, not IPD')
    if not (RUN/'approval.json').exists():raise RuntimeError('NEEDS_WORKLOAD_APPROVAL')
    verify_integrity(create=True)
    if not (RUN/'teacher-collection-complete.json').exists():child('teacher')
    if not (RUN/'packed-manifest.json').exists():child('pack')
    # CPU scoring overlaps GPU training, without selecting/filtering any targets.
    score_out=(RUN/'score-teacher.stdout.log').open('a');score_err=(RUN/'score-teacher.stderr.log').open('a')
    scorer=None
    if not (RUN/'scores/teacher-summary.json').exists():
        scorer=subprocess.Popen([sys.executable,__file__,'score','--arm','teacher'],stdout=score_out,stderr=score_err,cwd=ROOT)
    try:
        if not (RUN/'local-gpu-preflight.json').exists():wait_gpu();child('preflight')
        for arm in ARMS:
            if not (RUN/'arms'/arm/'training-complete.json').exists():wait_gpu();child('train',arm)
            if not (RUN/'arms'/arm/'reload-check.json').exists():wait_gpu();child('reload',arm)
        if not (RUN/'models-frozen.json').exists():freeze_models()
        for arm in ['N-0','M-0']+ARMS:
            if not (RUN/(arm+'-generation-complete.json')).exists():wait_gpu();child('evaluate',arm)
            if not (RUN/'scores'/(arm+'-summary.json')).exists():child('score',arm)
        if scorer is not None and scorer.wait()!=0:raise RuntimeError('Teacher SQL scoring failed')
        summarize()
        event('COMPLETE',report=str(ROOT/'reports/bird-tokenizer-generator-control-2026-10-21/results.md'))
    except BaseException as exc:
        event('FAILED',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if scorer is not None and scorer.poll() is None:scorer.terminate();scorer.wait()
        score_out.close();score_err.close()

def summarize():
    import numpy as np
    verify_integrity()
    arms=['N-0','M-0']+ARMS
    scores={a:read(RUN/'scores'/(a+'-summary.json')) for a in arms}
    per={a:{r['id']:r for r in read(RUN/'scores'/(a+'-per-item.json'))} for a in arms}
    ids=[r['id'] for r in read(RUN/'inputs/eval-items.json')]
    assert all(set(per[a])==set(ids) for a in arms)
    # Database-cluster bootstrap keeps all items in each sampled DB together.
    dbs=sorted({r['db_id'] for r in per['N-0'].values()});rng=np.random.default_rng(SEED)
    contrasts={'source_N':{'N-G':1,'N-R':-1},'source_M':{'M-G':1,'M-R':-1},
        'interaction':{'M-G':1,'M-R':-1,'N-G':-1,'N-R':1},
        **{a+'_gain':{a:1,a[0]+'-0':-1} for a in ARMS}}
    effects={}
    for name,weights in contrasts.items():
        values={i:sum(w*bool(per[a][i].get('official_ex')) for a,w in weights.items()) for i in ids}
        totals={db:[values[i] for i in ids if per['N-0'][i]['db_id']==db] for db in dbs}
        draws=[]
        for _ in range(5000):
            sample=rng.choice(dbs,len(dbs),replace=True)
            draws.append(100*sum(sum(totals[d]) for d in sample)/sum(len(totals[d]) for d in sample))
        effects[name]=dict(pp=100*sum(values.values())/len(ids),db_cluster_bootstrap95_pp=np.quantile(draws,[.025,.975]).tolist())
    teacher_scores=read(RUN/'scores/teacher-summary.json')
    newbudget=ledger().summary();old=historical_spend()
    audit=dict(at=now(),status='PASS',scores=scores,teacher=teacher_scores,effects=effects,
        updates={a:read(RUN/'arms'/a/'training-complete.json') for a in ARMS},
        reloads={a:read(RUN/'arms'/a/'reload-check.json') for a in ARMS},
        historical_budget=old,new_budget=newbudget,method='sequence-KD diagnostic, NOT IPD',
        limitations=['single seed','migration includes recovery and template','source effect includes teacher answer changes',
            'neither generator fidelity nor privacy established','no IPD-vs-SFT superiority test'])
    if any(s['gold_technical_failures'] for s in list(scores.values())+[teacher_scores]):audit['status']='COMPLETE_WITH_GOLD_SCORING_FAILURES'
    dump(RUN/'completion-audit.json',audit)
    report=ROOT/'reports/bird-tokenizer-generator-control-2026-10-21/results.md'
    text='# 原词表与生成器路径对照\n\n'+audit['method']+'；单配对种子，冻结新200题。不是第四轮B修复。\n\n'
    text+='|组|正确/200|EX|\n|---|---:|---:|\n'
    for a in arms:text+=f"|{a}|{scores[a]['correct']}/200|{scores[a]['correct']/2:.2f}%|\n"
    text+='\n教师G/R：'+json.dumps(teacher_scores.get('by_source'),ensure_ascii=False)+'\n\n'
    text+='效应与按数据库聚类的95% bootstrap区间：\n\n```json\n'+json.dumps(effects,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='所有四组同来源逐字共享教师文本；完整提示+答案/EOS、每组200次独立更新、独立重载。N保留Qwen原词表，M沿用NOT_RECOVERED的固定B。迁移因素仍包含恢复与模板；单种子不作普遍归因，教师质量差异不是生成器保真或隐私证明。\n\n'
    text+='[完整审计]('+str(RUN/'completion-audit.json')+')。\n'
    report.write_text(text)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','prepare-eval','teacher','pack','preflight','train','reload','evaluate','score','coordinate','summarize']);p.add_argument('--arm');args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='prepare-eval':prepare_eval()
    elif args.command=='teacher':teacher()
    elif args.command=='pack':pack()
    elif args.command=='preflight':preflight()
    elif args.command=='train':train(args.arm)
    elif args.command=='reload':reload_check(args.arm)
    elif args.command=='evaluate':evaluate(args.arm)
    elif args.command=='score':score(args.arm)
    elif args.command=='coordinate':coordinate()
    elif args.command=='summarize':summarize()

if __name__=='__main__':main()
