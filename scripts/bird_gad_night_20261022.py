"""Bounded native-vocabulary BIRD GAD adaptation, with zero-network preparation.

No external-model client exists here. GPU commands require a root-owned launch
authorization bound to the protocol and immutable preparation manifest.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import bird_tokenizer_generator_control_20261021 as control
from privacy_synth.onpolicy_gad import (bradley_terry, clipped_grpo,
    group_advantages, last_nonpadding, response_log_probs)
read, dump, sha = control.read, control.dump, control.sha
OLD = control.RUN
RUN = ROOT/'runs/2026-10-22-night-dual/preparation/bird'
PROTOCOL = RUN/'protocol-v3.json'
INTEGRITY = RUN/'integrity-manifest-v4.json'
REGISTRY = ROOT/'runs/2026-10-21-bird-question-family-registry-v3/registry.json'
SEED = 20261022
STOP_REQUESTED = False


class GracefulStop(RuntimeError):
    """Stop at a saved item boundary, never halfway through an optimizer step."""


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def heartbeat(phase, **details):
    control.dump(RUN/'worker-heartbeat.json',dict(phase=phase,at_epoch=time.time(),pid=os.getpid(),**details))


def verify_integrity():
    for name,expected in read(INTEGRITY).items():
        if not Path(name).is_file() or sha(Path(name))!=expected:
            raise RuntimeError('Frozen input/code/base changed: '+name)


def frozen(path, obj):
    control.frozen(path, obj)


def append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream: stream.write(json.dumps(obj, ensure_ascii=False)+'\n')


def files_at(folder):
    return {str(p): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def sql_worker(pipe,path,sql,timeout):
    # Reuse the established exact canonical/hash implementation, but keep its
    # bounded scratch files in this new run rather than in the historical run.
    import bird_control_streaming_score_20261021 as streaming
    streaming.RUN=RUN
    streaming.worker(pipe,path,sql,timeout)


def execute_sql(path,sql,**_):
    timeout=30;ctx=mp.get_context('spawn');recv,send=ctx.Pipe(duplex=False)
    process=ctx.Process(target=sql_worker,args=(send,path,sql,timeout));process.start();send.close()
    try:
        if recv.poll(timeout+3):
            try:return recv.recv()
            except EOFError:return dict(status='error',error_type='WorkerExit')
        return dict(status='timeout')
    finally:
        if process.is_alive():process.terminate()
        process.join();recv.close()


def validation_gold_preflight():
    target=RUN/'validation-gold-preflight-v2.json'
    if target.exists():
        result=read(target)
        if result['gold_file_sha256']!=sha(RUN/'validation64-gold.local.json'):raise RuntimeError('Validation gold changed')
        if result['status']!='PASS':raise RuntimeError('Validation gold remains unscorable')
        return
    items=list(read(RUN/'validation64-gold.local.json').items())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda item:dict(id=item[0],**execute_sql(item[1]['db_path'],item[1]['sql'])),items))
    old={r['id']:r for r in read(RUN/'validation-gold-preflight.json')['results']}
    matched=sum(r['status']=='ok' and r['summary']==old[r['id']].get('summary') for r in results if old[r['id']]['status']=='ok')
    result=dict(status='PASS' if all(r['status']=='ok' for r in results) and matched==63 else 'FAIL',
        denominator=64,gold_scorable=sum(r['status']=='ok' for r in results),original_scorable63_exact_summary_matches=matched,
        gold_file_sha256=sha(RUN/'validation64-gold.local.json'),scorer='existing canonical external-memory hash',
        timeout_seconds=30,max_rows=5000000,max_scratch_bytes_per_worker=2*1024**3,final_gold_materialized=False,results=results)
    frozen(target,result)
    if result['status']!='PASS':raise RuntimeError('Validation gold/scorer preflight failed; preserve denominator and stop')


def cache_audit():
    rows = [r for r in read(OLD/'inputs/teacher-items.json') if r['source']=='R']
    assert len(rows)==200 and len({r['source_index'] for r in rows})==200
    complete = read(OLD/'teacher-collection-complete.json')
    assert complete['status']=='COMPLETE'
    teacher_identity = read(OLD/'teacher-identity.json')
    records=[]; request_files={}
    for p in (OLD/'requests').glob('*.gz'):
        record=json.load(gzip.open(p, 'rt'))
        if record['http_status']==200: request_files[record['call_id']] = (p, record)
    for row in rows:
        assert control.digest(row['user'])==row['prompt_sha256']
        p=OLD/'responses/teacher'/(row['id']+'.json');response=read(p)
        assert response['http_status']==200 and response['finish_reason']=='stop' and response['content']
        assert response['prompt_sha256']==row['prompt_sha256']
        assert control.digest(response['content'])==response['teacher_text_sha256']==complete['teacher_manifest'][row['id']]
        rp, raw = request_files[response['call_id']]
        request=raw['request'];payload=raw['payload']
        assert raw['id']==row['id'] and request==control.body(row)
        assert request['thinking']=={'type':'disabled'} and request['model']=='deepseek-flash'
        assert payload['system_fingerprint']==teacher_identity['fingerprint']==response['fingerprint']
        assert payload['choices'][0]['message']['content']==response['content']
        records.append(dict(id=row['id'],source_index=row['source_index'],db_id=row['db_id'],
            prompt_sha256=row['prompt_sha256'],teacher_text_sha256=response['teacher_text_sha256'],
            response_file=str(p),response_file_sha256=sha(p),request_file=str(rp),request_file_sha256=sha(rp)))
    nr=OLD/'arms/N-R';assert read(nr/'training-complete.json')['updates']==200
    assert read(nr/'reload-check.json')['status']=='PASS'
    checkpoint=nr/'student-adapter/student'
    assert (checkpoint/'adapter_model.safetensors').is_file()
    totals=[]
    for p in [ROOT/'runs/2026-10-20-bird-six-arm-ipd/api-budget.sqlite',OLD/'api-budget.sqlite']:
        with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as db:
            calls,pico,inflight=db.execute('SELECT COUNT(*),SUM(pico),SUM(1-settled) FROM calls').fetchone()
        assert not inflight
        totals.append(dict(path=str(p),attempts=calls,conservative_usd=pico/1e12,inflight=inflight))
    return dict(status='CACHE_MATCH',new_external_calls=0,teachers=records,teacher_identity=teacher_identity,
        teacher_model_revision='provider model alias + saved system_fingerprint; no immutable provider revision supplied',
        teacher_mode='disabled',teacher_sampling_temperature=0,teacher_max_tokens=1024,
        nr_checkpoint=str(checkpoint),nr_checkpoint_files=files_at(checkpoint),
        nr_training_complete_sha256=sha(nr/'training-complete.json'),nr_reload_sha256=sha(nr/'reload-check.json'),
        historical_BIRD_ledgers=totals,combined_BIRD_conservative_usd=sum(x['conservative_usd'] for x in totals),
        accounting_scope='these two BIRD ledgers only, not all project/model/agent spend')


def split_prepare():
    """Select new public-dev roles without reading any historical scores/gold."""
    import prepare_bird_ipd_comparison_20261019 as selection
    dev=read(control.RAW/'dev.json');by_id={r['question_id']:r for r in dev}
    fp_to_id={selection.fingerprint(r):r['question_id'] for r in dev}
    registry=read(REGISTRY);excluded=set();evidence=[]
    assert len(registry['families'])==13
    for family in registry['families'].values():
        p=Path(family['identity_manifest']);assert sha(p)==family['identity_sha256']
        if family['split']=='dev':
            ids={r['official_question_id'] for r in read(p)};excluded.update(ids)
            evidence.append(dict(path=str(p),sha256=sha(p),dev_ids=sorted(ids),kind='registry_public_dev_membership'))
    assert len(excluded)==700
    # Supplement canonical families with saved ad-hoc manifests/runtime records.
    # Prepared training pools are not called actual training exposures. All official
    # train identities are split-disjoint from the new official-dev validation/test.
    names={'eval-items.json','frozen-items.json','eval300-identities.json','selection.json','audit.json',
           'items.json','full-items.json','selection-items.json','audit-items.json','full-context-validation16.json'}
    def visit(obj,found):
        if isinstance(obj,dict):
            db=obj.get('db_id',obj.get('source_db_id'))
            if db:
                for key in ['question_id','source_question_id','source_id','official_question_id']:
                    q=obj.get(key)
                    if isinstance(q,int) and q in by_id and by_id[q]['db_id']==db: found.add(q)
                if 'question' in obj:
                    fp=selection.fingerprint(dict(db_id=db,question=obj['question'],evidence=obj.get('evidence',obj.get('expert_evidence',''))))
                    if fp in fp_to_id:found.add(fp_to_id[fp])
                if isinstance(obj.get('user'),str):
                    try:
                        context=json.loads(obj['user'].split('\n\n',1)[1])
                        fp=selection.fingerprint(dict(db_id=db,question=context['question'],evidence=context.get('expert_evidence','')))
                        if fp in fp_to_id:found.add(fp_to_id[fp])
                    except (ValueError,KeyError,IndexError): pass
            for k,v in obj.items():
                if k not in {'input_ids','prompt_ids','logits'}:visit(v,found)
        elif isinstance(obj,list):
            for value in obj: visit(value,found)
    for folder,dirs,files in os.walk(ROOT/'runs'):
        dirs[:]=[d for d in dirs if d not in {'api-requests','requests','responses','trajectories','checkpoints','snapshots','code-snapshot','code','completion-audits','preparation','families'}]
        for name in files:
            if name not in names:continue
            p=Path(folder)/name
            if p.stat().st_size>30_000_000:raise RuntimeError('Unreviewed large role manifest '+str(p))
            found=set();visit(read(p),found)
            if found:
                excluded.update(found);evidence.append(dict(path=str(p),sha256=sha(p),dev_ids=sorted(found),kind='historical_manifest_or_runtime'))
    excluded_fp={selection.fingerprint(by_id[i]) for i in excluded}
    pool=[];seen=set()
    for index,row in enumerate(dev):
        fp=selection.fingerprint(row)
        if row['question_id'] in excluded or fp in excluded_fp or fp in seen:continue
        seen.add(fp);pool.append(dict(source_index=index,question_id=row['question_id'],db_id=row['db_id'],
            difficulty=row['difficulty'],question_evidence_sha256=fp))
    validation,_=selection.stratified(pool,64,lambda r:(r['db_id'],r['difficulty']),SEED)
    valids={r['question_id'] for r in validation};valfps={r['question_evidence_sha256'] for r in validation}
    final,_=selection.stratified([r for r in pool if r['question_id'] not in valids and r['question_evidence_sha256'] not in valfps],200,lambda r:(r['db_id'],r['difficulty']),SEED+1)
    frozen(RUN/'validation64-identities.json',validation);frozen(RUN/'final200-identities.sealed.json',final)
    frozen(RUN/'exposure-audit.json',dict(registry_sha256=sha(REGISTRY),registry_families=13,
        source_file_sha256=sha(control.RAW/'dev.json'),excluded_dev_ids=sorted(excluded),excluded_count=len(excluded),
        remaining_pool=len(pool),historical_evidence=evidence,validation_n=64,final_n=200,overlap=0,
        roles='new official-dev validation64 for checkpoint selection, separate new official-dev final200 sealed until freeze',
        official_train_overlap=0,prepared_recovery_pool_is_not_actual_exposure=True,selection_uses_scores=False))


def materialize(identities, include_gold=True):
    """Reuse existing complete public DB views; gold remains local."""
    contexts={};paths={}
    for folder in [control.OLD, OLD]:
        gp=folder/('eval-gold-final.local.json' if folder==control.OLD else 'eval-gold.local.json')
        gold=read(gp) if include_gold else None
        for row in read(folder/('eval-items.json' if folder==control.OLD else 'inputs/eval-items.json')):
            prefix,content=row['user'].split('\n\n',1)
            contexts[row['db_id']]=(prefix,json.loads(content))
            if include_gold:paths[row['db_id']]=gold[row['id']]['db_path']
    dev=read(control.RAW/'dev.json');rows=[];gold={}
    for identity in identities:
        row=dev[identity['source_index']];assert row['question_id']==identity['question_id']
        prefix,context=contexts[row['db_id']];context=dict(context,question=row['question'],expert_evidence=row['evidence'])
        user=prefix+'\n\n'+json.dumps(context,ensure_ascii=False,indent=2)
        item=dict(identity,id=f"bird-gad-{row['question_id']:04d}",user=user,prompt_sha256=control.digest(user))
        rows.append(item)
        if include_gold:gold[item['id']]=dict(db_id=row['db_id'],db_path=paths[row['db_id']],sql=row['SQL'])
    return rows,gold


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    audit=cache_audit();frozen(RUN/'cache-audit.json',audit)
    split_prepare()
    rows=[r for r in read(OLD/'inputs/teacher-items.json') if r['source']=='R']
    frozen(RUN/'train200.json',rows)
    validation,gold=materialize(read(RUN/'validation64-identities.json'))
    frozen(RUN/'validation64-items.json',validation);frozen(RUN/'validation64-gold.local.json',gold)
    validation_gold_preflight()
    from transformers import AutoTokenizer
    base=read(OLD/'protocol.json')['source_checkpoint_N']
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    historic=max(len(tok.encode(control.render(r['user'],'N',tok),add_special_tokens=False)) for r in read(OLD/'inputs/eval-items.json')+rows)
    bounds={}
    # Inspect only final input text/length here. No final SQL is selected or
    # materialized; it is released only after all candidate models freeze.
    final_inputs,_=materialize(read(RUN/'final200-identities.sealed.json'),include_gold=False)
    for name,data in [('train',rows),('validation',validation),('final_inputs',final_inputs)]:
        lengths=[len(tok.encode(control.render(r['user'],'N',tok),add_special_tokens=False)) for r in data]
        bounds[name]=dict(count=len(data),max_prompt_tokens=max(lengths),total_prompt_tokens=sum(lengths))
    if max(bounds[r]['max_prompt_tokens'] for r in ['validation','final_inputs'])>historic:
        raise RuntimeError('Heldout input exceeds historical complete-context bound; no truncation allowed')
    final_input_manifest=[dict(id=r['id'],prompt_sha256=r['prompt_sha256'],prompt_tokens=len(tok.encode(control.render(r['user'],'N',tok),add_special_tokens=False))) for r in final_inputs]
    frozen(RUN/'final200-input-integrity.json',final_input_manifest)
    frozen(RUN/'token-bound-audit-v2.json',dict(historic_max_prompt_tokens=historic,**bounds,full_prompts=True,truncation=False,
        final_gold_materialized=False,final_input_integrity_sha256=sha(RUN/'final200-input-integrity.json')))
    protocol=dict(status='PREPARED_NOT_GPU_VALIDATED_NOT_LAUNCHED',method='GAD native-vocabulary BIRD adaptation',
        base=base,initial_actor_adapter=audit['nr_checkpoint'],reference='immutable independent copy of N-R adapter',
        train_sha256=sha(RUN/'train200.json'),cache_audit_sha256=sha(RUN/'cache-audit.json'),
        validation64_sha256=sha(RUN/'validation64-items.json'),final200_identity_sha256=sha(RUN/'final200-identities.sealed.json'),
        seed=SEED,group_size=8,temperature=.8,max_new_tokens=1024,max_policy_updates=200,
        discriminator_warmup_updates=200,discriminator_update='one group BT mean after each fresh actor group',
        warmup='reuse completed N-R SeqKD; then sequential D-only one-pass warmup with real N-R rollouts; adaptation of joint warmup',
        lr=1e-6,clip=.2,beta=.001,optimizer='AdamW weight_decay0',gradient_clip=1,
        policy_loss='equal response mean of token-mean clipped GRPO + k3 sampled KL to immutable N-R; temperature .8 consistently',
        discriminator='independent N-R initialized LoRA plus random linear scalar head on last actual nonpadding hidden state',
        shared_base=True,base_frozen=True,adapter_roles=['actor','reference','discriminator'],
        save_every=5,validate_every=50,checkpoint_selection='validation64 EX desc, parse desc, trunc asc, earliest update',
        max_trials=3,trial_roles=['default GAD','continued SeqKD same N-R same train200 additional200 cap','one evidence-based technical fix only'],
        no_stale_replay=True,external_calls=0,teacher_cache_immutable=True,full_prompt=True,
        prompt_max_tokens=historic,gpu_allocator_gib=9,host_limit_gib=14,
        discriminator_warmup_rollouts=200,discriminator_warmup_group_size=1,policy_rollouts=1600,total_training_rollouts=1800,
        final_test='new official-dev200; input hashes/lengths checked before freeze, final SQL gold materialized only after all eligible checkpoints frozen; reevaluate N-0 N-R GAD SeqKD',
        limitations=['BIRD public proxy; no privacy/fidelity proof','single seed and 200 questions',
            'LoRA/shared frozen base/sequential warmup/one prompt per update are resource adaptations, not exact author reproduction',
            'D reward is optimization signal, SQL EX is independent quality metric'])
    protocol['supersedes_protocol_sha256']=sha(RUN/'protocol-v2.json')
    protocol['amendment']='v3 preserves v2 single-pair200 D warmup + N8/200 policy, complete-input audit and safe resume; replaces row-capped SQL result storage with the existing exact same canonical/set-hash external-memory scorer after one of64 validation gold queries exceeded50k rows; no membership/denominator change'
    protocol['sql_scorer']=dict(canonical_and_comparison='unchanged original canonical row/set hash',storage='existing external-memory SQLite count/hash',
        timeout_seconds=30,max_rows=5000000,max_scratch_bytes_per_worker=2*1024**3,workers=2,
        validation_gold_preflight_sha256=sha(RUN/'validation-gold-preflight-v2.json'))
    protocol['token_bound_audit_sha256']=sha(RUN/'token-bound-audit-v2.json')
    protocol['resume']='explicit --resume only, complete boundary checkpoint with exact adapters/head/optimizers/RNG/log hashes; real-model preflight must validate resume'
    frozen(PROTOCOL,protocol)
    conditional=dict(status='CONDITIONAL_ONLY_DO_NOT_SEND',provider='DeepSeek official',model='deepseek-flash',thinking='disabled',
        calls=200,max_technical_attempts=220,max_input_tokens_per_call=16384,max_output_tokens_per_call=1024,
        input_token_ceiling=200*16384,output_token_ceiling=200*1024,
        input_data='same 200 frozen public BIRD REAL prompts; no gold SQL or protected records',
        cheaper_alternative='verified existing 200 cached teacher outputs requires zero API calls',
        condition='only cache integrity/teacher identity mismatch, then explicit per-workload approval required',
        cost='not re-priced; reserve determined by root approved current provider rates before any request')
    frozen(RUN/'conditional-refresh-work-package.json',conditional)
    paths=[RUN/name for name in ['protocol.json','protocol-v2.json','protocol-v3.json','train200.json','cache-audit.json','exposure-audit.json','validation64-items.json',
        'validation64-gold.local.json','final200-identities.sealed.json','final200-input-integrity.json','token-bound-audit-v2.json',
        'conditional-refresh-work-package.json','validation-gold-preflight.json','validation-gold-preflight-v2.json']]
    paths += [Path(__file__),ROOT/'src/privacy_synth/onpolicy_gad.py',ROOT/'tests/test_onpolicy_gad.py']
    paths += [ROOT/'scripts'/name for name in ['bird_tokenizer_generator_control_20261021.py','ipd_native_migration_20261019.py','rescore_bird_student_inplace.py','bird_control_streaming_score_20261021.py']]
    from rescore_bird_student_inplace import BIRD_PATH
    paths += [BIRD_PATH]
    paths += list((RUN/'primary').glob('*'))
    paths += [Path(r[key]) for r in audit['teachers'] for key in ['request_file','response_file']]
    # The separate .jinja template is executable input to native prompt rendering.
    paths += [p for p in Path(base).iterdir() if p.is_file()]
    paths += [ROOT/'pyproject.toml',ROOT/'uv.lock']
    paths += list((OLD/'arms/N-R/student-adapter/student').glob('*'))
    frozen(INTEGRITY,{str(p):sha(p) for p in paths if p.is_file()})
    frozen(RUN/'preparation-complete-v4.json',dict(status='READY_FOR_AUTHORIZED_GPU_PREFLIGHT',external_calls=0,optimizer_updates=0,
        protocol_sha256=sha(PROTOCOL),integrity_sha256=sha(INTEGRITY),
        teacher_cache=200,validation64=64,final200_sealed=True,real_model_gradient_reload='NOT_RUN'))
    frozen(RUN/'preparation-active.json',dict(protocol_path=str(PROTOCOL),protocol_sha256=sha(PROTOCOL),
        integrity_path=str(INTEGRITY),integrity_sha256=sha(INTEGRITY),preparation_path=str(RUN/'preparation-complete-v4.json')))
    print(json.dumps(read(RUN/'preparation-complete-v4.json'),indent=2))


def guard(command):
    verify_integrity()
    import night_dual_20261022 as runtime
    runtime.root_authorization(RUN.parents[1])
    runtime.supervised_worker_verified(RUN.parents[1])
    approval=read(RUN/'launch-authorization.json')
    if approval.get('authorized') is not True or approval.get('protocol_sha256')!=sha(PROTOCOL) or approval.get('integrity_sha256')!=sha(INTEGRITY):
        raise RuntimeError('Root launch authorization does not bind this exact preparation')
    if command not in approval.get('allowed_commands',[]):raise RuntimeError('Command not authorized: '+command)


def activate(model, head, role):
    model.set_adapter(role)
    for name,param in model.named_parameters():param.requires_grad_(role!='reference' and 'lora_' in name and ('.'+role+'.') in name)
    for param in head.parameters():param.requires_grad_(role=='discriminator')


def role_hash(model,role):
    import torch
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    for k,v in sorted(get_peft_model_state_dict(model,adapter_name=role).items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def setup(checkpoint=None):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForImageTextToText
    p=read(PROTOCOL);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    tok=AutoTokenizer.from_pretrained(p['base'],local_files_only=True)
    base=AutoModelForImageTextToText.from_pretrained(p['base'],local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
    path=lambda role: Path(checkpoint)/role if checkpoint else Path(p['initial_actor_adapter'])
    model=PeftModel.from_pretrained(base,path('actor'),adapter_name='actor',is_trainable=True)
    model.load_adapter(path('reference'),adapter_name='reference',is_trainable=False)
    model.load_adapter(path('discriminator'),adapter_name='discriminator',is_trainable=True)
    dim=model.get_input_embeddings().weight.shape[1]
    head=torch.nn.Linear(dim,1,bias=False,device='cuda',dtype=torch.float32)
    if checkpoint:head.load_state_dict(torch.load(Path(checkpoint)/'score-head.pt',map_location='cuda',weights_only=True))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    activate(model,head,'actor')
    return torch,tok,model,head


def prompt_ids(tok,row):
    ids=tok.encode(control.render(row['user'],'N',tok),add_special_tokens=False)
    if len(ids)>read(PROTOCOL)['prompt_max_tokens']:raise RuntimeError('Complete prompt exceeds frozen bound; no clipping')
    return ids


def hidden(model,ids):
    return model.get_base_model().model(input_ids=ids,attention_mask=ids.new_ones(ids.shape),use_cache=False).last_hidden_state


def rollout(torch,tok,model,head,row,n=8,greedy=False,base_only=False):
    if not base_only:activate(model,head,'actor')
    model.eval();prompt=prompt_ids(tok,row)
    stops=[tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')]
    records=[]
    for index in range(n):
        heartbeat('ROLLOUT',id=row['id'],group_index=index,group_size=n)
        ids=torch.tensor([prompt],device='cuda')
        with torch.inference_mode():
            seq=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=1024,
                do_sample=not greedy,temperature=.8 if not greedy else None,top_k=0,top_p=1.,
                eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True,
                cache_implementation='static',max_cache_len=((len(prompt)+1279)//256)*256)
        response=seq[0,len(prompt):].tolist();end=next((i+1 for i,t in enumerate(response) if t in stops),len(response));response=response[:end]
        if not response:raise RuntimeError('Empty token response')
        records.append(dict(input_ids=prompt+response,prompt_length=len(prompt),response_ids=response,
            content=tok.decode(response,skip_special_tokens=True),finish_reason='stop' if response[-1] in stops else 'length',
            actual_eos=response[-1] in stops,group_index=index))
        del seq,ids
        heartbeat('ROLLOUT_SAVED_IN_MEMORY',id=row['id'],group_index=index,response_tokens=len(response))
    return records


def logprobs(torch,model,record):
    ids=torch.tensor([record['input_ids']],device='cuda')
    return response_log_probs(hidden(model,ids),model.get_output_embeddings().weight,ids,record['prompt_length'],temperature=.8)


def dscore(torch,model,head,ids):
    tokens=torch.tensor([ids],device='cuda');h=hidden(model,tokens)
    return head(last_nonpadding(h,torch.ones_like(tokens)).float()).reshape(())


def update_d(torch,model,head,optimizer,teacher_ids,records):
    activate(model,head,'discriminator');model.train();optimizer.zero_grad(set_to_none=True)
    before=role_hash(model,'discriminator');head_before=head.weight.detach().clone();losses=[]
    for record in records:
        loss=bradley_terry(dscore(torch,model,head,teacher_ids),dscore(torch,model,head,record['input_ids']))/len(records)
        losses.append(float(loss.detach()));loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]+list(head.parameters())
    norm=torch.nn.utils.clip_grad_norm_(params,1)
    if not bool(torch.isfinite(norm)) or float(norm)<=0:raise RuntimeError('Invalid discriminator gradient')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    if role_hash(model,'discriminator')==before and torch.equal(head.weight,head_before):raise RuntimeError('No discriminator update')
    return dict(loss=sum(losses),gradient_norm=float(norm))


def nested_hash(value):
    """Stable state digest for tensor optimizer/RNG state, including metadata."""
    import torch
    h=hashlib.sha256()
    def visit(obj):
        if isinstance(obj,torch.Tensor):
            h.update(str((str(obj.dtype),tuple(obj.shape))).encode())
            h.update(obj.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj,dict):
            for key in sorted(obj,key=str):h.update(str(key).encode());visit(obj[key])
        elif isinstance(obj,(list,tuple)):
            h.update(type(obj).__name__.encode())
            for item in obj:visit(item)
        else:h.update(repr(obj).encode())
    visit(value);return h.hexdigest()


def checkpoint_verified(folder):
    manifest=read(folder/'manifest.json')
    if manifest['protocol_sha256']!=sha(PROTOCOL):raise RuntimeError('Checkpoint protocol differs')
    for name,expected in manifest['files'].items():
        if not Path(name).is_file() or sha(Path(name))!=expected:raise RuntimeError('Checkpoint changed: '+name)
    return manifest


def restore_optimizers_and_rng(torch,folder,optimizers):
    manifest=checkpoint_verified(folder)
    states=torch.load(folder/'optimizers.pt',map_location='cpu',weights_only=True)
    for key,opt in optimizers.items():
        opt.load_state_dict(states[key])
        if nested_hash(opt.state_dict())!=manifest['optimizer_hashes'][key]:raise RuntimeError('Optimizer reload differs: '+key)
    rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    return manifest


def save_checkpoint(torch,model,head,folder,update,stage,optimizers,progress=None):
    if folder.exists():raise RuntimeError('Checkpoint already exists; no implicit resume')
    folder.mkdir(parents=True)
    model.save_pretrained(folder,selected_adapters=['actor','reference','discriminator'])
    torch.save(head.state_dict(),folder/'score-head.pt')
    torch.save({k:v.state_dict() for k,v in optimizers.items()},folder/'optimizers.pt')
    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),folder/'rng.pt')
    dump(folder/'manifest.json',dict(update=update,stage=stage,role_hashes={r:role_hash(model,r) for r in ['actor','reference','discriminator']},
        head_sha256=sha(folder/'score-head.pt'),protocol_sha256=sha(PROTOCOL),
        optimizer_hashes={k:nested_hash(v.state_dict()) for k,v in optimizers.items()},progress=progress,files=files_at(folder)))


def preflight():
    guard('preflight');torch,tok,model,head=setup();rows=read(RUN/'train200.json')
    row=max(rows,key=lambda r:len(prompt_ids(tok,r)));initial={r:role_hash(model,r) for r in ['actor','reference','discriminator']}
    if len(set(initial.values()))!=1:raise RuntimeError('NR initialization mismatch')
    records=rollout(torch,tok,model,head,row)
    activate(model,head,'discriminator');dp=[p for p in model.parameters() if p.requires_grad]+list(head.parameters())
    do=torch.optim.AdamW(dp,lr=1e-6,weight_decay=0,foreach=False)
    target=control.supervised_ids(row['user'],read(OLD/'responses/teacher'/(row['id']+'.json'))['content'],'N',tok)
    metric=update_d(torch,model,head,do,target['input_ids'],records)
    assert role_hash(model,'actor')==initial['actor'] and role_hash(model,'reference')==initial['reference']
    assert all(p.grad is None for n,p in model.named_parameters() if '.actor.' in n or '.reference.' in n)
    with torch.no_grad():
        activate(model,head,'discriminator');rewards=torch.stack([dscore(torch,model,head,r['input_ids']) for r in records]);advantages=group_advantages(rewards)
        activate(model,head,'reference');reference=[logprobs(torch,model,r).cpu() for r in records]
        activate(model,head,'actor');old=[logprobs(torch,model,r).cpu() for r in records]
    activate(model,head,'actor');ap=[p for p in model.parameters() if p.requires_grad]
    ao=torch.optim.AdamW(ap,lr=1e-6,weight_decay=0,foreach=False)
    actor_step(torch,model,head,ao,records,old,reference,advantages)
    assert role_hash(model,'actor')!=initial['actor'] and role_hash(model,'reference')==initial['reference']
    out=RUN/'gpu-preflight-checkpoint';save_checkpoint(torch,model,head,out,1,'disposable_preflight',{'actor':ao,'discriminator':do})
    expected_random=torch.rand(8,device='cuda').cpu()
    probe=records[0]
    activate(model,head,'actor');model.eval()
    with torch.no_grad():live=logprobs(torch,model,probe).cpu()
    hashes={r:role_hash(model,r) for r in initial};peak=torch.cuda.max_memory_allocated()/2**20
    del model,head,dp,ap,do,ao;gc.collect();torch.cuda.empty_cache()
    torch,tok,model,head=setup(out)
    activate(model,head,'actor');ao=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6,weight_decay=0,foreach=False)
    activate(model,head,'discriminator');do=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(head.parameters()),lr=1e-6,weight_decay=0,foreach=False)
    restore_optimizers_and_rng(torch,out,{'actor':ao,'discriminator':do})
    assert torch.equal(expected_random,torch.rand(8,device='cuda').cpu())
    activate(model,head,'actor');model.eval()
    with torch.no_grad():again=logprobs(torch,model,probe).cpu()
    error=float((live-again).abs().max())
    assert error==0 and hashes=={r:role_hash(model,r) for r in initial}
    longest=max(read(RUN/'validation64-items.json'),key=lambda r:len(prompt_ids(tok,r)))
    long_response=rollout(torch,tok,model,head,longest,n=1,greedy=True)[0]
    dump(RUN/'gpu-preflight.json',dict(status='PASS',optimizer_updates_disposable=2,training_updates=0,
        prompt_tokens=probe['prompt_length'],fresh_real_rollouts=8,finish_reasons=dict(Counter(r['finish_reason'] for r in records)),
        actor_reference_discriminator_isolation=True,reload_max_logprob_error=error,peak_cuda_mib=max(peak,torch.cuda.max_memory_allocated()/2**20),discriminator=metric,
        resume_verified=True,optimizer_state_and_rng_exact=True,long_validation_prompt_tokens=long_response['prompt_length'],
        long_validation_finish_reason=long_response['finish_reason']))


def actor_step(torch,model,head,optimizer,records,old,reference,advantages):
    activate(model,head,'actor');model.train();optimizer.zero_grad(set_to_none=True)
    before=role_hash(model,'actor');disc=role_hash(model,'discriminator');ref=role_hash(model,'reference');losses=[]
    for i,r in enumerate(records):
        loss=clipped_grpo(logprobs(torch,model,r),old[i].to('cuda'),reference[i].to('cuda'),advantages[i])/len(records)
        loss.backward();losses.append(float(loss.detach()))
    params=[p for p in model.parameters() if p.requires_grad];norm=torch.nn.utils.clip_grad_norm_(params,1)
    if not bool(torch.isfinite(norm)):raise RuntimeError('Invalid actor gradient')
    if float(norm)==0:
        optimizer.zero_grad(set_to_none=True)
        return dict(loss=sum(losses),gradient_norm=0.,before=before,after=before,optimizer_updated=False,
            zero_signal_group=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert role_hash(model,'reference')==ref and role_hash(model,'discriminator')==disc
    if role_hash(model,'actor')==before:raise RuntimeError('Actor parameter hash unchanged')
    return dict(loss=sum(losses),gradient_norm=float(norm),before=before,after=role_hash(model,'actor'),optimizer_updated=True)


def training_progress(out,phase,completed,actual_updates):
    return dict(phase=phase,completed=completed,actual_updates=actual_updates,
        log_hashes={name:sha(out/name) for name in ['warmup.jsonl','updates.jsonl'] if (out/name).exists()})


def interrupt_training(torch,model,head,optimizers,out,trial,progress):
    folder=out/'checkpoints'/f"interrupted-{progress['phase']}{progress['completed']:04d}"
    save_checkpoint(torch,model,head,folder,progress['completed'],trial,optimizers,progress)
    dump(out/'interruption-checkpoint.json',dict(status='SAVED_AT_COMPLETE_ITEM_BOUNDARY',trial=trial,
        checkpoint=str(folder),checkpoint_manifest_sha256=sha(folder/'manifest.json'),progress=progress,
        resume_verified=read(RUN/'gpu-preflight.json').get('resume_verified') is True))
    heartbeat('INTERRUPTED_SAVED',trial=trial,**progress)
    raise GracefulStop('Training saved at complete item boundary')


def train(trial,resume=False):
    guard('train')
    preflight_result=read(RUN/'gpu-preflight.json')
    if preflight_result['status']!='PASS':raise RuntimeError('Real GPU preflight required')
    if (RUN/'models-frozen.json').exists():raise RuntimeError('Final evaluation already released; training prohibited')
    out=RUN/'trials'/trial
    checkpoint=None;progress=dict(phase='warmup' if trial=='gad' else 'policy',completed=0,actual_updates=0)
    if resume:
        if preflight_result.get('resume_verified') is not True:raise RuntimeError('Resume not verified by real-model preflight')
        if (out/'resume-used.json').exists():raise RuntimeError('At most one explicit recovery is permitted')
        marker=read(out/'interruption-checkpoint.json');checkpoint=Path(marker['checkpoint'])
        if marker['trial']!=trial or marker['resume_verified'] is not True or sha(checkpoint/'manifest.json')!=marker['checkpoint_manifest_sha256']:
            raise RuntimeError('Interruption marker does not bind this trial/checkpoint')
        progress=checkpoint_verified(checkpoint)['progress']
        for name,expected in progress['log_hashes'].items():
            if sha(out/name)!=expected:raise RuntimeError('Training log advanced beyond safe resume boundary')
        frozen(out/'resume-used.json',dict(checkpoint_manifest_sha256=marker['checkpoint_manifest_sha256']))
    else:
        if out.exists():raise RuntimeError('Existing trial cannot implicitly resume')
        out.mkdir(parents=True)
    torch,tok,model,head=setup(checkpoint);rows=read(RUN/'train200.json')
    if resume:
        contract=read(out/'trial-contract.json');immutable_reference=contract['immutable_reference'];initial_actor=contract['initial_actor']
    else:
        immutable_reference=role_hash(model,'reference');initial_actor=role_hash(model,'actor')
        frozen(out/'trial-contract.json',dict(trial=trial,initial_actor=initial_actor,immutable_reference=immutable_reference,
            protocol_sha256=sha(PROTOCOL),train_sha256=sha(RUN/'train200.json')))
    activate(model,head,'actor');ao=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6,weight_decay=0,foreach=False)
    activate(model,head,'discriminator');do=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(head.parameters()),lr=1e-6,weight_decay=0,foreach=False)
    optimizers={'actor':ao,'discriminator':do}
    if checkpoint:restore_optimizers_and_rng(torch,checkpoint,optimizers)
    if role_hash(model,'reference')!=immutable_reference:raise RuntimeError('Frozen N-R reference changed')
    if trial=='gad' and progress['phase']=='warmup':
        # Existing SeqKD generator warmup is reused. D warmup is sequential and
        # explicitly not the paper's simultaneous G/D warmup.
        for i,row in enumerate(rows,1):
            if i<=progress['completed']:continue
            records=rollout(torch,tok,model,head,row,n=1)
            teacher=control.supervised_ids(row['user'],read(OLD/'responses/teacher'/(row['id']+'.json'))['content'],'N',tok)
            metric=update_d(torch,model,head,do,teacher['input_ids'],records)
            assert role_hash(model,'actor')==initial_actor and role_hash(model,'reference')==immutable_reference
            append(out/'warmup.jsonl',dict(update=i,id=row['id'],fresh_responses=records,**metric))
            state=training_progress(out,'warmup',i,0)
            if i%5==0:save_checkpoint(torch,model,head,out/'checkpoints'/f'warmup{i:04d}',i,'D_sequential_warmup',optimizers,state)
            heartbeat('D_WARMUP_ITEM_COMPLETE',trial=trial,completed=i)
            if STOP_REQUESTED:interrupt_training(torch,model,head,optimizers,out,trial,state)
        progress=dict(phase='policy',completed=0,actual_updates=0)
    actual_updates=progress['actual_updates'];start=progress['completed'] if progress['phase']=='policy' else 0
    if start and start%50==0:
        try:evaluate_rows(torch,tok,model,head,read(RUN/'validation64-items.json'),out/'validation'/f'step{start:04d}',read(RUN/'validation64-gold.local.json'))
        except GracefulStop:interrupt_training(torch,model,head,optimizers,out,trial,training_progress(out,'policy',start,actual_updates))
    for i,row in enumerate(rows,1):
        if i<=start:continue
        began=time.monotonic()
        if trial=='gad':
            actor_before=role_hash(model,'actor');records=rollout(torch,tok,model,head,row)
            model.eval()
            with torch.no_grad():
                activate(model,head,'actor');old=[logprobs(torch,model,r).cpu() for r in records]
                activate(model,head,'reference');reference=[logprobs(torch,model,r).cpu() for r in records]
                activate(model,head,'discriminator');rewards=torch.stack([dscore(torch,model,head,r['input_ids']) for r in records]);advantages=group_advantages(rewards)
            assert actor_before==role_hash(model,'actor')
            metric=actor_step(torch,model,head,ao,records,old,reference,advantages)
            teacher=control.supervised_ids(row['user'],read(OLD/'responses/teacher'/(row['id']+'.json'))['content'],'N',tok)
            dm=update_d(torch,model,head,do,teacher['input_ids'],records)
            actual_updates+=int(metric['optimizer_updated'])
            record=dict(fresh_responses=records,rewards=rewards.tolist(),advantages=advantages.tolist(),discriminator=dm,**metric)
        elif trial=='seqkd':
            activate(model,head,'actor');model.train();ao.zero_grad(set_to_none=True)
            teacher=control.supervised_ids(row['user'],read(OLD/'responses/teacher'/(row['id']+'.json'))['content'],'N',tok)
            ids=torch.tensor([teacher['input_ids']],device='cuda');labels=torch.tensor([teacher['labels']],device='cuda')
            loss=control.migration.chunked_clm_loss(hidden(model,ids),model.get_output_embeddings().weight,ids,128,labels)*teacher['supervised_tokens']/(teacher['answer_bytes']+1)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)) or float(norm)<=0:raise RuntimeError('Nonfinite SeqKD update')
            ao.step();ao.zero_grad(set_to_none=True);actual_updates+=1;record=dict(method='continued SeqKD control, not GAD',loss=float(loss.detach()),gradient_norm=float(norm),optimizer_updated=True)
            del ids,labels,loss
        else:raise ValueError('Only default GAD/continued SeqKD registered; third fix requires a separately frozen amendment')
        assert role_hash(model,'reference')==immutable_reference
        append(out/'updates.jsonl',dict(update=i,id=row['id'],source_index=row['source_index'],seconds=time.monotonic()-began,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,**record))
        state=training_progress(out,'policy',i,actual_updates)
        if i%5==0:save_checkpoint(torch,model,head,out/'checkpoints'/f'policy{i:04d}',i,trial,optimizers,state)
        heartbeat('POLICY_ITEM_COMPLETE',trial=trial,completed=i,actual_optimizer_updates=actual_updates)
        if STOP_REQUESTED:interrupt_training(torch,model,head,optimizers,out,trial,state)
        if i%50==0:
            try:evaluate_rows(torch,tok,model,head,read(RUN/'validation64-items.json'),out/'validation'/f'step{i:04d}',read(RUN/'validation64-gold.local.json'))
            except GracefulStop:interrupt_training(torch,model,head,optimizers,out,trial,state)
    dump(out/'training-complete.json',dict(status='TRAINING_COMPLETE_RELOAD_PENDING',source_groups=200,actual_optimizer_updates=actual_updates,
        zero_signal_groups=200-actual_updates,initial_actor=initial_actor,final_actor=role_hash(model,'actor'),reference_sha256=immutable_reference,final_checkpoint=str(out/'checkpoints/policy0200')))


def evaluate_rows(torch,tok,model,head,rows,out,gold,base_only=False):
    out.mkdir(parents=True,exist_ok=True)
    frozen(out/'evaluation-contract.json',dict(protocol_sha256=sha(PROTOCOL),base_only=base_only,
        actor_sha256=role_hash(model,'actor'),rows_sha256=control.digest(json.dumps(rows,sort_keys=True)),
        gold_sha256=control.digest(json.dumps(gold,sort_keys=True))))
    for row in rows:
        if STOP_REQUESTED:raise GracefulStop('Evaluation saved complete responses; explicit supervised restart required')
        if (out/(row['id']+'.json')).exists():
            saved=read(out/(row['id']+'.json'))
            if saved['id']!=row['id'] or saved['prompt_sha256']!=row['prompt_sha256']:raise RuntimeError('Saved evaluation membership/input differs')
            continue
        response=rollout(torch,tok,model,head,row,n=1,greedy=True,base_only=base_only)[0]
        frozen(out/(row['id']+'.json'),dict(id=row['id'],prompt_sha256=row['prompt_sha256'],**response))
        heartbeat('EVALUATION_ITEM_SAVED',folder=str(out),id=row['id'])
        # Evaluation can be restarted from immutable responses. The trainer
        # finishes this validation before saving a safe interruption marker.
    # This is the existing independent exact SQL scorer, never discriminator.
    bird=control.load_bird();bird.execute_summary=execute_sql
    def one(row):
        g=gold[row['id']];expected=bird.execute_summary(g['db_path'],g['sql']);r=read(out/(row['id']+'.json'))
        if expected['status']!='ok':return dict(id=row['id'],official_ex=False,gold_technical_failure=True,finish_reason=r['finish_reason'])
        return dict(id=row['id'],finish_reason=r['finish_reason'],**bird.score_response(r['content'],dict(g,expected=expected['summary']),finish_reason=r['finish_reason']))
    with ThreadPoolExecutor(max_workers=2) as pool:scores=list(pool.map(one,rows))
    dump(out/'scores.json',scores)
    summary=dict(denominator=len(rows),correct=sum(bool(r.get('official_ex')) for r in scores),parse_ok=sum(bool(r.get('parse_ok')) for r in scores),truncated=sum(r['finish_reason']=='length' for r in scores),gold_technical_failures=sum(bool(r.get('gold_technical_failure')) for r in scores))
    dump(out/'summary.json',summary);return summary


def select_models():
    """CPU-only checkpoint selection on validation, before final gold release."""
    verify_integrity();result={}
    for trial in ['gad','seqkd']:
        out=RUN/'trials'/trial;complete=read(out/'training-complete.json')
        assert complete['source_groups']==200 and 0<complete['actual_optimizer_updates']<=200
        updates=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in updates]==list(range(1,201))
        assert [r['id'] for r in updates]==[r['id'] for r in read(RUN/'train200.json')]
        choices=[]
        for step in [50,100,150,200]:
            p=out/'validation'/f'step{step:04d}'/'summary.json';s=read(p)
            assert s['denominator']==64 and s['gold_technical_failures']==0
            choices.append(((-s['correct'],-s['parse_ok'],s['truncated'],step),step,s,sha(p)))
        _,step,summary,summary_sha=min(choices)
        checkpoint=out/'checkpoints'/f'policy{step:04d}'
        result[trial]=dict(step=step,validation=summary,validation_sha256=summary_sha,
            checkpoint=str(checkpoint),checkpoint_manifest_sha256=sha(checkpoint/'manifest.json'))
    frozen(RUN/'selected-models.json',result)
    print(json.dumps(result,indent=2))


def reload_selected(trial):
    guard('reload');selected=read(RUN/'selected-models.json')[trial];folder=Path(selected['checkpoint'])
    manifest=checkpoint_verified(folder)
    torch,tok,model,head=setup(folder);actual={r:role_hash(model,r) for r in ['actor','reference','discriminator']}
    assert actual==manifest['role_hashes']
    assert sha(folder/'score-head.pt')==manifest['head_sha256']
    activate(model,head,'actor');model.eval()
    row=read(RUN/'train200.json')[0]
    response=read(OLD/'responses/teacher'/(row['id']+'.json'))['content']
    packed=control.supervised_ids(row['user'],response,'N',tok)
    with torch.no_grad():value=logprobs(torch,model,packed)
    assert bool(torch.isfinite(value).all())
    frozen(RUN/f'{trial}-selected-reload.json',dict(status='PASS',checkpoint_manifest_sha256=sha(folder/'manifest.json'),
        actual_role_hashes=actual,finite_teacher_logprob_tokens=value.numel(),
        verification='independent process load exact parameter hashes and finite actual forward; GPU preflight separately checks exact output reload'))


def freeze_models():
    verify_integrity();selected=read(RUN/'selected-models.json');frozen_models={}
    for trial in ['gad','seqkd']:
        check=read(RUN/f'{trial}-selected-reload.json');assert check['status']=='PASS'
        assert check['checkpoint_manifest_sha256']==selected[trial]['checkpoint_manifest_sha256']
        frozen_models[trial]=dict(selected[trial],files=files_at(Path(selected[trial]['checkpoint'])))
    frozen(RUN/'models-frozen.json',dict(models=frozen_models,initial_N_R=read(RUN/'cache-audit.json')['nr_checkpoint_files'],
        protocol_sha256=sha(PROTOCOL),final_gold_opened_before_freeze=False))
    # Materialize full final prompts/gold only once all model and recipe choices freeze.
    rows,gold=materialize(read(RUN/'final200-identities.sealed.json'))
    expected={r['id']:r['prompt_sha256'] for r in read(RUN/'final200-input-integrity.json')}
    if {r['id']:r['prompt_sha256'] for r in rows}!=expected:raise RuntimeError('Final inputs differ from prelaunch audit')
    frozen(RUN/'final200-items.json',rows);frozen(RUN/'final200-gold.local.json',gold)


def final_evaluate(arm):
    guard('final-eval');freeze=read(RUN/'models-frozen.json')
    if arm in ['gad','seqkd']:
        checkpoint=Path(freeze['models'][arm]['checkpoint'])
        assert files_at(checkpoint)==freeze['models'][arm]['files']
    else:checkpoint=None
    torch,tok,model,head=setup(checkpoint)
    if arm=='N-0':
        # PEFT disabled for all forwards/generation, restoring original native base.
        with model.disable_adapter():
            evaluate_rows(torch,tok,model,head,read(RUN/'final200-items.json'),RUN/'final-evaluation'/arm,read(RUN/'final200-gold.local.json'),base_only=True)
    else:
        evaluate_rows(torch,tok,model,head,read(RUN/'final200-items.json'),RUN/'final-evaluation'/arm,read(RUN/'final200-gold.local.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','train','select','reload','freeze','final-eval']);parser.add_argument('--trial',choices=['gad','seqkd'],default='gad')
    parser.add_argument('--eval-arm',choices=['N-0','N-R','gad','seqkd'],default='gad')
    parser.add_argument('--resume',action='store_true',help='Explicitly restore one saved training interruption boundary')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,request_stop)
    if args.command=='prepare':prepare()
    elif args.command=='preflight':preflight()
    elif args.command=='train':train(args.trial,args.resume)
    elif args.command=='select':select_models()
    elif args.command=='reload':reload_selected(args.trial)
    elif args.command=='freeze':freeze_models()
    else:final_evaluate(args.eval_arm)


if __name__=='__main__':
    try:main()
    except GracefulStop as exc:
        print(str(exc),flush=True);sys.exit(75)
