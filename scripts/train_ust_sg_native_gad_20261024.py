#!/usr/bin/env python3
"""Task-generic native-vocabulary GAD. No API client or SQL/data-generation code."""
from __future__ import annotations
import argparse, gc, json, os, signal, sys, time, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import eval_ust_sg_qwen_modes_20261024 as q
from privacy_synth.onpolicy_gad import response_log_probs,group_advantages
# Reuse the historical GPU-validated tensor/update/adapter isolation functions.
from bird_gad_night_20261022 import (activate,role_hash,hidden,logprobs,dscore,
    update_d,actor_step,nested_hash)
TRAIN_ROOT=ROOT/'runs/2026-10-24-ust-sg-onpolicy'
STOP=False
def stop(*_):
    global STOP
    STOP=True

def rows(spec):
    p=q.bound(spec);return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def teacher_rows(manifest_path,expected):
    m=q.load(manifest_path)
    if m.get('all_valid') is False:raise PermissionError('Frozen teacher collection contains invalid rows')
    if m.get('count',len(expected))!=len(expected):raise PermissionError('Teacher declared count mismatch')
    for binding in m.get('bindings',[]):q.bound(binding)
    data=rows(m['records_file']) if 'records_file' in m else m['records']
    ids=[str(x['id']) for x in data]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected) or len(ids)!=len(expected):
        raise PermissionError('Teacher membership differs; no dropping/filling/repeating')
    by={str(x['id']):x for x in data}
    for x in data:
        if 'source_result' in x:q.bound(x['source_result'])
        if x.get('valid') is not True or not isinstance(x.get('text'),str) or not x['text'].strip():
            raise PermissionError('Invalid teacher row blocks complete training qualification')
        if x.get('finish_reason','stop')!='stop':raise PermissionError('Truncated teacher row prohibited')
    return by

def data(manifest,teacher):
    m=q.load(manifest);train=rows(m['phases']['train']['queries']);val=rows(m['phases']['validation']['queries'])
    for phase,items in [('train',train),('validation',val)]:
        if len(items)!=m['phases'][phase]['count']:raise PermissionError('Frozen count mismatch')
        if len({str(x['id']) for x in items})!=len(items):raise PermissionError('Duplicate ID')
        for r in items:
            if [x['role'] for x in r['messages']]!=['system','user']:raise PermissionError('Expected exact zero-shot two-message Train/Val')
    if {str(x['group']) for x in train}&{str(x['group']) for x in val}:raise PermissionError('Train/Val group leakage')
    return m,train,val,teacher_rows(teacher,[str(r['id']) for r in train])

def prompt(tok,row):
    text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok.encode(text,add_special_tokens=False)
    if not ids or len(ids)+2048>32768:raise PermissionError('Full native prompt exceeds bound; truncation forbidden')
    return ids

def pack(tok,row,text):
    prefix=prompt(tok,row)
    answer=tok.encode(text,add_special_tokens=False)
    if not answer or tok.decode(answer,skip_special_tokens=False)!=text:raise PermissionError('Teacher native roundtrip differs')
    eos=tok.convert_tokens_to_ids('<|im_end|>')
    if eos is None or eos<0:raise PermissionError('Native EOS missing')
    if answer[-1]!=eos:answer.append(eos)
    if len(prefix)+len(answer)>32768:raise PermissionError('Teacher context exceeds bound')
    return dict(input_ids=prefix+answer,prompt_length=len(prefix))

def protocol(manifest,teacher):
    m,tr,va,_=data(manifest,teacher)
    return dict(task=m['task'],manifest_path=str(Path(manifest).resolve()),manifest_sha256=q.sha(manifest),
        teacher_manifest_path=str(Path(teacher).resolve()),teacher_manifest_sha256=q.sha(teacher),
        trainer_sha256=q.sha(__file__),dependencies={str(ROOT/p):q.sha(ROOT/p) for p in [
            'src/privacy_synth/onpolicy_gad.py','scripts/bird_gad_night_20261022.py',
            'scripts/preflight_qwen35_lora.py','scripts/eval_ust_sg_qwen_modes_20261024.py']},
        runtime=q.runtime_bindings(),base=str(q.DEFAULT_MODEL),base_assets={p.name:q.sha(p) for p in sorted(q.DEFAULT_MODEL.iterdir()) if p.is_file()},
        train_count=len(tr),validation_count=len(va),native_vocabulary=True,thinking='off',
        new_lora_r=8,lora_alpha=16,lora_dropout=0,actor_sft_epochs=1,
        sequential_discriminator_warmup_epochs=1,policy_groups=len(tr),group_size=8,
        rollout_max_new_tokens=256,temperature=.8,top_k=0,top_p=1,
        actor_lr=1e-6,discriminator_lr=1e-6,clip=.2,beta=.001,
        validation_steps=sorted(set(max(1,len(tr)*i//4) for i in range(1,5))),selection='correct then valid then fewer capped then earlier',
        no_gold_in_training=True,external_calls=0,prior_bird_adapter_transferred=False)

def prepare(manifest,teacher,out):
    p=protocol(manifest,teacher);q.immutable(out/'protocol.json',p)
    # Tokenizer-only native roundtrip and complete-context checks; no weight/GPU load.
    q.setup_native()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(q.DEFAULT_MODEL),local_files_only=True,trust_remote_code=False)
    _,tr,va,teachers=data(manifest,teacher)
    audit=[dict(id=str(r['id']),prompt_tokens=len(prompt(tok,r)),
        teacher_tokens=len(pack(tok,r,teachers[str(r['id'])]['text'])['input_ids'])-len(prompt(tok,r))) for r in tr]
    q.immutable(out/'token-audit.json',dict(train=audit,validation=[dict(id=str(r['id']),prompt_tokens=len(prompt(tok,r))) for r in va],
        tokenizer_only=True,gpu_used=False))
    return dict(status='PREPARED',protocol_sha256=q.sha(out/'protocol.json'))

def verify(out):
    p=q.load(out/'protocol.json')
    if protocol(p['manifest_path'],p['teacher_manifest_path'])!=p:raise PermissionError('Frozen training preparation changed')
    auth=q.load(out/'authorization.json')
    if auth.get('approved') is not True or auth.get('protocol_sha256')!=q.sha(out/'protocol.json'):
        raise PermissionError('Bound local training authorization missing')
    return p

def setup(out,checkpoint=None):
    p=verify(out)
    import run_cityfm_repair_qwen_20261022 as guard
    from run_ust_sg_fast_profile_20261024 import resources_ok
    if not resources_ok(guard.probe(out),True)[0]:raise RuntimeError('Immediate resource floor unmet; no stable idle wait')
    q.setup_native()
    import torch
    from transformers import AutoTokenizer,AutoModelForImageTextToText
    from peft import LoraConfig,get_peft_model,PeftModel
    from preflight_qwen35_lora import _text_lora_modules
    torch.set_num_threads(4);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
    torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
    tok=AutoTokenizer.from_pretrained(p['base'],local_files_only=True,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(p['base'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
    if checkpoint:
        model=PeftModel.from_pretrained(base,checkpoint/'actor',adapter_name='actor',is_trainable=True)
        for name in ('reference','discriminator'):model.load_adapter(checkpoint/name,adapter_name=name,is_trainable=name!='reference')
    else:
        config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=_text_lora_modules(base))
        model=get_peft_model(base,config,adapter_name='actor')
        for name in ('reference','discriminator'):model.add_adapter(name,config)
    head=torch.nn.Linear(model.get_input_embeddings().weight.shape[1],1,bias=False,device='cuda',dtype=torch.float32)
    if checkpoint:head.load_state_dict(torch.load(checkpoint/'score-head.pt',map_location='cuda',weights_only=True))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    activate(model,head,'actor')
    return torch,tok,model,head

def clone_warmed(model):
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    state={k:v.detach().clone() for k,v in get_peft_model_state_dict(model,adapter_name='actor').items()}
    for name in ('reference','discriminator'):set_peft_model_state_dict(model,state,adapter_name=name)
    if len({role_hash(model,r) for r in ('actor','reference','discriminator')})!=1:raise RuntimeError('Warm clone differs')

def rollouts(torch,tok,model,head,row,n=8,greedy=False,thinking='off',messages=None):
    activate(model,head,'actor');model.eval()
    if messages is None:prefix=prompt(tok,row)
    else:
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking=='on')
        prefix=tok.encode(text,add_special_tokens=False)
    cap=8192 if thinking=='on' else 256
    if len(prefix)+cap>32768:raise RuntimeError('Complete input+output exceeds context; no clipping')
    stops=[tok.convert_tokens_to_ids(x) for x in ('<|im_end|>','<|endoftext|>')]
    result=[]
    for i in range(n):
        ids=torch.tensor([prefix],device='cuda');started=time.monotonic()
        kwargs=dict(max_new_tokens=cap,do_sample=not greedy,num_beams=1,eos_token_id=stops,
            pad_token_id=tok.pad_token_id,use_cache=True)
        if not greedy:kwargs.update(temperature=.8,top_k=0,top_p=1.)
        with torch.inference_mode():seq=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),**kwargs)
        emitted=seq[0,len(prefix):].tolist()
        if not emitted:raise RuntimeError('Empty generated tokens')
        raw=tok.decode(emitted,skip_special_tokens=False)
        result.append(dict(input_ids=prefix+emitted,prompt_length=len(prefix),response_ids=emitted,
            content=tok.decode(emitted,skip_special_tokens=True),raw_generated=raw,
            finish_reason='stop' if emitted[-1] in stops else 'length',group_index=i,
            generation_seconds=time.monotonic()-started))
        del seq,ids
    return result

def checkpoint(out,folder,torch,model,head,optimizers,stage,step,actual_updates=0):
    if folder.exists():raise PermissionError('No overwrite/implicit training resume')
    def size(value):
        if hasattr(value,'numel'):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(size(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(map(size,value))
        return 0
    estimate=sum(v.numel()*v.element_size() for n,v in model.named_parameters() if 'lora_' in n)
    estimate+=sum(size(opt.state_dict()) for opt in optimizers.values())+64*2**20
    current=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if current+estimate>4*2**30:raise RuntimeError('Checkpoint staging would exceed4GiB task output cap')
    folder.mkdir(parents=True)
    model.save_pretrained(folder,selected_adapters=['actor','reference','discriminator'])
    torch.save(head.state_dict(),folder/'score-head.pt')
    torch.save({k:v.state_dict() for k,v in optimizers.items()},folder/'optimizers.pt')
    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),folder/'rng.pt')
    q.immutable(folder/'manifest.json',dict(protocol_sha256=q.sha(out/'protocol.json'),stage=stage,step=step,
        actual_updates=actual_updates,role_hashes={r:role_hash(model,r) for r in ('actor','reference','discriminator')},
        optimizer_hashes={k:nested_hash(v.state_dict()) for k,v in optimizers.items()},
        files={str(p.relative_to(folder)):q.sha(p) for p in folder.rglob('*') if p.is_file()}))

def verify_checkpoint(out,folder):
    m=q.load(folder/'manifest.json')
    if m['protocol_sha256']!=q.sha(out/'protocol.json'):raise PermissionError('Checkpoint protocol mismatch')
    for f,h in m['files'].items():
        if q.sha(folder/f)!=h:raise PermissionError('Checkpoint changed')
    return m

def optimizers(torch,model,head):
    activate(model,head,'actor')
    actor=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-6,weight_decay=0,foreach=False)
    activate(model,head,'discriminator')
    disc=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad]+list(head.parameters()),lr=1e-6,weight_decay=0,foreach=False)
    return dict(actor=actor,discriminator=disc)

def policy(torch,model,head,opt,records):
    model.eval()
    with torch.no_grad():
        activate(model,head,'actor');old=[logprobs(torch,model,r).cpu() for r in records]
        activate(model,head,'reference');ref=[logprobs(torch,model,r).cpu() for r in records]
        activate(model,head,'discriminator');rewards=torch.stack([dscore(torch,model,head,r['input_ids']) for r in records])
        advantages=group_advantages(rewards)
    return actor_step(torch,model,head,opt,records,old,ref,advantages),rewards.tolist(),advantages.tolist()

def policy_signal_summary(work,count):
    """Count saved adversarial signals separately from all optimizer updates."""
    updates=zero_adversarial=0
    for i in range(1,count+1):
        record=q.load(work/'policy'/f'{i:04d}.json')
        advantages=record['advantages']
        if not advantages:raise PermissionError('Saved policy group has no advantages')
        updates+=int(record['actor']['optimizer_updated'])
        zero_adversarial+=int(all(float(a)==0. for a in advantages))
    return dict(actual_optimizer_updates=updates,zero_gradient_groups=count-updates,
        zero_adversarial_signal_groups=zero_adversarial)

def evaluate_validation(out,folder,torch,tok,model,head,items,m):
    from run_llm_task_sweep_20261023 import score_prediction
    q.immutable(folder/'contract.json',dict(protocol_sha256=q.sha(out/'protocol.json'),actor_sha256=role_hash(model,'actor')))
    result=[]
    for row in items:
        path=folder/(q.digest(str(row['id']))+'.json')
        if path.exists():
            result.append(q.load(path));continue
        if path.with_suffix('.started.json').exists():raise PermissionError('Unresolved validation sample; no replay')
        q.immutable(path.with_suffix('.started.json'),dict(id=str(row['id'])))
        response=rollouts(torch,tok,model,head,row,n=1,greedy=True)[0]
        parsed=q.classify(response['raw_generated'],'off',m['phases']['validation']['scorer'],row.get('metadata',{}),response['finish_reason'])
        record=dict(id=str(row['id']),**parsed,finish_reason=response['finish_reason'])
        q.immutable(folder/(q.digest(str(row['id']))+'.json'),record);result.append(record)
    q.immutable(folder/'predictions-frozen.json',dict(complete=True,count=len(items),
        predictions=[dict(path=str(folder/(q.digest(str(r['id']))+'.json')),sha256=q.sha(folder/(q.digest(str(r['id']))+'.json'))) for r in items]))
    gold=q.load(q.bound(m['phases']['validation']['gold']))
    counts=[score_prediction(r,dict(kind='qa',scorer=m['phases']['validation']['scorer']),gold[r['id']],items[i].get('metadata',{})) for i,r in enumerate(result)]
    summary=dict(correct=sum(c for c,v in counts),valid=sum(v for c,v in counts),total=len(items),
        length_capped=sum(r['finish_reason']=='length' for r in result))
    q.immutable(folder/'scores.json',summary);return summary

def preflight(out):
    p=verify(out);m,tr,va,teachers=data(p['manifest_path'],p['teacher_manifest_path'])
    folder=out/'disposable-preflight'
    if folder.exists():raise PermissionError('Preflight exists; no automatic rerun')
    torch,tok,model,head=setup(out);clone_warmed(model);opts=optimizers(torch,model,head)
    before={r:role_hash(model,r) for r in ('actor','reference','discriminator')}
    row=max(tr,key=lambda r:len(prompt(tok,r)));fresh=rollouts(torch,tok,model,head,row)
    dm=update_d(torch,model,head,opts['discriminator'],pack(tok,row,teachers[str(row['id'])]['text'])['input_ids'],fresh)
    if role_hash(model,'actor')!=before['actor'] or role_hash(model,'reference')!=before['reference']:raise RuntimeError('D changed frozen role')
    metric,_,_=policy(torch,model,head,opts['actor'],fresh)
    if not metric['optimizer_updated']:raise RuntimeError('Functional preflight yielded no policy gradient')
    checkpoint(out,folder,torch,model,head,opts,'DISPOSABLE',1)
    expected_random=torch.rand(8,device='cuda').cpu()
    activate(model,head,'actor');model.eval()
    with torch.no_grad():live=logprobs(torch,model,fresh[0]).cpu()
    peak=torch.cuda.max_memory_allocated()
    del model,head,opts;gc.collect();torch.cuda.empty_cache()
    torch,tok,model,head=setup(out,folder);cm=verify_checkpoint(out,folder)
    opts=optimizers(torch,model,head)
    states=torch.load(folder/'optimizers.pt',map_location='cpu',weights_only=True)
    for name,opt in opts.items():
        opt.load_state_dict(states[name])
        if nested_hash(opt.state_dict())!=cm['optimizer_hashes'][name]:raise RuntimeError('Optimizer reload differs')
    rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    if not torch.equal(expected_random,torch.rand(8,device='cuda').cpu()):raise RuntimeError('RNG reload differs')
    if {r:role_hash(model,r) for r in before}!=cm['role_hashes']:raise RuntimeError('Reload adapter hash differs')
    activate(model,head,'actor');model.eval()
    with torch.no_grad():again=logprobs(torch,model,fresh[0]).cpu()
    error=float((live-again).abs().max())
    if error!=0:raise RuntimeError('Independent reload forward differs')
    q.immutable(out/'gpu-preflight.json',dict(status='PASS',disposable_optimizer_updates=2,training_updates=0,
        fresh_rollouts=8,native_vocabulary=True,optimizer_state_and_rng_exact=True,reload_max_logprob_error=error,peak_allocated_bytes=peak,
        discriminator=dm,actor=metric,protocol_sha256=q.sha(out/'protocol.json')))
    return dict(status='PASS',gpu_preflight_sha256=q.sha(out/'gpu-preflight.json'))

def retention(out,latest):
    """Delete only verified owned checkpoint payloads; immutable tombstones survive."""
    if not out.resolve().is_relative_to(TRAIN_ROOT.resolve()):raise PermissionError('Checkpoint output outside owned new task run')
    work=(out/'training').resolve();base=(work/'checkpoints').resolve()
    if base.parent!=work or not base.is_relative_to(out.resolve()):raise PermissionError('Checkpoint path escaped task output')
    keep={f"W{q.load(out/'protocol.json')['train_count']:04d}",latest.name}
    scored=[]
    for p in (work/'validation').glob('policy*/scores.json'):
        value=q.load(p);step=int(p.parent.name.removeprefix('policy'))
        scored.append(((-value['correct'],-value['valid'],value['length_capped'],step),p.parent.name))
    if scored:keep.add(min(scored)[1])
    for folder in sorted(base.iterdir()) if base.exists() else []:
        if not folder.is_dir() or folder.name in keep:continue
        resolved=folder.resolve()
        if resolved.parent!=base or folder.is_symlink():raise PermissionError('Refuse checkpoint symlink/path escape')
        manifest=verify_checkpoint(out,folder)
        allowed=set(manifest['files'])|{'manifest.json'}
        actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
        if actual!=allowed:raise PermissionError('Unexpected checkpoint payload; no deletion')
        for p in folder.rglob('*'):
            if p.is_symlink():raise PermissionError('Refuse symlink checkpoint payload')
        q.immutable(work/'checkpoint-tombstones'/(folder.name+'.json'),dict(
            path=str(folder),manifest_sha256=q.sha(folder/'manifest.json'),manifest=manifest,
            reason='bounded W/best/latest retention',kept=sorted(keep)))
        shutil.rmtree(folder)
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total>4*2**30:raise RuntimeError('Task output exceeds4GiB ceiling; stopped')
    return keep

def restore(torch,model,head,opts,out,folder):
    m=verify_checkpoint(out,folder)
    if {r:role_hash(model,r) for r in m['role_hashes']}!=m['role_hashes']:raise PermissionError('Resume role hashes differ')
    states=torch.load(folder/'optimizers.pt',map_location='cpu',weights_only=True)
    for k,opt in opts.items():
        opt.load_state_dict(states[k])
        if nested_hash(opt.state_dict())!=m['optimizer_hashes'][k]:raise PermissionError('Resume optimizer mismatch')
    rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    return m

def resume_cursor(out):
    work=out/'training';pointer=q.load(work/'latest-checkpoint.json')
    folder=Path(pointer['path'])
    if folder.resolve().parent!=(work/'checkpoints').resolve():raise PermissionError('Resume pointer outside owned checkpoint directory')
    if q.sha(folder/'manifest.json')!=pointer['manifest_sha256']:raise PermissionError('Resume pointer changed')
    m=verify_checkpoint(out,folder);stage=m['stage'];step=m['step']
    for marker in work.rglob('*.started.json'):
        if not marker.with_name(marker.name.replace('.started.json','.json')).exists():
            raise PermissionError('PARTIAL_UNSAFE: unresolved fresh generation; no resampling')
    phases=['sft','warmup','policy'];limits={'sft':0,'warmup':0,'policy':0}
    if stage=='sft':limits['sft']=step
    elif stage=='W':limits['sft']=q.load(out/'protocol.json')['train_count']
    elif stage=='D':limits.update(sft=q.load(out/'protocol.json')['train_count'],warmup=step)
    elif stage=='policy':limits.update(sft=q.load(out/'protocol.json')['train_count'],warmup=q.load(out/'protocol.json')['train_count'],policy=step)
    else:raise PermissionError('Unsupported safe resume stage')
    for phase in phases:
        done=sorted(int(p.stem) for p in (work/phase).glob('*.json') if not p.name.endswith('.started.json'))
        if done!=list(range(1,limits[phase]+1)):
            raise PermissionError('PARTIAL_UNSAFE: saved work differs from exact checkpoint cursor; no replay')
    return folder,m

def resume_check(out):
    verify(out)
    folder,m=resume_cursor(out)
    return dict(status='SAFE_RESUME',checkpoint=str(folder),checkpoint_manifest_sha256=q.sha(folder/'manifest.json'),
        stage=m['stage'],step=m['step'],actual_updates=m.get('actual_updates',0),gpu_used=False)

def train(out,resume=False):
    p=verify(out)
    if q.load(out/'gpu-preflight.json')['status']!='PASS':raise PermissionError('Real fresh-task functional preflight required')
    work=out/'training';folder=None;cursor=dict(stage='sft',step=0,actual_updates=0)
    if resume:folder,cursor=resume_cursor(out)
    else:
        if work.exists():raise PermissionError('Existing training requires explicit verified --resume')
        work.mkdir()
    m,tr,va,teachers=data(p['manifest_path'],p['teacher_manifest_path'])
    torch,tok,model,head=setup(out,folder);opts=optimizers(torch,model,head)
    if folder:restore(torch,model,head,opts,out,folder)
    actual=cursor.get('actual_updates',0)
    def boundary(stage,i):
        folder=work/'checkpoints'/f'{stage}{i:04d}'
        checkpoint(out,folder,torch,model,head,opts,stage,i,actual)
        pointer=dict(path=str(folder),manifest_sha256=q.sha(folder/'manifest.json'))
        tmp=work/'latest-checkpoint.tmp';tmp.write_text(json.dumps(pointer));tmp.replace(work/'latest-checkpoint.json')
        retention(out,folder)
        if STOP:raise RuntimeError('USER_STOP_SAVED_COMPLETE_GROUP_BOUNDARY')
    if cursor['stage']=='sft':
        for i,row in enumerate(tr,1):
            if i<=cursor['step']:continue
            activate(model,head,'actor');model.train();opt=opts['actor'];opt.zero_grad(set_to_none=True)
            packed=pack(tok,row,teachers[str(row['id'])]['text'])
            ids=torch.tensor([packed['input_ids']],device='cuda')
            loss=-response_log_probs(hidden(model,ids),model.get_output_embeddings().weight,ids,packed['prompt_length'],temperature=1.).mean()
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1)
            if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite SFT')
            opt.step();opt.zero_grad(set_to_none=True)
            q.immutable(work/'sft'/f'{i:04d}.json',dict(id=str(row['id']),loss=float(loss.detach()),gradient_norm=float(norm)))
            del ids,loss
            if i%5==0 or STOP:boundary('sft',i)
        clone_warmed(model);opts=optimizers(torch,model,head)
        boundary('W',len(tr));cursor=dict(stage='W',step=len(tr))
    if cursor['stage']=='W':
        evaluate_validation(out,work/'validation'/'W',torch,tok,model,head,va,m)
    reference=role_hash(model,'reference')
    if cursor['stage'] in ('sft','W','D'):
        warm_start=cursor['step'] if cursor['stage']=='D' else 0
        for i,row in enumerate(tr,1):
            if i<=warm_start:continue
            q.immutable(work/'warmup'/f'{i:04d}.started.json',dict(id=str(row['id']),stage='D_WARMUP'))
            fresh=rollouts(torch,tok,model,head,row,n=1)
            metric=update_d(torch,model,head,opts['discriminator'],pack(tok,row,teachers[str(row['id'])]['text'])['input_ids'],fresh)
            q.immutable(work/'warmup'/f'{i:04d}.json',dict(id=str(row['id']),fresh_responses=fresh,**metric))
            if i%5==0 or STOP:boundary('D',i)
        cursor=dict(stage='policy',step=0)
    start=cursor['step'] if cursor['stage']=='policy' else 0
    if start in p['validation_steps'] and not (work/'validation'/f'policy{start:04d}'/'scores.json').exists():
        evaluate_validation(out,work/'validation'/f'policy{start:04d}',torch,tok,model,head,va,m)
    for i,row in enumerate(tr,1):
        if i<=start:continue
        q.immutable(work/'policy'/f'{i:04d}.started.json',dict(id=str(row['id']),stage='POLICY',actor_before=role_hash(model,'actor')))
        fresh=rollouts(torch,tok,model,head,row,n=8)
        metric,rewards,advantages=policy(torch,model,head,opts['actor'],fresh)
        dm=update_d(torch,model,head,opts['discriminator'],pack(tok,row,teachers[str(row['id'])]['text'])['input_ids'],fresh)
        if role_hash(model,'reference')!=reference:raise RuntimeError('Frozen W reference changed')
        actual+=int(metric['optimizer_updated'])
        q.immutable(work/'policy'/f'{i:04d}.json',dict(id=str(row['id']),fresh_responses=fresh,
            rewards=rewards,advantages=advantages,actor=metric,discriminator=dm))
        if i%5==0 or STOP:boundary('policy',i)
        if i in p['validation_steps'] or i==len(tr):
            evaluate_validation(out,work/'validation'/f'policy{i:04d}',torch,tok,model,head,va,m)
            retention(out,work/'checkpoints'/f'policy{i:04d}')
    signals=policy_signal_summary(work,len(tr))
    if signals['actual_optimizer_updates']!=actual:raise RuntimeError('Saved policy update count differs')
    q.immutable(work/'complete.json',dict(policy_groups=len(tr),**signals,reference_sha256=reference))
    return select(out)

def select(out):
    p=verify(out);work=out/'training';complete=q.load(work/'complete.json')
    if complete['policy_groups']!=p['train_count']:raise PermissionError('Incomplete policy groups')
    options=[]
    for step in sorted(set([s for s in p['validation_steps'] if s<=p['train_count']]+[p['train_count']])):
        summary=q.load(work/'validation'/f'policy{step:04d}'/'scores.json')
        options.append(((-summary['correct'],-summary['valid'],summary['length_capped'],step),step,summary))
    _,step,summary=min(options)
    folder=work/'checkpoints'/f'policy{step:04d}'
    verify_checkpoint(out,folder)
    value=dict(checkpoint=str(folder),checkpoint_manifest_sha256=q.sha(folder/'manifest.json'),
        validation=summary,step=step,protocol_sha256=q.sha(out/'protocol.json'))
    q.immutable(out/'selected.json',value);return value

def reload(out):
    p=verify(out)
    chosen=q.load(out/'selected.json');folder=Path(chosen['checkpoint'])
    if q.sha(folder/'manifest.json')!=chosen['checkpoint_manifest_sha256']:raise PermissionError('Selected checkpoint changed')
    cm=verify_checkpoint(out,folder);torch,tok,model,head=setup(out,folder)
    actual={r:role_hash(model,r) for r in cm['role_hashes']}
    if actual!=cm['role_hashes']:raise RuntimeError('Selected reload differs')
    m,tr,_,teachers=data(p['manifest_path'],p['teacher_manifest_path'])
    activate(model,head,'actor');model.eval()
    packed=pack(tok,tr[0],teachers[str(tr[0]['id'])]['text'])
    with torch.no_grad():probe=logprobs(torch,model,packed)
    if not bool(torch.isfinite(probe).all()):raise RuntimeError('Selected forward nonfinite')
    q.immutable(out/'selected-reload.json',dict(status='PASS',selected_sha256=q.sha(out/'selected.json'),role_hashes=actual))
    return dict(status='PASS')

def evaluate(out,arm):
    """Generate exact old fixed8 evaluation only; global final scoring is separate."""
    p=verify(out)
    task=p['task'];kind,thinking=arm.split('-')
    baseline=ROOT/'runs/2026-10-24-ust100-sg60-modes'
    manifest=baseline/'local-qwen-manifest.json'
    if not (baseline/'local-qwen-original2b'/'predictions-frozen.json').is_file():
        raise PermissionError('Original full320 prediction freeze required before candidate evaluation')
    if kind=='GAD':
        chosen=q.load(out/'selected.json')
        check=q.load(out/'selected-reload.json')
        if check.get('status')!='PASS' or check['selected_sha256']!=q.sha(out/'selected.json'):raise PermissionError('Selected independent reload missing')
        folder=Path(chosen['checkpoint'])
    else:folder=out/'training'/'checkpoints'/f"W{p['train_count']:04d}"
    cm=verify_checkpoint(out,folder)
    torch,tok,model,head=setup(out,folder)
    if {r:role_hash(model,r) for r in cm['role_hashes']}!=cm['role_hashes']:raise PermissionError('Evaluation reload differs')
    _,spec,items=q.manifest_rows(manifest,task)
    target=out/'evaluation'/arm
    contract=dict(protocol_sha256=q.sha(out/'protocol.json'),checkpoint_manifest_sha256=q.sha(folder/'manifest.json'),
        original_evaluation_manifest_sha256=q.sha(manifest),task=task,arm=arm,count=len(items),
        original_fixed8_messages=True,max_new_tokens=q.CAPS[thinking],greedy=True,gold_read=False)
    q.immutable(target/'contract.json',contract)
    for row in items:
        path=target/'responses'/(q.digest(str(row['id']))+'.json')
        if path.exists():
            if q.load(path)['evaluation_contract_sha256']!=q.sha(target/'contract.json'):raise PermissionError('Cached candidate prediction differs')
            continue
        if path.with_suffix('.started.json').exists():raise PermissionError('Unresolved candidate generation; no resampling')
        if STOP:raise RuntimeError('USER_STOP_CANDIDATE_AT_SAVED_BOUNDARY')
        q.immutable(path.with_suffix('.started.json'),dict(id=str(row['id']),evaluation_contract_sha256=q.sha(target/'contract.json')))
        response=rollouts(torch,tok,model,head,row,n=1,greedy=True,thinking=thinking,messages=row['adapted_messages'])[0]
        parsed=q.classify(response['raw_generated'],thinking,spec['scorer'],row.get('metadata',{}),response['finish_reason'])
        q.immutable(path,dict(id=str(row['id']),task=task,thinking=thinking,**parsed,
            evaluation_contract_sha256=q.sha(target/'contract.json'),raw_generated=response['raw_generated'],
            output_ids=response['response_ids'],input_tokens=response['prompt_length'],output_tokens=len(response['response_ids']),
            generation_seconds=response['generation_seconds'],finish_reason=response['finish_reason'],
            max_new_tokens=q.CAPS[thinking],decoding='greedy',input_truncated=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    q.immutable(target/'predictions-frozen.json',dict(complete=True,expected=len(items),
        contract_sha256=q.sha(target/'contract.json'),gold_read=False,predictions=[
            dict(path=str(target/'responses'/(q.digest(str(r['id']))+'.json')),
                 sha256=q.sha(target/'responses'/(q.digest(str(r['id']))+'.json'))) for r in items]))
    return dict(status='FROZEN_UNSCORED',task=task,arm=arm,count=len(items),gold_read=False)

def main():
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','train','select','reload','evaluate','resume-check'])
    parser.add_argument('--manifest',type=Path);parser.add_argument('--teacher-manifest',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--arm',choices=['W-off','GAD-off','GAD-on'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output
    if args.action=='evaluate' and not args.arm:parser.error('evaluate requires --arm')
    if args.action=='resume-check':result=resume_check(out)
    elif args.action=='prepare':
        if not args.manifest or not args.teacher_manifest:parser.error('prepare requires both frozen manifests')
        result=prepare(args.manifest,args.teacher_manifest,out)
    else:
        from night_dual_20261022 import lock,cgroup_limits_verified
        import run_cityfm_repair_qwen_20261022 as guard
        with lock(out/'worker.lock'),lock(guard.HISTORICAL/'locks/gpu.lock'):
            cgroup_limits_verified()
            result=evaluate(out,args.arm) if args.action=='evaluate' else train(out,resume=args.resume) if args.action=='train' else globals()[args.action](out)
    print(json.dumps(result))
if __name__=='__main__':main()

