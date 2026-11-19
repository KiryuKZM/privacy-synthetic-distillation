"""Prepare/conditionally recover a real native-DeepSeek-vocabulary Qwen checkpoint.

prepare is CPU-only. recover/reload require an explicit passed score gate plus
--allow-gpu. This is vocabulary capability repair, never evidence of IPD.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/kiryu/privacy-synthetic-distillation')
META = ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18'
PREV = ROOT/'runs/2026-10-19-014315-qwen-deepseek-overnight'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
BASE = ROOT/'artifacts/hf-home/hub/models--Qwen--Qwen3.5-2B/snapshots'/REVISION
VOCAB = 129280
SEED = 20261019
_PUBLIC_TRAIN = None
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'artifacts/kernels')]
os.environ.setdefault('HF_HOME', str(ROOT/'artifacts/hf-home'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
inc = ROOT/'artifacts/python-dev/include'
if inc.exists():
    os.environ['CPATH'] = str(inc/'python3.12')+os.pathsep+str(inc)+os.pathsep+os.environ.get('CPATH', '')
    os.environ['C_INCLUDE_PATH'] = os.environ['CPATH']

def now(): return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
def read(p): return json.loads(Path(p).read_text())
def dump(p, x):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n'); tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def state(run, phase, **kw):
    event={'at':now(), 'phase':phase, 'pid':os.getpid(), **kw}
    dump(run/'status.json',event)
    with (run/'events.jsonl').open('a') as f: f.write(json.dumps(event, ensure_ascii=False)+'\n')
    print(json.dumps(event, ensure_ascii=False),flush=True)
def formatter():
    spec=importlib.util.spec_from_file_location('official_ds_encoding', META/'encoding--encoding.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def codecs():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(BASE/'tokenizer.json')), Tokenizer.from_file(str(META/'tokenizer.json'))
def prompt_text(row):
    return formatter().encode_messages([
        {'role':'system','content':row['system']},
        {'role':'user','content':row.get('user_a2',row.get('user'))},
        {'role':'assistant','content':'','wo_eos':True}],thinking_mode='chat')
def source_sql(row):
    global _PUBLIC_TRAIN
    sql=row.get('target_sql',row.get('gold_entry',{}).get('SQL',row.get('gold_entry',{}).get('gold_sql')))
    if sql is None and 'train_json_index' in row:
        if _PUBLIC_TRAIN is None:
            _PUBLIC_TRAIN=read(ROOT/'data/raw/candidate-qualification/bird/train.json')
        entry=_PUBLIC_TRAIN[row['train_json_index']]
        assert entry['db_id']==row.get('db_id',row.get('source_db_id'))
        sql=entry['SQL']
    return sql
def encode_row(row, tok):
    sql=source_sql(row)
    if not isinstance(sql,str): raise ValueError('No target SQL for '+str(row.keys()))
    prompt=prompt_text(row); answer=json.dumps({'sql':sql}, ensure_ascii=False)
    full=formatter().encode_messages([
        {'role':'system','content':row['system']},
        {'role':'user','content':row.get('user_a2',row.get('user'))},
        {'role':'assistant','content':answer}],thinking_mode='chat')
    assert full == prompt+answer+'<｜end▁of▁sentence｜>'
    pi=tok.encode(prompt,add_special_tokens=False).ids
    ids=tok.encode(full,add_special_tokens=False).ids
    assert ids[:len(pi)]==pi and ids[0]==0 and ids[-1]==1 and len(ids)>len(pi)
    assert tok.decode(ids,skip_special_tokens=False)==full
    return {'id':row.get('id',row.get('request_id')), 'db_id':row.get('db_id',row.get('source_db_id')),
            'input_ids':ids,'prompt_ids':pi,'prompt_length':len(pi), 'length':len(ids),
            'answer':answer,'system':row['system'],'user':row.get('user_a2',row.get('user')),
            'gold_status':row.get('gold_status'),'train_json_index':row.get('train_json_index')}

def added_recipes(source):
    m=read(META/'embedding-initialization-map.json')
    explicit={
        '<｜begin▁of▁sentence｜>':('<|im_start|>','semantic_conversation_start_approximation'),
        '<｜end▁of▁sentence｜>':('<|im_end|>','semantic_assistant_end'),
        '<｜▁pad▁｜>':('<|endoftext|>','semantic_padding'),
        '<｜User｜>':('<|im_start|>user\n','semantic_role_header_mean'),
        '<｜Assistant｜>':('<|im_start|>assistant\n','semantic_role_header_mean'),
        '<｜System｜>':('<|im_start|>system\n','semantic_role_header_mean'),
        '<think>':('<think>','source_exact_think_start'),
        '</think>':('</think>','source_exact_think_end')}
    out=[]
    for x in m['added_tokens_requiring_explicit_mapping']:
        text=x['content']; tid=source.token_to_id(text)
        if text in explicit:
            spelling,kind=explicit[text]; ids=source.encode(spelling,add_special_tokens=False).ids
        elif tid is not None:
            ids=[tid]; kind='source_exact_added_spelling_unvalidated_semantics'
        else:
            ids=source.encode(text,add_special_tokens=False).ids
            kind='literal_spelling_mean_UNVALIDATED_RESERVED_FALLBACK'
        assert ids and max(ids)<248320
        out.append({**x,'source_ids':ids,'initialization':kind,
                    'semantics_validated':False,'used_control_contract':text in explicit})
    return out

def prepare(run):
    run.mkdir(parents=True,exist_ok=True)
    source,native=codecs(); mapping=read(META/'embedding-initialization-map.json')
    added=added_recipes(source)
    ids=[r['target_id'] for k in ['exact_byte_copy','compositional_mean_initialization'] for r in mapping[k]]+[r['target_id'] for r in added]
    assert len(ids)==len(set(ids))==VOCAB and min(ids)==0 and max(ids)==VOCAB-1
    assert native.get_vocab_size(with_added_tokens=True)==VOCAB
    raw_train=read(PREV/'expanded-train.json'); raw_val=read(PREV/'validation-items.json')
    train=[encode_row(r,native) for r in raw_train]
    val=[encode_row(r,native) for r in raw_val]
    eligible=[r for r in train if r['length']<=1024]
    eligible.sort(key=lambda r:r['id']); random.Random(SEED).shuffle(eligible)
    assert len(train)==274 and len(val)==64 and len(eligible)>=32
    assert not {r['id'] for r in train}&{r['id'] for r in val}
    used=set(t for r in train+val for t in r['input_ids'])
    control=[r for r in added if r['target_id'] in used]
    assert all(r['used_control_contract'] for r in control), control
    dump(run/'added-initialization-recipes.json',added)
    dump(run/'recovery-train.json',eligible)
    dump(run/'recovery-validation.json',val)
    probes=sorted(eligible,key=lambda r:r['id'])[:4]
    dump(run/'generation-probes.json',probes)
    protocol={
        'at':now(),'status':'PREPARED_NOT_RUN','base_revision':REVISION,'fresh_base':True,
        'selected_sft_adapter_used':False,'native_vocabulary':VOCAB,'tie_word_embeddings':True,
        'bos_token_id':0,'eos_token_id':1,'pad_token_id':2,
        'legacy_tokenizer_config_pad_id':1,'pad_override':'explicit2; attention/labels mask; EOS1 labels retained',
        'formatter':'pinned official encode_messages(thinking_mode=chat), assistant wo_eos=True',
        'initialization_counts':{'exact_byte_copy':108771,'compositional_mean':19226,'added_explicit_or_declared_fallback':1283},
        'used_added_controls':control,
        'scope':'Task-limited native vocabulary capability repair CE, NOT IPD or broad capability restoration',
        'training':{'updates':32,'microbatch':1,'grad_accum':4,'max_length':1024,'learning_rate':3e-4,
                    'optimizer':'AdamW','weight_decay':0,'gradient_clip':1,'seed':SEED,
                    'trainable':'shared input embedding and output head only; FP32 parameter, autocast BF16 compute',
                    'frozen':'all other text/vision weights; no LoRA in repair',
                    'loss':'full sequence causal CE; validation additionally answer-only NLL',
                    'save':'cast trained vocabulary matrix to BF16, recompute metrics, full checkpoint; original preserved'},
        'generation':{'public_train_probes':4,'max_new_tokens':128,'do_sample':False,'eos_token_id':1,'pad_token_id':2},
        'train':{'original':274,'eligible':len(eligible),'excluded_full_length':274-len(eligible),
                 'database_counts':dict(Counter(r['db_id'] for r in eligible))},
        'validation':{'fixed_denominator':64,'nll_eligible':sum(r['length']<=1024 for r in val),
                      'overlength_unscored':sum(r['length']>1024 for r in val),
                      'known_reference_execution_failures':sum(r.get('gold_status')!='ok' for r in val),
                      'nll_target_note':'Saved public reference SQL is a text target; no execution-accuracy claim, including known reference errors',
                      'no_accuracy_claim':True},
        'test200_or_sealed_read':False,
        'gpu_gate':'recover/reload require --allow-gpu AND explicit status PASS / qualified_for_migration true in score-gate JSON',
        'code_sha256':sha(__file__),
        'assets':{str(p):sha(p) for p in [META/'tokenizer.json',META/'encoding--encoding.py',META/'embedding-initialization-map.json',BASE/'config.json',PREV/'expanded-train.json',PREV/'validation-items.json',ROOT/'data/raw/candidate-qualification/bird/train.json']}}
    dump(run/'protocol.json',protocol)
    dump(run/'cpu-checks.json',{'at':now(),'status':'PASS','mapping_target_coverage':len(ids),
        'data_sequences_roundtrip':len(train)+len(val),'eos_labels_retained':338,
        'used_added_controls':len(control),'no_gpu_model_load':True,
        'train_ids_sha256':sha(run/'recovery-train.json'),'validation_sha256':sha(run/'recovery-validation.json')})
    state(run,'PREPARED_NOT_RUN',train_eligible=len(eligible),validation_eligible=protocol['validation']['nll_eligible'])

def require_gate(args):
    if not args.allow_gpu or not args.score_gate: raise RuntimeError('GPU work locked: --allow-gpu and passed score gate required')
    gate=read(args.score_gate)
    if gate.get('status')!='PASS' or gate.get('qualified_for_migration') is not True:
        raise RuntimeError('Score gate not explicitly qualified_for_migration PASS')
    return {'path':str(args.score_gate),'sha256':sha(args.score_gate),'content':gate}

def assign_module(model,name,module):
    parts=name.split('.'); parent=model
    for part in parts[:-1]: parent=getattr(parent,part)
    setattr(parent,parts[-1],module)

def convert(model,run,torch):
    original=model.get_input_embeddings(); output=model.get_output_embeddings()
    assert original.weight.data_ptr()==output.weight.data_ptr(), 'Expected tied original head'
    source_weight=original.weight.detach()
    assert tuple(source_weight.shape)==(248320,2048)
    input_name=next(n for n,m in model.named_modules() if m is original)
    output_name=next(n for n,m in model.named_modules() if m is output)
    for p in model.parameters(): p.requires_grad_(False)
    new=torch.nn.Embedding(VOCAB,2048,padding_idx=2,device=source_weight.device,dtype=torch.float32)
    recipes=read(META/'embedding-initialization-map.json')
    with torch.no_grad():
        exact=recipes['exact_byte_copy']
        for start in range(0,len(exact),2048):
            group=exact[start:start+2048]
            ti=torch.tensor([r['target_id'] for r in group],device=source_weight.device)
            si=torch.tensor([r['source_id'] for r in group],device=source_weight.device)
            new.weight[ti]=source_weight[si].float()
        composed=recipes['compositional_mean_initialization']+read(run/'added-initialization-recipes.json')
        groups={}
        for row in composed: groups.setdefault(len(row['source_ids']),[]).append(row)
        for rows in groups.values():
            for start in range(0,len(rows),256):
                group=rows[start:start+256]
                ti=torch.tensor([r['target_id'] for r in group],device=source_weight.device)
                si=torch.tensor([r['source_ids'] for r in group],device=source_weight.device)
                new.weight[ti]=source_weight[si].float().mean(dim=1)
        assert bool(torch.isfinite(new.weight).all())
        # Verify each exact-copy row while the original matrix is still present.
        maxdiff=0.
        for start in range(0,len(exact),2048):
            group=exact[start:start+2048]
            ti=torch.tensor([r['target_id'] for r in group],device=source_weight.device)
            si=torch.tensor([r['source_id'] for r in group],device=source_weight.device)
            maxdiff=max(maxdiff,float((new.weight[ti]-source_weight[si].float()).abs().max()))
        assert maxdiff==0
    head=torch.nn.Linear(2048,VOCAB,bias=False,device=source_weight.device,dtype=torch.float32)
    head.weight=new.weight
    assign_module(model,input_name,new); assign_module(model,output_name,head)
    for conf in [model.config,getattr(model.config,'text_config',None),getattr(model,'generation_config',None)]:
        if conf is not None:
            conf.vocab_size=VOCAB; conf.bos_token_id=0; conf.eos_token_id=1; conf.pad_token_id=2
            conf.tie_word_embeddings=True
    model.tie_weights()
    assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    assert tuple(model.get_output_embeddings().weight.shape)==(VOCAB,2048)
    del original,output,source_weight
    gc.collect(); torch.cuda.empty_cache()
    dump(run/'conversion.json',{'at':now(),'status':'PASS','input_module':input_name,'output_module':output_name,
        'shape':[VOCAB,2048],'shared_parameter_count':VOCAB*2048,'exact_copy_maxdiff':maxdiff,
        'native_embedding_dtype':'float32','all_other_parameters_frozen':all(not p.requires_grad for p in model.parameters() if p is not new.weight)})
    return model

def forward(model,ids,torch,labels=None):
    x=torch.tensor([ids],device='cuda',dtype=torch.long)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,labels=labels)

def measure(model,rows,torch):
    model.eval(); out=[]
    with torch.no_grad():
        for row in rows:
            if row['length']>1024:
                out.append({'id':row['id'],'status':'OVERLENGTH_UNSCORED','length':row['length']}); continue
            logits=forward(model,row['input_ids'],torch).logits[0,:-1].float()
            targets=torch.tensor(row['input_ids'][1:],device='cuda')
            losses=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
            cut=row['prompt_length']-1
            out.append({'id':row['id'],'status':'ok','full_nll':float(losses.mean()),
                        'answer_nll':float(losses[cut:].mean()),'answer_tokens':len(row['input_ids'])-row['prompt_length']})
    good=[r for r in out if r['status']=='ok']
    return {'fixed_denominator':len(rows),'scored':len(good),'unscored':len(rows)-len(good),
            'mean_full_nll':sum(r['full_nll'] for r in good)/max(1,len(good)),
            'mean_answer_nll':sum(r['answer_nll'] for r in good)/max(1,len(good)),'items':out}

def generate_probes(model,rows,tok,torch,native=True,qwen_tok=None):
    model.eval(); out=[]
    # The shared recovery matrix is FP32. For generation, rotary key states
    # otherwise inherit FP32 while projected values are BF16, breaking static
    # cache initialization. Retain the exact master tensor and use a temporary
    # BF16 view for this no-gradient probe, matching delivery precision.
    parameter=model.get_input_embeddings().weight
    master=parameter.data if parameter.dtype==torch.float32 else None
    if master is not None: parameter.data=master.to(torch.bfloat16)
    for row in rows:
        if native: ids=row['prompt_ids']; stop=1; pad=2
        else:
            text=qwen_tok.apply_chat_template([{'role':'system','content':row['system']},{'role':'user','content':row['user']}],add_generation_prompt=True,enable_thinking=False,tokenize=False)
            ids=qwen_tok.encode(text,add_special_tokens=False); stop=qwen_tok.convert_tokens_to_ids('<|im_end|>'); pad=qwen_tok.pad_token_id
        x=torch.tensor([ids],device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=128,do_sample=False,eos_token_id=stop,pad_token_id=pad,cache_implementation='static')
        generated=result[0,len(ids):].tolist()
        text=tok.decode(generated,skip_special_tokens=False) if native else qwen_tok.decode(generated,skip_special_tokens=False)
        out.append({'id':row['id'],'token_ids':generated,'text':text,'stopped_eos':bool(generated and generated[-1]==stop),'prompt_tokens':len(ids)})
    if master is not None: parameter.data=master
    return out

def native_tokenizer():
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(str(META/'tokenizer.json')),
        bos_token='<｜begin▁of▁sentence｜>',eos_token='<｜end▁of▁sentence｜>',pad_token='<｜▁pad▁｜>')
    assert (tok.bos_token_id,tok.eos_token_id,tok.pad_token_id)==(0,1,2) and len(tok)==VOCAB
    return tok

def recover(run,args):
    gate=require_gate(args); protocol=read(run/'protocol.json')
    if protocol['code_sha256']!=sha(__file__): raise RuntimeError('Code changed since CPU preparation; rerun prepare before GPU launch')
    for path,expected in protocol['assets'].items():
        if sha(path)!=expected: raise RuntimeError('Prepared source asset changed: '+path)
    assert not (run/'done.json').exists(),'Already complete; do not overwrite'
    import torch
    import preflight_qwen35_lora as pf
    gpu=pf.gpu_query()
    if float(gpu['memory_free_mib'])<11000 or float(gpu['memory_used_mib'])>2048: raise RuntimeError('GPU occupied or insufficient free memory')
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(); t0=time.monotonic()
    state(run,'LOADING_FRESH_BASE',gate=gate)
    _,oldtok,model,loaded,errors=pf._load_model(torch)
    probes=read(run/'generation-probes.json'); _,native=codecs()
    dump(run/'original-qwen-train-probes.json',generate_probes(model,probes,native,torch,False,oldtok))
    model=convert(model,run,torch)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    val=read(run/'recovery-validation.json'); train=read(run/'recovery-train.json')
    before=measure(model,val,torch); dump(run/'native-before-nll.json',before)
    dump(run/'native-before-train-probes.json',generate_probes(model,probes,native,torch))
    parameter=model.get_input_embeddings().weight
    assert parameter.dtype==torch.float32 and parameter.requires_grad
    optim=torch.optim.AdamW([parameter],lr=3e-4,weight_decay=0)
    losses=[]; model.train(); optim.zero_grad(set_to_none=True)
    for update in range(32):
        lsum=0.
        for micro in range(4):
            row=train[(update*4+micro)%len(train)]
            labels=torch.tensor([row['input_ids']],device='cuda')
            loss=forward(model,row['input_ids'],torch,labels).loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite native repair loss')
            (loss/4).backward(); lsum+=float(loss.detach())/4
        norm=torch.nn.utils.clip_grad_norm_([parameter],1.)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite embedding gradient')
        before_rows=parameter[:32].detach().clone()
        optim.step(); optim.zero_grad(set_to_none=True)
        change=float((parameter[:32].detach()-before_rows).abs().sum())
        if update==0 and change==0: raise RuntimeError('No actual vocabulary parameter update')
        event={'at':now(),'update':update+1,'loss':lsum,'gradient_norm':float(norm),'vocabulary_row_probe_delta_l1':change,'learning_rate':optim.param_groups[0]['lr']}
        losses.append(event)
        with (run/'training.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        state(run,'RECOVERY_TRAINING',**{k:v for k,v in event.items() if k!='at'})
    del optim
    gc.collect(); torch.cuda.empty_cache()
    # Delivery is BF16; measure the delivered precision, not a different FP32 artifact.
    model.get_input_embeddings().to(dtype=torch.bfloat16)
    model.get_output_embeddings().weight=model.get_input_embeddings().weight
    model.config.use_cache=True
    after=measure(model,val,torch); dump(run/'native-after-nll.json',after)
    generations=generate_probes(model,probes,native,torch)
    dump(run/'native-after-train-probes.json',generations)
    checkpoint=run/'native-checkpoint'; checkpoint.mkdir(exist_ok=True)
    for p in model.parameters():p.requires_grad_(False)
    model.save_pretrained(checkpoint,safe_serialization=True,max_shard_size='2GB')
    native_tokenizer().save_pretrained(checkpoint)
    shutil.copyfile(META/'encoding--encoding.py',checkpoint/'deepseek_encoding.py')
    dump(checkpoint/'native-contract.json',{'text_only':True,'vocabulary':VOCAB,'bos':0,'eos':1,'pad':2,
        'formatter':'deepseek_encoding.encode_messages thinking_mode=chat, wo_eos=True on assistant prefix',
        'repair_only_not_ipd':True,'base_revision':REVISION})
    manifest={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
    dump(run/'checkpoint-sha256.json',manifest)
    dump(run/'done.json',{'at':now(),'status':'RECOVERY_COMPLETE_NOT_IPD','updates':32,'elapsed_seconds':time.monotonic()-t0,
        'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2,'loaded_as':loaded,'load_errors':errors,
        'before_answer_nll':before['mean_answer_nll'],'after_answer_nll':after['mean_answer_nll'],
        'source_head_tied':True,'checkpoint':str(checkpoint),'scope':protocol['scope'],'score_gate':gate})
    state(run,'RECOVERY_COMPLETE_NOT_IPD',checkpoint=str(checkpoint))

def reload_check(run,args):
    gate=require_gate(args)
    import torch
    from transformers import AutoModelForImageTextToText
    ck=run/'native-checkpoint'
    model=AutoModelForImageTextToText.from_pretrained(ck,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
    tok=native_tokenizer(); _,native=codecs()
    assert model.get_input_embeddings().weight.shape[0]==VOCAB
    assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    saved=read(run/'native-after-train-probes.json')
    got=generate_probes(model,read(run/'generation-probes.json'),native,torch)
    matches=[a['token_ids']==b['token_ids'] for a,b in zip(saved,got)]
    assert len(matches)==4 and all(matches)
    dump(run/'reload-check.json',{'at':now(),'status':'PASS','source_head_tied':True,'vocabulary':VOCAB,
        'probe_exact_token_matches':sum(matches),'probe_denominator':4,'native_pad_id':tok.pad_token_id,'score_gate':gate})


# Capability recovery is separate from the historical32-update functional run.
# It never imports the paid IPD client and never reads the evaluation300.
REPAIR_OLD=ROOT/'runs/2026-10-19-official-teacher-support-ipd/migration'
REPAIR_RAW=ROOT/'data/raw/candidate-qualification/bird'
REPAIR_SEED=20261020


def task_key(row):
    return row['db_id']+'|'+ ' '.join(row['question'].casefold().split())


def repair_prompt(row):
    return formatter().encode_messages([{'role':'user','content':row['user']},
        {'role':'assistant','content':'','wo_eos':True}],thinking_mode='chat')


def repair_prepare(run):
    """Freeze new public-train-only train/selection/audit memberships before learning."""
    if (run/'protocol.json').exists():raise RuntimeError('Repair preparation already frozen')
    from rescore_bird_student_inplace import load_bird
    from collections import defaultdict
    import numpy as np
    from array import array
    bird=load_bird();run.mkdir(parents=True,exist_ok=True)
    import tomllib
    config_path=ROOT/'configs/bird_B_retoke_repair_20261020.toml'
    settings=tomllib.loads(config_path.read_text())['repair']
    source=read(REPAIR_RAW/'train.json');metadata={r['db_id']:r for r in read(REPAIR_RAW/'train_tables.json')}
    meanings=read(REPAIR_RAW/'train_column_meaning.json')
    old=read(REPAIR_OLD/'recovery-train.json')+read(REPAIR_OLD/'recovery-validation.json')
    excluded_indices={r['train_json_index'] for r in old}
    ipd=read(ROOT/'runs/2026-10-19-bird-ipd-comparison-design/train200-candidate-identities.json')
    excluded_indices.update(r['source_index'] for r in ipd)
    excluded_keys={task_key(source[i]) for i in excluded_indices}
    dbdir=PREV/'public-training-databases';dbpaths={p.stem:p for p in dbdir.glob('*.sqlite')}
    pools=defaultdict(list)
    for i,row in enumerate(source):
        if i not in excluded_indices and task_key(row) not in excluded_keys and row['db_id'] in dbpaths:
            pools[row['db_id']].append(i)
    contexts={};select=[];audit=[];gold_checks=[];seen=set(excluded_keys)
    for db in sorted(dbpaths):
        context=bird.database_context(dbpaths[db],{});context.pop('original_description_csv',None)
        context['original_column_meanings']={k:v for k,v in meanings.items() if k.startswith(db+'|')}
        contexts[db]=context
        candidates=sorted(pools[db]);random.Random(REPAIR_SEED+sum(map(ord,db))).shuffle(candidates)
        retained=[]
        for i in candidates:
            row=source[i];key=task_key(row)
            if key in seen:continue
            seen.add(key)
            gold=bird.execute_summary(str(dbpaths[db]),row['SQL'],timeout=30,max_rows=50000)
            gold_checks.append(dict(source_index=i,db_id=db,status=gold['status'],error=gold.get('error')))
            if gold['status']!='ok':continue
            user=bird.SYSTEM+'\n\n'+json.dumps({**context,'question':row['question'],'expert_evidence':row['evidence']},ensure_ascii=False,indent=2)
            retained.append(dict(id=f'train-{i:05d}',source_index=i,db_id=db,user=user,
                answer=json.dumps({'sql':row['SQL']},ensure_ascii=False),
                gold_entry=dict(db_path=str(dbpaths[db]),expected=gold['summary'])))
            if len(retained)==16:break
        if len(retained)!=16:raise RuntimeError('Insufficient independent executable tasks in '+db)
        select+=retained[:8];audit+=retained[8:]
        state(run,'REPAIR_PUBLIC_SPLIT',database=db,selection=len(select),audit=len(audit))
    assert len(select)==len(audit)==64
    heldout_indices={r['source_index'] for r in select+audit}
    heldout_keys={task_key(source[i]) for i in heldout_indices}
    rejected_keys={task_key(source[r['source_index']]) for r in gold_checks if r['status']!='ok'}
    banned=excluded_keys|heldout_keys|rejected_keys
    def metadata_context(db):
        if db in contexts:return contexts[db]
        meta=metadata[db];ddl=[];cols=meta['column_names_original']
        for ti,table in enumerate(meta['table_names_original']):
            names=[(j,c) for j,(t,c) in enumerate(cols) if t==ti]
            definitions=[bird.q(c)+' '+{'text':'TEXT','integer':'INTEGER','number':'NUMERIC','real':'REAL','boolean':'INTEGER','date':'TEXT','datetime':'TEXT'}.get(meta['column_types'][j].lower(),'TEXT') for j,c in names]
            for a,b in meta['foreign_keys']:
                if cols[a][0]==ti:
                    definitions.append('FOREIGN KEY ('+bird.q(cols[a][1])+') REFERENCES '+bird.q(meta['table_names_original'][cols[b][0]])+' ('+bird.q(cols[b][1])+')')
            ddl.append('CREATE TABLE '+bird.q(table)+' ('+', '.join(definitions)+');')
        context=dict(schema_ddl='\n\n'.join(ddl),original_column_meanings={k:v for k,v in meanings.items() if k.startswith(db+'|')})
        contexts[db]=context;return context
    eligible=[];unique=set(banned)
    for i,row in enumerate(source):
        key=task_key(row)
        if i in excluded_indices or i in heldout_indices or key in unique or row['db_id'] in ['donor','food_inspection']:continue
        if not row['SQL'].lstrip().lower().startswith(('select','with')):continue
        unique.add(key);eligible.append(i)
    random.Random(REPAIR_SEED).shuffle(eligible)
    _,tok=codecs();fmt=formatter();tokens=array('I');documents=[]
    for i in eligible:
        row=source[i];context=metadata_context(row['db_id'])
        user=bird.SYSTEM+'\n\n'+json.dumps({**context,'question':row['question'],'expert_evidence':row['evidence']},ensure_ascii=False,indent=2)
        text=fmt.encode_messages([{'role':'user','content':user},
             {'role':'assistant','content':json.dumps({'sql':row['SQL']},ensure_ascii=False)}],thinking_mode='chat')
        ids=tok.encode(text,add_special_tokens=False).ids
        assert ids[0]==0 and ids[-1]==1
        tokens.extend(ids);documents.append(dict(source_index=i,db_id=row['db_id'],tokens=len(ids)))
    np.save(run/'clm-tokens.npy',np.asarray(tokens,dtype=np.uint32))
    dump(run/'selection.json',select);dump(run/'audit.json',audit)
    dump(run/'gold-eligibility.json',gold_checks);dump(run/'training-documents.json',documents)
    screen=[r['id'] for db in sorted(dbpaths) for r in [x for x in select if x['db_id']==db][:2]]
    protocol=dict(at=now(),status='AUTHORIZED_LOCAL_CAPABILITY_RECOVERY',seed=REPAIR_SEED,
        literature='ReTok2410.04335 sections3.2-4.1; domain CLM adaptation, not exact paper replication',
        base=str(BASE),base_revision=REVISION,start='fresh byte-copy/mean initialization, no oldB or IPD adapter',
        vocabulary=VOCAB,tied_head=True,formatter='official DS chat; single user; unchanged across recovery and validation',
        train_documents=len(documents),train_tokens=len(tokens),training_databases=dict(Counter(x['db_id'] for x in documents)),
        selection64=[x['id'] for x in select],audit64=[x['id'] for x in audit],screen16=screen,
        old338_and_IPD200_disjoint=True,question_keys_disjoint=True,food_donor_excluded_from_training=True,
        evaluation300_read=False,external_model_calls=0,
        training=dict(**settings,microbatch=1,schedule='cosine',
            objective='standard full-sequence causal LM, includes actual EOS; no custom weighted answer loss',
            trainable='FP32 shared embedding and output head only',frozen='all remaining weights',
            optimizer='FP32 AdamW with CPU master/moments; GPU gradients; mathematical gradient copying only'),
        validation=dict(screen_max_tokens=256,full_max_tokens=1024,greedy=True,batch_size=1,
            selection='screen EX>=original and truncations<=original, answerNLL/byte<=1.10*original; then full64 validation',
            full_gate='EX>=original, parse>=original-2, truncations<=original+2, answerNLL/byte<=1.10*original; original EX must be nonzero',
            final_audit='same gate on untouched64 only after selecting/fixing checkpoint; failure ends run without further audit-driven tuning',
            scope='public-train SQL capability check, not broad ability or statistical noninferiority proof'),
        code_sha256=sha(__file__),assets={str(p):sha(p) for p in [config_path,REPAIR_RAW/'train.json',REPAIR_RAW/'train_tables.json',REPAIR_RAW/'train_column_meaning.json',run/'clm-tokens.npy',run/'selection.json',run/'audit.json']})
    dump(run/'protocol.json',protocol);state(run,'REPAIR_PREPARED',train_documents=len(documents),train_tokens=len(tokens),selection=64,audit=64,external_calls=0)


def chunked_clm_loss(hidden,weight,ids,chunk_tokens=128,labels=None):
    """Exact shifted CE, optionally with standard ignored prompt labels."""
    import torch
    from torch.utils.checkpoint import checkpoint
    h=hidden[:,:-1].reshape(-1,hidden.shape[-1]);targets=(ids if labels is None else labels)[:,1:].reshape(-1)
    valid=targets!=-100;count=int(valid.sum())
    if count==0:raise ValueError('No supervised answer targets')
    total=h.new_zeros((),dtype=torch.float32)
    def term(x,w,y):return torch.nn.functional.cross_entropy(torch.nn.functional.linear(x,w).float(),y,reduction='sum',ignore_index=-100)
    for start in range(0,len(targets),chunk_tokens):
        if not bool(valid[start:start+chunk_tokens].any()):continue
        total=total+checkpoint(term,h[start:start+chunk_tokens],weight,targets[start:start+chunk_tokens],use_reentrant=False)
    return total/count


def complete_task_index(tokens,documents,assistant_id,suffix_ids,max_length):
    """Keep whole documents only; never truncate schema/question or invent EOS."""
    import numpy as np
    retained=[];excluded=[];offset=0
    for document in documents:
        size=document['tokens'];ids=tokens[offset:offset+size]
        if len(ids)!=size or int(ids[0])!=0 or int(ids[-1])!=1:raise ValueError('Invalid complete document boundary')
        roles=np.flatnonzero(ids==assistant_id)
        if len(roles)!=1:raise ValueError('Ambiguous assistant boundary')
        start=int(roles[0])+1
        if ids[start:start+len(suffix_ids)].tolist()!=list(suffix_ids):raise ValueError('Nonthinking prefix mismatch')
        start+=len(suffix_ids)
        if start>=size-1:raise ValueError('Missing actual answer')
        item=dict(**document,offset=offset,answer_start=start,supervised_tokens=size-start)
        (retained if size<=max_length else excluded).append(item);offset+=size
    if offset!=len(tokens):raise ValueError('Document/corpus lengths differ')
    return retained,excluded


def complete_task_arrays(tokens,document):
    import numpy as np
    lo=document['offset'];size=document['tokens']
    ids=np.array(tokens[lo:lo+size],dtype=np.int64)
    if len(ids)!=size or ids[0]!=0 or ids[-1]!=1:raise ValueError('Complete task was clipped')
    labels=np.full(size,-100,dtype=np.int64);labels[document['answer_start']:]=ids[document['answer_start']:]
    return ids,labels


def repair_resource_wait(run):
    import subprocess
    idle_since=None
    protocol_path=run/'protocol.json'
    cap=read(protocol_path).get('training',{}).get('cuda_allocator_cap_gib',8) if protocol_path.exists() else 8
    required=max(9000,(cap+1)*1024 if cap>8 else 9000)
    lease_path=os.environ.get('PRIVACY_SYNTH_GPU_LEASE')
    if lease_path:
        lease=read(lease_path)
        if lease.get('pid')==os.getppid() and lease.get('run')==str(run.resolve()):
            q=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free,used,utilization=map(float,q.stdout.strip().split(','))
            if free>=required and utilization<=10:
                state(run,'GPU_PHASE_HANDOFF_CONFIRMED',free_mib=free,used_mib=used,utilization_percent=utilization,coordinator_pid=lease['pid'])
                return
    while True:
        q=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        # This experiment is single-GPU; refuse an ambiguous multi-GPU reading.
        free,used,utilization=map(float,q.stdout.strip().split(','))
        measured=time.monotonic()
        if free>=required and utilization<=10:
            if idle_since is None:idle_since=measured
        else:
            idle_since=None
        idle_seconds=0 if idle_since is None else measured-idle_since
        details=dict(free_mib=free,used_mib=used,utilization_percent=utilization,
            required_free_mib=required,maximum_utilization_percent=10,
            idle_seconds=idle_seconds,required_idle_seconds=60)
        if idle_seconds>=60:
            state(run,'GPU_IDLE_CONFIRMED',**details)
            return
        state(run,'WAITING_GPU_IDLE',**details)
        # Owner explicitly elected to retain the other program and wait.
        # No timeout, model allocation or optimizer update while queued.
        time.sleep(30)


def repair_load(torch,base):
    from transformers import AutoModelForImageTextToText
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    return AutoModelForImageTextToText.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')


def repair_evaluate_model(run,model,tok,tag,split,torch,screen=False,original=False):
    rows=read(run/(split+'.json'));protocol=read(run/'protocol.json')
    if screen:rows=[r for r in rows if r['id'] in protocol['screen16']]
    mode='screen' if screen else split;folder=run/'evaluation'/tag/mode;folder.mkdir(parents=True,exist_ok=True)
    contract=dict(protocol_sha256=sha(run/'protocol.json'),tag=tag,split=mode,
        weights=REVISION if original else sha(run/'snapshots'/tag/'embedding.safetensors'))
    if not original and protocol.get('body_lora'):
        contract['merged_model_manifest_sha256']=sha(run/'snapshots'/tag/'export-manifest.json')
    if (folder/'contract.json').exists() and read(folder/'contract.json')!=contract:raise RuntimeError('Validation weights or inputs changed')
    dump(folder/'contract.json',contract)
    parameter=model.get_input_embeddings().weight;master=parameter.data if parameter.dtype==torch.float32 else None
    if master is not None:parameter.data=master.to(torch.bfloat16)
    model.eval();nll=[];max_new=256 if screen else 1024
    stops=[tok.convert_tokens_to_ids(t) for t in ['<|im_end|>','<|endoftext|>']] if original else [1]
    try:
        for row in rows:
            if (folder/(row['id']+'.json')).exists():continue
            text=tok.apply_chat_template([{'role':'user','content':row['user']}],add_generation_prompt=True,enable_thinking=False,tokenize=False) if original else repair_prompt(row)
            ids=tok.encode(text,add_special_tokens=False);full=tok.encode(text+row['answer'],add_special_tokens=False)
            assert full[:len(ids)]==ids
            x=torch.tensor([full],device='cuda')
            with torch.inference_mode():
                logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=len(full)-len(ids)+1).logits
                loss=torch.nn.functional.cross_entropy(logits[0,:-1].float(),x[0,len(ids):],reduction='sum')
                nats_per_byte=float(loss)/len(row['answer'].encode())
            del logits,loss,x
            x=torch.tensor([ids],device='cuda')
            with torch.inference_mode():
                out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=max_new,do_sample=False,
                    eos_token_id=stops,pad_token_id=tok.pad_token_id,cache_implementation='static',disable_compile=True,use_cache=True)
            generated=out[0,len(ids):].tolist();finish='stop' if generated and generated[-1] in stops else 'length'
            dump(folder/(row['id']+'.json'),dict(id=row['id'],content=tok.decode(generated,skip_special_tokens=True),
                token_ids=generated,finish_reason=finish,nats_per_byte=nats_per_byte,at=now()))
            del out,x
        state(run,'REPAIR_GENERATION_COMPLETE',tag=tag,split=mode,denominator=len(rows),cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20)
    finally:
        if master is not None:parameter.data=master
        gc.collect();torch.cuda.empty_cache()
    import subprocess
    subprocess.run([sys.executable,__file__,'repair-score','--run',str(run),'--tag',tag,'--split',mode],check=True)
    return read(folder/'summary.json')


def repair_score(run,tag,split):
    from rescore_bird_student_inplace import load_bird
    bird=load_bird();execute=bird.execute_summary
    bird.execute_summary=lambda path,sql,**kw:execute(path,sql,timeout=30,max_rows=50000)
    rows=read(run/('selection.json' if split=='screen' else split+'.json'))
    if split=='screen':rows=[r for r in rows if r['id'] in read(run/'protocol.json')['screen16']]
    folder=run/'evaluation'/tag/split;metrics=[]
    for row in rows:
        response=read(folder/(row['id']+'.json'))
        score=bird.score_response(response['content'],row['gold_entry'],finish_reason=response['finish_reason'])
        metrics.append(dict(score,id=row['id'],truncated=response['finish_reason']=='length',nats_per_byte=response['nats_per_byte']))
    summary=dict(denominator=len(rows),correct=sum(bool(x.get('official_ex')) for x in metrics),
        parse_ok=sum(bool(x.get('parse_ok')) for x in metrics),execution_ok=sum(x.get('execution_status')=='ok' for x in metrics),
        truncated=sum(x['truncated'] for x in metrics),answer_nats_per_byte=sum(x['nats_per_byte'] for x in metrics)/len(metrics))
    dump(folder/'scores.json',metrics);dump(folder/'summary.json',summary);print(json.dumps(dict(tag=tag,split=split,**summary)),flush=True)


def repair_gate(candidate,original,screen=False):
    allowance=0 if screen else 2
    return (candidate['denominator']==original['denominator'] and (screen or original['correct']>0)
        and candidate['correct']>=original['correct'] and candidate['parse_ok']>=original['parse_ok']-allowance
        and candidate['truncated']<=original['truncated']+allowance
        and candidate['answer_nats_per_byte']<=1.10*original['answer_nats_per_byte'])


def repair_run(run,args):
    if not args.allow_gpu:raise RuntimeError('Local repair requires --allow-gpu')
    protocol=read(run/'protocol.json')
    if protocol['code_sha256']!=sha(__file__):raise RuntimeError('Frozen repair code changed')
    for path,value in protocol['assets'].items():
        if sha(path)!=value:raise RuntimeError('Frozen repair input changed: '+path)
    if (run/'final.json').exists():raise RuntimeError('Repair already has final result')
    repair_resource_wait(run)
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    torch.manual_seed(REPAIR_SEED);torch.cuda.manual_seed_all(REPAIR_SEED)
    state(run,'REPAIR_ORIGINAL_REFERENCE')
    original_screen=run/'evaluation/original/screen/summary.json'
    if not original_screen.exists():
        model=repair_load(torch,BASE);qt=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        repair_evaluate_model(run,model,qt,'original','selection',torch,screen=True,original=True)
        del model;gc.collect();torch.cuda.empty_cache()
    original_screen=read(original_screen)
    state(run,'REPAIR_FRESH_INITIALIZATION')
    model=repair_load(torch,BASE)
    shutil.copyfile(REPAIR_OLD/'added-initialization-recipes.json',run/'added-initialization-recipes.json')
    torch.cuda.set_per_process_memory_fraction(protocol['training']['cuda_allocator_cap_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=convert(model,run,torch);model.config.use_cache=False
    if protocol.get('initial_embedding'):
        from safetensors.torch import load_file
        initial=load_file(protocol['initial_embedding'])['embedding']
        with torch.no_grad():model.get_input_embeddings().weight.copy_(initial)
        del initial
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameter=model.get_input_embeddings().weight
    master=torch.nn.Parameter(parameter.detach().cpu().clone())
    settings=protocol['training']
    optimizer=torch.optim.AdamW([master],lr=settings['learning_rate'],weight_decay=settings['weight_decay'],foreach=False)
    tokens=np.load(run/'clm-tokens.npy',mmap_mode='r');length=protocol['training']['sequence_length']
    complete=settings.get('data_mode')=='complete_task_answer_only'
    documents=read(run/'complete-task-index.json') if complete else None
    blocks=len(documents) if complete else (len(tokens)-1)//(length-1)
    order=np.random.default_rng(REPAIR_SEED).permutation(blocks)
    probe=torch.tensor(np.array(tokens[:64],dtype=np.int64)[None],device='cuda')
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        probe_labels=probe.clone()
        if complete:probe_labels[:,:32]=-100
        direct=model(input_ids=probe,attention_mask=torch.ones_like(probe),use_cache=False,labels=probe_labels).loss
        hidden=model.model(input_ids=probe,attention_mask=torch.ones_like(probe),use_cache=False).last_hidden_state
        chunked=chunked_clm_loss(hidden,parameter,probe,settings['head_loss_chunk_tokens'],labels=probe_labels)
        error=float((direct-chunked).abs())
    dump(run/'real-model-loss-equivalence.json',dict(full_loss=float(direct),chunked_loss=float(chunked),absolute_difference=error,at=now()))
    del direct,hidden,chunked,probe,probe_labels
    if error>2e-3:raise RuntimeError('Chunked real model objective differs from standard CLM')
    start=0
    if args.resume:
        saved=torch.load(run/'resume.pt',map_location='cpu',weights_only=True)
        if saved['protocol_sha256']!=sha(run/'protocol.json'):raise RuntimeError('Resume protocol mismatch')
        with torch.no_grad():master.copy_(saved['master']);parameter.copy_(master)
        optimizer.load_state_dict(saved['optimizer']);start=saved['update'];del saved
        state(run,'REPAIR_RESUMED',update=start)
    elif (run/'training.jsonl').exists():raise RuntimeError('Existing training requires explicit --resume')
    def save(update):
        temp=run/'resume.pt.tmp';torch.save(dict(update=update,master=master.detach(),optimizer=optimizer.state_dict(),protocol_sha256=sha(run/'protocol.json')),temp);temp.replace(run/'resume.pt')
        dump(run/'resume.json',dict(update=update,protocol_sha256=sha(run/'protocol.json'),at=now()))
    def candidate(update):
        tag=f'step{update:04d}';folder=run/'snapshots'/tag;folder.mkdir(parents=True,exist_ok=True)
        from safetensors.torch import save_file
        save_file({'embedding':master.detach().to(torch.bfloat16).contiguous()},str(folder/'embedding.safetensors'))
        model.gradient_checkpointing_disable()
        screen=repair_evaluate_model(run,model,native_tokenizer(),tag,'selection',torch,screen=True)
        if repair_gate(screen,original_screen,screen=True):
            reference=run/'evaluation/original/selection/summary.json'
            if not reference.exists():
                # Free the native model during reference generation; reload only
                # immutable original weights and retain CPU optimizer state.
                return 'NEEDS_FULL_REFERENCE',tag,screen
            full=repair_evaluate_model(run,model,native_tokenizer(),tag,'selection',torch)
            if repair_gate(full,read(reference)):
                dump(run/'selected.json',dict(tag=tag,update=update,selection=full,at=now()))
                return 'SELECTED',tag,full
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return 'CONTINUE',tag,screen
    if args.resume and (run/'latest-validation.json').exists():
        previous=read(run/'latest-validation.json')
        if previous['result']=='NEEDS_FULL_REFERENCE' and (run/'evaluation/original/selection/summary.json').exists():
            result,tag,metric=candidate(start)
            dump(run/'latest-validation.json',dict(result=result,tag=tag,metric=metric,at=now()))
            if result=='SELECTED':state(run,result,tag=tag,update=start);return
    processed_tokens=sum(documents[int(order[i%blocks])]['tokens'] for i in range(start*settings['grad_accum'])) if complete else start*settings['grad_accum']*length
    supervised_tokens=sum(documents[int(order[i%blocks])]['supervised_tokens'] for i in range(start*settings['grad_accum'])) if complete else start*settings['grad_accum']*(length-1)
    for update in range(start,protocol['training']['max_updates']):
        model.train();parameter.grad=None;optimizer.zero_grad(set_to_none=True)
        warmup=settings['warmup_updates'];maximum=settings['max_updates']
        lr=settings['learning_rate']*min(1.,(update+1)/warmup)*.5*(1+math.cos(math.pi*max(0,update-warmup)/max(1,maximum-warmup)))
        optimizer.param_groups[0]['lr']=lr;losses=[];began=time.monotonic()
        for micro in range(settings['grad_accum']):
            block=int(order[(update*settings['grad_accum']+micro)%blocks]);offset=block*(length-1)
            if complete:
                raw,targets=complete_task_arrays(tokens,documents[block])
                ids=torch.tensor(raw[None],device='cuda');labels=torch.tensor(targets[None],device='cuda')
                supervised_tokens+=documents[block]['supervised_tokens']
            else:
                ids=torch.tensor(np.array(tokens[offset:offset+length],dtype=np.int64)[None],device='cuda');labels=None
                supervised_tokens+=ids.shape[1]-1
            processed_tokens+=ids.shape[1]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
                loss=chunked_clm_loss(hidden,parameter,ids,protocol['training']['head_loss_chunk_tokens'],labels=labels)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite recovery loss')
            (loss/settings['grad_accum']).backward();losses.append(float(loss.detach()));del hidden,loss,ids,labels
        norm=torch.nn.utils.clip_grad_norm_([parameter],settings['gradient_clip'])
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite embedding gradient')
        master.grad=parameter.grad.detach().cpu();parameter.grad=None
        before=master[:32].detach().clone();optimizer.step();optimizer.zero_grad(set_to_none=True)
        delta=float((master[:32].detach()-before).abs().sum())
        with torch.no_grad():parameter.copy_(master)
        record=dict(at=now(),update=update+1,loss=sum(losses)/len(losses),learning_rate=lr,gradient_norm=float(norm),
            row_probe_delta_l1=delta,seen_tokens=processed_tokens,supervised_tokens=supervised_tokens,complete_prompt_for_every_target=complete,seconds=time.monotonic()-began,
            cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20,external_calls=0)
        with (run/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        state(run,'REPAIR_TRAINING',**{k:v for k,v in record.items() if k!='at'})
        if (update+1)%settings['resume_interval']==0 or update==start:save(update+1)
        if (update+1)%settings['validation_interval']==0:
            save(update+1);result,tag,metric=candidate(update+1)
            dump(run/'latest-validation.json',dict(result=result,tag=tag,metric=metric,at=now()))
            if result!='CONTINUE':
                # A resource-bounded coordinator resumes from this checkpoint
                # after obtaining the independent original full reference.
                state(run,result,tag=tag,update=update+1);return
    save(protocol['training']['max_updates'])
    dump(run/'final.json',dict(status='NOT_RESTORED_WITHIN_FROZEN_LOCAL_BUDGET',updates=protocol['training']['max_updates'],ipd_still_stopped=True,external_calls=0,at=now()))
    state(run,'NOT_RESTORED_WITHIN_FROZEN_LOCAL_BUDGET',ipd_still_stopped=True)

def repair_reference(run,args):
    if not args.allow_gpu:raise RuntimeError('Local reference requires --allow-gpu')
    repair_resource_wait(run)
    import torch
    from transformers import AutoTokenizer
    model=repair_load(torch,BASE);tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    repair_evaluate_model(run,model,tok,'original',args.split,torch,original=True)


def repair_finalize(run,args):
    """One fixed candidate, untouched audit64, export only after the frozen gate."""
    if not args.allow_gpu:raise RuntimeError('Local final audit requires --allow-gpu')
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    selected=read(run/'selected.json');tag=selected['tag']
    if not (run/'evaluation/original/audit/summary.json').exists():
        repair_resource_wait(run);model=repair_load(torch,BASE)
        repair_evaluate_model(run,model,AutoTokenizer.from_pretrained(BASE,local_files_only=True),'original','audit',torch,original=True)
        del model;gc.collect();torch.cuda.empty_cache()
    repair_resource_wait(run);model=repair_load(torch,REPAIR_OLD/'native-checkpoint')
    weight=load_file(str(run/'snapshots'/tag/'embedding.safetensors'))['embedding']
    with torch.no_grad():model.get_input_embeddings().weight.copy_(weight)
    del weight
    assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    for p in model.parameters():p.requires_grad_(False)
    result=repair_evaluate_model(run,model,native_tokenizer(),tag,'audit',torch)
    reference=read(run/'evaluation/original/audit/summary.json');passed=repair_gate(result,reference)
    final=dict(status='CAPABILITY_RECOVERED_PUBLIC_SQL' if passed else 'FINAL_AUDIT_FAILED_NOT_RESTORED',
        selected=selected,audit=result,original_audit=reference,external_calls=0,ipd_still_stopped=True,
        evidence_scope='single seed,64 selection+64 held-out public train tasks; not broad capability or statistical noninferiority proof',at=now())
    if passed:
        checkpoint=run/'native-checkpoint';model.save_pretrained(checkpoint,safe_serialization=True,max_shard_size='2GB')
        native_tokenizer().save_pretrained(checkpoint);shutil.copyfile(META/'encoding--encoding.py',checkpoint/'deepseek_encoding.py')
        dump(checkpoint/'native-contract.json',dict(vocabulary=VOCAB,bos=0,eos=1,pad=2,formatter='official DS chat single-user thinking off',
             recovery_only_not_ipd=True,protocol_sha256=sha(run/'protocol.json'),capability_gate=str(run/'final.json')))
        from safetensors import safe_open
        source_map=read(BASE/'model.safetensors.index.json')['weight_map']
        dest_map=read(checkpoint/'model.safetensors.index.json')['weight_map'];checked=0
        for name,shard in dest_map.items():
            if name.endswith('embed_tokens.weight') or name=='lm_head.weight':continue
            with safe_open(BASE/source_map[name],framework='pt',device='cpu') as a,safe_open(checkpoint/shard,framework='pt',device='cpu') as b:
                left=a.get_tensor(name);right=b.get_tensor(name)
                if not torch.equal(left.to(right.dtype),right):raise RuntimeError('Frozen trunk changed: '+name)
            checked+=1
        final['frozen_trunk_tensors_equal_at_delivered_precision']=checked
        probe=read(run/'audit.json')[0];ids=native_tokenizer().encode(repair_prompt(probe),add_special_tokens=False)
        x=torch.tensor([ids],device='cuda')
        with torch.inference_mode():before=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.cpu()
        del model;gc.collect();torch.cuda.empty_cache();model=repair_load(torch,checkpoint);model.eval()
        with torch.inference_mode():after=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.cpu()
        difference=float((before-after).abs().max())
        if difference!=0:raise RuntimeError('Export independent-reload mismatch')
        final.update(checkpoint=str(checkpoint),independent_reload_max_logit_difference=difference,
                     checkpoint_files={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()})
    dump(run/'final.json',final);state(run,final['status'],selected_tag=tag,ipd_still_stopped=True)


def repair_prepare_complete(run,args):
    """Freeze second-attempt whole-task answer supervision without new task leaks."""
    import tomllib
    import numpy as np
    if (run/'protocol.json').exists():raise RuntimeError('Complete-task attempt already frozen')
    config_path=ROOT/'configs/bird_B_repair_attempt2_20261021.toml'
    config=tomllib.loads(config_path.read_text());settings=config['repair']
    source=ROOT/config['initialization']['source_run'];snapshot=config['initialization']['snapshot']
    old=read(source/'protocol.json')
    for path,value in old['assets'].items():
        if sha(path)!=value:raise RuntimeError('Original frozen corpus changed: '+path)
    initial=source/'snapshots'/snapshot/'embedding.safetensors'
    if not initial.exists():raise RuntimeError('Initial saved embedding missing')
    run.mkdir(parents=True,exist_ok=True)
    for name in ['selection.json','audit.json','training-documents.json','split-audit.json','gold-eligibility.json']:
        shutil.copyfile(source/name,run/name)
    corpus=run/'clm-tokens.npy'
    if not corpus.exists():os.link(source/'clm-tokens.npy',corpus)
    tokens=np.load(corpus,mmap_mode='r');tok=native_tokenizer()
    kept,excluded=complete_task_index(tokens,read(run/'training-documents.json'),
        tok.encode('<｜Assistant｜>',add_special_tokens=False)[0],tok.encode('</think>',add_special_tokens=False),settings['sequence_length'])
    if not kept:raise RuntimeError('No complete training tasks')
    dump(run/'complete-task-index.json',kept);dump(run/'length-excluded-training-tasks.json',excluded)
    dump(run/'training-data-audit.json',dict(status='PASS',whole_prompt_for_every_retained_answer=True,
        prompt_label_weight=0,answer_and_actual_eos_label_weight=1,clipped_tasks=0,
        retained_tasks=len(kept),length_excluded_tasks=len(excluded),max_tokens=settings['sequence_length'],
        retained_by_database=dict(Counter(x['db_id'] for x in kept)),
        input_tokens=sum(x['tokens'] for x in kept),supervised_answer_and_eos_tokens=sum(x['supervised_tokens'] for x in kept),
        source_split_audit_sha256=sha(source/'split-audit.json'),selection_sha256=sha(run/'selection.json'),audit_sha256=sha(run/'audit.json'),
        evaluation300_read=False,external_calls=0))
    protocol=dict(old)
    protocol.update(at=now(),status='AUTHORIZED_LOCAL_CAPABILITY_RECOVERY_ATTEMPT2',scientific_attempt=2,
        literature='ReTok embedding/head adaptation plus Llama2 section3.1 assistant-only autoregressive supervision; bounded local domain adaptation, not exact reproduction',
        start='Existing step0256 embedding selected by observed selection screen, no old IPD adapters',initial_embedding=str(initial),
        train_documents=len(kept),train_tokens=sum(x['tokens'] for x in kept),
        preserved_source_corpus_tokens=len(tokens),training_databases=dict(Counter(x['db_id'] for x in kept)),
        training=dict(**settings,microbatch=1,schedule='cosine',objective='Standard answer-only causal CE including actual EOS; entire task retained; no prompt targets or arbitrary cuts',
            trainable='FP32 shared embedding and output head only',frozen='all remaining weights',
            optimizer='FP32 AdamW CPU master/moments; per-example mean answer CE, average2complete tasks perupdate'),
        code_sha256=sha(__file__))
    protocol.pop('resource_queue',None)
    assets=[config_path,initial,source/'protocol.json',source/'split-audit.json',run/'clm-tokens.npy',run/'selection.json',run/'audit.json',run/'complete-task-index.json',run/'training-documents.json']
    protocol['assets']={str(p):sha(p) for p in assets}
    dump(run/'protocol.json',protocol)
    snapshot_dir=run/'code-snapshot';snapshot_dir.mkdir()
    for path in [Path(__file__),config_path,ROOT/'tests/test_vocab_capability_repair.py']:
        shutil.copyfile(path,snapshot_dir/path.name)
    state(run,'REPAIR_COMPLETE_TASK_PREPARED',retained_tasks=len(kept),excluded_length=len(excluded),whole_prompt_answer_fraction=1.,external_calls=0)


def repair_snapshot_evaluate(run,args):
    if not args.allow_gpu:raise RuntimeError('Local snapshot inference requires --allow-gpu')
    if not args.tag:raise ValueError('Snapshot tag required')
    repair_resource_wait(run)
    import torch
    from safetensors.torch import load_file
    model=repair_load(torch,REPAIR_OLD/'native-checkpoint');model.eval()
    weight=load_file(str(run/'snapshots'/args.tag/'embedding.safetensors'))['embedding']
    with torch.no_grad():model.get_input_embeddings().weight.copy_(weight)
    del weight
    for p in model.parameters():p.requires_grad_(False)
    repair_evaluate_model(run,model,native_tokenizer(),args.tag,args.split,torch)


def body_target_modules(model,spec):
    import torch
    root=spec.get('text_root','model.language_model')
    names=[name for name,module in model.named_modules()
        if name.startswith(root+'.') and 'visual' not in name and isinstance(module,torch.nn.Linear)]
    if not names:raise RuntimeError('No text-body linear modules selected')
    return sorted(names)


def body_wrap(model,spec,adapter_path=None,trainable=True):
    from peft import LoraConfig,get_peft_model,PeftModel
    for parameter in model.parameters():parameter.requires_grad_(False)
    targets=body_target_modules(model,spec)
    if adapter_path is None:
        model=get_peft_model(model,LoraConfig(r=spec['rank'],lora_alpha=spec['alpha'],lora_dropout=spec['dropout'],
            target_modules=targets,bias='none',task_type='CAUSAL_LM'),adapter_name='recovery')
    else:model=PeftModel.from_pretrained(model,adapter_path,adapter_name='recovery',is_trainable=trainable)
    if model.get_input_embeddings().weight.requires_grad or model.get_output_embeddings().weight.requires_grad:
        raise RuntimeError('Frozen embedding/head became trainable')
    learned=[name for name,p in model.named_parameters() if p.requires_grad]
    if trainable and (not learned or any('lora_' not in name for name in learned)):
        raise RuntimeError('Unexpected body recovery trainable parameter')
    return model,targets


def body_save_adapter(model,path):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    path.mkdir(parents=True,exist_ok=True)
    values={key:value.detach().cpu().contiguous() for key,value in get_peft_model_state_dict(model,adapter_name='recovery').items()}
    temp=path/'adapter_model.safetensors.tmp';save_file(values,str(temp));temp.replace(path/'adapter_model.safetensors')
    model.peft_config['recovery'].save_pretrained(path)


def body_base_model(run,torch,embedding_path=None):
    from safetensors.torch import load_file
    protocol=read(run/'protocol.json');model=repair_load(torch,REPAIR_OLD/'native-checkpoint')
    torch.cuda.set_per_process_memory_fraction(protocol['training']['cuda_allocator_cap_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    weight=load_file(str(embedding_path or protocol['initial_embedding']))['embedding']
    with torch.no_grad():model.get_input_embeddings().weight.copy_(weight)
    del weight
    for p in model.parameters():p.requires_grad_(False)
    model.eval();return model


def body_prepare(run,args):
    import tomllib
    config_path=ROOT/'configs/bird_B_repair_attempt3_20261021.toml'
    config=tomllib.loads(config_path.read_text());source=ROOT/config['initialization']['source_run'];tag=config['initialization']['snapshot']
    if (run/'protocol.json').exists():raise RuntimeError('Final body attempt already frozen')
    prior=read(source/'protocol.json')
    for path,expected in prior['assets'].items():
        if sha(path)!=expected:raise RuntimeError('Prior complete-task input changed: '+path)
    run.mkdir(parents=True,exist_ok=True)
    copied=['selection.json','audit.json','complete-task-index.json','training-documents.json','training-data-audit.json','length-excluded-training-tasks.json','split-audit.json','gold-eligibility.json']
    for name in copied:shutil.copyfile(source/name,run/name)
    if not (run/'clm-tokens.npy').exists():os.link(source/'clm-tokens.npy',run/'clm-tokens.npy')
    initial=source/'snapshots'/tag/'embedding.safetensors'
    protocol=dict(prior);protocol.update(at=now(),scientific_attempt=3,status='AUTHORIZED_FINAL_BODY_RECOVERY_ATTEMPT',
        start='Frozen transferred embedding/head from attempt2step96; fresh body recovery LoRA, no old IPD adapter',
        initial_embedding=str(initial),reference_run=str(source),training=dict(config['repair']),body_lora=dict(config['body_lora']),
        final_scientific_attempt=True,code_sha256=sha(__file__),
        literature='ReTok section3.3 post-embedding body adaptation; LoRA; QLoRA adapter-coverage ablation; BF16 engineering adaptation, not exact paper replication')
    protocol['training'].update(microbatch=1,objective='Assistant-answer-only CE plus actual EOS; entire task; embedding/head/base frozen; body LoRA only',
        optimizer='AdamW on FP32 recovery LoRA parameters, no protected data or external teacher',full_validation='At each improved screen primary score; always evaluate final best screen; early stop on unchanged full64 recovery gate')
    assets=[config_path,initial,source/'protocol.json',run/'clm-tokens.npy',*(run/name for name in copied)]
    protocol['assets']={str(path):sha(path) for path in assets};dump(run/'protocol.json',protocol)
    snapshot=run/'code-snapshot';snapshot.mkdir()
    for path in [Path(__file__),config_path,ROOT/'tests/test_vocab_capability_repair.py']:
        shutil.copyfile(path,snapshot/path.name)
    dump(run/'reference-reuse.json',dict(source_run=str(source),source_protocol_sha256=sha(source/'protocol.json'),
        selection_sha256=sha(run/'selection.json'),source_selection_sha256=sha(source/'selection.json'),
        original_revision=REVISION,validation=protocol['validation'],reason='Same frozen prompts/items, original model and decoding; no final audit generated'))
    state(run,'BODY_REPAIR_PREPARED',scientific_attempt=3,whole_tasks=len(read(run/'complete-task-index.json')),external_calls=0)


def body_verify_protocol(run):
    protocol=read(run/'protocol.json')
    if protocol.get('scientific_attempt')!=3 or not protocol.get('final_scientific_attempt'):
        raise RuntimeError('Only authorized third body recovery attempt permitted')
    if protocol['code_sha256']!=sha(__file__):raise RuntimeError('Final body code changed')
    for path,expected in protocol['assets'].items():
        if sha(path)!=expected:raise RuntimeError('Final body frozen input changed: '+path)
    return protocol


def body_init(run,args):
    if not args.allow_gpu:raise RuntimeError('GPU permission flag required')
    protocol=body_verify_protocol(run);repair_resource_wait(run)
    import torch
    torch.manual_seed(REPAIR_SEED);torch.cuda.manual_seed_all(REPAIR_SEED)
    model=body_base_model(run,torch)
    prompt=native_tokenizer().encode(repair_prompt(read(run/'selection.json')[0]),add_special_tokens=False)
    probe=torch.tensor([prompt],device='cuda')
    with torch.inference_mode():before=model(input_ids=probe,attention_mask=torch.ones_like(probe),use_cache=False,logits_to_keep=1).logits.cpu()
    model,targets=body_wrap(model,protocol['body_lora']);model.eval()
    with torch.inference_mode():after=model(input_ids=probe,attention_mask=torch.ones_like(probe),use_cache=False,logits_to_keep=1).logits.cpu()
    difference=float((before-after).abs().max())
    if difference!=0:raise RuntimeError('Fresh body adapter changed initial model logits')
    del before,after,probe
    # Dense HF labels and the chunked masked CE use identical model computation.
    ids=torch.tensor([prompt[:96]],device='cuda');labels=ids.clone();labels[:,:64]=-100
    with torch.inference_mode():
        direct=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,labels=labels).loss
        hidden=model.get_base_model().model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        chunked=chunked_clm_loss(hidden,model.get_output_embeddings().weight,ids,protocol['training']['head_loss_chunk_tokens'],labels=labels)
    error=float((direct-chunked).abs())
    if error>2e-3:raise RuntimeError('Body masked objective differs from dense HF loss')
    del direct,chunked,hidden
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model.get_base_model().model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        gradient_probe=chunked_clm_loss(hidden,model.get_output_embeddings().weight,ids,128,labels=labels)
    gradient_probe.backward()
    missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:raise RuntimeError('Selected body adapters bypassed in real forward: '+str(missing))
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise RuntimeError('Nonfinite real body preflight gradient')
    model.zero_grad(set_to_none=True);model.gradient_checkpointing_disable();model.eval()
    del hidden,gradient_probe,ids,labels
    folder=run/'snapshots/step0000';folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'embedding.safetensors').exists():os.link(protocol['initial_embedding'],folder/'embedding.safetensors')
    body_save_adapter(model,folder/'body-adapter')
    learned=[(name,p.numel()) for name,p in model.named_parameters() if p.requires_grad]
    dump(run/'body-targets.json',targets)
    dump(run/'body-preflight.json',dict(status='NUMERICAL_AND_TRAINABLE_SCOPE_PASS',initial_logit_difference=difference,
        dense_chunked_masked_loss_difference=error,all_selected_adapter_gradients_reached=True,optimizer_updates=0,target_modules=len(targets),trainable_parameters=sum(n for _,n in learned),
        trainable_names=[name for name,_ in learned],embedding_and_head_frozen=True,external_calls=0,at=now()))
    state(run,'BODY_INITIALIZED',target_modules=len(targets),trainable_parameters=sum(n for _,n in learned),optimizer_updates=0)


def body_weight_audit(checkpoint,initial_embedding,targets,torch):
    from safetensors import safe_open
    from safetensors.torch import load_file
    original=read(BASE/'model.safetensors.index.json')['weight_map'];dest=read(checkpoint/'model.safetensors.index.json')['weight_map']
    initial=load_file(str(initial_embedding))['embedding'];allowed={name+'.weight' for name in targets}
    unchanged=0;changed=[];embedding_count=0
    for name,shard in dest.items():
        with safe_open(checkpoint/shard,framework='pt',device='cpu') as f:right=f.get_tensor(name)
        if name.endswith('embed_tokens.weight') or name=='lm_head.weight':
            if not torch.equal(right,initial.to(right.dtype)):raise RuntimeError('Frozen transferred embedding changed')
            embedding_count+=1;continue
        with safe_open(BASE/original[name],framework='pt',device='cpu') as f:left=f.get_tensor(name).to(right.dtype)
        if torch.equal(left,right):unchanged+=1
        elif name in allowed:
            delta=float((left.float()-right.float()).abs().max())
            if not math.isfinite(delta):raise RuntimeError('Nonfinite merged body weight')
            changed.append(dict(name=name,max_difference=delta))
        else:raise RuntimeError('Untargeted base weight changed: '+name)
    if not embedding_count:raise RuntimeError('Exported embedding missing')
    return dict(unchanged_nonembedding_tensors=unchanged,changed_permitted_target_tensors=changed,
        frozen_embedding_tensors_equal=embedding_count,comparison='Original weights rounded to delivered dtype; intentional LoRA target deltas permitted only')


def body_export(run,args):
    if not args.allow_gpu:raise RuntimeError('GPU permission flag required')
    tag=args.tag;folder=run/'snapshots'/tag;manifest=folder/'export-manifest.json';protocol=read(run/'protocol.json')
    checkpoint=folder/'native-checkpoint'
    if manifest.exists():
        existing=read(manifest)
        for name,expected in existing['checkpoint_files'].items():
            if sha(checkpoint/name)!=expected:raise RuntimeError('Existing canonical export changed')
        state(run,'BODY_EXPORT_ALREADY_VERIFIED',tag=tag)
        return
    repair_resource_wait(run)
    import torch
    model=body_base_model(run,torch,folder/'embedding.safetensors');targets=[]
    if (folder/'body-adapter/adapter_model.safetensors').exists():
        model,targets=body_wrap(model,protocol['body_lora'],folder/'body-adapter',trainable=False)
        model=model.merge_and_unload(safe_merge=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    if any('lora_' in name for name,_ in model.named_parameters()):raise RuntimeError('Canonical model still contains a recovery adapter')
    if model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr():raise RuntimeError('Untied export')
    prompt=native_tokenizer().encode(repair_prompt(read(run/'selection.json')[0]),add_special_tokens=False)
    x=torch.tensor([prompt],device='cuda')
    with torch.inference_mode():before=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.cpu()
    checkpoint.mkdir(parents=True,exist_ok=True);model.config.use_cache=True
    model.save_pretrained(checkpoint,safe_serialization=True,max_shard_size='2GB');native_tokenizer().save_pretrained(checkpoint)
    shutil.copyfile(META/'encoding--encoding.py',checkpoint/'deepseek_encoding.py')
    dump(checkpoint/'native-contract.json',dict(vocabulary=VOCAB,bos=0,eos=1,pad=2,recovery_only_not_ipd=True,
        formatter='official DS chat single-user thinking off',protocol_sha256=sha(run/'protocol.json'),source_tag=tag,
        body_recovery_merged=bool(targets),capability_status='NOT_YET_CERTIFIED_BY_FINAL_AUDIT',initial_embedding_sha256=sha(folder/'embedding.safetensors')))
    audit=body_weight_audit(checkpoint,folder/'embedding.safetensors',targets,torch)
    if tag!='step0000' and targets and not audit['changed_permitted_target_tensors']:raise RuntimeError('Trained adapter produced no merged body change')
    del model;gc.collect();torch.cuda.empty_cache();model=repair_load(torch,checkpoint);model.eval()
    with torch.inference_mode():after=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.cpu()
    difference=float((before-after).abs().max())
    if difference!=0:raise RuntimeError('Merged canonical model independent-reload mismatch')
    result=dict(status='CANONICAL_EXPORT_AND_RELOAD_PASS',tag=tag,checkpoint=str(checkpoint),body_targets=targets,
        body_weight_audit=audit,independent_reload_max_logit_difference=difference,
        checkpoint_files={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},external_calls=0,at=now())
    dump(manifest,result);state(run,'BODY_EXPORTED',tag=tag,reload_max_logit_difference=difference)


def body_validate(run,args):
    if not args.allow_gpu:raise RuntimeError('GPU permission flag required')
    tag=args.tag;folder=run/'snapshots'/tag;manifest=read(folder/'export-manifest.json')
    for name,expected in manifest['checkpoint_files'].items():
        if sha(folder/'native-checkpoint'/name)!=expected:raise RuntimeError('Canonical validation checkpoint changed')
    repair_resource_wait(run)
    import torch
    model=repair_load(torch,folder/'native-checkpoint');model.eval()
    for p in model.parameters():p.requires_grad_(False)
    reference=Path(read(run/'protocol.json')['reference_run'])
    if sha(run/'selection.json')!=sha(reference/'selection.json'):raise RuntimeError('Original reference task inputs changed')
    if args.split=='selection':result=repair_evaluate_model(run,model,native_tokenizer(),tag,'selection',torch)
    else:result=repair_evaluate_model(run,model,native_tokenizer(),tag,'selection',torch,screen=True)
    original=read(reference/('evaluation/original/selection/summary.json' if args.split=='selection' else 'evaluation/original/screen/summary.json'))
    passed=repair_gate(result,original,screen=args.split!='selection')
    dump(folder/('full-validation-decision.json' if args.split=='selection' else 'screen-decision.json'),dict(passed=passed,metric=result,reference=original,at=now()))
    if args.split=='selection' and passed:
        dump(run/'selected.json',dict(tag=tag,selection=result,reference=original,checkpoint=manifest['checkpoint'],at=now()))
        state(run,'SELECTED',tag=tag,full_selection_pass=True)


def body_run(run,args):
    if not args.allow_gpu:raise RuntimeError('GPU permission flag required')
    protocol=body_verify_protocol(run)
    if read(run/'snapshots/step0000/export-manifest.json')['status']!='CANONICAL_EXPORT_AND_RELOAD_PASS':
        raise RuntimeError('Zero-update canonical export preflight missing')
    repair_resource_wait(run)
    import torch
    import numpy as np
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    torch.manual_seed(REPAIR_SEED);torch.cuda.manual_seed_all(REPAIR_SEED)
    model=body_base_model(run,torch)
    model,targets=body_wrap(model,protocol['body_lora'],run/'snapshots/step0000/body-adapter')
    if targets!=read(run/'body-targets.json'):raise RuntimeError('Runtime body target list changed')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    optimizer=torch.optim.AdamW(params,lr=protocol['training']['learning_rate'],weight_decay=0,foreach=False)
    start=0
    if args.resume:
        saved=torch.load(run/'resume.pt',map_location='cpu',weights_only=True)
        if saved['protocol_sha256']!=sha(run/'protocol.json') or saved['parameter_names']!=[name for name,_ in named]:raise RuntimeError('Body resume configuration mismatch')
        set_peft_model_state_dict(model,saved['adapter'],adapter_name='recovery')
        optimizer.load_state_dict(saved['optimizer']);start=saved['update']
        states={int(v['step']) for v in optimizer.state.values()}
        if states!={start}:raise RuntimeError('Body AdamW step not restored exactly')
        # The adapter state is checked at the same model boundary used by saving.
        actual=get_peft_model_state_dict(model,adapter_name='recovery')
        if any(not torch.equal(actual[k].detach().cpu(),v) for k,v in saved['adapter'].items()):raise RuntimeError('Body adapter weights not restored exactly')
        del saved
        state(run,'BODY_RESUMED',update=start,optimizer_step=start)
    elif (run/'training.jsonl').exists():raise RuntimeError('Existing body training requires explicit resume')
    tokens=np.load(run/'clm-tokens.npy',mmap_mode='r');documents=read(run/'complete-task-index.json');settings=protocol['training']
    order=np.random.default_rng(REPAIR_SEED).permutation(len(documents));accum=settings['grad_accum']
    processed=sum(documents[int(order[i%len(order)])]['tokens'] for i in range(start*accum))
    supervised=sum(documents[int(order[i%len(order)])]['supervised_tokens'] for i in range(start*accum))
    boundary=min(settings['max_updates'],((start//settings['validation_interval'])+1)*settings['validation_interval'])
    if start>=settings['max_updates']:raise RuntimeError('Final body update budget already exhausted')
    for update in range(start,boundary):
        began=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
        warmup=settings['warmup_updates'];maximum=settings['max_updates']
        lr=settings['learning_rate']*min(1.,(update+1)/warmup)*.5*(1+math.cos(math.pi*max(0,update-warmup)/max(1,maximum-warmup)))
        optimizer.param_groups[0]['lr']=lr;losses=[]
        before=params[-1].detach().clone()
        for micro in range(accum):
            document=documents[int(order[(update*accum+micro)%len(order)])]
            raw,targets_array=complete_task_arrays(tokens,document);ids=torch.tensor(raw[None],device='cuda');labels=torch.tensor(targets_array[None],device='cuda')
            processed+=len(raw);supervised+=document['supervised_tokens']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=model.get_base_model().model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
                loss=chunked_clm_loss(hidden,model.get_output_embeddings().weight,ids,settings['head_loss_chunk_tokens'],labels=labels)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite body recovery loss')
            (loss/accum).backward();losses.append(float(loss.detach()));del hidden,loss,ids,labels
        norm=torch.nn.utils.clip_grad_norm_(params,settings['gradient_clip'])
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite body gradient')
        if model.get_input_embeddings().weight.grad is not None or model.get_input_embeddings().weight.requires_grad:raise RuntimeError('Embedding freeze violated')
        optimizer.step();delta=float((params[-1].detach()-before).abs().sum());del before
        if not math.isfinite(delta):raise RuntimeError('Nonfinite body parameter delta')
        adapter={k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model,adapter_name='recovery').items()}
        temp=run/'resume.pt.tmp';torch.save(dict(update=update+1,protocol_sha256=sha(run/'protocol.json'),adapter=adapter,
            optimizer=optimizer.state_dict(),parameter_names=[name for name,_ in named]),temp);temp.replace(run/'resume.pt')
        dump(run/'resume.json',dict(update=update+1,protocol_sha256=sha(run/'protocol.json'),at=now()))
        record=dict(at=now(),update=update+1,loss=sum(losses)/len(losses),learning_rate=lr,gradient_norm=float(norm),
            parameter_probe_delta_l1=delta,seen_tokens=processed,supervised_tokens=supervised,
            complete_prompt_for_every_target=True,embedding_and_head_frozen=True,trainable_parameters=sum(p.numel() for p in params),
            seconds=time.monotonic()-began,cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20,external_calls=0)
        with (run/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        state(run,'BODY_TRAINING',**{k:v for k,v in record.items() if k!='at'})
    tag=f'step{boundary:04d}';folder=run/'snapshots'/tag;folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'embedding.safetensors').exists():os.link(protocol['initial_embedding'],folder/'embedding.safetensors')
    body_save_adapter(model,folder/'body-adapter')
    state(run,'BODY_NEEDS_VALIDATION',tag=tag,update=boundary,optimizer_saved=True)


def body_coordinate(run,args):
    if not args.allow_gpu:raise RuntimeError('GPU permission flag required')
    import subprocess
    protocol=body_verify_protocol(run);repair_resource_wait(run)
    dump(run/'gpu-lease.json',dict(pid=os.getpid(),run=str(run.resolve()),at=now()))
    env=dict(os.environ,PRIVACY_SYNTH_GPU_LEASE=str((run/'gpu-lease.json').resolve()))
    def stage(mode,tag=None,split=None,resume=False):
        command=[sys.executable,__file__,mode,'--run',str(run),'--allow-gpu']
        if tag:command+=['--tag',tag]
        if split:command+=['--split',split]
        if resume:command+=['--resume']
        state(run,'BODY_COORDINATING',next_mode=mode,tag=tag)
        subprocess.run(command,env=env,check=True)
    if not (run/'body-preflight.json').exists():stage('body-init')
    if not (run/'snapshots/step0000/export-manifest.json').exists():stage('body-export','step0000')
    history=read(run/'body-validation-history.json') if (run/'body-validation-history.json').exists() else []
    while not (run/'selected.json').exists():
        current=read(run/'resume.json')['update'] if (run/'resume.json').exists() else 0
        if current>=protocol['training']['max_updates']:break
        stage('body-run',resume=current>0)
        current=read(run/'resume.json')['update'];tag=f'step{current:04d}'
        stage('body-export',tag);stage('body-validate',tag,'screen')
        metric=read(run/'snapshots'/tag/'screen-decision.json')['metric']
        primary=(metric['correct'],metric['parse_ok'],-metric['truncated'])
        prior=max(((x['screen']['correct'],x['screen']['parse_ok'],-x['screen']['truncated']) for x in history),default=(-1,-1,-999))
        entry=dict(tag=tag,update=current,screen=metric,full_selection_evaluated=False)
        if primary>prior:
            stage('body-validate',tag,'selection');entry['full_selection_evaluated']=True
            entry['selection']=read(run/'snapshots'/tag/'full-validation-decision.json')['metric']
        history.append(entry);dump(run/'body-validation-history.json',history)
    if not (run/'selected.json').exists():
        if not history:raise RuntimeError('No body validation candidate')
        best=min(history,key=lambda x:(-x['screen']['correct'],-x['screen']['parse_ok'],x['screen']['truncated'],x['screen']['answer_nats_per_byte'],x['update']))
        if not best['full_selection_evaluated']:
            stage('body-validate',best['tag'],'selection');best['full_selection_evaluated']=True
            best['selection']=read(run/'snapshots'/best['tag']/'full-validation-decision.json')['metric'];dump(run/'body-validation-history.json',history)
    status='BODY_FULL_SELECTION_PASS' if (run/'selected.json').exists() else 'FINAL_BODY_ATTEMPT_FINISHED_NOT_RESTORED'
    dump(run/'body-training-result.json',dict(status=status,updates=read(run/'resume.json')['update'],maximum_scientific_attempts=3,
        audit_not_run=True,ipd_not_started=True,external_calls=0,at=now()))
    state(run,status,ipd_not_started=True,audit_not_run=True)


def repair_coordinate(run,args):
    import subprocess
    if not args.allow_gpu:raise RuntimeError('Local repair requires --allow-gpu')
    modes=['repair-run']
    if (run/'resume.pt').exists():modes+=['--resume']
    while True:
        state(run,'REPAIR_COORDINATING',next_mode=modes[0])
        result=subprocess.run([sys.executable,__file__,*modes,'--run',str(run),'--allow-gpu'])
        if result.returncode:raise RuntimeError('Local repair child failed; checkpoints retained; no automatic uncertain restart')
        if (run/'final.json').exists():return
        status=read(run/'status.json')['phase']
        if status=='NEEDS_FULL_REFERENCE':
            subprocess.run([sys.executable,__file__,'repair-reference','--run',str(run),'--split','selection','--allow-gpu'],check=True)
            modes=['repair-run','--resume']
        elif status=='SELECTED':
            subprocess.run([sys.executable,__file__,'repair-finalize','--run',str(run),'--allow-gpu'],check=True);return
        else:raise RuntimeError('Unexpected recovery state '+status)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['prepare','recover','reload','repair-prepare','repair-run','repair-score','repair-reference','repair-finalize','repair-coordinate','repair-snapshot-evaluate','repair-prepare-complete','body-prepare','body-init','body-export','body-validate','body-run','body-coordinate']); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--score-gate',type=Path); p.add_argument('--allow-gpu',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--tag');p.add_argument('--split',choices=['selection','audit','screen'],default='selection');args=p.parse_args()
    try:
        if args.mode=='prepare':prepare(args.run)
        elif args.mode=='recover':recover(args.run,args)
        elif args.mode=='reload':reload_check(args.run,args)
        elif args.mode=='repair-prepare':repair_prepare(args.run)
        elif args.mode=='repair-run':repair_run(args.run,args)
        elif args.mode=='repair-score':repair_score(args.run,args.tag,args.split)
        elif args.mode=='repair-reference':repair_reference(args.run,args)
        elif args.mode=='repair-finalize':repair_finalize(args.run,args)
        elif args.mode=='repair-prepare-complete':repair_prepare_complete(args.run,args)
        elif args.mode=='repair-snapshot-evaluate':repair_snapshot_evaluate(args.run,args)
        elif args.mode=='body-prepare':body_prepare(args.run,args)
        elif args.mode=='body-init':body_init(args.run,args)
        elif args.mode=='body-export':body_export(args.run,args)
        elif args.mode=='body-validate':body_validate(args.run,args)
        elif args.mode=='body-run':body_run(args.run,args)
        elif args.mode=='body-coordinate':body_coordinate(args.run,args)
        else:repair_coordinate(args.run,args)
    except Exception as exc:
        import traceback
        args.run.mkdir(parents=True,exist_ok=True)
        dump(args.run/(args.mode+'-failure.json'),{'at':now(),'mode':args.mode,'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise
