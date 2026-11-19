"""CPU-only exact option cycles with a visible, fail-closed weak-order rule.

This one-off adapter does not change the frozen task-contract module. All old
inputs/results remain untouched. Preparation does not authorize GPU training.
"""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import copy,hashlib,json,os,random,re,sys
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_dev_semantic_transforms_20261106 as transform
import ust_position_balanced_training_prepare_20261106 as reference
from privacy_synth.ust_task_contract import (ContractError,certify_native_ranking_reference,NATIVE_UNDERSTANDING_NOTE,LEARNED_UNDERSTANDING_NOTE)
N=R/'runs/2026-11-06-ust-retention-information-v1';O=N/'option-cycle-grouping-weak-v3'
SOURCE=R/'runs/2026-11-06-ust-autonomous-repair-v1/position-training-v2/synthetic-balanced200'
FIXED8=SOURCE.parent/'fixed8.local.json'
RULE="Ranking convention: compare the displayed values in non-increasing order. Equal values may appear in either order. In the options, '>' marks ranking order and permits equality."
ARMS=('same-source-four','random-four','distinct-source-balanced-four');SEED=2026110701
LETTERS='ABCD';IDENTITY={i:i for i in range(1,5)}
def load(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return dict(path=str(p.relative_to(R)) if p.is_relative_to(R) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify_pins(pins):
 for pin in pins:assert bind(R / pin['path'])==pin,pin['path']
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()
def note_parts(question):
 matches=list(re.finditer(r'^## Note\n\n',question,re.M))
 if len(matches)!=1:raise ContractError('Exactly one canonical Note header is required')
 match=matches[0];return question[:match.end()],question[match.end():]
def clarify_native(question):
 head,note=note_parts(question)
 if note not in (NATIVE_UNDERSTANDING_NOTE,LEARNED_UNDERSTANDING_NOTE):raise ContractError('Unknown complete original Note')
 return head+RULE+'\n'+note

def unwrap_explicit_weak(question):
 head,note=note_parts(question)
 originals=[old for old in (NATIVE_UNDERSTANDING_NOTE,LEARNED_UNDERSTANDING_NOTE) if note==RULE+'\n'+old]
 if len(originals)!=1:raise ContractError('The entire explicit weak-order Note must match exactly; no extra or duplicate instructions')
 return head+originals[0]

def certify_explicit_weak_reference(question,answer):
 base=unwrap_explicit_weak(question);cert=certify_native_ranking_reference(base,answer)
 return dict(cert,scope='explicit_visible_descending_weak_order_only',ranking_semantics='non_increasing_displayed_values',ties='equal_values_either_order',option_symbol='> is rank order and permits equality',original_native_scope=cert['scope'],explicit_rule=RULE)

def independent_numeric(question):
 base=unwrap_explicit_weak(question);parts=transform.sections(base);intro,options,suffix=transform._options(parts['Question']);assert not suffix
 metric=reference.UNDER[intro];values={}
 for location,body in transform._data_blocks(parts['Data']).items():
  line=next(line for line in body.splitlines() if line.startswith('- '+metric+': '));raw=line.split(': ',1)[1]
  if raw in ('NaN','NaNh','NaN h'):
   assert metric in ('waiting_time','charging_time')
   required='NaN indicates no station nearby and the highest '+metric.replace('_',' ')+'.'
   assert next(x for x in parts['Data Schema'].splitlines() if x.startswith('- '+metric+':')).endswith(required)
   values[location]=(1,Decimal(0))
  else:values[location]=(0,Decimal(raw[:-1]))
 answers=[];strict=[];relations={}
 for letter,body in options.items():
  seq=[int(v) for v in re.findall('[1-4]',body)];assert len(seq)==4 and set(seq)=={1,2,3,4}
  weak=all(values[a]>=values[b] for a,b in zip(seq,seq[1:]));strong=all(values[a]>values[b] for a,b in zip(seq,seq[1:]))
  if weak:answers.append(letter)
  if strong:strict.append(letter)
  relations[letter]=dict(location_order=seq,weak_compatible=weak,literal_strict_compatible=strong)
 return dict(metric=metric,compatible_answers=answers,literal_strict_answers=strict,has_displayed_ties=len(set(values.values()))<4,has_target_NaN=any(v[0] for v in values.values()),option_relations=relations)

def groups(variants):
 sources=[];lookup={}
 for row in variants:
  if row['source_id'] not in lookup:sources.append(row['source_id']);lookup[row['source_id']]={}
  assert row['answer'] not in lookup[row['source_id']];lookup[row['source_id']][row['answer']]=row['id']
 assert len(sources)>=4 and all(set(x)==set(LETTERS) for x in lookup.values())
 permutation=list(sources);random.Random(SEED).shuffle(permutation)
 a=[[lookup[source][letter] for letter in LETTERS] for source in permutation]
 ids=[r['id'] for r in variants];random.Random(SEED).shuffle(ids);b=[ids[i:i+4] for i in range(0,len(ids),4)]
 c=[[lookup[permutation[(i+j)%len(sources)]][letter] for j,letter in enumerate(LETTERS)] for i in range(len(sources))]
 return dict(zip(ARMS,(a,b,c))),permutation

def qualify_groups(variants,batches):
 byid={r['id']:r for r in variants};allids=Counter(byid.keys());n=len(variants)//4;proof={}
 assert len(byid)==len(variants)==4*n
 for arm,bb in batches.items():
  assert len(bb)==n and all(len(b)==4 for b in bb)
  flat=[i for b in bb for i in b];assert Counter(flat)==allids
  assert set(Counter(byid[i]['source_id'] for i in flat).values())=={4}
  rows=[]
  for b in bb:
   source_n=len({byid[i]['source_id'] for i in b});letters=[byid[i]['answer'] for i in b]
   if arm==ARMS[0]:assert source_n==1 and letters==list(LETTERS)
   if arm==ARMS[2]:assert source_n==4 and letters==list(LETTERS)
   rows.append(dict(distinct_sources=source_n,answer_letters=letters))
  proof[arm]=dict(batches=n,microexamples=len(flat),multiset_sha256=digest(sorted(flat)),source_exposures=dict(Counter(byid[i]['source_id'] for i in flat)),
   distinct_sources_histogram=dict(Counter(r['distinct_sources'] for r in rows)),balanced_letter_batches=sum(Counter(r['answer_letters'])==Counter(LETTERS) for r in rows),per_batch=rows)
 assert len({x['multiset_sha256'] for x in proof.values()})==1
 return proof

def prepare():
 assert not O.exists(),'Immutable namespace, no automatic retry';O.mkdir(parents=True)
 save(O/'started.json',dict(action='CPU_DATA_PREPARATION_ONLY',GPU=0,API=0,physics=0))
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='1'
 pins=[bind(__file__),bind(R/'tests/test_ust_option_cycle_weak_data_v2.py'),bind(R/'scripts/ust_option_cycle_weak_data_20261106.py'),bind(SOURCE/'training.local.json'),bind(SOURCE/'protocol.json'),bind(FIXED8),bind(Path(transform.__file__)),bind(Path(reference.__file__)),bind(R/'src/privacy_synth/ust_task_contract.py')]
 from transformers import AutoTokenizer
 import torch
 import probe_overnight_train_fit_20261025 as packing
 assert not torch.cuda.is_initialized();p=load(SOURCE/'protocol.json');tok=AutoTokenizer.from_pretrained(p['base'],local_files_only=True,trust_remote_code=False);fixed8=load(FIXED8)
 pins.append(bind(Path(packing.__file__)))
 for path in Path(p['base']).iterdir():
  if path.is_file() and path.suffix in ('.json','.txt','.model','.tiktoken'):pins.append(bind(path))
 rows=load(SOURCE/'training.local.json');source_rows=[(i,r) for i,r in enumerate(rows) if r['metadata']['ability']=='st_understanding'];assert len(source_rows)==68 and len({r['id'] for _,r in source_rows})==68
 variants=[];checks=[];examples={};source_certificates=[];token_lengths=[]
 for source_index,(training_index,row) in enumerate(source_rows):
  old_answer=reference.target(row);old_cert=certify_native_ranking_reference(row['question'],old_answer);assert old_cert['trusted_gold']
  assert transform.canonical(row['question'])==row['question'],'No whitespace/body correction permitted'
  old_refs=reference.understood_references(row);oldparts=transform.sections(row['question']);source_certificates.append(dict(source_id=row['id'],certificate=old_cert,scope_limit='Legacy trust conditional on weak-order interpretation; old visible wording did not state ties'))
  for variant_index in range(4):
   option_order=LETTERS[variant_index:]+LETTERS[:variant_index];changed=transform.transform(row['question'],IDENTITY,option_order);mapping=changed['old_to_new_letter'];answer=mapping[old_answer]
   question=clarify_native(changed['question']);candidate=copy.deepcopy(row);candidate.update(id=f'ust-weak-cycle-v2-{source_index:03}-{variant_index}',source_id=row['id'],source_index=source_index,source_training_index=training_index,question=question,target='<Answer>'+answer+'</Answer>',answer=answer,variant_index=variant_index,old_to_new_letter=mapping)
   with (O/'attempted-variants.local.jsonl').open('a') as f:f.write(json.dumps(candidate,ensure_ascii=False)+'\n')
   cert=certify_explicit_weak_reference(question,answer);numeric=independent_numeric(question);assert cert['trusted_gold'] and cert['compatible_answers']==numeric['compatible_answers']==[answer]
   assert candidate['metadata']==row['metadata'];newparts=transform.sections(unwrap_explicit_weak(question))
   assert all(newparts[k]==v for k,v in oldparts.items() if k!='Question')
   assert transform._options(newparts['Question'])[0::2]==transform._options(oldparts['Question'])[0::2]
   inverse=transform.transform(unwrap_explicit_weak(question),IDENTITY,[mapping[x] for x in LETTERS]);assert inverse['question']==row['question']
   assert reference.understood_references(candidate)==old_refs
   if variant_index==0:assert unwrap_explicit_weak(question)==row['question'] and candidate['target']==row['target']
   prompt,target=packing.packing(tok,candidate,dict(text=candidate['target']),'matched8',fixed8)
   assert len(prompt)+len(target)<=16384 and target[-1]==tok.convert_tokens_to_ids('<|im_end|>')
   assert len(target)==7
   token_proof=dict(input_tokens=len(prompt),target_tokens=len(target),total_tokens=len(prompt)+len(target),prompt_ids_sha256=digest(prompt),target_ids_sha256=digest(target),no_truncation=True)
   checks.append(dict(variant_id=candidate['id'],source_id=row['id'],variant_index=variant_index,old_to_new_letter=mapping,option_order=option_order,location_map=IDENTITY,
    inverse_exact_original=True,all_non_option_body_text_exact=True,only_note_prefix_added=True,metadata_preserved=True,reference_guard=old_refs,old_native_conditional_certificate=old_cert,new_explicit_certificate=cert,independent_numeric=numeric,token_qualification=token_proof))
   variants.append(candidate);token_lengths.append(len(prompt))
   if variant_index==0:
    for kind,condition in [('displayed_tie',numeric['has_displayed_ties']),('strict_numeric_order',not numeric['has_displayed_ties']),('target_NaN',numeric['has_target_NaN'])]:
     if condition and kind not in examples:examples[kind]=dict(selection='First matching row in frozen source order; no model outputs read',original_training_row=row,clarified_variant=candidate,certificate=cert,numeric_relations=numeric)
 batches,permutation=groups(variants);batch_checks=qualify_groups(variants,batches)
 assert len(variants)==272 and len(checks)==272 and all(len({r['answer'] for r in variants if r['source_id']==source['id']})==4 for _,source in source_rows)
 save(O/'variants.local.json',variants);save(O/'source68.local.json',[r for _,r in source_rows]);save(O/'clarified-own68.local.json',[r for r in variants if r['variant_index']==0]);save(O/'fixed8.local.json',fixed8)
 save(O/'transformation-records.local.json',checks);save(O/'before-after-cases.local.json',examples)
 save(O/'qualification.local.json',dict(status='ALL_272_EXPLICIT_WEAK_ORDER_VARIANTS_PASS',rows=272,source_questions=68,rejected=0,dropped=0,refilled=0,
  legacy_certificates=source_certificates,displayed_tie_source_cases=sum(x['independent_numeric']['has_displayed_ties'] for x in checks if x['variant_index']==0),
  target_NaN_source_cases=sum(x['independent_numeric']['has_target_NaN'] for x in checks if x['variant_index']==0),
  input_tokens=dict(min=min(token_lengths),max=max(token_lengths)),target_tokens=7,group_qualification=batch_checks,source_permutation=permutation))
 data_pins=[bind(O/x) for x in ('variants.local.json','source68.local.json','clarified-own68.local.json','fixed8.local.json','transformation-records.local.json','before-after-cases.local.json','qualification.local.json','attempted-variants.local.jsonl')]
 for arm,bb in batches.items():
  save(O/arm/'batches.local.json',bb)
  save(O/arm/'data-protocol.json',dict(arm=arm,epoch_batches=68,batch_size=4,source_questions=68,epoch_microexamples=272,batches=bind(O/arm/'batches.local.json'),variants=bind(O/'variants.local.json'),source=bind(__file__),seed=SEED))
  data_pins += [bind(O/arm/'batches.local.json'),bind(O/arm/'data-protocol.json')]
 assert not torch.cuda.is_initialized();verify_pins(pins)
 save(O/'data-protocol.json',dict(status='ALL_THREE_GROUPINGS_FROZEN_CPU_ONLY',source=bind(__file__),bindings=pins+data_pins,arms=list(ARMS),seed=SEED,source_training=bind(SOURCE/'training.local.json'),
  variants=bind(O/'variants.local.json'),own68=bind(O/'clarified-own68.local.json'),fixed8=bind(O/'fixed8.local.json'),source_questions=68,variant_count=272,variant0='clarified source; old input recovered exactly only by removing the complete fixed Note prefix',
  note_prefix=RULE,contract='explicit non-increasing displayed-value ranking; equal values in either order; > ranking symbol permits equality',
  qualifier=dict(script=bind(__file__),function='certify_explicit_weak_reference'),old_gold_boundary='Legacy certificates are conditional on weak-order interpretation, not proof of literal strict > truth',
  grouping=dict(A='Same seed-permuted source, four variants in A/B/C/D correct-letter micro order',B='Ordinary seeded shuffle of all272 variant IDs then consecutive groups4; no filtering/refill',C='Same source permutation, fixed offsets0/1/2/3 with letter A/B/C/D; all distinct sources per batch'),
  prospective_training=dict(authorized_now=False,epochs=2,repeat_identical_batch_order=True,optimizer_calls_per_arm=136,microexamples_per_arm=544,total_optimizer_calls=408,lr=1e-4,loss='mean of4 per-example fullresponse token-mean CE',clip_once_per_batch=True,Adam_once_per_batch=True,same_original_INIT=True),
  prospective_evaluation=dict(authorized_now=False,freshINIT=308,each_trained_model=376,three_models=1128,total_new_raw=1436,step68='clarified identity-variant own68',step136='all272 clarified variants +36 unchanged Dev understanding',reuse_old_own_raw=False,Dev_wording_unchanged_report_separately=True),
  GPU_launch_gate='All four continuations and eight backend diagnostic units closed, root authorization and independent data/runner preflight',new_training=0,GPU=0,API=0,physics=0,Common_access=False,Audit_access=False,Confirm_access=False,Final_access=False))
 save(O/'terminal.json',dict(status='COMPLETE_CPU_DATA_ONLY',data_protocol=bind(O/'data-protocol.json'),variants=272,batches_per_arm=68,new_training=0,GPU=0,API=0,physics=0))
 print('CPU prepared272 variants /3x68batches; explicit weak-order all68;GPU0/API0')
if __name__=='__main__':
 try:prepare()
 except BaseException as e:
  if O.exists():save(O/'failure.json',dict(type=type(e).__name__,error=str(e),no_automatic_retry=True,GPU=0,API=0))
  raise
