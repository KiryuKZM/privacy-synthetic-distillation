"""Prepare three causal diagnostic training inputs, without GPU/model execution.

A replays the exact original learned200 bytes. B changes only the static
understanding template bundle. C retains the same68 understanding records in
original relative order, excluding132 other abilities; C has68 vs200 updates
and is an interference/exposure diagnostic, not an equal-resource utility test.
"""
from pathlib import Path
from decimal import Decimal
import collections,copy,hashlib,json,re,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import ust_dev_semantic_transforms_20261106 as t
from ust_template_crossover_prepare_20261106 import swapped
O=R/'runs/2026-11-06-ust-autonomous-repair-v1/causal-training'
P=R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1'
INIT=R/'runs/2026-10-25-qwen-transfer-repair/seqkd200/USTBench/2b/work/checkpoints/step0000'
ARMS=('A-original200-replay','B-understanding-real-template200','C-understanding-only68')

def load(p):return json.loads(p.read_text())
def bind(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def target(row):
 return re.fullmatch(r'<Answer>([ABCD])</Answer>',row['target']).group(1) if row['metadata']['ability']=='st_understanding' else json.loads(row['target'])['answer']

def independent_reference(question):
 # Deliberately independent of native-gate acceptance: this consumes exact known
 # ranking patterns and verifies visible ordering, not arbitrary prose semantics.
 bits=re.split(r'^## ([^\n]+)\n',question,flags=re.M)
 assert not bits[0].strip() and len(bits)%2==1
 p={bits[i]:bits[i+1].strip() for i in range(1,len(bits),2)}
 prompt=p['Question'];metric='distance' if 'average distance' in prompt else 'waiting_time' if 'average waiting time' in prompt else 'charging_time' if 'average charging time' in prompt else None
 assert metric
 blocks=re.split(r'(?im)^location ([1-4]):\s*\n',p['Data']);assert len(blocks)==9
 vals={};missing=0
 for i in range(1,len(blocks),2):
  tok=re.search(r'^- '+metric+r':\s*(\S+)\s*$',blocks[i+1],re.M).group(1)
  suffix='m' if metric=='distance' else 'h'
  if tok in ('NaN','NaN'+suffix):
   phrase='NaN indicates no station nearby and the highest '+metric.replace('_',' ')+'.'
   line=[s for s in p['Data Schema'].splitlines() if s.startswith('- '+metric+':')]
   assert len(line)==1 and line[0].endswith(phrase)
   vals[int(blocks[i])]=Decimal('Infinity');missing+=1
  else:
   assert tok.endswith(suffix); vals[int(blocks[i])]=Decimal(tok[:-1]);assert vals[int(blocks[i])].is_finite()
 options=re.findall(r'^([ABCD])\.\s*([1-4](?:\s*>\s*[1-4]){3})\s*$',prompt,re.M)
 assert len(options)==4 and set(vals)=={1,2,3,4}
 accepted=[]
 for letter,body in options:
  seq=list(map(int,re.findall('[1-4]',body)));assert set(seq)==set(vals)
  if all(vals[a]>=vals[b] for a,b in zip(seq,seq[1:])):accepted.append(letter)
 return dict(compatible_answers=accepted,metric=metric,NaN_requested_metric_count=missing,visible_NaN_policy='highest' if missing else 'not_used')

def main():
 O.mkdir(parents=True,exist_ok=False)
 original_path=P/'synthetic-reference-ce/training200.local.json';real_path=P/'real-reference-ce/training200.local.json'
 original=load(original_path);real=load(real_path)
 assert len(original)==200 and collections.Counter(x['metadata']['ability'] for x in original)==dict(st_understanding=68,planning=66,reflection=66)
 template=t.sections(next(r for r in real if r['metadata']['ability']=='st_understanding')['question'])
 a=original;b=copy.deepcopy(original);c=[copy.deepcopy(r) for r in original if r['metadata']['ability']=='st_understanding']
 checks=[]
 for i,(before,after) in enumerate(zip(a,b),1):
  if before['metadata']['ability']!='st_understanding':assert before==after;continue
  assert before['question']==t.canonical(before['question'])
  after['question']=swapped(before['question'],template)
  x=t.sections(before['question']);y=t.sections(after['question'])
  assert x['Data']==y['Data'] and x['Question']==y['Question'] and x['Task Description']==y['Task Description']
  assert {k:v for k,v in before.items() if k!='question'}=={k:v for k,v in after.items() if k!='question'}
  check_a=independent_reference(before['question']);check_b=independent_reference(after['question'])
  assert check_a==check_b and check_a['compatible_answers']==[target(before)]
  checks.append(dict(source_step=i,id=before['id'],Data_and_Question_byte_equal=True,metadata_target_identity_equal=True,**check_a))
 assert len(checks)==68 and sum(x['NaN_requested_metric_count']>0 for x in checks)==1
 assert len(c)==68 and all(x==y for x,y in zip(c,[r for r in a if r['metadata']['ability']=='st_understanding']))
 oldp=load(P/'synthetic-reference-ce/protocol.json');init=load(INIT/'manifest.json');assert init['actual_updates']==0
 common=dict(base=oldp['base'],seed=oldp['seed'],lr=oldp['lr'],lora=oldp['lora'],loss=oldp['loss'],
   initialization=bind(INIT/'manifest.json'),expected_role_hashes=init['role_hashes'],expected_optimizer_hashes=init['optimizer_hashes'],
   restore_initial_optimizer_and_RNG=True,optimizer=dict(name='AdamW',weight_decay=0,foreach=False,gradient_clip_norm=1),
   fixed8=bind(P/'fixed8.local.json'),packing_source=bind(R/'scripts/probe_overnight_train_fit_20261025.py'),
   CE_source=bind(R/'scripts/matched_qwen_objective_control_20261102.py'),teacher_API=0,
   no_new_fewshots=True,no_scene_or_option_reordering=True,no_targets_changed=True)
 counts={}
 for arm,rows in zip(ARMS,(a,b,c)):
  d=O/arm;d.mkdir()
  if arm==ARMS[0]:
   with (d/'training.local.json').open('xb') as f:f.write(original_path.read_bytes())
   assert (d/'training.local.json').read_bytes()==original_path.read_bytes()
  else:save(d/'training.local.json',rows)
  counts[arm]={ability:dict(n=sum(r['metadata']['ability']==ability for r in rows),labels=dict(collections.Counter(target(r) for r in rows if r['metadata']['ability']==ability))) for ability in ['st_understanding','planning','reflection']}
  save(d/'data-protocol.json',dict(status='DATA_PREPARED_NO_TRAINING',arm=arm,steps=len(rows),training=bind(d/'training.local.json'),
    common_training_spec=common,ability_label_counts=counts[arm],
    causal_interpretation='A exact original replay control' if arm==ARMS[0] else 'B compared with A: static understanding template bundle only; all200 rows/order/targets retained,132 other questions identical' if arm==ARMS[1] else 'C compared with A: same68 understanding exposures/order, remove132 planning/reflection updates. Unequal total update budgets68vs200; interference/schedule diagnostic, not equal-resource replacement utility.',
    label_validation='All68 understanding targets independently verified under visible Decimal/NaN-highest rules. Planning/reflection retained as historical experimental supervision, not certified gold.',
    native_contract_gate='Pending separate root gate audit; unsupported template is not a label mismatch. No rows may be removed or rewritten on this status without new protocol.'))
 save(O/'qualification.local.json',dict(status='PASS',A_original_training_bytes_identical=True,B_changed_questions=68,B_other_questions_identical=132,
   C_same_understanding_records_and_relative_order=True,checks=checks,ability_label_counts=counts,no_GPU=True,no_API=True,no_training=True))
 save(O/'data-protocol.json',dict(status='ALL_THREE_DATA_PREPARED_NO_GPU_EXECUTION',source=bind(Path(__file__)),arms=list(ARMS),
   bindings=[bind(original_path),bind(real_path),bind(R/'scripts/ust_template_crossover_prepare_20261106.py'),bind(R/'scripts/ust_dev_semantic_transforms_20261106.py')],
   outputs=[bind(O/x/'data-protocol.json') for x in ARMS]+[bind(O/'qualification.local.json')],
   intended_estimands=['A reproducibility of original learned collapse','B static template training intervention','C mixed-ability interference/schedule diagnostic with unequal total updates'],
   inference_and_training_runner='Not started. Root separately freezes model runtime, gate status, checkpoint schedule, evaluation inputs, caps and cumulative spend guard.'))
 print(json.dumps(dict(status='PASS',arms=ARMS,steps=[200,200,68],labels=counts,output=str(O))))
if __name__=='__main__':main()
