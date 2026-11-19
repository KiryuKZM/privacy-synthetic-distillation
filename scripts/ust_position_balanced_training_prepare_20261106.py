"""Train-only deterministic position construction; no model or Dev selection."""
from pathlib import Path
import collections,copy,hashlib,itertools,json,re,sys
from decimal import Decimal
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import ust_dev_semantic_transforms_20261106 as t
from privacy_synth.ust_task_contract import certify_native_ranking_reference
O=R/'runs/2026-11-06-ust-autonomous-repair-v1/position-training'
P=R/'runs/2026-11-05-ust-real-synthetic-paired-ce-v1'
ARMS=('real-identity200','real-balanced200','synthetic-balanced200')
SEED='ust-position-construction-20261106-v1'
LP=list(itertools.permutations(range(1,5)));OP=list(itertools.permutations('ABCD'))
GOALS={
 'enhancing overall station coverage.':'coverage',
 'minimizing overall travel time.':'travel_time',
 'reducing overall waiting time.':'waiting_time',
 'reducing overall charging time.':'charging_time'}
PLAN_PREFIX='Evaluate each candidate location based on the provided data. Identify which location would deliver the most substantial benefits for an EV charging station, focusing on '
UNDER={
 'Based on the data provided, rank the locations by average distance to nearby charging stations, from the farthest to the closest.':'distance',
 'Based on the data provided, rank the locations by average waiting time at near charging stations, from the longest to the shortest.':'waiting_time',
 'Based on the data provided, rank the locations by average charging time at near charging stations, from the longest to the shortest.':'charging_time'}

def load(p):return json.loads(p.read_text())
def bind(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def hash_key(*xs):return hashlib.sha256('\x1f'.join([SEED,*map(str,xs)]).encode()).hexdigest()
def target(row):return re.fullmatch(r'<Answer>([ABCD])</Answer>',row['target']).group(1) if row['metadata']['ability']=='st_understanding' else json.loads(row['target'])['answer']
def stratum(row):
 s=t.sections(row['question']);prefix,_,_=t._options(s['Question']);a=row['metadata']['ability']
 if a=='st_understanding':assert prefix in UNDER;return a+'/'+UNDER[prefix]
 assert a in ('planning','reflection') and prefix.startswith(PLAN_PREFIX)
 goal=prefix.removeprefix(PLAN_PREFIX);assert goal in GOALS;return a+'/'+GOALS[goal]

def understood_references(row):
 """Fail closed on known unsupported reference forms, not only roundtrip text."""
 s=t.sections(row['question']);a=row['metadata']['ability'];stratum(row)
 allowed={'Task Description','Data Schema','Data','Analysis','Question','Decision','Reason','Environment Changes','Note'}
 assert set(s)<=allowed,'Unsupported section'
 if a=='reflection':assert set(s)==allowed
 else:assert not {'Decision','Reason','Environment Changes'}&set(s)
 # Static descriptions describe fields/output grammar, never numbered candidates.
 for k in ['Task Description','Data Schema','Note']:
  assert not re.search(r'\b(?:locations?|sites?|candidates?)\s*[#:]?\s*\d',s[k],re.I),'Unsupported static numbered reference'
 blocks=t._data_blocks(s['Data']);assert list(blocks)==[1,2,3,4]
 for body in blocks.values():
  lines=body.splitlines();assert len(lines)==5
  for k,line in zip(['coverage','demand','distance','waiting_time','charging_time'],lines):
   m=re.fullmatch('- '+k+r': (\S+)',line);assert m
   val=re.sub('[mh]$','',m.group(1));assert val=='NaN' or Decimal(val).is_finite()
 prefix,opts,suffix=t._options(s['Question']);assert not suffix
 if a!='st_understanding':assert all(re.fullmatch(r'location [1-4]',v,re.I) for v in opts.values())
 analysis=s.get('Analysis');analysis_refs=[]
 if analysis and analysis!='N/A':
  lines=analysis.split('\n\n');assert len(lines)==5
  for title,chunk in zip(['Coverage','Demand','Distance','Waiting time','Charging time'],lines):
   lead,body=chunk.split('\n',1);assert lead=='- '+title+' ranking:'
   pieces=body.split(' > ');assert len(pieces)==4
   locs=[]
   for piece in pieces:
    m=re.fullmatch(r'(location\s+)([1-4]) \(([^()]+)\)',piece,re.I);assert m,'Unsupported analysis reference'
    val=re.sub('[mh]$','',m.group(3));assert val=='NaN' or Decimal(val).is_finite()
    locs.append(int(m.group(2)));analysis_refs.append([title,int(m.group(2)),m.group(3)])
   assert set(locs)=={1,2,3,4}
 if a=='reflection':
  assert s['Decision'] in 'ABCD' and len(s['Decision'])==1
  # Remove only exactly supported tokens. Any residual number or uppercase
  # answer letter/ordinal referent is rejected, not silently left unmapped.
  reason=s['Reason'];res=t.LOC.sub('LOCATION_REF',reason)
  res=t.LABEL_REF.sub(lambda m:'LETTER_REF' if m.group(2) in 'ABCD' else m.group(0),res)
  assert not re.search(r'\d|\b[A-D]\b',res),'Unsupported bare reason reference'
  assert not re.search(r'\b(?:first|second|third|fourth)\s+(?:location|site|option|candidate)\b',res,re.I),'Unsupported ordinal reason reference'
  lines=s['Environment Changes'].splitlines();assert len(lines)==4
  for name,line in zip(['Station coverage','Charging time','Travel time','Waiting time'],lines):
   assert re.fullmatch(re.escape(name)+r' (?:increased|decreased) by [+-]?(?:\d+(?:\.\d*)?|\.\d+)%',line),'Unsupported environment feedback'
 return dict(status='PASS_SUPPORTED_TRAIN_REFERENCES',analysis_refs=analysis_refs,
   reason_location_refs=[int(m.group(2)) for m in t.LOC.finditer(s.get('Reason',''))],
   reason_letter_refs=[m.group(2) for m in t.LABEL_REF.finditer(s.get('Reason','')) if m.group(2) in 'ABCD'])

def mapped_target(row,new_letter):
 if row['metadata']['ability']=='st_understanding':return '<Answer>'+new_letter+'</Answer>'
 obj=json.loads(row['target']);before=dict(obj);obj['answer']=new_letter
 assert {k:v for k,v in obj.items() if k!='answer'}=={k:v for k,v in before.items() if k!='answer'}
 return json.dumps(obj,separators=(',',':'))

def location_of_answer(row):
 opts=t._options(t.sections(row['question'])['Question'])[1];body=opts[target(row)]
 if row['metadata']['ability']=='st_understanding':return int(re.findall('[1-4]',body)[0])
 return int(re.fullmatch(r'location ([1-4])',body,re.I).group(1))

def associations(rows):
 groups=collections.defaultdict(list)
 for row in rows:groups[stratum(row)].append(row)
 out={}
 for k,z in sorted(groups.items()):
  letters=collections.Counter(target(r) for r in z);locations=collections.Counter(location_of_answer(r) for r in z)
  joint=collections.Counter(target(r)+'/'+str(location_of_answer(r)) for r in z)
  out[k]=dict(n=len(z),correct_letter={x:letters[x] for x in 'ABCD'},correct_location_or_top_rank={str(x):locations[x] for x in range(1,5)},
   joint_letter_location=dict(sorted(joint.items())),joint_cells=len(joint),
   dominant_letter_fraction=max(letters.values())/len(z),dominant_location_fraction=max(locations.values())/len(z),
   location_semantics='top displayed rank member, not unique argmax when tied' if k.startswith('st_understanding') else 'selected physical candidate after renamed IDs')
 return out

def balanced(domain,rows):
 groups=collections.defaultdict(list)
 for row in rows:groups[stratum(row)].append(row)
 planned={}
 for k,z in groups.items():
  offset=int(hash_key(domain,'stratum-offset',k),16)%4
  ordered=sorted(z,key=lambda r:hash_key(domain,'stratum-order',r['id']))
  for i,r in enumerate(ordered):planned[r['id']]=dict(letter='ABCD'[(i+offset)%4],stratum=k,within_stratum_hash_rank=i,stratum_offset=offset)
 out=[];records=[];rejects=[]
 for i,row in enumerate(rows):
  try:
   refs=understood_references(row);design=planned[row['id']];old=target(row);desired=design['letter']
   loc=dict(zip(range(1,5),LP[int(hash_key(domain,'independent-location',row['id']),16)%24]))
   eligible=[op for op in OP if op['ABCD'.index(desired)]==old];assert len(eligible)==6
   order=eligible[int(hash_key(domain,'remaining-option-order',row['id']),16)%6]
   transformed=t.transform(row['question'],loc,order);mapping=transformed['old_to_new_letter'];assert mapping[old]==desired
   new=copy.deepcopy(row);new['question']=transformed['question'];new['target']=mapped_target(row,desired)
   assert {k:v for k,v in row.items() if k not in ('question','target')}=={k:v for k,v in new.items() if k not in ('question','target')}
   newrefs=understood_references(new)
   assert newrefs['analysis_refs']==[[title,loc[n],v] for title,n,v in refs['analysis_refs']]
   assert newrefs['reason_location_refs']==[loc[n] for n in refs['reason_location_refs']]
   assert newrefs['reason_letter_refs']==[mapping[k] for k in refs['reason_letter_refs']]
   inverse=t.transform(new['question'],{v:k for k,v in loc.items()},[mapping[x] for x in 'ABCD'])
   assert inverse['question']==t.canonical(row['question'])
   assert mapped_target(new,old)==row['target']
   before=t.sections(row['question']);after=t.sections(new['question'])
   assert collections.Counter(t._data_blocks(before['Data']).values())==collections.Counter(t._data_blocks(after['Data']).values())
   if row['metadata']['ability']=='reflection':assert before['Environment Changes']==after['Environment Changes']
   cert=None
   if row['metadata']['ability']=='st_understanding':
    oldcert=certify_native_ranking_reference(row['question'],old);cert=certify_native_ranking_reference(new['question'],desired)
    assert oldcert['trusted_gold'] and cert['trusted_gold']
    assert cert['compatible_answers']==[desired]
   records.append(dict(original_index=i,source_id=row['id'],stratum=design['stratum'],within_stratum_hash_rank=design['within_stratum_hash_rank'],stratum_offset=design['stratum_offset'],
     old_target=old,new_target=desired,old_correct_location_or_top_rank=location_of_answer(row),new_correct_location_or_top_rank=location_of_answer(new),
     location_map=loc,option_order=list(order),old_to_new_letter=mapping,checks=transformed['checks'],references=refs,
     inverse_canonical_equal=True,numeric_data_multiset_equal=True,all_reference_mapping_equal=True,feedback_unchanged=True,
     original_identity_canonical_changed_whitespace=row['question']!=t.canonical(row['question']),native_gate=cert,
     label_status='trusted_displayed_ranking' if cert else 'original_label_identity_preserved_optimality_uncertified'))
   out.append(new)
  except Exception as e:rejects.append(dict(original_index=i,source_id=row['id'],type=type(e).__name__,message=str(e)))
 if rejects:return None,records,rejects
 assert len(out)==200 and [r['id'] for r in out]==[r['id'] for r in rows]
 for k,x in associations(out).items():assert max(x['correct_letter'].values())-min(x['correct_letter'].values())<=1
 return out,records,[]

def main():
 O.mkdir(parents=True,exist_ok=False)
 sources={'real':P/'real-reference-ce/training200.local.json','synthetic':P/'synthetic-reference-ce/training200.local.json'}
 datasets={k:load(p) for k,p in sources.items()};results={};allrejects=[]
 for name,rows in datasets.items():
  assert len(rows)==200 and len({r['id'] for r in rows})==200
  out,maps,rejects=balanced(name,rows);results[name]=(out,maps);allrejects.extend(dict(domain=name,**x) for x in rejects)
 save(O/'rejected.local.json',allrejects)
 if allrejects:
  save(O/'blocked.json',dict(status='FAIL_CLOSED_UNSUPPORTED_TRAIN_INPUT',rejected=len(allrejects),no_rows_dropped_or_refilled=True));raise RuntimeError('Unsupported rows retained; no training data emitted')
 common=load(R/'runs/2026-11-06-ust-autonomous-repair-v1/causal-training/A-original200-replay/data-protocol.json')['common_training_spec']
 assoc={};paths=[]
 for arm in ARMS:
  domain='synthetic' if arm.startswith('synthetic') else 'real';identity=arm=='real-identity200';rows=datasets[domain] if identity else results[domain][0]
  d=O/arm;d.mkdir()
  if identity:
   with (d/'training.local.json').open('xb') as f:f.write(sources[domain].read_bytes())
  else:save(d/'training.local.json',rows);save(d/'transformation-records.local.json',results[domain][1])
  assoc[arm]=associations(rows)
  p=dict(status='CPU_DATA_FROZEN_NO_MODEL_EXECUTION',arm=arm,steps=200,training=bind(d/'training.local.json'),source_training=bind(sources[domain]),
    transformation_records=None if identity else bind(d/'transformation-records.local.json'),common_training_spec=common,
    ability_quotas=dict(collections.Counter(r['metadata']['ability'] for r in rows)),
    supervision='Existing published/gain reference labels; original label identity mapped. Planning/reflection optimality not newly certified. Understanding independently native-gated.',
    data_change='Exact original real file bytes' if identity else 'Whole-scene location/row and option mapping; target identity preserved. Shared transformer normalizes whitespace.',
    no_extra_records=True,no_Dev_read=True,no_GPU=True,no_API=True)
  save(d/'data-protocol.json',p);paths.append(bind(d/'data-protocol.json'))
 qualification=dict(status='PASS',balanced_rows=400,real_identity_rows=200,rejected=0,all_mapped_references_verified=True,
   all_inverse_checks_pass=True,independent_native_understanding_checks=136,all_understanding_trusted=True,
   all_Data_values_units_multisets_preserved=True,all_feedback_preserved=True,
   source_associations={k:associations(v) for k,v in datasets.items()},new_associations=assoc,
   canonical_whitespace_changes={k:sum(x['original_identity_canonical_changed_whitespace'] for x in v[1]) for k,v in results.items()},
   location_sampling='Independent hash over24permutations per sourceID; not forced marginal or joint balance. Residual associations recorded, no Dev selection.',
   letter_sampling='Within source+ability+requestedgoal/metric, sourceID hash order with fixedstratumoffset cyclicA/B/C/D; remaining3options independenthash among6validorders.',
   privacy='No external requests. Original source rows remain local.',no_training=True)
 save(O/'qualification.local.json',qualification)
 save(O/'data-protocol.json',dict(status='ALL_THREE_DATA_FROZEN',source=bind(Path(__file__)),arms=list(ARMS),seed=SEED,
   source_bindings=[bind(p) for p in sources.values()]+[bind(R/'scripts/ust_dev_semantic_transforms_20261106.py'),bind(R/'src/privacy_synth/ust_task_contract.py')],
   arm_protocols=paths,qualification=bind(O/'qualification.local.json'),rejected=bind(O/'rejected.local.json'),common_training_spec=common,
   causal_scope='Same sourcequestions/labels/order200exposures; changes candidate numbering and answer position construction plus canonical whitespace for real. Not new planning-gold certification.',
   no_Dev_golds_or_scores_used=True,no_GPU=True,no_API=True,no_training=True))
 print(json.dumps(dict(status='PASS',arms=ARMS,balanced_rows=400,understanding_native_verified=136,whitespace_changes=qualification['canonical_whitespace_changes'],
   max_dominant_location_by_arm={a:max(v['dominant_location_fraction'] for v in d.values()) for a,d in assoc.items()})))
if __name__=='__main__':main()
