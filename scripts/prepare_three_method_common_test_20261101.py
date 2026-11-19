"""Freeze the final SAME local displayed-fact Test only after all trained endpoints.
523 sealed rooms x three existing families =1569 questions, not523 questions.
No room-label gold, external requests, training, checkpoint choice or filtering.
"""
from pathlib import Path
import argparse,hashlib,json,math,re,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from privacy_synth.spatial_native_qa import FAMILIES,build_question,independent_answer
from night_diffusion_spatial_ipd_cpu_20261031 import array_records
import three_method_native_ipd_20261101 as e
O=e.O;P=O/'common-test-v1';SOURCE=R/'runs/2026-10-22-matterport-acceptance/geometry/native-inputs.local.json'
SOURCE_SHA='0f085f93d50b2dff601d07e0aaee465c2e1c7ff1b19a7867a081ea91c38d5185';SEED=2026110100

def native_row(row,fit_vocab):
 expected={'record_id','building_graph_index','room_node_index','source_split','room_size_native','objects'}
 assert set(row)==expected and row['source_split']=='test' and row['objects']
 assert type(row['building_graph_index']) is int and type(row['room_node_index']) is int
 assert row['record_id']==f"mpcat40-graph-{row['building_graph_index']:02d}-room-{row['room_node_index']:04d}"
 def vec(v):return type(v) is list and len(v)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in v)
 assert vec(row['room_size_native'])
 for obj in row['objects']:
  assert set(obj)=={'category','relative_position_native'} and type(obj['category']) is str and obj['category'] and not re.search(r'\s',obj['category']) and vec(obj['relative_position_native'])
 # A new category is preserved as its literal input name; no unseen-to-known coercion.
 # The extension is local to this scene, not a fitted aggregate Test vocabulary.
 vocab=list(fit_vocab)+sorted({o['category'] for o in row['objects']}-set(fit_vocab))
 transformed=dict(record_id=row['record_id'],building_index=row['building_graph_index'],count=len(row['objects']),
  room_size_native=row['room_size_native'],objects=[dict(category_id=vocab.index(x['category']),position_native=x['relative_position_native']) for x in row['objects']])
 return transformed,vocab

def scene_questions(row,fit_vocab,ordinal):
 translated,vocab=native_row(row,fit_vocab);questions=[];gold=[]
 for j,family in enumerate(FAMILIES):
  built=build_question(translated,vocab,family,SEED+3*ordinal+j)
  assert independent_answer(built['prompt'],family,built['anchor'])==built['answer']
  identifier=f'test-{ordinal:04d}-{family}'
  messages=[dict(role='system',content=built['system']),dict(role='user',content=built['prompt'])]
  body={k:built[k] for k in ('system','prompt','family','object_count','anchor','options','singleton','encoding')}
  questions.append(dict(id=identifier,ordinal=3*ordinal+j,room_ordinal=ordinal,record_id=row['record_id'],building_index=row['building_graph_index'],messages=messages,**body))
  gold.append(dict(id=identifier,answer=built['answer'],answer_category=built['answer_category']))
 return questions,gold

def actual_endpoints_ready():
 a=e.load(O/'student-controller-v3/terminal.json');b=e.load(O/'latest-ipd-controller-v1/terminal.json')
 assert a['status']=='ALL18_W_OFF_ON_UNITS_FIT_RELOAD_COMPLETE_IPD_AND_COMMON_TEST_STILL_PENDING'
 assert b['status']=='ALL6_LATEST_NATIVE_QWEN_IPD50_ACTUAL_UPDATES_AND_INDEPENDENT_RELOAD_COMPLETE'
 assert len(a['units'])==18 and len(b['results'])==6
 endpoints=[]
 for gen in e.m.GENS:
  for seed in e.m.SEEDS:
   for stage in ('warm','off','on','ipd'):
    out=e.m.unit(gen,seed,stage);t=e.load(out/'fit-terminal.json');reload=e.load(out/'independent-reload.json')
    assert reload['status']=='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS'
    assert reload['actual_updates']==t['actual_updates'] and reload['step']==(288 if stage=='warm' else 50)
    if stage=='ipd':assert reload['actual_updates']>0
    manifest=Path(t['checkpoint'])/'manifest.json';e.m.lib.verify_checkpoint(out,manifest.parent)
    endpoints.append(dict(generator=gen,seed=seed,stage=stage,unit=str(out.relative_to(R)),manifest=e.bind(manifest),reload=e.bind(out/'independent-reload.json')))
 qual=e.load(e.P/'probability-qualification-v1/terminal.json')
 assert qual['status']=='ACTUAL_LATEST_QWEN_NATIVE_PROBABILITY_CODEC_CACHE_AND_COMBINED_MEMORY_PASS' and qual['artifact']['stat']==e.file_stat(e.MODEL)
 return endpoints

def prepare():
 assert not (P/'proposal.json').exists();proof=e.load(P/'public-CPU-proof.json')
 assert proof['source']==e.bind(Path(__file__)) and proof['Test_feature_payload_decoded'] is False
 pins=[Path(__file__),R/'src/privacy_synth/spatial_native_qa.py',R/'src/privacy_synth/choice_answer.py',e.m.PLAN,
  R/'scripts/night_diffusion_spatial_ipd_cpu_20261031.py',R/'runs/2026-10-31-spatial-joint-composition-inner-v1/normalization.local.json',P/'public-CPU-proof.json']
 e.once(P/'proposal.json',dict(approved_CPU_freeze_by_Root=True,source=e.bind(Path(__file__)),bindings=[e.bind(x) for x in pins],source_payload_SHA256=SOURCE_SHA,
  activation='Only after all24 actual W/off/on/IPD fit+independent-reload and exact real teacher qualification; no future manifest hashes',
  sealed_rooms=523,families=list(FAMILIES),expected_questions=1569,seed=SEED,order='source record_id ascending; fixed family order',
  preserve_all_objects_and_rows=True,no_refill_or_accuracy_selection=True,no_original_room_label_gold=True,
  out_of_fit_categories='literal names preserved in per-scene codec extension; no global Test statistics or model fit',
  common_context_cap=16384,maximum_new_tokens=32,too_long='retain every question and count context failure in each applicable endpoint; never truncate or remove',
  question_gold_in_separate_local_file=True,all_raw_then_once_score=True,API_calls=0,GPU_calls=0,Test_feature_payload_decoded=False))
 print('COMMON_TEST1569_CPU_PROPOSAL_FROZEN_TEST_NOT_OPENED')

def freeze():
 p=e.load(P/'proposal.json');assert p['source']==e.bind(Path(__file__))
 for b in p['bindings']:assert e.bind(R/b['path'])==b,b['path']
 endpoints=actual_endpoints_ready();assert not (P/'started.json').exists(),'No sealed Test rebuild or partial replay'
 assert e.sha(SOURCE)==SOURCE_SHA
 e.once(P/'started.json',dict(epoch=time.time(),source=e.bind(Path(__file__)),actual_trained_endpoints=endpoints))
 vocab=e.load(R/'runs/2026-10-31-spatial-joint-composition-inner-v1/normalization.local.json')['category_vocab'];rows=[]
 # Lexically reject all other splits before decoding feature payloads.
 for record in array_records(SOURCE.read_text()):
  split=re.findall(r'"source_split"\s*:\s*"([^"\\]*)"',record);assert len(split)==1
  if split[0]=='test':rows.append(json.loads(record))
 rows.sort(key=lambda row:row['record_id']);assert len(rows)==len({x['record_id'] for x in rows})==523
 fit=e.load(R/'runs/2026-10-31-spatial-set-diffusion-v1/fit.local.json');dev=e.load(R/'runs/2026-10-31-spatial-set-diffusion-v1/dev.local.json')
 assert not ({x['building_graph_index'] for x in rows}&({x['building_index'] for x in fit}|{x['building_index'] for x in dev}))
 inputs=[];golds=[]
 for i,row in enumerate(rows):
  e.check_stop();qs,gs=scene_questions(row,vocab,i);inputs+=qs;golds+=gs
 assert len(inputs)==len(golds)==1569 and [x['id'] for x in inputs]==[x['id'] for x in golds]
 for filename,values in [('queries1569.local.jsonl',inputs),('gold1569.local.jsonl',golds)]:
  with (P/filename).open('x') as f:
   for value in values:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
   f.flush();__import__('os').fsync(f.fileno())
 e.once(P/'terminal.json',dict(status='SAME1569_LOCAL_QUERIES_AND_SEPARATE_FACT_GOLD_FROZEN_NO_DECODE_OR_SCORE',rooms=523,questions=1569,
  queries=e.bind(P/'queries1569.local.jsonl'),gold=e.bind(P/'gold1569.local.jsonl'),actual_trained_endpoints=endpoints,Test_used_for_training=False,
  model_decodes=0,score_runs=0,GPU_calls=0,API_calls=0,no_original_room_label_gold=True,epoch=time.time()))
 print('ALL1569_COMMON_LOCAL_TEST_QUERIES_FROZEN_NO_MODEL_SCORE',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','freeze']);v=a.parse_args()
 (prepare if v.action=='prepare' else freeze)()