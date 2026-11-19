#!/usr/bin/env python3
"""Zero-model-API CityFM public-data acceptance and frozen payload preparation."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from privacy_synth.location_classification import (CLASSES,ROLES,digest,recompute_labels,semantic_categories,role_allocation,balanced_members,prompt,score,baseline_features)
REV='1bf4985c7e86665bb38818cd0e443914bfbb7e46'
INSPECTED={'32395524','32414926'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def load(path):return json.loads(path.read_text())
def physical_geometry(record,transformer):
 from shapely.geometry import Polygon
 from shapely import transform
 pts=record.get('points',[])
 # Official CityFM points are [latitude, longitude], not GeoJSON order.
 if len(pts)<3:raise ValueError('too_few_points')
 if any(not (isinstance(p,(list,tuple)) and len(p)==2 and 0<=float(p[0])<=3 and 102<=float(p[1])<=105) for p in pts):raise ValueError('coordinate_range')
 poly=transform(Polygon([(float(p[1]),float(p[0])) for p in pts]),transformer.transform,interleaved=False)
 if not poly.is_valid or poly.is_empty or poly.area<=0:raise ValueError('invalid_geometry')
 return poly

def build(raw,out):
 from shapely.geometry import Point,box
 from shapely.strtree import STRtree
 from pyproj import Transformer
 if out.exists():raise RuntimeError('Immutable preparation directory already exists; choose a fresh explicitly named output')
 files=('land_use.json','land_use_data.csv','buildings_untagged.json','buildings_context.json','entities.json')
 provenance={f:{'path':str(raw/f),'bytes':(raw/f).stat().st_size,'sha256':sha(raw/f),'url':f'https://raw.githubusercontent.com/PasqualeTurin/CityFM/{REV}/Singapore/Data/{f}'} for f in files}
 labels=load(raw/'land_use.json');targets=load(raw/'buildings_untagged.json');contexts=load(raw/'buildings_context.json');entities=load(raw/'entities.json')
 with (raw/'land_use_data.csv').open(newline='',encoding='utf-8-sig') as f:csvrows=list(csv.DictReader(f))
 computed,best=recompute_labels(csvrows)
 mismatch=[{'id':k,'released':labels.get(k),'recomputed':computed.get(k)} for k in sorted(set(labels)|set(computed)) if labels.get(k)!=computed.get(k)]
 audit={'revision':REV,'source':provenance,'planning_proxy_not_actual_activity':True,'raw_counts':{'labels':len(labels),'targets':len(targets),'contexts':len(contexts),'entities':len(entities),'csv_rows':len(csvrows)},'label_mismatch_count':len(mismatch),'label_mismatches':mismatch,'label_counts':dict(Counter(labels.values())),'native_context_radius_m':100,'grid_m':2000,'boundary_buffer_m':100,'metric_crs':'EPSG:3414','seed':20261022,'inspected_dev_audit_only':sorted(INSPECTED)}
 transformer=Transformer.from_crs('EPSG:4326','EPSG:3414',always_xy=True)
 geoms={};excluded={};all_geographic_cells=set();source_area_ratios=[];tagkeys=Counter();leaky_target=Counter()
 for k,t in targets.items():
  if k not in computed:excluded[k]='no_retained_label';continue
  if computed[k] not in CLASSES:excluded[k]='unexpected_label';continue
  if k in INSPECTED:excluded[k]='previously_inspected_audit_only';continue
  try:g=physical_geometry(t,transformer)
  except (ValueError,TypeError,KeyError) as e:excluded[k]='geometry:'+str(e);continue
  if not math.isfinite(float(t.get('area',0))) or float(t.get('area',0))<=0:excluded[k]='invalid_source_area';continue
  source_area_ratios.append(float(t['area'])/g.area)
  tagkeys.update(t.get('tags',{}).keys())
  for key,val in t.get('tags',{}).items():
   if key in ('name','amenity','shop','landuse','office','building:use','wikidata','wikipedia') or (key=='building' and val not in ('yes','no')):leaky_target[key]+=1
  x,y=transformer.transform(float(t['lon']),float(t['lat']));cell=(math.floor(x/2000),math.floor(y/2000))
  all_geographic_cells.add(cell)
  minx,miny,maxx,maxy=g.bounds
  if not (minx>=cell[0]*2000+100 and maxx<=(cell[0]+1)*2000-100 and miny>=cell[1]*2000+100 and maxy<=(cell[1]+1)*2000-100):excluded[k]='grid_boundary_buffer';continue
  geoms[k]=(g,x,y,cell)
 # Polygon associations identify overlapping buildings and campus/compound entities.
 entity_geoms=[];entity_poly_ids=[];invalid_entity_geoms=0
 for k,e in entities.items():
  if e.get('type')=='polygon':
   try:g=physical_geometry(e,transformer)
   except (ValueError,TypeError,KeyError):invalid_entity_geoms+=1;continue
   entity_geoms.append(g);entity_poly_ids.append(k)
 entity_tree=STRtree(entity_geoms)
 keys=list(geoms);target_tree=STRtree([geoms[k][0] for k in keys]);associations=defaultdict(set);overlap_edges=[]
 for i,k in enumerate(keys):
  g=geoms[k][0]
  for j in target_tree.query(g,predicate='intersects'):
   j=int(j)
   if j<=i:continue
   if g.intersection(geoms[keys[j]][0]).area>0:
    associations['target_overlap:'+min(k,keys[j])+':'+max(k,keys[j])].update((k,keys[j]));overlap_edges.append([k,keys[j]])
  for j in entity_tree.query(g,predicate='intersects'):
   j=int(j);eg=entity_geoms[j]
   if eg.intersection(g).area>0:associations['polygon:'+entity_poly_ids[j]].add(k)
 # Group allocation stays geographic; associations spanning cells are conservatively excluded.
 crosscell_assoc={a:sorted(ids) for a,ids in associations.items() if len({geoms[k][3] for k in ids})>1}
 for ids in crosscell_assoc.values():
  for k in ids:excluded[k]='association_spans_grid_cells'
 cells=role_allocation(all_geographic_cells);rows=[];context_audit=Counter();context_ids_by_target={};overlap_contexts={};area_quantiles={}
 if source_area_ratios:
  vals=sorted(source_area_ratios);area_quantiles={str(q):vals[min(len(vals)-1,int(q*(len(vals)-1)))] for q in (0,.01,.5,.99,1)}
 for k,(g,x,y,cell) in geoms.items():
  if k in excluded:continue
  listed=contexts.get(k)
  if not isinstance(listed,list):excluded[k]='missing_context_list';continue
  neighbors=[];used=set();overlapping=[];bad=False
  for nid0 in listed:
   nid=str(nid0)
   if nid in used:context_audit['duplicate_neighbor_ids']+=1;continue
   used.add(nid)
   if nid==k:context_audit['self_context']+=1;continue
   if nid not in entities:context_audit['unresolved_neighbor']+=1;bad=True;continue
   e=entities[nid]
   try:
    nx,ny=transformer.transform(float(e['lon']),float(e['lat']));distance=math.hypot(nx-x,ny-y)
    if not all(math.isfinite(v) for v in (nx,ny,distance)):raise ValueError()
   except (ValueError,TypeError,KeyError):context_audit['invalid_neighbor_coordinate']+=1;bad=True;continue
   # Native source integer-rounded distance may include points slightly above100m.
   if distance>101:context_audit['outside_native_radius']+=1;bad=True;continue
   if distance>100:context_audit['100_to_101m_excluded']+=1;continue
   # The target and overlapping/enclosing same-site polygons cannot supply an answer tag.
   if e.get('type')=='polygon':
    try:
     eg=physical_geometry(e,transformer)
     if eg.intersection(g).area>0:overlapping.append(nid);context_audit['overlapping_semantic_context_removed']+=1;continue
    except (ValueError,TypeError,KeyError):context_audit['invalid_neighbor_geometry']+=1;bad=True;continue
   if g.covers(Point(nx,ny)):
    overlapping.append(nid);context_audit['target_interior_context_removed']+=1;continue
   cats=semantic_categories(e.get('tags',{}));dx=nx-x;dy=ny-y
   angle=(math.degrees(math.atan2(dx,dy))+360)%360
   neighbors.append({'_id':nid,'categories':cats,'dx_m':round(dx,1),'dy_m':round(dy,1),'distance_m':round(distance,1),'bearing_deg':round(angle,1)})
  if bad:excluded[k]='unresolved_or_invalid_context';continue
  neighbors.sort(key=lambda n:(n['distance_m'],n['_id']))
  counts=Counter(c for n in neighbors for c in n['categories'])
  # Source physical attributes are recomputed in projected meters; no IDs, names, use, or absolute location.
  geo={'area_m2':round(g.area,1),'perimeter_m':round(g.length,1),'compactness':round(4*math.pi*g.area/g.length**2,4),'bbox_width_m':round(g.bounds[2]-g.bounds[0],1),'bbox_height_m':round(g.bounds[3]-g.bounds[1],1),'vertex_count':len(g.exterior.coords)-1}
  rel=[{key:value for key,value in n.items() if key!='_id'} for n in neighbors[:24]]
  payload={'target':geo,'context_radius_m':100,'total_neighbors':len(neighbors),'all_neighbor_category_counts':dict(sorted(counts.items())),'nearest_neighbors':rel}
  context_ids_by_target[k]={n['_id'] for n in neighbors};overlap_contexts[k]=overlapping
  rows.append({'id':k,'label':computed[k],'role':cells[cell],'cell':cell,'payload':payload})
 # Entity exposure includes target identity, all visible-count neighbors, and removed same-site associations.
 exposures=defaultdict(set)
 for r in rows:
  for eid in {r['id']}|context_ids_by_target[r['id']]|set(overlap_contexts[r['id']]):exposures[eid].add(r['role'])
 conflicts={k:sorted(v) for k,v in exposures.items() if len(v)>1}
 for r in rows:
  if ({r['id']}|context_ids_by_target[r['id']]|set(overlap_contexts[r['id']]))&conflicts.keys():excluded[r['id']]='shared_target_context_cross_role'
 row_roles={r['id']:r['role'] for r in rows}
 assoc_roles={a:{row_roles[k] for k in ids if k in row_roles} for a,ids in associations.items() if len(ids)>1}
 crossrole_assoc={a:sorted(v) for a,v in assoc_roles.items() if len(v)>1}
 for a in crossrole_assoc:
  for k in associations[a]:excluded[k]='overlap_campus_association_cross_role'
 rows=[r for r in rows if r['id'] not in excluded]
 dev,dev_counts,dev_ok=balanced_members(rows,ROLES[1],4);screen,screen_counts,screen_ok=balanced_members(rows,ROLES[2],12)
 selected={ROLES[0]:[r['id'] for r in rows if r['role']==ROLES[0]],ROLES[1]:dev,ROLES[2]:screen,ROLES[3]:[r['id'] for r in rows if r['role']==ROLES[3]]}
 retained_exposures=defaultdict(set)
 for r in rows:
  for eid in {r['id']}|context_ids_by_target[r['id']]|set(overlap_contexts[r['id']]):retained_exposures[eid].add(r['role'])
 assert not any(len(v)>1 for v in retained_exposures.values())
 assert not any(k in selected[ROLES[3]] for k in INSPECTED)
 audit.update({'source_area_over_metric_area_quantiles':area_quantiles,'target_tag_keys':dict(tagkeys),'target_direct_leak_fields_count':dict(leaky_target),'leak_policy':'Target tags/names/IDs/absolute coordinates are never serialized; same-footprint/interior semantic neighbors removed; all semantic values fixed-whitelisted.', 'entity_invalid_polygon_count':invalid_entity_geoms,'context_audit':dict(context_audit),'exclusion_counts':dict(Counter(excluded.values())),'eligible_natural_role_class_counts':{role:dict(Counter(r['label'] for r in rows if r['role']==role)) for role in ROLES},'dev_available_per_class':dev_counts,'screen_available_per_class':screen_counts,'dev32_complete':dev_ok,'screen96_complete':screen_ok,'overlap_edges_count':len(overlap_edges),'crosscell_association_count':len(crosscell_assoc),'pre_exclusion_shared_entity_conflicts':len(conflicts),'pre_exclusion_crossrole_associations':len(crossrole_assoc),'post_exclusion_shared_entity_conflicts':0,'group_allocation':dict(Counter(cells.values())),'selected_role_counts':{k:len(v) for k,v in selected.items()},'status':'STOP_LABEL_MISMATCH' if mismatch else 'STOP_BALANCED_QUOTA' if not(dev_ok and screen_ok) else 'PREPARED_PENDING_MODEL_PACKAGE_APPROVAL'})
 if mismatch:audit['status']='STOP_LABEL_MISMATCH'
 out.mkdir(parents=True)
 write(out/'audit.json',audit);write(out/'exclusions.json',excluded);write(out/'overlap-associations.json',{'target_overlap_edges':overlap_edges,'crosscell_associations':crosscell_assoc,'crossrole_associations':crossrole_assoc,'removed_same_site_contexts':overlap_contexts,'pre_exclusion_shared_entity_conflicts':conflicts})
 memberships=[{'id':r['id'],'role':r['role'],'grid':list(r['cell']),'selected':r['id'] in selected[r['role']],'payload_sha256':digest(r['payload']),'all_context_entity_ids':sorted(context_ids_by_target[r['id']])} for r in rows]
 jsonl(out/'memberships.jsonl',memberships)
 write(out/'membership-freeze.json',{'seed':20261022,'roles':{role:{'ids':ids,'sha256':digest(ids)} for role,ids in selected.items()},'membership_file_sha256':sha(out/'memberships.jsonl'),'provenance_sha256':digest(provenance),'label_source':'maxPERCENTAGE + official dismiss/rename; planning proxy','no_rerandomization':True})
 byid={r['id']:r for r in rows}
 for role,ids in selected.items():
  folder=out/role;folder.mkdir()
  jsonl(folder/'inputs.jsonl',[{'id':k,'payload':byid[k]['payload']} for k in ids])
  jsonl(folder/'gold.local.jsonl',[{'id':k,'class':byid[k]['label']} for k in ids])
  if role!=ROLES[3]:jsonl(folder/'requests.skeleton0.jsonl',[{'id':k,**prompt(byid[k]['payload'])} for k in ids])
 # No sealed payload tokenization, inference, scoring or training.
 baseline={'status':'not_run'}
 if not mismatch:
  from sklearn.feature_extraction import DictVectorizer
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  fit=[r for r in rows if r['role']==ROLES[0]]
  if len({r['label'] for r in fit})==len(CLASSES):
   majority=sorted(Counter(r['label'] for r in fit).items(),key=lambda p:(-p[1],p[0]))[0][0]
   model=make_pipeline(DictVectorizer(sparse=False),StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=20261022))
   model.fit([baseline_features(r['payload']) for r in fit],[r['label'] for r in fit])
   baseline={'status':'fit_only_fixed_logistic','fit_n':len(fit),'majority_label':majority,'settings':{'C':1,'max_iter':500,'hypersearch':False,'features':'geometry + whole-neighborhood type counts + nearest24 distance bins; fit-only vocabulary/scaler'}}
   for name,rr in [('natural_screen',[r for r in rows if r['role']==ROLES[2]]),('balanced_screen96',[byid[k] for k in screen] if screen_ok else [])]:
    if rr:
     y=[r['label'] for r in rr];pred=model.predict([baseline_features(r['payload']) for r in rr])
     baseline[name]={'majority':score(y,[json.dumps({'class':majority}) for _ in rr]),'logistic':score(y,[json.dumps({'class':x}) for x in pred])}
 write(out/'baseline.local.json',baseline)
 # Three skeletons are frozen candidates only; no prompts chosen with screen/final results.
 write(out/'prompt-skeletons.json',{'variants':[prompt({},i)['system'] for i in range(3)],'max_adaptation_rounds':3,'dev_stage_order':'one sample per class smoke8, then remaining24 of fixedDev32; candidate screen96 once only after prompt/input freeze','output_max_tokens':64,'input_max_tokens':8192,'gate':{'correct_min':63,'n':96,'accuracy_min':.65,'majority_margin':.10,'logistic_margin':.05,'macro_f1_min':.60,'valid_rate_min':.95,'no_leak_required':True}})
 tokenizer_path=ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json'
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(tokenizer_path));requests=[]
 # Candidates over8192 cause a STOP rather than truncation or task replacement.
 for stage,ids in [('dev32',dev),('screen96',screen if screen_ok else [])]:
  for k in ids:
   for variant in range(3) if stage=='dev32' else [0]:
    req=prompt(byid[k]['payload'],variant)
    rendered='<｜begin▁of▁sentence｜>'+req['system']+'<｜User｜>'+req['user']+'<｜Assistant｜>'
    requests.append({'id':k,'stage':stage,'variant':variant,'input_tokens':len(tok.encode(rendered,add_special_tokens=False).ids),'output_cap':64,'payload_sha256':digest(byid[k]['payload'])})
 over=[r for r in requests if r['input_tokens']>8192]
 package={'status':'BLOCKED_BALANCED_QUOTA' if not(dev_ok and screen_ok) else 'BLOCKED_LABEL_MISMATCH' if mismatch else 'BLOCKED_OVER8192' if over else 'AWAITING_EXPLICIT_MODEL_BUDGET_CONFIRMATION','provider':'DeepSeek official (proposed only)','model':'deepseek-flash thinking-off (proposed only)','request_rows':requests,'tokenizer_path':str(tokenizer_path),'tokenizer_sha256':sha(tokenizer_path),'token_estimation':'Exact cached local tokenizer on declared DS wrapper; actual API accounting may differ. No request sent.','max_calls_no_retries':len(requests),'input_tokens_no_retries':sum(r['input_tokens'] for r in requests),'output_tokens_max_no_retries':64*len(requests),'over8192':over,'data_transmitted_if_approved':'Public proxy only: metric target geometry summaries; sanitized surrounding categories/counts and nearest24 relative geometry. No target tags/names/IDs/addresses/use/wiki/absolute coords or gold.','cheaper_local_alternative':'Fixed fit-only majority/logistic baselines completed when fit supports all classes; no external model needed for schema/scoring.','sealed_evaluation':'Not authorized, not opened for inference/scoring/training','cost_usd':'Requires current provider pricing confirmation; no invented price estimate.','approval':None}
 write(out/'external-workpackage.proposed.json',package)
 write(out/'artifact-manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
 return audit,baseline,package

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=ROOT/'data/raw/cityfm-singapore'/REV);p.add_argument('--output',type=Path,default=ROOT/'runs/2026-10-22-night-dual/preparation/cityfm');args=p.parse_args()
 a,b,w=build(args.raw,args.output);print(json.dumps({'status':a['status'],'quota_dev':a['dev32_complete'],'quota_screen':a['screen96_complete'],'output':str(args.output),'workpackage_status':w['status']}))
if __name__=='__main__':main()


