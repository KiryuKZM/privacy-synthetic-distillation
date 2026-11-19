"""Native UST road simulator audit. No LLM, API, training or sealed QA access."""
from pathlib import Path
import os,sys,json,hashlib,copy,pickle,io,importlib,signal,time,traceback,random,math,argparse
R=Path(__file__).resolve().parents[1];O=R/'runs/2026-11-10-ust-original-env-audit-v1';S=O/'source'
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(S));os.chdir(S)
import numpy as np
import networkx as nx
from UST_tasks.road_planning.env_utils import my_graph as graphmod
from UST_tasks.road_planning.env_utils import road
from UST_tasks.road_planning.env_utils.utils import Config
from UST_tasks.road_planning.env import RoadPlanningEnvironment
ALLOWED={'builtins':{'dict','set','frozenset','list','tuple'},'numpy':{'ndarray','dtype'},'numpy.core.multiarray':{'_reconstruct','scalar'},'numpy._core.multiarray':{'_reconstruct','scalar'},'networkx.classes.graph':{'Graph'},'networkx.classes.reportviews':{'EdgeView','NodeView','DegreeView'},'networkx.classes.coreviews':{'AdjacencyView','AtlasView'}}
class Safe(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='my_graph' and name in {'MyGraph','MyNode','MyEdge','MyFace'}:return getattr(graphmod,name)
  if name in ALLOWED.get(module,set()):return getattr(importlib.import_module(module),name)
  raise ValueError(('unapproved_pickle_global',module,name))
def load_graph(slum):
 g=Safe(io.BytesIO((S/f'UST_tasks/road_planning/Data/{slum}.mg').read_bytes())).load()
 g.define_roads();g.define_interior_parcels();g.td_dict_init();g.feature_init();return g
# Sole source adapter: allowlisted old module mapping for file loading; same initializer calls.
road.load_graph=load_graph
MAPS=['CapeTown1','CapeTown2','Epworth_Demo','india'];GAMMA=.9
START=time.monotonic();TRANSITIONS=0

def primitive(v):
 if isinstance(v,dict):return {str(k):primitive(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [primitive(x) for x in v]
 if isinstance(v,np.generic):return primitive(v.item())
 if isinstance(v,float) and not math.isfinite(v):return str(v)
 return v
def dump(v):return json.dumps(primitive(v),sort_keys=True,ensure_ascii=False,indent=2)
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(dump(v)+'\n')
def status(**kw):
 p=O/'status.json';tmp=O/'status.tmp';tmp.write_text(dump(dict(pid=os.getpid(),epoch=time.time(),elapsed=time.monotonic()-START,transitions=TRANSITIONS,**kw)));tmp.replace(p)
def check():
 if (O/'STOP').exists():raise InterruptedError('own STOP requested')
def alarm(*_):raise TimeoutError('7200 second job cap reached; completed cases resumable')
def new(name):
 e=RoadPlanningEnvironment(Config('demo',name,1,False,str(O/'native-logs')));e.reset();return e
def legal(e):return [int(i) for i,v in enumerate(e._mg._get_edge_mask()) if v!=0] if not e._done else []
def fingerprint(e):
 g=e._mg
 return primitive(dict(stage=e._stage,done=e._done,history=e._action_history,built=[g.edge_list.index(x) for x in g.road_edges],interior=len(g.interior_parcels),avg=g.get_f2f_avgdist(),cache=g.f2f_avg,cost=g.total_road_cost))
def native_step(e,a):
 global TRANSITIONS
 check();assert not e._done and a in legal(e)
 before=e._mg.f2f_avg
 _,reward,done,info=e.step(a);TRANSITIONS+=1;g=e._mg
 # Recompute reward algebra without calling stateful reward functions again.
 travel=0. if len(g.interior_parcels) or g.del_parcel_num else (before-g.get_f2f_avgdist())/(before-g.f2f_avg_min)
 connect=(-1. if g.del_parcel_num==0 and len(g.interior_parcels) else g.del_parcel_num)/g.max_del_interior_parcels
 expected=connect+e.cfg.reward_specs.get('dis_weight',1.)*12*travel-e.cfg.reward_specs.get('cost_weight',1.)*.8*g.current_road_cost/g.max_road_cost
 return float(reward),bool(done),dict(info=info,algebra_reward=float(expected),algebra_abs_error=abs(float(reward-expected)))
def geometry(g):
 return dict(nodes=[[float(n.x),float(n.y)] for n in g.node_list],regions=[[g.node_list.index(n) for n in f.nodes] for f in g.inner_facelist],edges=[dict(id=i,nodes=[g.node_list.index(n) for n in e.nodes],weight=float(g.G[e.nodes[0]][e.nodes[1]]['weight']),built=e in g.road_edges) for i,e in enumerate(g.edge_list)])
def independent_distance(g):
 # All pairs shortest paths freshly recomputed from actual built edges, no native caches.
 a=geometry(g);G=nx.Graph();G.add_weighted_edges_from((e['nodes'][0],e['nodes'][1],e['weight']) for e in a['edges'] if e['built']);ds=dict(nx.all_pairs_dijkstra_path_length(G));pairs=[]
 for i,f in enumerate(a['regions']):
  for j,h in enumerate(a['regions']):
   if i==j:continue
   v=min([ds[x].get(y,1000.) for x in f if x in ds for y in h if y in ds] or [1000.]);native=float(g.td_dict_face[g.inner_facelist[i]][g.inner_facelist[j]])
   pairs.append((i,j,v,native))
 value=sum(x[2] for x in pairs)/len(pairs)
 return dict(recomputed=value,native=float(g.get_f2f_avgdist()),max_pair_error=max(abs(x[2]-x[3]) for x in pairs),mismatches=[x for x in pairs if abs(x[2]-x[3])>1e-8][:12])
def onestep(e):
 vals=[]
 for a in legal(e):
  c=copy.deepcopy(e);oldi=len(c._mg.interior_parcels);oldavg=c._mg.get_f2f_avgdist();r,d,chk=native_step(c,a)
  vals.append(dict(action=a,reward=r,connected=oldi-len(c._mg.interior_parcels),distance_reduction=float(oldavg-c._mg.get_f2f_avgdist()),done=d,algebra_abs_error=chk['algebra_abs_error']))
 return vals

def run_episode(name,policy):
 folder=O/'episodes'/name/policy;folder.mkdir(parents=True,exist_ok=True);e=new(name);roots=[];stage2seen=0
 for t in range(200):
  check()
  if e.env._done:break
  status(state='RUNNING',phase='native_episode',map=name,policy=policy,step=t)
  obs=e.get_observation();e.available_roads=[x['road_id'] for x in obs['available_roads']]
  if t==0 or (e.env._stage=='full_connected' and stage2seen in (0,3)):roots.append((t,copy.deepcopy(e)))
  if e.env._stage=='full_connected':stage2seen+=1
  p=folder/f'step-{t:03}.json';old=fingerprint(e.env)
  if p.exists():
   row=json.loads(p.read_text());assert row['before']==old;a=row['action']
   e.env_step(obs,f'road {a}');assert row['after']==fingerprint(e.env);continue
  vals=onestep(e.env)
  key='reward' if policy=='native_reward_greedy' else ('connected' if e.env._stage=='connecting' else 'distance_reduction')
  a=max(vals,key=lambda x:(x[key],-x['action']))['action'];shadow=copy.deepcopy(e.env);reward,done,chk=native_step(shadow,a)
  text=e.get_data_texts();unchanged=old==fingerprint(e.env)
  feedback=e.env_step(obs,f'road {a}');after=fingerprint(e.env)
  assert after==fingerprint(shadow),'wrapper and raw transition disagree'
  replay=copy.deepcopy(e.env);rereward,reinfo=replay.get_reward_info()
  region_ids={x['region_id'] for x in obs['connected_regions']+obs.get('unconnected_regions',[])}
  row=dict(step=t,action=a,policy=policy,objective=key,before=old,after=after,observation=obs,visible_text=text,task_info=json.loads((S/'UST_tasks/road_planning/Data/task_info.json').read_text()),full_geometry=geometry(e.env._mg),all_one_step=vals,reward=reward,reward_check=chk,feedback=feedback,independent_distance=independent_distance(e.env._mg),text_generation_preserved_state=unchanged,repeat_reward_without_action=float(rereward),reward_repeated_equal=abs(reward-float(rereward))<1e-9,all_regions_present=len(region_ids)==len(e.env._mg.inner_facelist),unprinted_config={'reward_specs':e.env.cfg.reward_specs,'total_road_steps':e.env._total_road_steps,'connecting_steps':e.env._connecting_steps,'full_connected_steps':e.env._full_connected_steps})
  save(p,row);print(name,policy,t,a,'reward',reward,'stage',after['stage'],flush=True)
 else:raise RuntimeError('episode step cap')
 if not (folder/'complete.json').exists():save(folder/'complete.json',dict(status='COMPLETE_NATIVE_EPISODE',steps=t,metrics=e.get_metrics(),final=fingerprint(e.env),policy=policy,LLM=False))
 return roots

def two_step(name,t,wrapper):
 out=O/'lookahead'/name/f'step-{t:03}';out.mkdir(parents=True,exist_ok=True);e=wrapper.env
 for a in legal(e):
  check();p=out/f'root-{a:03}.json'
  if p.exists():continue
  status(state='RUNNING',phase='exact_h2',map=name,step=t,root_action=a)
  c=copy.deepcopy(e);r,d,ch=native_step(c,a);rows=[]
  for b in legal(c):
   x=copy.deepcopy(c);rr,dd,ck=native_step(x,b);rows.append(dict(action=b,reward=rr,return2=r+GAMMA*rr,final=fingerprint(x),algebra_abs_error=ck['algebra_abs_error']))
  best=max(rows,key=lambda z:(z['return2'],-z['action'])) if rows else None
  save(p,dict(first=a,reward=r,second=rows,best=best,value=best['return2'] if best else r,complete_all_legal=True))
 rows=[json.loads((out/f'root-{a:03}.json').read_text()) for a in legal(e)]
 greedy=max(rows,key=lambda z:(z['reward'],-z['first']));best=max(rows,key=lambda z:(z['value'],-z['first']))
 trajectories=[]
 for label,x in [('greedy_first_best_second',greedy),('exact_h2',best)]:
  w=copy.deepcopy(wrapper);trace=[]
  for a in [x['first']]+([x['best']['action']] if x['best'] else []):
   obs=w.get_observation();w.available_roads=[z['road_id'] for z in obs['available_roads']];text=w.get_data_texts();shadow=copy.deepcopy(w.env);r,d,ch=native_step(shadow,a);fb=w.env_step(obs,f'road {a}');assert fingerprint(w.env)==fingerprint(shadow)
   trace.append(dict(action=a,visible_text=text,feedback=fb,reward=r,after=fingerprint(w.env),distance_check=independent_distance(w.env._mg)))
  trajectories.append(dict(policy=label,steps=trace))
 if not (out/'comparison.json').exists():save(out/'comparison.json',dict(status='EXACT_H2_NATIVE_REWARD',discount=GAMMA,greedy_first=greedy['first'],optimal_first=best['first'],strict_nonmyopic=best['value']>greedy['value']+1e-9,advantage=best['value']-greedy['value'],trajectories=trajectories,original_published_QA_gold_verified=False))
 # H5 is a sampled lower bound, never an exact reference. Three independent fixed seeds.
 for seed in (1101,1102,1103):
  p=out/f'h5-seed-{seed}.json'
  if p.exists():continue
  rng=random.Random(seed);sample=[]
  for a in legal(e):
   status(state='RUNNING',phase='sampled_h5',map=name,step=t,seed=seed,root_action=a)
   paths=[]
   for rep in range(8):
    c=copy.deepcopy(e);ret=0.;hist=[]
    for depth in range(5):
     actions=legal(c)
     if not actions:break
     action=a if depth==0 else rng.choice(actions);reward,done,ck=native_step(c,action);ret+=GAMMA**depth*reward;hist.append(dict(action=action,reward=reward,after=fingerprint(c)))
    paths.append(dict(return5=ret,steps=hist))
   sample.append(dict(first=a,best=max(paths,key=lambda z:z['return5']),paths=paths))
  save(p,dict(status='SAMPLED_H5_NOT_GOLD',seed=seed,rollouts_per_first=8,horizon=5,discount=GAMMA,candidates=sample,winner=max(sample,key=lambda z:(z['best']['return5'],-z['first']))['first']))
 print('LOOKAHEAD_COMPLETE',name,t,flush=True)

def summarize():
 episodes=[];steps=[];h2=[];h5=[]
 for p in sorted((O/'episodes').glob('*/*/complete.json')):episodes.append(dict(path=str(p.relative_to(O)),**json.loads(p.read_text())))
 for p in sorted((O/'episodes').glob('*/*/step-*.json')):steps.append(json.loads(p.read_text()))
 for p in sorted((O/'lookahead').glob('*/*/comparison.json')):
  v=json.loads(p.read_text());h2.append(dict(path=str(p.relative_to(O)),strict_nonmyopic=v['strict_nonmyopic'],advantage=v['advantage']))
 for d in sorted((O/'lookahead').glob('*/*')):
  s=[json.loads(p.read_text()) for p in d.glob('h5-seed-*.json')]
  if s:h5.append(dict(path=str(d.relative_to(O)),winners=[x['winner'] for x in s],all_three_complete=len(s)==3))
 report=dict(episodes=episodes,steps=len(steps),native_reward_algebra_mismatches=sum(x['reward_check']['algebra_abs_error']>1e-8 for x in steps),distance_cache_mismatches=sum(x['independent_distance']['max_pair_error']>1e-8 for x in steps),repeated_reward_changes=sum(not x['reward_repeated_equal'] for x in steps),missing_region_steps=sum(not x['all_regions_present'] for x in steps),h2=h2,h5=h5,API_calls=0,GPU=False,Final_read=False,limits=['Deterministic policies, not LLM scores.','H2 exact over native legal actions and native RL rewards; not original H5 QA gold.','H5 sampled eight continuations per first action, three seeds; not optimality proof.','Observation dependence and configuration omissions do not alone prove unanswerability on a fixed known map.','Native source bodies unchanged; restricted loader remaps old pickle module names.','Native saves/data.csv is scratch only; immutable JSON trajectories are evidence.'])
 tmp=O/'summary.tmp';tmp.write_text(dump(report));tmp.replace(O/'summary.json');return report

def main():
 signal.signal(signal.SIGALRM,alarm);signal.alarm(7200)
 import fcntl
 lock=(O/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=O/'protocol.json';protocol=json.loads(p.read_text());assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==protocol['script_sha256']
 for row in json.loads((O/'source-manifest.json').read_text())['files']+json.loads((O/'asset-manifest.json').read_text()):assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256']
 errors=[]
 try:
  for name in MAPS:
   try:
    roots=run_episode(name,'native_reward_greedy');run_episode(name,'stage_goal_greedy')
    for t,w in roots:two_step(name,t,w);summarize()
   except (TimeoutError,InterruptedError):raise
   except Exception:
    err=dict(map=name,traceback=traceback.format_exc());errors.append(err);save(O/f'failure-{name}-{time.time_ns()}.json',err);print(err['traceback'],flush=True)
   summarize()
  report=summarize();status(state='COMPLETE' if not errors else 'COMPLETE_WITH_FAILURES',errors=errors,episodes=len(report['episodes']))
 except BaseException:
  summarize();status(state='STOPPED_RESUMABLE',traceback=traceback.format_exc());raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
