"""Complete stage-two road task; deterministic simulator-independent graph arithmetic.
Not a reproduction of UST's unreleased QA gold generator. No model/API/file IO.
"""
import copy,json,math
import numpy as np

class RoadTask:
 def __init__(self,spec):
  self.spec=spec;self.n=len(spec['nodes']);self.regions=spec['regions'];self.edges={e['id']:e for e in spec['edges']}
  self.initial_built=frozenset(e['id'] for e in spec['edges'] if e['built']);self.initial_remaining=spec['remaining_steps']
  self.maxcost=spec['max_road_cost'];self.floor=spec['f2f_avg_min'];self.dw=spec['distance_weight'];self.cw=spec['cost_weight'];self.gamma=spec['discount']
  self.pairs=[(i,j) for i in range(len(self.regions)) for j in range(i+1,len(self.regions))]
  width=max(map(len,self.regions));self.region_index=np.array([f+[self.n]*(width-len(f)) for f in self.regions],dtype=int)
  self.pair_index=np.triu_indices(len(self.regions),1)
  self.cache={};self.initial=self.from_built(self.initial_built)
 def from_built(self,built):
  d=np.full((self.n,self.n),1000.,dtype=float);np.fill_diagonal(d,0.);active=set()
  for a in built:
   e=self.edges[a];u,v=e['nodes'];d[u,v]=d[v,u]=e['weight'];active.update((u,v))
  for k in range(self.n):d=np.minimum(d,d[:,k,None]+d[None,k,:])
  return (frozenset(built),d,frozenset(active),self.average(d,active))
 def average(self,d,active):
  masked=np.full((self.n+1,self.n+1),1000.,dtype=float)
  aa=np.asarray(sorted(active),dtype=int);masked[np.ix_(aa,aa)]=d[np.ix_(aa,aa)]
  to_region=masked[:,self.region_index].min(axis=2)
  region_dist=to_region[self.region_index,:].min(axis=1)
  return float(region_dist[self.pair_index].mean())
 def legal(self,state):
  built,d,active,avg=state
  if len(built)-len(self.initial_built)>=self.initial_remaining:return []
  return sorted(a for a,e in self.edges.items() if a not in built and (e['nodes'][0] in active or e['nodes'][1] in active))
 def step(self,state,a):
  if a not in self.legal(state):raise ValueError('illegal road '+str(a))
  built,d,active,before=state;e=self.edges[a];u,v=e['nodes'];w=e['weight']
  nd=np.minimum(d,np.minimum(d[:,u,None]+w+d[None,v,:],d[:,v,None]+w+d[None,u,:]));na=active|{u,v};after=self.average(nd,na)
  reward=self.dw*(before-after)/(before-self.floor)+self.cw*w/self.maxcost
  return (built|{a},nd,na,after),float(reward)
 def replay(self,plan):
  state=self.initial;trace=[];value=0.
  for i,a in enumerate(plan):
   before=state[3];state,r=self.step(state,int(a));value+=self.gamma**i*r;trace.append(dict(action=int(a),reward=r,distance_before=before,distance_after=state[3],legal_next=self.legal(state)))
  return dict(return_value=value,trace=trace,final_distance=state[3])
 def beam(self,horizon=5,width=64):
  beam=[(0.,(),self.initial)]
  for depth in range(horizon):
   candidates={}
   for value,path,state in beam:
    actions=self.legal(state)
    if not actions:
     k=state[0];old=candidates.get(k)
     if old is None or value>old[0]:candidates[k]=(value,path,state)
    for a in actions:
     ns,r=self.step(state,a);v=value+self.gamma**depth*r;k=ns[0];old=candidates.get(k)
     if old is None or v>old[0]+1e-12 or (abs(v-old[0])<=1e-12 and path+(a,)<old[1]):candidates[k]=(v,path+(a,),ns)
   beam=sorted(candidates.values(),key=lambda x:(-x[0],x[1]))[:width]
  value,path,state=beam[0]
  return dict(status='DETERMINISTIC_LOWER_BOUND_NOT_GOLD',horizon=horizon,width=width,plan=list(path),return_value=value)
 def exact2(self):
  rows=[]
  for a in self.legal(self.initial):
   s,r=self.step(self.initial,a);cont=[(r+self.gamma*self.step(s,b)[1],b) for b in self.legal(s)]
   v,b=max(cont,key=lambda x:(x[0],-x[1])) if cont else (r,None)
   rows.append(dict(first=a,second=b,value=v))
  return dict(status='EXACT_ALL_LEGAL_H2',values=rows,best=max(rows,key=lambda x:(x['value'],-x['first'])))

def render_full(s):
 payload={k:s[k] for k in ['regions','edges']}
 # Include full precision values; all potential future roads, not only one-step frontier.
 return 'Complete road network and ALL region memberships. Regions are numbered by list index. Each edge has its road ID, endpoints, exact weight, and current built flag.\n'+json.dumps(payload,separators=(',',':'))
def system(s):
 return ('You are planning roads in stage 2 of a deterministic urban simulator. Plan exactly FIVE consecutive legal road constructions. No examples or tools are provided. Return only <plan>[road_id1,road_id2,road_id3,road_id4,road_id5]</plan>.\n'
 'An unbuilt road is legal when at least one endpoint is already incident to a built road. Building activates both endpoints and may unlock further roads. Do not repeat roads. Region distance is the shortest built-road path between any ACTIVE boundary node of each region; take the arithmetic mean over all unordered pairs of ALL regions, allowing zero distance for a shared active node.\n'
 f'Maximize sum(t=0..4) {s["discount"]}^t * reward_t. reward_t = {s["distance_weight"]}*(D_before-D_after)/(D_before-{s["f2f_avg_min"]!r}) + ({s["cost_weight"]})*road_weight/{s["max_road_cost"]!r}. Current mean D={s["initial_distance"]!r}. There are {s["remaining_steps"]} constructions left before environment termination. All five requested constructions fit.\n'
 'If information is insufficient, do not invent missing geometry. You may return <insufficient/> instead; it will be recorded as abstention, not a legal plan.')
