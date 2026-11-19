"""Fit-only invertible conditional residual extension; original controls frozen."""
from pathlib import Path
import json,sys,time
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import ust_joint_generator_repair_20261107 as parent
base=parent.base;O=parent.O;load=parent.load;save=parent.save
class ResidualRepair(parent.Repair):
 def __init__(self,fit,arm):
  self.arm=arm;self.codec=base.Codec(fit,True);self.source=self.codec.encode(fit);self.fit=fit
  v=parent.values(fit).reshape(-1,5);c=np.c_[np.ones(len(v)),np.log(v[:,1]),np.log(v[:,2])]
  charge=np.isfinite(v[:,4])&(v[:,4]>0);self.beta_charge=np.linalg.lstsq(c[charge],np.log(v[charge,4]),rcond=None)[0]
  wait=np.isfinite(v[:,3])&(v[:,3]>0);assert np.all(v[wait,4]>0)
  wc=np.c_[c[wait],np.log(v[wait,4])];self.beta_wait=np.linalg.lstsq(wc,np.log(v[wait,3]),rcond=None)[0]
  self.x=self.encode(fit);self.marg,self.root,self.proof=parent.observed_correlation(self.x)
 def encode(self,rr):
  old=self.codec.encode(rr);o=parent.values(rr);states=old[:,33:].reshape(-1,4,2).astype(int);assert np.array_equal(states[:,:,0]==2,states[:,:,1]==2)
  y=old[:,:33].copy()
  for loc in range(4):
   context=np.c_[np.ones(len(rr)),np.log(o[:,loc,1]),np.log(o[:,loc,2])]
   for offset,field in [(0,3),(1,4)]:
    j=12+2*loc+offset;valid=states[:,loc,offset]==0;y[:,j]=np.nan
    vv=np.log(o[valid,loc,field]);cc=context[valid]
    if field==4:vv-=cc@self.beta_charge
    elif self.arm=='conditional_charge_and_wait':vv-=np.c_[cc,np.log(o[valid,loc,4])]@self.beta_wait
    y[valid,j]=vv
  return np.c_[y,states[:,:,0]*3+states[:,:,1]]
 def legacy(self,y):
  old=np.zeros((len(y),41));old[:,:33]=y[:,:33];pair=y[:,33:].astype(int);states=np.stack([pair//3,pair%3],axis=2);old[:,33:]=states.reshape(-1,8)
  # Demand and distance encoded coordinates invert to their log source units.
  logd=y[:,4:8]*self.codec.sd[4:8]+self.codec.mu[4:8];logr=y[:,8:12]*self.codec.sd[8:12]+self.codec.mu[8:12]
  for loc in range(4):
   context=np.c_[np.ones(len(y)),logd[:,loc],logr[:,loc]];charge_log=y[:,13+2*loc]+context@self.beta_charge
   wait_log=y[:,12+2*loc].copy()
   if self.arm=='conditional_charge_and_wait':wait_log+=np.c_[context,charge_log]@self.beta_wait
   for off,logs in enumerate([wait_log,charge_log]):
    j=12+2*loc+off;take=states[:,loc,off]==0;old[:,j]=0;old[take,j]=(logs[take]-self.codec.mu[j])/self.codec.sd[j]
  return old
 def roundtrip(self):
  y=np.nan_to_num(self.x,nan=0);out=self.codec.decode(self.legacy(y))
  for a,b in zip(self.fit,out):
   assert not b['failures'];assert a['observations']==b['observations'],a['id'];assert np.allclose(a['feedbacks'],b['feedbacks'],rtol=1e-12,atol=1e-12)
  return dict(rows=len(out),all_displayed_values_exact=True)
def main():
 p=load(O/'conditional-residual-protocol.json');orig=load(O/'protocol.json');fit=load(R/orig['fit']['path']);dev=load(R/orig['development']['path']);audit=parent.Audit(fit,dev);report={}
 for arm in p['arms']:
  model=ResidualRepair(fit,arm);folder=O/arm;folder.mkdir();save(folder/'codec-roundtrip.json',model.roundtrip());save(folder/'fitted-parameters.local.json',dict(charge_coefficients=model.beta_charge.tolist(),wait_coefficients=model.beta_wait.tolist(),marginals=[v.tolist() for v in model.marg],root=model.root.tolist(),proof=model.proof))
  report[arm]={}
  for seed in p['seeds']:
   rr,raw=model.sample(p['n_per_seed'],seed)
   with (folder/f'raw-{seed}.npy').open('xb') as f:np.save(f,raw)
   with (folder/f'rows-{seed}.local.jsonl').open('x') as f:
    for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')
   z=audit.metrics(rr);report[arm][str(seed)]=z;save(folder/f'metrics-{seed}.json',z)
   print(arm,seed,json.dumps({k:z[k] for k in ['paired_NaN_mismatch','structural_failures','mean_absolute_spearman_error_Fit','mean_absolute_spearman_error_Dev','joint_overlap_Dev','exact_joint_scene_clones']}),flush=True)
 save(O/'conditional-residual-results.json',dict(status='ALL6_DATASETS_COMPLETE',arms=report,source_sha256=parent.digest(__file__),GPU=0,API=0))
if __name__=='__main__':main()
