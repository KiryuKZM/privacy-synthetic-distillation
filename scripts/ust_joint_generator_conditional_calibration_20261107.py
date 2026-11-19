"""Fixed two-field pooled monotone map after conditional residual dependence."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.stats import ks_2samp
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import ust_joint_generator_repair_20261107 as b
from ust_joint_generator_residual_20261107 import ResidualRepair
O=b.O
class Calibrated:
 def __init__(self,fit,probe_n=16384,probe_seed=2026110715):
  self.model=ResidualRepair(fit,'conditional_charge_and_wait');_,raw=self.model.sample(probe_n,probe_seed);self.tables=[]
  for off,field in [(0,3),(1,4)]:
   cols=np.arange(12+off,20,2);states=raw[:,33+np.arange(off,8,2)];z=raw[:,cols]*self.model.codec.sd[cols]+self.model.codec.mu[cols];probe=np.sort(z[states==0]);v=b.values(fit)[:,:,field];target=np.sort(np.log(v[np.isfinite(v)&(v>0)]));assert len(probe)>64
   self.tables.append((probe,target))
  self.probe=raw
 def transform(self,raw):
  out=raw.copy()
  for off,(probe,target) in enumerate(self.tables):
   cols=np.arange(12+off,20,2)
   for j in cols:
    take=out[:,33+j-12]==0;v=out[take,j]*self.model.codec.sd[j]+self.model.codec.mu[j]
    u=(np.searchsorted(probe,v,side='left')+np.searchsorted(probe,v,side='right'))/(2*len(probe));mapped=target[np.minimum((u*len(target)).astype(int),len(target)-1)];out[take,j]=(mapped-self.model.codec.mu[j])/self.model.codec.sd[j]
  return out
 def sample(self,n,seed):
  _,raw=self.model.sample(n,seed);mapped=self.transform(raw);return self.model.codec.decode(mapped),mapped
 def consistency(self,before,after):
  assert np.array_equal(before[:,32:],after[:,32:])
  keep=[j for j in range(32) if not 12<=j<20];assert np.array_equal(before[:,keep],after[:,keep])
  return dict(categories_unchanged=True,coverage_demand_distance_feedback_unchanged=True,positive_times_only=True)
def main():
 p=b.load(O/'conditional-calibration-protocol.json');orig=b.load(O/'protocol.json');fit=b.load(R/orig['fit']['path']);dev=b.load(R/orig['development']['path']);model=Calibrated(fit,p['probe_n'],p['probe_seed']);folder=O/'conditional_calibrated';folder.mkdir();audit=b.Audit(fit,dev)
 with (folder/'probe-raw.npy').open('xb') as f:np.save(f,model.probe)
 b.save(folder/'tables.local.json',[dict(log_probe=v.tolist(),Fit_positive_log_inverse=t.tolist()) for v,t in model.tables])
 _,raw=model.model.sample(p['check_n'],p['check_seed']);mapped=model.transform(raw);checks=[]
 for off in [0,1]:
  cols=np.arange(12+off,20,2);state=mapped[:,33+np.arange(off,8,2)];v=mapped[:,cols]*model.model.codec.sd[cols]+model.model.codec.mu[cols];v=v[state==0];checks.append(dict(field=['waiting_time','charging_time'][off],positive_count=len(v),KS_Fit=float(ks_2samp(v,model.tables[off][1],method='asymp').statistic)))
 b.save(folder/'independent-marginal-check.json',dict(n=p['check_n'],checks=checks,**model.consistency(raw,mapped)))
 results={}
 for seed in p['output_seeds']:
  rr,raw=model.sample(p['output_n'],seed)
  with (folder/f'raw-{seed}.npy').open('xb') as f:np.save(f,raw)
  with (folder/f'rows-{seed}.local.jsonl').open('x') as f:
   for row in rr:f.write(json.dumps(row,allow_nan=False)+'\n')
  m=audit.metrics(rr);results[str(seed)]=m;b.save(folder/f'metrics-{seed}.json',m);print(seed,json.dumps({k:m[k] for k in ['structural_failures','paired_NaN_mismatch','tail_count','mean_absolute_spearman_error_Fit','mean_absolute_spearman_error_Dev','joint_overlap_Dev','exact_joint_scene_clones']}),flush=True)
 b.save(O/'conditional-calibration-results.json',dict(status='COMPLETE_ALL3_SEEDS',results=results,source_sha256=b.digest(__file__),probe=16384,check=8192,generated=3000,GPU=0,API=0))
if __name__=='__main__':main()
