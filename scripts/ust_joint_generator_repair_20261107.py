"""Three predeclared CPU copula controls; no protected evaluation or model calls."""
from pathlib import Path
import json,sys,hashlib,time,itertools,collections
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import rankdata,spearmanr,ks_2samp
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import ust_full_schema_generator_20261104 as base
O=R/'runs/2026-11-07-ust-joint-generator-repair-v1';F=('coverage','demand','distance','waiting_time','charging_time')
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def values(rr):return np.asarray([r['observations'] for r in rr],dtype=float)
def observed_correlation(x,shrink=.1):
 """Ranks of observed entries only; missing numeric entries are not zero fills."""
 x=np.asarray(x,float);z=np.full(x.shape,np.nan);marg=[]
 for j in range(x.shape[1]):
  mask=np.isfinite(x[:,j]);v=x[mask,j];assert len(v)>=2
  z[mask,j]=ndtri((rankdata(v,method='average')-.5)/len(v));marg.append(np.sort(v))
 corr=np.eye(x.shape[1]);counts=np.zeros(corr.shape,dtype=int)
 for a,b in itertools.combinations(range(x.shape[1]),2):
  m=np.isfinite(z[:,a])&np.isfinite(z[:,b]);counts[a,b]=counts[b,a]=int(m.sum())
  c=float(np.corrcoef(z[m,a],z[m,b])[0,1]) if m.sum()>=3 and np.std(z[m,a])>1e-12 and np.std(z[m,b])>1e-12 else 0.
  corr[a,b]=corr[b,a]=c
 eigen,Q=np.linalg.eigh(corr);project=(Q*np.maximum(eigen,1e-8))@Q.T
 sd=np.sqrt(np.diag(project));project/=sd[:,None]*sd[None,:]
 final=(1-shrink)*project+shrink*np.eye(len(corr));root=np.linalg.cholesky(final)
 return marg,root,dict(pairwise_min_eigenvalue=float(eigen.min()),pairwise_negative_eigenvalues=int((eigen<0).sum()),final_min_eigenvalue=float(np.linalg.eigvalsh(final).min()),pair_observed_counts=counts.tolist(),correlation=final.tolist())
class Repair:
 def __init__(self,fit,arm):
  self.arm=arm;self.codec=base.Codec(fit,True);self.source=self.codec.encode(fit);self.fit=fit
  self.x=self.encode(fit);self.marg,self.root,self.proof=observed_correlation(self.x)
 def encode(self,rr):
  old=self.codec.encode(rr);o=values(rr);states=old[:,33:].reshape(-1,4,2).astype(int)
  assert np.array_equal(states[:,:,0]==2,states[:,:,1]==2),'Original source has inconsistent missingness'
  if self.arm=='legacy_correlation':return old
  if self.arm=='joint_state_only':return np.c_[old[:,:33],states[:,:,0]*3+states[:,:,1]]
  assert self.arm=='joint_state_observed_times'
  numeric=old[:,:32].copy();numeric[:,12:20]=np.log1p(o[:,:,3:]).reshape(-1,8)
  return np.c_[numeric,old[:,32],(states[:,:,0]==2).astype(float)]
 def legacy(self,y):
  if self.arm=='legacy_correlation':return y.copy()
  old=np.zeros((len(y),41));old[:,:33]=y[:,:33]
  if self.arm=='joint_state_only':
   pair=y[:,33:].astype(int);old[:,33:]=np.stack([pair//3,pair%3],axis=2).reshape(-1,8)
  else:
   tt=np.expm1(y[:,12:20]);missing=np.repeat(y[:,33:].astype(bool),2,axis=1)
   states=np.where(missing,2,np.where(tt==0,1,0));old[:,33:]=states
   for j in range(8):
    take=states[:,j]==0
    old[:,12+j]=0
    old[take,12+j]=(np.log(tt[take,j])-self.codec.mu[12+j])/self.codec.sd[12+j]
  return old
 def sample(self,n,seed):
  # Same rectangular independent-normal stream across dimensions/arms.
  z=np.random.default_rng(seed).normal(size=(n,41))[:,:self.x.shape[1]]@self.root.T;u=ndtr(z)
  y=np.column_stack([v[np.minimum((u[:,j]*len(v)).astype(int),len(v)-1)] for j,v in enumerate(self.marg)])
  raw=self.legacy(y);return self.codec.decode(raw),raw
 def roundtrip(self):
  y=self.x.copy()
  if self.arm=='joint_state_observed_times':
   for j in range(12,20):y[~np.isfinite(y[:,j]),j]=0
  out=self.codec.decode(self.legacy(y));differ=[]
  for a,b in zip(self.fit,out):
   assert not b['failures'];assert a['observations']==b['observations'],(self.arm,a['id']);assert np.allclose(a['feedbacks'],b['feedbacks'],rtol=1e-12,atol=1e-12)
  return dict(rows=len(out),all_displayed_values_exact=True,feedback_max_tolerance=1e-12)
def flat_spearman(v):
 p=v.reshape(-1,5);out={}
 for a,b in itertools.combinations(range(5),2):
  m=np.isfinite(p[:,a])&np.isfinite(p[:,b]);out[F[a]+'/'+F[b]]=float(spearmanr(p[m,a],p[m,b]).statistic)
 return out
class Audit:
 def __init__(self,fit,dev):
  self.fit=values(fit);self.dev=values(dev);self.q=[];self.bounds=[];self.pairs=list(itertools.combinations(range(5),2))
  for j in range(5):
   a=self.fit[:,:,j].ravel();a=a[np.isfinite(a)&(a>0)];self.q.append(np.quantile(a,[.25,.5,.75]));self.bounds.append(np.quantile(a,[.01,.99]))
  self.fh,self.fj=self.hist(self.fit);self.dh,self.dj=self.hist(self.dev);self.fc=flat_spearman(self.fit);self.dc=flat_spearman(self.dev)
  fs=self.signature(self.fit);self.med=np.nanmedian(fs,axis=0);self.scale=np.nanquantile(fs,.75,axis=0)-np.nanquantile(fs,.25,axis=0);self.scale[self.scale<1e-6]=1
  self.fit_z=np.nan_to_num((fs-self.med)/self.scale,nan=0);self.exact={self.key(v) for v in self.fit}
 def key(self,v):return json.dumps(sorted([[None if np.isnan(a) else float(a) for a in x] for x in v],key=lambda x:json.dumps(x)),sort_keys=True)
 def signature(self,v):return np.log1p(np.sort(v,axis=1)).reshape(len(v),20)
 def hist(self,v):
  p=v.reshape(-1,5);codes=np.array([np.where(np.isnan(p[:,j]),0,np.where(p[:,j]==0,1,2+np.searchsorted(self.q[j],p[:,j]))) for j in range(5)]).T
  return np.array([np.bincount(codes[:,j],minlength=6)/len(p) for j in range(5)]),np.array([np.bincount(codes[:,a]*6+codes[:,b],minlength=36)/len(p) for a,b in self.pairs])
 def metrics(self,rr):
  valid=[r for r in rr if 'observations' in r];v=values(valid);n=len(v);p=v.reshape(-1,5);m,j=self.hist(v);c=flat_spearman(v);fields={};tail=np.zeros(n,dtype=bool)
  for i,f in enumerate(F):
   a=p[:,i];fref=self.fit[:,:,i].ravel();dref=self.dev[:,:,i].ravel();finite=lambda a:a[np.isfinite(a)];lo,hi=self.bounds[i]
   tail|=np.any((v[:,:,i]>0)&np.isfinite(v[:,:,i])&((v[:,:,i]<lo)|(v[:,:,i]>hi)),axis=1)
   fields[f]=dict(KS_Fit=float(ks_2samp(finite(fref),finite(a),method='asymp').statistic),KS_Dev=float(ks_2samp(finite(dref),finite(a),method='asymp').statistic),zero=float((a==0).mean()),NaN=float(np.isnan(a).mean()),positive_quantiles=np.quantile(a[np.isfinite(a)&(a>0)],[.01,.5,.99]).tolist())
  span=np.max(v[:,:,0],axis=1)-np.min(v[:,:,0],axis=1);zz=np.nan_to_num((self.signature(v)-self.med)/self.scale,nan=0);near=[]
  for chunk in np.array_split(zz,max(1,len(zz)//128)):near.extend(np.sqrt(np.mean((chunk[:,None,:]-self.fit_z[None,:,:])**2,axis=2).min(axis=1)).tolist())
  return dict(total=len(rr),decoded=n,structural_failures=sum(bool(r['failures']) for r in rr),paired_NaN_mismatch=int(np.any(np.isnan(v[:,:,3])!=np.isnan(v[:,:,4]),axis=1).sum()),time_NaN=int(np.isnan(v[:,:,3:]).any(axis=(1,2)).sum()),tail_count=int(tail.sum()),coverage_range_median=float(np.median(span)),coverage_range_p95=float(np.quantile(span,.95)),fields=fields,spearman=c,mean_absolute_spearman_error_Fit=float(np.mean([abs(c[k]-self.fc[k]) for k in c])),mean_absolute_spearman_error_Dev=float(np.mean([abs(c[k]-self.dc[k]) for k in c])),marginal_overlap_Fit=float(np.minimum(m,self.fh).sum(axis=1).mean()),joint_overlap_Fit=float(np.minimum(j,self.fj).sum(axis=1).mean()),marginal_overlap_Dev=float(np.minimum(m,self.dh).sum(axis=1).mean()),joint_overlap_Dev=float(np.minimum(j,self.dj).sum(axis=1).mean()),exact_joint_scene_clones=sum(self.key(a) in self.exact for a in v),nearest_Fit_field_sorted_signature_RMS_quantiles=np.quantile(near,[0,.01,.5,1]).tolist(),copy_metric_is_not_privacy_proof=True)
def main():
 p=load(O/'protocol.json')
 for key in ['fit','development','coverage_base']:assert digest(R/p[key]['path'])==p[key]['sha256']
 fit=load(R/p['fit']['path']);dev=load(R/p['development']['path']);assert len(fit)==145 and len(dev)==24
 audit=Audit(fit,dev);report={};start=time.monotonic()
 save(O/'run-started.json',dict(epoch=time.time(),script_sha256=digest(__file__),GPU=0,API=0))
 save(O/'reference-only-metrics.json',dict(Fit=audit.metrics([dict(r,failures=[]) for r in fit]),Dev=audit.metrics([dict(r,failures=[]) for r in dev])))
 for arm in p['arms']:
  model=Repair(fit,arm);folder=O/arm;folder.mkdir();save(folder/'codec-roundtrip.json',model.roundtrip());save(folder/'fitted-parameters.local.json',dict(marginals=[v.tolist() for v in model.marg],root=model.root.tolist(),proof=model.proof,mean=model.codec.mu.tolist(),scale=model.codec.sd.tolist(),coverage_base=model.codec.base))
  report[arm]={}
  for seed in p['seeds']:
   rr,raw=model.sample(p['samples_per_arm_seed'],seed)
   with (folder/f'raw-{seed}.npy').open('xb') as f:np.save(f,raw)
   with (folder/f'rows-{seed}.local.jsonl').open('x') as f:
    for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')
   z=audit.metrics(rr);report[arm][str(seed)]=z;save(folder/f'metrics-{seed}.json',z)
   print(arm,seed,json.dumps({k:z[k] for k in ['paired_NaN_mismatch','structural_failures','mean_absolute_spearman_error_Fit','mean_absolute_spearman_error_Dev','joint_overlap_Dev','exact_joint_scene_clones']}),flush=True)
 save(O/'results.json',dict(status='ALL9_GENERATOR_DATASETS_COMPLETE',arms=report,protocol_sha256=digest(O/'protocol.json'),source_sha256=digest(__file__),GPU=0,API=0,original_benchmark_unchanged=True,Common_Audit_Confirm_Final_access=False))
 save(O/'generator-terminal.json',dict(status='COMPLETE',seconds=time.monotonic()-start,generated_candidate_sets=9000,GPU=0,API=0));print('COMPLETE',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
