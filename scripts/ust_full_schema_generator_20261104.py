"""Bounded local experiment on original UST observations AND action outcomes.

Uses the unmodified official TabDiff model/loss/sample implementations. This is
not a reproduction of the undisclosed physical episode/H5 label construction.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
import numpy as np

R=Path(__file__).resolve().parents[1]
D=R/'runs/2026-11-05-ust-environment-generator-ipd-v1'
P=D/'original-gain-schema-v2'
SEED=2026110429
CONFIG=dict(updates=2000,batch_size=64,num_layers=2,factor=8,dim_t=128,
            num_timesteps=50,learning_rate=.001)

def load(p): return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def bind(p):
    p=Path(p);return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rows(p,x):
    with Path(p).open('x') as f:
        for a in x:f.write(json.dumps(a,allow_nan=False)+'\n')

class Codec:
    """32 numeric + goal and eight positive/zero/NaN categorical columns."""
    def __init__(self,fit,shared):
        self.shared=shared;self.base=load(P/'fit-coverage-normalizer.local.json')['value']
        z,cat,mask=self.raw(fit)
        self.mu=np.array([z[mask[:,j],j].mean() for j in range(32)])
        self.sd=np.array([z[mask[:,j],j].std() for j in range(32)])
        self.sd[self.sd<1e-6]=1.
    def raw(self,rr):
        nums=[];cats=[];masks=[]
        for r in rr:
            o=np.array([[np.nan if v is None else v for v in x] for x in r['observations']]);g=np.array(r['feedbacks'])
            cg=np.log(g[:,0]);c=np.r_[cg.mean(),(cg-cg.mean())[:3]] if self.shared else cg
            ts=o[:,3:].reshape(-1);state=np.where(np.isnan(ts),2,np.where(ts==0,1,0))
            positive=state==0;logs=np.zeros(8);logs[positive]=np.log(ts[positive])
            v=np.r_[c,np.log(o[:,1]),np.log(o[:,2]),logs,np.log(1-g[:,1:]).reshape(-1)]
            assert len(v)==32 and np.isfinite(v).all()
            nums.append(v);cats.append(np.r_[r['goal'],state]);masks.append(np.r_[np.ones(12,bool),positive,np.ones(12,bool)])
        return np.array(nums),np.array(cats),np.array(masks)
    def encode(self,rr):
        z,c,m=self.raw(rr);v=(z-self.mu)/self.sd;v[~m]=0
        return np.c_[v,c]
    def decode(self,raw):
        zz=raw[:,:32]*self.sd+self.mu;out=[]
        for i,(z,cat) in enumerate(zip(zz,raw[:,32:])):
            issues=[]
            if not np.isfinite(z).all() or not np.isfinite(cat).all():
                out.append(dict(index=i,failures=['nonfinite_latent']));continue
            if any(v!=int(v) for v in cat) or not 0<=cat[0]<4 or any(v not in (0,1,2) for v in cat[1:]):
                out.append(dict(index=i,failures=['invalid_category']));continue
            lc=z[:4]
            if self.shared:lc=z[0]+np.r_[z[1:4],-z[1:4].sum()]
            with np.errstate(over='ignore',invalid='ignore'):
                covg=np.exp(lc);demand=np.exp(z[4:8]);distance=np.exp(z[8:12]);ts=np.exp(z[12:20]);other=1-np.exp(z[20:32]).reshape(4,3)
            ts=np.array([np.nan if s==2 else 0. if s==1 else v for v,s in zip(ts,cat[1:])]).reshape(4,2)
            obs=np.c_[self.base*(1+covg),demand,distance,ts];gain=np.c_[covg,other]
            if not np.isfinite(gain).all() or not np.isfinite(obs[:,:3]).all() or np.isinf(ts).any():
                out.append(dict(index=i,failures=['nonfinite_decoded']));continue
            # Preserve author's coverage precision3, other values precision2.
            display=obs.copy();display[:,0]=np.round(display[:,0],3);display[:,1:]=np.round(display[:,1:],2)
            if np.any(display[:,:3]<=0):issues.append('rounded_positive_observation_nonpositive')
            if np.any((cat[1:]==0)&(display[:,3:].reshape(-1)<=0)):issues.append('positive_time_rounded_to_zero')
            out.append(dict(index=i,goal=int(cat[0]),observations=[[None if np.isnan(v) else float(v) for v in x] for x in display],
                feedbacks=gain.tolist(),local_gain_answers=['ABCD'[j] for j,v in enumerate(gain[:,int(cat[0])]) if v==gain[:,int(cat[0])].max()],failures=issues))
        return out
    def proof(self,rr):
        raw=self.encode(rr);allproof=[]
        for dtype in (np.float64,np.float32):
            dec=self.decode(raw.astype(dtype));maxerr=0.;display_differences=[];label_changes=[]
            for a,b in zip(rr,dec):
                assert not b['failures'],b
                if a['observations']!=b['observations']:
                    display_differences.append(a['id'])
                    assert dtype==np.float32,(a['id'],a['observations'],b['observations'])
                err=np.max(np.abs(np.array(a['feedbacks'])-np.array(b['feedbacks'])));maxerr=max(maxerr,float(err))
                if a['local_original_answer'] not in b['local_gain_answers']:label_changes.append(a['id'])
            assert not label_changes,label_changes
            allproof.append(dict(dtype=str(dtype),rows=len(rr),exact_display=not display_differences,
                float32_display_quantization_changed_rows=display_differences,original_label_retained=True,max_gain_error=maxerr))
        return allproof

def model(device):
    import torch
    sys.path.insert(0,str(R/'vendor/tabdiff'));sys.path.insert(0,str(R/'artifacts/tabdiff-site'))
    from tabdiff.modules.main_modules import UniModMLP,Model
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    cats=np.array([4]+[3]*8)
    net=UniModMLP(d_numerical=32,categories=(cats+1).tolist(),num_layers=2,d_token=4,n_head=1,factor=8,dim_t=128)
    net=Model(net,precond=True,sigma_data=1.,net_conditioning='sigma').to(device)
    return UnifiedCtimeDiffusion(cats,32,net,None,num_timesteps=50,
        scheduler='power_mean_per_column',cat_scheduler='log_linear_per_column',noise_dist='uniform_t',
        edm_params=dict(precond=True,sigma_data=1.,net_conditioning='sigma'),
        noise_schedule_params=dict(sigma_min=.002,sigma_max=80.,rho=7.,eps_max=.001,eps_min=.00001,rho_init=7.,rho_offset=5.,k_init=-6.,k_offset=1.),
        sampler_params=dict(stochastic_sampler=True,second_order_correction=True),device=device).to(device)

def fidelity(fit,dev,generated):
    from scipy.stats import ks_2samp
    def vals(rr):
        return np.array([[np.nan if v is None else v for x in r['observations'] for v in x] for r in rr])
    def ties(rr):return sum(len(set(x[0] for x in r['observations']))<4 for r in rr)/len(rr)
    vv=[r for r in generated if not r['failures']]
    if not vv:return dict(raw=len(generated),structural_pass=0)
    f=vals(fit);v=vals(dev);g=vals(vv);columns=[]
    for j in range(5):
        a=f[:,j::5].ravel();b=v[:,j::5].ravel();c=g[:,j::5].ravel();af=a[np.isfinite(a)];bf=b[np.isfinite(b)];cf=c[np.isfinite(c)]
        lo,hi=np.quantile(af,[.01,.99])
        columns.append(dict(field=['coverage','demand','distance','waiting_time','charging_time'][j],
            Fit_quantiles=np.quantile(af,[0,.01,.5,.99,1]).tolist(),Dev_quantiles=np.quantile(bf,[0,.01,.5,.99,1]).tolist(),
            generated_quantiles=np.quantile(cf,[0,.01,.5,.99,1]).tolist(),
            KS_vs_Fit=float(ks_2samp(af,cf).statistic),Dev_KS_vs_Fit=float(ks_2samp(af,bf).statistic),
            Fit_zero=float((a==0).mean()),Dev_zero=float((b==0).mean()),generated_zero=float((c==0).mean()),
            Fit_NaN=float(np.isnan(a).mean()),Dev_NaN=float(np.isnan(b).mean()),generated_NaN=float(np.isnan(c).mean()),
            Fit_lowtail=float((af<lo).mean()),Dev_lowtail=float((bf<lo).mean()),generated_lowtail=float((cf<lo).mean()),
            Fit_hightail=float((af>hi).mean()),Dev_hightail=float((bf>hi).mean()),generated_hightail=float((cf>hi).mean())))
    # Same Fit-derived robust coordinate system and threshold for both comparators.
    mu=np.nanmedian(f,axis=0);scale=np.nanquantile(f,.75,axis=0)-np.nanquantile(f,.25,axis=0);scale[scale<1e-6]=1.
    def zz(a):return np.nan_to_num((a-mu)/scale,nan=0.)
    zf,zv,zg=zz(f),zz(v),zz(g);scores=lambda a:np.sqrt(np.mean(a*a,axis=1));threshold=float(np.quantile(scores(zf),.995))
    nearest=np.sqrt(np.min(np.mean((zg[:,None,:]-zf[None,:,:])**2,axis=2),axis=1))
    return dict(raw=len(generated),structural_pass=len(vv),no_refill=True,columns=columns,
        coverage_any_display_tie=dict(Fit=ties(fit),Dev=ties(dev),generated=ties(vv)),
        statistical_outlier_candidate=dict(Fit_threshold=threshold,quantile=.995,Fit=float((scores(zf)>threshold).mean()),
            Dev=float((scores(zv)>threshold).mean()),generated=float((scores(zg)>threshold).mean()),human_verified=False),
        nearest_Fit_robust_RMS_quantiles=np.quantile(nearest,[0,.01,.5,1]).tolist(),exact_display_clones=int((nearest<1e-10).sum()),privacy_proof=False,
        requested_gain_best_at_original_goal_slot=float(np.mean([r['goal']==np.argmax(np.array(r['feedbacks'])[:,r['goal']]) for r in vv])))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['direct','shared','copula']);a=parser.parse_args()
    os.environ['WANDB_MODE']='disabled';os.environ['WANDB_DISABLED']='true'
    out=D/('full-schema-'+a.method+'-v1');out.mkdir(exist_ok=False)
    fit=load(P/'Fit.full-scenes.local.json');dev=load(P/'SelectDev.full-scenes.local.json');codec=Codec(fit,a.method!='direct')
    save(out/'protocol.json',dict(source=bind(__file__),Fit=bind(P/'Fit.full-scenes.local.json'),SelectDev=bind(P/'SelectDev.full-scenes.local.json'),
        coverage_base=bind(P/'fit-coverage-normalizer.local.json'),config=CONFIG,method=a.method,seed=SEED,
        unmodified_official_TabDiff=True,numeric_columns=32,categorical_cardinality=[4]+[3]*8,
        external_API=0,Audit_Final_read=False,raw100_then1000_if_95percent_structural=True,source_physical_H5_not_reproduced=True))
    save(out/'codec-proof.json',dict(Fit=codec.proof(fit),SelectDev=codec.proof(dev)))
    save(out/'preprocessing.local.json',dict(mean=codec.mu.tolist(),std=codec.sd.tolist(),base=codec.base,shared=codec.shared,Fit_only=True))
    x=codec.encode(fit).astype(np.float32);start=time.monotonic()
    if a.method=='copula':
        from scipy.special import ndtr,ndtri
        ranks=np.empty_like(x,dtype=np.float64)
        for j in range(x.shape[1]):
            _,inv,count=np.unique(x[:,j],return_inverse=True,return_counts=True);mid=(np.cumsum(count)-count/2.)/len(x);ranks[:,j]=ndtri(mid[inv])
        cov=np.cov(ranks,rowvar=False);cov=.9*cov+.1*np.diag(np.diag(cov))+np.eye(x.shape[1])*1e-6
        def sample(n,seed):
            u=ndtr(np.random.default_rng(seed).multivariate_normal(np.zeros(x.shape[1]),cov,n))
            return np.column_stack([np.sort(x[:,j])[np.minimum((u[:,j]*len(x)).astype(int),len(x)-1)] for j in range(x.shape[1])])
    else:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED);np.random.seed(SEED)
        assert torch.cuda.is_available();device=torch.device('cuda');torch.cuda.set_per_process_memory_fraction(.18)
        net=model(device);optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=0);tensor=torch.from_numpy(x).to(device);history=[]
        for step in range(1,2001):
            if (D/'STOP').exists():raise InterruptedError('Own STOP')
            if time.monotonic()-start>3600:raise TimeoutError('Frozen fit wall bound')
            ix=torch.randint(len(x),(64,),device=device);optim.zero_grad();d,c=net.mixed_loss(tensor[ix]);loss=d+c
            assert torch.isfinite(loss);loss.backward();optim.step()
            if step%200==0:
                history.append(dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-start));print(a.method,history[-1],flush=True)
        checkpoint=out/'step2000.pt';torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),step=2000,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),source=bind(__file__),config=CONFIG),checkpoint)
        fresh=model(device);s=torch.load(checkpoint,map_location=device,weights_only=True);fresh.load_state_dict(s['model'])
        assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in net.state_dict().items());del net;net=fresh;net.eval()
        save(out/'training.json',dict(history=history,actual_updates=2000,all_model_tensors_exact_after_fresh_reload=True,checkpoint=bind(checkpoint),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        def sample(n,seed):
            torch.manual_seed(seed);return np.concatenate([net.sample(min(64,n-i)).cpu().numpy() for i in range(0,n,64)])
    results=[]
    for n in (100,1000):
        raw=sample(n,SEED+n);np.save(out/f'raw-{n}.npy',raw);rr=codec.decode(raw);rows(out/f'raw-{n}.local.jsonl',rr)
        report=fidelity(fit,dev,rr);save(out/f'fidelity-{n}.json',report);results.append(report)
        if n==100 and report['structural_pass']<95:break
    save(out/'terminal.json',dict(status='LOCAL_FULL_SCHEMA_EXPERIMENT_COMPLETE',method=a.method,seconds=time.monotonic()-start,API=0,
        results=[{k:v for k,v in rr.items() if k!='columns'} for rr in results],acceptance='REQUIRES_FIDELITY_AND_TASK_REVIEW'))
    print(json.dumps(load(out/'terminal.json')),flush=True)

if __name__=='__main__':main()
