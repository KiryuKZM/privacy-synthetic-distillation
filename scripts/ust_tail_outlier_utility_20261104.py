"""Prospectively frozen local subgroups, evaluated only after the fixed study.

No teacher, model inference, checkpoint choice or second gold/scoring pass. Uses
existing once-scored correct/valid flags and the exact corresponding inputs.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,sys,time
import numpy as np

R=Path(__file__).resolve().parents[1];D=R/'runs/2026-11-05-ust-environment-generator-ipd-v1'
S=D/'full-schema-mainline-v3';O=D/'tail-outlier-utility-v1'
F=('coverage','demand','distance','waiting_time','charging_time')
DATASETS=('direct-full-v1','shared-cal-v2','copula-corr-v2')
FLAGS=('upper_tail','lower_tail','any_tail','time_NaN','time_NaN_mismatch','statistical_outlier_candidate','Fit_unseen_joint_bin','Fit_rare_seen_joint_bin')
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def bind(p):
    p=Path(p).resolve();return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def bins(v,quart):
    result=[]
    for j,x in enumerate(v):result.append(0 if np.isnan(x) else 1 if x==0 else 2+int(np.searchsorted(quart[j%5],x,side='right')))
    return np.array([result[slot*5+a]*6+result[slot*5+b] for slot in range(4) for a,b in itertools.combinations(range(5),2)])

def freeze():
    O.mkdir(exist_ok=False)
    fp=D/'original-gain-schema-v2/Fit.full-scenes.local.json';fit=load(fp)
    v=np.array([[np.nan if a is None else a for row in r['observations'] for a in row] for r in fit]);assert v.shape==(145,20)
    med=np.nanmedian(v,axis=0);scale=np.nanquantile(v,.75,axis=0)-np.nanquantile(v,.25,axis=0);scale[scale<1e-6]=1.
    scores=np.sqrt(np.mean(np.nan_to_num((v-med)/scale,nan=0.)**2,axis=1));threshold=float(np.quantile(scores,.995))
    assert abs(threshold-load(D/'full-schema-shared-calibrated-v2/fidelity-1000.json')['statistical_outlier_candidate']['Fit_threshold'])<1e-12
    limits=[]
    for j in range(5):
        c=v[:,j::5].ravel();c=c[np.isfinite(c)&(c>0)];limits.append(np.quantile(c,[.01,.99]).tolist())
    old=D.parent/'2026-10-26-generator-discovery/branches/UST';a=module(old/'audit_dev_fidelity_v3.py','subgroup_old_fidelity')
    p=load(old/'dev-fidelity-v3/protocol.json');ref=a.read_source('Fit176',p['inputs']['Fit176']);q=load(old/'dev-fidelity-v3/point-results.json')['transform']['field_positive_quartiles']
    bb=np.stack([bins(row,q) for row in ref['V']]);counts=np.array([np.bincount(bb[:,j],minlength=36) for j in range(40)])
    assert ((counts>0)&(counts<=8)).sum()==263
    save(O/'thresholds.local.json',dict(Fit145=bind(fp),Fit176=p['inputs']['Fit176'],positive_field_q01_q99=limits,
        robust_median20=med.tolist(),robust_IQR20=scale.tolist(),outlier_threshold=threshold,outlier_quantile=.995,
        Fit176_positive_quartiles=q,Fit176_joint_counts=counts.tolist(),Fit_rare_counts=[1,8]))
    save(O/'protocol.json',dict(status='FROZEN_BEFORE_COMMON_INFERENCE_OR_SCORE',source=bind(__file__),thresholds=bind(O/'thresholds.local.json'),
        study_eval_plan=bind(S/'evaluation-plan.json'),flags=FLAGS,subgroups_overlap=True,
        required_dependency=str((S/'controller-terminal.json').relative_to(R)),cohort_sizes=dict(native_synthetic=297,permuted_synthetic=99,historical_Common100=100),
        endpoint_count=7,bootstrap_replicates=2000,bootstrap_seed=2026110457,
        zero_cases='NA_NO_CASES; no refill or omitted subgroup',interval_minimum_groups=2,
        original_Common_input_gold_read=False,private_Audit_Final_read=False,additional_API_or_model_generation=0,
        caveat='Exploratory conditional fixed-run subgroup report; no privacy/anomaly/H5/generalization proof; robust position scores can change with relabeling'))
    print('TAIL_OUTLIER_THRESHOLDS_FROZEN_NO_COMMON_ACCESS',flush=True)

def flags(values,t):
    v=np.asarray(values,float);upper=lower=False
    for j,x in enumerate(v):
        if np.isfinite(x) and x>0:
            lo,hi=t['positive_field_q01_q99'][j%5];upper|=bool(x>hi);lower|=bool(x<lo)
    z=np.nan_to_num((v-np.array(t['robust_median20']))/np.array(t['robust_IQR20']),nan=0.)
    score=float(np.sqrt(np.mean(z*z)));wait=np.isnan(v[3::5]);charge=np.isnan(v[4::5]);b=bins(v,t['Fit176_positive_quartiles'])
    counts=np.array(t['Fit176_joint_counts']);hits=counts[np.arange(40),b]
    return dict(upper_tail=upper,lower_tail=lower,any_tail=upper or lower,time_NaN=bool((wait|charge).any()),
        time_NaN_mismatch=bool((wait!=charge).any()),statistical_outlier_candidate=score>t['outlier_threshold'],
        Fit_unseen_joint_bin=bool((hits==0).any()),Fit_rare_seen_joint_bin=bool(((hits>=1)&(hits<=8)).any())),score

def run():
    protocol=load(O/'protocol.json');assert protocol['source']==bind(__file__) and protocol['thresholds']==bind(O/'thresholds.local.json')
    assert protocol['study_eval_plan']==bind(S/'evaluation-plan.json')
    save(O/'started.json',dict(PID=os.getpid(),source=bind(__file__),epoch=time.time(),API=0))
    begin=time.monotonic()
    while not (S/'controller-terminal.json').exists():
        if (S/'STOP').exists():raise InterruptedError('Own study STOP')
        if (S/'controller-failure.json').exists():
            save(O/'dependency-failure.json',dict(status='SOURCE_STUDY_FAILED_NO_SUBGROUP_EVALUATION',source_failure=bind(S/'controller-failure.json')));return 2
        if time.monotonic()-begin>12*3600:raise TimeoutError('Observer bound only; no source job restart')
        time.sleep(10)
    assert load(S/'controller-terminal.json')['status']=='SIX_MATCHED_FITS_RELOADS_AND_LOCAL_EVALUATION_COMPLETE'
    assert load(S/'evaluation/results.json')['status']=='ALL_FROZEN_ENDPOINTS_SCORED_ONCE'
    frozen=load(S/'evaluation/all-raw-frozen.json');assert frozen['total']==3472 and frozen['raw_each']==496
    sys.path[:0]=[str(S),str(R/'scripts'),str(R/'src')]
    ev=module(S/'local_evaluate.py','post_score_frozen_eval');parser=module(R/'scripts/audit_ust100_input_labels_20261024.py','post_score_features')
    questions=ev.questions();t=load(O/'thresholds.local.json');tags=[]
    for row in questions:
        parsed=parser.parse_data(parser.sections(row['messages'][-1]['content'])['Data'])
        values=[np.nan if parsed[k][f]['number']=='NaN' else float(parsed[k][f]['number']) for k in range(1,5) for f in F]
        ff,score=flags(values,t);tags.append(dict(id=row['id'],cohort=row['cohort'],ability=row['metadata']['ability'],group=row['group'],flags=ff,outlier_score=score))
    assert len(tags)==496
    save(O/'input-flags.local.json',tags)
    endpoints=ev.endpoints();graded={}
    for name in endpoints:
        rows=load(S/'evaluation'/('graded-'+name+'.local.json'));assert [r['id'] for r in rows]==[r['id'] for r in tags];graded[name]=rows
    report={};paired={};rng=np.random.default_rng(2026110457)
    for cohort,n in protocol['cohort_sizes'].items():
        base=[i for i,r in enumerate(tags) if r['cohort']==cohort];assert len(base)==n
        report[cohort]={};paired[cohort]={}
        for flag in ['all',*FLAGS]:
            take=[i for i in base if flag=='all' or tags[i]['flags'][flag]]
            gnames=sorted({tags[i]['group'] for i in take});group=[[i for i in take if tags[i]['group']==g] for g in gnames]
            report[cohort][flag]=dict(n=len(take),source_groups=len(group),status='MEASURED' if take else 'NA_NO_CASES',
                endpoints={name:dict(correct=sum(graded[name][i]['correct'] for i in take),valid=sum(graded[name][i]['valid'] for i in take),
                    length_capped=sum(graded[name][i]['length'] for i in take),total=len(take)) for name in endpoints})
            paired[cohort][flag]={}
            for d in DATASETS:
                a,b=graded[d+'--ipd'],graded[d+'--opd'];delta=np.array([int(x['correct'])-int(y['correct']) for x,y in zip(a,b)])
                point=float(delta[take].mean()) if take else None;ci=None
                if len(group)>=2:
                    draws=[]
                    for _ in range(2000):
                        sample=[i for k in rng.integers(len(group),size=len(group)) for i in group[k]];draws.append(float(delta[sample].mean()))
                    ci=np.quantile(draws,[.025,.975]).tolist()
                paired[cohort][flag][d]=dict(IPD_minus_OPD=point,paired_group_bootstrap95=ci,n=len(take),groups=len(group))
    save(O/'results.json',dict(status='FROZEN_TAIL_OUTLIER_SUBGROUP_UTILITY_COMPLETE',subgroups=report,IPD_minus_OPD=paired,
        thresholds=bind(O/'thresholds.local.json'),source_once_scored_result=bind(S/'evaluation/results.json'),
        original_gold_reads=0,additional_model_inference=0,API_calls=0,subgroups_overlap=True,rare_or_outlier_truth_human_verified=False,
        caveat=protocol['caveat']))
    save(O/'terminal.json',dict(status='COMPLETE',rows=496,endpoints=7,original_gold_reads=0,API_calls=0,epoch=time.time()))
    print('FIXED_TAIL_OUTLIER_SUBGROUPS_COMPLETE',flush=True);return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);args=p.parse_args()
    if args.action=='freeze':freeze()
    else:raise SystemExit(run())
