#!/usr/bin/env python3
"""Read-only CPU1 post-hoc native GAD audit. Never loads model/torch or writes frozen runs."""
import argparse,array,collections,hashlib,json,math,os,statistics,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/2026-10-25-overnight-onpolicy'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import eval_ust_sg_qwen_modes_20261024 as q
from run_llm_task_sweep_20261023 import score_prediction

def path(x):
 x=str(x); prefix='/home/kiryu/privacy-synthetic-distillation/'
 return ROOT/x[len(prefix):] if x.startswith(prefix) else ROOT/x if not Path(x).is_absolute() else Path(x)
def load(p):return json.loads(path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()
def rows(spec):
 p=path(spec['path']);assert sha(p)==spec['sha256'];return [json.loads(x) for x in p.read_text().splitlines() if x]
def stats(v):
 v=list(v)
 if not v:return dict(n=0)
 s=sorted(v);return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),min=s[0],max=s[-1],p90=s[min(len(s)-1,int(.9*len(s)))])
def corr(x,y):
 if len(x)<2:return None
 mx,my=statistics.mean(x),statistics.mean(y);a=sum((v-mx)**2 for v in x);b=sum((v-my)**2 for v in y)
 return sum((v-mx)*(w-my) for v,w in zip(x,y))/math.sqrt(a*b) if a*b else None
def records(folder):return [load(f) for f in sorted(folder.glob('*.json')) if '.started.' not in f.name]
def curve(v):return dict(first10=stats(v[:10]),middle10=stats(v[max(0,len(v)//2-5):len(v)//2+5]),last10=stats(v[-10:]),all=stats(v))
def eval_compare(u):
 arms={a:{r['id']:r for r in records(u/'evaluation'/a/'responses')} for a in ['W-off','GAD-off']}; w,g=arms.values();assert set(w)==set(g)
 return dict(total=len(w),same_output_ids=sum(w[k]['output_ids']==g[k]['output_ids'] for k in w),same_raw_text=sum(w[k]['raw_generated']==g[k]['raw_generated'] for k in w),same_prediction_including_invalid=sum(w[k]['prediction']==g[k]['prediction'] for k in w),same_valid_prediction=sum(w[k]['valid'] and g[k]['valid'] and w[k]['prediction']==g[k]['prediction'] for k in w))
def safefile(p):
 with path(p).open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));offset=8+n
 return h,offset
def tensor(p,key,h,offset):
 d=h[key];start,end=d['data_offsets']
 with path(p).open('rb') as f:f.seek(offset+start);b=f.read(end-start)
 if d['dtype']=='F32':a=array.array('f');a.frombytes(b);return a
 if d['dtype']=='BF16':return [struct.unpack('<f',b'\0\0'+b[i:i+2])[0] for i in range(0,len(b),2)]
 if d['dtype']=='F16':return [x[0] for x in struct.iter_unpack('<e',b)]
 raise ValueError(d['dtype'])
def adapter_delta(a,b):
 ha,oa=safefile(a);hb,ob=safefile(b);keys=set(ha)-{'__metadata__'};assert keys==set(hb)-{'__metadata__'}
 groups={}; per=[]
 for k in sorted(keys):
  va=tensor(a,k,ha,oa);vb=tensor(b,k,hb,ob);assert len(va)==len(vb)
  tag='A' if '.lora_A.' in k else 'B';s=groups.setdefault(tag,dict(elements=0,changed=0,l2_before_sq=0.,l2_after_sq=0.,delta_sq=0.,max_abs_delta=0.,dtypes=set()))
  n=sum(x!=y for x,y in zip(va,vb));d2=sum((float(x)-float(y))**2 for x,y in zip(va,vb));a2=sum(float(x)**2 for x in va);b2=sum(float(x)**2 for x in vb);mx=max(abs(float(x)-float(y)) for x,y in zip(va,vb))
  s['elements']+=len(va);s['changed']+=n;s['l2_before_sq']+=a2;s['l2_after_sq']+=b2;s['delta_sq']+=d2;s['max_abs_delta']=max(s['max_abs_delta'],mx);s['dtypes'].update([ha[k]['dtype'],hb[k]['dtype']])
  per.append(dict(key=k,shape=ha[k]['shape'],dtype=ha[k]['dtype'],changed=n,elements=len(va),l2_before=math.sqrt(a2),l2_after=math.sqrt(b2),delta_l2=math.sqrt(d2),max_abs_delta=mx))
 for s in groups.values():
  s['dtypes']=sorted(s['dtypes']);s['delta_l2']=math.sqrt(s.pop('delta_sq'));s['l2_before']=math.sqrt(s.pop('l2_before_sq'));s['l2_after']=math.sqrt(s.pop('l2_after_sq'));s['relative_delta']=s['delta_l2']/s['l2_before'] if s['l2_before'] else None;s['changed_fraction']=s['changed']/s['elements']
 return dict(before=str(a.relative_to(ROOT)),after=str(b.relative_to(ROOT)),before_sha256=sha(a),after_sha256=sha(b),groups=groups,tensors=per)
def scored(task,scorer,row,r,gold):
 parsed=q.classify(r['raw_generated'],'off',scorer,row.get('metadata',{}),r['finish_reason']);correct=bool(score_prediction(parsed,dict(kind='qa',scorer=scorer),gold[str(row['id'])],row.get('metadata',{}))[0])
 return dict(valid=bool(parsed['valid']),correct=correct,capped=r['finish_reason']=='length',length=len(r['response_ids']),prediction=parsed['prediction'],reason=parsed.get('reason'))
def sample_summary(items):
 return dict(total=len(items),correct=sum(r['correct'] for r in items),valid=sum(r['valid'] for r in items),capped=sum(r['capped'] for r in items),length=stats(r['length'] for r in items),by_family={k:dict(total=len(v),correct=sum(r['correct'] for r in v),valid=sum(r['valid'] for r in v),capped=sum(r['capped'] for r in v)) for k,v in group(items,'family').items()})
def group(items,key):
 d=collections.defaultdict(list)
 for r in items:d[r[key]].append(r)
 return d

def audit_unit(task,model,skip_tensors=False):
 u=RUN/'training'/task/model;work=u/'training';p=load(u/'protocol.json');m=load(p['manifest_path']);train=rows(m['phases']['train']['queries']);by={str(r['id']):r for r in train};goldspec=m['phases']['train']['gold'];assert sha(goldspec['path'])==goldspec['sha256'];gold=load(goldspec['path']);scorer=m['phases']['train']['scorer'];teacher_manifest=load(p['teacher_manifest_path']);teacher={str(r['id']):r for r in rows(teacher_manifest['records_file'])};field='ability' if task=='USTBench' else 'question_type'
 sft=records(work/'sft');warm=records(work/'warmup');pol=records(work/'policy');assert len(sft)==200 and len(warm)==200 and len(pol)==50
 out=dict(task=task,model=model,protocol_sha256=sha(u/'protocol.json'),selection=load(u/'selected.json'),complete=load(work/'complete.json'),evaluation_agreement=eval_compare(u),sft_loss=curve([r['loss'] for r in sft]),sft_gradient_norm=stats(r['gradient_norm'] for r in sft),actor_gradient_norm=stats(r['actor']['gradient_norm'] for r in pol),actor_loss=curve([r['actor']['loss'] for r in pol]),D_warm_loss=curve([r['loss'] for r in warm]),D_policy_loss=curve([r['discriminator']['loss'] for r in pol]),D_gradient_norm=stats([r['gradient_norm'] for r in warm]+[r['discriminator']['gradient_norm'] for r in pol]),validation={name:load(work/'validation'/name/'scores.json') for name in ['W','policy0025','policy0050']},gpu_preflight=load(u/'gpu-preflight.json'))
 audit=load(u/'token-audit.json')['train'];out['tokens']=dict(prompt=stats(r['prompt_tokens'] for r in audit),teacher_with_eos=stats(r['teacher_tokens'] for r in audit),teacher_with_eos_gt256=sum(r['teacher_tokens']>256 for r in audit),max_total=max(r['prompt_tokens']+r['teacher_tokens'] for r in audit),input_truncated=False)
 allwarm=[];allpol=[];groups=[];details=[];teachwrong=[]
 for item in warm:
  row=by[str(item['id'])];r=scored(task,scorer,row,item['fresh_responses'][0],gold);r['family']=row['metadata'][field];allwarm.append(r)
 for i,item in enumerate(pol,1):
  row=by[str(item['id'])];members=[]
  for j,r in enumerate(item['fresh_responses']):
   v=scored(task,scorer,row,r,gold);v.update(family=row['metadata'][field],reward=item['rewards'][j],advantage=item['advantages'][j],id=str(row['id']),step=i,index=j);members.append(v);allpol.append(v)
  top=max(range(8),key=lambda j:item['rewards'][j]);correct=[v for v in members if v['correct']];incorrect=[v for v in members if not v['correct']]
  groups.append(dict(step=i,id=str(row['id']),correct=len(correct),valid=sum(v['valid'] for v in members),capped=sum(v['capped'] for v in members),unique_token_sequences=len({tuple(r['response_ids']) for r in item['fresh_responses']}),unique_parsed_predictions=len({json.dumps(v['prediction'],sort_keys=True) for v in members}),reward_std=statistics.stdev(item['rewards']),advantage_std=statistics.stdev(item['advantages']),top_reward_correct=members[top]['correct'],reward_correct_minus_wrong=(statistics.mean(v['reward'] for v in correct)-statistics.mean(v['reward'] for v in incorrect)) if correct and incorrect else None))
  t=teacher[str(row['id'])];tc=bool(score_prediction(t,dict(kind='qa',scorer=scorer),gold[str(row['id'])],row.get('metadata',{}))[0])
  if not tc:teachwrong.append(dict(id=str(row['id']),step=i,student_correct=len(correct),mean_reward=statistics.mean(item['rewards']),teacher_prediction=t['prediction'],same_teacher_prediction=sum(v['prediction']==t['prediction'] for v in members)))
 out['W_train_saved_sampling']=dict(**sample_summary(allwarm),measurement='one pre-existing T=.8/max256 sampled answer per Train prompt, W fixed during D warmup; NOT greedy fit or fresh independent evaluation')
 out['policy_saved_sampling']=sample_summary(allpol);out['policy_groups']=groups;out['teacher_wrong_policy_groups']=teachwrong
 out['reward_alignment']=dict(reward_correct_corr=corr([v['reward'] for v in allpol],[int(v['correct']) for v in allpol]),reward_length_corr=corr([v['reward'] for v in allpol],[v['length'] for v in allpol]),advantage_correct_corr=corr([v['advantage'] for v in allpol],[int(v['correct']) for v in allpol]),advantage_length_corr=corr([v['advantage'] for v in allpol],[v['length'] for v in allpol]),mixed_correctness_groups=sum(g['reward_correct_minus_wrong'] is not None for g in groups),correct_reward_lower_in_mixed_groups=sum(g['reward_correct_minus_wrong'] is not None and g['reward_correct_minus_wrong']<0 for g in groups),top_reward_correct_groups=sum(g['top_reward_correct'] for g in groups),capped_positive_advantage=sum(v['capped'] and v['advantage']>0 for v in allpol),incorrect_positive_advantage=sum(not v['correct'] and v['advantage']>0 for v in allpol),invalid_positive_advantage=sum(not v['valid'] and v['advantage']>0 for v in allpol))
 out['policy_member_audit']=allpol;out['policy_reward_std']=stats(g['reward_std'] for g in groups)
 selected=path(out['selection']['checkpoint']);W=work/'checkpoints/W0200';wm=load(W/'manifest.json');sm=load(selected/'manifest.json')
 for folder,cm in [(W,wm),(selected,sm)]:
  for role in ['actor','reference','discriminator']:
   rel=role+'/adapter_model.safetensors';assert sha(folder/rel)==cm['files'][rel]
 out['role_checks']=dict(W_actor_reference_equal=wm['role_hashes']['actor']==wm['role_hashes']['reference'],selected_reference_unchanged=wm['role_hashes']['reference']==sm['role_hashes']['reference'],W_actor_selected_actor_different=wm['role_hashes']['actor']!=sm['role_hashes']['actor'],W_reload=load(u/'W-reload.json'),selected_reload=load(u/'selected-reload.json'),adapter_config=load(W/'actor/adapter_config.json'))
 if not skip_tensors:
  out['adapter_W_to_selected']=adapter_delta(W/'actor/adapter_model.safetensors',selected/'actor/adapter_model.safetensors')
  out['D_adapter_W_to_selected']=adapter_delta(W/'discriminator/adapter_model.safetensors',selected/'discriminator/adapter_model.safetensors')
 out['norm_clipping_counts']=dict(SFT_preclip_norm_gt1=sum(r['gradient_norm']>1 for r in sft),actor_preclip_norm_gt1=sum(r['actor']['gradient_norm']>1 for r in pol))
 if not skip_tensors:
  import re
  keys=[x['key'] for x in out['adapter_W_to_selected']['tensors']];out['actual_adapter_scope']=dict(layers=sorted({int(re.search(r'layers\.(\d+)\.',k)[1]) for k in keys}),modules=sorted({k.split('.')[-3] for k in keys}),tensors=len(keys),actor_parameters=sum(v['elements'] for v in out['adapter_W_to_selected']['groups'].values()))
 out['actor_hash_changed_groups']=sum(r['actor']['before']!=r['actor']['after'] for r in pol)
 out['evidence_not_logged']=['per-answer-token gradients','per-group explicit KL and clip ratio','W teacher-forced NLL measured after full epoch on same Train200','base same-condition sampled Train200','fresh greedy Train200 fit','initial adapter A snapshot','D teacher score and BT margin per pair']
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--skip-tensors',action='store_true');ap.add_argument('--output',default='reports/overnight-onpolicy-2026-10-25/diagnosis/training-replay.json');a=ap.parse_args()
 if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 started=time.monotonic();out=dict(schema='posthoc-training-audit-v1',source_sha256=sha(__file__),new_api_calls=0,gpu_used=False,base_model_loaded=False,train_gold_posthoc_only=True,units=[])
 for task in ['USTBench','SGQA']:
  for model in ['08','2b']:
   v=audit_unit(task,model,a.skip_tensors);out['units'].append(v);print(json.dumps({k:v[k] for k in ['task','model','evaluation_agreement','tokens','W_train_saved_sampling','policy_saved_sampling','reward_alignment']},ensure_ascii=False),flush=True)
 out['wall_seconds']=time.monotonic()-started;p=path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print('SAVED',p,'sha256',sha(p),'seconds',out['wall_seconds'])
if __name__=='__main__':main()



