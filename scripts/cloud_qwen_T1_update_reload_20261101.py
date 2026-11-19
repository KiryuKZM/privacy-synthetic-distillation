"""One-use cached fixed-prefix public T=1 cloud-Max IPD update/reload smoke.
No API, dataset, Test, local teacher, repeat fit or weight downloads. Complete K5 support.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,ast,copy,fcntl,gc,hashlib,json,math,os,resource,signal,subprocess,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
INHERITED_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES')
import three_method_native_ipd_v4_20261101 as previous
e=previous.e;m=e.m
from privacy_synth.ipd import decide_token,source_split_kl
if INHERITED_CUDA is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=INHERITED_CUDA
P=R/'runs/2026-11-01-cloud-qwen-T1-update-reload-v1';PROBE=R/'runs/2026-11-01-cloud-qwen-probability-generic-v1'
UNIT='codex-cloud-qwen-T1-update-reload-20261101-v1.service';STOP=False
load,bind,once,atomic=m.load,m.bind,m.once,m.atomic
OLD_STOP=m.stopped;BASE_CHECK=e.check_stop

def stopped():return STOP or (P/'STOP').exists() or OLD_STOP()
m.stopped=stopped

def stop(*_):
 global STOP;STOP=True;e.stop()

def check():
 if stopped():raise InterruptedError('Own cloud T1 STOP')
 BASE_CHECK()
e.check_stop=check
m.P=P;m.S=P;e.P=P
# Change only the former union16-width assertion, retaining the exact checked tensors/loss/update.
source=R/'scripts/three_method_native_ipd_20261101.py';node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='support_tensors');original=copy.deepcopy(node)
WIDTH_BEFORE=ast.parse('16 <= len(ids) <= 32',mode='eval').body
WIDTH_AFTER=ast.parse('len(ids)==5 and trace["support_kind"]=="complete_teacher_top5"',mode='eval').body
class ExactWidthOnly(ast.NodeTransformer):
 def __init__(self,before,after):self.before=ast.dump(before,include_attributes=False);self.after=after;self.count=0
 def generic_visit(self,n):
  if ast.dump(n,include_attributes=False)==self.before:self.count+=1;return copy.deepcopy(self.after)
  return super().generic_visit(n)
change=ExactWidthOnly(WIDTH_BEFORE,WIDTH_AFTER);node=change.visit(node);assert change.count==1
undo=ExactWidthOnly(WIDTH_AFTER,WIDTH_BEFORE);inverse=undo.visit(copy.deepcopy(node));assert undo.count==1
assert ast.dump(inverse,include_attributes=False)==ast.dump(original,include_attributes=False)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(Path(__file__))+'::only_teacher5_support_width','exec'),e.__dict__)


def feedback(response,mapping):
 assert response['status']=='RESPONDED' and response['index']==0 and response['model_returned']=='qwen3.8-max' and not response['thinking_content_present']
 positions=response['probability']['top5'];assert len(positions)==1
 point=positions[0];top=point['top_logprobs'];assert len(top)==5
 candidates=[v for v in mapping['observations'] if v['call']==0];bybytes={tuple(v['bytes']):v for v in candidates}
 result=[]
 for item in top:
  proof=bybytes.get(tuple(item['bytes']));assert proof and proof['unique_single_student_event'] and proof['exact_single_ID_decoded_bytes']
  assert len(proof['student_ID_candidates'])==1 and math.isfinite(item['logprob']);result.append(dict(id=proof['student_ID_candidates'][0],bytes=item['bytes'],logprob=item['logprob'],token=item['token']))
 assert len({x['id'] for x in result})==5
 emitted=bybytes.get(tuple(point['bytes']));assert emitted and emitted['unique_single_student_event'] and emitted['exact_single_ID_decoded_bytes']
 return dict(support=result,emitted=emitted['student_ID_candidates'][0],emitted_bytes=point['bytes'],source='actual_one_API_policy_draw_from_same_fixed_prefix; not_argmax_or_Top5_resample')


def verify(out):
 assert Path(out).resolve()==(P/'student').resolve(),'Only separate new smoke namespace'
 g=load(P/'root-decision.json');p=load(out/'protocol.json');assert g['approved'] and g['source']==bind(Path(__file__)) and g['protocol']==bind(out/'protocol.json')
 for b in g['bindings']:assert bind(R/b['path'])==b
 assert p['actual_update_cap']==p['trajectory_positions']==1 and p['support_K']==5 and p['stage']=='cloud_max_T1_smoke'
 for name,info in p['base_asset_stat'].items():
  st=(Path(p['base'])/name).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==tuple(info)
 return p

def setup(out,checkpoint):
 fn=m.core.seeded_setup(m.native,verify);fn.__globals__['formal_setup_probe']=m.setup_probe
 return fn(out,checkpoint)

def prepare():
 out=P/'student';proof=load(P/'CPU-proof.json');assert proof['status']=='CLOUD_T1_CPU_CONTROLS_PASS' and proof['source']==bind(Path(__file__))
 # The originally first registered W is fixed independently of exposed Test outcomes.
 w=R/'runs/2026-10-31-new-generator-three-method-comparison/students-v2/seed01/2026102751/warm';wt=load(w/'fit-terminal.json');wp=load(w/'protocol.json');cm=m.lib.verify_checkpoint(w,Path(wt['checkpoint']))
 assert load(w/'independent-reload.json')['status']=='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS'
 raw=load(PROBE/'call-0.json');mapping=load(PROBE/'observed-token-byte-map.json');f=feedback(raw,mapping);bodies=load(PROBE/'requests2-frozen.json')['bodies'];body=bodies[0]
 assert body['messages']==[dict(role='user',content='Complete the phrase with one English word: The sky is')]
 assert hashlib.sha256(json.dumps(body,separators=(',',':')).encode()).hexdigest()==raw['body_sha256']
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(wp['base'],local_files_only=True,trust_remote_code=False);rendered=tok.apply_chat_template(body['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);prefix=tok.encode(rendered,add_special_tokens=False)
 assert prefix and len(prefix)<1024 and tok.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)==rendered
 for x in f['support']:assert tok.decode([x['id']],skip_special_tokens=False,clean_up_tokenization_spaces=False).encode()==bytes(x['bytes'])
 once(P/'public-fixture.json',dict(original_probe_index=0,API_response=bind(PROBE/'call-0.json'),request_body=body,rendered_student_prefix=rendered,student_prompt_ids=prefix,teacher_feedback=f,student_template_separate_from_hidden_hosted_template=True,common_condition='same exact logical public role/content initial fixture; own documented chat renderers; no generated history',one_use_cached_teacher_sample=True))
 protocol=dict(stage='cloud_max_T1_smoke',base=wp['base'],base_asset_stat=wp['base_asset_stat'],seed=2026102751,student_sampling_seed=2026102751,lr=1e-4,generator='first_registered_W_source_only_not_new_generator_fit',actual_update_cap=1,trajectory_positions=1,support_K=5,source_W=bind(w/'fit-terminal.json'),source_W_manifest=bind(Path(wt['checkpoint'])/'manifest.json'),source_W_reload=bind(w/'independent-reload.json'),source_W_roles=cm['role_hashes'],source_W_path=str(w.relative_to(R)),fixture=bind(P/'public-fixture.json'),teacher_cache_reused_once_at_same_fixed_prefix=True,teacher_sample_actual_API_emission_not_Top5_argmax=True,no_full_native_ID_or_multi_step_or_quality_claim=True,API_calls_new=0,old24_models_Test_and_scores_unchanged=True)
 once(out/'protocol.json',protocol)
 pins=[Path(__file__),P/'CPU-proof.json',P/'public-fixture.json',PROBE/'actual-result-review.json',PROBE/'observed-token-byte-map.json',R/'src/privacy_synth/ipd.py',R/'scripts/three_method_native_ipd_20261101.py',R/'scripts/three_method_native_student_v2_20261101.py',R/'reports/cloud-qwen-ipd-prefix-event-qualification-20261101.md',w/'independent-reload.json']
 once(P/'root-decision.json',dict(approved=True,actor='Root',source=bind(Path(__file__)),protocol=bind(out/'protocol.json'),bindings=[bind(x) for x in pins],authority='Existing active latest-cloud-Qwen genuine IPD goal and delegated in-scope decisions; additional public one-position engineering qualification, not old-model retraining or new utility experiment',unit=UNIT,maximum_new_optimizer_updates=1,maximum_reload=1,API_calls=0,no_teacher_weights_or_protected_data_or_Test=True,original_total_USD500_nonadditional=True,resources='singleGPUlock/sharedCPU4host14GiBnoSwap/CUDA9/startWin30run20/GPU11util10quiet20/temp80/disk20/output3GiB',maximum_seconds=900))
 print('CLOUD_T1_ONE_UPDATE_AND_EXACT_RELOAD_ROOT_BOUND',flush=True)


def initialize(out,checkpoint):
 torch,tok,model,head=setup(out,checkpoint);torch.set_num_threads(4)
 assert len(tok)==248077
 return torch,tok,model,head


def fit():
 out=P/'student';p=verify(out);check();assert not (out/'fit-started.json').exists(),'Unknown fit never replayed'
 m.preflight();frame=m.snapshot(True);e.BASELINE_FREE=frame['GPU']['free_MiB'];once(out/'fit-started.json',dict(epoch=time.time(),source=bind(Path(__file__)),API_cache=bind(PROBE/'call-0.json')))
 begin=time.monotonic();folder=R/p['source_W_manifest']['path'];folder=folder.parent;torch,tok,model,head=initialize(out,folder);m.lib.activate(model,head,'actor')
 assert {k:m.lib.role_hash(model,k) for k in p['source_W_roles']}==p['source_W_roles']
 oldrng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True);torch.set_rng_state(oldrng['cpu']);torch.cuda.set_rng_state_all(oldrng['cuda'])
 opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=p['lr'],weight_decay=0,foreach=False);assert not opt.state
 initial={k:m.lib.role_hash(model,k) for k in ('actor','reference','discriminator')};head_before=m.lib.nested_hash(head.state_dict());once(out/'initial-role-hashes.json',dict(roles=initial,head=head_before,fresh_optimizer=True,W_optimizer_restored=False))
 m.verify=verify;m.save(out,0,0,torch,model,head,opt);e.snapshot();fixture=load(P/'public-fixture.json');q=fixture['teacher_feedback'];prefix=fixture['student_prompt_ids'];model.eval();stepper=e.StudentStepper(torch,model)
 sample_begin=time.monotonic();sv=stepper.score(prefix);sk=sv[:248077].topk(5).indices.tolist();rng=torch.Generator(device='cuda').manual_seed(p['student_sampling_seed']);sp=int(torch.multinomial(sv[:248077].softmax(-1),1,generator=rng));tk=[x['id'] for x in q['support']]
 teacher_used=q['emitted'] if sp not in tk else None;choice=decide_token('S',sp,teacher_used,set(sk),set(tk));support=sorted(tk);teacher_scores={x['id']:x['logprob'] for x in q['support']};idx=torch.tensor(support,device='cuda')
 trace=dict(prompt_ids=prefix,actor_before=initial['actor'],support_kind='complete_teacher_top5',legacy_field_union_ids_is_packing_only_not_union=True,teacher_probability_kind='actual_provider_logprobs_under_neutral_requested_transforms',teacher_model='qwen3.8-max',one_use_API_teacher_draw=q['emitted'],teacher_draw_consumed=teacher_used is not None,one_use_cached_API=fixture['API_response'],student_sampling_seed=p['student_sampling_seed'],tokens=[dict(position=0,driver_before='S',student_proposal=sp,teacher_proposal=teacher_used,student_topk_ids=sk,teacher_topk_ids=tk,union_ids=support,student_union_logits=sv[idx].detach().cpu().tolist(),teacher_union_logits=[teacher_scores[k] for k in support],**asdict(choice))],events={choice.event:1},sources={choice.source:1},actual_raw_teacher_feedback=True,teacher_final_text_not_used=True,generated_history_reencoding_steps=0)
 stepper.close();del sv;torch.cuda.empty_cache();once(P/'teacher-cache-consumed-once.json',dict(API_response=fixture['API_response'],position=0,emitted_teacher_draw=q['emitted'],actual_used=teacher_used is not None,student_proposal=sp,epoch=time.time()));once(out/'trace.json',trace)
 check();e.snapshot();once(out/'update.started.json',dict(actor_before=initial['actor'],epoch=time.time(),source=choice.source));metric=e.update(torch,model,head,opt,trace);torch.cuda.synchronize()
 assert metric['optimizer_updated'] and metric['actor_before']!=metric['actor_after'] and metric['gradient_norm']>0
 assert all(m.lib.role_hash(model,k)==initial[k] for k in ('reference','discriminator')) and m.lib.nested_hash(head.state_dict())==head_before
 once(out/'update.json',dict(metric,scheduled=1,actual=1,epoch=time.time()));checkpoint,cm=m.save(out,1,1,torch,model,head,opt);last=e.snapshot()
 once(out/'fit-terminal.json',dict(status='PUBLIC_T1_ACTUAL_CLOUD_PROBABILITY_UPDATE_RELOAD_PENDING',scheduled=1,actual_updates=1,checkpoint=str(checkpoint),manifest=bind(checkpoint/'manifest.json'),initial_roles=initial,final_actor=metric['actor_after'],source=choice.source,event=choice.event,setup_and_fit_seconds=time.monotonic()-begin,sampling_update_checkpoint_wall_seconds=time.monotonic()-sample_begin,CPU_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CUDA_allocated_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),last_resource=last,teacher_API_calls_new=0,epoch=time.time()))
 print('ACTUAL_CLOUD_T1_UPDATE',choice.source,choice.event,'GRAD',metric['gradient_norm'],flush=True)


def reload():
 out=P/'student';p=verify(out);check();assert not (out/'reload-started.json').exists(),'Unknown reload never replayed';term=load(out/'fit-terminal.json');folder=Path(term['checkpoint']);m.preflight();e.BASELINE_FREE=m.snapshot(True)['GPU']['free_MiB'];once(out/'reload-started.json',dict(epoch=time.time(),manifest=bind(folder/'manifest.json')));tick=time.monotonic();torch,tok,model,head=initialize(out,folder);m.lib.activate(model,head,'actor');opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=p['lr'],weight_decay=0,foreach=False);cm=m.lib.restore(torch,model,head,dict(actor=opt),out,folder);rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
 assert torch.equal(torch.get_rng_state(),rng['cpu']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda']))
 steps=[float(v['step']) for v in opt.state.values() if 'step' in v];assert steps and all(x==1 for x in steps) and cm['step']==cm['actual_updates']==1
 assert cm['role_hashes']['actor']==term['final_actor'];frame=e.snapshot()
 once(out/'independent-reload.json',dict(status='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS',manifest=bind(folder/'manifest.json'),step=1,actual_updates=1,optimizer_state_steps=steps,roles=cm['role_hashes'],reload_seconds=time.monotonic()-tick,no_new_optimizer_updates=True,resource=frame,epoch=time.time()));print('CLOUD_T1_EXACT_RELOAD_PASS',flush=True)


def run():
 name=next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));assert Path(name).name==UNIT and os.environ.get('INVOCATION_ID') and os.getuid()!=0,'Exact actual service owner required'
 verify(P/'student');check();once(P/'started.json',dict(epoch=time.time(),invocation=os.environ.get('INVOCATION_ID'),source=bind(Path(__file__)),actions=['fit','reload'],API_calls_new=0));begin=time.monotonic()
 for action in ('fit','reload'):
  check();cmd=[sys.executable,str(Path(__file__)),action];child=subprocess.Popen(cmd,cwd=R);once(P/'children'/f'{action}.started.json',dict(pid=child.pid,arguments=cmd,epoch=time.time()));tick=time.monotonic()
  try:
   while child.poll() is None:
    check();assert time.monotonic()-tick<420;time.sleep(.5)
   once(P/'children'/f'{action}.json',dict(exit_code=child.returncode,seconds=time.monotonic()-tick,epoch=time.time()));assert child.returncode==0,'Preserve actual failed/unknown stage; no auto retry'
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=30)
 once(P/'terminal.json',dict(status='PUBLIC_CACHED_FIXED_PREFIX_T1_CLOUD_MAX_IPD_UPDATE_AND_EXACT_RELOAD_PASS',actual_updates=1,positions=1,fit=bind(P/'student/fit-terminal.json'),reload=bind(P/'student/independent-reload.json'),trace=bind(P/'student/trace.json'),teacher_API=bind(PROBE/'call-0.json'),API_calls_new=0,whole_cloud_IPD_or_utility_or_multi_step_complete=False,whole_goal_complete=False,seconds=time.monotonic()-begin,epoch=time.time()))
 print('CLOUD_MAX_T1_UPDATE_RELOAD_SMOKE_COMPLETE',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','fit','reload']);v=a.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if v.action=='prepare':prepare()
  elif v.action=='run':run()
  else:
   with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(fit if v.action=='fit' else reload)()
 except BaseException as ex:
  p=P/(v.action+'-failure.json')
  if not p.exists():once(p,dict(type=type(ex).__name__,message=str(ex),epoch=time.time(),no_automatic_replay=True))
  raise
