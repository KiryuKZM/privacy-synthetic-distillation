"""Bounded actual per-prefix cloud-Max K5 IPD2, one update and independent reload."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import argparse,ast,copy,fcntl,hashlib,json,math,os,resource,signal,sqlite3,ssl,subprocess,sys,time,urllib.error,urllib.request
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import cloud_qwen_T1_update_reload_20261101 as b
import qualify_cloud_qwen_prefix_replay_20261101 as q
api=q.old;e=b.e;m=b.m;load,bind,once,atomic=b.load,b.bind,b.once,b.atomic
P=R/'runs/2026-11-01-cloud-qwen-string-prefix-IPD2-v1';UNIT='codex-cloud-qwen-string-prefix-IPD2-20261101-v1.service';MODEL='qwen3.8-max'
b.P=P;b.UNIT=UNIT;m.P=P;m.S=P;e.P=P
phase_source=R/'scripts/three_method_native_student_v2_20261101.py'
phase_node=next(x for x in ast.parse(phase_source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='phase_bytes')
exec(compile(ast.fix_missing_locations(ast.Module(body=[phase_node],type_ignores=[])),str(phase_source)+'::unchanged_original_phase_bytes','exec'),m.__dict__)
e.base_snapshot=e.clone(m.snapshot,P=P,check_stop=b.check)
assert e.base_snapshot.__globals__['P']==e.snapshot.__globals__['P']==m.snapshot.__globals__['P']==P
assert m.phase_bytes.__globals__['S']==P and b.reload.__globals__['P']==P

# Reuse exact established ByteLevel/added-event table builder with actual current tokenizer.
f=R/'scripts/map_cloud_qwen_observed_bytes_20261101.py';tree=ast.parse(f.read_text());oldfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');nodes=[];taking=False
for n in oldfn.body:
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='backend' for x in n.targets):taking=True
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='observed' for x in n.targets):break
 if taking:nodes.append(copy.deepcopy(n))
assert nodes
fn=ast.parse('def make_lookup(tok):\n return None').body[0];fn.body=nodes+[ast.parse('return lookup').body[0]];exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(f)+'::unchanged_event_builder','exec'),globals())

def event(item,tok,lookup):
 raw=bytes(item['bytes']);ids=lookup.get(raw,[]);assert len(ids)==1 and 0<=ids[0]<248077,'Unqualified or ambiguous byte event'
 i=ids[0]
 if i in set(tok.all_special_ids):
  assert i==tok.eos_token_id and item['token']=='<|im_end|>' and raw==b'<|im_end|>','No unsupported control alias'
  return dict(id=i,kind='TURN_END',bytes=item['bytes'],provider_label=item['token'],contract='official end-of-turn application event; hosted ID/emission equivalence unclaimed')
 return dict(id=i,kind='BYTES',bytes=item['bytes'],provider_label=item['token'])

def messages(prefix):return [dict(role='user',content=q.USER),dict(role='assistant',content=prefix,partial=True)]
def render(tok,prefix):return tok.apply_chat_template(messages(prefix),tokenize=False,add_generation_prompt=False,continue_final_message=True,enable_thinking=False)
def advance(tok,prefix,ids,selected):
 if selected['kind']=='TURN_END':return prefix,list(ids),True
 suffix=bytes(selected['bytes']).decode('utf-8',errors='strict');assert suffix
 current=render(tok,prefix);following=render(tok,prefix+suffix)
 assert tok.encode(current,add_special_tokens=False)==ids and tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==current
 assert following==current+suffix and tok.encode(following,add_special_tokens=False)==ids+[selected['id']],'Actual selected event changes canonical student token boundary'
 return prefix+suffix,ids+[selected['id']],False

def body(prefix):return dict(model=MODEL,max_tokens=1,enable_thinking=False,stream=False,temperature=1,top_p=1,top_k=None,presence_penalty=0,repetition_penalty=1,logprobs=True,top_logprobs=5,messages=messages(prefix))
def validate_body(value,index):
 assert index in (0,1)
 expected=q.PREFIX if index==0 else load(P/'selected-0.json')['visible_prefix_after']
 if index:assert not load(P/'selected-0.json')['terminal']
 assert value==body(expected) and len(json.dumps(value).encode())<2048,'Only fixed public fixture plus actual selected event may leave'
 return True

def verify(out):
 assert Path(out).resolve()==(P/'student').resolve()
 g=load(P/'root-decision.json');p=load(out/'protocol.json');assert g['approved'] and g['source']==bind(Path(__file__)) and g['protocol']==bind(out/'protocol.json')
 for pin in g['bindings']:assert bind(R/pin['path'])==pin
 assert p['stage']=='cloud_max_text_prefix2_smoke' and p['trajectory_position_cap']==2 and p['actual_update_cap']==1 and p['support_K']==5
 for name,info in p['base_asset_stat'].items():
  st=(Path(p['base'])/name).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==tuple(info)
 return p
b.verify=verify;m.verify=verify

def prepare():
 proof=load(P/'CPU-proof.json');assert proof['status']=='CLOUD_TEXT_PREFIX2_CPU_CONTROLS_PASS' and proof['source']==bind(Path(__file__)) and proof['tests_passed']>=10
 review=load(P/'preparation/source-review.json');assert review['status']=='PASS' and review['source']==bind(Path(__file__))
 proposed=load(P/'proposed-protocol.json');assert proposed['max_new_HTTP']==2 and proposed['max_new_parameter_updates']==1
 w=R/'runs/2026-10-31-new-generator-three-method-comparison/students-v2/seed01/2026102751/warm';wt=load(w/'fit-terminal.json');wp=load(w/'protocol.json');cm=m.lib.verify_checkpoint(w,Path(wt['checkpoint']));assert load(w/'independent-reload.json')['status']=='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS'
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(wp['base'],local_files_only=True,trust_remote_code=False);lookup=make_lookup(tok);eos=event(dict(token='<|im_end|>',bytes=list(b'<|im_end|>')),tok,lookup);initial=render(tok,q.PREFIX);ids=tok.encode(initial,add_special_tokens=False);assert ids and len(ids)<1024 and tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==initial
 once(P/'public-fixture.json',dict(user=q.USER,initial_assistant_prefix=q.PREFIX,student_rendered_initial=initial,student_prompt_ids=ids,turn_end=eos,hidden_teacher_template_equality_not_claimed=True,max_positions=2))
 with sqlite3.connect(api.LEDGER.resolve().as_uri()+'?mode=ro',uri=True) as db:
  db.execute('BEGIN');cfg=json.loads(db.execute('SELECT config FROM settings').fetchone()[0]);assert cfg['global_pico']==500*10**12
  count,spent=db.execute("SELECT COUNT(*),COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose!='historical-smoke'").fetchone();worst=2*api.cost(q.IN_RESERVE,1);combined=db.execute('SELECT COALESCE(SUM(pico),0) FROM calls').fetchone()[0]+db.execute('SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls').fetchone()[0];assert count+2<=111 and spent+worst<=10*10**12 and combined+worst<=500*10**12
  actual=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name IN ('sgqa_global500_qwen_insert','sgqa_global500_legacy_insert')").fetchall());expected={}
  import re
  for n in ast.walk(ast.parse(Path(api.__file__).read_text())):
   if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.lstrip().startswith('CREATE TRIGGER IF NOT EXISTS sgqa_global500_'):expected[re.search(r'EXISTS\s+(sgqa_global500_\w+)',n.value).group(1)]=n.value
  norm=lambda s:re.sub(r'\s+','',s.upper()).replace('IFNOTEXISTS','').rstrip(';');assert set(actual)==set(expected) and all(norm(actual[n])==norm(expected[n]) for n in expected)
 once(P/'budget-before.json',dict(prior_accounted_USD=combined/10**12,old_nonhistorical_qwen_calls=count,old111HTTP10USD_caps_unchanged=True,new_HTTP_cap=2,input_reservation_each=q.IN_RESERVE,output_cap_each=1,worst_new_reserved_USD=worst/10**12,original500_cumulative_nonadditional=True,invoice=False,epoch=time.time()))
 out=P/'student';protocol=dict(stage='cloud_max_text_prefix2_smoke',base=wp['base'],base_asset_stat=wp['base_asset_stat'],seed=2026102751,student_sampling_seed=2026102751,lr=1e-4,generator='first_originally_registered_W_readonly',actual_update_cap=1,trajectory_position_cap=2,support_K=5,source_W=bind(w/'fit-terminal.json'),source_W_manifest=bind(Path(wt['checkpoint'])/'manifest.json'),source_W_reload=bind(w/'independent-reload.json'),source_W_roles=cm['role_hashes'],fixture=bind(P/'public-fixture.json'),teacher='qwen3.8-max/off actual fresh stateless Partial(canonical UTF8 prefix) each position',not_claimed='hosted native-ID replay/immutable weights/full vocab/utility/privacy',early_terminal_and_invalid_no_refill=True)
 once(out/'protocol.json',protocol)
 pins=[phase_source,P/'preparation/source-review.json',Path(__file__),Path(b.__file__),Path(q.__file__),Path(api.__file__),f,P/'CPU-proof.json',P/'proposed-protocol.json',P/'public-fixture.json',P/'budget-before.json',R/'src/privacy_synth/ipd.py',R/'scripts/three_method_native_ipd_20261101.py',R/'reports/cloud-qwen-prefix-conditioned-ipd-review-20261101.md']
 once(P/'root-decision.json',dict(approved=True,actor='Root',source=bind(Path(__file__)),protocol=bind(out/'protocol.json'),bindings=[bind(x) for x in pins],authority='Active cloud-Qwen IPD goal, delegated in-scope decisions, additional finite public2-position engineering qualification before original approved synthetic mainline; no oldmodels retrained',maximum_HTTP=2,maximum_updates=1,maximum_reload=1,maximum_seconds=900,resources='singleGPUlock/sharedCPU4host14GiBnoSwap/CUDA9/startWin30run20/GPU11util10quiet20/temp80/disk20/output3GiB',new_USD_peak_cap=worst/10**12,original_budget_USD500_nonadditional=True,data='Hardcoded public numberlist plus actual selected events only; no project/real/Test/gold/stats',cheaper_alternative='Flash/local27B cannot qualify latest hosted Max policy',unit=UNIT,no_retries_unknown_replay_or_other_app_controls=True))
 print('CLOUD_TEXT_PREFIX2_ROOT_BOUND',flush=True)

def call_api(prefix,index):
 verify(P/'student');b.check();start=P/f'call-{index}.started.json';out=P/f'call-{index}.json';assert not start.exists() and not out.exists(),'No unknown API replay'
 value=body(prefix);validate_body(value,index);once(P/f'body-{index}.json',value);secret=api.key();raw=json.dumps(value,separators=(',',':')).encode();h=hashlib.sha256(raw).hexdigest();budget=api.MaxBudget(api.LEDGER);row=budget.reserve('cloud-max-text-prefix-IPD2-20261101-'+str(index),'ipd-public-text-prefix2',MODEL,h,q.IN_RESERVE,1);once(start,dict(index=index,ledger_id=row,body_sha256=h,epoch=time.time(),one_HTTP_only=True));tick=time.monotonic();usage=None;result=dict(index=index,requested=MODEL,ledger_id=row,body_sha256=h)
 try:
  req=urllib.request.Request(api.URL,data=raw,method='POST',headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'});opener=urllib.request.build_opener(api.NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
  with opener.open(req,timeout=60) as response:answer=json.loads(response.read(2**20));result['http_status']=response.status
  u=answer.get('usage',{})
  if all(type(u.get(k)) is int for k in ('prompt_tokens','completion_tokens')):usage={k:u[k] for k in ('prompt_tokens','completion_tokens')}
  assert answer.get('model')==MODEL,'Changed actual model identity';reasoning=bool((answer.get('choices') or [{}])[0].get('message',{}).get('reasoning_content'));result.update(status='THINKING_OFF_VIOLATED' if reasoning else 'RESPONDED',model_returned=answer['model'],thinking_content_present=reasoning,system_fingerprint=answer.get('system_fingerprint'),usage=usage,generic_response=answer)
 except urllib.error.HTTPError as ex:result.update(status='HTTP_ERROR',http_status=ex.code,raw_error_not_logged=True)
 except Exception as ex:result.update(status='MODEL_IDENTITY_CHANGED' if isinstance(ex,AssertionError) else 'CALL_OUTCOME_UNKNOWN',error_type=type(ex).__name__,no_retry=True)
 once(P/f'call-{index}.observed.json',dict(result,observed_before_budget_settlement=True,epoch=time.time()));budget.settle(row,usage);result.update(seconds=time.monotonic()-tick,epoch=time.time(),usage_known=usage is not None);once(out,result);atomic(P/'budget-live.json',budget.summary());assert result['status']=='RESPONDED','Actual call preserved; no retry'
 return result

def collect(torch,tok,model,fixture,actor_before,p):
 prefix=fixture['initial_assistant_prefix'];ids=fixture['student_prompt_ids'];initial_ids=list(ids);lookup=make_lookup(tok);model.eval();stepper=e.StudentStepper(torch,model);rng=torch.Generator(device='cuda').manual_seed(p['student_sampling_seed']);rng_start=rng.get_state().cpu().tolist();driver='S';tokens=[];sources=Counter();events=Counter();api_seconds=0.;terminal=False
 for index in range(p['trajectory_position_cap']):
  b.check();e.snapshot();current_render=render(tok,prefix);assert tok.encode(current_render,add_special_tokens=False)==ids;logits=stepper.score(ids);sk=logits[:248077].topk(5).indices.tolist();sp=int(torch.multinomial(logits[:248077].softmax(-1),1,generator=rng)) if driver=='S' else None
  response=call_api(prefix,index);api_seconds+=response['seconds'];point=q.position(response,0);assert len(response['generic_response']['choices'][0]['logprobs']['content'])==1;top=point['top_logprobs'];mapped=[event(x,tok,lookup) for x in top];assert len({x['id'] for x in mapped})==5;teacher_draw=event(point,tok,lookup);tk=[x['id'] for x in mapped];used_teacher=teacher_draw['id'] if driver=='T' or sp not in tk else None;choice=b.decide_token(driver,sp,used_teacher,set(sk),set(tk));selected=next((x for x in mapped if x['id']==choice.token_id),teacher_draw);assert selected['id']==choice.token_id
  nextprefix,nextids,terminal=advance(tok,prefix,ids,selected);support=sorted(tk);scores={x['id']:y['logprob'] for x,y in zip(mapped,top)};idx=torch.tensor(support,device='cuda');tokens.append(dict(position=index,driver_before=driver,student_proposal=sp,teacher_proposal=used_teacher,actual_teacher_API_draw=teacher_draw['id'],student_topk_ids=sk,teacher_topk_ids=tk,union_ids=support,student_union_logits=logits[idx].detach().cpu().tolist(),teacher_union_logits=[scores[i] for i in support],teacher_response=bind(P/f'call-{index}.json'),visible_prefix_before=prefix,mapped_complete_Top5=mapped,selected_event=selected,**asdict(choice)));sources[choice.source]+=1;events[choice.event]+=1
  once(P/f'selected-{index}.json',dict(visible_prefix_before=prefix,visible_prefix_after=nextprefix,canonical_student_ids_before=ids,canonical_student_ids_after=nextids,selected_event=selected,decision=asdict(choice),terminal=terminal,actual_API=bind(P/f'call-{index}.json')));prefix,ids,driver=nextprefix,nextids,choice.next_driver;del logits
  if terminal:break
 trace=dict(prompt_ids=initial_ids,actor_before=actor_before,support_kind='complete_teacher_top5',legacy_union_field_is_packing_only=True,tokens=tokens,sources=dict(sources),events=dict(events),teacher_policy='actual stateless Partial(canonical UTF8 string state) each reached position',actual_teacher_call_positions=len(tokens),max_positions=2,early_turn_end=terminal,sampling_RNG_start=rng_start,sampling_RNG_after=rng.get_state().cpu().tolist(),sampling_RNG_not_resumed_after_rollout=True,student_forward_calls=stepper.calls,no_forced_takeover_or_handback=True)
 stepper.close();torch.cuda.empty_cache();return trace,api_seconds

def fit():
 out=P/'student';p=verify(out);b.check();assert not (out/'fit-started.json').exists(),'No unknown fit replay';m.preflight();e.BASELINE_FREE=m.snapshot(True)['GPU']['free_MiB'];once(out/'fit-started.json',dict(epoch=time.time(),source=bind(Path(__file__))))
 begin=time.monotonic();folder=(R/p['source_W_manifest']['path']).parent;torch,tok,model,head=b.initialize(out,folder);m.lib.activate(model,head,'actor');assert {k:m.lib.role_hash(model,k) for k in p['source_W_roles']}==p['source_W_roles'];oldrng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True);torch.set_rng_state(oldrng['cpu']);torch.cuda.set_rng_state_all(oldrng['cuda']);opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=p['lr'],weight_decay=0,foreach=False);assert not opt.state
 initial={k:m.lib.role_hash(model,k) for k in ('actor','reference','discriminator')};headhash=m.lib.nested_hash(head.state_dict());once(out/'initial-roles.json',dict(roles=initial,head=headhash,fresh_optimizer=True));m.save(out,0,0,torch,model,head,opt);fixture=load(P/'public-fixture.json');trace,api_seconds=collect(torch,tok,model,fixture,initial['actor'],p);tokens=trace['tokens'];sources=trace['sources'];events=trace['events'];terminal=trace['early_turn_end']
 once(out/'trace.json',trace);b.check();e.snapshot();once(out/'update.started.json',dict(epoch=time.time(),actor_before=initial['actor'],actual_positions=len(tokens)));metric=e.update(torch,model,head,opt,trace);torch.cuda.synchronize();assert metric['optimizer_updated'] and metric['gradient_norm']>0 and metric['actor_before']!=metric['actor_after'];assert all(m.lib.role_hash(model,k)==initial[k] for k in ('reference','discriminator')) and m.lib.nested_hash(head.state_dict())==headhash
 once(out/'update.json',dict(metric,scheduled=1,actual=1,epoch=time.time()));checkpoint,cm=m.save(out,1,1,torch,model,head,opt);frame=e.snapshot();once(out/'fit-terminal.json',dict(status='PUBLIC_TEXT_PREFIX_IPD_ACTUAL_UPDATE_RELOAD_PENDING',actual_updates=1,positions=len(tokens),source_counts=dict(sources),event_counts=dict(events),early_turn_end=terminal,checkpoint=str(checkpoint),manifest=bind(checkpoint/'manifest.json'),initial_roles=initial,final_actor=metric['actor_after'],setup_fit_seconds=time.monotonic()-begin,API_seconds=api_seconds,actual_API_calls=len(tokens),CPU_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CUDA_allocated_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),resource=frame,epoch=time.time()));print('ACTUAL_TEXT_PREFIX_IPD_UPDATE',len(tokens),dict(sources),dict(events),flush=True)

def run():
 cg=next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));assert Path(cg).name==UNIT and os.environ.get('INVOCATION_ID') and os.getuid()!=0
 verify(P/'student');b.check();once(P/'started.json',dict(invocation=os.environ['INVOCATION_ID'],source=bind(Path(__file__)),epoch=time.time()));begin=time.monotonic()
 for action in ('fit','reload'):
  child=subprocess.Popen([sys.executable,str(Path(__file__)),action],cwd=R);once(P/'children'/f'{action}.started.json',dict(pid=child.pid,epoch=time.time()));tick=time.monotonic()
  try:
   while child.poll() is None:b.check();assert time.monotonic()-tick<420;time.sleep(.5)
   once(P/'children'/f'{action}.json',dict(exit_code=child.returncode,seconds=time.monotonic()-tick,epoch=time.time()));assert child.returncode==0,'Preserve failed/unknown stage; no retry'
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=30)
 fit_term=load(P/'student/fit-terminal.json');once(P/'terminal.json',dict(status='PUBLIC_STRING_PREFIX_K5_IPD_ACTUAL_UPDATE_AND_EXACT_RELOAD_PASS',positions=fit_term['positions'],actual_updates=1,source_counts=fit_term['source_counts'],event_counts=fit_term['event_counts'],fit=bind(P/'student/fit-terminal.json'),reload=bind(P/'student/independent-reload.json'),trace=bind(P/'student/trace.json'),actual_API_calls=fit_term['actual_API_calls'],seconds=time.monotonic()-begin,not_full_synthetic_training_or_same_Test_teacher=True,whole_goal_complete=False,epoch=time.time()));print('TEXT_PREFIX_IPD2_UPDATE_RELOAD_COMPLETE',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','fit','reload']);v=a.parse_args();signal.signal(signal.SIGTERM,b.stop);signal.signal(signal.SIGINT,b.stop)
 try:
  if v.action=='prepare':prepare()
  elif v.action=='run':run()
  else:
   with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(fit if v.action=='fit' else b.reload)()
 except BaseException as ex:
  fail=P/(v.action+'-failure.json')
  if not fail.exists():once(fail,dict(type=type(ex).__name__,message=str(ex)[:600],epoch=time.time(),no_automatic_retry=True))
  raise
