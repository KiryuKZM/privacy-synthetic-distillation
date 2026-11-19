"""Original six public-synthetic schedules, new hosted-Max text-prefix K5 IPD branches."""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import argparse,copy,fcntl,hashlib,json,os,resource,signal,sqlite3,ssl,subprocess,sys,time,urllib.error,urllib.request
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import cloud_qwen_string_prefix_IPD2_20261101 as h
import cloud_qwen_ipd_mainline_budget_20261101 as money
b=h.b;e=h.e;m=h.m;api=money.api;load,bind,once,atomic=h.load,h.bind,h.once,h.atomic
MAIN=R/'runs/2026-11-01-cloud-qwen-synthetic-IPD-mainline-v1';P=MAIN/'continuation-control-v2';OLD=R/'runs/2026-10-31-new-generator-three-method-comparison';UNIT='codex-cloud-qwen-synthetic-IPD-control-v2-20261101-v1.service';MODEL=money.MODEL
GENS=('seed01','seed02');SEEDS=(2026102751,2026102752,2026102753)
h.P=P;b.P=P;m.P=P;m.S=MAIN;e.P=P;e.base_snapshot=e.clone(m.snapshot,P=P,check_stop=b.check)
assert m.phase_bytes.__globals__['S']==MAIN and e.base_snapshot.__globals__['P']==e.snapshot.__globals__['P']==P and b.reload.__globals__['P']==P

# Two distinct existing special events, never a probability-column fold or literal suffix.
OriginalEvent=h.event

def map_event(item,tok,lookup):
 if item['token']=='<|endoftext|>' and bytes(item['bytes'])==b'<|endoftext|>':
  ids=lookup.get(b'<|endoftext|>',[]);expected=tok.convert_tokens_to_ids('<|endoftext|>')
  assert len(ids)==1 and ids[0]==expected and expected in tok.all_special_ids and expected!=tok.eos_token_id and tok.convert_ids_to_tokens(expected)=='<|endoftext|>'
  return dict(id=expected,kind='SEQUENCE_END',bytes=item['bytes'],provider_label=item['token'],contract='official EOD application event; distinct from im_end/EOT; hosted native-ID equivalence unclaimed')
 return OriginalEvent(item,tok,lookup)
h.event=map_event
import ast
advance_node=next(x for x in ast.parse(Path(h.__file__).read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='advance');original_advance=copy.deepcopy(advance_node);first_if=advance_node.body[0];assert isinstance(first_if,ast.If) and ast.unparse(first_if.test)=="selected['kind'] == 'TURN_END'"
before_test=copy.deepcopy(first_if.test);first_if.test=ast.parse("selected['kind'] in ('TURN_END','SEQUENCE_END')",mode='eval').body;inverse=copy.deepcopy(advance_node);inverse.body[0].test=before_test;assert ast.dump(inverse,include_attributes=False)==ast.dump(original_advance,include_attributes=False)
exec(compile(ast.fix_missing_locations(ast.Module(body=[advance_node],type_ignores=[])),str(Path(__file__))+'::two_distinct_terminal_kinds_only','exec'),h.__dict__)
BaseStopped=m.stopped
m.stopped=lambda:(MAIN/'STOP').exists() or BaseStopped()

def unit(gen,seed):
 assert gen in GENS and seed in SEEDS
 return P/'students'/gen/str(seed)/'cloud-ipd'
def frozen(path,value):
 if path.exists():assert load(path)==value;return
 once(path,value)
def verify(out):
 out=Path(out).resolve();p=load(out/'protocol.json');assert out==unit(p['generator'],p['seed']).resolve() and out.is_relative_to(P.resolve())
 g=load(P/'root-decision.json');assert g['approved'] and g['source']==bind(Path(__file__)) and bind(out/'protocol.json') in g['unit_protocols']
 for pin in g['bindings']:assert bind(R/pin['path'])==pin
 assert p['schedule_start']==(7 if (p['generator'],p['seed'])==('seed01',2026102751) else 1) and p['prior_slots']==p['schedule_start']-1
 assert p['stage']=='cloud_ipd' and p['lr']==1e-4 and p['maximum_optimizer_attempts']==50 and p['max_positions']==32 and p['support_K']==5 and p['lora']==dict(r=8,alpha=16,dropout=0)
 assert p['inputs']==bind(R/p['inputs']['path']) and p['W_manifest']==bind(R/p['W_manifest']['path'])
 for name,info in p['base_asset_stat'].items():
  st=(Path(p['base'])/name).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==tuple(info)
 return p
h.verify=verify;b.verify=verify;m.verify=verify

def ledger_snapshot():
 with sqlite3.connect(api.LEDGER.resolve().as_uri()+'?mode=ro',uri=True) as db:
  db.execute('BEGIN');snap={}
  for name in ('settings','calls','qwen_max_calls'):
   rows=db.execute('SELECT * FROM '+name+' ORDER BY rowid').fetchall();snap[name]=dict(rows=len(rows),sha256=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest())
  snap['original_global_triggers']=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name IN ('sgqa_global500_qwen_insert','sgqa_global500_legacy_insert')").fetchall());return snap

def prepare():
 proof=load(P/'preparation/worker-CPU-proof-v2.json');assert proof['source']==bind(Path(__file__)) and proof['status']=='MAINLINE_WORKER_CPU_PASS'
 review=load(P/'preparation/worker-source-review-v2.json');assert review['status']=='PASS' and review['source']==bind(Path(__file__))
 breview=load(MAIN/'preparation/budget-source-review.json');assert breview['status']=='PASS' and breview['source']==bind(Path(money.__file__))
 resume=load(P/'preparation/old-cursor-audit.json');assert resume['status']=='STOPPED_MAINLINE_ZERO_UPDATE_AND_KNOWN_API_CLOSURE_VERIFIED' and resume['old_actual_model_updates']==0 and resume['next_original_task']==7 and resume['unknown_API']==resume['inflight_API']==0
 original=load(MAIN/'preparation/original-six-bindings.json');assert original['maximum_scheduled_updates']==300 and original['original_requests576']==bind(R/original['original_requests576']['path'])
 wire=[json.loads(s) for s in (R/original['original_requests576']['path']).read_text().splitlines()];assert len(wire)==576 and len({x['id'] for x in wire})==576;by={x['id']:x for x in wire}
 from transformers import AutoTokenizer
 first=load(R/original['units'][0]['original_local_IPD_protocol']['path']);tok=AutoTokenizer.from_pretrained(first['base'],local_files_only=True,trust_remote_code=False);assert len(tok)==248077
 group_rows={}
 for gen in GENS:
  records=[u for u in original['units'] if u['generator']==gen];order=records[0]['fixed_task_ids'];assert len(order)==len(set(order))==50 and all(u['fixed_task_ids']==order for u in records);rows=[]
  assert bind(R/records[0]['source_corpus']['path'])==records[0]['source_corpus'];corpus=load(R/records[0]['source_corpus']['path']);bycorpus={x['id']:x for x in corpus['selected']}
  for identifier in order:
   x=by[identifier];messages=x['body']['messages'];assert x['generator']==gen and [y['role'] for y in messages]==['system','user'] and all(set(y)=={'role','content'} for y in messages) and messages==bycorpus[identifier]['messages']
   rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(rendered,add_special_tokens=False);assert ids and len(ids)+32<=16384 and tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==rendered
   rows.append(dict(id=identifier,messages=messages,student_base_rendered=rendered,student_prompt_ids=ids,original_public_wire_body_sha=x['body_sha256']))
  frozen(P/'inputs'/f'{gen}-fifty.json',rows);group_rows[gen]=rows
 protocols=[]
 for u in original['units']:
  gen,seed=u['generator'],u['seed'];assert bind(R/u['original_local_IPD_protocol']['path'])==u['original_local_IPD_protocol'];oldp=load(R/u['original_local_IPD_protocol']['path']);wman=R/u['source_W_manifest']['path'];wout=wman.parent.parent.parent;cm=m.lib.verify_checkpoint(wout,wman.parent);assert load(R/u['source_W_reload']['path'])['status']=='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS';assert bind(wman)==u['source_W_manifest']
  p=dict(generator=gen,seed=seed,stage='cloud_ipd',base=oldp['base'],base_asset_stat=oldp['base_asset_stat'],lr=1e-4,lora=dict(r=8,alpha=16,dropout=0),native_vocabulary=True,thinking=False,maximum_optimizer_attempts=50,max_positions=32,support_K=5,inputs=bind(P/'inputs'/f'{gen}-fifty.json'),post50_ids=u['fixed_task_ids'],W_manifest=u['source_W_manifest'],W_reload=u['source_W_reload'],W_roles=cm['role_hashes'],teacher=MODEL,teacher_policy='fresh actual hosted Partial(visible prefix); initial system/user only, not native-decode equivalence',terminal_missing_feedback_policy='strict provider terminal-unobserved: retain verified prefix loss, mark partial; zero prefix no update; no EOS/source invented',actual_unmappable_policy='retain entire trajectory as unqualified/no gradient/no refill',checkpoint_every=5,latest2full=True,no_oracle_or_real_gold_in_loss=True,no_Test_selection=True)
  first_unit=(gen,seed)==('seed01',2026102751);p.update(schedule_start=7 if first_unit else 1,prior_slots=6 if first_unit else 0,prior_API_calls=28 if first_unit else 0,prior_observed_closed_positions=18 if first_unit else 0,prior_interrupted_selected_positions=5 if first_unit else 0,prior_feedback_counts={'unqualified_actual_event':5,'interrupted_without_update':1} if first_unit else {},prior_events={'student_accepted':18} if first_unit else {},prior_sources={'S':18} if first_unit else {},control_map_version='two_distinct_terminal_events_and_zero_output_omitted_vector_v2')
  if first_unit:p.update(resume_manifest=resume['checkpoint'],resume_checkpoint_folder=resume['checkpoint_folder'],resume_checkpoint_unit=resume['checkpoint_unit'],resume_actual_updates=0,old_cursor_audit=bind(P/'preparation/old-cursor-audit.json'))
  frozen(unit(gen,seed)/'protocol.json',p);protocols.append(bind(unit(gen,seed)/'protocol.json'))
 if not (P/'preparation/budget-installed.json').exists():
  before=ledger_snapshot();budget=money.MainlineBudget(api.LEDGER);after=ledger_snapshot();assert before==after,'Pool registration changed old rows/settings/triggers';once(P/'preparation/budget-installed.json',dict(status='NEW_PURPOSE_POOL_REGISTERED_OLD_ROWS_SETTINGS_GLOBAL_TRIGGERS_BYTE_EXACT',source=bind(Path(money.__file__)),before=before,after=after,actual_summary=budget.mainline_summary(),new_API_calls=0,epoch=time.time()))
 pins=[Path(__file__),Path(h.__file__),Path(b.__file__),Path(money.__file__),Path(api.__file__),R/'scripts/three_method_native_student_v2_20261101.py',R/'scripts/three_method_native_ipd_20261101.py',R/'src/privacy_synth/ipd.py',P/'preparation/worker-CPU-proof-v2.json',P/'preparation/worker-source-review-v2.json',MAIN/'preparation/budget-CPU-proof.json',MAIN/'preparation/budget-source-review.json',P/'preparation/budget-installed.json',MAIN/'preparation/original-six-bindings.json',R/'reports/cloud-qwen-terminal-unobserved-prefix-contract-20261101.md',R/'reports/cloud-qwen-synthetic-mainline-readiness-20261101.md',P/'preparation/old-cursor-audit.json',R/'reports/cloud-qwen-two-terminal-event-contract-20261101.md',R/'reports/cloud-qwen-zero-output-omitted-logprobs-contract-20261101.md',MAIN/'root-decision.json',R/'scripts/cloud_qwen_synthetic_ipd_mainline_20261101.py']
 for gen in GENS:pins.append(P/'inputs'/f'{gen}-fifty.json')
 once(P/'root-decision.json',dict(approved=True,actor='Root',source=bind(Path(__file__)),bindings=[bind(x) for x in pins],unit_protocols=protocols,authority='User explicitly requires cloud latest Qwen genuineIPD within original500 and delegated all in-scope decisions; evidence-bound control/parser continuation starts original task7 after5invalid+1interrupted/0updates; other5groups fresh50; no old task replay, max294new updates, no old completedtraining repeat',new_update_cap=294,original_scheduled300=True,prior_untrained_slots6_preserved=True,same_existing_pool_not_reset=True,new_HTTP_cap=9600,new_pool_USD_cap400=True,original_total_USD500_nonadditional=True,teacher=MODEL,data='Only exact old authorized public-synthetic system/user messages plus actual generated prefix, no IDs/metadata/gold/protectedTest/stats',method='teacher-supportK5/stateless-text-prefix source-splitKL; oldnativeunionK16/localteacher controls preserved',resources='singleGPUlock/sharedCPU4host14GiBnoSwap/CUDA9/startWin30run20/free11util10quiet20/temp80/disk20/wholephase3GiB',unit=UNIT,no_other_app_closure_or_weight_download=True,cheaper_alternative='Local27B/Flash cannot replace requested latest cloudMax probabilities; public oracle is not a teacher probability source'))
 print('SIX_CLOUD_IPD_MAINLINE_ROOT_BOUND_SAME_CUMULATIVE500',flush=True)

def terminal_unobserved(result):
 if result.get('http_status')!=200 or result.get('status')!='RESPONDED' or result.get('model_returned')!=MODEL or result.get('thinking_content_present'):return False
 cs=result.get('generic_response',{}).get('choices');
 if not isinstance(cs,list) or len(cs)!=1:return False
 c=cs[0];msg=c.get('message',{});lp=c.get('logprobs',{})
 return c.get('finish_reason')=='stop' and msg.get('role')=='assistant' and msg.get('content')=='' and not any(msg.get(k) for k in ('tool_calls','function_call','refusal','reasoning_content')) and isinstance(lp,dict) and lp.get('content')==[]

OriginalTerminal=terminal_unobserved

def terminal_unobserved(result):
 if OriginalTerminal(result):return True
 if result.get('status')!='RESPONDED' or result.get('http_status')!=200 or result.get('model_returned')!=MODEL or result.get('thinking_content_present') or not result.get('usage_known'):return False
 usage=result.get('usage',{})
 if type(usage.get('completion_tokens')) is not int or usage['completion_tokens']!=0:return False
 cs=result.get('generic_response',{}).get('choices')
 if not isinstance(cs,list) or len(cs)!=1:return False
 c=cs[0];msg=c.get('message',{})
 return 'logprobs' not in c and c.get('finish_reason')=='stop' and isinstance(msg,dict) and msg.get('role')=='assistant' and msg.get('content')=='' and not any(msg.get(k) for k in ('tool_calls','function_call','refusal','reasoning_content'))

def make_body(messages,prefix):
 assert [x['role'] for x in messages]==['system','user'] and all(set(x)=={'role','content'} and isinstance(x['content'],str) for x in messages)
 msgs=copy.deepcopy(messages)
 if prefix:msgs.append(dict(role='assistant',content=prefix,partial=True))
 value=dict(model=MODEL,messages=msgs,max_tokens=1,enable_thinking=False,stream=False,temperature=1,top_p=1,top_k=None,presence_penalty=0,repetition_penalty=1,logprobs=True,top_logprobs=5);assert len(json.dumps(value).encode())<=131072;return value

def call_feedback(out,p,row,step,index,prefix,folder,budget):
 b.check();path=folder/f'call-{index}.json';start=folder/f'call-{index}.started.json';assert not path.exists() and not start.exists(),'Never replay unknown API position'
 assert row in load(R/p['inputs']['path']) and 1<=step<=50 and row['id']==p['post50_ids'][step-1] and 0<=index<32
 if index:
  previous=load(folder/f'selected-{index-1}.json');assert previous['visible_prefix_after']==prefix and not previous['terminal']
 else:assert prefix==''
 value=make_body(row['messages'],prefix);once(folder/f'body-{index}.json',value);raw=json.dumps(value,separators=(',',':')).encode();sha=hashlib.sha256(raw).hexdigest();tag=f"cloud-main-v1/{p['generator']}/{p['seed']}/step{step:03d}/task{hashlib.sha256(row['id'].encode()).hexdigest()[:12]}/pos{index:02d}"
 secret=api.key();record=budget.reserve(tag,money.PURPOSE,MODEL,sha);once(start,dict(tag=tag,ledger_id=record,body_sha256=sha,epoch=time.time(),one_HTTP_only=True));tick=time.monotonic();usage=None;result=dict(requested=MODEL,ledger_id=record,tag=tag,body_sha256=sha)
 try:
  req=urllib.request.Request(api.URL,data=raw,method='POST',headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'});opener=urllib.request.build_opener(api.NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
  with opener.open(req,timeout=60) as response:answer=json.loads(response.read(2**20));result['http_status']=response.status
  usage0=answer.get('usage',{})
  if all(type(usage0.get(k)) is int for k in ('prompt_tokens','completion_tokens')):usage={k:usage0[k] for k in ('prompt_tokens','completion_tokens')}
  assert answer.get('model')==MODEL,'Actual hosted model identity changed';reasoning=bool((answer.get('choices') or [{}])[0].get('message',{}).get('reasoning_content'));result.update(status='THINKING_OFF_VIOLATED' if reasoning else 'RESPONDED',model_returned=answer['model'],thinking_content_present=reasoning,system_fingerprint=answer.get('system_fingerprint'),usage=usage,generic_response=answer)
 except urllib.error.HTTPError as ex:result.update(status='HTTP_ERROR',http_status=ex.code,raw_error_not_logged=True)
 except Exception as ex:result.update(status='MODEL_IDENTITY_CHANGED' if isinstance(ex,AssertionError) else 'CALL_OUTCOME_UNKNOWN',error_type=type(ex).__name__,no_retry=True)
 once(folder/f'call-{index}.observed.json',dict(result,observed_before_settlement=True,epoch=time.time()));budget.settle(record,usage);result.update(seconds=time.monotonic()-tick,epoch=time.time(),usage_known=usage is not None);once(path,result);atomic(P/'budget-live.json',budget.mainline_summary());atomic(P/'live-event.json',dict(kind='actual_teacher_HTTP_response',unit=str(out.relative_to(R)),step=step,position=index,tag=tag,epoch=time.time()))
 assert result['status']=='RESPONDED' and usage is not None,'Preserve failed/unknown API and stop affected stage; no replay'
 return result

def trajectory(torch,tok,model,p,row,step,actor,budget,out,lookup):
 folder=out/'traces'/f'{step:04d}';base=row['student_base_rendered'];ids=list(row['student_prompt_ids']);assert tok.encode(base,add_special_tokens=False)==ids;initial=list(ids);prefix='';driver='S';tokens=[];events=Counter();sources=Counter();api_seconds=0.;calls=0;feedback='complete';invalid=None;ended=False
 rng=torch.Generator(device='cuda').manual_seed(e.rng_seed(p['generator'],p['seed'],step,row['id']));stepper=e.StudentStepper(torch,model);advance=e.clone(h.advance,render=lambda tok,s:base+s);model.eval();tick=time.monotonic()
 try:
  for index in range(32):
   b.check();e.snapshot();assert tok.encode(base+prefix,add_special_tokens=False)==ids;sv=stepper.score(ids);sk=sv[:248077].topk(5).indices.tolist();sp=int(torch.multinomial(sv[:248077].softmax(-1),1,generator=rng)) if driver=='S' else None
   response=call_feedback(out,p,row,step,index,prefix,folder,budget);calls+=1;api_seconds+=response['seconds']
   if terminal_unobserved(response):
    feedback='partial_terminal_unobserved';invalid=dict(kind='paid_terminal_unobserved',position=index,student_proposal_attempted_not_verified=sp,actual_API=bind(folder/f'call-{index}.json'));once(folder/'terminal-unobserved.json',invalid);break
   try:
    point=h.q.position(response,0);assert len(response['generic_response']['choices'])==1 and len(response['generic_response']['choices'][0]['logprobs']['content'])==1
    mapped=[h.event(x,tok,lookup) for x in point['top_logprobs']];assert len({x['id'] for x in mapped})==5;teacher=h.event(point,tok,lookup);tk=[x['id'] for x in mapped];tp=teacher['id'] if driver=='T' or sp not in tk else None;choice=b.decide_token(driver,sp,tp,set(sk),set(tk));selected=next((x for x in mapped if x['id']==choice.token_id),teacher);assert selected['id']==choice.token_id;nextprefix,nextids,ended=advance(tok,prefix,ids,selected)
   except (AssertionError,KeyError,IndexError,UnicodeError,TypeError) as ex:
    feedback='unqualified_actual_event';invalid=dict(kind='unqualified_actual_event_entire_trajectory_nograd',position=index,error_type=type(ex).__name__,message=str(ex)[:180],actual_API=bind(folder/f'call-{index}.json'));once(folder/'unqualified.json',invalid);break
   support=sorted(tk);scores={x['id']:y['logprob'] for x,y in zip(mapped,point['top_logprobs'])};idx=torch.tensor(support,device='cuda');tokens.append(dict(position=index,driver_before=driver,student_proposal=sp,teacher_proposal=tp,actual_teacher_API_draw=teacher['id'],student_topk_ids=sk,teacher_topk_ids=tk,union_ids=support,student_union_logits=sv[idx].detach().cpu().tolist(),teacher_union_logits=[scores[i] for i in support],actual_API=bind(folder/f'call-{index}.json'),visible_prefix_before=prefix,selected_event=selected,**asdict(choice)));events[choice.event]+=1;sources[choice.source]+=1
   once(folder/f'selected-{index}.json',dict(visible_prefix_after=nextprefix,canonical_ids_before=ids,canonical_ids_after=nextids,decision=asdict(choice),selected_event=selected,terminal=ended));prefix,ids,driver=nextprefix,nextids,choice.next_driver;del sv
   if ended:break
 finally:stepper.close()
 return dict(id=row['id'],prompt_ids=initial,actor_before=actor,support_kind='complete_teacher_top5',tokens=tokens,events=dict(events),sources=dict(sources),feedback_status=feedback,boundary=invalid,actual_API_calls=calls,API_seconds=api_seconds,observed_positions=len(tokens),max_attempted_positions=32,early_turn_end=ended,finish_reason=('turn_end' if ended else 'feedback_censored' if feedback=='partial_terminal_unobserved' else 'unqualified' if feedback=='unqualified_actual_event' else 'length'),seconds=time.monotonic()-tick,student_cached_calls=stepper.calls,teacher_policy='actual stateless API text-prefix conditional/completeK5',source_split_on_observed_positions_only=True,no_refill=True)

def loss_eligible(trace):
 assert trace['feedback_status'] in ('complete','partial_terminal_unobserved','unqualified_actual_event')
 return bool(trace['tokens']) and trace['feedback_status']!='unqualified_actual_event'

def save_paused(out,committed,actual,torch,model,head,opt):
 folder=out/'checkpoints'/f'step{committed:04d}'
 if folder.exists():m.lib.verify_checkpoint(out,folder)
 else:m.save(out,committed,actual,torch,model,head,opt)
 once(out/'paused.json',dict(last_committed=committed,actual_updates=actual,checkpoint=str(folder),unknown_started_preserved=True,automatic_resume=False,epoch=time.time()))

def fit(out):
 p=verify(out);b.check();assert not (out/'fit-started.json').exists(),'No unknown fit replay';m.preflight();e.BASELINE_FREE=m.snapshot(True)['GPU']['free_MiB'];once(out/'fit-started.json',dict(epoch=time.time(),scheduled=50,source=bind(Path(__file__))))
 stage=time.monotonic();folder=(R/p['resume_checkpoint_folder']) if p.get('resume_manifest') else (R/p['W_manifest']['path']).parent
 if p.get('resume_manifest'):assert bind(folder/'manifest.json')==p['resume_manifest'],'Exact audited checkpoint5 manifest mismatch before model load'
 torch,tok,model,head=b.initialize(out,folder);m.lib.activate(model,head,'actor');assert {k:m.lib.role_hash(model,k) for k in p['W_roles']}==p['W_roles'];rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda']);opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False);assert not opt.state
 if p.get('resume_manifest'):
  cm=m.lib.restore(torch,model,head,dict(actor=opt),R/p['resume_checkpoint_unit'],folder);assert cm['step']==5 and cm['actual_updates']==0 and not opt.state and cm['role_hashes']==p['W_roles']
 initial={k:m.lib.role_hash(model,k) for k in ('actor','reference','discriminator')};headhash=m.lib.nested_hash(head.state_dict());baseversions={k:v._version for k,v in model.named_parameters() if 'lora_' not in k};once(out/'warm-source.json',dict(manifest=p['W_manifest'],roles=initial,fresh_optimizer_object=True,zero_state_restore=p.get('resume_manifest'),W_not_retrained=True));m.save(out,p['prior_slots'],0,torch,model,head,opt);setup=time.monotonic()-stage
 rows=load(R/p['inputs']['path']);assert [x['id'] for x in rows]==p['post50_ids'];lookup=h.make_lookup(tok);budget=money.MainlineBudget(api.LEDGER);actual=0;calls=p['prior_API_calls'];observed=p['prior_observed_closed_positions'];zero_feedback=0;loss_positions=0;states=Counter(p['prior_feedback_counts']);all_events=Counter(p['prior_events']);all_sources=Counter(p['prior_sources']);cool=pace=compute=api_seconds=checkpoint_seconds=0.;committed=p['prior_slots'];begin=time.monotonic()
 try:
  for step,row in enumerate(rows[p['schedule_start']-1:],p['schedule_start']):
   b.check();assert time.monotonic()-begin<14400,'Bounded4h fit';cool+=e.cooldown();tick=time.monotonic();time.sleep(.5);pace+=time.monotonic()-tick;actor=m.lib.role_hash(model,'actor');once(out/'updates'/f'{step:04d}.started.json',dict(step=step,id=row['id'],actor_before=actor,epoch=time.time()))
   trace=trajectory(torch,tok,model,p,row,step,actor,budget,out,lookup);once(out/'traces'/f'{step:04d}.json',trace);calls+=trace['actual_API_calls'];observed+=trace['observed_positions'];api_seconds+=trace['API_seconds'];states[trace['feedback_status']]+=1;all_events.update(trace['events']);all_sources.update(trace['sources']);tick=time.monotonic()
   if loss_eligible(trace):metric=e.update(torch,model,head,opt,trace);torch.cuda.synchronize()
   else:metric=dict(optimizer_updated=False,actor_before=actor,actor_after=actor,loss=None,gradient_norm=None,input_tokens=len(trace['prompt_ids']),target_tokens=0,events=trace['events'],sources=trace['sources'],update_seconds=None,update_not_executed=True,nograd_reason=trace['feedback_status'],no_SFT_or_gold_fallback=True)
   compute+=time.monotonic()-tick;actual+=int(metric['optimizer_updated']);zero_feedback+=int(trace['observed_positions']==0);loss_positions+=metric['target_tokens'];assert all(m.lib.role_hash(model,k)==initial[k] for k in ('reference','discriminator')) and m.lib.nested_hash(head.state_dict())==headhash;assert all(v._version==baseversions[k] for k,v in model.named_parameters() if k in baseversions)
   metric.update(step=step,id=row['id'],feedback_status=trace['feedback_status'],observed_positions=trace['observed_positions'],API_calls=trace['actual_API_calls'],API_seconds=trace['API_seconds'],trajectory_seconds=trace['seconds'],actual_updates=actual,epoch=time.time())
   if step%5==0 or step==50:
    tick=time.monotonic();checkpoint,cm=m.save(out,step,actual,torch,model,head,opt);checkpoint_seconds+=time.monotonic()-tick
   once(out/'updates'/f'{step:04d}.json',metric);committed=step;e.snapshot();atomic(P/'progress.json',dict(kind='actual_cloud_IPD_scheduled_task_commit',unit=str(out.relative_to(R)),generator=p['generator'],seed=p['seed'],scheduled_committed=step,scheduled_total=50,actual_updates=actual,actual_API_calls=calls,observed_positions=observed,feedback_counts=dict(states),events=dict(all_events),sources=dict(all_sources),epoch=time.time()));atomic(P/'live-event.json',dict(kind='actual_optimizer_or_nograd_task_commit',actual_optimizer_updated=metric['optimizer_updated'],unit=str(out.relative_to(R)),step=step,epoch=time.time()));print('CLOUD_MAIN_IPD',p['generator'],p['seed'],step,'UPDATES',actual,'CALLS',calls,'FEEDBACK',trace['feedback_status'],flush=True)
 except InterruptedError:
  save_paused(out,committed,actual,torch,model,head,opt);raise

 once(out/'fit-terminal.json',dict(status='ORIGINAL_CLOUD50_ACCOUNTED_WITH_PRESERVED_PRIOR_SLOTS_RELOAD_PENDING',scheduled=50,scheduled_executed_this_attempt=50-p['prior_slots'],prior_untrained_slots=p['prior_slots'],prior_interrupted_selected_positions=p['prior_interrupted_selected_positions'],actual_updates=actual,has_actual_IPD_update=actual>0,checkpoint=str(checkpoint),manifest=bind(checkpoint/'manifest.json'),initial_roles=initial,final_actor=m.lib.role_hash(model,'actor'),source_counts=dict(all_sources),events=dict(all_events),feedback_counts=dict(states),actual_API_calls=calls,observed_positions=observed,zero_feedback_trajectories=zero_feedback,loss_covered_positions=loss_positions,setup_seconds=setup,fit_seconds=time.monotonic()-begin,stage_wall_seconds=time.monotonic()-stage,API_seconds=api_seconds,update_compute_seconds=compute,pacing_seconds=pace,cooling_seconds=cool,checkpoint_seconds=checkpoint_seconds,CPU_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CUDA_allocated_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),resource=e.snapshot(),no_Test_or_gold_or_oracle_in_loss=True,epoch=time.time()))

def reload(out):
 p=verify(out);b.check();assert not (out/'reload-started.json').exists(),'No unknown reload replay';term=load(out/'fit-terminal.json');folder=Path(term['checkpoint']);m.preflight();e.BASELINE_FREE=m.snapshot(True)['GPU']['free_MiB'];once(out/'reload-started.json',dict(epoch=time.time(),manifest=bind(folder/'manifest.json')));tick=time.monotonic();torch,tok,model,head=b.initialize(out,folder);m.lib.activate(model,head,'actor');opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0,foreach=False);cm=m.lib.restore(torch,model,head,dict(actor=opt),out,folder);rng=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True);assert torch.equal(torch.get_rng_state(),rng['cpu']) and all(torch.equal(a,z) for a,z in zip(torch.cuda.get_rng_state_all(),rng['cuda']));steps=[float(v['step']) for v in opt.state.values() if 'step' in v];assert (steps and all(v==term['actual_updates'] for v in steps)) if term['actual_updates'] else not steps;assert cm['step']==50 and cm['actual_updates']==term['actual_updates'] and cm['role_hashes']['actor']==term['final_actor'];once(out/'independent-reload.json',dict(status='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS',step=50,actual_updates=term['actual_updates'],optimizer_state_steps=steps,manifest=bind(folder/'manifest.json'),roles=cm['role_hashes'],seconds=time.monotonic()-tick,resource=e.snapshot(),no_new_updates=True,epoch=time.time()))

def run():
 cg=next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));assert Path(cg).name==UNIT and os.environ.get('INVOCATION_ID') and os.getuid()!=0
 for gen in GENS:
  for seed in SEEDS:verify(unit(gen,seed))
 once(P/'started.json',dict(source=bind(Path(__file__)),invocation=os.environ['INVOCATION_ID'],registered_order=[(g,s) for g in GENS for s in SEEDS],epoch=time.time()));begin=time.monotonic();done=[]
 for gen in GENS:
  for seed in SEEDS:
   out=unit(gen,seed)
   for action in ('fit','reload'):
    b.check();assert not (out/('fit-started.json' if action=='fit' else 'reload-started.json')).exists(),'No completed or unknown model action replay';child=subprocess.Popen([sys.executable,str(Path(__file__)),action,'--generator',gen,'--seed',str(seed)],cwd=R);record=P/'children'/f'{gen}-{seed}-{action}';once(record.with_suffix('.started.json'),dict(pid=child.pid,action=action,epoch=time.time()));tick=time.monotonic()
    try:
     while child.poll() is None:b.check();assert time.monotonic()-tick<(15000 if action=='fit' else 420);time.sleep(.5)
     once(record.with_suffix('.json'),dict(exit_code=child.returncode,seconds=time.monotonic()-tick,epoch=time.time()));assert child.returncode==0,'Preserve actual failure, no automatic stage retry'
    finally:
     if child.poll() is None:child.terminate();child.wait(timeout=90)
   term=load(out/'fit-terminal.json');assert load(out/'independent-reload.json')['status']=='EXACT_ROLES_OPTIMIZER_RNG_RELOAD_PASS';done.append(dict(generator=gen,seed=seed,fit=bind(out/'fit-terminal.json'),reload=bind(out/'independent-reload.json'),actual_updates=term['actual_updates']));atomic(P/'completed-units.json',dict(units=done,actual_updates=sum(x['actual_updates'] for x in done),epoch=time.time()))
 once(P/'terminal.json',dict(status='SIX_CLOUD_IPD_STUDENTS_FIT_AND_RELOAD_CLOSED',units=done,scheduled=300,prior_untrained_slots6_preserved=True,scheduled_executed_this_queue294=True,actual_updates=sum(x['actual_updates'] for x in done),all_units_have_nonzero_update=all(x['actual_updates']>0 for x in done),seconds=time.monotonic()-begin,local_same_Test_student9414_eval_pending=True,cloud_protected_Test_teacher_unfulfilled=True,whole_goal_complete=False,epoch=time.time()));print('SIX_CLOUD_IPD_FIT_RELOAD_CLOSED',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','fit','reload']);a.add_argument('--generator',choices=GENS);a.add_argument('--seed',type=int,choices=SEEDS);v=a.parse_args();signal.signal(signal.SIGTERM,b.stop);signal.signal(signal.SIGINT,b.stop)
 try:
  if v.action=='prepare':prepare()
  elif v.action=='run':run()
  else:
   out=unit(v.generator,v.seed)
   with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(fit if v.action=='fit' else reload)(out)
 except BaseException as ex:
  if isinstance(ex,InterruptedError):
   target=P/'paused.json' if not v.generator else unit(v.generator,v.seed)/'paused.json'
   if not target.exists():once(target,dict(status='STOP_PRESERVED_NO_AUTOMATIC_RESUME',epoch=time.time(),unknown_started_preserved=True))
   raise SystemExit(0)
  target=P/(v.action+'-failure.json') if not v.generator else unit(v.generator,v.seed)/(v.action+'-failure.json')
  if not target.exists():once(target,dict(type=type(ex).__name__,message=str(ex)[:600],epoch=time.time(),unknown_started_preserved=True,no_automatic_retry=True))
  raise
