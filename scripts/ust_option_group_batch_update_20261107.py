"""One experimental batch4 CE update; no CLI, model initialization or data access.

Four response-token-mean losses /4, four immediate backwards, then one clip and
one unconditional Adam call. Same7-token targets; no prompt loss or native tail CE.
"""
import time

def optimizer_steps(opt,params):
 assert [id(p) for g in opt.param_groups for p in g['params']]==[id(p) for p in params]
 if not opt.state:return [0]*len(params)
 assert len(opt.state)==len(params) and all(p in opt.state for p in params)
 return [int(opt.state[p]['step'].item()) for p in params]

def batch4_update(torch,model,head,opt,traces,lib,check,semantic=248077,native=248320):
 assert len(traces)==4 and all(len(t['oracle_target_ids'])==7 for t in traces)
 lib.activate(model,head,'actor');model.train();opt.zero_grad(set_to_none=True)
 params=[p for p in model.parameters() if p.requires_grad];assert len(params)==48
 before=lib.role_hash(model,'actor');steps_before=optimizer_steps(opt,params);assert len(set(steps_before))==1
 micros=[];started=time.monotonic()
 for index,trace in enumerate(traces):
  check();target=trace['oracle_target_ids'];prompt=trace['prompt_ids'];assert len(prompt)>0 and len(prompt)+len(target)<=16384
  ids=torch.tensor([prompt+target[:-1]],device='cuda');gold=torch.tensor(target,device='cuda')
  logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(target)).logits[0].float()
  assert tuple(logits.shape)==(7,native)
  loss=torch.nn.functional.cross_entropy(logits[:,:semantic],gold)
  if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite batch4 micro CE')
  (loss/4.).backward()
  micros.append(dict(index=index,id=trace['id'],loss=float(loss.detach()),scaled_loss=float(loss.detach()/4.),input_tokens=len(prompt),target_tokens=7))
  assert optimizer_steps(opt,params)==steps_before,'Adam advanced inside microbatch'
  del loss,logits,ids,gold
 assert all(p.grad is not None for p in params),'Every actor tensor must receive an accumulated gradient'
 if any(not bool(torch.isfinite(p.grad).all()) for p in params):raise RuntimeError('Nonfinite accumulated actor gradient')
 assert lib.role_hash(model,'actor')==before,'Parameters changed before the sole optimizer call'
 norm=torch.nn.utils.clip_grad_norm_(params,1.)
 if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite aggregate gradient norm')
 check();opt.step();steps_after=optimizer_steps(opt,params)
 assert steps_after==[x+1 for x in steps_before],'Every one of48 Adam clocks must advance once'
 opt.zero_grad(set_to_none=True);after=lib.role_hash(model,'actor');assert all(p.grad is None for p in model.parameters())
 return dict(method='batch4-response-mean-CE',microexamples=micros,loss=sum(x['loss'] for x in micros)/4.,gradient_norm=float(norm),optimizer_calls=1,optimizer_updated=True,
  parameter_changed=before!=after,actor_before=before,actor_after=after,state_steps_before=steps_before,state_steps_after=steps_after,
  micro_forward_backward_calls=4,clipping_calls=1,gradient_zero_still_advances_Adam=True,update_seconds=time.monotonic()-started,API=0)
