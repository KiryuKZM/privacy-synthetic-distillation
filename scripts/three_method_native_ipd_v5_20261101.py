"""Text-only cached position IDs repair; all original IPD math and fixed tasks retained."""
from pathlib import Path
import argparse,ast,fcntl,signal,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from three_method_native_ipd_v4_20261101 import e
ORIGINAL_STEPPER=e.StudentStepper
node=next(x for x in ast.parse((R/'scripts/three_method_native_ipd_20261101.py').read_text()).body if isinstance(x,ast.ClassDef) and x.name=='StudentStepper')
changed=0
for x in ast.walk(node):
 if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='self' and x.func.attr=='model':
  assert not any(k.arg=='position_ids' for k in x.keywords)
  x.keywords.append(ast.keyword(arg='position_ids',value=ast.parse('self.torch.arange(len(prefix)-ids.shape[1],len(prefix),device=self.device,dtype=self.torch.long).unsqueeze(0)',mode='eval').body));changed+=1
assert changed==1
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(Path(__file__))+'::one-text-position-keyword','exec'),e.__dict__)
e.P=e.O/'latest-ipd-engine-v5';e.__file__=str(Path(__file__).resolve())
e.base_snapshot=e.clone(e.m.snapshot,P=e.P,check_stop=e.check_stop);e.save=e.clone(e.m.save,verify=e.verify,check_stop=e.check_stop)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['fit','reload']);a.add_argument('--generator',choices=e.m.GENS,required=True);a.add_argument('--seed',type=int,choices=e.m.SEEDS,required=True);v=a.parse_args();signal.signal(signal.SIGTERM,e.stop);signal.signal(signal.SIGINT,e.stop)
 out=e.m.unit(v.generator,v.seed,'ipd')
 try:
  with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(e.fit if v.action=='fit' else e.reload)(out)
 except BaseException as ex:e.once(out/(v.action+'-failure.json'),dict(type=type(ex).__name__,message=str(ex),epoch=time.time(),no_model_replay=True));raise