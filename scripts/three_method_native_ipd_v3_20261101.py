"""Operational dependency adaptation only; original IPD math/worker stays frozen."""
from pathlib import Path
import argparse,fcntl,signal,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import three_method_native_student_v3_20261101 as student_adapter
from three_method_native_ipd_v2_20261101 import e
e.P=e.O/'latest-ipd-engine-v3';e.__file__=str(Path(__file__).resolve())
e.base_snapshot=e.clone(e.m.snapshot,P=e.P,check_stop=e.check_stop)
e.save=e.clone(e.m.save,verify=e.verify,check_stop=e.check_stop)
def dependencies():
 a=e.O/'student-controller-v5/terminal.json';b=e.Q3/'terminal.json'
 assert e.load(a)['status']=='ALL18_W_OFF_ON_UNITS_FIT_RELOAD_COMPLETE_IPD_AND_COMMON_TEST_STILL_PENDING'
 assert e.load(b)['status']=='DIRECT_BF16_Q3_K_M_ARTIFACT_SHA_AND_CODEC_VERIFIED_NOT_INFERENCE_QUALIFIED'
 return [e.bind(a),e.bind(b)]
e.dependencies=dependencies
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','reload']);p.add_argument('--generator',choices=e.m.GENS,required=True);p.add_argument('--seed',type=int,choices=e.m.SEEDS,required=True);v=p.parse_args();signal.signal(signal.SIGTERM,e.stop);signal.signal(signal.SIGINT,e.stop)
 out=e.m.unit(v.generator,v.seed,'ipd')
 try:
  with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(e.fit if v.action=='fit' else e.reload)(out)
 except BaseException as ex:e.once(out/(v.action+'-failure.json'),dict(error_type=type(ex).__name__,message=str(ex),epoch=time.time()));raise