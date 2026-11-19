"""Same actual Q3 teacher/IPD math, one16k sequence instead of an unused second16k.
GPU placement12layers is prospective resource placement, never accuracy selection.
"""
from pathlib import Path
import argparse,fcntl,signal,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from three_method_native_ipd_v3_20261101 import e
from three_method_native_teacher_client_20261101 import NativeTeacher
e.P=e.O/'latest-ipd-engine-v4';e.__file__=str(Path(__file__).resolve());e.BINARY=R/'artifacts/three-method-llama-native-single-seq-20261101'
e.base_snapshot=e.clone(e.m.snapshot,P=e.P,check_stop=e.check_stop);e.save=e.clone(e.m.save,verify=e.verify,check_stop=e.check_stop)
class SingleTeacher(NativeTeacher):
 @staticmethod
 def slot(slot):assert type(slot) is int and slot==0
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  assert self.ready['n_seq_max']==1 and self.ready['context_tokens']==16384

def teacher_open(folder,qualification=False):
 profile=e.load(e.P/('profile-proposal.json' if qualification else 'qualified-profile.json'))
 assert profile['ngl']==12 and profile['threads']==2 and profile['n_seq_max']==1 and profile['context_per_sequence']==16384
 return SingleTeacher(e.BINARY,e.MODEL,e.BACKEND,folder/'native-teacher.stderr.txt',12,threads=2,guard=e.native_guard,timeout=900,qualification=qualification)
e.teacher_open=teacher_open
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['fit','reload']);a.add_argument('--generator',choices=e.m.GENS,required=True);a.add_argument('--seed',type=int,choices=e.m.SEEDS,required=True);v=a.parse_args();signal.signal(signal.SIGTERM,e.stop);signal.signal(signal.SIGINT,e.stop)
 out=e.m.unit(v.generator,v.seed,'ipd')
 try:
  with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(e.fit if v.action=='fit' else e.reload)(out)
 except BaseException as ex:e.once(out/(v.action+'-failure.json'),dict(error_type=type(ex).__name__,message=str(ex),epoch=time.time(),no_model_replay=True));raise