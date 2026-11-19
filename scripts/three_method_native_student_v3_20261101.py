"""Repair ONLY legacy-bound on-policy sampling; keep original native CE/math/serializer.
Six on units in a new namespace; W/off stay at their exact existing paths.
"""
from pathlib import Path
import argparse,copy,fcntl,signal,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import three_method_native_student_v2_20261101 as m
from three_method_uncoupled_native_sampling_20261101 import uncoupled_sampler
O=m.O;NEW=O/'students-on-native-v3';P=O/'student-on-engine-v3';OLD_UNIT=m.unit;OLD_AUTHORIZE=m.authorize;OLD_S=m.S

def unit(gen,seed,stage):
 original=OLD_UNIT(gen,seed,stage)
 return NEW/gen/str(seed)/stage if stage=='on' else original
m.unit=unit
m.lib.rollouts=uncoupled_sampler()

def phase_bytes():
 return sum(x.stat().st_size for root in (OLD_S,NEW) for x in root.rglob('*') if x.is_file())
m.phase_bytes=phase_bytes

def authorize(out,action):
 p=m.verify(out)
 if p['stage']!='on':return OLD_AUTHORIZE(out,action)
 g=m.load(P/'root-grant.json');assert g['approved'] and g['source']==m.bind(Path(__file__)) and action in g['actions']
 assert str(out.relative_to(R)) in g['allowed_units']
 for b in g['bindings']:assert m.bind(R/b['path'])==b,b['path']
 m.check_stop();return p
m.authorize=authorize

def prepare():
 assert not (P/'root-grant.json').exists()
 proof=m.load(O/'on-sampler-fix-v1/cpu-proof-and-zero-step.json');assert proof['actual_optimizer_commits']==0 and proof['actual_rollouts']==0 and proof['RNG_equals_W']
 cpu=m.load(P/'cpu-proof.json');assert cpu['source']==m.bind(Path(__file__))
 units=[]
 for gen in m.GENS:
  for seed in m.SEEDS:
   old=OLD_UNIT(gen,seed,'on');out=unit(gen,seed,'on');assert not out.exists()
   p=copy.deepcopy(m.load(old/'protocol.json'));p.update(unit_path=str(out),
    sampler='exact original native function, legacyCTX/cached-seed wrapper removed',
    sampler_source='scripts/train_ust_sg_native_gad_20261024.py::rollouts',RNG='same actual W CUDA RNG restored once, eight consecutive current-actor draws; no per-draw reseed or quality refill',
    repair_before_any_actual_on_rollout=True)
   m.once(out/'protocol.json',p);m.verify(out);units.append(out)
 pins=[Path(__file__),R/'scripts/three_method_uncoupled_native_sampling_20261101.py',O/'on-sampler-fix-v1/cpu-proof-and-zero-step.json',P/'cpu-proof.json',
  Path(m.__file__),m.P/'root-grant.json',m.P/'protocol.json',R/'scripts/train_ust_sg_native_gad_20261024.py',R/'scripts/train_sg_teacher_label_grpo_20261025.py',
  R/'scripts/bird_gad_night_20261022.py',R/'src/privacy_synth/onpolicy_gad.py',m.PLAN,*[u/'protocol.json' for u in units]]
 m.once(P/'root-grant.json',dict(approved=True,actor='Root',source=m.bind(Path(__file__)),bindings=[m.bind(x) for x in pins],allowed_units=[str(u.relative_to(R)) for u in units],actions=['fit','reload'],
  authority='Original explicit six true fresh-current-actor GRPO50 goal; actual first oldon failed before any sampling/optimizer, exact zero-step evidence, preserve oldinitialization and all started/failure',
  known_zero_old_on_reinitialized_from_same_W=True,new_optimizer_resumes=0,maximum_scheduled_tasks=300,maximum_group_draws=2400,
  mathematical_policy_update_and_CE_unchanged=True,no_W_or_completed_off_retraining=True,original_guards_unchanged=True,API_calls=0))
 print('SIX_REPAIRED_NATIVE_ON_PROTOCOLS_ROOT_BOUND_NO_COMPLETED_W_OFF_RETRAIN')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','fit','reload']);a.add_argument('--generator',choices=m.GENS);a.add_argument('--seed',type=int,choices=m.SEEDS);a.add_argument('--stage',choices=m.STAGES);v=a.parse_args();signal.signal(signal.SIGTERM,m.stop);signal.signal(signal.SIGINT,m.stop)
 if v.action=='prepare':prepare()
 else:
  out=unit(v.generator,v.seed,v.stage)
  try:
   m.once(out/('runtime-adapter-'+v.action+'.json'),dict(adapter=m.bind(Path(__file__)),sampler='native original AST; noCTX or rollout-cache/reseed',original_engine=m.bind(Path(m.__file__)),epoch=time.time()))
   with (R/'runs/2026-10-22-night-dual/locks/gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(m.fit if v.action=='fit' else m.reload)(out)
  except BaseException as ex:
   m.once(out/(v.action+'-adapter-failure.json'),dict(error_type=type(ex).__name__,message=str(ex),epoch=time.time(),no_unknown_replay=True));raise