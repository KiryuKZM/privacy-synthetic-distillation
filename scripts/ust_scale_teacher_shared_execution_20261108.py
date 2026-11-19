"""Exact teacher/oracle input equivalence; attach shared executions, no extra GPU."""
from pathlib import Path
import os,sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import ust_scale_teacher_recover_one_20261108 as rec
t=rec.t;O=rec.O

def audit():
 assert t.load(O/'training-comparison.json')['different_target_count']==0
 a=t.load(t.T/'variants.local.json');b=[v for v in t.load(t.P/'variants.local.json') if v['id'].startswith('current_n1088/')];assert a==b and len(a)==4352
 at=t.load(t.T/'tokens.local.json');bt=t.load(t.P/'tokens.local.json');assert all(value==bt[key] for key,value in at.items())
 assert t.load(t.T/'fixed8.local.json')==t.load(t.P/'fixed8.local.json')
 bindings=[t.pin(__file__),t.pin(rec.__file__),t.pin(t.study.__file__),t.pin(O/'labels.local.json'),t.pin(O/'result.json'),t.pin(t.T/'protocol.json'),t.pin(t.P/'protocol.json')]
 for seed in t.study.SEEDS:
  for name in ('protocol.json','schedule.local.json','queries.local.json'):
   x=t.T/f'teacher_n1088_s{seed}'/name;y=t.P/f'current_n1088_s{seed}'/name
   assert t.load(x)==t.load(y);bindings.extend([t.pin(x),t.pin(y)])
 t.save(O/'shared-execution-proof.json',dict(status='EXACT_INPUT_AND_LABEL_EQUIVALENCE',bindings=bindings,views=4352,token_records=len(at),training_order_seeds=t.study.SEEDS,teacher_labels_from_1088_initial_calls_plus1_explicit_recovery=True,actual_additional_teacher_fits=0,planned_three_fits_covered_by_existing_identical_current1088_fits=True,not_independent_replicates=True,not_a_teacher_vs_oracle_noise_contrast=True,Final_access=False))

def wait_and_attach():
 p=t.load(O/'shared-execution-proof.json')
 for b in p['bindings']:assert t.pin(R/b['path'])==b
 t.save(O/'shared-waiter-started.json',dict(PID=os.getpid(),PPID=os.getppid(),SID=os.getsid(0),epoch=time.time(),GPU=False))
 deadline=time.monotonic()+172800
 while not (t.P/'terminal.json').exists():
  assert not (t.P/'failure.json').exists(),'Main suite failed; no success claim'
  assert not (O/'STOP').exists() and time.monotonic()<deadline
  time.sleep(10)
 specs=t.load(t.P/'specs.json');results=t.load(t.P/'results.json');attached={}
 for seed in t.study.SEEDS:
  key=f'current_n1088_s{seed}';u=t.P/key;assert t.load(u/'fit-terminal.json')['actual_updates']==2176;assert t.load(u/'evaluation-terminal.json')['raw']==310
  h=t.study.helper();_,pins=h.checkpoint_pins(u,u/'checkpoints/step2176',2176)
  attached[str(seed)]=dict(shared_existing_arm=key,results=results['scores'][key],checkpoint_pins=pins,fit=t.pin(u/'fit-terminal.json'),evaluation=t.pin(u/'evaluation-terminal.json'),raw_grades=t.pin(u/'graded.local.json'))
 for b in p['bindings']:assert t.pin(R/b['path'])==b
 t.save(O/'shared-execution-results.json',dict(status='COMPLETE_USING3_EXACTLY_EQUIVALENT_MAIN_RUNS',orders=attached,additional_fits=0,independent_teacher_vs_oracle_replicates=False,budget=t.budget(),Final_access=False))
 t.save(t.S/'recovery-terminal.json',dict(status='COMPLETE_AFTER_ONE_APPROVED_API_RECOVERY_WITH_SHARED_STUDENT_EXECUTIONS',results=t.pin(O/'shared-execution-results.json'),original_failure_retained=True,additional_teacher_fits=0,epoch=time.time()))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:t.save(O/f'shared-failure-{time.time_ns()}.json',dict(error=str(e),epoch=time.time()));raise
