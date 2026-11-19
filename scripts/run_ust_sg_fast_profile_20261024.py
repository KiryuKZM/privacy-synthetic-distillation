#!/usr/bin/env python3
"""Explicit fast-idle local profile; untouched scientific runner and cached answers."""
from __future__ import annotations
import argparse, contextlib, json, time, subprocess
from pathlib import Path
import eval_ust_sg_qwen_modes_20261024 as q
from run_ust_sg_gaming_profile_20261024 import swap, RUNNER_SHA, MANIFEST_SHA, PREFLIGHT_SHA
ROOT=q.ROOT
RUN=ROOT/'runs/2026-10-24-ust100-sg60-modes'
OUT=RUN/'local-qwen-original2b'
PROFILE=RUN/'fast-idle-profile-v1'
CONTRACT=PROFILE/'contract.json'
MANIFEST=RUN/'local-qwen-manifest.json'
UNIT='ust-sg-qwen-gaming-20261024.service'
APPROVAL=ROOT/'runs/2026-10-24-ust-sg-onpolicy/user-approval.json'

class ResourceFloor(Exception):pass

def resources_ok(s,startup=True):
    checks={k:s.get(k,-1)>=20 for k in ('windows_free_gib','c_free_gib','linux_free_gib')}
    if startup:checks['gpu_free_gib']=s.get('gpu_free_gib',-1)>=9
    return all(checks.values()),checks

def contract_value():
    return dict(name='fast-idle-v1',wrapper_sha256=q.sha(__file__),original_runner_sha256=RUNNER_SHA,
        manifest_sha256=MANIFEST_SHA,preflight_sha256=PREFLIGHT_SHA,
        user_approval_sha256=q.sha(APPROVAL),thinking='on',max_new_tokens=8192,greedy=True,
        batch_size=1,cpu_threads=4,cuda_allocator_cap_gib=9,start_gpu_free_gib=9,
        host_ram_floor_gib=20,disk_floor_gib=20,idle_utilization_required=False,
        stable_wait_seconds=0,token_sleep_seconds=0,per_question_unload=False,cooldown_seconds=0,
        arbitrary_deadline_epoch=None,execution_policy_amended=True,api_calls=0)

def prepare():
    q.immutable(CONTRACT,contract_value())
    return dict(status='PREPARED',contract_sha256=q.sha(CONTRACT),gpu_used=False)

def verify():
    if q.load(CONTRACT)!=contract_value():raise PermissionError('Fast contract/source changed')
    for p,h in [(q.__file__,RUNNER_SHA),(MANIFEST,MANIFEST_SHA),(OUT/'preflight.json',PREFLIGHT_SHA)]:
        if q.sha(p)!=h:raise PermissionError('Frozen scientific inputs changed')
    auth=q.load(PROFILE/'authorization.json')
    if auth.get('root_approved') is not True or auth.get('contract_sha256')!=q.sha(CONTRACT):
        raise PermissionError('Explicit fast-profile authorization missing')

def unresolved():
    return [p for p in OUT.glob('*/*/responses/*.started.json')
            if not p.with_name(p.name.replace('.started.json','.json')).exists()]

def cached():
    return {str(p.relative_to(ROOT)):q.sha(p) for p in OUT.glob('*/*/responses/*.json')
            if not p.name.endswith('.started.json')}

def handoff(query=None,stop=None,sleep=time.sleep):
    """Only drain the owned old queue after its current answer has committed."""
    query=query or (lambda:subprocess.run(['systemctl','--user','show',UNIT,
        '--property=ActiveState','--property=ExecStart'],capture_output=True,text=True,check=True).stdout)
    stop=stop or (lambda:subprocess.run(['systemctl','--user','stop',UNIT],check=True))
    verify()
    receipt=PROFILE/'handoff.json'
    if receipt.exists():
        value=q.load(receipt)
        if unresolved():raise PermissionError('Unresolved started item after saved handoff')
        for p,h in value['preserved_cache'].items():
            if q.sha(ROOT/p)!=h:raise PermissionError('Old committed response changed')
        return value
    while True:
        state=query()
        if 'ActiveState=active' not in state and 'ActiveState=activating' not in state:
            if unresolved():raise PermissionError('Old queue inactive with unresolved generation; no resampling')
            break
        if 'run_ust_sg_gaming_queue_20261024.py' not in state:
            raise PermissionError('Owned fixed unit command mismatch')
        if not unresolved():
            # The queue has a mandatory 60-second saved-answer cooldown. Do not stop mid-generation.
            before=cached()
            q.immutable(PROFILE/'handoff-stop-intent.json',dict(unit=UNIT,epoch=time.time(),
                contract_sha256=q.sha(CONTRACT),preserved_cache=before))
            stop()
            if unresolved():raise PermissionError('Handoff raced a new start; retained unresolved, no retry')
            break
        sleep(2)
    value=dict(unit=UNIT,epoch=time.time(),contract_sha256=q.sha(CONTRACT),
               preserved_cache=cached(),unresolved_started=0,no_resampling=True)
    q.immutable(receipt,value)
    return value

def annotate(path,obj,stats,writer):
    path=Path(path)
    if path.parent.name=='responses' and not path.name.endswith('.started.json'):
        obj=dict(obj,execution_profile=dict(name='fast-idle-v1',
            contract_path=str(CONTRACT.relative_to(ROOT)),contract_sha256=q.sha(CONTRACT),
            wrapper_source_sha256=q.sha(__file__),original_runner_sha256=RUNNER_SHA,
            execution_policy_amended=True,completed_epoch=time.time(),actual_overrides=dict(stats)))
    return writer(path,obj)

@contextlib.contextmanager
def fast_runtime(stats):
    import run_cityfm_repair_qwen_20261022 as guard
    original=q.immutable
    def check(*args,**kwargs):
        snapshot=guard.probe(RUN)
        ok,checks=resources_ok(snapshot,True)
        stats.update(start_resource_snapshot=snapshot,start_resource_checks=checks)
        if not ok:raise ResourceFloor('Immediate GPU9GiB/host20GiB floor unmet; no model loaded')
        return snapshot
    with contextlib.ExitStack() as stack:
        stack.enter_context(swap(guard,'wait_resources',check))
        stack.enter_context(swap(guard,'resources_ok',resources_ok))
        stack.enter_context(swap(q,'immutable',lambda p,o:annotate(p,o,stats,original)))
        yield

def run():
    from night_dual_20261022 import lock
    verify()
    with lock(PROFILE/'profile.lock'):
        handoff()
        if unresolved():raise PermissionError('Unresolved started generation forbids new inference')
        stats=dict(torch_threads=4,cuda_allocator_cap_gib=9,start_gpu_free_gib=9,
            stable_wait_seconds=0,token_sleep_seconds=0,per_question_unload=False,
            cooldown_seconds=0,arbitrary_deadline_epoch=None)
        try:
            with fast_runtime(stats):
                for task in ('USTBench','SGQA'):
                    q.run(MANIFEST,q.DEFAULT_MODEL,task,'on',OUT)
        except ResourceFloor as e:
            return dict(status='WAITING_IMMEDIATE_RESOURCE_FLOOR',reason=str(e),gold_read=False)
        if not q.freeze_all(MANIFEST,OUT):raise RuntimeError('Full320 freeze missing; no gold access')
        result=q.score(MANIFEST,OUT)
        q.immutable(PROFILE/'completed.json',dict(status='FULL320_FROZEN_SCORED',
            contract_sha256=q.sha(CONTRACT),scores_sha256=q.sha(OUT/'scores.json'),epoch=time.time()))
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--prepare',action='store_true');a.add_argument('--run',action='store_true')
    a.add_argument('--handoff-only',action='store_true')
    args=p.parse_args()
    print(json.dumps(prepare() if args.prepare else handoff() if args.handoff_only else run()))
if __name__=='__main__':main()

