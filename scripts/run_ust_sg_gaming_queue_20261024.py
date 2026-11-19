#!/usr/bin/env python3
"""Owned gaming queue: one isolated response, cooldown, bounded pilot, no gold."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/home/kiryu/privacy-synthetic-distillation')
RUN = ROOT / 'runs/2026-10-24-ust100-sg60-modes'
PROFILE = RUN / 'gaming-profile-v1'
WRAPPER = ROOT / 'scripts/run_ust_sg_gaming_profile_20261024.py'
DEADLINE = 1792816127.9174702
STOP = False
GOLD_READ = False

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def status(state, **kw):
    atomic(PROFILE/'queue-heartbeat.json', dict(status=state, epoch=time.time(),
        deadline_epoch=DEADLINE, gold_read=GOLD_READ, api_calls=0, **kw))

def stop(*_):
    global STOP
    STOP = True

def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)

def committed():
    found = []
    for path in (RUN/'local-qwen-original2b').glob('*/on/responses/*.json'):
        if path.name.endswith('.started.json'):
            continue
        record = load(path)
        if record.get('execution_profile',{}).get('contract_sha256') == sha(PROFILE/'contract.json'):
            found.append((path,record))
    return sorted(found,key=lambda x:x[1]['execution_profile']['completed_epoch'])

def approve_continuation():
    items=committed()
    if len(items)!=3:
        raise RuntimeError('Exactly three pilot records required')
    for path,r in items:
        p=r['execution_profile']
        a=p['actual_overrides']
        if (r.get('error') or r.get('finish_reason')=='technical_failure'
            or r.get('peak_allocated_bytes',10**20)>6.5*2**30
            or not 0<r.get('peak_reserved_bytes',0)<=6.5*2**30
            or a.get('actual_cuda_allocator_cap_gib',100)>6.5
            or a.get('max_new_tokens')!=8192 or a.get('do_sample') is not False
            or a.get('throttle_calls',0)<=0 or a.get('actual_torch_threads')!=1
            or p.get('wrapper_source_sha256')!=sha(WRAPPER)):
            raise RuntimeError('Pilot mechanical or resource validation failed')
    receipt = dict(authorized_by='root/user_slow_local_request',
        contract_sha256=sha(PROFILE/'contract.json'),
        pilot_responses=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p,r in items],
        max_total_new_responses=160, deadline_epoch=DEADLINE,
        validation='mechanical_and_resource_only_no_gold', epoch=time.time())
    target=PROFILE/'continuation.json'
    if target.exists():
        raise RuntimeError('Continuation receipt already exists unexpectedly')
    target.write_text(json.dumps(receipt,indent=2))
    return receipt

def main():
    global GOLD_READ
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    contract=load(PROFILE/'contract.json')
    if contract['wrapper_sha256']!=sha(WRAPPER) or contract['deadline_epoch']!=DEADLINE:
        raise RuntimeError('Source or deadline mismatch')
    attempts=0
    completed_tasks=set()
    task_order=['USTBench','USTBench','SGQA']
    for n in range(100000):
        if STOP or time.time()>=DEADLINE:
            status('STOPPED_ORIGINAL_DEADLINE_OR_USER',saved=len(committed()))
            return
        saved=committed()
        if len(saved)==3 and not (PROFILE/'continuation.json').exists():
            if contract.get('maximum_with_bound_continuation')!=160:
                status('PILOT_COMPLETE_WAITING_CONTINUATION',saved=3)
                return
            approve_continuation()
            status('PILOT_VALIDATED_SLOW_CONTINUATION',saved=3)
        if len(saved)>=160 or len(completed_tasks)==2:
            import eval_ust_sg_qwen_modes_20261024 as q
            if not q.freeze_all(RUN/'local-qwen-manifest.json',RUN/'local-qwen-original2b'):
                raise RuntimeError('All-mode prediction freeze incomplete')
            GOLD_READ=True
            q.score(RUN/'local-qwen-manifest.json',RUN/'local-qwen-original2b')
            reporter=ROOT/'scripts/report_ust_sg_modes_20261024.py'
            if q.sha(reporter)!='77bbb370e6d7dfb94f0bd3189de8d1d55bedcea4b0efa2359b4e9f596253a5b9':
                raise RuntimeError('Reporter source changed')
            subprocess.run([sys.executable,str(reporter)],cwd=ROOT,check=True)
            status('COMPLETE_FROZEN_SCORED_SMALLER_QUEUE_HELD_FOR_GAME',saved=len(saved))
            return
        if len(saved)<3:
            task=task_order[len(saved)]
        else:
            available=[t for t in ['USTBench','SGQA'] if t not in completed_tasks]
            task=available[(len(saved)-3)%len(available)]
        attempts+=1
        log=PROFILE/'worker-logs'/f'{attempts:05d}-{int(time.time())}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        status('STARTING_ONE_QUESTION_OR_RESOURCE_CHECK',saved=len(saved),task=task)
        with log.open('w') as output:
            child=subprocess.Popen([sys.executable,str(WRAPPER),'--once',task],
                cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    if STOP or time.time()>=DEADLINE:
                        terminate(child)
                        status('STOPPED_ORIGINAL_DEADLINE_OR_USER',saved=len(committed()),
                            interrupted_item_not_resampled=True)
                        return
                    status('WORKER_ACTIVE',task=task,saved=len(committed()),pid=child.pid,log=str(log))
                    time.sleep(min(30,max(0.1,DEADLINE-time.time())))
            finally:
                terminate(child)
        lines=log.read_text(errors='replace').splitlines()
        result=None
        for line in reversed(lines):
            try:
                v=json.loads(line)
                if isinstance(v,dict) and 'status' in v:
                    result=v
                    break
            except ValueError:
                pass
        if child.returncode!=0 or result is None:
            status('STOPPED_WORKER_FAILURE_NO_RETRY',returncode=child.returncode,log=str(log))
            return
        state=result['status']
        status(state,worker=result,saved=len(committed()))
        if state=='TASK_COMPLETE_NO_NEW_GENERATION':
            completed_tasks.add(task)
        elif state=='ONE_RESPONSE_SAVED_GPU_PROCESS_EXIT':
            pass
        elif state in ('WAITING_COOLDOWN','WAITING_RESOURCE_NO_RESPONSE'):
            pass
        elif state=='PILOT_LIMIT_REACHED':
            status('PILOT_COMPLETE_WAITING_CONTINUATION',saved=len(committed()))
            return
        else:
            status('STOPPED_WORKER_TERMINAL',worker=result,saved=len(committed()))
            return
        until=time.time()+(60 if result.get('generated') else 30)
        while not STOP and time.time()<min(until,DEADLINE):
            time.sleep(min(10,max(.1,min(until,DEADLINE)-time.time())))

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('STOPPED_SUPERVISOR_FAILURE_NO_RETRY',reason=str(exc))
        raise



