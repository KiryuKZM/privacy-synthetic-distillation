#!/usr/bin/env python3
"""Explicit gaming resource profile: one original-Qwen ON answer per process.

No checkpoint/prompt/decoder/scorer changes. The new resource policy and wrapper
binding are recorded in every new response. Preparation and tests use no GPU.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
import eval_ust_sg_qwen_modes_20261024 as q

ROOT=q.ROOT
RUN=ROOT/'runs/2026-10-24-ust100-sg60-modes'
OUT=RUN/'local-qwen-original2b'
PROFILE=RUN/'gaming-profile-v1'
CONTRACT=PROFILE/'contract.json'
MANIFEST=RUN/'local-qwen-manifest.json'
DEADLINE=1792816127.9174702
RUNNER_SHA='ac80fea5b3af5257dae954db3c4ce7b71503cd1a6fcb79323cc771ce64f7c821'
MANIFEST_SHA='4f24de8cd9cba801f94d98d02a740eec8adc1dcf95f2cee3a9b5f48e45d7e4c4'
PREFLIGHT_SHA='f457f2672d2098e4cd8b19bd81ba8751b5df3704104a37c3b271fa5b26ca8236'
PILOT_LIMIT=3


class GamingBatchSaved(Exception):
    def __init__(self,path,record):
        self.path,self.record=Path(path),record


class GamingWaiting(Exception):
    pass


class GamingTechnicalStop(Exception):
    pass


class SleepCriteria:
    """Cooperative delay after generated tokens; never changes the stop decision."""
    def __init__(self,synchronize,sleep=time.sleep,seconds=.1):
        self.synchronize,self.sleep,self.seconds=synchronize,sleep,seconds
        self.calls=0
    def __call__(self,input_ids,scores,**kwargs):
        self.synchronize()
        self.sleep(self.seconds)
        self.calls+=1
        return False


def resources_ok(snapshot,startup=True):
    checks={'windows_ram':snapshot.get('windows_free_gib',-1)>=20,
            'windows_disk':snapshot.get('c_free_gib',-1)>=20,
            'workspace_disk':snapshot.get('linux_free_gib',-1)>=20}
    if startup:checks['gpu_free']=snapshot.get('gpu_free_gib',-1)>=8.5
    return all(checks.values()),checks


def contract_value():
    return {'name':'gaming-cooperative-v1','wrapper_sha256':q.sha(__file__),
        'original_runner_sha256':RUNNER_SHA,'manifest_sha256':MANIFEST_SHA,'preflight_sha256':PREFLIGHT_SHA,
        'deadline_epoch':DEADLINE,'pilot_new_response_limit':PILOT_LIMIT,
        'maximum_with_bound_continuation':160,'continuation_receipt':'gaming-profile-v1/continuation.json','thinking':'on',
        'max_new_tokens':8192,'greedy':True,'batch_size':1,'questions_per_process':1,
        'minimum_between_questions_seconds':60,'cpu_quota_percent':100,'torch_threads':1,'nice_minimum':10,
        'cuda_allocator_cap_gib':6.5,'start_gpu_free_gib':8.5,'gpu_utilization_idle_required':False,
        'windows_ram_floor_gib':20,'disk_floor_gib':20,'sleep_seconds_per_generated_token':.1,
        'stopping_criterion_always_false':True,'unload_on_process_exit':True,
        'hard_gpu_percent_or_fps_guarantee':False,'prefill_may_spike':True,
        'execution_policy_amended':True,'prompts_models_caps_scorers_unchanged':True,'api_calls':0}


def prepare():
    q.immutable(CONTRACT,contract_value())
    return {'status':'PREPARED','contract_sha256':q.sha(CONTRACT),'gpu_used':False}


def verify_contract():
    value=q.load(CONTRACT)
    if value!=contract_value():raise PermissionError('Gaming profile source/contract changed')
    for path,expected in [(q.__file__,RUNNER_SHA),(MANIFEST,MANIFEST_SHA),(OUT/'preflight.json',PREFLIGHT_SHA)]:
        if q.sha(path)!=expected:raise PermissionError('Original frozen runner/inputs changed')
    receipt=q.load(OUT/'queue-start.json')
    if receipt['deadline_epoch']!=DEADLINE:raise PermissionError('Original deadline changed')
    return value


def existing_profile_responses():
    results=[]
    for path in OUT.glob('*/on/responses/*.json'):
        if path.name.endswith('.started.json'):continue
        record=q.load(path)
        if record.get('execution_profile',{}).get('contract_sha256')==q.sha(CONTRACT):
            results.append((path,record))
    return results


def effective_limit(saved):
    path=PROFILE/'continuation.json'
    if not path.exists():return PILOT_LIMIT,None
    receipt=q.load(path)
    pilot=sorted(saved,key=lambda item:item[1]['execution_profile']['completed_epoch'])[:3]
    if (receipt.get('authorized_by')!='root/user_slow_local_request'
        or receipt.get('contract_sha256')!=q.sha(CONTRACT)
        or receipt.get('max_total_new_responses')!=160 or receipt.get('deadline_epoch')!=DEADLINE
        or len(pilot)!=3):
        raise PermissionError('Continuation is not bound to the original three-item gaming pilot')
    expected=[{'path':str(path.relative_to(ROOT)),'sha256':q.sha(path)} for path,_ in pilot]
    if receipt.get('pilot_responses')!=expected:raise PermissionError('Pilot response hashes changed')
    for _,record in pilot:
        meta=record.get('execution_profile',{});actual=meta.get('actual_overrides',{})
        if (record.get('error') or meta.get('contract_sha256')!=q.sha(CONTRACT)
            or meta.get('wrapper_source_sha256')!=q.sha(__file__)
            or actual.get('actual_torch_threads')!=1 or actual.get('actual_cuda_allocator_cap_gib',100)>6.5
            or actual.get('throttle_calls',0)<=0 or actual.get('max_new_tokens')!=8192
            or actual.get('do_sample') is not False or not 0<record.get('peak_reserved_bytes',0)<=6.5*2**30):
            raise PermissionError('Pilot safety/provenance checks did not pass; no continuation')
    return 160,q.sha(path)


def pending(task):
    _,_,rows=q.manifest_rows(MANIFEST,task)
    for marker in OUT.glob('*/*/responses/*.started.json'):
        if not marker.with_name(marker.name.replace('.started.json','.json')).exists():
            raise PermissionError('Unresolved started item; automatic resampling forbidden')
    return next((row for row in rows if not q.prediction_path(OUT,task,'on',row['id']).exists()),None)


def enforce_own_cpu():
    from night_dual_20261022 import cgroup_limits_verified
    scope=cgroup_limits_verified()
    quota,period=scope['cpu_quota']
    if quota=='max' or int(quota)/int(period)>1:
        raise PermissionError('Gaming worker requires owned CPUQuota<=100%')
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice<10:raise PermissionError('Gaming worker requires Nice>=10')
    return {'cgroup':scope,'nice':nice}


@contextlib.contextmanager
def swap(obj,name,value):
    old=obj.__dict__.get(name)
    existed=name in obj.__dict__
    setattr(obj,name,value)
    try:yield
    finally:
        if existed:setattr(obj,name,old)
        else:delattr(obj,name)


def commit_one(path,obj,stats,writer):
    path=Path(path)
    if path.parent.name!='responses' or path.name.endswith('.started.json'):
        return writer(path,obj)
    annotated=dict(obj,execution_profile={
        'name':'gaming-cooperative-v1','contract_path':str(CONTRACT.relative_to(ROOT)),
        'contract_sha256':q.sha(CONTRACT),'wrapper_source_sha256':q.sha(__file__),
        'original_runner_sha256':RUNNER_SHA,'execution_policy_amended':True,
        'completed_epoch':time.time(),'actual_overrides':dict(stats)})
    writer(path,annotated)
    if obj.get('error'):
        raise GamingTechnicalStop('Technical failure saved invalid; no automatic retry')
    raise GamingBatchSaved(path,annotated)


def install_torch(torch,transformers,stats,stack):
    original_threads=torch.set_num_threads
    original_fraction=torch.cuda.set_per_process_memory_fraction
    def threads(requested):
        stats['requested_torch_threads']=requested;stats['actual_torch_threads']=1
        return original_threads(1)
    def memory(requested,*args,**kwargs):
        total=torch.cuda.get_device_properties(0).total_memory
        cap=6.5*2**30/total
        actual=min(requested,cap)
        stats.update(requested_cuda_fraction=requested,actual_cuda_fraction=actual,
                     actual_cuda_allocator_cap_gib=actual*total/2**30)
        return original_fraction(actual,*args,**kwargs)
    stack.enter_context(swap(torch,'set_num_threads',threads))
    stack.enter_context(swap(torch.cuda,'set_per_process_memory_fraction',memory))
    for name in ('AutoModelForImageTextToText','AutoModelForCausalLM'):
        cls=getattr(transformers,name)
        loader=cls.from_pretrained
        def wrapped_loader(*args,_loader=loader,**kwargs):
            model=_loader(*args,**kwargs)
            generate=model.generate
            def throttled(*args,**kwargs):
                if kwargs.get('max_new_tokens')!=8192 or kwargs.get('do_sample') is not False:
                    raise PermissionError('Frozen thinking decoder changed')
                criterion=SleepCriteria(torch.cuda.synchronize)
                items=list(kwargs.get('stopping_criteria') or [])
                kwargs['stopping_criteria']=transformers.StoppingCriteriaList(items+[criterion])
                try:return generate(*args,**kwargs)
                finally:
                    stats.update(throttle_calls=criterion.calls,
                                 cooperative_sleep_seconds=criterion.calls*.1,
                                 max_new_tokens=kwargs['max_new_tokens'],do_sample=kwargs['do_sample'])
            model.generate=throttled
            return model
        stack.enter_context(swap(cls,'from_pretrained',staticmethod(wrapped_loader)))


@contextlib.contextmanager
def gaming_runtime(stats):
    import run_cityfm_repair_qwen_20261022 as guard
    original_immutable,original_setup=q.immutable,q.setup_native
    with contextlib.ExitStack() as stack:
        def start_check(*args,**kwargs):
            if time.time()>=DEADLINE:raise GamingWaiting('Original deadline reached')
            snapshot=guard.probe(RUN)
            ok,checks=resources_ok(snapshot,True)
            stats.update(start_resource_snapshot=snapshot,start_resource_checks=checks)
            if not ok:raise GamingWaiting('Waiting for 8.5GiB GPU margin and host floors; utilization is unrestricted')
            return snapshot
        def setup():
            # Recheck after the original full checkpoint hash audit and before weights load.
            start_check()
            native=original_setup()
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                os.environ[key]='1'
            import torch
            import transformers
            install_torch(torch,transformers,stats,stack)
            return native
        stack.enter_context(swap(guard,'wait_resources',start_check))
        stack.enter_context(swap(guard,'resources_ok',resources_ok))
        stack.enter_context(swap(q,'setup_native',setup))
        stack.enter_context(swap(q,'immutable',lambda path,obj:commit_one(path,obj,stats,original_immutable)))
        yield


def once(task):
    from night_dual_20261022 import lock,StopPhase
    contract=verify_contract()
    with lock(PROFILE/'pilot.lock'):
        if time.time()>=DEADLINE:return {'status':'ORIGINAL_DEADLINE_REACHED','generated':0}
        if (PROFILE/'terminal-stop.json').exists():return {'status':'PROFILE_STOPPED_NO_RETRY','generated':0}
        saved=existing_profile_responses()
        limit,continuation_sha=effective_limit(saved)
        if len(saved)>=limit:return {'status':'PILOT_LIMIT_REACHED' if limit==3 else 'PROFILE_SCOPE_LIMIT_REACHED','generated':0}
        if saved:
            elapsed=time.time()-max(r['execution_profile']['completed_epoch'] for _,r in saved)
            if elapsed<60:return {'status':'WAITING_COOLDOWN','remaining_seconds':60-elapsed,'generated':0}
        row=pending(task)
        if row is None:return {'status':'TASK_COMPLETE_NO_NEW_GENERATION','generated':0}
        actual_cpu=enforce_own_cpu()
        stats={'cpu_quota_percent':100,'actual_cpu_scope':actual_cpu,'nice_minimum':10,
               'questions_per_process':1,'sleep_seconds_per_generated_token':.1,
               'gpu_utilization_idle_required':False,'start_gpu_free_gib':8.5,
               'minimum_between_questions_seconds':60,'original_deadline_epoch':DEADLINE,
               'continuation_receipt_sha256':continuation_sha,'effective_response_limit':limit}
        try:
            with gaming_runtime(stats):
                q.run(MANIFEST,q.DEFAULT_MODEL,task,'on',OUT)
            return {'status':'NO_NEW_RESPONSE','generated':0}
        except GamingBatchSaved as saved:
            expected=q.prediction_path(OUT,task,'on',row['id'])
            if saved.path!=expected:raise PermissionError('Gaming item differs from first frozen pending item')
            return {'status':'ONE_RESPONSE_SAVED_GPU_PROCESS_EXIT','generated':1,'task':task,'id':str(row['id']),
                    'response_path':str(saved.path),'execution_profile':saved.record['execution_profile'],
                    'input_tokens':saved.record.get('input_tokens'),'output_tokens':saved.record.get('output_tokens'),
                    'generation_seconds':saved.record.get('generation_seconds'),
                    'peak_allocated_bytes':saved.record.get('peak_allocated_bytes'),
                    'peak_reserved_bytes':saved.record.get('peak_reserved_bytes'),'gold_read':False}
        except GamingWaiting as exc:
            return {'status':'WAITING_RESOURCE_NO_RESPONSE','generated':0,'reason':str(exc),'gold_read':False}
        except (GamingTechnicalStop,StopPhase) as exc:
            q.immutable(PROFILE/'terminal-stop.json',{'reason':str(exc),'epoch':time.time(),
                         'contract_sha256':q.sha(CONTRACT),'no_retry':True})
            return {'status':'TECHNICAL_OR_RESOURCE_ABORT_NO_RETRY','reason':str(exc),'gold_read':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--once',choices=('USTBench','SGQA'))
    args=parser.parse_args()
    print(json.dumps(prepare() if args.prepare else once(args.once),ensure_ascii=False))


if __name__=='__main__':main()

