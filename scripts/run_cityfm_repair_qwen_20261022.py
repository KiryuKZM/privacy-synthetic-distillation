#!/usr/bin/env python3
"""Conditional original-Qwen CityFM baseline; preparation/checks are GPU-free.

Reuses the established untouched-native loader/stop tokens, scorer, host probe
and historical GPU lock. Does not change the frozen repaired-API runner/package.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from privacy_synth.location_classification import digest, score, qualification_gate
from prepare_cityfm_repair_20261022 import sha, load
from night_dual_20261022 import atomic, lock, host_snapshot, StopPhase
from run_cityfm_repair_20261022 import require_approval, lines, RUN, HISTORICAL

REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
SNAPSHOT = ROOT/'artifacts/hf-home/hub/models--Qwen--Qwen3.5-2B/snapshots'/REVISION
POLICY = {'gpu_start_free_gib':11, 'gpu_start_util_max_percent':10,
          'gpu_stable_seconds':60, 'windows_free_gib':20, 'disk_free_gib':20,
          'cuda_allocator_gib':9, 'cpu_threads':4, 'output_max_tokens':64,
          'input_max_tokens':8192, 'oom_stop_count':2, 'poll_seconds':20}


def prepare(run=RUN):
    """Hash already cached assets/code; do not initialize a tokenizer/model/GPU."""
    run = Path(run)
    path = run/'qwen-baseline/contract-v1.json'
    if path.exists():
        raise RuntimeError('Immutable Qwen contract already exists')
    original = load(HISTORICAL/'baseline-model.json')
    bindings = {}
    for name, expected in original.items():
        if sha(Path(name)) != expected:
            raise PermissionError('Cached original model changed: '+name)
    for asset in sorted(SNAPSHOT.iterdir()):
        if asset.is_file():
            bindings[str(asset)] = sha(asset)
    if 'chat_template.jinja' not in {Path(p).name for p in bindings}:
        raise RuntimeError('Cached native chat template missing')
    paths = [Path(__file__), ROOT/'scripts/eval_qwen35_untuned_30_20261014.py',
             ROOT/'scripts/night_dual_20261022.py', ROOT/'scripts/run_cityfm_repair_20261022.py',
             ROOT/'src/privacy_synth/location_classification.py',
             run/'workpackages-v1/dev32.json', run/'workpackages-v1/screen96.json',
             run/'protocol/repair-role-freeze.json', run/'protocol/repair-screen96-gold.local.json',
             run/'protocol/fallback-overlap-audit.json', run/'fresh-baseline/result.json',
             HISTORICAL/'baseline-model.json',ROOT/'pyproject.toml',ROOT/'uv.lock']
    paths += sorted((ROOT/'artifacts/kernels').rglob('*.py'))
    bindings.update({str(p):sha(p) for p in paths})
    contract = {'status':'PREPARED_CONDITIONAL_ON_QUALIFIED_NEW_DS_SCREEN',
                'model':'original cached Qwen3.5-2B', 'revision':REVISION,
                'snapshot':str(SNAPSHOT), 'native_vocabulary':True,
                'thinking':'disabled', 'decoding':'greedy', 'do_sample':False,
                'max_new_tokens':64, 'batch_size':1, 'policy':POLICY,
                'gpu_lock':str(HISTORICAL/'locks/gpu.lock'), 'bindings':bindings,
                'new_weights_downloaded':False, 'checkpoint_loaded_during_preparation':False,
                'selection':'exact new Screen96 and Dev-selected approved system/user strings; no prompt reconstruction'}
    atomic(path,contract)
    return {'path':str(path),'sha256':sha(path),'cached_assets':len(list(SNAPSHOT.iterdir())),
            'model_loaded':False,'gpu_used':False}


def validate_teacher_evidence(run, packet):
    """Recompute fixed-denominator qualification from immutable request caches."""
    run = Path(run)
    result_path = run/'results/screen96.json'
    if not result_path.is_file():
        raise PermissionError('Qualified new DS Screen96 receipt is missing')
    result = load(result_path)
    if result.get('package_sha256') != sha(run/'workpackages-v1/screen96.json'):
        raise PermissionError('DS screen receipt package mismatch')
    dev_path = run/'results/dev32.json'
    if not dev_path.is_file():
        raise PermissionError('Dev-selected prompt receipt missing')
    dev = load(dev_path)
    variant = dev.get('selected_variant')
    if variant not in (1,2) or result.get('selected_variant') != variant or not dev.get('dev_gate_pass'):
        raise PermissionError('DS screen does not use the passing Dev-selected prompt')
    roles = load(run/'protocol/repair-role-freeze.json')['roles']
    ids = roles['CityFM-SG-GeoRepairScreen96']['ids']
    rows = [r for r in lines(packet['requests_path']) if r['variant'] == variant]
    if len(ids) != 96 or len(set(ids)) != 96 or result.get('ids') != ids or [r['id'] for r in rows] != ids:
        raise PermissionError('DS/new-Qwen fixed96membership/order mismatch')
    if digest(ids) != roles['CityFM-SG-GeoRepairScreen96']['sha256']:
        raise PermissionError('Screen membership hash mismatch')
    outputs = []
    for row in rows:
        key = f"v{variant}:{row['id']}"
        if packet['allowed_request_sha256'].get(key) != digest(row['request']):
            raise PermissionError('Selected DS request is not the approved exact input')
        cached_path = run/'requests/screen96'/(key+'.json')
        if not cached_path.is_file():
            raise PermissionError('Actual DS response cache missing: '+key)
        cached = load(cached_path)
        if cached.get('request_sha256') != digest(row['request']):
            raise PermissionError('Actual DS response/input hash mismatch')
        outputs.append(cached['text'])
    if outputs != result.get('outputs'):
        raise PermissionError('DS aggregate text differs from actual response caches')
    gold = {r['id']:r['class'] for r in load(run/'protocol/repair-screen96-gold.local.json')}
    scored = score([gold[k] for k in ids],outputs)
    if scored != result.get('scores'):
        raise PermissionError('DS screen score receipt does not recompute')
    baseline = load(run/'fresh-baseline/result.json')
    audit = load(run/'protocol/fallback-overlap-audit.json')
    no_overlap = (audit['post_exclusion_shared_entities'] == audit['fit_screen_grid_overlap_n'] == 0
                  and not audit['crosscell_site_violations'] and not audit['crossrole_target_overlap_violations']
                  and baseline['fit_screen_intersection'] == baseline['fit_whole_pool_intersection'] == 0)
    passed = qualification_gate(scored,baseline['majority']['accuracy'],baseline['logistic']['accuracy'],no_overlap)
    if not passed or result.get('qualification_pass') is not True:
        raise PermissionError('New DS Screen96 failed the predeclared qualification gate')
    return {'rows':rows,'gold':gold,'teacher_scores':scored,'selected_variant':variant,
            'teacher_receipt_sha256':sha(result_path),'membership_sha256':digest(ids)}


def eligibility(run=RUN):
    """Approval/qualification precede model imports, resources and GPU activity."""
    run = Path(run)
    packet = require_approval(run,'screen96')
    evidence = validate_teacher_evidence(run,packet)
    contract_path = run/'qwen-baseline/contract-v1.json'
    if not contract_path.is_file():
        raise PermissionError('New Qwen baseline contract missing')
    contract = load(contract_path)
    for path, expected in contract['bindings'].items():
        if not Path(path).is_file() or sha(Path(path)) != expected:
            raise PermissionError('Bound Qwen/code/inputs changed: '+path)
    return evidence, contract


def resources_ok(snapshot, startup=True):
    required = ('windows_free_gib','c_free_gib','linux_free_gib')
    if not all(key in snapshot for key in required):
        return False, {'missing_host_probe':True}
    checks = {'windows_ram':snapshot['windows_free_gib'] >= 20,
              'windows_disk':snapshot['c_free_gib'] >= 20,
              'workspace_disk':snapshot['linux_free_gib'] >= 20}
    if startup:
        checks.update({'gpu_free':snapshot.get('gpu_free_gib',-1) >= 11,
                       'gpu_idle':0 <= snapshot.get('gpu_util_percent',101) <= 10})
    return all(checks.values()), checks


def probe(run):
    result = host_snapshot(run)  # Existing probe enforces exactly one GPU.
    result['linux_free_gib'] = shutil.disk_usage(run).free/2**30
    return result


def wait_resources(run, probe_fn=probe, clock=time.monotonic, sleep=time.sleep, poll=20, stop=None):
    """Wait without weights/GPU allocation; a SIGTERM saves a resumable boundary."""
    stable = None
    while True:
        if stop and stop():
            raise StopPhase('Resource wait stopped; rerun resumes saved per-item cache')
        try:
            snapshot = probe_fn(run)
            ok, checks = resources_ok(snapshot,True)
        except Exception as error:
            snapshot, ok, checks = {'probe_error_type':type(error).__name__},False,{'probe_failed':True}
        now = clock()
        stable = (now if stable is None else stable) if ok else None
        atomic(Path(run)/'qwen-baseline/resource-status.json',
               {'status':'WAITING_STABLE_RESOURCE','snapshot':snapshot,'checks':checks,
                'stable_seconds':0 if stable is None else now-stable,'model_loaded':False})
        if stable is not None and now-stable >= 60:
            return snapshot
        sleep(min(poll,20))


def run_baseline(run=RUN):
    run = Path(run)
    evidence, contract = eligibility(run)
    folder = run/'qwen-baseline'
    result_path = folder/'summary.json'
    if result_path.exists():
        result = load(result_path)
        if result.get('teacher_receipt_sha256') != evidence['teacher_receipt_sha256'] or result.get('contract_sha256') != sha(folder/'contract-v1.json'):
            raise PermissionError('Saved Qwen summary binding mismatch')
        return result
    import signal
    stopped = [False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    cache = folder/'responses'
    cache.mkdir(parents=True,exist_ok=True)
    contract_hash = sha(folder/'contract-v1.json')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    with lock(run/'locks/qwen-repair.lock'), lock(Path(contract['gpu_lock'])):
        texts = []
        model = tokenizer = torch = None
        oom_count = sum(load(p).get('error') == 'CUDA_OOM' for p in cache.glob('*.json'))
        if oom_count >= 2:
            raise StopPhase('Two saved Qwen OOMs; no automatic restart')
        try:
            for index, row in enumerate(evidence['rows']):
                if stopped[0]:
                    raise StopPhase('Qwen stopped at saved item boundary; rerun resumes cache')
                path = cache/(row['id']+'.json')
                expected = digest(row['request'])
                if path.exists():
                    record = load(path)
                    if record.get('request_sha256') != expected or record.get('contract_sha256') != contract_hash or record.get('teacher_receipt_sha256') != evidence['teacher_receipt_sha256']:
                        raise PermissionError('Qwen cached item/input/model binding mismatch')
                else:
                    if model is not None:
                        ok, _ = resources_ok(probe(run),startup=False)
                        if not ok:
                            del model, tokenizer
                            model = tokenizer = None
                            gc.collect();torch.cuda.empty_cache()
                    if model is None:
                        wait_resources(run,stop=lambda:stopped[0])
                        # Revalidate after a potentially long resource wait.
                        eligibility(run)
                        import torch as torch_module
                        torch = torch_module
                        if torch.cuda.device_count() != 1:
                            raise StopPhase('Exactly one visible GPU required')
                        torch.set_num_threads(4)
                        torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
                        # Existing loader uses the pinned untouched cached native
                        # model, local_files_only, BF16, SDPA, no adapter/training.
                        import eval_qwen35_untuned_30_20261014 as native
                        tokenizer,model,loaded_as = native.load_model()
                        stops = native.stop_ids(tokenizer)
                    request = row['request']
                    rendered = tokenizer.apply_chat_template(
                        [{'role':'system','content':request['system']},{'role':'user','content':request['user']}],
                        add_generation_prompt=True,enable_thinking=False,tokenize=False)
                    inputs = tokenizer(rendered,return_tensors='pt',add_special_tokens=False)
                    length = inputs['input_ids'].shape[1]
                    if length > 8192:
                        raise StopPhase('Native Qwen request exceeds8192; no clipping/replacement')
                    record = {'id':row['id'],'request_sha256':expected,'contract_sha256':contract_hash,
                              'teacher_receipt_sha256':evidence['teacher_receipt_sha256'],'input_tokens':length}
                    try:
                        inputs = inputs.to('cuda')
                        with torch.inference_mode():
                            generated = model.generate(**inputs,max_new_tokens=64,do_sample=False,
                                                       eos_token_id=stops,pad_token_id=tokenizer.pad_token_id)
                        output_ids = generated[0,length:].tolist()
                        record.update(content=tokenizer.decode(output_ids,skip_special_tokens=True),
                                      output_tokens=len(output_ids),loaded_as=loaded_as,
                                      finish_reason='stop' if output_ids and output_ids[-1] in stops else 'length')
                        del generated
                    except torch.OutOfMemoryError:
                        oom_count += 1
                        record.update(content='',output_tokens=0,finish_reason='technical_failure',error='CUDA_OOM')
                        gc.collect();torch.cuda.empty_cache()
                    finally:
                        del inputs
                    atomic(path,record)
                texts.append(record['content'])
                atomic(folder/'progress.json',{'completed':index+1,'n':96,'oom_count':oom_count})
                if oom_count >= 2:
                    raise StopPhase('Two Qwen OOMs; fixed-denominator partial artifacts retained')
            ids = [r['id'] for r in evidence['rows']]
            scored = score([evidence['gold'][k] for k in ids],texts)
            result = {'status':'COMPLETE','n':96,'ids':ids,'scores':scored,
                      'teacher_scores':evidence['teacher_scores'],'selected_variant':evidence['selected_variant'],
                      'teacher_receipt_sha256':evidence['teacher_receipt_sha256'],'contract_sha256':contract_hash,
                      'model':contract['model'],'revision':REVISION,'thinking':'disabled','decoding':'greedy',
                      'max_new_tokens':64,'sealed_used':False,'trained':False,
                      'accuracy_gap_teacher_minus_qwen':evidence['teacher_scores']['accuracy']-scored['accuracy'],
                      'oom_count':oom_count}
            atomic(result_path,result)
            return result
        finally:
            if model is not None:
                del model,tokenizer
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','check','run'])
    parser.add_argument('--run',type=Path,default=RUN)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.run)
    elif args.action == 'check':
        evidence,_ = eligibility(args.run)
        result = {'eligible':True,'n':len(evidence['rows']),'model_loaded':False,'gpu_used':False}
    else:
        result = run_baseline(args.run)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
