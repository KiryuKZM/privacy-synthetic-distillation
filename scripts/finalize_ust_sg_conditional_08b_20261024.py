#!/usr/bin/env python3
"""Deterministic approved conditional 0.8B route; no DS calls or training.

--once returns WAITING until both original full score sets are frozen. Public
weights download only after a frozen passing gate and immutable task selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import eval_ust_sg_qwen_modes_20261024 as q
import report_ust_sg_modes_20261024 as report

ROOT, RUN = report.ROOT, report.RUN
OUT = RUN / 'local-qwen-original08b'
CONTROL = RUN / 'conditional-08b'
DEADLINE = 1792816127.9174702
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
MODEL = 'Qwen/Qwen3.5-0.8B'
SHARD = 'model.safetensors-00001-of-00001.safetensors'
WEIGHT_BYTES = 1746942600
WEIGHT_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
         'model.safetensors.index.json', SHARD]
MODEL_PATH = ROOT / 'artifacts/hf-home/hub/models--Qwen--Qwen3.5-0.8B/snapshots' / REVISION
RUNNER = ROOT / 'scripts/eval_ust_sg_qwen_modes_20261024.py'
MANIFEST = RUN / 'local-qwen-manifest.json'
EXPECTED_RUNNER = 'ac80fea5b3af5257dae954db3c4ce7b71503cd1a6fcb79323cc771ce64f7c821'
EXPECTED_MANIFEST = '4f24de8cd9cba801f94d98d02a740eec8adc1dcf95f2cee3a9b5f48e45d7e4c4'
EXPECTED_REPORTER = '77bbb370e6d7dfb94f0bd3189de8d1d55bedcea4b0efa2359b4e9f596253a5b9'
HEADER = 'kind\tname\tupstream\tlocal_path\tversion_or_revision\tbytes\tsha256\tstatus'


def disk_ok(minimum):
    return all(shutil.disk_usage(path).free >= minimum * 2**30 for path in (ROOT, Path('/mnt/c')))


def frozen_gate():
    packet = q.load(RUN / 'execution-packet.json')
    ds = report.read_scores(ROOT / packet['run_dir'], 'DS')
    original = report.read_scores(RUN / 'local-qwen-original2b', 'Qwen2B')
    if ds['status'] != 'FROZEN_SCORED' or original['status'] != 'FROZEN_SCORED':
        return None
    report.main()  # Existing reporter freezes its exact gate before this selection.
    trigger = report.trigger(ds, original)
    path = RUN / 'smaller-model-trigger.json'
    if q.load(path) != trigger:
        raise PermissionError('Reporter trigger receipt differs')
    return trigger


def source_bindings():
    result = {str(RUNNER): EXPECTED_RUNNER, str(MANIFEST): EXPECTED_MANIFEST,
              str(Path(report.__file__)): EXPECTED_REPORTER, str(Path(__file__)): q.sha(__file__)}
    for path, expected in result.items():
        if q.sha(path) != expected:
            raise PermissionError('Frozen conditional source/input changed')
    return result


def selected_tasks(trigger):
    tasks = trigger.get('qualifying_tasks', [])
    if len(tasks) != len(set(tasks)) or not set(tasks) <= set(report.COUNTS):
        raise PermissionError('Unknown/duplicate qualifying tasks')
    return [task for task in report.COUNTS if task in tasks]


def register_assets(snapshot, root=ROOT):
    """Append public checkpoint provenance only; preserve every old byte."""
    from night_dual_20261022 import lock
    root, snapshot = Path(root), Path(snapshot)
    registry = root / 'sources/assets.tsv'
    cells = []
    for name in FILES:
        path = snapshot / name
        relative = path.relative_to(root).as_posix()
        if not path.resolve().is_relative_to((root / 'artifacts').resolve()):
            raise PermissionError('Downloaded asset resolves outside project artifacts')
        values = ['weight' if name == SHARD else 'asset', MODEL + '/' + name,
            f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}', relative, REVISION,
            str(path.stat().st_size), q.sha(path), 'downloaded; registered before local model use']
        if any(any(c in value for c in '\t\r\n\0') for value in values):
            raise PermissionError('Unsafe registry cell')
        cells.append(values)
    with lock(root / 'sources/.assets-append.lock'):
        prefix = registry.read_bytes()
        if prefix.splitlines()[0].decode('utf-8-sig') != HEADER or not prefix.endswith(b'\n'):
            raise PermissionError('Registry header/newline differs')
        existing = {}
        for line in prefix.decode('utf-8-sig').splitlines()[1:]:
            fields = line.split('\t')
            if len(fields) == 8:
                existing.setdefault(fields[3], []).append(fields)
        added = []
        for values in cells:
            if values[3] in existing:
                if any(row[2:7] != values[2:7] for row in existing[values[3]]):
                    raise PermissionError('Conflicting asset registration')
            else:
                added.append('\t'.join(values))
        suffix = ''.join(line + '\n' for line in added).encode()
        with registry.open('ab') as target:
            target.write(suffix); target.flush(); os.fsync(target.fileno())
        after = registry.read_bytes()
        if after != prefix + suffix:
            raise PermissionError('Registry old prefix was not preserved exactly')
    return {'old_prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'old_prefix_bytes': len(prefix),
        'added_rows': added, 'registered_assets': cells, 'registry_sha256': hashlib.sha256(after).hexdigest()}


def validate_selection():
    selection = q.load(CONTROL / 'selection.json')
    for path, expected in selection['source_bindings'].items():
        if q.sha(path) != expected:
            raise PermissionError('Selected conditional source/input changed')
    trigger = frozen_gate()
    review = report.root_data_review()
    if review is None or q.sha(RUN / 'root-data-review.json') != selection.get('data_review_sha256'):
        raise PermissionError('Final root data review changed or is unresolved')
    numeric = selected_tasks(trigger) if trigger else []
    eligible = [task for task in numeric if review['tasks'][task]['status'] == 'PASS']
    blocked = {task: value.get('reason', '') for task, value in review['tasks'].items() if value['status'] == 'BLOCKED'}
    if (trigger is None or eligible != selection['tasks'] or selection.get('numerical_qualifying_tasks') != numeric
        or selection.get('data_eligible_tasks') != eligible or selection.get('blocked_tasks') != blocked
        or q.sha(RUN / 'smaller-model-trigger.json') != selection['trigger_sha256']):
        raise PermissionError('Frozen numeric/data-eligible task selection changed')
    if selection['deadline_epoch'] != DEADLINE or time.time() >= DEADLINE:
        raise PermissionError('Original local queue deadline reached')
    return selection


def download():
    validate_selection()
    if not disk_ok(25):
        raise RuntimeError('Insufficient disk for download plus preserved 20GiB floor')
    os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DOWNLOAD_TIMEOUT='30',
                      HF_HUB_ETAG_TIMEOUT='15')
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo_id=MODEL, revision=REVISION, token=False,
        cache_dir=str(ROOT / 'artifacts/hf-home/hub'), allow_patterns=FILES,
        max_workers=2, etag_timeout=15))
    if snapshot != MODEL_PATH or not set(FILES) <= {p.name for p in snapshot.iterdir() if p.is_file()}:
        raise PermissionError('Unexpected pinned snapshot path/files')
    weight = snapshot / SHARD
    if weight.stat().st_size != WEIGHT_BYTES or q.sha(weight) != WEIGHT_SHA:
        raise PermissionError('Official 0.8B shard byte count/SHA mismatch')
    if not disk_ok(20):
        raise RuntimeError('Post-download disk floor reached')
    registration = register_assets(snapshot)
    files = {name: {'bytes': (snapshot / name).stat().st_size, 'sha256': q.sha(snapshot / name)} for name in FILES}
    q.immutable(CONTROL / 'download-receipt.json', {'model': MODEL, 'revision': REVISION,
        'snapshot': str(snapshot), 'files': files, 'registered_before_model_use': True,
        'registration': registration, 'selection_sha256': q.sha(CONTROL / 'selection.json')})
    return {'status': 'DOWNLOADED_AND_REGISTERED'}


def owned_command(argv, timeout, logfile):
    """Only this newly created process group may receive deadline signals."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('ab') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = proc.wait(timeout=timeout)
        except BaseException:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
            raise
    if result:
        raise RuntimeError(f'Owned conditional stage exited {result}')


def stage(name, argv, outputs, command=owned_command, timeout=None):
    started, done = CONTROL / 'stages' / (name + '.started.json'), CONTROL / 'stages' / (name + '.done.json')
    if done.exists():
        receipt = q.load(done)
        if receipt['argv'] != argv:
            raise PermissionError('Saved stage command changed')
        for path, expected in receipt['outputs'].items():
            if q.sha(path) != expected:
                raise PermissionError('Completed stage output changed')
        return
    if started.exists():
        raise PermissionError('Unresolved conditional stage; no automatic retry or answer replay')
    remaining = DEADLINE - time.time()
    if remaining < 60 or not disk_ok(20):
        raise RuntimeError('Insufficient remaining time or disk before new stage')
    q.immutable(started, {'argv': argv, 'deadline_epoch': DEADLINE, 'selection_sha256': q.sha(CONTROL / 'selection.json')})
    command(argv, min(remaining, timeout or remaining), CONTROL / 'logs' / (name + '.log'))
    paths = outputs()
    if not paths:
        raise PermissionError('Stage produced no verified outputs')
    q.immutable(done, {'argv': argv, 'outputs': {str(path): q.sha(path) for path in paths}})


def response_outputs(task, mode):
    _, _, rows = q.manifest_rows(MANIFEST, task)
    outputs = []
    for row in rows:
        path = q.prediction_path(OUT, task, mode, row['id'])
        record = q.load(path)
        if (record.get('manifest_sha256') != EXPECTED_MANIFEST or record.get('thinking') != mode
            or record.get('task') != task or record.get('id') != str(row['id'])):
            raise PermissionError('Small model response identity mismatch')
        outputs.append(path)
    return outputs


def once():
    if time.time() >= DEADLINE:
        return {'status': 'ORIGINAL_DEADLINE_REACHED', 'downloaded_now': False}
    if (CONTROL / 'terminal-failure.json').exists():
        return {'status': 'TERMINAL_FAILURE_NO_RETRY', 'downloaded_now': False}
    trigger = frozen_gate()
    if trigger is None:
        return {'status': 'WAITING_FULL_FROZEN_DS_AND_2B', 'downloaded_now': False}
    review = report.root_data_review()
    if review is None:
        return {'status': 'WAITING_FINAL_ROOT_DATA_REVIEW', 'downloaded_now': False}
    numeric = selected_tasks(trigger)
    tasks = [task for task in numeric if review['tasks'][task]['status'] == 'PASS']
    blocked = {task: value.get('reason', '') for task, value in review['tasks'].items() if value['status'] == 'BLOCKED'}
    review_sha = q.sha(RUN / 'root-data-review.json')
    if not tasks:
        q.immutable(CONTROL / 'no-trigger.json', {'trigger_sha256': q.sha(RUN / 'smaller-model-trigger.json'),
            'tasks': [], 'numerical_qualifying_tasks': numeric, 'data_review_sha256': review_sha, 'blocked_tasks': blocked})
        return {'status': 'NO_DATA_ELIGIBLE_TASK' if numeric else 'NO_TASK_QUALIFIES', 'downloaded_now': False}
    source = source_bindings()
    source[str(RUN / 'root-data-review.json')] = review_sha
    minimum = max(7200, 14400 * sum(report.COUNTS[t] for t in tasks) / 160)
    if not (CONTROL / 'selection.json').exists() and (DEADLINE - time.time() < minimum or not disk_ok(25)):
        return {'status': 'WAITING_TIME_OR_DISK_INSUFFICIENT_FOR_NEW_DOWNLOAD', 'downloaded_now': False}
    selection = {'tasks': tasks, 'numerical_qualifying_tasks': numeric, 'data_eligible_tasks': tasks,
        'data_review_sha256': review_sha, 'blocked_tasks': blocked, 'model': MODEL, 'revision': REVISION, 'deadline_epoch': DEADLINE,
        'trigger_sha256': q.sha(RUN / 'smaller-model-trigger.json'), 'source_bindings': source,
        'root_preapproved_conditional_route': True, 'both_modes': True, 'no_training': True,
        'minimum_initial_remaining_seconds': minimum, 'time_allowance_is_not_completion_guarantee': True}
    q.immutable(CONTROL / 'selection.json', selection)
    validate_selection()
    from night_dual_20261022 import cgroup_limits_verified
    cgroup_limits_verified()
    python = str(ROOT / '.venv/bin/python')
    try:
        stage('download', [python, str(Path(__file__)), '--download-only'],
            lambda: [CONTROL / 'download-receipt.json'], timeout=1800)
        receipt = q.load(CONTROL / 'download-receipt.json')
        if not receipt.get('registered_before_model_use') or receipt['selection_sha256'] != q.sha(CONTROL / 'selection.json'):
            raise PermissionError('Checkpoint provenance was not registered before model use')
        args = [python, str(RUNNER), '--manifest', str(MANIFEST), '--model-path', str(MODEL_PATH),
                '--output', str(OUT), '--experiment-tasks', *tasks]
        stage('preflight', args + ['--preflight-zero-gpu'], lambda: [OUT / 'preflight.json'])
        for mode in ('off', 'on'):
            for task in tasks:
                stage(task + '-' + mode, args + ['--task', task, '--thinking', mode],
                    lambda task=task, mode=mode: response_outputs(task, mode))
        stage('score', args + ['--score'], lambda: [OUT / 'scores.json', OUT / 'predictions-frozen.json'])
        report.main()
        q.immutable(CONTROL / 'complete.json', {'tasks': tasks, 'scores_sha256': q.sha(OUT / 'scores.json'),
            'selection_sha256': q.sha(CONTROL / 'selection.json'), 'api_calls': 0})
        return {'status': 'COMPLETE', 'tasks': tasks}
    except BaseException as exc:
        q.immutable(CONTROL / 'terminal-failure.json', {'error_type': type(exc).__name__,
            'error': str(exc), 'deadline_epoch': DEADLINE, 'no_automatic_retry': True})
        raise


def main():
    from night_dual_20261022 import lock
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--once', action='store_true')
    actions.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if args.download_only:
        print(json.dumps(download()))
    else:
        with lock(CONTROL / 'controller.lock'):
            print(json.dumps(once()))


if __name__ == '__main__':
    main()

