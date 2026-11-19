#!/usr/bin/env python3
"""Original local Qwen, fixed UST/SGQA messages, practical off/on budgets.

Preflight imports only the tokenizer. Run never opens gold. Score requires the
whole selected task x mode experiment frozen. No model/API calls at import.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
DEFAULT_MODEL = ROOT / 'artifacts/hf-home/hub/models--Qwen--Qwen3.5-2B/snapshots' / REVISION
CAPS = {'off': 256, 'on': 8192}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, obj):
    """Create once; a resume may reproduce identical JSON but cannot overwrite."""
    path = Path(path)
    if path.exists():
        if load(path) != obj:
            raise PermissionError('Immutable output changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def local_path(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def bound(spec):
    path = local_path(spec['path'])
    if sha(path) != spec['sha256']:
        raise PermissionError('Bound file changed: ' + str(path))
    return path


def manifest_rows(manifest_path, task):
    packet = load(manifest_path)
    for path, expected in packet.get('bindings', {}).items():
        if sha(local_path(path)) != expected:
            raise PermissionError('Manifest binding changed: ' + path)
    if 'data_manifest' in packet:
        bound(packet['data_manifest'])
    spec = packet['tasks'][task]
    if spec['scorer'] != {'USTBench': 'ust', 'SGQA': 'sgqa'}[task]:
        raise PermissionError('Unexpected scorer')
    rows = [json.loads(s) for s in bound(spec['queries']).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows) != spec['count'] or len({str(r['id']) for r in rows}) != len(rows):
        raise PermissionError('Fixed denominator / identity mismatch')
    for row in rows:
        m = row['adapted_messages']
        if len(m) != 18 or m[0]['role'] != 'system' or m[-1]['role'] != 'user':
            raise PermissionError('Expected exact fixed8 canonical messages')
        if [x['role'] for x in m[1:-1]] != ['user', 'assistant'] * 8:
            raise PermissionError('Demonstration order changed')
        if any(not isinstance(x['content'], str) for x in m):
            raise PermissionError('Text-only messages required')
    return packet, spec, rows


def model_assets(path):
    path = Path(path)
    if not path.is_dir() or (path / 'adapter_config.json').exists():
        raise PermissionError('Original local checkpoint required; adapters forbidden')
    files = {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()}
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja'}
    if not required <= files.keys() or not any(n.endswith('.safetensors') for n in files):
        raise PermissionError('Incomplete original model/tokenizer checkpoint')
    return files


def setup_native():
    # Reuse established local kernel/header setup; this import loads no weights.
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import eval_qwen35_untuned_30_20261014 as native
    return native


def rendered_input(tokenizer, row, thinking, context_cap):
    text = tokenizer.apply_chat_template(row['adapted_messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking == 'on')
    ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
    if len(ids) + CAPS[thinking] > context_cap:
        raise PermissionError('Context overflow; truncation/replacement forbidden: ' + str(row['id']))
    return text, ids


def split_final(raw, thinking):
    """Only generated final content; never recover an answer from scratchpad."""
    text = raw
    for token in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(token):
            text = text[:-len(token)]
    closed = '</think>' in text
    if (thinking == 'on' and not closed) or ('<think>' in text and not closed):
        return {'text': '', 'reasoning': text, 'closed_thinking': False,
                'empty_visible_reasoning': not text.strip(), 'no_final': True}
    if closed:
        reasoning, final = text.rsplit('</think>', 1)
        reasoning = reasoning.removeprefix('<think>').strip()
    else:
        reasoning, final = '', text
    return {'text': final.strip(), 'reasoning': reasoning, 'closed_thinking': closed,
            'empty_visible_reasoning': not reasoning.strip(), 'no_final': not final.strip()}


def classify(raw, thinking, scorer, metadata, finish_reason='stop'):
    from run_llm_task_sweep_20261023 import parse_task
    final = split_final(raw, thinking)
    parsed = parse_task(final['text'], scorer, metadata)
    if final['no_final']:
        parsed = {'valid': False, 'prediction': None, 'reason': 'NO_GENERATED_FINAL'}
    if finish_reason != 'stop':
        parsed = {'valid': False, 'prediction': None, 'reason': 'LENGTH_TRUNCATION' if finish_reason == 'length' else 'TECHNICAL_FAILURE'}
    return dict(final, **parsed)


def runtime_bindings():
    paths = [Path(__file__), ROOT / 'scripts/eval_qwen35_untuned_30_20261014.py',
        ROOT / 'scripts/run_cityfm_repair_qwen_20261022.py',
        ROOT / 'scripts/night_dual_20261022.py', ROOT / 'scripts/run_llm_task_sweep_20261023.py',
        ROOT / 'scripts/build_llm_task_sweep_qa_20261023.py',
        ROOT / 'src/privacy_synth/pilot_ledger.py', ROOT / 'src/privacy_synth/location_classification.py',
        ROOT / 'src/privacy_synth/spatial_scene.py', ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    paths += sorted((ROOT / 'artifacts/kernels').rglob('*.py'))
    return {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers')}}


def preflight(manifest_path, model_path, output, tasks):
    setup_native()
    from transformers import AutoTokenizer
    packet = load(manifest_path)
    context_cap = min(32768, int(packet.get('local_qwen', {}).get('context_cap', 32768)))
    if packet.get('local_qwen', {}).get('max_new_tokens', CAPS) != CAPS:
        raise PermissionError('Frozen practical output caps differ')
    assets = model_assets(model_path)
    cfg = load(Path(model_path) / 'config.json')
    native_cap = cfg.get('text_config', cfg).get('max_position_embeddings')
    if not isinstance(native_cap, int) or context_cap > native_cap:
        raise PermissionError('Unverified native model context limit')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    audit = {}
    for task in tasks:
        _, _, rows = manifest_rows(manifest_path, task)
        audit[task] = {}
        for thinking in CAPS:
            entries = []
            for row in rows:
                text, ids = rendered_input(tokenizer, row, thinking, context_cap)
                entries.append({'id': str(row['id']), 'messages_sha256': digest(row['adapted_messages']),
                    'metadata_sha256': digest(row.get('metadata', {})),
                    'rendered_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'input_ids_sha256': digest(ids), 'input_tokens': len(ids)})
            audit[task][thinking] = entries
    result = {'manifest_sha256': sha(manifest_path), 'runner_sha256': sha(__file__),
        'model_path': str(Path(model_path).resolve()), 'model_assets': assets, 'tasks': tasks,
        'context_cap': context_cap, 'native_context_cap': native_cap,
        'max_new_tokens': CAPS, 'runtime_bindings': runtime_bindings(), 'audit': audit, 'weights_loaded': False,
        'gpu_used': False, 'input_truncation': False}
    immutable(Path(output) / 'preflight.json', result)
    return {'status': 'PREFLIGHT_PASS', 'audit_sha256': sha(Path(output) / 'preflight.json'),
        'counts': {k: len(v['off']) for k, v in audit.items()},
        'max_input_tokens': max(x['input_tokens'] for v in audit.values() for a in v.values() for x in a),
        'weights_loaded': False, 'gpu_used': False}


def verify_preflight(manifest_path, model_path, output):
    audit = load(Path(output) / 'preflight.json')
    if audit['manifest_sha256'] != sha(manifest_path) or audit['runner_sha256'] != sha(__file__):
        raise PermissionError('Preflight source/manifest changed')
    if audit['runtime_bindings'] != runtime_bindings():
        raise PermissionError('Pinned local runtime or reused implementation changed')
    if audit['model_path'] != str(Path(model_path).resolve()) or audit['model_assets'] != model_assets(model_path):
        raise PermissionError('Model/tokenizer asset hash mismatch')
    return audit


def prediction_path(output, task, thinking, rid):
    return Path(output) / task / thinking / 'responses' / (digest(str(rid)) + '.json')


def freeze_all(manifest_path, output):
    """Gold stays unopened until every predeclared task/mode is fixed."""
    audit_path = Path(output) / 'preflight.json'
    audit = load(audit_path)
    specs = []
    for task in audit['tasks']:
        _, _, rows = manifest_rows(manifest_path, task)
        for thinking in CAPS:
            for row in rows:
                path = prediction_path(output, task, thinking, row['id'])
                if not path.is_file():
                    return False
                pred = load(path)
                if pred['id'] != str(row['id']) or pred['task'] != task or pred['thinking'] != thinking:
                    raise PermissionError('Prediction identity mismatch')
                if pred['manifest_sha256'] != sha(manifest_path) or pred['preflight_sha256'] != sha(audit_path):
                    raise PermissionError('Prediction binding mismatch')
                specs.append({'path': str(path.resolve()), 'sha256': sha(path)})
    immutable(Path(output) / 'predictions-frozen.json', {'manifest_sha256': sha(manifest_path),
        'preflight_sha256': sha(audit_path), 'complete': True, 'expected': len(specs), 'predictions': specs})
    return True


def run(manifest_path, model_path, task, thinking, output):
    import signal
    audit = verify_preflight(manifest_path, model_path, output)
    if task not in audit['tasks']:
        raise PermissionError('Task was not frozen in this local experiment')
    _, spec, rows = manifest_rows(manifest_path, task)
    from run_cityfm_repair_qwen_20261022 import wait_resources, resources_ok, probe, HISTORICAL
    from night_dual_20261022 import lock, cgroup_limits_verified, StopPhase
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    Path(output).mkdir(parents=True, exist_ok=True)
    with lock(Path(output) / 'inference.lock'), lock(HISTORICAL / 'locks/gpu.lock'):
        cgroup_limits_verified()
        pending = [r for r in rows if not prediction_path(output, task, thinking, r['id']).exists()]
        if (Path(output) / 'predictions-frozen.json').exists() and pending:
            raise PermissionError('Frozen predictions forbid more generation')
        if not pending:
            return {'status': 'CACHED_COMPLETE', 'all_predictions_frozen': freeze_all(manifest_path, output)}
        # Interrupted generation is retained as unresolved, never silently resampled.
        for row in pending:
            if prediction_path(output, task, thinking, row['id']).with_suffix('.started.json').exists():
                raise StopPhase('Unresolved started item requires review; automatic resampling forbidden')
        wait_resources(output, stop=lambda: stopped[0])
        verify_preflight(manifest_path, model_path, output)
        native = setup_native()
        import torch
        import transformers
        torch.set_num_threads(4)
        if torch.cuda.device_count() != 1:
            raise StopPhase('Exactly one visible GPU required')
        torch.cuda.set_per_process_memory_fraction(9 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        cfg = load(Path(model_path) / 'config.json')
        cls = transformers.AutoModelForImageTextToText if cfg.get('model_type') in ('qwen3_5', 'qwen3_5_moe') else transformers.AutoModelForCausalLM
        model = cls.from_pretrained(str(model_path), dtype=torch.bfloat16, device_map='cuda',
            attn_implementation='sdpa', local_files_only=True, trust_remote_code=False)
        if any('lora' in n.lower() for n, _ in model.named_parameters()):
            raise PermissionError('Adapter detected')
        model.eval()
        model.requires_grad_(False)
        stops = native.stop_ids(tokenizer)
        expected_rows = {r['id']: r for r in audit['audit'][task][thinking]}
        for index, row in enumerate(rows):
            path = prediction_path(output, task, thinking, row['id'])
            if path.exists():
                old = load(path)
                if old['manifest_sha256'] != sha(manifest_path) or old['preflight_sha256'] != sha(Path(output) / 'preflight.json'):
                    raise PermissionError('Cached prediction changed')
                continue
            if stopped[0]:
                raise StopPhase('Stopped at immutable saved-item boundary')
            if not resources_ok(probe(output), startup=False)[0]:
                raise StopPhase('Host resource floor reached; saved items retained')
            text, ids = rendered_input(tokenizer, row, thinking, audit['context_cap'])
            expected = expected_rows[str(row['id'])]
            if digest(ids) != expected['input_ids_sha256'] or hashlib.sha256(text.encode()).hexdigest() != expected['rendered_sha256']:
                raise PermissionError('Native rendering changed after preflight')
            immutable(path.with_suffix('.started.json'), {'manifest_sha256': sha(manifest_path),
                'preflight_sha256': sha(Path(output) / 'preflight.json'), 'id': str(row['id'])})
            inputs = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors='pt').to('cuda')
            torch.cuda.synchronize()
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, num_beams=1,
                        max_new_tokens=CAPS[thinking], eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                emitted = generated[0, len(ids):].tolist()
                raw = tokenizer.decode(emitted, skip_special_tokens=False)
                finish_reason = 'stop' if emitted and emitted[-1] in stops else 'length'
                parsed = classify(raw, thinking, spec['scorer'], row.get('metadata', {}), finish_reason)
                record = dict(parsed, id=str(row['id']), task=task, thinking=thinking,
                    manifest_sha256=sha(manifest_path), preflight_sha256=sha(Path(output) / 'preflight.json'),
                    raw_generated=raw, output_ids=emitted, input_tokens=len(ids), output_tokens=len(emitted),
                    generation_seconds=elapsed, tokens_per_second=len(emitted) / elapsed,
                    finish_reason=finish_reason, output_truncated=finish_reason == 'length',
                    input_truncated=False, decoding='greedy', max_new_tokens=CAPS[thinking],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), index=index)
                immutable(path, record)
                del generated, inputs
            except torch.OutOfMemoryError:
                immutable(path, {'id': str(row['id']), 'task': task, 'thinking': thinking,
                    'manifest_sha256': sha(manifest_path), 'preflight_sha256': sha(Path(output) / 'preflight.json'),
                    'text': '', 'valid': False, 'prediction': None, 'error': 'CUDA_OOM',
                    'generation_seconds': time.monotonic() - started, 'input_tokens': len(ids),
                    'output_tokens': 0, 'finish_reason': 'technical_failure'})
                raise StopPhase('CUDA_OOM saved invalid; no automatic retry')
        return {'status': 'COMPLETE', 'task': task, 'thinking': thinking,
            'all_predictions_frozen': freeze_all(manifest_path, output)}


def score(manifest_path, output):
    from run_llm_task_sweep_20261023 import score_prediction
    from night_dual_20261022 import lock
    with lock(Path(output) / 'inference.lock'):
        frozen = load(Path(output) / 'predictions-frozen.json')
        if frozen['manifest_sha256'] != sha(manifest_path) or frozen.get('complete') is not True:
            raise PermissionError('Complete prediction freeze required before gold')
        for item in frozen['predictions']:
            if sha(item['path']) != item['sha256']:
                raise PermissionError('Frozen prediction changed')
        summaries = {}
        for task in load(Path(output) / 'preflight.json')['tasks']:
            _, spec, rows = manifest_rows(manifest_path, task)
            gold = load(bound(spec['gold']))
            for thinking in CAPS:
                items = [load(prediction_path(output, task, thinking, r['id'])) for r in rows]
                correct, valid = 0, 0
                for row, pred in zip(rows, items):
                    c, v = score_prediction(pred, {'kind': 'qa', 'scorer': spec['scorer']},
                        gold[str(row['id'])], row.get('metadata', {}))
                    correct += c
                    valid += v
                summaries[task + '/' + thinking] = {'correct': correct, 'total': len(rows),
                    'accuracy': correct / len(rows), 'valid': valid,
                    'length_capped': sum(p['finish_reason'] == 'length' for p in items),
                    'no_final': sum(p.get('no_final', not p.get('text')) for p in items),
                    'empty_visible_reasoning': sum(p.get('empty_visible_reasoning', False) for p in items),
                    'input_tokens': sum(p['input_tokens'] for p in items),
                    'output_tokens': sum(p['output_tokens'] for p in items),
                    'generation_seconds': sum(p['generation_seconds'] for p in items)}
        result = {'manifest_sha256': sha(manifest_path),
            'prediction_freeze_sha256': sha(Path(output) / 'predictions-frozen.json'), 'scores': summaries}
        immutable(Path(output) / 'scores.json', result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--task', choices=('USTBench', 'SGQA'))
    p.add_argument('--thinking', choices=('off', 'on'))
    p.add_argument('--experiment-tasks', nargs='+', choices=('USTBench', 'SGQA'), default=['USTBench', 'SGQA'])
    p.add_argument('--output', type=Path, required=True)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--preflight-zero-gpu', action='store_true')
    action.add_argument('--score', action='store_true')
    args = p.parse_args()
    if len(set(args.experiment_tasks)) != len(args.experiment_tasks):
        p.error('Duplicate experiment task')
    if args.preflight_zero_gpu:
        result = preflight(args.manifest, args.model_path, args.output, args.experiment_tasks)
    elif args.score:
        result = score(args.manifest, args.output)
    else:
        if not args.task or not args.thinking:
            p.error('Generation requires --task and --thinking')
        result = run(args.manifest, args.model_path, args.task, args.thinking, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

