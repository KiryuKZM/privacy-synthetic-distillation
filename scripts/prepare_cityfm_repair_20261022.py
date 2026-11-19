#!/usr/bin/env python3
"""Prepare paired old/repaired Dev32 locally. No credentials/network/model load."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from privacy_synth.cityfm_representation import (build_payload, compact_payload,
                                               expand_neighbors, metric_polygon,
                                               CONTEXT_KEYS, PHYSICAL_KEYS)
from privacy_synth.location_classification import (digest, prompt, score, recompute_labels,
                                                 SEMANTIC_KEYS, SEMANTIC_VALUES)

REV = '1bf4985c7e86665bb38818cd0e443914bfbb7e46'
CASES = ('117919999', '241209957', '172245189', '159638789', '188884827', '172769457')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n'
                            for r in rows), encoding='utf-8')


def token_count(tokenizer, request):
    rendered = '<｜begin▁of▁sentence｜>'+request['system']+'<｜User｜>'+request['user']+'<｜Assistant｜>'
    return len(tokenizer.encode(rendered, add_special_tokens=False).ids)


def prepare_fixed_screen(raw, role_freeze, output):
    """Serialize preselected fresh-screen IDs only; do not read screen gold."""
    from pyproj import Transformer
    from tokenizers import Tokenizer
    if output.exists():
        raise RuntimeError('Immutable output exists; choose a new output namespace')
    freeze = load(role_freeze)
    ids = freeze['roles']['CityFM-SG-GeoRepairScreen96']['ids']
    assert len(ids) == len(set(ids)) == 96
    assert digest(ids) == freeze['roles']['CityFM-SG-GeoRepairScreen96']['sha256']
    targets = load(raw/'buildings_untagged.json')
    entities = load(raw/'entities.json')
    contexts = load(raw/'buildings_context.json')
    tokenizer_path = ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3414', always_xy=True)
    requests = []
    audits = []
    payloads = []
    for k in ids:
        payload, audit = build_payload(k, targets[k], contexts[k], entities, transformer)
        request = prompt(payload, 0)
        tokens = token_count(tokenizer, request)
        encoding = 'expanded'
        if tokens > 8192:
            compressed = compact_payload(payload)
            assert expand_neighbors(compressed) == expand_neighbors(payload)
            candidate = prompt(compressed, 0)
            compressed_tokens = token_count(tokenizer, candidate)
            if compressed_tokens < tokens:
                payload, request, tokens = compressed, candidate, compressed_tokens
                encoding = 'lossless_semantic_profiles'
        requests.append({'id': k, 'condition': 'repaired-input_same-prompt0', 'request': request,
                         'request_sha256': digest(request), 'input_tokens': tokens, 'output_cap': 64})
        audits.append({'id': k, **audit, 'encoding': encoding, 'input_tokens': tokens})
        payloads.append({'id': k, 'payload': payload})
    output.mkdir(parents=True)
    jsonl(output/'inputs.jsonl', payloads)
    jsonl(output/'requests.proposed.jsonl', requests)
    write(output/'input-audits.local.json', audits)
    over = [{'id': r['id'], 'tokens': r['input_tokens']} for r in requests if r['input_tokens'] > 8192]
    summary = {'status': 'STOP_OVER8192' if over else 'PREPARED_PENDING_NEW_WORKLOAD_APPROVAL',
               'n': 96, 'membership_sha256': digest(ids), 'role_freeze_file_sha256': sha(role_freeze),
               'screen_gold_read': False, 'old_sealed_read': False, 'model_calls': 0, 'gpu_used': False,
               'input_tokens': {'sum': sum(r['input_tokens'] for r in requests),
                                'max': max(r['input_tokens'] for r in requests)},
               'over8192': over, 'all_retained_neighbor_positions_and_semantics_preserved': True,
               'historical_old_logistic_fit_exposure': 'All96 belong to historical fit; fresh baseline must exclude the whole predeclared screen pool and be retrained.'}
    write(output/'validation-summary.json', summary)
    write(output/'external-workpackage.proposed.json', {
        'status': 'BLOCKED_OVER8192' if over else 'AWAITING_EXPLICIT_NEW_MODEL_BUDGET_APPROVAL',
        'approval': None, 'provider': 'DeepSeek official', 'model': 'deepseek-flash thinking-off',
        'calls_no_retries': 96, 'input_tokens_local_wrapper_estimate': summary['input_tokens']['sum'],
        'output_tokens_max': 96*64, 'input_max': 8192, 'output_max': 64,
        'data': 'Public proxy only: target physical tags and relative unsimplified footprint, independent public neighbor names/raw semantics/relative positions; no target identity/use/absolute coordinates/gold.',
        'expected_cost': 'Root to verify current provider price and concrete budget route before approval.',
        'cheaper_alternative': 'Local sparse name+semantic+physical structured baseline on fresh fit partition; no model API needed for scoring or schema audit.',
        'http_capability': False, 'historical_approval_not_reused': True})
    write(output/'artifact-manifest.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.glob('*')) if p.is_file()})
    return summary


def prepare(raw, historical, output):
    from pyproj import Transformer
    from tokenizers import Tokenizer
    if output.exists():
        raise RuntimeError('Immutable output exists; choose a new output namespace')
    output.mkdir(parents=True)
    targets = load(raw/'buildings_untagged.json')
    entities = load(raw/'entities.json')
    contexts = load(raw/'buildings_context.json')
    labels = load(raw/'land_use.json')
    with (raw/'land_use_data.csv').open(encoding='utf-8-sig', newline='') as f:
        recomputed, _ = recompute_labels(csv.DictReader(f))
    dev = load(historical/'classification/dev-variant0.json')
    approved = load(historical/'workpackages-v2/cityfm_dev-requests.json')
    old = {r['id']: r for r in approved if r['variant'] == 0}
    ids = dev['ids']
    assert len(ids) == len(set(ids)) == 32
    assert all(labels[k] == recomputed[k] for k in ids)
    assert score([labels[k] for k in ids], dev['outputs']) == dev['scores']
    tokenizer_path = ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3414', always_xy=True)
    pairs = []
    audits = []
    requests = []
    stats = Counter()
    for k in ids:
        assert digest(old[k]['request']) == old[k]['request_sha256']
        previous = json.loads(old[k]['request']['user'])
        payload, audit = build_payload(k, targets[k], contexts[k], entities, transformer)
        request = prompt(payload, 0)
        input_tokens = token_count(tokenizer, request)
        encoding = 'expanded'
        if input_tokens > 8192:
            compressed = compact_payload(payload)
            assert expand_neighbors(compressed) == expand_neighbors(payload)
            candidate = prompt(compressed, 0)
            compressed_tokens = token_count(tokenizer, candidate)
            if compressed_tokens < input_tokens:
                payload, request, input_tokens = compressed, candidate, compressed_tokens
                encoding = 'lossless_semantic_profiles'
        # Model-visible dictionaries have no target source identity or semantics.
        assert set(payload['target']['physical_tags']) <= set(PHYSICAL_KEYS)
        assert len(payload['target']['footprint_xy_m']) == len(metric_polygon(targets[k], transformer).exterior.coords)
        retained_ids = {a['id'] for a in audit['retained']}
        expanded = expand_neighbors(payload)
        assert len(expanded) == payload['total_neighbors'] == len(retained_ids)
        expected_semantics = Counter(tuple(sorted((key, str(value)) for key, value in entities[nid].get('tags', {}).items()
                                                  if key in CONTEXT_KEYS)) for nid in retained_ids)
        assert Counter(tuple(sorted(row[0].items())) for row in expanded) == expected_semantics
        assert all(row[3] <= 100.1 for row in expanded)
        restored_other = sum(key in SEMANTIC_KEYS and str(value).lower().strip() not in SEMANTIC_VALUES
                             for nid in retained_ids for key, value in entities[nid].get('tags', {}).items())
        stats.update({'native_associations': len(contexts[k]), 'old_retained': previous['total_neighbors'],
                      'repaired_retained': payload['total_neighbors'], 'retained_names': audit['names_retained'],
                      'retained_healthcare': audit['healthcare_retained'], 'raw_values_previously_other': restored_other,
                      'targets_with_levels': 'building:levels' in payload['target']['physical_tags'],
                      'targets_with_height': 'height' in payload['target']['physical_tags'],
                      'old_empty': previous['total_neighbors'] == 0, 'repaired_empty': payload['total_neighbors'] == 0})
        pairs.append({'id': k, 'old_payload': previous, 'repaired_payload': payload,
                      'old_request_sha256': old[k]['request_sha256'],
                      'repaired_request_sha256': digest(request), 'gold_local': labels[k],
                      'old_output': dict(zip(ids, dev['outputs']))[k]})
        audits.append({'id': k, **audit, 'encoding': encoding, 'input_tokens': input_tokens,
                       'raw_values_previously_other': restored_other})
        requests.append({'id': k, 'condition': 'repaired-input_same-prompt0',
                         'request': request, 'request_sha256': digest(request),
                         'input_tokens': input_tokens, 'output_cap': 64})
    byid = {p['id']: p for p in pairs}
    case_diffs = []
    for k in CASES:
        p = byid[k]
        old_neighbor_semantics = {c for n in p['old_payload']['nearest_neighbors'] for c in n['categories']}
        case_diffs.append({**p, 'old_visible_categories': sorted(old_neighbor_semantics),
                           'repaired_full_context': expand_neighbors(p['repaired_payload']),
                           'local_provenance': next(a for a in audits if a['id'] == k)})
    jsonl(output/'paired-inputs.local.jsonl', pairs)
    jsonl(output/'repaired-requests.proposed.jsonl', requests)
    write(output/'input-audits.local.json', audits)
    write(output/'five-errors-one-correct.local.json', case_diffs)
    over = [{'id': r['id'], 'tokens': r['input_tokens']} for r in requests if r['input_tokens'] > 8192]
    summary = {'n': 32, 'label_matches': 32, 'historical_request_hash_matches': 32,
               'old_score_recomputed_identical': True, 'historical_correct': dev['scores']['correct'],
               'restoration_assertions_passed': True, 'statistics': dict(stats),
               'input_tokens': {'sum': sum(r['input_tokens'] for r in requests),
                                'min': min(r['input_tokens'] for r in requests),
                                'max': max(r['input_tokens'] for r in requests)},
               'over8192': over, 'model_calls': 0, 'gpu_used': False,
               'sealed_access': False, 'screen_access': False,
               'source_revision': REV, 'source_hashes': {name: sha(raw/name) for name in
                 ('buildings_untagged.json', 'entities.json', 'buildings_context.json', 'land_use.json', 'land_use_data.csv')},
               'tokenizer_sha256': sha(tokenizer_path),
               'status': 'STOP_OVER8192' if over else 'PREPARED_PENDING_NEW_WORKLOAD_APPROVAL'}
    write(output/'validation-summary.json', summary)
    package = {'status': 'BLOCKED_OVER8192' if over else 'AWAITING_EXPLICIT_NEW_MODEL_BUDGET_APPROVAL',
               'approval': None, 'provider': 'DeepSeek official', 'model': 'deepseek-flash thinking-off',
               'calls_no_retries': 32, 'conditions': ['repaired-input_same-prompt0'],
               'input_tokens_local_wrapper_estimate': summary['input_tokens']['sum'],
               'output_tokens_max': 32*64, 'input_max': 8192, 'output_max': 64,
               'expected_cost': 'Not estimated without current verified provider price; root to approve concrete route.',
               'data': 'Public CityFM proxy only: local relative target footprint and physical tags; independent 100m public context raw names and semantic values, including healthcare; all relative context positions. No target name/use/address/wiki/absolute ID/absolute coordinates; no gold.',
               'cheaper_alternative': 'Zero-API local schema/provenance/scoring verification completed; structured fit-only classifiers can diagnose this route, but do not establish LLM teacher quality.',
               'historical_approval_not_reused': True, 'http_capability': False,
               'token_estimate_note': 'Cached DS tokenizer and declared wrapper, actual API accounting may differ.'}
    write(output/'external-workpackage.proposed.json', package)
    write(output/'artifact-manifest.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.glob('*')) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw/cityfm-singapore'/REV)
    parser.add_argument('--historical', type=Path, default=ROOT/'runs/2026-10-22-night-dual')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/2026-10-22-cityfm-input-repair/dev32')
    parser.add_argument('--fixed-screen-role-freeze', type=Path,
                        help='Prepare only frozen repair Screen96 identities, without reading gold')
    args = parser.parse_args()
    result = (prepare_fixed_screen(args.raw, args.fixed_screen_role_freeze, args.output)
              if args.fixed_screen_role_freeze else prepare(args.raw, args.historical, args.output))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
