#!/usr/bin/env python3
"""Local baseline and exact-package-gated CityFM repair runner.

Import/check/prepare/baseline never touch credentials or the provider. Only run,
after exact new approval, can instantiate the existing official secret loader.
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from privacy_synth.location_classification import digest, score, qualification_gate, parse_label
from privacy_synth.pilot_ledger import SqliteBudget
from privacy_synth.cityfm_representation import build_payload, repaired_baseline_features
from prepare_cityfm_repair_20261022 import sha, load, REV
from night_dual_20261022 import atomic, lock

RUN = ROOT/'runs/2026-10-22-cityfm-input-repair'
HISTORICAL = ROOT/'runs/2026-10-22-night-dual'
RATES = {'hit': .006, 'miss': .3, 'output': 1.2}
OLD_USD = Decimal('0.017907408')
PARENT_USD = Decimal('500')
SPECS = {'dev32': {'calls': 70, 'logical': 64, 'retry': 6},
         'screen96': {'calls': 106, 'logical': 96, 'retry': 10}}


def variant_request(request, variant):
    if variant == 1:
        return request
    if variant != 2:
        raise ValueError('Only two predeclared conditions')
    old = 'Classify the target building planning-use proxy from its physical shape and surrounding public map context.'
    new = ('Infer the CityFM planning-use proxy assigned to this footprint: the largest mapped planning-use overlap, '
           'not necessarily a single containing parcel, the current function of the small structure, or its nearest POI. '
           'Use the supplied physical shape and independent surrounding public map context as evidence; '
           'do not treat a nearby facility as the target automatically.')
    assert request['system'].startswith(old)
    return {**request, 'system': new+request['system'][len(old):]}


def dev_gate(scored):
    return scored['n'] == 32 and scored['correct'] >= 21 and scored['macro_f1'] >= .6 and scored['valid_rate'] >= .95


def prompt_diagnostic(ids, gold, outputs):
    errors = [{'id':k, 'gold':gold[k], 'prediction':parse_label(text)}
              for k,text in zip(ids,outputs) if parse_label(text) != gold[k]]
    pairs = Counter((e['gold'],e['prediction']) for e in errors if e['prediction'] is not None)
    repeated = [{'gold':g, 'prediction':p, 'count':n} for (g,p),n in sorted(pairs.items()) if n >= 2]
    invalid = sum(e['prediction'] is None for e in errors)
    return {'errors':errors, 'repeated_gold_prediction_confusions':repeated, 'invalid_count':invalid,
            'trigger':bool(repeated or invalid >= 2),
            'reason':'Repeated confusion or invalidity warrants the predeclared task-definition clarification; this is a diagnostic hypothesis, not proven prompt causation.'}


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def baseline(run=RUN):
    """Fresh fit only, single fixed sparse logistic model, no model selection."""
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    from pyproj import Transformer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    import joblib
    from threadpoolctl import threadpool_limits
    run = Path(run)
    folder = run/'fresh-baseline'
    if folder.exists():
        raise RuntimeError('Fresh baseline namespace exists; no silent refit')
    folder.mkdir()
    roles = load(run/'protocol/repair-role-freeze.json')['roles']
    fit = roles['CityFM-SG-GeoRepairFit']['ids']
    screen = roles['CityFM-SG-GeoRepairScreen96']['ids']
    pool = roles['CityFM-SG-GeoRepairScreenPool']['ids']
    assert len(fit) == 26158 and len(screen) == 96
    assert not set(fit) & set(pool) and set(screen) <= set(pool)
    assert all(digest(v['ids']) == v['sha256'] for v in roles.values())
    raw = ROOT/'data/raw/cityfm-singapore'/REV
    targets = load(raw/'buildings_untagged.json')
    contexts = load(raw/'buildings_context.json')
    entities = load(raw/'entities.json')
    labels = load(raw/'land_use.json')
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3414', always_xy=True)
    def features():
        for i, k in enumerate(fit):
            payload, _ = build_payload(k, targets[k], contexts[k], entities, transformer)
            if i % 1000 == 0:
                atomic(folder/'progress.json', {'phase': 'features', 'completed': i, 'total': len(fit)})
            yield repaired_baseline_features(payload)
    model = make_pipeline(DictVectorizer(sparse=True), StandardScaler(with_mean=False),
                          LogisticRegression(C=1, max_iter=500, random_state=20261022))
    with threadpool_limits(limits=4):
        model.fit(features(), [labels[k] for k in fit])
        payloads = {r['id']: r['payload'] for r in lines(run/'repair-screen96/inputs.jsonl')}
        predictions = model.predict([repaired_baseline_features(payloads[k]) for k in screen])
    majority = sorted(Counter(labels[k] for k in fit).items(), key=lambda x: (-x[1], x[0]))[0][0]
    frozen_gold = {r['id']: r['class'] for r in load(run/'protocol/repair-screen96-gold.local.json')}
    assert all(frozen_gold[k] == labels[k] for k in screen)
    result = {'status': 'FRESH_FIXED_BASELINE_COMPLETE', 'fit_n': len(fit), 'fit_ids_sha256': digest(fit),
              'screen_ids_sha256': digest(screen), 'whole_pool_excluded_n': len(pool),
              'fit_screen_intersection': 0, 'fit_whole_pool_intersection': 0,
              'settings': {'C': 1, 'max_iter': 500, 'seed': 20261022, 'threads_max': 4,
                           'sparse': True, 'scaler': 'StandardScaler(with_mean=False)', 'hypersearch': False,
                           'features': 'fixed repaired_baseline_features, including neighbor names; fit-only vocabulary/scaler'},
              'majority_label': majority, 'majority': score([frozen_gold[k] for k in screen],
                    [json.dumps({'class': majority}) for _ in screen]),
              'logistic': score([frozen_gold[k] for k in screen], [json.dumps({'class': x}) for x in predictions]),
              'screen_historical_old_logistic_fit_exposure': True,
              'new_model_screen_exposure': 'predict only, no tuning',
              'n_features': len(model[0].vocabulary_), 'iterations': model[-1].n_iter_.tolist(),
              'role_freeze_sha256': sha(run/'protocol/repair-role-freeze.json')}
    joblib.dump(model, folder/'model.local.joblib')
    atomic(folder/'result.json', result)
    atomic(folder/'predictions.local.json', [{'id': k, 'class': str(p)} for k, p in zip(screen, predictions)])
    return result


def packages(run=RUN):
    """Freeze actual requests and exact local code/data/budget without approval."""
    run = Path(run)
    folder = run/'workpackages-v1'
    if folder.exists():
        raise RuntimeError('Immutable packages exist; do not replace their approval identity')
    provider = ROOT/'reports/night-dual-2026-10-22/provider-verification.json'
    assert load(provider)['rates_usd_per_million_peak'] == RATES
    assert load(run/'fresh-baseline/result.json')['status'] == 'FRESH_FIXED_BASELINE_COMPLETE'
    old_ledger = HISTORICAL/'budget/new-round.sqlite'
    with sqlite3.connect('file:'+str(old_ledger)+'?mode=ro', uri=True) as conn:
        pico = conn.execute('SELECT COALESCE(SUM(pico),0) FROM calls').fetchone()[0]
        assert Decimal(pico)/SqliteBudget.SCALE == OLD_USD
        assert conn.execute('SELECT COUNT(*) FROM calls WHERE settled=0').fetchone()[0] == 0
    bindings = {}
    paths = [Path(__file__), ROOT/'scripts/prepare_cityfm_repair_20261022.py',
             ROOT/'src/privacy_synth/cityfm_representation.py', ROOT/'src/privacy_synth/location_classification.py',
             ROOT/'src/privacy_synth/pilot_ledger.py', ROOT/'src/privacy_synth/round_budget.py',
             ROOT/'scripts/night_dual_20261022.py', ROOT/'scripts/deepseek_ipd_interface.py',
             ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json',
             run/'protocol/repair-role-freeze.json', run/'protocol/fallback-overlap-audit.json',
             run/'fresh-baseline/result.json', provider, old_ledger, ROOT/'pyproject.toml', ROOT/'uv.lock']
    paths += [run/'fresh-baseline/model.local.joblib', run/'protocol/geometry-identity-audit.json']
    raw = ROOT/'data/raw/cityfm-singapore'/REV
    paths += [raw/name for name in ('buildings_untagged.json','buildings_context.json','entities.json','land_use.json','land_use_data.csv')]
    paths += [run/'dev32/paired-inputs.local.jsonl', run/'protocol/repair-screen96-gold.local.json']
    for path in paths:
        bindings[str(path)] = sha(path)
    folder.mkdir()
    result = {}
    for phase, source in [('dev32', run/'dev32/repaired-requests.proposed.jsonl'),
                          ('screen96', run/'repair-screen96/requests.proposed.jsonl')]:
        rows = lines(source)
        spec = SPECS[phase]
        assert len(rows) == (32 if phase == 'dev32' else 96) and all(r['input_tokens'] <= 8192 for r in rows)
        from tokenizers import Tokenizer
        from prepare_cityfm_repair_20261022 import token_count
        tok = Tokenizer.from_file(str(ROOT/'artifacts/qwen-deepseek-tokenizer-transfer-2026-10-18/tokenizer.json'))
        candidates = [{**row, 'variant':v, 'request':variant_request(row['request'],v)} for v in (1,2) for row in rows]
        for row in candidates:
            row['request_sha256'] = digest(row['request'])
            row['input_tokens'] = token_count(tok,row['request'])
            assert row['input_tokens'] <= 8192
        candidate_path = folder/(phase+'-requests.jsonl')
        candidate_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in candidates))
        packet = {'status': 'FROZEN_AWAITING_NEW_INPUT_PACKAGE_CONFIRMATION', 'approval': None,
                  'phase': phase, 'provider': 'DeepSeek official', 'model': 'deepseek-flash',
                  'thinking': 'disabled', 'temperature': 0, 'input_max': 8192, 'output_max': 64,
                  'logical_calls_max': spec['logical'], 'http_max': spec['calls'], 'technical_retries_max': spec['retry'],
                  'usd_cap': 10, 'parent_usd_cap': 500, 'historical_accounted_usd': str(OLD_USD),
                  'new_shared_available_usd': str(PARENT_USD-OLD_USD), 'rates': RATES,
                  'input_tokens_by_variant': {str(v):sum(r['input_tokens'] for r in candidates if r['variant']==v) for v in (1,2)},
                  'output_max_no_retries': spec['logical']*64,
                  'reservation_input_tokens': spec['calls']*8192,
                  'reservation_output_tokens': spec['calls']*64,
                  'reservation_usd': float(Decimal(spec['calls'])*(Decimal(8192)*Decimal('.3')+Decimal(64)*Decimal('1.2'))/1000000),
                  'requests_path': str(candidate_path), 'allowed_request_sha256': {f"v{r['variant']}:{r['id']}": r['request_sha256'] for r in candidates},
                  'bindings': {**bindings, str(source): sha(source), str(candidate_path):sha(candidate_path)},
                  'dev_gate': {'correct_min': 21, 'n': 32, 'macro_f1_min': .6, 'valid_rate_min': .95},
                  'screen_gate': {'accuracy_min': .65, 'correct_min': 63, 'macro_f1_min': .6,
                                  'majority_margin': .10, 'fresh_logistic_margin': .05, 'valid_rate_min': .95},
                  'adaptation': 'v1 repaired input + unchanged prompt0. v2 identical input, task-definition clarification only, triggered exclusively by v1 Dev failure plus >=2 same gold-to-prediction confusions or >=2 invalid outputs. Save exact errors before v2. Select Dev macroF1 then accuracy then v1; if neither passes stop. Screen exactly once using selected prompt. No fewshot/new representation/correctness retry.',
                  'prompt_systems': {str(v):variant_request(rows[0]['request'],v)['system'] for v in (1,2)},
                  'data': 'Independent public CityFM OSM neighbors: names, raw semantic values including healthcare, campus/site spatial relations and every relative position; target physical levels/height and complete relative footprint. Target names/use/address/wiki/IDs/absolute coordinates and gold are not transmitted.',
                  'public_proxy_only': True, 'historical_screen_local_fit_exposure': True,
                  'cheaper_alternative': 'Fresh fixed sparse logistic and majority have actually run; local schema/scoring need no API. They cannot establish teacher quality.'}
        path = folder/(phase+'.json')
        atomic(path, packet)
        result[phase] = {'path': str(path), 'sha256': sha(path), 'reservation_usd': packet['reservation_usd']}
    atomic(folder/'index.json', result)
    return result


def require_approval(run, phase):
    """This must execute before any env/key/proxy/network/ledger creation."""
    run = Path(run)
    packet_path = run/'workpackages-v1'/(phase+'.json')
    approval_path = run/'approvals'/(phase+'.json')
    if not packet_path.is_file() or not approval_path.is_file():
        raise PermissionError('New restored-input package approval is required before provider, key or network access')
    packet, approval = load(packet_path), load(approval_path)
    required = {'authorized': True, 'package_sha256': sha(packet_path), 'provider': 'DeepSeek official',
                'model': 'deepseek-flash', 'thinking': 'disabled', 'usd_cap': 10, 'parent_usd_cap': 500}
    if any(approval.get(k) != v for k, v in required.items()):
        raise PermissionError('Exact restored-input model/budget/package approval mismatch')
    if not isinstance(approval.get('approved_at_epoch'), (int,float)) or not approval['approved_at_epoch'] <= time.time() < approval['approved_at_epoch']+12*3600:
        raise PermissionError('Approval expired, missing or future')
    for path, expected in packet['bindings'].items():
        if not Path(path).is_file() or sha(Path(path)) != expected:
            raise PermissionError('Frozen restored-input binding changed: '+path)
    if phase == 'screen96':
        dev_path = run/'results/dev32.json'
        if not dev_path.is_file() or load(dev_path).get('dev_gate_pass') is not True:
            raise PermissionError('Fresh screen is blocked until the fixed Dev32 gate passes')
        dev = load(dev_path)
        if dev.get('package_sha256') != sha(run/'workpackages-v1/dev32.json'):
            raise PermissionError('Dev result is not bound to the current package')
    return packet


class RepairClient:
    def __init__(self, run, phase):
        self.run, self.phase = Path(run), phase
        self.packet = require_approval(self.run, phase)
        pools = {name: {'calls': spec['calls'], 'input_tokens': spec['calls']*8192,
                        'output_tokens': spec['calls']*64, 'usd': 10} for name, spec in SPECS.items()}
        self.ledger = SqliteBudget(self.run/'budget/repair.sqlite', str(PARENT_USD-OLD_USD), pools, RATES)
        import deepseek_ipd_interface as existing
        self.key = os.environ.get('DEEPSEEK_API_KEY') or existing.load_key()
        existing.apply_proxy()

    def request(self, row):
        import urllib.request
        import urllib.error
        require_approval(self.run, self.phase)
        request, item = row['request'], f"v{row['variant']}:{row['id']}"
        if self.packet['allowed_request_sha256'].get(item) != digest(request) or request['max_tokens'] != 64:
            raise PermissionError('Request is not the approved exact input')
        if self.phase == 'screen96' and row['variant'] != load(self.run/'results/dev32.json')['selected_variant']:
            raise PermissionError('Screen may use only the Dev-selected fixed prompt')
        if self.phase == 'dev32' and row['variant'] == 2:
            prior = self.run/'results/dev32-v1.json'
            if not prior.is_file():
                raise PermissionError('Prompt2 requires completed v1 diagnostic evidence')
            v1 = load(prior)
            gold = {r['id']:r['gold_local'] for r in lines(self.run/'dev32/paired-inputs.local.jsonl')}
            rescored = score([gold[k] for k in v1['ids']],v1['outputs'])
            if dev_gate(rescored) or not prompt_diagnostic(v1['ids'],gold,v1['outputs'])['trigger']:
                raise PermissionError('Prompt2 diagnostic trigger is absent')
        folder = self.run/'requests'/self.phase
        folder.mkdir(parents=True, exist_ok=True)
        final = folder/(item+'.json')
        if final.exists():
            cached = load(final)
            if cached['request_sha256'] != digest(request):
                raise RuntimeError('Cached request hash mismatch')
            return cached['text']
        for attempt in range(2):
            tag = f'{self.phase}:{item}:attempt{attempt}'
            state = folder/(item+f'.attempt{attempt}.json')
            if state.exists():
                saved = load(state)
                if saved.get('status') == 'TECHNICAL_RETRYABLE' and attempt == 0:
                    continue
                raise RuntimeError('Existing uncertain/completed attempt blocks duplicate send: '+tag)
            if attempt:
                used_retries = sum(1 for p in folder.glob('*.attempt1.json'))
                if used_retries >= self.packet['technical_retries_max']:
                    raise RuntimeError('Package technical retry allowance exhausted')
            call = self.ledger.reserve(self.phase, tag, 8192, 64)
            atomic(state, {'status': 'RESERVED_BEFORE_SEND', 'call_id': call, 'request_sha256': digest(request)})
            body = {'model': 'deepseek-flash', 'messages': [{'role':'system','content':request['system']},
                       {'role':'user','content':request['user']}], 'thinking': {'type':'disabled'},
                    'temperature': 0, 'max_tokens': 64, 'stream': False, 'response_format': {'type':'json_object'}}
            settled = False
            try:
                req = urllib.request.Request('https://api.deepseek.com/chat/completions', data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+self.key, 'Content-Type':'application/json'})
                with urllib.request.urlopen(req, timeout=120) as response:
                    data = json.load(response)
                # Save response before accounting: an interruption cannot silently resend it.
                atomic(state, {'status':'RECEIVED_UNSETTLED', 'call_id':call, 'response':data,
                               'request_sha256':digest(request)})
                self.ledger.settle(call, data.get('usage')); settled = True
                msg = data['choices'][0]['message']
                if str(data.get('model','')).lower() not in ('deepseek-flash','deepseek-v4.1-flash-1017') or msg.get('reasoning_content'):
                    raise RuntimeError('Provider model/thinking mismatch')
                identity = {'model':data['model'], 'system_fingerprint':data.get('system_fingerprint')}
                if not identity['system_fingerprint']:
                    raise RuntimeError('Provider revision fingerprint absent')
                identity_path = self.run/'provider-identity.json'
                if identity_path.exists() and load(identity_path) != identity:
                    raise RuntimeError('Provider revision drift')
                if not identity_path.exists():
                    atomic(identity_path, identity)
                text = msg.get('content','')
                atomic(final, {'id':item, 'text':text, 'response':data, 'request_sha256':digest(request)})
                atomic(state, {'status':'RECEIVED_SETTLED', 'call_id':call, 'request_sha256':digest(request)})
                return text
            except urllib.error.HTTPError as error:
                self.ledger.settle(call, None); settled = True
                retryable = error.code in (429,500,502,503,504) and attempt == 0
                atomic(state, {'status':'TECHNICAL_RETRYABLE' if retryable else 'HTTP_ERROR_UNKNOWN_USAGE',
                               'call_id':call, 'http_status':error.code, 'request_sha256':digest(request)})
                if retryable:
                    continue
                raise RuntimeError('Provider HTTP error '+str(error.code)) from None
            except Exception:
                if not settled:
                    with sqlite3.connect(self.ledger.path) as conn:
                        already = conn.execute('SELECT settled FROM calls WHERE id=?',(call,)).fetchone()[0]
                    if not already:
                        self.ledger.settle(call, None)
                raise
        raise RuntimeError('Technical retry exhausted')


def execute(run=RUN, phase='dev32'):
    run = Path(run)
    packet = require_approval(run, phase)
    with lock(run/'locks/repair-api.lock'):
        result_path = run/'results'/(phase+'.json')
        if result_path.exists():
            result = load(result_path)
            if result['package_sha256'] != sha(run/'workpackages-v1'/(phase+'.json')):
                raise RuntimeError('Result package mismatch')
            return result
        client = RepairClient(run, phase)
        all_rows = lines(packet['requests_path'])
        if phase == 'dev32':
            gold = {r['id']:r['gold_local'] for r in lines(run/'dev32/paired-inputs.local.jsonl')}
        else:
            gold = {r['id']:r['class'] for r in load(run/'protocol/repair-screen96-gold.local.json')}
        rounds = []
        variants = (1,2) if phase == 'dev32' else (load(run/'results/dev32.json')['selected_variant'],)
        for variant in variants:
            if phase == 'dev32' and variant == 2:
                evidence = prompt_diagnostic(rounds[0]['ids'],gold,rounds[0]['outputs'])
                atomic(run/'results/v1-prompt-diagnostic.local.json',evidence)
                if dev_gate(rounds[0]['scores']) or not evidence['trigger']:
                    break
            rows = [r for r in all_rows if r['variant']==variant]
            outputs = []
            for row in rows:
                outputs.append(client.request(row))
                atomic(run/'results/progress.json', {'phase':phase, 'variant':variant, 'complete':len(outputs), 'total':len(rows)})
            scored = score([gold[r['id']] for r in rows],outputs)
            current = {'variant':variant,'ids':[r['id'] for r in rows],'outputs':outputs,'scores':scored}
            rounds.append(current)
            atomic(run/'results'/(phase+f'-v{variant}.json'),current)
        selected = max(rounds,key=lambda r:(r['scores']['macro_f1'],r['scores']['accuracy'],-r['variant']))
        scored = selected['scores']
        result = {'phase':phase, 'ids':selected['ids'], 'outputs':selected['outputs'], 'scores':scored,
                  'selected_variant':selected['variant'],'rounds':rounds,
                  'package_sha256':sha(run/'workpackages-v1'/(phase+'.json')), 'ledger':client.ledger.summary()}
        if phase == 'dev32':
            result['dev_gate_pass'] = dev_gate(scored)
            result['status'] = 'DEV_PASS_SCREEN_REQUIRES_OWN_APPROVAL' if result['dev_gate_pass'] else 'STOP_DEV_FAIL_NO_AUTOMATIC_ADAPTATION'
        else:
            base = load(run/'fresh-baseline/result.json')
            audit = load(run/'protocol/fallback-overlap-audit.json')
            no_overlap = audit['post_exclusion_shared_entities'] == audit['fit_screen_grid_overlap_n'] == 0 and not audit['crosscell_site_violations'] and not audit['crossrole_target_overlap_violations']
            result['qualification_pass'] = qualification_gate(scored, base['majority']['accuracy'], base['logistic']['accuracy'], no_overlap)
            result['status'] = 'QUALIFIED_PUBLIC_PROXY_ONLY' if result['qualification_pass'] else 'STOP_SCREEN_FAIL'
        atomic(result_path, result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['baseline','prepare','check','run'])
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--phase', choices=list(SPECS), default='dev32')
    args = parser.parse_args()
    if args.action == 'baseline': result = baseline(args.run)
    elif args.action == 'prepare': result = packages(args.run)
    elif args.action == 'check':
        require_approval(args.run,args.phase); result = {'approved':True}
    else: result = execute(args.run,args.phase)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
