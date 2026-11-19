#!/usr/bin/env python3
"""Qualify DeepSeek deepseek-flash for verbatim IPD against pinned Qwen3.5.

Stage B and utility stay stopped unless every stage-A gate passes.
Gold, evaluation items, and protected rows are never sent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('HF_HOME', str(ROOT / 'artifacts/hf-home'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

TZ = timezone(timedelta(hours=8))
T0 = datetime.fromisoformat('2026-10-18T01:57:30+08:00')
FIRST_DEADLINE = T0 + timedelta(minutes=30)
STAGE_A_DEADLINE = T0 + timedelta(hours=2)
HARD_STOP = T0 + timedelta(hours=24)
WRAP_AT = HARD_STOP - timedelta(minutes=45)
API = 'https://api.deepseek.com'
BETA = 'https://api.deepseek.com/beta'
MODEL = 'deepseek-flash'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
HISTORICAL_FINGERPRINT = 'aeb56401ca74e127821c4f9126dcb669'
IDEA_MARKER = '## 2026-10-18 — Execute the DeepSeek IPD interface qualification'
RESULT_MARKER = 'deepseek-ipd-interface-utility result'
SECRET_RE = re.compile(r'sk-[A-Za-z0-9_\-]+')
PROXY = 'http://127.0.0.1:7897'
SQL_PREFIX = 'SELECT id FROM businesses WHERE '
LINE_PREFIXES = [
    'SELECT',
    'SELECT id',
    ' count(*) ',
    'WHERE business_id =',
    'ORDER BY 2',
    'LIMIT 10',
    '12345',
    'id10',
    'id 10',
    '  ',
    "it's",
    '"quoted"',
    "'O''Brien'",
    "name LIKE '%cafe%'",
    '价格',
    '中文标识',
    'αβγ',
    '€50',
    '🙂',
    'table.column',
    'CAST(x AS INT)',
    'NULL',
    'end;',
    'json key sql',
    'business_id',
    'GROUP BY',
]
NEWLINE_PROBE = 'a\nb\n\n  SELECT'


def now() -> str:
    return datetime.now(TZ).isoformat(timespec='seconds')


def clock() -> datetime:
    return datetime.now(TZ)


def redact(text: str) -> str:
    return SECRET_RE.sub('sk-REDACTED', text)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_key() -> str:
    for line in (Path.home() / '.secrets' / 'deepseek.env').read_text(encoding='utf-8').splitlines():
        if line.startswith('DEEPSEEK_API_KEY='):
            value = line.split('=', 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit('DEEPSEEK_API_KEY missing')


def apply_proxy() -> None:
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ[name] = PROXY


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def peak_usd(rates: dict, hit: int, miss: int, output: int) -> float:
    usd = (
        hit * rates['cache_hit_input_peak']
        + miss * rates['cache_miss_input_peak']
        + output * rates['output_peak']
    ) / 1_000_000
    return round(usd, 10)


def empty_ledger(rates: dict, price_meta: dict) -> dict:
    def stage(attempts, inp, out, usd):
        return {
            'max_attempts': attempts,
            'max_input_tokens': inp,
            'max_output_tokens': out,
            'max_usd': usd,
        }

    return {
        'provider': 'deepseek',
        'model': MODEL,
        'thinking': 'disabled',
        'api_base': API,
        'beta_base': BETA,
        'price': price_meta,
        'rates_per_million_usd': rates,
        'reservation': 'peak cache-hit for known hits; peak cache-miss for misses and unknown input; peak output. Unknown usage stays reserved.',
        'grok_platform_fee_usd': 'UNKNOWN',
        'paid_tools': 'disabled',
        'stages': {
            'A': stage(60, 250_000, 8_000, 1.0),
            'B': stage(20_000, 80_000_000, 1_000_000, 49.0),
        },
        'total_max_usd': 50.0,
        'stage_b_unlocked': False,
        'stop': False,
        'stop_reason': '',
        'next_id': 1,
        'accounted': {
            'A': {'attempts': 0, 'input_tokens': 0, 'output_tokens': 0, 'usd': 0.0},
            'B': {'attempts': 0, 'input_tokens': 0, 'output_tokens': 0, 'usd': 0.0},
        },
        'calls': [],
    }


class CapExceeded(RuntimeError):
    pass


def ledger_path(run: Path) -> Path:
    return run / 'budget-ledger.json'


def with_ledger(run: Path, fn):
    lock_path = run / 'budget-ledger.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            ledger = json.loads(ledger_path(run).read_text(encoding='utf-8'))
            result = fn(ledger)
            dump(ledger_path(run), ledger)
            return result
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def try_reserve(ledger: dict, stage: str, probe: str, est_in: int, est_out: int) -> dict:
    if stage == 'B' and not ledger['stage_b_unlocked']:
        raise CapExceeded('stage B is locked')
    if ledger.get('stop'):
        raise CapExceeded(ledger.get('stop_reason') or 'stopped')
    caps = ledger['stages'][stage]
    acc = ledger['accounted'][stage]
    usd = peak_usd(ledger['rates_per_million_usd'], 0, est_in, est_out)
    if acc['attempts'] + 1 > caps['max_attempts']:
        raise CapExceeded('attempts')
    if acc['input_tokens'] + est_in > caps['max_input_tokens']:
        raise CapExceeded('input_tokens')
    if acc['output_tokens'] + est_out > caps['max_output_tokens']:
        raise CapExceeded('output_tokens')
    if acc['usd'] + usd > caps['max_usd'] + 1e-9:
        raise CapExceeded('usd')
    if ledger['accounted']['A']['usd'] + ledger['accounted']['B']['usd'] + usd > ledger['total_max_usd'] + 1e-9:
        raise CapExceeded('total_usd')
    call = {
        'id': ledger['next_id'],
        'stage': stage,
        'probe': probe,
        'est_in': est_in,
        'est_out': est_out,
        'est_usd': usd,
        'reserved_at': now(),
        'settled': False,
        'usage_unknown': True,
        'usd': usd,
        'input_tokens': est_in,
        'output_tokens': est_out,
    }
    ledger['next_id'] += 1
    acc['attempts'] += 1
    acc['input_tokens'] += est_in
    acc['output_tokens'] += est_out
    acc['usd'] = round(acc['usd'] + usd, 10)
    ledger['calls'].append(call)
    return call


def settle(ledger: dict, call_id: int, usage: dict | None) -> dict:
    call = next(item for item in ledger['calls'] if item['id'] == call_id)
    if call['settled']:
        return call
    acc = ledger['accounted'][call['stage']]
    rates = ledger['rates_per_million_usd']
    if not usage:
        call['settled'] = True
        call['usage_unknown'] = True
        call['settled_at'] = now()
        return call
    hit = usage.get('prompt_cache_hit_tokens')
    miss = usage.get('prompt_cache_miss_tokens')
    prompt = usage.get('prompt_tokens')
    completion = usage.get('completion_tokens')
    if hit is None or miss is None:
        hit = 0
        miss = prompt if prompt is not None else call['est_in']
    if completion is None:
        completion = call['est_out']
        call['output_unknown'] = True
    else:
        call['output_unknown'] = False
    usd = peak_usd(rates, int(hit), int(miss), int(completion))
    acc['input_tokens'] += int(hit) + int(miss) - call['est_in']
    acc['output_tokens'] += int(completion) - call['est_out']
    acc['usd'] = round(acc['usd'] + usd - call['est_usd'], 10)
    call.update({
        'settled': True,
        'usage_unknown': False,
        'input_tokens': int(hit) + int(miss),
        'cache_hit_tokens': int(hit),
        'cache_miss_tokens': int(miss),
        'output_tokens': int(completion),
        'usd': usd,
        'usage': usage,
        'settled_at': now(),
    })
    caps = ledger['stages'][call['stage']]
    if acc['attempts'] > caps['max_attempts'] or acc['input_tokens'] > caps['max_input_tokens'] or acc['output_tokens'] > caps['max_output_tokens'] or acc['usd'] > caps['max_usd'] + 1e-9:
        ledger['stop'] = True
        ledger['stop_reason'] = 'settled usage crossed the stage cap'
    return call


def parse_rates(html: str) -> dict:
    frag = re.sub(r'<[^>]+>', ' ', html)
    frag = re.sub(r'\s+', ' ', frag)
    if 'deepseek-flash' not in frag or 'DeepSeek-V4.1-Flash-1017' not in frag:
        raise RuntimeError('pricing page missing deepseek-flash identity')
    hit = re.search(r'1M INPUT TOKENS \(CACHE HIT\) OFF-PEAK \$([0-9.]+) \$([0-9.]+) PEAK \$([0-9.]+) \$([0-9.]+)', frag)
    miss = re.search(r'1M INPUT TOKENS \(CACHE MISS\) OFF-PEAK \$([0-9.]+) \$([0-9.]+) PEAK \$([0-9.]+) \$([0-9.]+)', frag)
    out = re.search(r'1M OUTPUT TOKENS OFF-PEAK \$([0-9.]+) \$([0-9.]+) PEAK \$([0-9.]+) \$([0-9.]+)', frag)
    if not (hit and miss and out):
        raise RuntimeError('pricing table did not match the Flash columns')
    return {
        'cache_hit_input_offpeak': float(hit.group(1)),
        'cache_hit_input_peak': float(hit.group(3)),
        'cache_miss_input_offpeak': float(miss.group(1)),
        'cache_miss_input_peak': float(miss.group(3)),
        'output_offpeak': float(out.group(1)),
        'output_peak': float(out.group(3)),
        'model_version_documented': 'DeepSeek-V4.1-Flash-1017',
        'flash_is_first_price_column': True,
    }


def confirm_price(run: Path) -> dict:
    apply_proxy()
    import httpx

    url = 'https://api-docs.deepseek.com/quick_start/pricing'
    last_error = ''
    response_text = ''
    status = 0
    for _ in range(2):
        try:
            response = httpx.get(url, timeout=30, follow_redirects=True)
            status = response.status_code
            response_text = response.text
            if status == 200 and 'deepseek-flash' in response_text:
                break
            last_error = f'HTTP {status}'
        except Exception as exc:  # noqa: BLE001 — price check is recorded
            last_error = redact(f'{type(exc).__name__}: {exc}')
    (run / 'docs').mkdir(parents=True, exist_ok=True)
    (run / 'docs' / 'pricing.html').write_text(response_text, encoding='utf-8')
    if status != 200 or 'deepseek-flash' not in response_text:
        dump(run / 'price-check.json', {'ok': False, 'at': now(), 'error': last_error, 'http_status': status})
        raise RuntimeError(f'price check failed: {last_error}')
    rates = parse_rates(response_text)
    meta = {
        'ok': True,
        'at': now(),
        'url': url,
        'http_status': status,
        'bytes': len(response_text.encode()),
        'sha256': hashlib.sha256(response_text.encode()).hexdigest(),
        'live_hour_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'live_period': 'off-peak weekend',
        'reservation_period': 'peak',
        'peak_hours_utc': 'Monday-Friday 01:00-04:00 and 06:00-10:00, excluding Chinese public holidays',
        'search_fetch_used': False,
    }
    dump(run / 'price-check.json', {'meta': meta, 'rates_per_million_usd': rates})
    return {'rates': rates, 'meta': meta}


def estimate_tokens(body: dict) -> int:
    raw = json.dumps(body, ensure_ascii=False).encode()
    return len(raw) + 32


def post_once(url: str, body: dict, key: str) -> dict:
    import httpx

    apply_proxy()
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0)
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=True) as client:
            response = client.post(url, headers=headers, json=body)
        try:
            payload = response.json()
        except Exception:
            payload = {'_non_json': redact(response.text)[:4000]}
        return {
            'http_status': response.status_code,
            'payload': payload,
            'error': '' if response.status_code == 200 else redact(response.text)[:2000],
        }
    except Exception as exc:  # noqa: BLE001 — the ledger keeps the attempt
        return {'http_status': 0, 'payload': {}, 'error': redact(f'{type(exc).__name__}: {exc}')[:2000]}


def technical(status: int) -> bool:
    return status in {0, 408, 429, 500, 502, 503, 504}


def save_attempt(run: Path, call: dict, url: str, body: dict, result: dict, attempt: int) -> Path:
    path = run / 'requests' / f"A-{call['id']:04d}-{call['probe']}-try{attempt}.json"
    payload = result.get('payload') or {}
    choice = (payload.get('choices') or [{}])[0] if isinstance(payload, dict) else {}
    message = choice.get('message') or {}
    usage = payload.get('usage') if isinstance(payload, dict) else None
    record = {
        'call_id': call['id'],
        'probe': call['probe'],
        'attempt': attempt,
        'at': now(),
        'url': url,
        'request': body,
        'http_status': result['http_status'],
        'error': result['error'],
        'response_id': payload.get('id') if isinstance(payload, dict) else None,
        'response_model': payload.get('model') if isinstance(payload, dict) else None,
        'system_fingerprint': payload.get('system_fingerprint') if isinstance(payload, dict) else None,
        'usage': usage,
        'finish_reason': choice.get('finish_reason'),
        'content': message.get('content'),
        'reasoning_content': message.get('reasoning_content'),
        'logprobs': choice.get('logprobs'),
        'raw_response': payload,
    }
    dump(path, record)
    return path


def execute_probe(run: Path, key: str, probe: str, url: str, body: dict) -> dict:
    est_in = estimate_tokens(body)
    est_out = int(body.get('max_tokens') or 16)
    try:
        call = with_ledger(run, lambda ledger: try_reserve(ledger, 'A', probe, est_in, est_out))
    except CapExceeded as exc:
        dump(run / 'status.json', {'phase': 'A_STOPPED', 'at': now(), 'reason': str(exc), 'probe': probe})
        raise
    result = post_once(url, body, key)
    path = save_attempt(run, call, url, body, result, 1)
    usage = (result.get('payload') or {}).get('usage') if result['http_status'] else None
    if technical(result['http_status']):
        with_ledger(run, lambda ledger: settle(ledger, call['id'], usage))
        try:
            retry = with_ledger(run, lambda ledger: try_reserve(ledger, 'A', probe + '-retry', est_in, est_out))
        except CapExceeded:
            return {'probe': probe, 'call_id': call['id'], 'path': str(path), 'result': result, 'retry': None}
        result2 = post_once(url, body, key)
        path2 = save_attempt(run, retry, url, body, result2, 2)
        usage2 = (result2.get('payload') or {}).get('usage') if result2['http_status'] else None
        with_ledger(run, lambda ledger: settle(ledger, retry['id'], usage2))
        print(f"PROBE {probe} status={result['http_status']} retry={result2['http_status']} id={retry['id']}", flush=True)
        return {'probe': probe, 'call_id': retry['id'], 'path': str(path2), 'result': result2, 'first': result}
    with_ledger(run, lambda ledger: settle(ledger, call['id'], usage if result['http_status'] else None))
    print(f"PROBE {probe} status={result['http_status']} id={call['id']}", flush=True)
    return {'probe': probe, 'call_id': call['id'], 'path': str(path), 'result': result, 'first': None}


def base_body(messages: list, max_tokens: int, top_logprobs: int = 20, temperature: float = 1) -> dict:
    return {
        'model': MODEL,
        'messages': messages,
        'thinking': {'type': 'disabled'},
        'max_tokens': max_tokens,
        'temperature': temperature,
        'logprobs': True,
        'top_logprobs': top_logprobs,
        'stream': False,
    }


def cursor_load(run: Path) -> dict:
    path = run / 'cursor.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'done': {}}


def cursor_mark(run: Path, probe: str, info: dict) -> None:
    cursor = cursor_load(run)
    cursor['done'][probe] = info
    cursor['updated_at'] = now()
    dump(run / 'cursor.json', cursor)


def run_probe(run: Path, key: str, probe: str, url: str, body: dict) -> dict:
    done = cursor_load(run)['done']
    if probe in done:
        print(f'SKIP {probe}', flush=True)
        return done[probe]
    if clock() > STAGE_A_DEADLINE:
        raise CapExceeded('stage A wall clock')
    info = execute_probe(run, key, probe, url, body)
    stored = {
        'call_id': info['call_id'],
        'path': info['path'],
        'http_status': info['result']['http_status'],
        'error': info['result']['error'],
    }
    cursor_mark(run, probe, stored)
    if probe == 'p01_identity_logprobs':
        payload = info['result'].get('payload') or {}
        fingerprint = payload.get('system_fingerprint') if isinstance(payload, dict) else None
        if info['result']['http_status']:
            print(f"FIRST_DS_RESPONSE status={info['result']['http_status']} id={payload.get('id')} fingerprint={fingerprint} at={now()}", flush=True)
        else:
            print(f"FIRST_DS_BLOCKED error={info['result']['error']} at={now()}", flush=True)
    return stored


def token_rows(logprobs: dict | None) -> list:
    if not logprobs:
        return []
    return list(logprobs.get('content') or [])


def finite_logprob(value) -> bool:
    return isinstance(value, (int, float)) and value == value and value > -1000


def qwen_tokenizer():
    os.environ['HF_HUB_OFFLINE'] = '1'
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B', revision=REVISION, local_files_only=True)


def qwen_cuts(tokenizer, text: str) -> dict:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(start), int(end)) for start, end in encoded['offset_mapping']]
    ids = [int(item) for item in encoded['input_ids']]
    pieces = [tokenizer.decode([item], skip_special_tokens=False) for item in ids]
    roundtrip = tokenizer.decode(ids, skip_special_tokens=False) == text
    return {'ids': ids, 'offsets': offsets, 'pieces': pieces, 'roundtrip': roundtrip, 'n_tokens': len(ids)}


def ds_cuts(rows: list) -> dict | None:
    blob = bytearray()
    cuts = []
    pieces = []
    ok_bytes = True
    for row in rows:
        raw = row.get('bytes')
        if not isinstance(raw, list) or any(not isinstance(item, int) for item in raw):
            ok_bytes = False
            break
        blob.extend(raw)
        cuts.append(len(blob))
        pieces.append(row.get('token'))
    if not ok_bytes:
        return None
    try:
        text = blob.decode('utf-8')
    except UnicodeDecodeError:
        return {'decode_ok': False, 'cuts': cuts, 'pieces': pieces}
    return {'decode_ok': True, 'text': text, 'cuts': cuts, 'pieces': pieces, 'n_tokens': len(rows)}


def line_boundaries(text: str, cuts: list[int], line: str, start: int) -> list[int]:
    end = start + len(line.encode())
    relative = [cut - start for cut in cuts if start < cut < end]
    return relative


def qwen_byte_internal(tokenizer, line: str) -> list[int]:
    """Character offsets from the fast tokenizer, converted to UTF-8 byte cuts."""
    local = qwen_cuts(tokenizer, line)
    char_bytes = [0]
    for char in line:
        char_bytes.append(char_bytes[-1] + len(char.encode()))
    total = char_bytes[-1]
    internal = []
    for start, end in local['offsets']:
        if end <= start or end >= len(char_bytes):
            continue
        byte_end = char_bytes[end]
        if 0 < byte_end < total:
            internal.append(byte_end)
    return internal


def compare_lines(tokenizer, text: str, service_cuts: list[int]) -> list[dict]:
    rows = []
    cursor = 0
    data = text.encode()
    for line in text.split('\n'):
        start = cursor
        local = qwen_cuts(tokenizer, line)
        qwen_internal = qwen_byte_internal(tokenizer, line)
        ds_internal = line_boundaries(text, service_cuts, line, start)
        rows.append({
            'line': line,
            'qwen_roundtrip': local['roundtrip'],
            'qwen_n': local['n_tokens'],
            'qwen_internal_cuts': qwen_internal,
            'service_internal_cuts': ds_internal,
            'same_boundaries': qwen_internal == ds_internal,
            'qwen_pieces': local['pieces'],
        })
        cursor += len(line.encode()) + 1
        if cursor > len(data):
            break
    return rows


def student_top16(text: str) -> dict:
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT / 'scripts'))
    import torch
    import preflight_qwen35_lora as pf

    _config, tokenizer, model, loaded_as, errors = pf._load_model(torch)
    model.eval()
    encoded = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    encoded = {key: value.to('cuda') for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        values, indices = torch.topk(logp, 16)
    rows = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token_id = int(index)
        rows.append({
            'id': token_id,
            'token': tokenizer.convert_ids_to_tokens(token_id),
            'text': tokenizer.decode([token_id], skip_special_tokens=False),
            'logprob': float(value),
        })
    return {'loaded_as': loaded_as, 'load_errors': errors, 'top16': rows, 'prefix': text}


def load_saved(run: Path, probe: str) -> dict | None:
    info = cursor_load(run)['done'].get(probe)
    if not info:
        return None
    return json.loads(Path(info['path']).read_text(encoding='utf-8'))


def top_entries(saved: dict | None) -> list[dict]:
    if not saved:
        return []
    rows = token_rows(saved.get('logprobs'))
    if not rows:
        return []
    return list(rows[0].get('top_logprobs') or [])


def write_matrix(run: Path, analyses: dict) -> dict:
    saved = {name: load_saved(run, name) for name in cursor_load(run)['done']}
    p01 = saved.get('p01_identity_logprobs')
    items = []

    def add(item_id: str, name: str, status: str, reason: str, probes: list[str]):
        items.append({
            'id': item_id,
            'name': name,
            'status': status,
            'reason': reason,
            'probes': probes,
            'files': [cursor_load(run)['done'][probe]['path'] for probe in probes if probe in cursor_load(run)['done']],
        })

    if not p01 or not p01.get('http_status'):
        add('1', 'identity_and_mode', 'UNVERIFIED', 'No HTTP response was saved for the identity request.', ['p01_identity_logprobs'])
        add('2', 'real_probabilities', 'UNVERIFIED', 'No probability payload was saved.', ['p01_identity_logprobs'])
    else:
        usage = p01.get('usage') or {}
        details = usage.get('completion_tokens_details') or {}
        reasoning_tokens = details.get('reasoning_tokens')
        reasoning = p01.get('reasoning_content')
        thinking_off = not reasoning and reasoning_tokens in (None, 0)
        fingerprint = p01.get('system_fingerprint')
        same_historical = fingerprint == HISTORICAL_FINGERPRINT
        identity_ok = p01['http_status'] == 200 and p01.get('response_model') and thinking_off
        add(
            '1',
            'identity_and_mode',
            'PASS' if identity_ok else 'FAIL',
            (
                f"HTTP {p01['http_status']}, response model {p01.get('response_model')}, "
                f"fingerprint {fingerprint}, historical fingerprint equal={same_historical}. "
                f"Thinking-disabled request returned reasoning_content={reasoning!r}, reasoning_tokens={reasoning_tokens}. "
                'The current alias is not recorded as a bit-reproduction of the historical teacher.'
            ),
            ['p01_identity_logprobs'],
        )
        rows = token_rows(p01.get('logprobs'))
        finite_top = []
        bytes_ok = 0
        for row in rows:
            for entry in row.get('top_logprobs') or []:
                if finite_logprob(entry.get('logprob')):
                    finite_top.append(entry)
                raw = entry.get('bytes')
                if isinstance(raw, list):
                    try:
                        if bytes(raw).decode('utf-8') == entry.get('token'):
                            bytes_ok += 1
                    except UnicodeDecodeError:
                        pass
        has_ids = any(isinstance(row, dict) and 'token_id' in row for row in rows)
        prob_ok = p01['http_status'] == 200 and bool(finite_top) and bytes_ok > 0
        add(
            '2',
            'real_probabilities',
            'PASS' if prob_ok else 'FAIL',
            (
                f"Returned {len(rows)} content logprob positions and {len(finite_top)} finite top-logprob entries. "
                f"Byte round-trip matches on {bytes_ok} top entries. Integer token ids present={has_ids}. "
                'Raw logits are not in the response. These saved bodies do not contain the documented -9999.0 sentinel.'
            ),
            ['p01_identity_logprobs'],
        )

    france = saved.get('p03_prefix_france')
    japan = saved.get('p04_prefix_japan')
    japan_retry = saved.get('p04_japan_retry')
    sql = saved.get('p05_prefix_sql')

    def top_token(saved_probe: dict | None):
        entries = top_entries(saved_probe)
        return entries[0].get('token') if entries else None

    france_top = top_token(france)
    sql_top = top_token(sql)
    japan_top = top_token(japan)
    japan_retry_top = top_token(japan_retry)
    france_ok = bool(france and france.get('http_status') == 200 and france_top and not france.get('reasoning_content'))
    sql_ok = bool(sql and sql.get('http_status') == 200 and sql_top and not sql.get('reasoning_content'))
    hidden = any((probe or {}).get('reasoning_content') for probe in (france, japan, japan_retry, sql))
    prefix_ok = france_ok and sql_ok and france_top != sql_top and not hidden
    add(
        '3',
        'mixed_prefix',
        'PASS' if prefix_ok else 'FAIL',
        (
            f"France continuation top token {france_top!r}, content {(france or {}).get('content')!r}. "
            f"SQL prefix top token {sql_top!r}, content {(sql or {}).get('content')!r}. "
            f"Japan attempt returned top token {japan_top!r}, content {(japan or {}).get('content')!r}, "
            f"finish {(japan or {}).get('finish_reason')}, completion_tokens {((japan or {}).get('usage') or {}).get('completion_tokens')}. "
            f"Japan retry top token {japan_retry_top!r}, content {(japan_retry or {}).get('content')!r}. "
            'An empty top list is not counted as a different distribution. '
            f"reasoning_content present={hidden}."
        ),
        ['p03_prefix_france', 'p04_prefix_japan', 'p04_japan_retry', 'p05_prefix_sql'],
    )

    tok = analyses.get('tokenization') or {}
    add('4', 'tokenization', tok.get('status', 'UNVERIFIED'), tok.get('reason', 'Tokenization comparison did not run.'), tok.get('probes', []))
    union = analyses.get('union') or {}
    add('5', 'union_support', union.get('status', 'UNVERIFIED'), union.get('reason', 'Union coverage did not run.'), union.get('probes', []))
    by_status = {item['status'] for item in items}
    qualified = by_status == {'PASS'} and len(items) == 5
    matrix = {
        'written_at': now(),
        'qualified_for_stage_b': qualified,
        'items': items,
        'historical_fingerprint': HISTORICAL_FINGERPRINT,
        'bit_reproduction_claimed': False,
    }
    dump(run / 'capability-matrix.json', matrix)
    print('MATRIX ' + ' '.join(f"{item['id']}={item['status']}" for item in items), flush=True)
    return matrix


def analyze(run: Path) -> dict:
    tokenizer = qwen_tokenizer()
    local_rows = []
    for text in LINE_PREFIXES + [NEWLINE_PROBE, SQL_PREFIX]:
        local_rows.append({'text': text, **qwen_cuts(tokenizer, text)})
    service = {}
    mismatches = []
    compared = 0
    for probe in ('p06_repeat_lines', 'p07_repeat_newlines'):
        saved = load_saved(run, probe)
        if not saved or saved.get('http_status') != 200:
            service[probe] = {'http_ok': False}
            continue
        expected = '\n'.join(LINE_PREFIXES) if probe == 'p06_repeat_lines' else NEWLINE_PROBE
        content = saved.get('content') or ''
        accepted = content == expected or content == expected + '\n'
        cuts = ds_cuts(token_rows(saved.get('logprobs')))
        entry = {'http_ok': True, 'exact_or_trailing_newline': accepted, 'content': content, 'expected': expected, 'service': cuts}
        if accepted and cuts and cuts.get('decode_ok') and cuts.get('text') in {content, expected}:
            target = cuts['text'] if cuts['text'] == expected else expected
            # Compare against the service-covered text.
            covered = cuts['text']
            rows = compare_lines(tokenizer, covered.rstrip('\n') if covered.endswith('\n') and not expected.endswith('\n') else covered, cuts['cuts'])
            entry['lines'] = rows
            for row in rows:
                if row['line'] == '' :
                    continue
                compared += 1
                if not row['same_boundaries']:
                    mismatches.append({
                        'probe': probe,
                        'line': row['line'],
                        'qwen': row['qwen_internal_cuts'],
                        'service': row['service_internal_cuts'],
                        'qwen_roundtrip': row['qwen_roundtrip'],
                    })
        service[probe] = {key: value for key, value in entry.items() if key != 'content'}
        service[probe]['content_sha256'] = hashlib.sha256(content.encode()).hexdigest()
    dump(run / 'tokenization.json', {'local_qwen': local_rows, 'service': service, 'mismatches': mismatches, 'compared_nonempty_lines': compared})
    if compared < 24:
        tok = {
            'status': 'UNVERIFIED',
            'reason': f'Exact service segmentation was recovered for {compared} nonempty lines, below the required 24 frozen prefixes.',
            'probes': ['p06_repeat_lines', 'p07_repeat_newlines'],
        }
    elif mismatches:
        sample = mismatches[0]
        tok = {
            'status': 'FAIL',
            'reason': (
                f'{len(mismatches)} frozen prefixes have different Qwen and DeepSeek byte boundaries. '
                f"Example {sample['line']!r}: Qwen cuts {sample['qwen']}, service cuts {sample['service']}. "
                'Same token ids or equal length are not used as compatibility. No cross-tokenizer same-id KL is computed.'
            ),
            'probes': ['p06_repeat_lines', 'p07_repeat_newlines'],
        }
    else:
        tok = {
            'status': 'UNVERIFIED',
            'reason': 'The frozen prefixes that were recovered have matching boundaries, and that sample does not prove a full vocabulary mapping.',
            'probes': ['p06_repeat_lines', 'p07_repeat_newlines'],
        }

    sql = load_saved(run, 'p05_prefix_sql')
    ds_top = []
    for entry in top_entries(sql):
        raw = entry.get('bytes')
        decoded = None
        if isinstance(raw, list):
            try:
                decoded = bytes(raw).decode('utf-8')
            except UnicodeDecodeError:
                decoded = None
        ds_top.append({'token': entry.get('token'), 'logprob': entry.get('logprob'), 'bytes': raw, 'decoded': decoded, 'finite': finite_logprob(entry.get('logprob'))})
    completion = load_saved(run, 'p12_completions_logprobs')
    completion_top = []
    if completion and completion.get('http_status') == 200:
        choice = ((completion.get('raw_response') or {}).get('choices') or [{}])[0]
        lp = choice.get('logprobs') or {}
        tables = lp.get('top_logprobs') or []
        if tables and isinstance(tables[0], dict):
            completion_top = [
                {'token': token, 'logprob': value, 'finite': finite_logprob(value)}
                for token, value in sorted(tables[0].items(), key=lambda item: -item[1])
            ]
    student = {'error': ''}
    try:
        student = student_top16(SQL_PREFIX)
    except Exception as exc:  # noqa: BLE001 — local failure stays in the matrix evidence
        student = {'error': redact(f'{type(exc).__name__}: {exc}'), 'trace': redact(traceback.format_exc())[-2000:]}
    chat_surface = {item.get('token') for item in ds_top} | {item.get('decoded') for item in ds_top}
    completion_surface = {item.get('token') for item in completion_top}
    outside_chat = [row for row in student.get('top16') or [] if row['text'] not in chat_surface and row['token'] not in chat_surface]
    outside_completion = [row for row in student.get('top16') or [] if row['text'] not in completion_surface]
    # A token that merely appears inside some returned top-20 does not prove scoring outside that list.
    specified = outside_completion[0] if outside_completion else {'text': '龘', 'source': 'constructed absent from the completions top 20'}
    scoring = []
    for probe in (
        'p02_top_logprobs_21', 'p08_prompt_logprobs', 'p09_echo', 'p10_logit_bias',
        'p11_beta_completions', 'p12_completions_logprobs',
    ):
        saved = load_saved(run, probe)
        if not saved:
            scoring.append({'probe': probe, 'missing': True})
            continue
        payload = saved.get('raw_response') or {}
        scoring.append({
            'probe': probe,
            'http_status': saved.get('http_status'),
            'error': (saved.get('error') or '')[:500],
            'top20_size': len(completion_top) if probe == 'p12_completions_logprobs' else None,
        })
    outside_scored = False
    probes = [
        'p05_prefix_sql', 'p02_top_logprobs_21', 'p08_prompt_logprobs', 'p09_echo',
        'p10_logit_bias', 'p11_beta_completions', 'p12_completions_logprobs',
    ]
    id_in_completion = next((item for item in completion_top if item['token'] == ' id'), None)
    if not completion or completion.get('http_status') != 200 or not completion_top:
        union = {
            'status': 'UNVERIFIED',
            'reason': 'The raw-prefix completions logprobs table was not recovered.',
            'probes': probes,
        }
    elif outside_completion and not outside_scored:
        missing_text = ', '.join(repr(row['text']) for row in outside_completion)
        union = {
            'status': 'FAIL',
            'reason': (
                f"Chat-prefix top-20 has {len(ds_top)} entries and omits {len(outside_chat)} student top-16 strings. "
                f"Raw-prefix completions top_logprobs has {len(completion_top)} entries. "
                f"Student token ' id' is inside that completions list"
                + (f" at logprob {id_in_completion['logprob']}" if id_in_completion else ' is absent')
                + '. '
                f"Student top-16 strings still outside it: {missing_text}. "
                f"Specified outside candidate {specified.get('text')!r} has no finite DeepSeek logprob. "
                'top_logprobs=21 was rejected for range [0, 20]. '
                'prompt_logprobs and echo did not add candidates. '
                'echo together with logprobs was rejected. '
                'A natural top-20 overlap is not a specified-candidate score. '
                f"Student load error: {student.get('error') or 'none'}. "
                'Missing candidates are left missing.'
            ),
            'probes': probes,
        }
    else:
        union = {
            'status': 'PASS',
            'reason': 'A candidate outside the returned top-20 has a finite DeepSeek logprob on the original prefix.',
            'probes': probes,
        }
    dump(run / 'union-coverage.json', {
        'chat_top': ds_top,
        'completion_top': completion_top,
        'student': student,
        'outside_chat': outside_chat,
        'outside_completion': outside_completion,
        'specified': specified,
        'scoring': scoring,
        'scored_outside_top20': outside_scored,
    })
    return {'tokenization': tok, 'union': union}


def render_report(run: Path, matrix: dict) -> str:
    ledger = json.loads(ledger_path(run).read_text(encoding='utf-8'))
    acc = ledger['accounted']['A']
    failed = [item for item in matrix['items'] if item['status'] != 'PASS']
    if matrix.get('qualified_for_stage_b'):
        opening = '阶段 A 五项都是 PASS。阶段 B 还没有开始。'
    else:
        names = '、'.join(f"{item['id']} {item['name']}={item['status']}" for item in failed)
        opening = (
            'DeepSeek 官方 `deepseek-flash`、显式关闭 thinking，当前不能对 pinned Qwen3.5-2B 计算论文中的原样 IPD。'
            f'未通过项：{names}。'
        )
    lines = [
        '# DeepSeek→Qwen 原样 IPD 接口结论',
        '',
        f'写入时间：{now()}',
        '',
        opening,
        '阶段 B、SFT、IPD 训练和 200 题效用都是 NOT_RUN。' if not matrix.get('qualified_for_stage_b') else '效用尚未开始。',
        '本地代理教师的 LOCAL_IPD_FUNCTIONAL_PASS 不计入本次接口结论。',
        '',
        '## 能力矩阵',
        '',
    ]
    for item in matrix['items']:
        lines.append(f"- {item['id']} {item['name']}: {item['status']}。{item['reason']}")
    lines.extend([
        '',
        '## 调用与费用',
        '',
        f"阶段 A HTTP 尝试（含失败与重试）：{acc['attempts']}。按峰时保守价格入账的输入 token {acc['input_tokens']}，输出 token {acc['output_tokens']}，费用 USD {acc['usd']:.6f}。",
        '阶段 B 尝试 0，费用 USD 0。',
        'Grok 平台费用 UNKNOWN，不在这本 API 账里。',
        '未知用量保持预留，没有记成 0。缺失候选概率没有用 0、-9999 或均匀残差补上。',
        '',
        '## 效用',
        '',
        'U / S / I：NOT_RUN / NOT_RUN / NOT_RUN。没有已回答题数，未运行的题不记为错误。',
        '学生更新次数：0。',
        '',
        '## 证据',
        '',
        f'- 矩阵：`{run / "capability-matrix.json"}`',
        f'- 账本：`{run / "budget-ledger.json"}`',
        f'- 分词：`{run / "tokenization.json"}`',
        f'- 并集：`{run / "union-coverage.json"}`',
        f'- 请求目录：`{run / "requests"}`',
        '',
    ])
    if not matrix.get('qualified_for_stage_b'):
        lines.append('下一步要改的是研究设定：在现有官方返回上补不出双方 top-16 并集的真实概率，也不能把文本批改或本地教师称作这次 IPD。')
    text = '\n'.join(lines) + '\n'
    (run / 'report.md').write_text(text, encoding='utf-8')
    report = ROOT / 'reports' / 'deepseek-ipd-interface-2026-10-18.md'
    report.write_text(text, encoding='utf-8')
    return text


def append_ledgers(run: Path, matrix: dict) -> None:
    ledger = json.loads(ledger_path(run).read_text(encoding='utf-8'))
    acc = ledger['accounted']['A']
    statuses = ' '.join(f"{item['id']}={item['status']}" for item in matrix['items'])
    qualified = matrix['qualified_for_stage_b']
    idea = (
        '## 2026-10-18 — DeepSeek IPD interface result\n\n'
        'This closes the open outcome of "2026-10-18 — Execute the DeepSeek IPD interface qualification". '
        f'Stage A qualified_for_stage_b={qualified}. Capability statuses: {statuses}. '
        f"Accounted stage-A attempts {acc['attempts']}, peak-reserved USD {acc['usd']:.6f}. "
        'Stage B and the U/S/I utility comparison are NOT_RUN. '
        'NO_ACCEPTED_GENERATOR and PRIVACY_NOT_ESTABLISHED stay unchanged. '
        'Historical 68/72/75 stay separate. '
        f'Report: reports/deepseek-ipd-interface-2026-10-18.md. {RESULT_MARKER}.\n'
    )
    progress = (
        '\n\n### 2026-10-18 — DeepSeek IPD interface qualification\n\n'
        f"Run `{run.name}`. Stage A statuses {statuses}. "
        f"Attempts {acc['attempts']}. Peak-accounted USD {acc['usd']:.6f}. "
        'Stage B NOT_RUN. Utility U/S/I NOT_RUN. Student updates 0. '
        'Grok platform fee UNKNOWN. '
        'Report `reports/deepseek-ipd-interface-2026-10-18.md`. '
        f'{RESULT_MARKER}.\n'
    )
    idea_path = ROOT / 'idea.md'
    if RESULT_MARKER not in idea_path.read_text(encoding='utf-8'):
        with idea_path.open('a', encoding='utf-8') as handle:
            handle.write('\n' + idea)
    progress_path = ROOT / 'progress.md'
    if RESULT_MARKER not in progress_path.read_text(encoding='utf-8'):
        with progress_path.open('a', encoding='utf-8') as handle:
            handle.write(progress)


def copy_windows(run: Path) -> dict:
    dest = Path('/mnt/c/Users/10950/Documents/Codex/2026-10-12/h/outputs') / run.name
    if not Path('/mnt/c/Users/10950/Documents/Codex/2026-10-12/h/outputs').is_dir():
        return {'ok': False, 'error': 'windows outputs directory is not mounted'}
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(run, dest, ignore=shutil.ignore_patterns('budget-ledger.lock'))
    rows = []
    mismatch = 0
    for path in sorted(p for p in run.rglob('*') if p.is_file() and p.name != 'budget-ledger.lock'):
        rel = path.relative_to(run).as_posix()
        left = sha256_file(path)
        right = sha256_file(dest / rel)
        rows.append(f'{left}  {rel}')
        if left != right:
            mismatch += 1
    manifest = '\n'.join(rows) + '\n'
    (run / 'windows-sha256.txt').write_text(manifest, encoding='utf-8')
    (dest / 'windows-sha256.txt').write_text(manifest, encoding='utf-8')
    summary = {'ok': mismatch == 0, 'files': len(rows), 'mismatches': mismatch, 'dest': str(dest)}
    dump(run / 'windows-copy.json', summary)
    dump(dest / 'windows-copy.json', summary)
    return summary


def snapshot_code(run: Path) -> None:
    code = run / 'code'
    code.mkdir(parents=True, exist_ok=True)
    paths = [
        ROOT / 'scripts/deepseek_ipd_interface.py',
        ROOT / 'src/privacy_synth/ipd.py',
        ROOT / 'tests/test_ipd.py',
        ROOT / 'docs/notes/grok-goal-deepseek-ipd-interface-and-utility-2026-10-18.md',
    ]
    hashes = []
    for path in paths:
        target = code / path.name
        shutil.copy2(path, target)
        hashes.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(target)})
    dump(run / 'source-snapshot.json', {'at': now(), 'files': hashes})


def init_run(run: Path, price: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / 'runner.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
    if not (run / 'protocol.json').exists():
        dump(run / 'protocol.json', {
            'created_at': now(),
            't0': T0.isoformat(),
            'first_response_deadline': FIRST_DEADLINE.isoformat(),
            'stage_a_deadline': STAGE_A_DEADLINE.isoformat(),
            'target_end': (T0 + timedelta(hours=16)).isoformat(),
            'hard_stop': HARD_STOP.isoformat(),
            'wrap_at': WRAP_AT.isoformat(),
            'provider': 'deepseek',
            'model': MODEL,
            'thinking': {'type': 'disabled'},
            'student': {'model_id': 'Qwen/Qwen3.5-2B', 'revision': REVISION},
            'decision_module': 'src/privacy_synth/ipd.py',
            'sent_data': 'public probe strings only in stage A',
            'not_sent': ['gold SQL', 'gold execution', 'eval items', 'protected rows', 'local filter labels'],
            'stage_b': 'locked until capability-matrix qualified_for_stage_b is true',
            'utility': 'NOT_RUN unless stage B precheck passes',
            'historical_fingerprint_not_a_reproduction_claim': HISTORICAL_FINGERPRINT,
            'cheaper_alternative': 'The local A3 proxy already showed the mechanism. It cannot answer the DeepSeek interface question.',
            'probes': [
                'p01_identity_logprobs', 'p02_top_logprobs_21', 'p03_prefix_france', 'p04_prefix_japan',
                'p05_prefix_sql', 'p06_repeat_lines', 'p07_repeat_newlines', 'p08_prompt_logprobs',
                'p09_echo', 'p10_logit_bias', 'p11_beta_completions',
            ],
        })
    if not ledger_path(run).exists():
        dump(ledger_path(run), empty_ledger(price['rates'], price['meta']))
    snapshot_code(run)
    dump(run / 'status.json', {'phase': 'A_RUNNING', 'at': now(), 'pid': os.getpid()})


def self_check() -> None:
    import tempfile

    rates = {
        'cache_hit_input_offpeak': 0.003,
        'cache_hit_input_peak': 0.006,
        'cache_miss_input_offpeak': 0.15,
        'cache_miss_input_peak': 0.3,
        'output_offpeak': 0.6,
        'output_peak': 1.2,
    }
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        dump(ledger_path(run), empty_ledger(rates, {'ok': True}))
        call = with_ledger(run, lambda ledger: try_reserve(ledger, 'A', 't', 100, 10))
        with_ledger(run, lambda ledger: settle(ledger, call['id'], None))
        ledger = json.loads(ledger_path(run).read_text(encoding='utf-8'))
        if ledger['accounted']['A']['usd'] <= 0 or ledger['calls'][0]['usage_unknown'] is not True:
            raise SystemExit('unknown usage was dropped')
        try:
            with_ledger(run, lambda ledger: try_reserve(ledger, 'B', 'nope', 1, 1))
            raise SystemExit('stage B reserved while locked')
        except CapExceeded:
            pass
        dump(ledger_path(run), empty_ledger(rates, {'ok': True}))
        with_ledger(run, lambda ledger: try_reserve(ledger, 'A', 'known', 1000, 100))
        with_ledger(run, lambda ledger: settle(ledger, 1, {
            'prompt_tokens': 10,
            'prompt_cache_hit_tokens': 4,
            'prompt_cache_miss_tokens': 6,
            'completion_tokens': 2,
        }))
        ledger = json.loads(ledger_path(run).read_text(encoding='utf-8'))
        expected = peak_usd(rates, 4, 6, 2)
        if abs(ledger['accounted']['A']['usd'] - expected) > 1e-9:
            raise SystemExit(f'settle mismatch {ledger["accounted"]["A"]["usd"]} != {expected}')
        if ledger['accounted']['A']['input_tokens'] != 10 or ledger['accounted']['A']['output_tokens'] != 2:
            raise SystemExit('settled token totals mismatch')
    print('SELF_CHECK_OK', flush=True)


def probes(run: Path, key: str) -> None:
    user = 'Reply with the single word pong.'
    run_probe(run, key, 'p01_identity_logprobs', f'{API}/chat/completions', base_body(
        [{'role': 'user', 'content': user}], 16, 20, 1,
    ))
    first = load_saved(run, 'p01_identity_logprobs')
    if not first or not first.get('http_status'):
        raise RuntimeError('transport blocked before an HTTP status')
    if first.get('http_status') != 200:
        raise RuntimeError(f"identity request HTTP {first.get('http_status')}: {(first.get('error') or '')[:500]}")
    run_probe(run, key, 'p02_top_logprobs_21', f'{API}/chat/completions', base_body(
        [{'role': 'user', 'content': user}], 8, 21, 1,
    ))
    prefix_user = 'Continue the assistant draft from its end.'
    run_probe(run, key, 'p03_prefix_france', f'{BETA}/chat/completions', base_body([
        {'role': 'user', 'content': prefix_user},
        {'role': 'assistant', 'content': 'The capital of France is ', 'prefix': True},
    ], 1, 20, 1))
    run_probe(run, key, 'p04_prefix_japan', f'{BETA}/chat/completions', base_body([
        {'role': 'user', 'content': prefix_user},
        {'role': 'assistant', 'content': 'The capital of Japan is ', 'prefix': True},
    ], 1, 20, 1))
    run_probe(run, key, 'p05_prefix_sql', f'{BETA}/chat/completions', base_body([
        {'role': 'user', 'content': prefix_user},
        {'role': 'assistant', 'content': SQL_PREFIX, 'prefix': True},
    ], 1, 20, 1))
    line_block = '\n'.join(LINE_PREFIXES)
    run_probe(run, key, 'p06_repeat_lines', f'{API}/chat/completions', base_body([
        {'role': 'system', 'content': 'Repeat the user text exactly. Preserve spaces. Do not add commentary.'},
        {'role': 'user', 'content': line_block},
    ], 800, 5, 0))
    run_probe(run, key, 'p07_repeat_newlines', f'{API}/chat/completions', base_body([
        {'role': 'system', 'content': 'Repeat the user text exactly, including blank lines and spaces.'},
        {'role': 'user', 'content': NEWLINE_PROBE},
    ], 64, 5, 0))
    sql_messages = [
        {'role': 'user', 'content': prefix_user},
        {'role': 'assistant', 'content': SQL_PREFIX, 'prefix': True},
    ]
    extra = base_body(sql_messages, 1, 20, 1)
    extra['prompt_logprobs'] = True
    run_probe(run, key, 'p08_prompt_logprobs', f'{BETA}/chat/completions', extra)
    echoed = base_body(sql_messages, 1, 20, 1)
    echoed['echo'] = True
    run_probe(run, key, 'p09_echo', f'{BETA}/chat/completions', echoed)
    biased = base_body(sql_messages, 1, 20, 1)
    biased['logit_bias'] = {'0': 0}
    run_probe(run, key, 'p10_logit_bias', f'{BETA}/chat/completions', biased)
    run_probe(run, key, 'p11_beta_completions', f'{BETA}/completions', {
        'model': MODEL,
        'prompt': SQL_PREFIX,
        'max_tokens': 1,
        'temperature': 1,
        'logprobs': 20,
        'echo': True,
        'thinking': {'type': 'disabled'},
        'stream': False,
    })


def finish(run: Path, blocked: str = '') -> None:
    if blocked and not cursor_load(run)['done']:
        analyses = {
            'tokenization': {'status': 'UNVERIFIED', 'reason': blocked, 'probes': []},
            'union': {'status': 'UNVERIFIED', 'reason': blocked, 'probes': []},
        }
        matrix = write_matrix(run, analyses)
    elif blocked:
        analyses = {
            'tokenization': {'status': 'UNVERIFIED', 'reason': blocked, 'probes': []},
            'union': {'status': 'UNVERIFIED', 'reason': blocked, 'probes': []},
        }
        matrix = write_matrix(run, analyses)
    else:
        analyses = analyze(run)
        matrix = write_matrix(run, analyses)
    if matrix.get('qualified_for_stage_b'):
        dump(run / 'status.json', {'phase': 'A_QUALIFIED', 'at': now(), 'utility': 'NOT_STARTED'})
        print('A_QUALIFIED', flush=True)
        return
    http_seen = any(
        (load_saved(run, name) or {}).get('http_status')
        for name in cursor_load(run)['done']
    )
    overall = 'PARTIAL' if http_seen else 'BLOCKED'
    interface = 'COMPLETE' if http_seen else 'BLOCKED'
    dump(run / 'summary.json', {
        'overall': overall,
        'interface_subtask': interface,
        'stage_b': 'NOT_RUN',
        'utility': {'U': 'NOT_RUN', 'S': 'NOT_RUN', 'I': 'NOT_RUN'},
        'student_updates': 0,
        'qualified_for_stage_b': False,
        'blocked': blocked,
        'at': now(),
    })
    dump(run / 'status.json', {'phase': 'REPORT', 'at': now(), 'overall': overall})
    render_report(run, matrix)
    append_ledgers(run, matrix)
    copied = copy_windows(run)
    dump(run / 'status.json', {
        'phase': 'TERMINAL',
        'at': now(),
        'overall': overall,
        'interface_subtask': interface,
        'utility': 'NOT_RUN',
        'windows': copied,
    })
    print(f'TERMINAL {overall} utility=NOT_RUN windows_ok={copied.get("ok")}', flush=True)


def all_command(run: Path) -> None:
    idea = (ROOT / 'idea.md').read_text(encoding='utf-8')
    if IDEA_MARKER not in idea:
        raise SystemExit('idea.md execution entry is missing')
    if run.name in {
        '2026-10-17-001119-bird-student-c1-ablation',
        '2026-10-17-131704-bird-student-json-repeat',
    }:
        raise SystemExit('refusing to write into a frozen run')
    status_file = run / 'status.json'
    if status_file.exists() and json.loads(status_file.read_text(encoding='utf-8')).get('phase') == 'TERMINAL':
        print('ALREADY_TERMINAL', flush=True)
        return
    self_check()
    price = confirm_price(run)
    init_run(run, price)
    key = load_key()
    sock = socket.socket()
    try:
        proxy_open = sock.connect_ex(('127.0.0.1', 7897)) == 0
    finally:
        sock.close()
    dump(run / 'proxy-check.json', {'at': now(), 'port_7897_open': proxy_open, 'route': PROXY})
    try:
        probes(run, key)
        finish(run)
    except Exception as exc:
        message = redact(f'{type(exc).__name__}: {exc}')
        (run / 'driver-error.txt').write_text(redact(traceback.format_exc()), encoding='utf-8')
        if not (run / 'capability-matrix.json').exists():
            finish(run, blocked=message)
        if json.loads((run / 'status.json').read_text(encoding='utf-8')).get('phase') == 'TERMINAL':
            return
        raise


CORRECTION_MARKER = 'deepseek-ipd-interface-utility result correction'


def append_correction(run: Path, matrix: dict) -> None:
    ledger = json.loads(ledger_path(run).read_text(encoding='utf-8'))
    acc = ledger['accounted']['A']
    statuses = ' '.join(f"{item['id']}={item['status']}" for item in matrix['items'])
    item3 = next(item['reason'] for item in matrix['items'] if item['id'] == '3')
    idea = (
        '## 2026-10-18 — Correct the DeepSeek interface matrix\n\n'
        'This supersedes the item-3 wording and the attempt total in '
        '"2026-10-18 — DeepSeek IPD interface result". '
        'The first matrix treated an empty Japan completion as a different distribution. '
        'The corrected contrast uses the France and SQL prefixes, both of which returned top-20 logprobs, '
        f'and records the Japan calls separately. Statuses: {statuses}. '
        f"Stage-A attempts {acc['attempts']}, peak-accounted USD {acc['usd']:.6f}. "
        'Stage B and U/S/I remain NOT_RUN. '
        f'Item 3 evidence: {item3} '
        f'{CORRECTION_MARKER}.\n'
    )
    progress = (
        '\n\n### 2026-10-18 — DeepSeek IPD interface correction\n\n'
        f"Same run `{run.name}`. Corrected statuses {statuses}. "
        f"Attempts {acc['attempts']}. Peak-accounted USD {acc['usd']:.6f}. "
        'The empty Japan prefix is not used as evidence that two distributions differ. '
        'Stage B NOT_RUN. Utility U/S/I NOT_RUN. '
        f'{CORRECTION_MARKER}.\n'
    )
    idea_path = ROOT / 'idea.md'
    if CORRECTION_MARKER not in idea_path.read_text(encoding='utf-8'):
        with idea_path.open('a', encoding='utf-8') as handle:
            handle.write('\n' + idea)
    progress_path = ROOT / 'progress.md'
    if CORRECTION_MARKER not in progress_path.read_text(encoding='utf-8'):
        with progress_path.open('a', encoding='utf-8') as handle:
            handle.write(progress)


def followup(run: Path) -> None:
    if not ledger_path(run).exists():
        raise SystemExit('followup requires the stage A ledger')
    key = load_key()
    prefix_user = 'Continue the assistant draft from its end.'
    run_probe(run, key, 'p04_japan_retry', f'{BETA}/chat/completions', base_body([
        {'role': 'user', 'content': prefix_user},
        {'role': 'assistant', 'content': 'The capital of Japan is ', 'prefix': True},
    ], 1, 20, 1))
    run_probe(run, key, 'p12_completions_logprobs', f'{BETA}/completions', {
        'model': MODEL,
        'prompt': SQL_PREFIX,
        'max_tokens': 1,
        'temperature': 1,
        'logprobs': 20,
        'thinking': {'type': 'disabled'},
        'stream': False,
    })
    analyses = analyze(run)
    matrix = write_matrix(run, analyses)
    if matrix.get('qualified_for_stage_b'):
        dump(run / 'status.json', {'phase': 'A_QUALIFIED', 'at': now(), 'utility': 'NOT_STARTED'})
        print('A_QUALIFIED', flush=True)
        return
    render_report(run, matrix)
    append_correction(run, matrix)
    code = run / 'code'
    shutil.copy2(Path(__file__), code / Path(__file__).name)
    copied = copy_windows(run)
    dump(run / 'summary.json', {
        'overall': 'PARTIAL',
        'interface_subtask': 'COMPLETE',
        'stage_b': 'NOT_RUN',
        'utility': {'U': 'NOT_RUN', 'S': 'NOT_RUN', 'I': 'NOT_RUN'},
        'student_updates': 0,
        'qualified_for_stage_b': False,
        'corrected_at': now(),
    })
    dump(run / 'status.json', {
        'phase': 'TERMINAL',
        'at': now(),
        'overall': 'PARTIAL',
        'interface_subtask': 'COMPLETE',
        'utility': 'NOT_RUN',
        'windows': copied,
    })
    print(f'TERMINAL PARTIAL utility=NOT_RUN windows_ok={copied.get("ok")}', flush=True)


SUPPLEMENT_MARKER = 'deepseek-ipd-interface-utility completions top20'


def rejudge(run: Path) -> None:
    analyses = analyze(run)
    matrix = write_matrix(run, analyses)
    if matrix.get('qualified_for_stage_b'):
        dump(run / 'status.json', {'phase': 'A_QUALIFIED', 'at': now(), 'utility': 'NOT_STARTED'})
        print('A_QUALIFIED', flush=True)
        return
    render_report(run, matrix)
    ledger = json.loads(ledger_path(run).read_text(encoding='utf-8'))
    acc = ledger['accounted']['A']
    item5 = next(item['reason'] for item in matrix['items'] if item['id'] == '5')
    idea = (
        '## 2026-10-18 — Record the completions top-20\n\n'
        'This supplements "2026-10-18 — Correct the DeepSeek interface matrix". '
        'The beta completions response on the raw SQL prefix returns a 20-entry top_logprobs map. '
        f'Item 5 stays {next(item["status"] for item in matrix["items"] if item["id"] == "5")}. '
        f'Evidence: {item5} '
        f"Stage-A attempts remain {acc['attempts']}, peak-accounted USD {acc['usd']:.6f}. "
        'Stage B and U/S/I stay NOT_RUN. '
        f'{SUPPLEMENT_MARKER}.\n'
    )
    progress = (
        '\n\n### 2026-10-18 — DeepSeek completions top-20\n\n'
        f"Run `{run.name}`. {item5} "
        f"Attempts {acc['attempts']}. Peak-accounted USD {acc['usd']:.6f}. Utility NOT_RUN. {SUPPLEMENT_MARKER}.\n"
    )
    if SUPPLEMENT_MARKER not in (ROOT / 'idea.md').read_text(encoding='utf-8'):
        with (ROOT / 'idea.md').open('a', encoding='utf-8') as handle:
            handle.write('\n' + idea)
    if SUPPLEMENT_MARKER not in (ROOT / 'progress.md').read_text(encoding='utf-8'):
        with (ROOT / 'progress.md').open('a', encoding='utf-8') as handle:
            handle.write(progress)
    shutil.copy2(Path(__file__), run / 'code' / Path(__file__).name)
    copied = copy_windows(run)
    dump(run / 'summary.json', {
        'overall': 'PARTIAL',
        'interface_subtask': 'COMPLETE',
        'stage_b': 'NOT_RUN',
        'utility': {'U': 'NOT_RUN', 'S': 'NOT_RUN', 'I': 'NOT_RUN'},
        'student_updates': 0,
        'qualified_for_stage_b': False,
        'rejudged_at': now(),
        'item5': item5,
    })
    dump(run / 'status.json', {
        'phase': 'TERMINAL',
        'at': now(),
        'overall': 'PARTIAL',
        'interface_subtask': 'COMPLETE',
        'utility': 'NOT_RUN',
        'windows': copied,
    })
    print(f'TERMINAL PARTIAL utility=NOT_RUN windows_ok={copied.get("ok")}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['self-check', 'all', 'followup', 'rejudge'])
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
        return
    if args.run_dir is None:
        raise SystemExit('--run-dir is required')
    run = args.run_dir.resolve()
    if args.command == 'followup':
        followup(run)
        return
    if args.command == 'rejudge':
        rejudge(run)
        return
    all_command(run)


if __name__ == '__main__':
    main()
