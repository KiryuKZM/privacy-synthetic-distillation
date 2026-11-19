"""CPU-only, credential-free persistence for bounded benchmark discovery.

Network orchestration belongs to scripts/. This module reuses the existing
SQLite ledger and peak-price function; no historical runner is modified.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from privacy_synth.pilot_ledger import SqliteBudget
from privacy_synth.round_budget import BudgetError, FLASH, worst_case_usd

INPUT_CAP = 32768
HTTP_CAP = 2486
LOGICAL_CAP = 2260
INPUT_TOTAL = 81461248
OUTPUT_TOTAL = 20365312
PHASE_COUNTS = {'dev': 128, 'repair': 32, 'screen': 150, 'fresh': 200}
HISTORY = (
    'runs/2026-10-22-night-dual/budget/new-round.sqlite',
    'runs/2026-10-22-cityfm-input-repair/budget/repair.sqlite',
    'runs/2026-10-22-cityfm-factorial/budget/factorial.sqlite',
    'runs/2026-10-22-cityfm-evidence-thinking/budget/evidence.sqlite',
    'runs/2026-10-23-matterport-ds-pilot/budget/matterport.sqlite',
    'runs/2026-10-23-matterport-adaptation-execution/budget/adaptation.sqlite',
)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()) + '.' + str(time.time_ns()))
    with temporary.open('w', encoding='utf-8') as target:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write('\n'); target.flush(); os.fsync(target.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def once(path, value):
    if Path(path).exists():
        if load(path) != value:
            raise PermissionError('Immutable receipt changed: ' + str(path))
    else:
        atomic(path, value)


def local_path(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
        raise PermissionError('Path leaves repository')
    return path


def historical_snapshot(root):
    records = []
    for name in HISTORY:
        path = local_path(root, name)
        before = sha(path)
        with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            db.execute('PRAGMA query_only=ON')
            n, pico, inflight = db.execute('SELECT COUNT(*),COALESCE(SUM(pico),0),COALESCE(SUM(1-settled),0) FROM calls').fetchone()
        if sha(path) != before or inflight:
            raise BudgetError('Historical ledger changed or has unsettled calls')
        records.append({'path': name, 'sha256': before, 'attempts': n, 'pico': pico})
    return {'ledgers': records, 'accounted_pico': sum(r['pico'] for r in records)}


def pools():
    result = {}
    for prefix, count, attempts, usd in [('candidate', 6, 341, 12), ('fresh', 2, 220, 14)]:
        for slot in range(count):
            result[prefix + str(slot + 1)] = {'calls': attempts, 'input_tokens': attempts * INPUT_CAP,
                'output_tokens': attempts * 8192, 'usd': usd}
    return result


class DiscoveryBudget:
    """Fixed eight budget pools with transactional dynamic candidate binding."""
    def __init__(self, run):
        self.run = Path(run)
        self.ledger = SqliteBudget(self.run / 'budget/discovery.sqlite', 100, pools(), {'hit': .30, 'miss': .30, 'output': 1.20})
        self.path = self.ledger.path
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS packages(
                candidate TEXT NOT NULL,kind TEXT NOT NULL,pool TEXT UNIQUE NOT NULL,PRIMARY KEY(candidate,kind));
                CREATE TABLE IF NOT EXISTS stages(candidate TEXT NOT NULL,phase TEXT NOT NULL,
                manifest_sha TEXT NOT NULL,logical_count INTEGER NOT NULL,PRIMARY KEY(candidate,phase));
                CREATE TABLE IF NOT EXISTS attempts(call_id INTEGER PRIMARY KEY,request_key TEXT NOT NULL,
                candidate TEXT NOT NULL,phase TEXT NOT NULL,ordinal INTEGER NOT NULL,
                state TEXT NOT NULL,result TEXT,UNIQUE(request_key,ordinal));
                CREATE TABLE IF NOT EXISTS members(candidate TEXT NOT NULL,phase TEXT NOT NULL,
                record_id TEXT NOT NULL,group_id TEXT NOT NULL,PRIMARY KEY(candidate,phase,record_id));''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def register(self, manifest_sha, manifest, rows):
        candidate, phase = manifest['candidate_id'], manifest['phase']
        kind = 'fresh' if phase == 'fresh' else 'candidate'
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT manifest_sha,logical_count FROM stages WHERE candidate=? AND phase=?', (candidate, phase)).fetchone()
            if old:
                if old != (manifest_sha, len(rows)):
                    raise BudgetError('Stage already frozen to another manifest')
                return
            previous = db.execute('SELECT phase,record_id,group_id FROM members WHERE candidate=?', (candidate,)).fetchall()
            members = {str(r['record_id']): r['group_id'] for r in rows}
            if phase in ('repair', 'screen') and not any(p == 'dev' for p, _, _ in previous):
                raise BudgetError('Dev must be registered before repair/screen')
            if phase == 'fresh' and not any(p == 'screen' for p, _, _ in previous):
                raise BudgetError('Screen must be registered before fresh')
            if phase == 'repair' and members != {rid: group for p, rid, group in previous if p == 'dev'}:
                raise BudgetError('Repair must retain exact Dev membership/groups')
            for old_phase, rid, group in previous:
                if phase in ('dev', 'repair') and old_phase in ('dev', 'repair'):
                    continue
                if rid in members or group in set(members.values()):
                    raise BudgetError('Evaluation stages overlap record/group membership')
            package = db.execute('SELECT pool FROM packages WHERE candidate=? AND kind=?', (candidate, kind)).fetchone()
            if not package:
                occupied = {r[0] for r in db.execute('SELECT pool FROM packages')}
                pool = next((p for p in pools() if p.startswith(kind) and p not in occupied), None)
                if pool is None:
                    raise BudgetError('Candidate/fresh package limit')
                if kind == 'fresh' and not db.execute('SELECT 1 FROM packages WHERE candidate=? AND kind=?', (candidate, 'candidate')).fetchone():
                    raise BudgetError('Fresh candidate lacks a discovery package')
                db.execute('INSERT INTO packages VALUES(?,?,?)', (candidate, kind, pool))
            total = db.execute('SELECT COALESCE(SUM(logical_count),0) FROM stages').fetchone()[0]
            if total + len(rows) > LOGICAL_CAP:
                raise BudgetError('Global logical cap')
            db.execute('INSERT INTO stages VALUES(?,?,?,?)', (candidate, phase, manifest_sha, len(rows)))
            db.executemany('INSERT INTO members VALUES(?,?,?,?)', [(candidate, phase, rid, group) for rid, group in members.items()])

    def prior(self, key):
        with self.connect() as db:
            return db.execute('SELECT a.call_id,a.ordinal,a.state,a.result,c.settled FROM attempts a JOIN calls c ON c.id=a.call_id WHERE request_key=? ORDER BY ordinal', (key,)).fetchall()

    def reserve(self, candidate, phase, key, ordinal, output, parent_pico):
        """Caller holds the shared parent flock and an in-process mutex."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            kind = 'fresh' if phase == 'fresh' else 'candidate'
            package = db.execute('SELECT pool FROM packages WHERE candidate=? AND kind=?', (candidate, kind)).fetchone()
            if not package:
                raise BudgetError('Unregistered package')
            n, inp, out, amount = db.execute('SELECT COUNT(*),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COALESCE(SUM(pico),0) FROM calls').fetchone()
            # round_budget supplies the authoritative conservative price.
            reserve = SqliteBudget.money(worst_case_usd(FLASH, INPUT_CAP, output))
            if n + 1 > HTTP_CAP or inp + INPUT_CAP > INPUT_TOTAL or out + output > OUTPUT_TOTAL:
                raise BudgetError('Global HTTP/token cap')
            if amount + reserve > SqliteBudget.money(100) or parent_pico + amount + reserve > SqliteBudget.money(500):
                raise BudgetError('Child100/parent500 cap')
            active, own = db.execute('SELECT COUNT(*),COALESCE(SUM(a.candidate=?),0) FROM calls c JOIN attempts a ON a.call_id=c.id WHERE c.settled=0', (candidate,)).fetchone()
            if active >= 4 or own >= 2:
                raise BudgetError('concurrency')
            if ordinal == 2:
                prior = db.execute('SELECT state,result FROM attempts WHERE request_key=? AND ordinal=1', (key,)).fetchone()
                if not prior or prior[0] != 'RETRYABLE' or not prior[1]:
                    raise BudgetError('Retry lacks explicit response')
                retries = db.execute('SELECT COUNT(*) FROM attempts WHERE candidate=? AND ordinal=2 AND (phase="fresh")=?', (candidate, phase == 'fresh')).fetchone()[0]
                if retries >= (20 if phase == 'fresh' else 31):
                    raise BudgetError('Package retry cap')
            elif ordinal != 1:
                raise BudgetError('At most one technical retry')
            cap = pools()[package[0]]
            pn, pi, po, pp = db.execute('SELECT COUNT(*),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COALESCE(SUM(pico),0) FROM calls WHERE pool=?', package).fetchone()
            if pn + 1 > cap['calls'] or pi + INPUT_CAP > cap['input_tokens'] or po + output > cap['output_tokens'] or pp + reserve > SqliteBudget.money(cap['usd']):
                raise BudgetError('Package HTTP/token/USD cap')
            ident = db.execute('INSERT INTO calls(pool,tag,input_tokens,output_tokens,pico) VALUES(?,?,?,?,?)', (package[0], key + '|attempt' + str(ordinal), INPUT_CAP, output, reserve)).lastrowid
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?)', (ident, key, candidate, phase, ordinal, 'RESERVED', None))
            return ident

    def finish(self, ident, result, usage):
        """Result is durably journaled before settlement, allowing safe replay."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT result FROM attempts WHERE call_id=?', (ident,)).fetchone()
            if old is None:
                raise BudgetError('Unknown attempt')
            serialized = json.dumps(result, sort_keys=True)
            if old[0] is not None and old[0] != serialized:
                raise BudgetError('Journaled response changed')
            db.execute('UPDATE attempts SET state=?,result=? WHERE call_id=?', ('RECEIVED_UNSETTLED', serialized, ident))
        with self.connect() as db:
            settled = db.execute('SELECT settled FROM calls WHERE id=?', (ident,)).fetchone()[0]
        if not settled:
            self.ledger.settle(ident, usage)
        with self.connect() as db:
            db.execute('UPDATE attempts SET state=? WHERE call_id=?', (result['status'], ident))

    def snapshot(self):
        with self.connect() as db:
            calls, inp, out, pico, unknown, active = db.execute('SELECT COUNT(*),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM calls').fetchone()
            packages = [dict(candidate=c, kind=k, pool=p) for c, k, p in db.execute('SELECT candidate,kind,pool FROM packages')]
            stages = [dict(candidate=c, phase=p, manifest_sha=s, logical_count=n) for c, p, s, n in db.execute('SELECT * FROM stages')]
        return dict(http_attempts=calls, input_tokens=inp, output_tokens=out, conservative_pico=pico,
                    unknown_usage=unknown, inflight=active, packages=packages, stages=stages)
