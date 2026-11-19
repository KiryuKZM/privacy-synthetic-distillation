"""Atomic spend ledger for the 2026-10-11 overnight pilot.

The 2026-10-10 ledger in round_budget.py is a different round and is not
opened here. This ledger reserves peak cache-miss prices. Unknown usage keeps
the full reserve. It does not call a network.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import threading
import uuid
from decimal import Decimal
from pathlib import Path

from privacy_synth.round_budget import BudgetError, worst_case_usd


class SqliteBudget:
    """Configurable bounded-workload ledger without rewriting all call history.

    Integer picodollars, transactional reservations and conservative peak rates.
    Unknown usage retains its full reservation. No network or credential access.
    The historical fixed-round JSON entrypoints below remain unchanged.
    """
    SCALE = 10**12

    def __init__(self, path, global_usd, pools, rates):
        import sqlite3
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.rates={k:Decimal(str(v)) for k,v in rates.items()}
        config={'global_pico':self.money(global_usd),'pools':pools,'rates':{k:str(v) for k,v in self.rates.items()}}
        with sqlite3.connect(self.path,timeout=30) as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.executescript('''CREATE TABLE IF NOT EXISTS settings(config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY AUTOINCREMENT,pool TEXT NOT NULL,
                tag TEXT UNIQUE NOT NULL,input_tokens INTEGER NOT NULL,output_tokens INTEGER NOT NULL,
                pico INTEGER NOT NULL,settled INTEGER NOT NULL DEFAULT 0,usage_known INTEGER NOT NULL DEFAULT 0);''')
            old=c.execute('SELECT config FROM settings').fetchone()
            if old:
                if json.loads(old[0])!=config:raise BudgetError('SQLite ledger contract changed')
            else:c.execute('INSERT INTO settings VALUES(?)',(json.dumps(config,sort_keys=True),))
        self.config=config

    @classmethod
    def money(cls,value):
        return int((Decimal(str(value))*cls.SCALE).to_integral_value(rounding='ROUND_CEILING'))

    def charge(self,hit,miss,output):
        amount=(Decimal(hit)*self.rates['hit']+Decimal(miss)*self.rates['miss']+Decimal(output)*self.rates['output'])/Decimal(1000000)
        return self.money(amount)

    def reserve(self,pool,tag,input_tokens,output_tokens):
        import sqlite3
        if input_tokens<0 or output_tokens<0:raise BudgetError('Negative token reservation')
        cap=self.config['pools'][pool];cost=self.charge(0,input_tokens,output_tokens)
        with sqlite3.connect(self.path,timeout=30) as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute('SELECT id FROM calls WHERE tag=?',(tag,)).fetchone():
                raise BudgetError('Call tag exists; never blindly resend an uncertain request')
            n,inp,out,spent=c.execute('SELECT COUNT(*),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COALESCE(SUM(pico),0) FROM calls WHERE pool=?',(pool,)).fetchone()
            total=c.execute('SELECT COALESCE(SUM(pico),0) FROM calls').fetchone()[0]
            if n+1>cap['calls'] or inp+input_tokens>cap['input_tokens'] or out+output_tokens>cap['output_tokens']:
                raise BudgetError('Pool call/token cap')
            if spent+cost>self.money(cap['usd']) or total+cost>self.config['global_pico']:
                raise BudgetError('Pool/global USD cap')
            return c.execute('INSERT INTO calls(pool,tag,input_tokens,output_tokens,pico) VALUES(?,?,?,?,?)',
                             (pool,tag,input_tokens,output_tokens,cost)).lastrowid

    def settle(self,call_id,usage):
        import sqlite3
        with sqlite3.connect(self.path,timeout=30) as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT input_tokens,output_tokens,pico,settled FROM calls WHERE id=?',(call_id,)).fetchone()
            if row is None or row[3]:raise BudgetError('Missing or already-settled call')
            if usage is None:
                c.execute('UPDATE calls SET settled=1 WHERE id=?',(call_id,));return
            inp=int(usage['prompt_tokens']);out=int(usage['completion_tokens'])
            hit=int(usage.get('prompt_cache_hit_tokens',0));miss=inp-hit
            if min(inp,out,hit,miss)<0:raise BudgetError('Invalid usage')
            cost=self.charge(hit,miss,out)
            if inp>row[0] or out>row[1] or cost>row[2]:
                raise BudgetError('Usage exceeded conservative reservation; stop and inspect')
            c.execute('UPDATE calls SET input_tokens=?,output_tokens=?,pico=?,settled=1,usage_known=1 WHERE id=?',
                      (inp,out,cost,call_id))

    def summary(self):
        import sqlite3
        with sqlite3.connect(self.path,timeout=30) as c:
            rows=c.execute('SELECT pool,COUNT(*),SUM(input_tokens),SUM(output_tokens),SUM(pico),SUM(1-usage_known),SUM(1-settled) FROM calls GROUP BY pool').fetchall()
        return {pool:dict(attempts=n,input_tokens=inp,output_tokens=out,peak_accounted_usd=float(Decimal(cost)/self.SCALE),
                          unknown_usage=unknown,inflight=inflight) for pool,n,inp,out,cost,unknown,inflight in rows}

FLASH = "deepseek-flash"
PRO = "deepseek-v4-pro"
ROUND_ID = "2026-10-11-overnight-022356"
GLOBAL_USD = Decimal("500")
GLOBAL_CALLS = 2688
PRO_CALLS = 32
CONCURRENCY = 8
MEETING_CAP = 4
POOLS = {
    "orchestration": {
        "usd_cap": Decimal("120"),
        "call_cap": 64,
        "models": {FLASH, PRO},
        "input_cap": 16384,
        "output_cap": 131072,
    },
    "public_calibration": {
        "usd_cap": Decimal("40"),
        "call_cap": 320,
        "models": {FLASH},
        "input_cap": 4096,
        "output_cap": 131072,
    },
    "synthetic_labels": {
        "usd_cap": Decimal("240"),
        "call_cap": 2304,
        "models": {FLASH},
        "input_cap": 4096,
        "output_cap": 131072,
    },
    "fault_reserve": {
        "usd_cap": Decimal("100"),
        "call_cap": None,
        "models": {FLASH, PRO},
        "input_cap": 16384,
        "output_cap": 131072,
    },
}
_QUANT = Decimal("0.000000000001")
_THREAD_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def _thread_lock_for(path: Path) -> threading.Lock:
    """fcntl.flock does not exclude threads in this process. Pair it with this lock."""
    key = str(path)
    with _THREAD_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def _money(value: Decimal) -> str:
    quantized = value.quantize(_QUANT)
    if quantized == 0:
        return "0"
    return format(quantized, "f").rstrip("0").rstrip(".")


def _dec(value: str) -> Decimal:
    return Decimal(value)


def empty_state() -> dict:
    pools = {}
    for name, spec in POOLS.items():
        pools[name] = {
            "usd_cap": _money(spec["usd_cap"]),
            "call_cap": spec["call_cap"],
            "spent_usd": "0",
            "reserved_usd": "0",
            "calls": 0,
            "open_calls": 0,
        }
    return {
        "round_id": ROUND_ID,
        "historical_spend_included": False,
        "grok_orchestrator_usd": "unverified",
        "price_stamp": {
            "flash_miss_peak": "0.30",
            "flash_out_peak": "1.20",
            "pro_miss_peak": "1.32",
            "pro_out_peak": "3.96",
            "source": "https://api-docs.deepseek.com/quick_start/pricing",
        },
        "spent_usd": "0",
        "reserved_usd": "0",
        "calls": 0,
        "open_calls": 0,
        "pro_calls": 0,
        "pro_open_calls": 0,
        "meetings": [],
        "mst_enabled": False,
        "pools": pools,
        "open": {},
        "events": [],
    }


def _check_stamp(state: dict) -> None:
    if state.get("round_id") != ROUND_ID or state.get("historical_spend_included") is not False:
        raise BudgetError("ledger is not the 2026-10-11 pilot")
    if state.get("price_stamp") != empty_state()["price_stamp"]:
        raise BudgetError("ledger price stamp does not match this round")


def reservation_block(
    state: dict,
    pool: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> str | None:
    if pool not in POOLS:
        return "unknown_pool"
    spec = POOLS[pool]
    if model not in spec["models"]:
        return "model_not_in_pool"
    if input_tokens > spec["input_cap"] or output_tokens > spec["output_cap"]:
        return "token_cap"
    if input_tokens < 0 or output_tokens < 0:
        return "token_cap"
    amount = worst_case_usd(model, input_tokens, output_tokens)
    if amount <= 0:
        return "zero_reserve"
    if state["open_calls"] + 1 > CONCURRENCY:
        return "concurrency"
    call_cap = spec["call_cap"]
    pool_calls = state["pools"][pool]["calls"] + state["pools"][pool]["open_calls"]
    if call_cap is not None and pool_calls + 1 > call_cap:
        return "pool_calls"
    if state["calls"] + state["open_calls"] + 1 > GLOBAL_CALLS:
        return "global_calls"
    if model == PRO and state["pro_calls"] + state["pro_open_calls"] + 1 > PRO_CALLS:
        return "pro_calls"
    pool_exposure = _dec(state["pools"][pool]["spent_usd"]) + _dec(state["pools"][pool]["reserved_usd"])
    if pool_exposure + amount > _dec(state["pools"][pool]["usd_cap"]):
        return "pool_usd"
    global_exposure = _dec(state["spent_usd"]) + _dec(state["reserved_usd"])
    if global_exposure + amount > GLOBAL_USD:
        return "global_usd"
    return None


class PilotLedger:
    """File ledger. One lock covers read, decide, and replace."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @classmethod
    def create(cls, path: Path) -> "PilotLedger":
        ledger = cls(path)
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        if ledger.path.exists():
            raise BudgetError("ledger already exists")
        ledger._replace(empty_state())
        return ledger

    def status(self) -> dict:
        with self._lock():
            return self._read()

    def open_meeting(self, meeting_id: str) -> dict:
        def mutate(state: dict) -> dict:
            if meeting_id in state["meetings"]:
                raise BudgetError("meeting already open")
            if len(state["meetings"]) >= MEETING_CAP:
                raise BudgetError("meeting cap")
            state["meetings"].append(meeting_id)
            state["events"].append({"kind": "meeting", "meeting_id": meeting_id})
            return {"meetings": len(state["meetings"])}

        return self._mutate(mutate)

    def enable_mst(self, reason: str) -> None:
        if reason != "aim_failed_frozen_pilot_gate":
            raise BudgetError("MST reason is not the frozen failure condition")

        def mutate(state: dict) -> None:
            state["mst_enabled"] = True
            state["events"].append({"kind": "enable_mst", "reason": reason})
            return None

        self._mutate(mutate)

    def reallocate(self, source: str, destination: str, usd: str, reason: str) -> None:
        amount = _dec(usd)
        if amount <= 0 or source not in POOLS or destination not in POOLS or source == destination:
            raise BudgetError("reallocation is not valid")
        if not reason or len(reason) > 80:
            raise BudgetError("reallocation reason is missing")

        def mutate(state: dict) -> None:
            source_cap = _dec(state["pools"][source]["usd_cap"])
            source_exposure = _dec(state["pools"][source]["spent_usd"]) + _dec(state["pools"][source]["reserved_usd"])
            if source_cap - amount < source_exposure:
                raise BudgetError("reallocation would pass spent funds")
            state["pools"][source]["usd_cap"] = _money(source_cap - amount)
            state["pools"][destination]["usd_cap"] = _money(_dec(state["pools"][destination]["usd_cap"]) + amount)
            state["events"].append({
                "kind": "reallocate",
                "source": source,
                "destination": destination,
                "usd": _money(amount),
                "reason": reason,
            })
            return None

        self._mutate(mutate)

    def reserve(self, pool: str, model: str, input_tokens: int, output_tokens: int, purpose: str) -> dict:
        if not purpose or len(purpose) > 80 or any(ch.isspace() for ch in purpose):
            raise BudgetError("purpose must be a short token")

        def mutate(state: dict) -> dict:
            reason = reservation_block(state, pool, model, input_tokens, output_tokens)
            if reason is not None:
                raise BudgetError(reason)
            amount = worst_case_usd(model, input_tokens, output_tokens)
            reservation_id = uuid.uuid4().hex
            state["open"][reservation_id] = {
                "pool": pool,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reserved_usd": _money(amount),
                "purpose": purpose,
            }
            state["pools"][pool]["reserved_usd"] = _money(_dec(state["pools"][pool]["reserved_usd"]) + amount)
            state["pools"][pool]["open_calls"] += 1
            state["reserved_usd"] = _money(_dec(state["reserved_usd"]) + amount)
            state["open_calls"] += 1
            if model == PRO:
                state["pro_open_calls"] += 1
            state["events"].append({
                "kind": "reserve",
                "reservation_id": reservation_id,
                "pool": pool,
                "model": model,
                "reserved_usd": _money(amount),
                "purpose": purpose,
            })
            return {"reservation_id": reservation_id, "reserved_usd": _money(amount)}

        return self._mutate(mutate)

    def release(self, reservation_id: str) -> None:
        def mutate(state: dict) -> None:
            held = state["open"].get(reservation_id)
            if held is None:
                raise BudgetError("unknown reservation")
            amount = _dec(held["reserved_usd"])
            pool = held["pool"]
            del state["open"][reservation_id]
            state["pools"][pool]["reserved_usd"] = _money(_dec(state["pools"][pool]["reserved_usd"]) - amount)
            state["pools"][pool]["open_calls"] -= 1
            state["reserved_usd"] = _money(_dec(state["reserved_usd"]) - amount)
            state["open_calls"] -= 1
            if held["model"] == PRO:
                state["pro_open_calls"] -= 1
            state["events"].append({"kind": "release", "reservation_id": reservation_id})
            return None

        self._mutate(mutate)

    def commit(
        self,
        reservation_id: str,
        *,
        usage_known: bool,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> dict:
        def mutate(state: dict) -> dict:
            held = state["open"].get(reservation_id)
            if held is None:
                raise BudgetError("unknown reservation")
            reserved = _dec(held["reserved_usd"])
            if usage_known:
                if prompt_tokens is None or completion_tokens is None:
                    raise BudgetError("known usage needs token counts")
                spent = worst_case_usd(held["model"], prompt_tokens, completion_tokens)
            else:
                spent = reserved
            pool = held["pool"]
            del state["open"][reservation_id]
            state["pools"][pool]["reserved_usd"] = _money(_dec(state["pools"][pool]["reserved_usd"]) - reserved)
            state["pools"][pool]["spent_usd"] = _money(_dec(state["pools"][pool]["spent_usd"]) + spent)
            state["pools"][pool]["open_calls"] -= 1
            state["pools"][pool]["calls"] += 1
            state["reserved_usd"] = _money(_dec(state["reserved_usd"]) - reserved)
            state["spent_usd"] = _money(_dec(state["spent_usd"]) + spent)
            state["open_calls"] -= 1
            state["calls"] += 1
            if held["model"] == PRO:
                state["pro_open_calls"] -= 1
                state["pro_calls"] += 1
            state["events"].append({
                "kind": "commit",
                "reservation_id": reservation_id,
                "spent_usd": _money(spent),
                "usage_known": usage_known,
                "overage": spent > reserved,
            })
            return {"spent_usd": _money(spent), "overage": spent > reserved}

        return self._mutate(mutate)

    def _mutate(self, fn):
        with self._lock():
            draft = copy.deepcopy(self._read())
            result = fn(draft)
            self._replace(draft)
            return result

    def _lock(self):
        thread_lock = _thread_lock_for(self.lock_path)
        thread_lock.acquire()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            handle.close()
            thread_lock.release()
            raise
        return _Unlock(handle, thread_lock)

    def _read(self) -> dict:
        if not self.path.exists():
            raise BudgetError("ledger is missing")
        state = json.loads(self.path.read_text(encoding="utf-8"))
        _check_stamp(state)
        return state

    def _replace(self, state: dict) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


class _Unlock:
    def __init__(self, handle, thread_lock):
        self.handle = handle
        self.thread_lock = thread_lock

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc, tb):
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        finally:
            self.thread_lock.release()
