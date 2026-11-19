"""Atomic spend ledger for the 2026-10-10 overnight round.

The ledger reserves the peak cache-miss price before a request and counts an
HTTP attempt when the request is committed. It does not call a network and it
does not read data rows. Historical spend from earlier runs is not stored here.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import uuid
from decimal import Decimal
from pathlib import Path


class BudgetError(ValueError):
    """A reserve does not fit, or a ledger transition is illegal."""


FLASH = "deepseek-flash"
PRO = "deepseek-v4-pro"
PRICES = {
    FLASH: (Decimal("0.30"), Decimal("1.20")),
    PRO: (Decimal("1.32"), Decimal("3.96")),
}
API_MAX_OUTPUT_TOKENS = 393216
GLOBAL_USD = Decimal("500")
GLOBAL_CALLS = 2544
CONCURRENCY = 4
POOLS = {
    "orchestration": {
        "usd_cap": Decimal("70"),
        "call_cap": None,
        "meeting_cap": 6,
        "models": {FLASH, PRO},
        "input_cap": 32768,
        "output_cap": 32768,
    },
    "teacher_diagnosis": {
        "usd_cap": Decimal("50"),
        "call_cap": 256,
        "meeting_cap": None,
        "models": {FLASH},
        "input_cap": 8192,
        "output_cap": 131072,
    },
    "candidate_a": {
        "usd_cap": Decimal("170"),
        "call_cap": 1024,
        "meeting_cap": None,
        "models": {FLASH},
        "input_cap": 8192,
        "output_cap": 131072,
    },
    "candidate_b": {
        "usd_cap": Decimal("170"),
        "call_cap": 1024,
        "meeting_cap": None,
        "models": {FLASH},
        "input_cap": 8192,
        "output_cap": 131072,
    },
    "transport_reserve": {
        "usd_cap": Decimal("40"),
        "call_cap": 240,
        "meeting_cap": None,
        "models": {FLASH, PRO},
        "input_cap": 32768,
        "output_cap": 131072,
    },
}


_QUANT = Decimal("0.000000000001")


def _money(value: Decimal) -> str:
    quantized = value.quantize(_QUANT)
    if quantized == 0:
        return "0"
    return format(quantized, "f").rstrip("0").rstrip(".")


def _dec(value: str) -> Decimal:
    return Decimal(value)


def worst_case_usd(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Peak cache-miss bill. Cache hits and off-peak prices are not used."""
    if model not in PRICES:
        raise BudgetError("unknown model")
    if input_tokens < 0 or output_tokens < 0:
        raise BudgetError("token counts must be non-negative")
    if output_tokens > API_MAX_OUTPUT_TOKENS:
        raise BudgetError("output cap exceeds the documented API maximum")
    miss, output = PRICES[model]
    million = Decimal(1_000_000)
    prompt = Decimal(input_tokens) / million
    generated = Decimal(output_tokens) / million
    return prompt * miss + generated * output


def empty_state() -> dict:
    pools = {}
    for name in POOLS:
        pools[name] = {
            "spent_usd": "0",
            "reserved_usd": "0",
            "calls": 0,
            "open_calls": 0,
        }
    return {
        "round_id": "2026-10-10-overnight-083951",
        "historical_spend_included": False,
        "price_stamp": {
            "flash_miss": "0.30",
            "flash_out": "1.20",
            "pro_miss": "1.32",
            "pro_out": "3.96",
        },
        "spent_usd": "0",
        "reserved_usd": "0",
        "calls": 0,
        "open_calls": 0,
        "meetings": [],
        "candidate_b_enabled": False,
        "pools": pools,
        "open": {},
        "events": [],
    }


def _check_stamp(state: dict) -> None:
    expected = empty_state()["price_stamp"]
    if state.get("price_stamp") != expected or state.get("historical_spend_included") is not False:
        raise BudgetError("ledger price stamp does not match this round")


def reservation_block(
    state: dict,
    pool: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> str | None:
    """Return a reason when the reserve must not be sent. None means it fits."""
    if pool not in POOLS:
        return "unknown_pool"
    spec = POOLS[pool]
    if pool == "candidate_b" and not state.get("candidate_b_enabled"):
        return "candidate_b_not_enabled"
    if model not in spec["models"]:
        return "model_not_in_pool"
    if input_tokens > spec["input_cap"] or output_tokens > spec["output_cap"]:
        return "token_cap"
    if pool == "transport_reserve" and model == PRO and output_tokens > 32768:
        return "token_cap"
    if pool == "transport_reserve" and model == FLASH and input_tokens > 8192:
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
    pool_exposure = _dec(state["pools"][pool]["spent_usd"]) + _dec(state["pools"][pool]["reserved_usd"])
    if pool_exposure + amount > spec["usd_cap"]:
        return "pool_usd"
    global_exposure = _dec(state["spent_usd"]) + _dec(state["reserved_usd"])
    if global_exposure + amount > GLOBAL_USD:
        return "global_usd"
    return None


class Ledger:
    """File ledger. One lock covers read, decide, and replace."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @classmethod
    def create(cls, path: Path) -> "Ledger":
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
            if len(state["meetings"]) >= POOLS["orchestration"]["meeting_cap"]:
                raise BudgetError("meeting cap")
            state["meetings"].append(meeting_id)
            state["events"].append({"kind": "meeting", "meeting_id": meeting_id})
            return {"meetings": len(state["meetings"])}

        return self._mutate(mutate)

    def enable_candidate_b(self, reason: str) -> None:
        if reason not in {"development_failure", "fidelity_or_privacy_unmet", "council_c1"}:
            raise BudgetError("candidate B reason is not an authorized condition")

        def mutate(state: dict) -> None:
            state["candidate_b_enabled"] = True
            state["events"].append({"kind": "enable_candidate_b", "reason": reason})
            return None

        self._mutate(mutate)

    def reserve(
        self,
        pool: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str,
    ) -> dict:
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
            state["pools"][pool]["reserved_usd"] = _money(
                _dec(state["pools"][pool]["reserved_usd"]) + amount
            )
            state["pools"][pool]["open_calls"] += 1
            state["reserved_usd"] = _money(_dec(state["reserved_usd"]) + amount)
            state["open_calls"] += 1
            state["events"].append({
                "kind": "reserve",
                "reservation_id": reservation_id,
                "pool": pool,
                "model": model,
                "reserved_usd": _money(amount),
                "purpose": purpose,
            })
            return {
                "reservation_id": reservation_id,
                "reserved_usd": _money(amount),
            }

        return self._mutate(mutate)

    def release(self, reservation_id: str) -> None:
        """Drop a reserve that never became an HTTP attempt."""

        def mutate(state: dict) -> None:
            held = state["open"].get(reservation_id)
            if held is None:
                raise BudgetError("unknown reservation")
            amount = _dec(held["reserved_usd"])
            pool = held["pool"]
            del state["open"][reservation_id]
            state["pools"][pool]["reserved_usd"] = _money(
                _dec(state["pools"][pool]["reserved_usd"]) - amount
            )
            state["pools"][pool]["open_calls"] -= 1
            state["reserved_usd"] = _money(_dec(state["reserved_usd"]) - amount)
            state["open_calls"] -= 1
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
        """Count one HTTP attempt. Unknown usage keeps the full reserve."""

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
            overage = spent > reserved
            pool = held["pool"]
            del state["open"][reservation_id]
            state["pools"][pool]["reserved_usd"] = _money(
                _dec(state["pools"][pool]["reserved_usd"]) - reserved
            )
            state["pools"][pool]["spent_usd"] = _money(
                _dec(state["pools"][pool]["spent_usd"]) + spent
            )
            state["pools"][pool]["open_calls"] -= 1
            state["pools"][pool]["calls"] += 1
            state["reserved_usd"] = _money(_dec(state["reserved_usd"]) - reserved)
            state["spent_usd"] = _money(_dec(state["spent_usd"]) + spent)
            state["open_calls"] -= 1
            state["calls"] += 1
            state["events"].append({
                "kind": "commit",
                "reservation_id": reservation_id,
                "spent_usd": _money(spent),
                "usage_known": usage_known,
                "overage": overage,
            })
            return {"spent_usd": _money(spent), "overage": overage}

        return self._mutate(mutate)

    def _mutate(self, fn):
        with self._lock():
            state = self._read()
            draft = copy.deepcopy(state)
            result = fn(draft)
            self._replace(draft)
            return result

    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _Unlock(handle)

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
    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
