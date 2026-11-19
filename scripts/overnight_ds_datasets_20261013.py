#!/usr/bin/env python3
"""Authorized DeepSeek Flash diagnostic on frozen public FinQA, TAT-QA, and BIRD items.

This runner never sends gold answers, FinQA programs, TAT-QA derivations, or BIRD gold SQL.
It resumes from its own ledger and does not write historical experiment ledgers.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/home/kiryu/privacy-synthetic-distillation")
WORK = Path("/mnt/c/Users/10950/Documents/Codex/2026-10-12/h/work")
RAW = ROOT / "data/raw/candidate-qualification"
RUN = ROOT / "runs/2026-10-13-overnight-ds-datasets"
REPORT = ROOT / "reports/overnight-ds-datasets-2026-10-13.md"
REPORT_COPY = Path("/mnt/c/Users/10950/Documents/Codex/2026-10-12/h/outputs/overnight-ds-datasets-2026-10-13.md")
SECRET = Path.home() / ".secrets" / "deepseek.env"
API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
MAX_OUTPUT = 32768
MAX_INPUT = 65536
MAX_CALLS = 900
MAX_QUESTIONS = 600
TARGET = 200
MINIMUM = 100
EXTRA_TRIES = 2
FX_FALLBACK = Decimal("6.70582")
FX_FALLBACK_ASOF = "2026-10-12T23:57:31Z open.er-api.com"
CNY_CAP = Decimal("450.00")  # 500 CNY authorization with a 10% buffer
PEAK_MISS = Decimal("0.30")
PEAK_HIT = Decimal("0.006")
PEAK_OUT = Decimal("1.20")
SGT = ZoneInfo("Asia/Singapore")
SEED = 20261013
_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")

FINQA_MOD = WORK / "table_reasoning/qualification-prep/candidate_finqa.py"
TATQA_MOD = WORK / "spatial_planning/qualification-prep/candidate_tatqa.py"
BIRD_MOD = WORK / "relational_sql/qualification-prep/candidate_bird.py"
FINQA_PACKET = WORK / "table_reasoning/qualification-prep/packet-v2"
TATQA_FROZEN = WORK / "spatial_planning/qualification-prep/frozen"
BIRD_FROZEN = WORK / "relational_sql/qualification-prep/frozen-dev"


def redact(text: str) -> str:
    return _SECRET.sub("sk-REDACTED", text)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_module(name: str, path: Path):
    # Spawned BIRD workers re-import the function by module name. The name has to
    # match the file, and the directory has to be on the child interpreter's path.
    directory = str(path.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if directory not in parts:
        os.environ["PYTHONPATH"] = directory if not current else directory + os.pathsep + current
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_env() -> str:
    for raw in SECRET.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and not os.environ.get(key, "").strip():
            os.environ[key] = value
    token = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not token:
        raise SystemExit("DEEPSEEK_API_KEY empty")
    return token


def now_sgt() -> datetime:
    return datetime.now(SGT)


def deadline_for(start: datetime) -> datetime:
    candidate = start.replace(hour=1, minute=0, second=0, microsecond=0)
    if candidate <= start:
        candidate += timedelta(days=1)
    return candidate


def is_peak(moment: datetime) -> bool:
    utc = moment.astimezone(timezone.utc)
    if utc.weekday() >= 5:
        return False
    minutes = utc.hour * 60 + utc.minute
    return (60 <= minutes < 240) or (360 <= minutes < 600)


def money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))


def peak_usd(prompt: int, completion: int, hit: int = 0, miss: int = 0) -> Decimal:
    prompt, completion, hit, miss = (int(prompt), int(completion), int(hit), int(miss))
    if hit + miss == 0:
        miss = prompt
    elif prompt > hit + miss:
        miss += prompt - (hit + miss)
    million = Decimal(1_000_000)
    return (Decimal(hit) * PEAK_HIT + Decimal(miss) * PEAK_MISS + Decimal(completion) * PEAK_OUT) / million


def billed_usd(prompt: int, completion: int, hit: int, miss: int, moment: datetime, usage_known: bool, reserved: Decimal) -> Decimal:
    if not usage_known:
        return reserved
    cost = peak_usd(prompt, completion, hit, miss)
    return cost if is_peak(moment) else cost / 2


def wilson(correct: int, n: int) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    z = 1.959963984540054
    p = correct / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, center - half), min(1.0, center + half)


def hamilton(counts: dict[str, int], size: int) -> dict[str, int]:
    total = sum(counts.values())
    allocation = {key: counts[key] * size // total for key in sorted(counts)}
    remainder = size - sum(allocation.values())
    ranked = sorted(counts, key=lambda key: (-(counts[key] * size % total), key))
    for key in ranked[:remainder]:
        allocation[key] += 1
    return allocation


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Ledger:
    def __init__(self) -> None:
        self.path = RUN / "ledger.jsonl"
        self.lock = threading.Lock()
        self.rows = []
        if self.path.exists():
            self.rows = read_jsonl(self.path)
        self.fx = Decimal(str(self._fx()))
        settled = Decimal("0")
        calls = 0
        for row in self.rows:
            if row.get("event") == "attempt":
                calls += 1
                settled += Decimal(str(row.get("settled_cny") or "0"))
        self.settled = settled
        self.reserved = Decimal("0")
        self.calls = calls
        self.inflight_ids: set[str] = set()
        self.blocked = next((row.get("error") for row in self.rows if row.get("event") == "blocked"), "")

    def _fx(self) -> str:
        fx_path = RUN / "fx.json"
        if fx_path.exists():
            return str(json.loads(fx_path.read_text(encoding="utf-8"))["usd_cny"])
        rate = FX_FALLBACK
        source = "fallback " + FX_FALLBACK_ASOF
        try:
            import httpx
            response = httpx.get("https://open.er-api.com/v6/latest/USD", timeout=20.0, trust_env=True)
            payload = response.json()
            fetched = Decimal(str(payload["rates"]["CNY"]))
            if fetched > 0:
                rate = fetched
                source = "open.er-api.com " + str(payload.get("time_last_update_utc"))
        except Exception as exc:  # noqa: BLE001
            source += " fetch_failed " + type(exc).__name__
        write_json(fx_path, {"usd_cny": str(rate), "source": source, "buffer": "10 percent; circuit at 450 CNY of the 500 CNY cap"})
        return str(rate)

    def done_ids(self) -> set[str]:
        done = set()
        attempts = Counter()
        terminal = set()
        for row in self.rows:
            if row.get("event") != "attempt":
                continue
            ident = row["id"]
            attempts[ident] += 1
            if row.get("terminal"):
                terminal.add(ident)
        for ident, count in attempts.items():
            if ident in terminal or count >= 1 + EXTRA_TRIES:
                # A nonterminal last attempt that exhausted retries is also finished.
                last = [row for row in self.rows if row.get("event") == "attempt" and row["id"] == ident][-1]
                if last.get("terminal") or attempts[ident] >= 1 + EXTRA_TRIES:
                    done.add(ident)
        return done

    def requested_ids(self) -> set[str]:
        return {row["id"] for row in self.rows if row.get("event") == "attempt"}

    def try_reserve(self, ident: str, usd: Decimal) -> bool:
        cny = usd * self.fx
        with self.lock:
            if self.blocked:
                return False
            if self.calls >= MAX_CALLS:
                return False
            known = self.requested_ids() | self.inflight_ids
            if ident not in known and len(known) >= MAX_QUESTIONS:
                return False
            if self.settled + self.reserved + cny > CNY_CAP:
                return False
            self.reserved += cny
            self.calls += 1
            self.inflight_ids.add(ident)
            return True

    def release_unused_reserve(self, ident: str, usd: Decimal) -> None:
        with self.lock:
            self.reserved -= usd * self.fx
            self.calls -= 1
            self.inflight_ids.discard(ident)
            if self.reserved < 0:
                self.reserved = Decimal("0")
            if self.calls < 0:
                self.calls = 0

    def settle(self, reserved_usd: Decimal, actual_usd: Decimal, row: dict) -> None:
        cny = actual_usd * self.fx
        with self.lock:
            self.reserved -= reserved_usd * self.fx
            if self.reserved < 0:
                self.reserved = Decimal("0")
            self.settled += cny
            self.inflight_ids.discard(row.get("id"))
            row["settled_cny"] = money(cny)
            row["settled_usd"] = money(actual_usd)
            row["reserved_usd"] = money(reserved_usd)
            row["running_settled_cny"] = money(self.settled)
            row["running_calls"] = self.calls
            self.rows.append(row)
            append_jsonl(self.path, row)

    def block(self, reason: str) -> None:
        with self.lock:
            self.blocked = reason
            row = {"event": "blocked", "error": redact(reason), "at": now_sgt().isoformat()}
            self.rows.append(row)
            append_jsonl(self.path, row)


LEDGER = None


def estimate_tokens(prompt: str) -> int:
    return max(1, len(prompt) // 2)


def reserve_usd(prompt: str) -> Decimal:
    return peak_usd(min(MAX_INPUT, estimate_tokens(prompt)), MAX_OUTPUT)


PROTOCOL = {"mode": "flat", "locked": False}


def request_body(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT,
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    if PROTOCOL["mode"] == "nested":
        body["thinking"] = {"type": "enabled", "reasoning_effort": "high"}
    return body


def call_api(token: str, prompt: str) -> dict:
    import httpx
    body = request_body(prompt)
    started = now_sgt()
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=30.0)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=True) as client:
            response = client.post(API, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=body)
        status = response.status_code
        request_id = response.headers.get("x-request-id") or response.headers.get("x-ds-request-id") or ""
        try:
            payload = response.json()
        except Exception:
            payload = {"raw": redact(response.text)[:1000]}
        if not isinstance(payload, dict):
            payload = {"raw": redact(str(payload))[:1000]}
        error = "" if status == 200 else redact(response.text)[:800]
    except Exception as exc:  # noqa: BLE001
        return {"started": started, "http_status": 0, "error": redact(f"{type(exc).__name__}: {exc}")[:800],
                "payload": {}, "request_id": "", "unknown": "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()}
    return {"started": started, "ended": now_sgt(), "http_status": status, "error": error, "payload": payload,
            "request_id": request_id or str(payload.get("id") or ""), "unknown": False,
            "protocol_mode": PROTOCOL["mode"]}


def technical(remote: dict) -> bool:
    status = int(remote.get("http_status") or 0)
    if status in {0, 408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if status == 400 and "reasoning_effort" in (remote.get("error") or ""):
        return True
    return False


def global_block(remote: dict) -> str:
    status = int(remote.get("http_status") or 0)
    text = (remote.get("error") or "").lower()
    if status in {401, 402, 403} or any(word in text for word in ("insufficient", "balance", "payment required", "invalid api key", "authentication")):
        return f"http {status}"
    return ""


def content_of(payload: dict) -> tuple[str, str, str | None, dict]:
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("content") or "", message.get("reasoning_content") or "", choice.get("finish_reason"), usage


def score_finqa(finqa, content: str, gold: dict, finish_reason: str | None) -> dict:
    truncated = finish_reason == "length"
    result = finqa.score_answer(content, gold, truncated=truncated)
    result["truncated"] = truncated
    return result


def score_tatqa(tatqa, metric_class, content: str, gold: dict, finish_reason: str | None) -> dict:
    truncated = finish_reason == "length"
    prediction, error = (None, "truncated") if truncated else tatqa.parse_final_answer(content)
    answer = prediction["answer"] if prediction else []
    scale = prediction["scale"] if prediction else ""
    one = metric_class()
    one(gold, answer, pred_scale=scale)
    em, f1, _, _ = one.get_overall_metric()
    return {"em": em, "f1": f1, "correct": em == 1, "parsed": prediction is not None,
            "strict_scale_correct": prediction is not None and scale == gold.get("scale"),
            "format_error": error, "truncated": truncated, "status": error or "scored"}


def score_bird(bird, content: str, gold: dict, finish_reason: str | None) -> dict:
    return bird.score_response(content, gold, finish_reason=finish_reason)


def save_response(dataset: str, ident: str, row: dict) -> None:
    path = RUN / "responses" / dataset / f"{ident}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_self_test(finqa, tatqa, bird) -> None:
    gold_rows = read_jsonl(FINQA_PACKET / "gold.local.jsonl")
    first = gold_rows[0]
    synthetic = json.dumps({"answer": json.loads(first["gold_target"]) if first["gold_target_unit"] != "boolean" else first["gold_target"], "unit": first["gold_target_unit"]})
    # gold_target may be a numeric string. JSON number is required by the parser.
    if first["gold_target_unit"] == "boolean":
        synthetic = json.dumps({"answer": first["gold_target"], "unit": "boolean"})
    else:
        synthetic = json.dumps({"answer": float(first["gold_target"]) if "." in str(first["gold_target"]) else int(first["gold_target"]), "unit": first["gold_target_unit"]})
    scored = finqa.score_answer(synthetic, first, truncated=False)
    if not scored["correct"]:
        raise RuntimeError("FinQA scorer rejected its own canonical gold")
    wrong_scale = json.dumps({"answer": float(first["gold_target"]) / 100 if first["gold_target_unit"] == "percent" else first["gold_target"], "unit": "number" if first["gold_target_unit"] == "percent" else "percent"})
    if first["gold_target_unit"] == "percent":
        wrong = finqa.score_answer(json.dumps({"answer": float(first["gold_target"]) / 100, "unit": "percent"}), first, truncated=False)
        if wrong["correct"]:
            raise RuntimeError("FinQA scorer accepted both percentage points and a divided scale")
    tat_gold = json.loads((TATQA_FROZEN / "gold.json").read_text(encoding="utf-8"))
    metric = tatqa.load_official_metric(RAW / "tatqa")
    one_id, one_gold = next(iter(tat_gold.items()))
    self_score = score_tatqa(tatqa, metric, json.dumps({"answer": one_gold["answer"], "scale": one_gold["scale"]}), one_gold, "stop")
    if self_score["em"] != 1:
        raise RuntimeError("TAT-QA official metric rejected its own gold")
    if bird.canonical(1) != bird.canonical(1.0):
        raise RuntimeError("BIRD canonical numeric equivalence failed for 1 and 1.0")
    bird_gold = json.loads((BIRD_FROZEN / "gold.json").read_text(encoding="utf-8"))
    sample_id, sample = next(iter(bird_gold.items()))
    bird_score = bird.score_response(json.dumps({"sql": sample["sql"]}), sample, finish_reason="stop")
    if not bird_score["official_ex"]:
        raise RuntimeError(f"BIRD gold SQL did not reproduce its stored result: {sample_id}")
    (RUN / "self-test.json").write_text(json.dumps({"finqa": scored, "tatqa_em": self_score["em"], "bird_ex": True, "bird_checked": sample_id}, indent=2) + "\n", encoding="utf-8")


def existing_items() -> dict[str, list[dict]]:
    finqa_prompts = read_jsonl(FINQA_PACKET / "prompts.jsonl")
    finqa_gold = {row["exam_id"]: row for row in read_jsonl(FINQA_PACKET / "gold.local.jsonl")}
    finqa_manifest = {row["exam_id"]: row for row in read_jsonl(FINQA_PACKET / "manifest.jsonl")}
    finqa = []
    for row in finqa_prompts:
        meta = finqa_manifest[row["exam_id"]]
        finqa.append({"id": row["exam_id"], "dataset": "finqa", "prompt": row["prompt"], "prompt_sha256": sha256_text(row["prompt"]),
                      "stratum": meta.get("step_stratum") or finqa_gold[row["exam_id"]].get("step_stratum"),
                      "source_id": finqa_gold[row["exam_id"]]["question_id"], "origin": "frozen-v2-32"})
    tat_exam = read_jsonl(TATQA_FROZEN / "exam.jsonl")
    tat_gold = json.loads((TATQA_FROZEN / "gold.json").read_text(encoding="utf-8"))
    tatqa = []
    for row in tat_exam:
        gold = tat_gold[row["id"]]
        tatqa.append({"id": row["id"], "dataset": "tatqa", "prompt": row["prompt"], "prompt_sha256": row["prompt_sha256"],
                      "stratum": gold["answer_type"] + "|" + gold["answer_from"], "answer_type": gold["answer_type"],
                      "answer_from": gold["answer_from"], "source_id": row["question_uid"], "origin": "frozen-32"})
    bird_inputs = read_jsonl(BIRD_FROZEN / "qualification.jsonl")
    bird_gold = json.loads((BIRD_FROZEN / "gold.json").read_text(encoding="utf-8"))
    bird = []
    for row in bird_inputs:
        gold = bird_gold[row["id"]]
        bird.append({"id": row["id"], "dataset": "bird", "prompt": row["prompt"], "prompt_sha256": sha256_text(row["prompt"]),
                     "stratum": gold["db_id"] + "|" + gold["difficulty"], "difficulty": gold["difficulty"],
                     "db_id": gold["db_id"], "source_id": row["source_question_id"], "origin": "frozen-dev-32"})
    return {"finqa": finqa, "tatqa": tatqa, "bird": bird}


def freeze_expansion(finqa, tatqa, bird) -> dict[str, list[dict]]:
    frozen_path = RUN / "frozen-items.json"
    if frozen_path.exists():
        return json.loads(frozen_path.read_text(encoding="utf-8"))
    base = existing_items()
    # FinQA: one new eligible question per unused annual report, Hamilton on the eligible pool.
    rows = finqa.load_train(RAW / "finqa" / "train.json")
    used_reports = set()
    used_questions = set()
    for item in base["finqa"]:
        used_questions.add(item["source_id"])
    gold_rows = {row["question_id"]: row for row in read_jsonl(FINQA_PACKET / "gold.local.jsonl")}
    for row in gold_rows.values():
        used_reports.add(row["report_id"])
    eligible = []
    ineligible = Counter()
    for row in rows:
        label = finqa.canonical_gold(row["qa"])
        if label["source_gold_status"] != "SCORABLE":
            ineligible[label["source_scale_convention"]] += 1
            continue
        eligible.append((row, label))
    pop = Counter(finqa.stratum(row) for row, _ in eligible)
    already = Counter(item["stratum"] for item in base["finqa"])
    quotas = hamilton(dict(pop), TARGET)
    finqa_target_quotas = {key: quotas[key] for key in quotas}
    skipped_input = []
    finqa_extra_gold = []
    additional = {key: max(0, quotas[key] - already.get(key, 0)) for key in pop}
    extra = sum(already.values()) + sum(additional.values()) - TARGET
    while extra > 0:
        key = max(additional, key=lambda name: (additional[name], name))
        if additional[key] == 0:
            break
        additional[key] -= 1
        extra -= 1
    grouped = defaultdict(list)
    for row, label in eligible:
        report = finqa.report_id(row)
        if report in used_reports or row["id"] in used_questions:
            continue
        grouped[finqa.stratum(row)].append((finqa.stable_rank(SEED, finqa.stratum(row), report, row["id"]), row, label, report))
    for key in grouped:
        grouped[key].sort()
    seen_reports = set(used_reports)
    for key, count in additional.items():
        if count <= 0:
            continue
        taken = 0
        for _, row, label, report in grouped[key]:
            if report in seen_reports:
                continue
            prompt = finqa.render_prompt(row)
            if estimate_tokens(prompt) > MAX_INPUT:
                skipped_input.append({"dataset": "finqa", "source_id": row["id"], "characters": len(prompt)})
                continue
            ident = f"finqa-x-{len(base['finqa']):03d}"
            base["finqa"].append({"id": ident, "dataset": "finqa", "prompt": prompt, "prompt_sha256": sha256_text(prompt),
                                  "stratum": key, "source_id": row["id"], "report_id": report, "origin": "expanded-seed-20261013"})
            finqa_extra_gold.append({"exam_id": ident, **label, "question_id": row["id"], "report_id": report, "step_stratum": key,
                                     "program_step_count": finqa.step_count(row)})
            seen_reports.add(report)
            taken += 1
            if taken == count:
                break
    gold_out = RUN / "gold" / "finqa.jsonl"
    gold_out.parent.mkdir(parents=True, exist_ok=True)
    gold_rows = read_jsonl(FINQA_PACKET / "gold.local.jsonl") + finqa_extra_gold
    temporary_gold = gold_out.with_suffix(".jsonl.tmp")
    temporary_gold.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold_rows), encoding="utf-8")
    temporary_gold.replace(gold_out)
    # TAT-QA hierarchical quotas.
    contexts = json.loads((RAW / "tatqa" / "tatqa_dataset_train.json").read_text(encoding="utf-8"))
    used_contexts = {row.get("context_fingerprint") for row in read_jsonl(TATQA_FROZEN / "exam.jsonl")}
    used_uids = {item["source_id"] for item in base["tatqa"]}
    pools = defaultdict(list)
    type_counts = Counter()
    source_counts = defaultdict(Counter)
    for index, context in enumerate(contexts):
        fingerprint = tatqa.digest(tatqa.context_payload(context))
        for question in context["questions"]:
            stratum = question["answer_type"] + "|" + question["answer_from"]
            type_counts[question["answer_type"]] += 1
            source_counts[question["answer_type"]][question["answer_from"]] += 1
            pools[stratum].append((tatqa.digest([SEED, stratum, question["uid"], fingerprint]), index, context, question, fingerprint))
    type_quotas = hamilton(dict(type_counts), TARGET)
    stratum_quotas = {}
    for answer_type, quota in type_quotas.items():
        stratum_quotas.update({f"{answer_type}|{source}": value for source, value in hamilton(dict(source_counts[answer_type]), quota).items()})
    already = Counter(item["stratum"] for item in base["tatqa"])
    additional = {key: max(0, stratum_quotas.get(key, 0) - already.get(key, 0)) for key in stratum_quotas}
    extra = len(base["tatqa"]) + sum(additional.values()) - TARGET
    while extra > 0:
        key = max(additional, key=lambda name: (additional[name], name))
        if additional[key] == 0:
            break
        additional[key] -= 1
        extra -= 1
    tat_gold_path = RUN / "gold" / "tatqa.json"
    tat_gold = json.loads((TATQA_FROZEN / "gold.json").read_text(encoding="utf-8"))
    for key, count in additional.items():
        if count <= 0:
            continue
        taken = 0
        for _, index, context, question, fingerprint in sorted(pools[key]):
            if fingerprint in used_contexts or question["uid"] in used_uids:
                continue
            prompt = tatqa.render_prompt(context, question)
            if estimate_tokens(prompt) > MAX_INPUT:
                skipped_input.append({"dataset": "tatqa", "source_id": question["uid"], "characters": len(prompt)})
                continue
            ident = f"tatqa_x_{len(base['tatqa']):03d}"
            base["tatqa"].append({"id": ident, "dataset": "tatqa", "prompt": prompt, "prompt_sha256": sha256_text(prompt),
                                  "stratum": key, "answer_type": question["answer_type"], "answer_from": question["answer_from"],
                                  "source_id": question["uid"], "origin": "expanded-seed-20261013"})
            tat_gold[ident] = {"question_uid": question["uid"], "context_fingerprint": fingerprint,
                               "answer_type": question["answer_type"], "answer_from": question["answer_from"],
                               "answer": question["answer"], "scale": question["scale"]}
            used_contexts.add(fingerprint)
            used_uids.add(question["uid"])
            taken += 1
            if taken == count:
                break
    write_json(tat_gold_path, tat_gold)
    # BIRD within the four local databases.
    dev = json.loads((RAW / "bird" / "dev.json").read_text(encoding="utf-8"))
    bird_gold = json.loads((BIRD_FROZEN / "gold.json").read_text(encoding="utf-8"))
    db_paths = {}
    for entry in bird_gold.values():
        db_paths[entry["db_id"]] = entry["db_path"]
    descriptions = {}
    contexts = {}
    for db_id, db_path in db_paths.items():
        folder = Path(db_path).parent / "database_description"
        texts = {}
        for csv_path in sorted(folder.glob("*.csv")):
            raw = csv_path.read_bytes()
            try:
                texts[csv_path.name] = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                texts[csv_path.name] = raw.decode("cp1252")
        descriptions[db_id] = texts
        contexts[db_id] = bird.database_context(db_path, texts)
    used_q = {item["source_id"] for item in base["bird"]}
    population = [row for row in dev if row["db_id"] in db_paths]
    pop = Counter(row["db_id"] + "|" + row["difficulty"] for row in population)
    quotas = hamilton(dict(pop), TARGET)
    bird_target_quotas = {key: quotas[key] for key in quotas}
    already = Counter(item["stratum"] for item in base["bird"])
    additional = {key: max(0, quotas.get(key, 0) - already.get(key, 0)) for key in pop}
    extra = len(base["bird"]) + sum(additional.values()) - TARGET
    while extra > 0:
        key = max(additional, key=lambda name: (additional[name], name))
        if additional[key] == 0:
            break
        additional[key] -= 1
        extra -= 1
    grouped = defaultdict(list)
    for row in population:
        if row["question_id"] in used_q:
            continue
        stratum = row["db_id"] + "|" + row["difficulty"]
        grouped[stratum].append((hashlib.sha256(json.dumps([SEED, stratum, row["question_id"]]).encode()).hexdigest(), row))
    failures = []
    for key, count in additional.items():
        if count <= 0:
            continue
        taken = 0
        for _, row in sorted(grouped[key]):
            run = bird.execute_summary(db_paths[row["db_id"]], row["SQL"])
            if run["status"] != "ok":
                failures.append({"question_id": row["question_id"], "db_id": row["db_id"], "status": run["status"]})
                continue
            context = {**contexts[row["db_id"]], "question": row["question"], "expert_evidence": row["evidence"]}
            prompt = bird.SYSTEM + "\n\n" + json.dumps(context, ensure_ascii=False, indent=2)
            if estimate_tokens(prompt) > MAX_INPUT:
                skipped_input.append({"dataset": "bird", "source_id": row["question_id"], "characters": len(prompt)})
                continue
            ident = f"bird-x-{len(base['bird']):03d}"
            base["bird"].append({"id": ident, "dataset": "bird", "prompt": prompt, "prompt_sha256": sha256_text(prompt),
                                 "stratum": key, "difficulty": row["difficulty"], "db_id": row["db_id"],
                                 "source_id": row["question_id"], "origin": "expanded-seed-20261013"})
            bird_gold[ident] = {"db_id": row["db_id"], "db_path": db_paths[row["db_id"]], "sql": row["SQL"],
                                "source_question_id": row["question_id"], "difficulty": row["difficulty"],
                                "expected": run["summary"], "gold_execution_status": "ok"}
            taken += 1
            if taken == count:
                break
    write_json(RUN / "gold" / "bird.json", bird_gold)
    write_json(RUN / "bird-gold-skip.json", {"skipped_nonexecutable": failures, "count": len(failures)})
    # Drop prompts from the frozen public index? Keep them; gold is separate. The frozen file includes prompts because the runner needs them.
    # Gold is not inside items. Write items without gold.
    public = {name: [{k: v for k, v in item.items()} for item in rows] for name, rows in base.items()}
    write_json(frozen_path, public)
    write_json(RUN / "freeze-policy.json", {
        "seed": SEED,
        "finqa_eligible": len(eligible),
        "finqa_ineligible": dict(ineligible),
        "finqa_target_quotas": finqa_target_quotas,
        "tatqa_type_quotas": type_quotas,
        "tatqa_stratum_quotas": stratum_quotas,
        "bird_population": len(population),
        "bird_quotas": bird_target_quotas,
        "skipped_input_cap": skipped_input,
        "counts": {name: len(rows) for name, rows in base.items()},
        "rule": "Existing 32 are a fixed prefix. Additional items use seed 20261013, one FinQA report and one TAT-QA context, BIRD only inside the four local public-dev databases. Gold execution failures are skipped, not replaced after seeing model output.",
    })
    return public


def load_gold() -> dict[str, dict]:
    finqa = {row["exam_id"]: row for row in read_jsonl(RUN / "gold" / "finqa.jsonl")} if (RUN / "gold" / "finqa.jsonl").exists() else {row["exam_id"]: row for row in read_jsonl(FINQA_PACKET / "gold.local.jsonl")}
    tat_path = RUN / "gold" / "tatqa.json"
    tatqa = json.loads(tat_path.read_text(encoding="utf-8")) if tat_path.exists() else json.loads((TATQA_FROZEN / "gold.json").read_text(encoding="utf-8"))
    bird_path = RUN / "gold" / "bird.json"
    bird = json.loads(bird_path.read_text(encoding="utf-8")) if bird_path.exists() else json.loads((BIRD_FROZEN / "gold.json").read_text(encoding="utf-8"))
    return {"finqa": finqa, "tatqa": tatqa, "bird": bird}


def attempt_item(token: str, modules: dict, item: dict, gold: dict, attempt: int) -> dict:
    reserved = reserve_usd(item["prompt"])
    if not LEDGER.try_reserve(item["id"], reserved):
        return {"status": "not_sent", "reason": LEDGER.blocked or "budget_or_call_cap", "id": item["id"]}
    if estimate_tokens(item["prompt"]) > MAX_INPUT:
        LEDGER.release_unused_reserve(item["id"], reserved)
        return {"status": "not_sent", "reason": "input_over_cap", "id": item["id"]}
    state = {"settled": False}

    def mark() -> None:
        state["settled"] = True

    try:
        return _attempt_reserved(token, modules, item, gold, attempt, reserved, mark)
    finally:
        if not state["settled"]:
            LEDGER.release_unused_reserve(item["id"], reserved)


def _attempt_reserved(token: str, modules: dict, item: dict, gold: dict, attempt: int, reserved: Decimal, mark) -> dict:
    remote = call_api(token, item["prompt"])
    payload = remote.get("payload") or {}
    content, reasoning, finish, usage = content_of(payload)
    usage_known = bool(usage) and remote.get("http_status") == 200
    actual = billed_usd(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                        int(usage.get("prompt_cache_hit_tokens") or 0), int(usage.get("prompt_cache_miss_tokens") or 0),
                        remote["started"], usage_known, reserved)
    score = {"correct": False, "parsed": False, "status": "not_scored"}
    block = global_block(remote)
    adapted = False
    status = int(remote.get("http_status") or 0)
    if status == 200:
        PROTOCOL["locked"] = True
    elif status == 400 and "reasoning_effort" in (remote.get("error") or "") and not PROTOCOL["locked"] and PROTOCOL["mode"] == "flat":
        PROTOCOL["mode"] = "nested"
        adapted = True
        write_json(RUN / "protocol-adapt.json", {"mode": "nested", "at": now_sgt().isoformat(), "error": remote.get("error") or ""})
    elif status == 400 and any(word in (remote.get("error") or "").lower() for word in ("reasoning_effort", "unknown parameter", "invalid model", "thinking")):
        block = block or f"protocol http {status}"
    tech = (technical(remote) or bool(remote.get("unknown")) or adapted) and not block
    terminal = not tech
    if remote.get("http_status") == 200 and not block:
        try:
            if item["dataset"] == "finqa":
                score = score_finqa(modules["finqa"], content, gold, finish)
                score["correct"] = bool(score.get("correct"))
            elif item["dataset"] == "tatqa":
                score = score_tatqa(modules["tatqa"], modules["tatqa_metric"], content, gold, finish)
            else:
                score = score_bird(modules["bird"], content, gold, finish)
                score["correct"] = bool(score.get("official_ex"))
                score["parsed"] = bool(score.get("parse_ok"))
                score["status"] = "ex" if score["correct"] else ("truncated" if score.get("truncated") else "scored")
        except Exception as exc:  # noqa: BLE001
            score = {"correct": False, "parsed": False, "status": "score_error", "error": redact(str(exc))[:400]}
            terminal = True
    complete = remote.get("http_status") == 200 and finish not in {None, "length"} and not tech
    row = {
        "event": "attempt",
        "id": item["id"],
        "dataset": item["dataset"],
        "source_id": item.get("source_id"),
        "stratum": item.get("stratum"),
        "attempt": attempt,
        "started": remote["started"].isoformat(),
        "ended": (remote.get("ended") or now_sgt()).isoformat(),
        "http_status": remote.get("http_status"),
        "request_id": remote.get("request_id"),
        "response_id": payload.get("id"),
        "payload_model": payload.get("model"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "finish_reason": finish,
        "usage": usage,
        "technical_failure": tech,
        "result_unknown": bool(remote.get("unknown")),
        "complete_response": complete,
        "terminal": terminal,
        "score": {k: v for k, v in score.items() if k != "sql"},
        "error": remote.get("error") or "",
        "peak_window": is_peak(remote["started"]),
        "protocol_mode": remote.get("protocol_mode"),
    }
    LEDGER.settle(reserved, actual, row)
    mark()
    try:
        save_response(item["dataset"], item["id"], {
            "id": item["id"], "dataset": item["dataset"], "attempt": attempt, "prompt_sha256": item["prompt_sha256"],
            "content": content, "reasoning_content": reasoning, "finish_reason": finish, "usage": usage,
            "http_status": remote.get("http_status"), "request_id": remote.get("request_id"),
            "response_id": payload.get("id"), "payload_model": payload.get("model"),
            "score": row["score"], "error": row["error"],
        })
    except Exception:  # noqa: BLE001
        append_jsonl(RUN / "runner-errors.jsonl", {"id": item["id"], "error": redact(traceback.format_exc())[:1500]})
    if block:
        LEDGER.block(block + " " + (remote.get("error") or ""))
    return row


def write_status(start: datetime, stop_new: datetime, phase: str, note: str = "") -> None:
    counts = summarize()
    payload = {"phase": phase, "note": note, "started": start.isoformat(), "stop_new_requests": stop_new.isoformat(),
               "deadline": (stop_new + timedelta(minutes=15)).isoformat(), "updated": now_sgt().isoformat(),
               "calls": LEDGER.calls, "settled_cny": money(LEDGER.settled), "reserved_cny": money(LEDGER.reserved),
               "fx": str(LEDGER.fx), "blocked": LEDGER.blocked, "datasets": counts}
    write_json(RUN / "status.json", payload)
    report = render_report(start, stop_new, phase)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    REPORT_COPY.write_text(report, encoding="utf-8")
    try:
        refresh_progress(start, stop_new, phase)
    except Exception:  # noqa: BLE001
        append_jsonl(RUN / "runner-errors.jsonl", {"id": "progress", "error": redact(traceback.format_exc())[:1500]})


def summarize() -> dict:
    by_id = {}
    for row in LEDGER.rows:
        if row.get("event") == "attempt":
            by_id.setdefault(row["id"], []).append(row)
    out = {}
    for dataset in ("finqa", "tatqa", "bird"):
        rows = {ident: attempts for ident, attempts in by_id.items() if attempts[-1]["dataset"] == dataset}
        last = {ident: attempts[-1] for ident, attempts in rows.items()}
        complete = [row for row in last.values() if row.get("complete_response")]
        correct = [row for row in last.values() if row.get("score", {}).get("correct")]
        out[dataset] = {"requested": len(rows), "calls": sum(len(v) for v in rows.values()),
                        "complete": len(complete), "correct_among_requested": sum(1 for row in last.values() if row.get("score", {}).get("correct")),
                        "parsed": sum(1 for row in last.values() if row.get("score", {}).get("parsed") or row.get("score", {}).get("parse_ok")),
                        "truncated": sum(1 for row in last.values() if row.get("finish_reason") == "length"),
                        "technical": sum(1 for row in last.values() if row.get("technical_failure") and not row.get("complete_response")),
                        "correct_among_complete": sum(1 for row in complete if row.get("score", {}).get("correct"))}
    return out


def last_attempts() -> dict[str, dict]:
    grouped = {}
    for row in LEDGER.rows:
        if row.get("event") == "attempt":
            grouped.setdefault(row["id"], []).append(row)
    return {ident: attempts[-1] for ident, attempts in grouped.items()}


def _frozen_counts() -> dict[str, int | None]:
    path = RUN / "frozen-items.json"
    if not path.exists():
        return {"finqa": None, "tatqa": None, "bird": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: len(payload.get(name) or []) for name in ("finqa", "tatqa", "bird")}


def _usage_int(row: dict, key: str) -> int:
    usage = row.get("usage") or {}
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def render_report(start: datetime, stop_new: datetime, phase: str) -> str:
    try:
        return _render_report(start, stop_new, phase)
    except Exception:  # noqa: BLE001
        return "# Overnight DeepSeek dataset diagnostic\n\nReport render failed. The ledger and response files remain the record.\n"


def _render_report(start: datetime, stop_new: datetime, phase: str) -> str:
    counts = summarize()
    latest = last_attempts()
    frozen = _frozen_counts()
    attempts = [row for row in LEDGER.rows if row.get("event") == "attempt"]
    models = sorted({str(row.get("payload_model")) for row in attempts if row.get("payload_model")})
    prompt_tokens = sum(_usage_int(row, "prompt_tokens") for row in attempts)
    completion_tokens = sum(_usage_int(row, "completion_tokens") for row in attempts)
    lines = ["# Overnight DeepSeek dataset diagnostic", "",
             f"Started {start.isoformat()}. New requests stop at {stop_new.isoformat()}. Deadline {(stop_new + timedelta(minutes=15)).isoformat()}. Phase: {phase}.",
             "The goal file named 2026-10-13 08:00 Asia/Singapore. That clock had already passed. This run uses the later instruction: the next 01:00 Asia/Singapore after start.",
             f"Provider DeepSeek official API `https://api.deepseek.com/chat/completions`. Requested model id `{MODEL}`. Returned model: {', '.join(models) if models else 'no completed payload yet'}.",
             "Thinking enabled, `reasoning_effort` high, `max_tokens` 32768, `stream` false. No tools, no temperature. Gold, programs, derivations, and gold SQL stay local.",
             f"FX {LEDGER.fx} CNY per USD. Circuit {CNY_CAP} CNY, the 500 CNY cap minus a 10% buffer. Settled {money(LEDGER.settled)} CNY. In-flight reserve {money(LEDGER.reserved)} CNY. HTTP attempts counted {LEDGER.calls}. Prompt tokens {prompt_tokens}. Completion tokens {completion_tokens}.",
             "Reservation uses peak cache-miss rates, USD 0.30 / 1M input and USD 1.20 / 1M output. Known off-peak usage is billed at half. Missing usage keeps the peak reservation.",
             ""]
    if LEDGER.blocked:
        lines.append(f"BLOCKED reason: {LEDGER.blocked}")
        lines.append("")
    lines.append("## Measured results")
    lines.append("")
    lines.append("Full-denominator accuracy counts a technical failure as incorrect. Complete responses are finished HTTP 200 replies whose finish reason is not `length`. Truncation and format errors stay in the accuracy denominator and are not retried. Unrequested items are NOT_RUN.")
    lines.append("")
    lines.append("| dataset | frozen N | requested distinct | complete responses | correct / requested | correct / complete | coverage | main metric | Wilson 95% on requested | technical failures | truncated | calls | prompt tokens | completion tokens | settled CNY |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    metric_name = {"finqa": "numeric answer EM", "tatqa": "EM", "bird": "execution EX"}
    for dataset in ("finqa", "tatqa", "bird"):
        row = counts[dataset]
        n = row["requested"]
        k = row["correct_among_requested"]
        low, high = wilson(k, n)
        interval = "n/a" if low is None else f"{low:.3f}–{high:.3f}"
        coverage = "n/a" if n == 0 else f"{row['complete'] / n:.3f}"
        among = "n/a" if row["complete"] == 0 else f"{row['correct_among_complete']}/{row['complete']}"
        frozen_n = "not frozen" if frozen[dataset] is None else str(frozen[dataset])
        subset = [item for item in attempts if item.get("dataset") == dataset]
        settled = sum((Decimal(str(item.get("settled_cny") or "0")) for item in subset), Decimal("0"))
        lines.append(f"| {dataset} | {frozen_n} | {n} | {row['complete']} | {k}/{n} | {among} | {coverage} | {metric_name[dataset]} | {interval} | {row['technical']} | {row['truncated']} | {row['calls']} | {sum(_usage_int(item, 'prompt_tokens') for item in subset)} | {sum(_usage_int(item, 'completion_tokens') for item in subset)} | {money(settled)} |")
    lines.append("")
    tat_last = [row for row in latest.values() if row.get("dataset") == "tatqa"]
    if tat_last:
        f1_values = [float((row.get("score") or {}).get("f1") or 0) for row in tat_last]
        strict = sum(1 for row in tat_last if (row.get("score") or {}).get("strict_scale_correct"))
        lines.append(f"TAT-QA mean per-question F1 on requested items: {sum(f1_values) / len(f1_values):.4f}. Strict scale agreement: {strict}/{len(tat_last)}. Arithmetic and count F1 follow the pinned official metric. Official EM and strict scale are separate.")
        lines.append("")
    bird_last = [row for row in latest.values() if row.get("dataset") == "bird"]
    if bird_last:
        parsed = sum(1 for row in bird_last if (row.get("score") or {}).get("parse_ok"))
        executed = sum(1 for row in bird_last if (row.get("score") or {}).get("execution_ok"))
        lines.append(f"BIRD JSON parse ok {parsed}/{len(bird_last)}. Local read-only execution ok {executed}/{len(bird_last)}. EX is result-set agreement, not string match to gold SQL.")
        lines.append("")
    lines.append("## Strata")
    lines.append("")
    lines.append("Wilson intervals describe the scored items. They do not remove dependence inside an annual report, a TAT-QA context, or a BIRD database.")
    lines.append("")
    lines.append("| dataset | stratum | requested | correct | Wilson 95% |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    groups = {}
    for row in latest.values():
        groups.setdefault((row.get("dataset"), row.get("stratum")), []).append(row)
    for key in sorted(groups, key=lambda item: (str(item[0]), str(item[1]))):
        rows = groups[key]
        correct = sum(1 for row in rows if (row.get("score") or {}).get("correct"))
        low, high = wilson(correct, len(rows))
        interval = "n/a" if low is None else f"{low:.3f}–{high:.3f}"
        lines.append(f"| {key[0]} | {key[1]} | {len(rows)} | {correct} | {interval} |")
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    lines.append("Examples are the earliest correct and earliest incorrect finished response in each dataset. They do not replace the table. Snippets are the model content only.")
    lines.append("")
    for dataset in ("finqa", "tatqa", "bird"):
        rows = sorted((row for row in latest.values() if row.get("dataset") == dataset and row.get("complete_response")), key=lambda row: row.get("started") or "")
        shown = {"correct": 0, "incorrect": 0}
        for row in rows:
            kind = "correct" if (row.get("score") or {}).get("correct") else "incorrect"
            if shown[kind] >= 2:
                continue
            path = RUN / "responses" / dataset / f"{row['id']}.json"
            snippet = ""
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                snippet = (payload.get("content") or "").replace("\n", " ")[:180]
            lines.append(f"- {dataset} {row['id']} {kind} stratum `{row.get('stratum')}`: {snippet}")
            shown[kind] += 1
    lines.append("")
    lines.append("## NOT_RUN")
    lines.append("")
    lines.append("| dataset | status | reason |")
    lines.append("| --- | --- | --- |")
    lines.append("| SQA3D | NOT_RUN | not in the approved overnight package |")
    lines.append("| CityData / CityGeoQA | NOT_RUN | not in the approved overnight package |")
    lines.append("| InstructScene | NOT_RUN | not in the approved overnight package |")
    for dataset in ("finqa", "tatqa", "bird"):
        if counts[dataset]["requested"] == 0:
            lines.append(f"| {dataset} | NOT_RUN | no DeepSeek request was sent in this run |")
    lines.append("")
    lines.append("## Scope and credentials")
    lines.append("")
    lines.append("This is a screened public diagnostic, not an official FinQA, TAT-QA, or BIRD score. The public subset can overlap pretraining. FinQA uses the v2 eligible-train filter. TAT-QA one item per context is not one item per independent filing. BIRD uses four local development databases: superhero, student_club, toxicology, thrombosis_prediction. Prompts include schema, official descriptions, two sample rows, and official evidence. No cross-database generalization is claimed.")
    lines.append("")
    lines.append("Credential index: `~/.secrets/deepseek.env` is present and was read for `DEEPSEEK_API_KEY`. The key is not printed. No recharge and no model switch were performed.")
    lines.append("")
    lines.append(f"Per-item responses: `runs/2026-10-13-overnight-ds-datasets/responses/`. Ledger: `runs/2026-10-13-overnight-ds-datasets/ledger.jsonl`. This report: `reports/overnight-ds-datasets-2026-10-13.md`.")
    lines.append("")
    minimum_hits = []
    target_hits = []
    for dataset in ("finqa", "tatqa", "bird"):
        row = counts[dataset]
        if row["requested"] >= MINIMUM and row["complete"] >= 0.9 * row["requested"]:
            minimum_hits.append(dataset)
        if row["requested"] >= TARGET and row["complete"] >= 0.9 * row["requested"]:
            target_hits.append(dataset)
    lines.append(f"Minimum bar, two datasets at 100 requested with 90% complete: {len(minimum_hits) >= 2} ({', '.join(minimum_hits) if minimum_hits else 'none'}).")
    lines.append(f"Target bar, all three at 200 requested with 90% complete: {len(target_hits) == 3} ({', '.join(target_hits) if target_hits else 'none'}).")
    lines.append(f"Run label against those bars: {final_label(counts)}.")
    lines.append("")
    lines.append("High or low scores are not evidence about distillation. No student and no generator was run.")
    lines.append("")
    return "\n".join(lines) + "\n"


def final_label(counts: dict) -> str:
    def ok(dataset: str, minimum: int) -> bool:
        row = counts[dataset]
        return row["requested"] >= minimum and row["complete"] >= 0.9 * row["requested"]
    tested = [name for name in ("finqa", "tatqa", "bird") if counts[name]["requested"]]
    if all(ok(name, TARGET) for name in ("finqa", "tatqa", "bird")):
        return "COMPLETE"
    minimum_met = sum(ok(name, MINIMUM) for name in ("finqa", "tatqa", "bird")) >= 2
    if LEDGER.blocked and not minimum_met:
        return "BLOCKED"
    if minimum_met:
        return "PARTIAL"
    if not tested:
        return "BLOCKED" if LEDGER.blocked else "NOT_RUN"
    return "PARTIAL" if not LEDGER.blocked else "BLOCKED"


def refresh_progress(start: datetime, stop_new: datetime, phase: str) -> None:
    counts = summarize()
    label = final_label(counts)
    bits = []
    for dataset in ("finqa", "tatqa", "bird"):
        row = counts[dataset]
        bits.append(f"{dataset} {row['correct_among_requested']}/{row['requested']} correct, {row['complete']} complete, {row['calls']} calls")
    body = (
        f"Phase `{phase}`. Label `{label}`. Started {start.isoformat()}. "
        f"Stop new requests {stop_new.isoformat()}. Settled {money(LEDGER.settled)} CNY, reserve {money(LEDGER.reserved)} CNY, "
        f"HTTP count {LEDGER.calls}. " + ". ".join(bits) + ". "
        "Report: `reports/overnight-ds-datasets-2026-10-13.md`. "
        "Ledger: `runs/2026-10-13-overnight-ds-datasets/ledger.jsonl`. "
        "SQA3D, CityData, and InstructScene are NOT_RUN. "
        "Generator status stays NO_ACCEPTED_GENERATOR. This diagnostic does not change distillation."
    )
    if LEDGER.blocked:
        body += f" Blocked: {LEDGER.blocked}."
    heading = "### 2026-10-13 — Overnight DeepSeek FinQA, TAT-QA, and BIRD results"
    path = ROOT / "progress.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = f"\n{heading}\n\n{body}\n"
    if heading in text:
        start_at = text.index(heading)
        rest = text[start_at + len(heading):]
        nxt = rest.find("\n### ")
        if nxt == -1:
            text = text[:start_at].rstrip() + section
        else:
            text = text[:start_at].rstrip() + section + "\n" + rest[nxt + 1:]
    else:
        text = text.rstrip() + section
    path.write_text(text, encoding="utf-8")


def note_start(start: datetime, stop_new: datetime) -> None:
    heading = "### 2026-10-13 — Overnight DeepSeek diagnostic started"
    path = ROOT / "progress.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if heading in text:
        return
    body = (
        f"Approved package started {start.isoformat()}. New requests stop {stop_new.isoformat()}. "
        "DeepSeek official API, model id `deepseek-flash`, thinking enabled, reasoning_effort high. "
        "FinQA, TAT-QA, and BIRD public calibration items only. Circuit 450 CNY against a 500 CNY cap. "
        "The goal file's 08:00 deadline had passed; the run uses the next 01:00 Asia/Singapore. "
        "No result is claimed in this start note."
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{heading}\n\n{body}\n")


def _finish_notes(start: datetime, stop_new: datetime) -> None:
    errors = RUN / "runner-errors.jsonl"
    adapt = RUN / "protocol-adapt.json"
    if not (LEDGER.blocked or errors.exists() or adapt.exists()):
        return
    heading = "## 2026-10-13 — Overnight DeepSeek diagnostic failure"
    path = ROOT / "ProblemSolution.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if heading in text:
        return
    detail = []
    if LEDGER.blocked:
        detail.append(f"Blocked: {LEDGER.blocked}.")
    if adapt.exists():
        detail.append("A 400 mentioning reasoning_effort switched the unsent protocol from flat to nested before any locked success. See `protocol-adapt.json`.")
    if errors.exists():
        detail.append(f"Runner errors were appended to `{errors}`.")
    detail.append("No recharge and no model switch. Historical packets were not rewritten.")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{heading}\n\n" + " ".join(detail) + "\n")


def choose(items: dict[str, list[dict]], skip: set[str]) -> dict | None:
    pending = {name: [item for item in rows if item["id"] not in skip] for name, rows in items.items()}

    def count(name: str) -> int:
        return sum(1 for item in items[name] if item["id"] in skip)

    def take(name: str) -> dict | None:
        return pending[name][0] if pending[name] else None

    if (count("finqa") < MINIMUM or count("tatqa") < MINIMUM) and (pending["finqa"] or pending["tatqa"]):
        order = sorted(("finqa", "tatqa"), key=lambda name: (count(name), name))
        for name in order:
            item = take(name)
            if item:
                return item
    if count("bird") < MINIMUM:
        item = take("bird")
        if item:
            return item
    order = sorted(("finqa", "tatqa", "bird"), key=lambda name: (count(name), name))
    for name in order:
        if count(name) >= TARGET:
            continue
        item = take(name)
        if item:
            return item
    return None


def run() -> None:
    global LEDGER
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "responses").mkdir(exist_ok=True)
    start = now_sgt()
    stop_new = deadline_for(start) - timedelta(minutes=15)
    token = load_env()
    modules = {"finqa": load_module("candidate_finqa", FINQA_MOD),
               "tatqa": load_module("candidate_tatqa", TATQA_MOD),
               "bird": load_module("candidate_bird", BIRD_MOD)}
    write_json(RUN / "authorization.json", {
        "approved_in_user_message": "2026-10-13 overnight goal",
        "provider": "DeepSeek official API",
        "base_url": "https://api.deepseek.com",
        "model": MODEL,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "max_input_tokens": MAX_INPUT,
        "max_output_tokens": MAX_OUTPUT,
        "max_distinct_questions": MAX_QUESTIONS,
        "max_http_requests": MAX_CALLS,
        "cny_authorization": "500",
        "cny_circuit_with_10_percent_buffer": str(CNY_CAP),
        "concurrency": "2 then 4",
        "started": start.isoformat(),
        "stop_new_requests": stop_new.isoformat(),
        "deadline": deadline_for(start).isoformat(),
        "documented_prices_per_million_peak": {"cache_miss_input": "0.30", "cache_hit_input": "0.006", "output": "1.20"},
        "off_peak": "half of peak, UTC weekdays 01:00-04:00 and 06:00-10:00 are peak; reservation always uses peak",
        "price_source": "https://api-docs.deepseek.com/quick_start/pricing fetched 2026-10-13",
        "thinking_source": "https://api-docs.deepseek.com/guides/thinking_mode",
        "deadline_rule": "next 01:00 Asia/Singapore after start; goal-file 08:00 on 2026-10-13 had already passed",
        "historical_calls_approved_false_not_rewritten": True,
    })
    LEDGER = Ledger()
    note_start(start, stop_new)
    if not (RUN / "self-test.json").exists():
        local_self_test(modules["finqa"], modules["tatqa"], modules["bird"])
    modules["tatqa_metric"] = modules["tatqa"].load_official_metric(RAW / "tatqa")
    base = existing_items()
    gold = load_gold()
    freeze_box: dict = {}

    def _freeze() -> None:
        try:
            freeze_box["items"] = freeze_expansion(modules["finqa"], modules["tatqa"], modules["bird"])
        except Exception:  # noqa: BLE001
            freeze_box["error"] = traceback.format_exc()

    freezer = threading.Thread(target=_freeze, name="freeze-expansion", daemon=True)
    freezer.start()
    first = base["finqa"][0]
    prior = [row for row in LEDGER.rows if row.get("event") == "attempt" and row.get("id") == first["id"]]
    if not prior and first["id"] not in LEDGER.done_ids() and now_sgt() < stop_new:
        write_status(start, stop_new, "calling-first")
        row = attempt_item(token, modules, first, gold["finqa"][first["id"]], 1)
        print(f"FIRST_ATTEMPT {first['id']} http={row.get('http_status')} status={row.get('status') or row.get('score', {}).get('status')}", flush=True)
        write_status(start, stop_new, "first-attempt")
        if LEDGER.blocked:
            freezer.join()
            write_status(start, stop_new, final_label(summarize()), "blocked on first call")
            _finish_notes(start, stop_new)
            return
    freezer.join()
    if freeze_box.get("error"):
        append_jsonl(RUN / "runner-errors.jsonl", {"id": "freeze", "error": redact(freeze_box["error"])[:2000]})
        items = existing_items()
    else:
        items = freeze_box.get("items") or existing_items()
    gold = load_gold()
    write_status(start, stop_new, "running")
    slots = threading.Semaphore(2)
    futures = []
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    pool = ThreadPoolExecutor(max_workers=8)

    def wrap(item: dict, attempt: int, delay: float):
        if delay:
            time.sleep(delay)
        with slots:
            return attempt_item(token, modules, item, gold[item["dataset"]][item["id"]], attempt)

    claimed = set(LEDGER.requested_ids())
    by_id = {item["id"]: item for rows in items.values() for item in rows}
    retry_queue = []
    for ident in sorted(LEDGER.requested_ids() - LEDGER.done_ids()):
        if ident not in by_id:
            continue
        tries = sum(1 for row in LEDGER.rows if row.get("event") == "attempt" and row["id"] == ident)
        if tries <= EXTRA_TRIES:
            retry_queue.append((by_id[ident], tries + 1, min(30, 5 * tries)))
    limit = 2
    promoted = False
    hold_new = False
    last_status = time.monotonic()
    while True:
        now = now_sgt()
        still = []
        for entry, future in futures:
            if not future.done():
                still.append((entry, future))
                continue
            try:
                result = future.result()
            except Exception:  # noqa: BLE001
                append_jsonl(RUN / "runner-errors.jsonl", {"id": entry["id"], "error": redact(traceback.format_exc())[:1500]})
                continue
            if result.get("status") == "not_sent" and result.get("reason") != "input_over_cap":
                hold_new = True
                continue
            if result.get("technical_failure") and not result.get("terminal"):
                tries = sum(1 for row in LEDGER.rows if row.get("event") == "attempt" and row["id"] == entry["id"])
                if tries <= EXTRA_TRIES and now < stop_new and not LEDGER.blocked:
                    retry_queue.append((entry, tries + 1, min(30, 5 * tries)))
        futures = still
        if not promoted and sum(row["complete"] for row in summarize().values()) >= 8:
            slots.release()
            slots.release()
            limit = 4
            promoted = True
        running = len(futures)
        if now < stop_new and not LEDGER.blocked and not hold_new:
            while running < limit:
                if retry_queue:
                    item, attempt_no, delay = retry_queue.pop(0)
                    futures.append((item, pool.submit(wrap, item, attempt_no, delay)))
                    running += 1
                    continue
                item = choose(items, claimed)
                if item is None:
                    break
                claimed.add(item["id"])
                attempt_no = 1 + sum(1 for row in LEDGER.rows if row.get("event") == "attempt" and row["id"] == item["id"])
                futures.append((item, pool.submit(wrap, item, attempt_no, 0)))
                running += 1
        if time.monotonic() - last_status > 300:
            write_status(start, stop_new, "running")
            last_status = time.monotonic()
        if not futures and not retry_queue:
            break
        if not futures:
            break
        wait([future for _, future in futures], timeout=5, return_when=FIRST_COMPLETED)
    pool.shutdown(wait=True)
    label = final_label(summarize())
    write_status(start, stop_new, label, "runner finished")
    _finish_notes(start, stop_new)
    print(f"RUN_FINISHED {label}", flush=True)


if __name__ == "__main__":
    run()
