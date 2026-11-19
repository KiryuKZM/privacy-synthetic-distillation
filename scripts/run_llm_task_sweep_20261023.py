#!/usr/bin/env python3
"""Bounded task-native sweep. Preparation/check/review never read a credential.

Collection is disabled until an explicit approval binds the complete packet.
No historical executor, source asset or sealed final is modified by this file.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
POOL_LIMITS = {
    "MapEval": (128, 141, 4, 32768, 256, 8192),
    "USTBench": (140, 154, 4, 32768, 256, 8192),
    "SQA3D": (140, 154, 6, 65536, 256, 8192),
    "SGQA": (140, 154, 4, 32768, 256, 8192),
    "ScanQA": (140, 154, 6, 65536, 256, 8192),
    "TravelPlanner": (60, 66, 3, 32768, 4096, 16384),
    "ALFWorld": (1960, 2156, 23, 16384, 128, 4096),
    "ScienceWorld": (4000, 4400, 45, 16384, 128, 4096),
}
COUNTS = {k: (4 if k in ("ALFWorld", "ScienceWorld") else 8 if k == "TravelPlanner" else 16,
              48 if k == "MapEval" else 20 if k in ("ALFWorld", "ScienceWorld", "TravelPlanner") else 60)
          for k in POOL_LIMITS}
SUPPORTED_READY = {"MapEval", "USTBench", "SQA3D", "SGQA", "ScienceWorld"}
ARMS = ("native-off", "native-on-high", "adapted-off", "adapted-on-high")
REQUIRED_CODE = ("scripts/run_llm_task_sweep_20261023.py", "src/privacy_synth/pilot_ledger.py",
                 "src/privacy_synth/round_budget.py", "scripts/deepseek_ipd_interface.py")
SCIENCE_PYTHON = "artifacts/agent-worlds-20261023/venv/bin/python"
BUDGET_PATH = "runs/2026-10-23-llm-task-sweep/budget/llm-task-sweep.sqlite"


class SweepStop(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("wb") as target:
        target.write(encoded(value) + b"\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def once(path, value):
    if Path(path).exists():
        if load(path) != value:
            raise SweepStop("Immutable receipt differs: " + str(path))
    else:
        atomic(path, value)


def local_path(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
        raise PermissionError("Path leaves project")
    return path


def no_final(name):
    if any(re.search(r"final|sealed", part, re.I) for part in Path(name).parts):
        raise PermissionError("Official Final/Sealed content is ineligible")


@contextlib.contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def file_map(packet):
    files = packet.get("files", [])
    result = {r["path"]: r for r in files}
    if len(result) != len(files):
        raise PermissionError("Duplicate asset bindings")
    return result


def bound_path(packet, name, root=ROOT, read_gold=False):
    spec = file_map(packet).get(name)
    if spec is None:
        raise PermissionError("Unbound asset: " + name)
    path = local_path(root, name)
    if spec.get("kind") == "gold" and not read_gold:
        return path  # Bind its expected digest without opening gold during collection.
    if not path.is_file() or path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
        raise PermissionError("Asset/code hash differs: " + name)
    return path


def checked_interpreter(task, root=ROOT):
    spec = task.get("worker_interpreter", {})
    if spec.get("path") != SCIENCE_PYTHON:
        raise PermissionError("Only the explicitly configured project ScienceWorld interpreter is eligible")
    path = Path(root).resolve() / SCIENCE_PYTHON
    base = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    if path.resolve() != base or sha(path) != spec.get("resolved_sha256") or spec.get("version") != sys.version.split()[0]:
        raise PermissionError("ScienceWorld symlink/base-interpreter identity differs")
    return path


def check_packet(packet_path, root=ROOT, allow_stop=False):
    packet = load(packet_path)
    provider = packet.get("provider", {})
    if (packet.get("schema_version") != 1 or provider.get("name") != "DeepSeek official"
            or provider.get("endpoint") != "https://api.deepseek.com/chat/completions"
            or provider.get("model") != "deepseek-flash"
            or provider.get("model_revision") != "DeepSeek-V4.1-Flash-1017"
            or packet.get("modes") != ["off", "on-high"]):
        raise PermissionError("Provider/model/modes contract differs")
    if packet.get("execution_window_seconds") != 43200 or packet.get("start_epoch") is not None or packet.get("deadline_epoch") is not None:
        raise PermissionError("Static packet must bind a12h approval-relative window, without a premature start")
    if packet.get("budget_path") != BUDGET_PATH:
        raise PermissionError("Every package in this100USD route must share the canonical ledger")
    bounds = packet.get("bounds", {})
    expected = {"logical": 6708, "http": 7379, "global_usd": 100, "task_usd_sum": 95,
                "parent_usd": 500, "prior_conservative_usd": "2.397530706",
                "concurrency": 4, "no_progress_seconds": 1800, "oom_stop_count": 2}
    if any(bounds.get(k) != v for k, v in expected.items()):
        raise PermissionError("Global bounded workload changed")
    if not set(REQUIRED_CODE) <= set(file_map(packet)):
        raise PermissionError("Required executor/ledger/transport code is not bound")
    if any(file_map(packet)[name].get("kind") != "code" for name in REQUIRED_CODE):
        raise PermissionError("Required execution sources must be marked code")
    rates = provider.get("rates", {})
    if set(rates) != {"hit", "miss", "output"} or any(float(v) < 0 for v in rates.values()):
        raise PermissionError("Explicit verified price binding missing")
    for spec in packet["files"]:
        bound_path(packet, spec["path"], root)
    run = local_path(root, packet["run_dir"])
    if not allow_stop and ((run / "global-stop.json").exists() or (local_path(root, BUDGET_PATH).parent / "global-stop.json").exists()):
        raise PermissionError("Global stop is set")
    if not packet.get("tasks") or not set(packet["tasks"]) <= set(POOL_LIMITS):
        raise PermissionError("Unknown/empty task set")
    for name, task in packet["tasks"].items():
        if task.get("ready") is not True:
            continue
        if name not in SUPPORTED_READY:
            raise PermissionError("Task is conditional/blocked in this executor revision")
        if task.get("kind") not in ("qa", "agent"):
            raise PermissionError("Unsupported task kind")
        expected_kind = "agent" if name in ("ALFWorld", "ScienceWorld") else "qa"
        if task["kind"] != expected_kind:
            raise PermissionError("Task kind differs from fixed pool")
        expected_scorer = {"MapEval": "mapeval", "USTBench": "ust", "SQA3D": "sceneqa", "ScanQA": "sceneqa",
                           "SGQA": "sgqa", "TravelPlanner": "travel", "ALFWorld": "agent", "ScienceWorld": "agent"}[name]
        if task.get("scorer") != expected_scorer:
            raise PermissionError("Task-native scorer differs")
        scorer_code = "scripts/prepare_sceneqa_task_sweep_20261023.py" if expected_scorer == "sceneqa" else "scripts/build_llm_task_sweep_qa_20261023.py" if expected_scorer in ("mapeval", "ust", "sgqa") else None
        if scorer_code and file_map(packet).get(scorer_code, {}).get("kind") != "code":
            raise PermissionError("Native scorer source binding missing")
        if name in ("TravelPlanner", "ALFWorld"):
            raise PermissionError("Conditional blocked task has no validated execution adapter in this revision")
        for phase in ("dev", "screen"):
            spec = task["phases"][phase]
            no_final(spec["queries"])
            rows = read_rows(bound_path(packet, spec["queries"], root))
            if len(rows) != COUNTS[name][phase == "screen"] or len({str(r["id"]) for r in rows}) != len(rows):
                raise PermissionError("Exact frozen task membership/count differs")
            for row in rows:
                validate_row(row, task["kind"])
                if task["kind"] == "qa":
                    wanted_demos = 2 if name == "SQA3D" else 8
                    if sum(m["role"] == "assistant" for m in row["adapted_messages"]) != wanted_demos:
                        raise PermissionError("Adapted public Fit demonstration count differs")
                if name == "MapEval":
                    legal = row.get("metadata", {}).get("legal_options")
                    if not isinstance(legal, list) or len(legal) < 2 or any(type(x) is not int for x in legal) or legal != list(range(len(legal))):
                        raise PermissionError("MapEval numbered-option grammar missing")
                if name == "USTBench" and row.get("metadata", {}).get("ability") not in ("st_understanding", "planning", "reflection"):
                    raise PermissionError("Only the frozen UST process abilities are prepared")
            if task["kind"] == "qa":
                no_final(spec["gold"])
                if file_map(packet).get(spec["gold"], {}).get("kind") != "gold":
                    raise PermissionError("Gold must be explicitly labeled and bound")
        dev = read_rows(bound_path(packet, task["phases"]["dev"]["queries"], root))
        screen = read_rows(bound_path(packet, task["phases"]["screen"]["queries"], root))
        if {str(x["id"]) for x in dev} & {str(x["id"]) for x in screen}:
            raise PermissionError("Dev/Screen IDs overlap")
        if {x["group"] for x in dev} & {x["group"] for x in screen}:
            raise PermissionError("Dev/Screen worlds overlap")
        if task["kind"] == "agent":
            argv = task["worker_argv"]
            if name == "ScienceWorld" and (len(argv) != 3 or argv[1:] != ["scripts/agent_world_sweep_runtime_recovery_20261023.py", "scienceworld"]):
                raise PermissionError("Only the frozen native ScienceWorld JSONL worker is eligible")
            if argv[0] != "SELF_PYTHON":
                if argv[0] != SCIENCE_PYTHON:
                    raise PermissionError("Unapproved external interpreter")
                checked_interpreter(task, root)
            for path in argv[1:]:
                if "/" in path and not path.startswith("-"):
                    bound_path(packet, path, root)
            if name == "ScienceWorld":
                receipt = load(bound_path(packet, task["native_smoke_receipt"], root))
                if receipt.get("status") != "PASS" or not receipt.get("all_ok"):
                    raise PermissionError("ScienceWorld JSONL smoke is not finalized")
            manifest = load(bound_path(packet, task["episode_manifest"], root))
            episodes = {r["episode_id"]: r for r in manifest["episodes"]}
            for phase, rows in (("Dev", dev), ("Screen", screen)):
                for row in rows:
                    source = episodes.get(row["episode_id"], {})
                    if source.get("role") != phase or row["group"] != source.get("source_group") or row["task_name"] != source.get("task_name"):
                        raise PermissionError("Native episode role/group/task differs; Final/Fit ineligible")
    return packet


def approval_binding(packet, packet_path):
    return {"packet_sha256": sha(packet_path), "provider": packet["provider"]["name"],
            "model": packet["provider"]["model"], "model_revision": packet["provider"]["model_revision"],
            "modes": packet["modes"], "files_digest": digest(packet["files"]), "bounds": packet["bounds"],
            "execution_window_seconds": packet["execution_window_seconds"]}


def authorize(packet_path, approval_path, root=ROOT):
    packet = check_packet(packet_path, root)
    approval = load(approval_path)
    wanted = approval_binding(packet, packet_path)
    if (approval.get("approved") is not True or not approval.get("user_confirmation")
            or any(approval.get(k) != v for k, v in wanted.items())):
        raise PermissionError("Explicit approval does not bind exact workload/model/code/assets")
    if packet.get("repair_policy", {}).get("allow") and approval.get("allow_bounded_dev_instruction_repair") is not True:
        raise PermissionError("Adaptive instruction content was not explicitly approved")
    now = time.time()
    start, deadline = approval.get("start_epoch"), approval.get("deadline_epoch")
    if (type(start) not in (float, int) or type(deadline) not in (float, int)
            or deadline - start != packet["execution_window_seconds"] or not start <= now < deadline):
        raise PermissionError("Approved execution window is not active")
    run = local_path(root, packet["run_dir"])
    receipt_path = run / "start-receipt.json"
    if receipt_path.exists() and load(receipt_path) != start_binding(packet_path, approval_path, approval):
        raise PermissionError("Started run approval/window cannot be reset")
    return packet


def start_binding(packet_path, approval_path, approval):
    return {"packet_sha256": sha(packet_path), "approval_sha256": sha(approval_path),
            "start_epoch": approval["start_epoch"], "deadline_epoch": approval["deadline_epoch"]}


def start_run(packet, packet_path, approval_path, root=ROOT):
    run = local_path(root, packet["run_dir"])
    with lock(run / "start.lock"):
        once(run / "start-receipt.json", start_binding(packet_path, approval_path, load(approval_path)))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise PermissionError("Empty message array")
    if any(set(m) != {"role", "content"} or m["role"] not in ("system", "user", "assistant") or not isinstance(m["content"], str) for m in messages):
        raise PermissionError("Only exact plain-text message schema is allowed")
    if messages[-1]["role"] != "user":
        raise PermissionError("Query must be final user message")


def validate_row(row, kind):
    if not isinstance(row.get("group"), str) or not row["group"]:
        raise PermissionError("Missing group")
    if {"gold", "answer", "aliases", "gold_label", "feedbacks"} & set(row):
        raise PermissionError("Gold metadata in request row")
    if kind == "qa":
        for key in ("native_messages", "adapted_messages"):
            validate_messages(row[key])
        if row["native_messages"][-1] != row["adapted_messages"][-1]:
            raise PermissionError("Prompt arms changed query facts")
    elif any(k not in row for k in ("episode_id", "native_system", "adapted_system", "task_card", "task_name")):
        raise PermissionError("Incomplete agent query")


def build_body(messages, arm, pool, response_format=None):
    validate_messages(messages)
    on = arm.endswith("on-high")
    body = {"model": "deepseek-flash", "messages": messages,
            "thinking": {"type": "enabled" if on else "disabled"},
            "max_tokens": POOL_LIMITS[pool][5 if on else 4]}
    if on:
        body["reasoning_effort"] = "high"
    else:
        body["temperature"] = 0
    if response_format:
        if response_format != "json_object":
            raise PermissionError("Unknown response format")
        body["response_format"] = {"type": "json_object"}
    return body


def token_bound(packet, messages, root=ROOT):
    spec = packet.get("token_counter", {"kind": "utf8_upper_bound"})
    if spec["kind"] == "utf8_upper_bound":
        return len(encoded(messages)) + 128
    if spec["kind"] != "tokenizers":
        raise PermissionError("Unknown bound token counter")
    path = bound_path(packet, spec["path"], root)
    tokenizer = cached_tokenizer(str(path), file_map(packet)[spec["path"]]["sha256"])
    return sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) + 32 for m in messages) + 128


@lru_cache(maxsize=2)
def cached_tokenizer(path, expected_sha):
    from tokenizers import Tokenizer
    if sha(path) != expected_sha:
        raise PermissionError("Tokenizer binding changed")
    return Tokenizer.from_file(path)


def parse_task(content, scorer, metadata):
    if not isinstance(content, str):
        return {"valid": False, "prediction": None, "reason": "NO_FINAL_CONTENT"}
    try:
        if scorer == "mapeval":
            obj = json.loads(content)
            value = obj["option_no"]
            valid = isinstance(obj, dict) and set(obj) == {"option_no"} and type(value) is int and value in metadata["legal_options"]
            return {"valid": valid, "prediction": str(value) if valid else None, "reason": "OPTION_SCHEMA"}
        if scorer == "sgqa":
            match = re.fullmatch(r"\s*\[([^\[\]\n]+)\]\s*", content)
            return {"valid": bool(match), "prediction": match[1] if match else None, "reason": "ENTITY_BRACKET_SCHEMA"}
        if scorer == "ust":
            from build_llm_task_sweep_qa_20261023 import parse_ust
            prediction = parse_ust(content, metadata["ability"])
            return {"valid": prediction in ("A", "B", "C", "D"), "prediction": prediction, "reason": "NATIVE_UST_SCHEMA"}
        if scorer in ("sceneqa", "travel", "agent"):
            obj = json.loads(content)
            field = "action" if scorer == "agent" else "answer" if scorer == "sceneqa" else "plan"
            value = obj[field]
            valid = isinstance(obj, dict) and set(obj) == {field}
            if scorer == "agent":
                valid = valid and isinstance(value, str) and bool(value.strip()) and "\n" not in value and len(value) <= 300 and not value.lower().startswith("think:")
            elif scorer == "sceneqa":
                valid = valid and isinstance(value, str) and bool(value.strip())
            else:
                valid = valid and isinstance(value, list) and bool(value)
            return {"valid": bool(valid), "prediction": value if valid else None, "reason": "EXACT_" + field.upper() + "_SCHEMA"}
    except (ValueError, TypeError, KeyError):
        pass
    return {"valid": False, "prediction": None, "reason": "INVALID_SCHEMA"}


def checked_usage(payload, input_cap, output_cap):
    usage = payload.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] <= 0 for k in ("prompt_tokens", "completion_tokens")):
        return None
    hit = usage.get("prompt_cache_hit_tokens", 0)
    if type(hit) is not int or not 0 <= hit <= usage["prompt_tokens"]:
        return None
    miss = usage.get("prompt_cache_miss_tokens", usage["prompt_tokens"] - hit)
    if type(miss) is not int or miss != usage["prompt_tokens"] - hit:
        return None
    if usage["prompt_tokens"] > input_cap or usage["completion_tokens"] > output_cap:
        return None
    if "total_tokens" in usage and usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        return None
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        return None
    reasoning = details.get("reasoning_tokens")
    if reasoning is not None and (type(reasoning) is not int or not 0 <= reasoning <= usage["completion_tokens"]):
        return None
    return usage


def classify_wire(wire, body, pool, scorer, metadata, identity=None):
    payload = wire.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    choices = payload.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    status = wire.get("http_status", 0)
    usage = checked_usage(payload, POOL_LIMITS[pool][3], body["max_tokens"])
    # Only explicit rejection statuses without any final/reasoning output qualify.
    retryable = (status == 429 or 500 <= status <= 599) and not content and not message.get("reasoning_content")
    if status != 200:
        return {"status": "RETRYABLE_EXPLICIT_REJECTION" if retryable else "HTTP_UNKNOWN_OR_FAILED",
                "valid": False, "prediction": None, "retryable": retryable, "text": content or "",
                "mechanical_ok": False, "stop": not retryable}, usage
    observed = {"model": str(payload.get("model", "")).lower(), "fingerprint": payload.get("system_fingerprint")}
    ok_model = observed["model"] in ("deepseek-flash", "deepseek-v4.1-flash-1017")
    ok_fingerprint = isinstance(observed["fingerprint"], str) and bool(observed["fingerprint"].strip())
    response_id = payload.get("id")
    drift = (not ok_model or not ok_fingerprint or not isinstance(response_id, str) or not response_id.strip()
             or (identity is not None and observed != identity))
    mechanical = (not drift and usage is not None and choice.get("finish_reason") == "stop" and message.get("role") == "assistant"
                  and not message.get("tool_calls") and isinstance(content, str))
    parsed = parse_task(content, scorer, metadata)
    return {**parsed, "status": "RECEIVED" if mechanical else "PROVIDER_OR_USAGE_DRIFT", "text": content,
            "valid": bool(parsed["valid"] and mechanical), "mechanical_ok": mechanical, "stop": not mechanical,
            "retryable": False, "identity": observed, "response_id": payload.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "thinking_observation": {"requested": body["thinking"]["type"], "visible_reasoning_present": bool(message.get("reasoning_content")),
                "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")}}, usage


class Runtime:
    def __init__(self, packet, packet_path, approval_path, root=ROOT, transport=None, key_loader=None, budget_path=None):
        self.packet, self.packet_path, self.approval_path, self.root = packet, Path(packet_path), Path(approval_path), Path(root)
        self.run = local_path(root, packet["run_dir"])
        self.transport, self.key_loader, self.key = transport, key_loader, None
        self._verified = False
        self.extra_bindings = {}
        self.run.mkdir(parents=True, exist_ok=True)
        from privacy_synth.pilot_ledger import SqliteBudget
        # An injected ledger is used only by local transport fixtures; CLI collection
        # always uses the one canonical route ledger verified in check_packet().
        self.budget = SqliteBudget(budget_path or local_path(root, BUDGET_PATH), 100,
            {k: {"calls": v[1], "input_tokens": v[1] * v[3], "output_tokens": v[1] * v[5], "usd": v[2]} for k, v in POOL_LIMITS.items()}, packet["provider"]["rates"])
        with sqlite3.connect(self.budget.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS logical(tag TEXT PRIMARY KEY,pool TEXT NOT NULL,created REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS retry_slots(pool TEXT NOT NULL,tag TEXT UNIQUE NOT NULL)")

    def stop(self, reason):
        if not (self.run / "global-stop.json").exists():
            atomic(self.run / "global-stop.json", {"reason": reason, "at": time.time()})
        shared = self.budget.path.parent / "global-stop.json"
        if not shared.exists():
            atomic(shared, {"reason": reason, "at": time.time(), "run_dir": self.packet["run_dir"]})

    def guard(self):
        if not self._verified:
            authorize(self.packet_path, self.approval_path, self.root)
            self._packet_sha, self._approval_sha = sha(self.packet_path), sha(self.approval_path)
            self._window = load(self.approval_path)
            self._stats = {s["path"]: (local_path(self.root, s["path"]).stat().st_size,
                local_path(self.root, s["path"]).stat().st_mtime_ns) for s in self.packet["files"] if s.get("kind") != "gold"}
            self._verified = True
        else:
            if sha(self.packet_path) != self._packet_sha or sha(self.approval_path) != self._approval_sha:
                raise PermissionError("Approved packet/approval changed during collection")
            if ((self.run / "global-stop.json").exists() or (self.budget.path.parent / "global-stop.json").exists()
                    or not self._window["start_epoch"] <= time.time() < self._window["deadline_epoch"]):
                raise PermissionError("Global stop/deadline")
            for spec in self.packet["files"]:
                if spec.get("kind") == "gold":
                    continue
                path = local_path(self.root, spec["path"])
                stat = (path.stat().st_size, path.stat().st_mtime_ns)
                if spec.get("kind") == "code" or stat != self._stats[spec["path"]]:
                    bound_path(self.packet, spec["path"], self.root)
                self._stats[spec["path"]] = stat
        for path, expected in self.extra_bindings.items():
            if sha(path) != expected:
                raise PermissionError("Registered adaptive repair asset changed")

    def credentials(self):
        self.guard()  # Deliberately before import, environment-key read, loader or network.
        if self.key is None:
            if self.transport is None or self.key_loader is None:
                import deepseek_ipd_interface as official
                self.transport = self.transport or official.post_once
                self.key_loader = self.key_loader or (lambda: os.environ.get("DEEPSEEK_API_KEY") or official.load_key())
            self.key = self.key_loader()
            if not isinstance(self.key, str) or not self.key.strip():
                raise PermissionError("Missing explicitly selected DeepSeek credential")
        return self.key

    def request(self, pool, tag, body, scorer, metadata):
        self.guard()
        if token_bound(self.packet, body["messages"], self.root) > POOL_LIMITS[pool][3]:
            return {"status": "CONTEXT_CAP", "valid": False, "prediction": None, "mechanical_ok": True, "text": ""}
        call_dir = self.run / "calls" / pool / digest(tag)
        final = call_dir / "result.json"
        if final.exists():
            result = load(final)
            if result.get("body_sha256") not in (None, digest(body)):
                raise PermissionError("Attempted tag belongs to different request bytes")
            return result
        call_dir.mkdir(parents=True, exist_ok=True)
        with lock(self.budget.path.parent / "reserve.lock"):
            self.guard()
            with sqlite3.connect(self.budget.path) as db:
                old = db.execute("SELECT 1 FROM logical WHERE tag=?", (tag,)).fetchone()
                if old:
                    result = {"status": "UNKNOWN_PREVIOUS_SUBMISSION_NO_RESEND", "valid": False, "prediction": None, "mechanical_ok": False, "text": ""}
                    atomic(final, result)
                    return result
                if db.execute("SELECT COUNT(*) FROM logical WHERE pool=?", (pool,)).fetchone()[0] >= POOL_LIMITS[pool][0]:
                    raise SweepStop("Logical pool cap")
                if db.execute("SELECT COUNT(*) FROM logical").fetchone()[0] >= 6708:
                    raise SweepStop("Global logical cap")
                db.execute("INSERT INTO logical VALUES(?,?,?)", (tag, pool, time.time()))
        for attempt in (0, 1):
            with lock(self.budget.path.parent / "reserve.lock"):
                self.guard()
                if attempt:
                    with sqlite3.connect(self.budget.path) as db:
                        count = db.execute("SELECT COUNT(*) FROM retry_slots WHERE pool=?", (pool,)).fetchone()[0]
                        if count >= math.ceil(POOL_LIMITS[pool][0] * .1):
                            raise SweepStop("Technical retry allowance exhausted")
                        db.execute("INSERT INTO retry_slots VALUES(?,?)", (pool, tag))
                with sqlite3.connect(self.budget.path) as db:
                    if db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] >= 7379:
                        raise SweepStop("Global HTTP cap")
                ident = self.budget.reserve(pool, f"{tag}:http{attempt}", POOL_LIMITS[pool][3], body["max_tokens"])
                once(call_dir / f"attempt{attempt}.reserved.json", {"ledger_id": ident, "tag": tag, "body_sha256": digest(body), "request": body})
            key = self.credentials()
            try:
                wire = self.transport(self.packet["provider"]["endpoint"], body, key)
            except Exception as exc:
                wire = {"http_status": 0, "payload": {}, "error": type(exc).__name__}
            atomic(call_dir / f"attempt{attempt}.wire.json", wire)
            identity = load(self.run / "provider-identity.json") if (self.run / "provider-identity.json").exists() else None
            result, usage = classify_wire(wire, body, pool, scorer, metadata, identity)
            self.budget.settle(ident, usage)
            result.update({"tag": tag, "body_sha256": digest(body), "ledger_id": ident, "usage": usage})
            if result["mechanical_ok"]:
                with lock(self.run / "identity.lock"):
                    identity_path = self.run / "provider-identity.json"
                    if identity_path.exists() and load(identity_path) != result["identity"]:
                        result.update(status="PROVIDER_IDENTITY_RACE_DRIFT", valid=False, stop=True)
                    else:
                        once(identity_path, result["identity"])
            atomic(call_dir / f"attempt{attempt}.result.json", result)
            if result.get("retryable") and attempt == 0:
                continue
            atomic(final, result)
            if result.get("stop"):
                self.stop(result["status"])
            return result
        atomic(final, result)
        return result


class WorldWorker:
    def __init__(self, argv, log_path, root=ROOT, interpreter=None):
        env = dict(os.environ)
        env["_JAVA_OPTIONS"] = "-Xms64m -Xmx512m -XX:ActiveProcessorCount=2"
        self.log = Path(log_path).open("a", encoding="utf-8")
        command = [sys.executable if p == "SELF_PYTHON" else str(interpreter) if i == 0 and interpreter else str(local_path(root, p)) if "/" in p and not p.startswith("-") else p for i, p in enumerate(argv)]
        def limit_cpu():
            if hasattr(os, "sched_getaffinity"):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        self.process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1, start_new_session=True, preexec_fn=limit_cpu)
        self._read_buffer = b""

    def send(self, value, timeout=120):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._read_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise SweepStop("Native worker response deadline exceeded")
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    self.log.flush()
                    tail = Path(self.log.name).read_bytes()[-16384:].lower()
                    if b"outofmemory" in tail or b"cannot allocate memory" in tail or self.process.poll() in (-9, 137):
                        raise SweepStop("OOM_NATIVE_WORKER")
                    raise SweepStop("Native worker exited " + str(self.process.poll()))
                self._read_buffer += block
                if len(self._read_buffer) > 1048576:
                    raise SweepStop("Native worker JSONL response exceeds1MiB")
        line, self._read_buffer = self._read_buffer.split(b"\n", 1)
        response = json.loads(line.decode("utf-8"))
        if response.get("ok") is not True:
            raise SweepStop("Native worker error: " + str(response.get("error")))
        return response["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.send({"op": "close"}, timeout=10)
            except Exception:
                pass
        # The native JVM is a child in this owned session. Parent-only terminate
        # could orphan it when close/transport fails, even after Python exited.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                raise SweepStop("Owned native worker did not exit after SIGKILL") from exc
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    with contextlib.suppress(BrokenPipeError, OSError):
                        stream.close()
            self.log.close()


def public_observation(result):
    return {key: result[key] for key in ("observation", "look", "inventory", "task") if key in result}


def instruction_view(row, kind):
    """Remove only permitted generic instruction channels, preserving all facts/demos."""
    result = copy.deepcopy(row)
    instructions = {}
    if kind == "qa":
        for channel in ("native_messages", "adapted_messages"):
            messages = result[channel]
            if messages[0]["role"] != "system":
                raise PermissionError("Repair requires an existing first system instruction")
            instructions[channel] = messages[0]["content"]
            messages[0]["content"] = "<INSTRUCTION_CHANNEL>"
    else:
        for channel in ("native_system", "adapted_system", "task_card"):
            instructions[channel] = result[channel]
            result[channel] = "<INSTRUCTION_CHANNEL>"
    return result, instructions


def validate_repair(packet, packet_path, task_name, receipt_path, root=ROOT, phase="repair", allow_started_screen=False):
    if phase != "repair":
        raise PermissionError("Repairs are Dev-only; no Screen repair phase")
    policy = packet.get("repair_policy", {})
    if policy.get("allow") is not True or policy.get("max_per_task") != 1 or not policy.get("approved_scope_id"):
        raise PermissionError("Packet has no bounded adaptive repair scope")
    receipt_path = Path(receipt_path)
    if not receipt_path.resolve().is_relative_to(Path(root).resolve()):
        raise PermissionError("Repair receipt leaves project")
    receipt = load(receipt_path)
    if (receipt.get("status") != "ROOT_APPROVED_BOUNDED_DEV_INSTRUCTION_REPAIR" or receipt.get("root_approved") is not True
            or receipt.get("parent_packet_sha256") != sha(packet_path) or receipt.get("approved_scope_id") != policy["approved_scope_id"]
            or receipt.get("task") != task_name or receipt.get("phase") != "dev"):
        raise PermissionError("Root repair scope/lineage differs")
    if (receipt.get("model") != packet["provider"]["model"] or receipt.get("modes") != packet["modes"]
            or receipt.get("bounds_sha256") != digest(packet["bounds"])):
        raise PermissionError("Repair changed model/modes/budget contract")
    attest = receipt.get("attestations", {})
    required = ("same_query_facts", "same_fit_demo_ids", "no_query_gold_in_instructions", "screen_final_not_read", "data_scorable")
    if any(attest.get(key) is not True for key in required):
        raise PermissionError("Missing root semantic/public-rule review attestation")
    run = local_path(root, packet["run_dir"])
    if not allow_started_screen and (run / task_name / "screen/started.json").exists():
        raise PermissionError("Screen has started; instruction repair is permanently closed")
    selected = load(run / task_name / "dev/selection.json")
    if selected.get("packet_sha256") != sha(packet_path) or selected.get("screen_authorized") is not True or receipt.get("base_arm") != selected["selected_arm"]:
        raise PermissionError("Repair must retain original complete-Dev-selected model/mode")
    errors = receipt["dev_error_receipt"]
    error_path = local_path(root, errors["path"])
    if sha(error_path) != errors["sha256"]:
        raise PermissionError("Dev error report hash differs")
    error_report = load(error_path)
    if error_report.get("phase") != "dev" or error_report.get("screen_read") is not False or error_report.get("packet_sha256") != sha(packet_path) or error_report.get("data_scorable") is not True:
        raise PermissionError("Unscorable/data-failure repairs are not allowed")
    rules = receipt.get("public_rules", [])
    if not rules:
        raise PermissionError("Repair requires a bound published task rule or format citation")
    for rule in rules:
        if file_map(packet).get(rule["path"], {}).get("kind") not in ("public_rule", "code"):
            raise PermissionError("Gold/data files cannot serve as published instruction-rule citations")
        path = bound_path(packet, rule["path"], root)
        if sha(path) != rule["sha256"] or not rule.get("citation"):
            raise PermissionError("Public rule source/citation differs")
    task = packet["tasks"][task_name]
    old = read_rows(bound_path(packet, task["phases"]["dev"]["queries"], root))
    new_path = local_path(root, receipt["queries"]["path"])
    no_final(receipt["queries"]["path"])
    if sha(new_path) != receipt["queries"]["sha256"]:
        raise PermissionError("Repair query/instruction asset changed")
    new = read_rows(new_path)
    if len(old) != len(new) or {str(r["id"]) for r in old} != {str(r["id"]) for r in new} or len({str(r["id"]) for r in new}) != len(new):
        raise PermissionError("Repair changed Dev count/membership")
    by_id = {str(r["id"]): r for r in old}
    patches = {}
    for row in new:
        validate_row(row, task["kind"])
        old_facts, old_instructions = instruction_view(by_id[str(row["id"])], task["kind"])
        new_facts, new_instructions = instruction_view(row, task["kind"])
        if old_facts != new_facts:
            raise PermissionError("Repair changed data, Fit examples, metadata or query facts")
        for channel, value in new_instructions.items():
            if value != old_instructions[channel]:
                if channel in patches and patches[channel] != value:
                    raise PermissionError("Repair must be one uniform task instruction, not per-query hints")
                patches[channel] = value
    if not patches or digest(patches) != receipt.get("new_instruction_sha256"):
        raise PermissionError("Repair instruction hash differs or no instruction changed")
    # A channel declared changed must carry that same value on every Dev member.
    for row in new:
        _, instructions = instruction_view(row, task["kind"])
        if any(instructions[channel] != value for channel, value in patches.items()):
            raise PermissionError("Nonuniform instruction patch")
    return receipt, new, patches


def register_repair(budget_path, task_name, receipt_path, packet_path):
    """One transactional registration per task; identical resume is not a new repair."""
    with sqlite3.connect(budget_path, timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS repair_registry(task TEXT PRIMARY KEY,receipt_sha256 TEXT NOT NULL,parent_packet_sha256 TEXT NOT NULL)")
        old = db.execute("SELECT receipt_sha256,parent_packet_sha256 FROM repair_registry WHERE task=?", (task_name,)).fetchone()
        wanted = (sha(receipt_path), sha(packet_path))
        if old and old != wanted:
            raise PermissionError("Second task repair is forbidden")
        if not old:
            db.execute("INSERT INTO repair_registry VALUES(?,?,?)", (task_name, *wanted))
    return wanted[0]


def apply_instruction_patch(row, kind, patches):
    row = copy.deepcopy(row)
    if kind == "qa":
        for channel, text in patches.items():
            if row[channel][0]["role"] != "system":
                raise PermissionError("Screen instruction channel differs")
            row[channel][0]["content"] = text
    else:
        row.update(patches)
    return row


def task_selection(run, task_name):
    final = run / task_name / "dev/final-selection.json"
    return load(final if final.exists() else run / task_name / "dev/selection.json")


def agent_episode(runtime, task_name, task, row, arm, phase, out):
    interpreter = checked_interpreter(task, runtime.root) if task["worker_argv"][0] == SCIENCE_PYTHON else None
    worker = WorldWorker(task["worker_argv"], out / (digest([arm, row["id"]]) + ".worker.log"), runtime.root, interpreter)
    trace, success, valid_turns = [], False, 0
    try:
        state = worker.send({"op": "start", "episode_id": row["episode_id"]})
        initial = public_observation(state)
        header = row["adapted_system"] if arm.startswith("adapted") else row["native_system"]
        prefix = [{"role": "system", "content": header + "\n" + row["task_card"]}]
        if arm.startswith("adapted"):
            demo = row.get("fit_demo")
            if not isinstance(demo, dict) or set(demo) != {"task", "initial_observation", "steps"}:
                raise PermissionError("Exactly one allowlisted public Fit demonstration is required")
            prefix.append({"role": "user", "content": "Public Fit demonstration:\n" + json.dumps(demo, ensure_ascii=False)})
            prefix.append({"role": "assistant", "content": "I will follow the new task and its own observations."})
        history = []
        cap = 49 if task_name == "ALFWorld" else 100
        for turn in range(cap):
            messages = prefix + [{"role": "user", "content": "Initial task and observation:\n" + json.dumps(initial, ensure_ascii=False)}] + history
            # Full initial state, goal, demo and complete trajectory are preserved.
            body = build_body(messages, arm, task_name, "json_object")
            result = runtime.request(task_name, f"{task_name}/{phase}/{arm}/{row['id']}/turn{turn}", body, "agent", {})
            trace.append({"turn": turn, "prediction": result})
            if result["status"] == "CONTEXT_CAP":
                return {"id": str(row["id"]), "arm": arm, "valid": False, "success": False, "status": "CONTEXT_CAP", "trace": trace}
            if result.get("stop"):
                raise SweepStop(result["status"])
            valid_turns += int(result["valid"])
            action = result["prediction"] if result["valid"] else ""
            state = worker.send({"op": "step", "action": action})
            trace[-1]["native_result_local_only"] = state
            history += [{"role": "assistant", "content": result.get("text") or ""},
                        {"role": "user", "content": json.dumps(public_observation(state), ensure_ascii=False)}]
            success = bool(state.get("success"))
            if success or state.get("done"):
                break
        return {"id": str(row["id"]), "arm": arm, "success": success, "valid": bool(trace) and valid_turns == len(trace),
                "status": "EPISODE_FINISHED", "trace": trace,
                "completion_tokens": sum((r["prediction"].get("usage") or {}).get("completion_tokens", 0) for r in trace)}
    finally:
        worker.close()


def collect(packet_path, approval_path, task_name, phase, resume_unsubmitted=False, repair_receipt=None):
    packet = authorize(packet_path, approval_path)
    task = packet["tasks"][task_name]
    if task.get("ready") is not True:
        raise PermissionError("Task is conditional/not ready; freeze data and approval first")
    start_run(packet, packet_path, approval_path)
    runtime = Runtime(packet, packet_path, approval_path)
    out = runtime.run / task_name / phase
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / "phase.lock"):
        runtime.guard()
        if (out / "predictions-frozen.json").exists():
            raise PermissionError("Predictions are frozen; collection cannot restart")
        started = out / "started.json"
        if started.exists() and not resume_unsubmitted:
            raise PermissionError("Partial paid phase requires explicit unsubmitted-only resume proof")
        if started.exists() and load(started)["packet_sha256"] != sha(packet_path):
            raise PermissionError("A started phase cannot be rebound to another packet")
        if repair_receipt and phase != "repair":
            raise PermissionError("Explicit repair receipt can only register a Dev repair")
        if phase == "repair":
            if not repair_receipt:
                raise PermissionError("Dev repair requires a root-reviewed scope receipt")
            receipt, rows, patches = validate_repair(packet, packet_path, task_name, repair_receipt)
            with lock(runtime.budget.path.parent / "reserve.lock"):
                repair_sha = register_repair(runtime.budget.path, task_name, repair_receipt, packet_path)
            once(out / "repair-lineage.json", {"receipt_path": Path(repair_receipt).resolve().relative_to(ROOT).as_posix(),
                "receipt_sha256": repair_sha, "patches": patches, "base_arm": receipt["base_arm"]})
            runtime.extra_bindings = {Path(repair_receipt): repair_sha,
                local_path(ROOT, receipt["queries"]["path"]): receipt["queries"]["sha256"],
                local_path(ROOT, receipt["dev_error_receipt"]["path"]): receipt["dev_error_receipt"]["sha256"]}
            arms = [receipt["base_arm"]]
        else:
            rows = read_rows(bound_path(packet, task["phases"][phase]["queries"]))
        if phase == "screen":
            selected = task_selection(runtime.run, task_name)
            if selected["packet_sha256"] != sha(packet_path) or selected.get("screen_authorized") is not True:
                raise PermissionError("Selection belongs to another packet")
            lineage_path = runtime.run / task_name / "repair/repair-lineage.json"
            if lineage_path.exists():
                if not (runtime.run / task_name / "dev/final-selection.json").exists():
                    raise PermissionError("Registered repair must freeze/score before Screen")
                lineage = load(lineage_path)
                path = local_path(ROOT, lineage["receipt_path"])
                with sqlite3.connect(runtime.budget.path) as db:
                    registered = db.execute("SELECT receipt_sha256 FROM repair_registry WHERE task=?", (task_name,)).fetchone()
                if not registered or registered[0] != sha(path) or registered[0] != lineage["receipt_sha256"]:
                    raise PermissionError("Registered repair lineage differs")
                receipt, _, patches = validate_repair(packet, packet_path, task_name, path, allow_started_screen=True)
                runtime.extra_bindings = {path: registered[0], local_path(ROOT, receipt["queries"]["path"]): receipt["queries"]["sha256"]}
                if selected.get("use_repair"):
                    rows = [apply_instruction_patch(row, task["kind"], patches) for row in rows]
            arms = [selected["selected_arm"]]
        elif phase == "dev":
            arms = list(ARMS)
        if started.exists():
            with sqlite3.connect(runtime.budget.path) as db:
                known = db.execute("SELECT tag FROM logical WHERE tag LIKE ?", (f"{task_name}/{phase}/%",)).fetchall()
            atomic(out / f"resume-proof-{time.time_ns()}.json", {"submitted_tags_never_resent": [x[0] for x in known], "packet_sha256": sha(packet_path)})
        else:
            once(started, {"packet_sha256": sha(packet_path), "at": time.time(), "arms": arms})
        completed, last_progress = [], time.monotonic()
        try:
            # Query-major order makes the first2QA/first1episode a shared mechanical prelude.
            for row in rows:
                for arm in arms:
                    runtime.guard()
                    if time.monotonic() - last_progress > 1800:
                        raise SweepStop("Thirty minutes without progress")
                    result_path = out / (digest([row["id"], arm]) + ".prediction.json")
                    if result_path.exists():
                        result = load(result_path)
                    elif task["kind"] == "agent":
                        # Replaying a partly paid dynamic trajectory would change the query state.
                        prefix = f"{task_name}/{phase}/{arm}/{row['id']}/%"
                        with sqlite3.connect(runtime.budget.path) as db:
                            prior = db.execute("SELECT COUNT(*) FROM logical WHERE tag LIKE ?", (prefix,)).fetchone()[0]
                        if prior:
                            result = {"id": str(row["id"]), "arm": arm, "success": False, "valid": False, "status": "PARTIAL_EPISODE_NO_REPLAY"}
                        else:
                            result = agent_episode(runtime, task_name, task, row, arm, phase, out)
                    else:
                        messages = row["adapted_messages"] if arm.startswith("adapted") else row["native_messages"]
                        body = build_body(messages, arm, task_name, task.get("response_format"))
                        result = runtime.request(task_name, f"{task_name}/{phase}/{arm}/{row['id']}", body, task["scorer"], row.get("metadata", {}))
                        result.update(id=str(row["id"]), arm=arm)
                    atomic(result_path, result)
                    completed.append({"id": str(row["id"]), "arm": arm, "path": result_path.relative_to(ROOT).as_posix(), "sha256": sha(result_path)})
                    last_progress = time.monotonic()
                    if result.get("stop"):
                        raise SweepStop(result["status"])
                    prelude = 1 if task["kind"] == "agent" else 2
                    if len(completed) == prelude * len(arms):
                        # No labels or native goal values participate in this mechanical check.
                        mechanical = all(load(local_path(ROOT, p["path"])).get("mechanical_ok", True)
                                         and load(local_path(ROOT, p["path"])).get("valid", False) for p in completed)
                        once(out / "mechanical-prelude.json", {"passed": mechanical, "items": len(completed), "gold_read": False})
                        if not mechanical:
                            raise SweepStop("Mechanical prelude failed")
            once(out / "predictions-frozen.json", {"packet_sha256": sha(packet_path), "predictions": completed,
                 "expected": len(rows) * len(arms), "complete": True, "collected_all": True, "gold_read": False})
        except (MemoryError, SweepStop, PermissionError) as exc:
            if isinstance(exc, PermissionError):
                runtime.stop("AUTHORIZATION_OR_ASSET_DRIFT: " + str(exc))
            oom_count = 0
            if isinstance(exc, MemoryError) or "oom" in str(exc).lower() or "memory" in str(exc).lower():
                with lock(runtime.run / "oom.lock"):
                    path = runtime.run / "oom-count.json"
                    oom_count = (load(path)["count"] if path.exists() else 0) + 1
                    atomic(path, {"count": oom_count, "last_task": task_name, "last_phase": phase})
                if oom_count >= 2:
                    runtime.stop("Two OOM failures")
            atomic(out / "partial-stop.json", {"reason": str(exc), "predictions_completed": completed, "oom_count_global": oom_count})
            raise
    return {"status": "PREDICTIONS_FROZEN", "count": len(completed)}


def read_gold(path):
    no_final(path)
    path = Path(path)
    if path.suffix == ".json":
        return load(path)
    result = {}
    for row in read_rows(path):
        result[str(row.get("id", row.get("record_id")))] = row.get("gold", row.get("gold_label", row.get("aliases")))
    return result


def score_prediction(result, task, gold, metadata):
    if task["kind"] == "agent":
        return bool(result.get("success")), bool(result.get("valid"))
    if not result.get("valid"):
        return False, False
    scorer = task["scorer"]
    if scorer == "sceneqa":
        from prepare_sceneqa_task_sweep_20261023 import score_response
        scored = score_response(result["text"], gold)
        # Unknown is a refusal, not evidence of a correct grounded SQA answer.
        unknown = str(scored.get("prediction", "")).strip().lower() in ("unknown", "unanswerable", "not enough information")
        return bool(scored["correct"] and not unknown), bool(scored["valid"])
    if scorer == "sgqa":
        from build_llm_task_sweep_qa_20261023 import score_sgqa
        return score_sgqa(result["text"], str(gold)), True
    if scorer in ("mapeval", "ust"):
        return str(result["prediction"]) == str(gold), True
    raise PermissionError("Task scorer is conditional/not implemented; no fabricated score")


def review(packet_path, task_name, phase):
    packet = check_packet(packet_path, allow_stop=True)
    task = packet["tasks"][task_name]
    run = local_path(ROOT, packet["run_dir"])
    out = run / task_name / phase
    with lock(out / "phase.lock"):
        frozen = load(out / "predictions-frozen.json")
        if frozen.get("packet_sha256") != sha(packet_path) or frozen.get("complete") is not True:
            raise PermissionError("Complete predictions must freeze before phase gold")
        source_phase = "dev" if phase == "repair" else phase
        rows = read_rows(bound_path(packet, task["phases"][source_phase]["queries"]))
        by_id = {str(row["id"]): row for row in rows}
        predictions = []
        for spec in frozen["predictions"]:
            path = local_path(ROOT, spec["path"])
            if sha(path) != spec["sha256"]:
                raise PermissionError("Frozen prediction changed")
            predictions.append(load(path))
        if len(predictions) != frozen["expected"]:
            raise PermissionError("Prediction denominator differs")
        gold = {} if task["kind"] == "agent" else read_gold(bound_path(packet, task["phases"][source_phase]["gold"], read_gold=True))
        if task["kind"] == "qa" and not set(by_id) <= set(gold):
            raise PermissionError("Gold is missing frozen query members")
        expected_arms = set(ARMS) if phase == "dev" else {load(out / "repair-lineage.json")["base_arm"]} if phase == "repair" else {task_selection(run, task_name)["selected_arm"]}
        if {(r["id"], r["arm"]) for r in predictions} != {(rid, arm) for rid in by_id for arm in expected_arms}:
            raise PermissionError("Frozen prediction member/arm grid differs")
        arms = {}
        for result in predictions:
            correct, valid = score_prediction(result, task, gold.get(result["id"]), by_id[result["id"]].get("metadata", {}))
            arm = arms.setdefault(result["arm"], {"correct": 0, "valid": 0, "total": 0, "completion_tokens": 0})
            arm["correct"] += int(correct)
            arm["valid"] += int(valid)
            arm["total"] += 1
            arm["completion_tokens"] += result.get("completion_tokens", (result.get("usage") or {}).get("completion_tokens", 0))
        receipt = {"packet_sha256": sha(packet_path), "phase": phase, "arms": arms, "fixed_denominator": True, "gold_after_prediction_freeze": True}
        once(out / "score.json", receipt)
        if phase == "dev":
            # Success/accuracy first, validity second; off then shorter response breaks ties.
            selected = min(arms, key=lambda a: (-arms[a]["correct"], -arms[a]["valid"], a.endswith("on-high"), arms[a]["completion_tokens"], ARMS.index(a)))
            once(out / "selection.json", {"packet_sha256": sha(packet_path), "selected_arm": selected,
                  "rule": "Devcorrect/success, then valid, off-mode, shorter-output, fixed-arm-order", "score_sha256": sha(out / "score.json"),
                  "screen_read": False, "screen_authorized": frozen.get("collected_all") is True})
        elif phase == "repair":
            lineage = load(out / "repair-lineage.json")
            base_selection = load(run / task_name / "dev/selection.json")
            old = load(run / task_name / "dev/score.json")["arms"][lineage["base_arm"]]
            new = arms[lineage["base_arm"]]
            rank = lambda r: (r["correct"], r["valid"], -r["completion_tokens"])
            use_repair = frozen.get("collected_all") is True and rank(new) > rank(old)
            once(run / task_name / "dev/final-selection.json", {**base_selection, "use_repair": use_repair,
                "repair_score_sha256": sha(out / "score.json"), "root_repair_receipt_sha256": lineage["receipt_sha256"],
                "rule": "At mostone same-mode Dev instruction repair; success/correct,valid,shorter; exacttie retains original"})
        return receipt


def freeze_partial(packet_path, task_name, phase):
    """Seal a stopped phase with every missing member counted invalid; never resume it."""
    packet = check_packet(packet_path, allow_stop=True)
    task = packet["tasks"][task_name]
    run = local_path(ROOT, packet["run_dir"])
    out = run / task_name / phase
    with lock(out / "phase.lock"):
        if (out / "predictions-frozen.json").exists():
            return load(out / "predictions-frozen.json")
        started = load(out / "started.json")
        if started["packet_sha256"] != sha(packet_path):
            raise PermissionError("Partial phase belongs to another packet")
        source_phase = "dev" if phase == "repair" else phase
        rows = read_rows(bound_path(packet, task["phases"][source_phase]["queries"]))
        arms = started["arms"]
        predictions, missing = [], 0
        for row in rows:
            for arm in arms:
                path = out / (digest([row["id"], arm]) + ".prediction.json")
                if not path.exists():
                    missing += 1
                    atomic(path, {"id": str(row["id"]), "arm": arm, "valid": False, "success": False,
                        "status": "UNFINISHED_PHASE_MEMBER_COUNTED_INVALID", "prediction": None})
                predictions.append({"id": str(row["id"]), "arm": arm, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
        receipt = {"packet_sha256": sha(packet_path), "predictions": predictions, "expected": len(rows) * len(arms),
                   "complete": True, "collected_all": False, "missing_counted_invalid": missing, "gold_read": False,
                   "collection_permanently_closed": True}
        once(out / "predictions-frozen.json", receipt)
        return receipt


def repair_proposal(packet_path, task_name, queries_path, error_receipt_path):
    """Prepare a concrete Dev-only amendment; this does not authorize its collection."""
    packet = check_packet(packet_path, allow_stop=True)
    task = packet["tasks"][task_name]
    run = local_path(ROOT, packet["run_dir"])
    score = load(run / task_name / "dev/score.json")
    if score["packet_sha256"] != sha(packet_path):
        raise PermissionError("Repair requires this packet's frozen/scored Dev")
    no_final(queries_path)
    no_final(error_receipt_path)
    old = read_rows(bound_path(packet, task["phases"]["dev"]["queries"]))
    new = read_rows(queries_path)
    if {str(r["id"]) for r in old} != {str(r["id"]) for r in new} or len(old) != len(new):
        raise PermissionError("Repair membership must exactly equal original Dev")
    old_by_id = {str(r["id"]): r for r in old}
    patches = {}
    for row in new:
        validate_row(row, task["kind"])
        original = old_by_id[str(row["id"])]
        before, before_instruction = instruction_view(original, task["kind"])
        after, after_instruction = instruction_view(row, task["kind"])
        if before != after:
            raise PermissionError("Repair changed data/query facts/Fit demos/metadata")
        for channel, text in after_instruction.items():
            if text != before_instruction[channel]:
                if channel in patches and patches[channel] != text:
                    raise PermissionError("Repair instructions must be uniform across Dev")
                patches[channel] = text
    errors = load(error_receipt_path)
    if errors.get("phase") != "dev" or errors.get("screen_read") is not False or errors.get("packet_sha256") != sha(packet_path):
        raise PermissionError("Error receipt must attest Dev-only review")
    receipt = {"status": "DEV_REPAIR_PROPOSED_AWAITING_ROOT_PROTOCOL_REVIEW", "parent_packet_sha256": sha(packet_path),
        "approved_scope_id": packet.get("repair_policy", {}).get("approved_scope_id"), "root_approved": False,
        "task": task_name, "phase": "dev", "base_arm": load(run / task_name / "dev/selection.json")["selected_arm"],
        "model": packet["provider"]["model"], "modes": packet["modes"], "bounds_sha256": digest(packet["bounds"]),
        "queries": {"path": Path(queries_path).resolve().relative_to(ROOT).as_posix(), "sha256": sha(queries_path)},
        "dev_error_receipt": {"path": Path(error_receipt_path).resolve().relative_to(ROOT).as_posix(), "sha256": sha(error_receipt_path)},
        "new_instruction_sha256": digest(patches), "public_rules": [], "attestations": {},
        "members": [str(r["id"]) for r in new], "max_new_conditions": 1, "calls_started": 0,
        "next_gate": "Root reviews published-rule/format-only instructions and signs bounded scope receipt; original user approval must explicitly allow this adaptive content"}
    once(run / task_name / "dev-repair-proposal.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "approval-template", "collect", "review", "freeze-partial", "prepare-repair"))
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--task", choices=tuple(POOL_LIMITS))
    parser.add_argument("--phase", choices=("dev", "repair", "screen"))
    parser.add_argument("--resume-unsubmitted", action="store_true")
    parser.add_argument("--repair-queries", type=Path)
    parser.add_argument("--error-receipt", type=Path)
    parser.add_argument("--repair-receipt", type=Path)
    parser.add_argument("--confirmed-at", type=float)
    args = parser.parse_args()
    if args.action == "check":
        packet = check_packet(args.packet)
        result = {"status": "OFFLINE_CHECK_PASS", "ready_tasks": [k for k, v in packet["tasks"].items() if v.get("ready")], "calls": 0, "key_read": False}
    elif args.action == "approval-template":
        packet = check_packet(args.packet)
        result = {**approval_binding(packet, args.packet), "approved": False, "user_confirmation": None,
                  "allow_bounded_dev_instruction_repair": False, "start_epoch": args.confirmed_at,
                  "deadline_epoch": args.confirmed_at + packet["execution_window_seconds"] if args.confirmed_at is not None else None}
    elif args.action == "collect":
        if not args.approval or not args.task or not args.phase:
            parser.error("collect requires approval/task/phase")
        result = collect(args.packet, args.approval, args.task, args.phase, args.resume_unsubmitted, args.repair_receipt)
    elif args.action == "review":
        if not args.task or not args.phase:
            parser.error("review requires task/phase")
        result = review(args.packet, args.task, args.phase)
    elif args.action == "freeze-partial":
        if not args.task or not args.phase:
            parser.error("freeze-partial requires task/phase")
        result = freeze_partial(args.packet, args.task, args.phase)
    else:
        if not args.task or not args.repair_queries or not args.error_receipt:
            parser.error("prepare-repair requires task/repair-queries/error-receipt")
        result = repair_proposal(args.packet, args.task, args.repair_queries, args.error_receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
