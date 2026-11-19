#!/usr/bin/env python3
"""Re-score saved BIRD student answers with the original worker.

The ablation driver imports a heavy module before it spawns the scorer. The
worker then sets a 768 MiB address-space cap and cannot return the result, so
score_response records an execution error. This process stays small, calls the
same score_response, and replaces only the score fields. It does not generate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/kiryu/privacy-synthetic-distillation")
BIRD_PATH = Path("/mnt/c/Users/10950/Documents/Codex/2026-10-12/h/work/relational_sql/qualification-prep/candidate_bird.py")
GOLD_PATH = ROOT / "runs/2026-10-13-overnight-ds-datasets/gold/bird.json"


def load_bird():
    os.environ["PYTHONPATH"] = str(BIRD_PATH.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if str(BIRD_PATH.parent) not in sys.path:
        sys.path.insert(0, str(BIRD_PATH.parent))
    spec = importlib.util.spec_from_file_location("candidate_bird", BIRD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_bird"] = module
    spec.loader.exec_module(module)
    return module


def patch_file(path: Path, bird, gold: dict) -> str:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("score_source") == "candidate_bird_light_parent":
        return "skip"
    saved = row.get("score") or {}
    if saved.get("execution_status") == "ok" and saved.get("parse_ok"):
        row["score_source"] = "spawn_ok_kept"
        temporary = path.with_suffix(".json.tmp-rescore")
        temporary.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        return "kept_ok"
    content = row.get("content") or ""
    scored = bird.score_response(content, gold[row["id"]], finish_reason=row.get("finish_reason"))
    scored["correct"] = bool(scored.get("official_ex"))
    scored["parsed"] = bool(scored.get("parse_ok"))
    if "spawn_score" not in row:
        row["spawn_score"] = saved
    row["score"] = {key: value for key, value in scored.items() if key != "sql"}
    row["score_source"] = "candidate_bird_light_parent"
    temporary = path.with_suffix(".json.tmp-rescore")
    temporary.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return "rescored"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-age-seconds", type=float, default=2.0)
    args = parser.parse_args()
    bird = load_bird()
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    now = time.time()
    counts = {"skip": 0, "kept_ok": 0, "rescored": 0, "young": 0}
    flipped = []
    for arm in ("a2", "a3"):
        directory = args.run_dir / "responses" / arm
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            if now - path.stat().st_mtime < args.min_age_seconds:
                counts["young"] += 1
                continue
            before = json.loads(path.read_text(encoding="utf-8"))
            if before.get("score_source") in {"candidate_bird_light_parent", "spawn_ok_kept"}:
                counts["skip"] += 1
                continue
            kind = patch_file(path, bird, gold)
            counts[kind] = counts.get(kind, 0) + 1
            after = json.loads(path.read_text(encoding="utf-8"))
            old = bool((before.get("score") or {}).get("correct"))
            new = bool((after.get("score") or {}).get("correct"))
            if old != new:
                flipped.append(f"{arm}:{before['id']}:{old}->{new}")
    print(json.dumps({"counts": counts, "flipped": flipped}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
