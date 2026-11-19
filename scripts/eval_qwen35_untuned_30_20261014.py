#!/usr/bin/env python3
"""Score 30 frozen public questions with the local untuned Qwen3.5-2B.

Ten items from each of the 200 DeepSeek-requested FinQA, TAT-QA, and BIRD
questions. Selection uses strata and a fixed seed, never a score. No adapter,
no training, no external call. Gold stays local.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/kiryu/privacy-synthetic-distillation")
# Local Triton stand-in for the unbuilt causal_conv1d CUDA package. Transformers
# binds these functions when the Qwen3.5 module is imported, so the path has to
# exist before that import.
sys.path.insert(0, str(ROOT / "artifacts" / "kernels"))
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "hf-home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Triton compiles a CUDA shim that includes Python.h. The system image has no
# python3.12-dev, so the headers extracted from the Ubuntu dev packages live here.
_PY_INC = ROOT / "artifacts" / "python-dev" / "include"
if _PY_INC.is_dir():
    _cpath = f"{_PY_INC / 'python3.12'}{os.pathsep}{_PY_INC}"
    os.environ["CPATH"] = _cpath if not os.environ.get("CPATH") else _cpath + os.pathsep + os.environ["CPATH"]
    os.environ["C_INCLUDE_PATH"] = os.environ["CPATH"]

DS_RUN = ROOT / "runs" / "2026-10-13-overnight-ds-datasets"
RUN = ROOT / "runs" / "2026-10-14-qwen35-untuned-30"
REPORT = ROOT / "reports" / "qwen35-untuned-30-2026-10-14.md"
REPORT_COPY = Path("/mnt/c/Users/10950/Documents/Codex/2026-10-12/h/outputs/qwen35-untuned-30-2026-10-14.md")
MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SNAPSHOT = (
    ROOT / "artifacts" / "hf-home" / "hub" / "models--Qwen--Qwen3.5-2B" / "snapshots" / REVISION
)
SEED = 20261014
PER_DATASET = 10
MAX_NEW = 16384
MAX_BATCH = 4
MAX_PROMPT_SUM = 12000
DECODER = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "seed": SEED,
    "max_new_tokens": MAX_NEW,
}


def load_overnight():
    path = ROOT / "scripts" / "overnight_ds_datasets_20261013.py"
    spec = importlib.util.spec_from_file_location("overnight_ds_datasets_20261013", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def requested_ids() -> set[str]:
    ids = set()
    for line in (DS_RUN / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def hamilton(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {key: counts[key] * n / total for key in counts}
    quota = {key: math.floor(raw[key]) for key in counts}
    left = n - sum(quota.values())
    ranked = sorted(counts, key=lambda key: (-(raw[key] - quota[key]), -counts[key], key))
    for key in ranked:
        if left == 0:
            break
        if quota[key] < counts[key]:
            quota[key] += 1
            left -= 1
    if left != 0:
        raise SystemExit(f"could not allocate {n} items")
    return {key: value for key, value in quota.items() if value}


def select_items(frozen: dict, requested: set[str]) -> list[dict]:
    rng = random.Random(SEED)
    chosen = []
    quotas = {}
    for dataset in ("finqa", "tatqa", "bird"):
        pool = [row for row in frozen[dataset] if row["id"] in requested]
        counts = Counter(row["stratum"] for row in pool)
        quota = hamilton(dict(counts), PER_DATASET)
        quotas[dataset] = quota
        by_stratum = defaultdict(list)
        for row in pool:
            by_stratum[row["stratum"]].append(row)
        for stratum in sorted(quota):
            rows = sorted(by_stratum[stratum], key=lambda row: row["id"])
            chosen.extend(rng.sample(rows, quota[stratum]))
    if len(chosen) != 30 or len({row["id"] for row in chosen}) != 30:
        raise SystemExit("selection did not yield 30 distinct items")
    return chosen, quotas


def final_content(text: str) -> str:
    """Chat content after the thinking block. Stop tokens are not part of the answer."""
    if "</think>" not in text:
        return ""
    body = text.rsplit("</think>", 1)[1]
    for token in ("<|im_end|>", "<|endoftext|>"):
        body = body.replace(token, "")
    return body.strip()


def wilson(correct: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = correct / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre - margin) / den, 3), round((centre + margin) / den, 3)


def ds_correct(ids: set[str]) -> dict[str, bool]:
    last = {}
    for line in (DS_RUN / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in ids:
            last[row["id"]] = bool((row.get("score") or {}).get("correct"))
    return last


def pack(rows: list[dict]) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda row: (row["prompt_tokens"], row["id"]))
    batches = []
    current = []
    total = 0
    for row in ordered:
        if current and (len(current) >= MAX_BATCH or total + row["prompt_tokens"] > MAX_PROMPT_SUM):
            batches.append(current)
            current = []
            total = 0
        current.append(row)
        total += row["prompt_tokens"]
    if current:
        batches.append(current)
    return batches


class PresencePenalty:
    """Penalise tokens already emitted. The mask stays on GPU."""

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = penalty
        self.prompt_length = prompt_length
        self.seen = None

    def __call__(self, input_ids, scores):
        import torch

        if self.penalty == 0:
            return scores
        if self.seen is None or self.seen.shape != scores.shape:
            self.seen = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        if input_ids.shape[-1] > self.prompt_length:
            self.seen.scatter_(1, input_ids[:, -1:].to(device=scores.device), True)
            scores = scores - self.penalty * self.seen.to(dtype=scores.dtype)
        return scores


class StepProgress:
    """Log decode progress. It does not change scores."""

    def __init__(self, every: int = 1024):
        self.every = every
        self.steps = 0
        self.started = time.perf_counter()

    def __call__(self, input_ids, scores):
        self.steps += 1
        if self.steps % self.every == 0:
            elapsed = time.perf_counter() - self.started
            print(
                f"GEN_PROGRESS steps={self.steps} seconds={elapsed:.1f} step_s={self.steps / elapsed:.2f}",
                flush=True,
            )
        return scores


def load_model():
    import torch
    from transformers import AutoTokenizer
    import transformers

    if not SNAPSHOT.is_dir() or (SNAPSHOT / "adapter_config.json").exists():
        raise SystemExit("pinned untuned snapshot is missing or contains an adapter")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer.padding_side = "left"
    errors = []
    model = None
    loaded_as = None
    for class_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                MODEL_ID,
                revision=REVISION,
                dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
                local_files_only=True,
            )
            loaded_as = class_name
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    if model is None:
        raise SystemExit("no loader succeeded: " + " | ".join(errors))
    if any("lora" in name.lower() for name, _ in model.named_parameters()):
        raise SystemExit("loaded model is not the untouched base")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model, loaded_as


def stop_ids(tokenizer) -> list[int]:
    ids = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id < 0:
            raise SystemExit(f"missing stop token {token}")
        ids.append(token_id)
    return ids


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )


def generate_batch(model, tokenizer, rows: list[dict], stops: list[int], max_cache_len: int) -> list[dict]:
    import torch
    from transformers import LogitsProcessorList

    texts = [row["rendered"] for row in rows]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    prompt_length = int(encoded["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=DECODER["temperature"],
            top_p=DECODER["top_p"],
            top_k=DECODER["top_k"],
            min_p=DECODER["min_p"],
            repetition_penalty=DECODER["repetition_penalty"],
            logits_processor=LogitsProcessorList([
                PresencePenalty(DECODER["presence_penalty"], prompt_length),
                StepProgress(),
            ]),
            eos_token_id=stops,
            pad_token_id=tokenizer.pad_token_id,
            cache_implementation="static",
            max_cache_len=max_cache_len,
        )
    elapsed = time.perf_counter() - started
    stop = set(stops)
    produced = []
    for index, row in enumerate(rows):
        new_ids = output[index, prompt_length:]
        ids = [int(token) for token in new_ids.tolist()]
        true_count = 0
        for token in ids:
            true_count += 1
            if token in stop:
                break
        raw = tokenizer.decode(new_ids[:true_count], skip_special_tokens=False)
        new_tokens = true_count
        finish = "length" if new_tokens >= MAX_NEW else "stop"
        produced.append({
            "id": row["id"],
            "dataset": row["dataset"],
            "stratum": row["stratum"],
            "source_id": row.get("source_id"),
            "prompt_sha256": row["prompt_sha256"],
            "prompt_tokens": row["prompt_tokens"],
            "new_tokens": new_tokens,
            "finish_reason": finish,
            "truncated": finish == "length",
            "content": final_content(raw),
            "raw_completion": raw,
            "elapsed_seconds_batch": round(elapsed, 3),
        })
    produced_tokens = sum(item["new_tokens"] for item in produced)
    print(
        f"BATCH_DONE n={len(rows)} new_tokens={produced_tokens} seconds={elapsed:.1f} tok_s={produced_tokens / elapsed:.1f}",
        flush=True,
    )
    return produced


def score_rows(overnight, modules, gold, rows: list[dict]) -> None:
    for row in rows:
        item_gold = gold[row["dataset"]][row["id"]]
        finish = row["finish_reason"]
        content = row["content"]
        if row["dataset"] == "finqa":
            score = overnight.score_finqa(modules["finqa"], content, item_gold, finish)
            score["correct"] = bool(score.get("correct"))
        elif row["dataset"] == "tatqa":
            score = overnight.score_tatqa(modules["tatqa"], modules["tatqa_metric"], content, item_gold, finish)
        else:
            score = overnight.score_bird(modules["bird"], content, item_gold, finish)
            score["correct"] = bool(score.get("official_ex"))
            score["parsed"] = bool(score.get("parse_ok"))
        row["score"] = {key: value for key, value in score.items() if key != "sql"}


def write_report(rows: list[dict], quotas: dict, loaded_as: str, wall: float) -> None:
    by_id = ds_correct({row["id"] for row in rows})
    lines = [
        "# 未微调 Qwen3.5-2B 的 30 题诊断",
        "",
        f"模型 `{MODEL_ID}`，修订 `{REVISION}`，未加载 adapter。思考开启。解码是 Qwen3.5 通用文本的官方思考采样：temperature 1.0，top_p 0.95，top_k 20，min_p 0，presence_penalty 1.5，repetition_penalty 1.0。种子 `{SEED}`。每题最多 {MAX_NEW} 个新 token。加载类 `{loaded_as}`。KV cache 用静态长度，decode 走编译后的图。同一采样下，静态 cache 与动态 cache 的贪心和解码 token 在核对批上一致。因果卷积用本地 Triton 内核。",
        "",
        "题目来自 2026-10-13 DeepSeek 已经请求过的 200+200+200 道。每种按层做 Hamilton 分配，层内按题号排序后用种子抽取 10 道。抽取时不看 DeepSeek 或 Qwen 的对错。评分器与那次 DeepSeek 运行相同：只评思考块之后的正文；截断算错；不从思考过程里捞答案。",
        "",
        "## 结果",
        "",
        "| 数据集 | Qwen 正确 | 完整未截断 | DeepSeek 在这 10 题上正确 | Qwen Wilson 95% |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    new_tokens = 0
    for dataset in ("finqa", "tatqa", "bird"):
        group = [row for row in rows if row["dataset"] == dataset]
        correct = sum(bool(row["score"]["correct"]) for row in group)
        complete = sum(not row["truncated"] for row in group)
        teacher = sum(bool(by_id[row["id"]]) for row in group)
        lo, hi = wilson(correct, len(group))
        new_tokens += sum(row["new_tokens"] for row in group)
        lines.append(f"| {dataset} | {correct}/10 | {complete}/10 | {teacher}/10 | {lo:.3f}–{hi:.3f} |")
    lines.extend([
        "",
        f"30 题生成墙钟 {wall:.1f} 秒，新 token {new_tokens}，约 {new_tokens / wall:.1f} token/秒。这是同一批 10 题上的描述。Wilson 区间只描述这 10 题，不能把两种模型分出高低，也不能外推到各自的 200 题。",
        "",
        "## 分层配额",
        "",
    ])
    for dataset, quota in quotas.items():
        lines.append(f"- {dataset}: " + ", ".join(f"{key} {value}" for key, value in sorted(quota.items())))
    lines.extend(["", "## 逐题", "", "| 数据集 | 题号 | 层 | Qwen | DeepSeek | 新 token | 截断 |", "| --- | --- | --- | --- | --- | ---: | --- |"])
    for row in sorted(rows, key=lambda item: (item["dataset"], item["id"])):
        lines.append(
            f"| {row['dataset']} | `{row['id']}` | {row['stratum']} | {bool(row['score']['correct'])} | {bool(by_id[row['id']])} | {row['new_tokens']} | {row['truncated']} |"
        )
    lines.extend([
        "",
        "逐题正文和评分在 `runs/2026-10-14-qwen35-untuned-30/responses/`。没有外部请求，没有训练，生成器状态和蒸馏状态都没有改。",
        "",
    ])
    text = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    REPORT_COPY.write_text(text, encoding="utf-8")


def main() -> None:
    import torch

    RUN.mkdir(parents=True, exist_ok=True)
    for dataset in ("finqa", "tatqa", "bird"):
        (RUN / "responses" / dataset).mkdir(parents=True, exist_ok=True)
    frozen = json.loads((DS_RUN / "frozen-items.json").read_text(encoding="utf-8"))
    selection_path = RUN / "selection.json"
    if selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        chosen = saved["items"]
        quotas = saved["quotas"]
    else:
        chosen, quotas = select_items(frozen, requested_ids())
        payload = {
            "seed": SEED,
            "per_dataset": PER_DATASET,
            "rule": "Hamilton within the 200 requested items of each dataset. Stratum order is sorted by name. Items inside a stratum are sorted by id, then drawn with Random(20261014). Scores are not an input.",
            "quotas": quotas,
            "items": [
                {key: row[key] for key in ("id", "dataset", "stratum", "source_id", "prompt_sha256", "origin")}
                for row in chosen
            ],
        }
        selection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("SELECTION_FROZEN 30", flush=True)
    by_id = {row["id"]: row for dataset in frozen.values() for row in dataset}
    overnight = load_overnight()
    modules = {
        "finqa": overnight.load_module("candidate_finqa", overnight.FINQA_MOD),
        "tatqa": overnight.load_module("candidate_tatqa", overnight.TATQA_MOD),
        "bird": overnight.load_module("candidate_bird", overnight.BIRD_MOD),
    }
    modules["tatqa_metric"] = modules["tatqa"].load_official_metric(overnight.RAW / "tatqa")
    gold = overnight.load_gold()
    for row in chosen:
        if row["id"] not in gold[row["dataset"]]:
            raise SystemExit(f"missing gold for {row['id']}")

    tokenizer, model, loaded_as = load_model()
    stops = stop_ids(tokenizer)
    print(f"MODEL_LOADED {loaded_as}", flush=True)
    prepared = []
    for row in chosen:
        source = by_id[row["id"]]
        rendered = render(tokenizer, source["prompt"])
        tokens = tokenizer(rendered, return_tensors="pt")
        prepared.append({
            **row,
            "prompt": source["prompt"],
            "rendered": rendered,
            "prompt_tokens": int(tokens["input_ids"].shape[-1]),
        })
    batches = pack(prepared)
    (RUN / "batches.json").write_text(
        json.dumps([[row["id"] for row in batch] for batch in batches], indent=2) + "\n",
        encoding="utf-8",
    )
    max_cache_len = max(row["prompt_tokens"] for row in prepared) + MAX_NEW
    # Compile the static decode graph before the exam seed is applied. Greedy
    # warmup does not enter the saved CUDA generator state.
    for width in sorted({len(batch) for batch in batches}):
        dummy = tokenizer(["ok"] * width, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            model.generate(
                **dummy,
                max_new_tokens=1,
                do_sample=False,
                cache_implementation="static",
                max_cache_len=max_cache_len,
                eos_token_id=stops,
                pad_token_id=tokenizer.pad_token_id,
            )
    print(f"COMPILE_WARM cache={max_cache_len}", flush=True)
    state_path = RUN / "cuda-rng.pt"
    if state_path.exists():
        torch.cuda.set_rng_state(torch.load(state_path, map_location="cuda"))
    else:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    wall_started = time.perf_counter()
    for batch in batches:
        if all((RUN / "responses" / row["dataset"] / f"{row['id']}.json").exists() for row in batch):
            continue
        produced = generate_batch(model, tokenizer, batch, stops, max_cache_len)
        score_rows(overnight, modules, gold, produced)
        for row in produced:
            path = RUN / "responses" / row["dataset"] / f"{row['id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(path)
        torch.save(torch.cuda.get_rng_state(), state_path)
    wall = time.perf_counter() - wall_started
    rows = []
    for row in chosen:
        path = RUN / "responses" / row["dataset"] / f"{row['id']}.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) != 30:
        raise SystemExit("missing scored responses")
    write_report(rows, quotas, loaded_as, wall)
    (RUN / "summary.json").write_text(json.dumps({
        "loaded_as": loaded_as,
        "revision": REVISION,
        "decoder": DECODER,
        "wall_seconds": wall,
        "correct": {
            dataset: sum(bool(row["score"]["correct"]) for row in rows if row["dataset"] == dataset)
            for dataset in ("finqa", "tatqa", "bird")
        },
    }, indent=2) + "\n", encoding="utf-8")
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
