"""Local post-training quantization comparison on the fixed UST fresh72 cohort."""
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNS = ROOT / "runs"
REFERENCE = RUNS / "2026-11-10-ust-fresh100-zero-demo-v1/student-synth8-cap1024-v1"
OUT = RUNS / "2026-11-11-ust-quantization-v1"
MODEL = "current_n68_s831"
ACTOR = RUNS / f"2026-11-08-ust-scale-v1/{MODEL}/checkpoints/step0136/actor"
MODES = ("bf16", "int8", "nf4")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    import bitsandbytes as bnb
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from eval_ust_sg_qwen_modes_20261024 import classify
    from ust_fresh_zero_demo_student_cap1024_20261110 import decoder1024

    assert not (OUT / "protocol.json").exists()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.28)
    base = Path(read(REFERENCE / MODEL / "protocol.json")["base"])
    queries = read(REFERENCE / "queries.local.json")
    answers = read(REFERENCE / "answers.local.json")
    tokens = read(REFERENCE / "tokens.local.json")
    assert len(queries) == len(answers) == 72
    assert len(read(REFERENCE / "fixed8.local.json")) == 17
    inputs = [
        Path(__file__), *ACTOR.iterdir(), *base.glob("*.safetensors"),
        REFERENCE / "queries.local.json", REFERENCE / "answers.local.json",
        REFERENCE / "tokens.local.json", REFERENCE / "fixed8.local.json",
        *sorted((REFERENCE / MODEL / "raw").glob("[0-9][0-9][0-9][0-9].json")),
    ]
    pins = {str(path): digest(path) for path in inputs}
    write(OUT / "protocol.json", {
        "model": MODEL, "checkpoint_step": 136, "base": str(base),
        "conditions": MODES, "queries": 72, "demos": "exact frozen synthetic8",
        "cap": 1024, "device": torch.cuda.get_device_name(),
        "bitsandbytes": bnb.__version__, "torch": torch.__version__,
        "merge": "PEFT safe_merge into BF16, common source for all three exported models",
        "latency": "serial greedy batch1, one warmup per mode excluded",
        "pins_sha256": pins, "api_calls": 0, "new_training": 0,
    })
    decode = decoder1024()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    options = dict(local_files_only=True, dtype=torch.bfloat16,
                   device_map={"": 0}, attn_implementation="sdpa")

    def load(source, **extra):
        return AutoModelForImageTextToText.from_pretrained(source, **options, **extra).eval()

    def clear():
        gc.collect()
        torch.cuda.empty_cache()

    def generate(model, query):
        rope = model.model.rope_deltas
        assert rope is None or not rope.any().item()
        model.model.rope_deltas = None
        return decode(torch, tokenizer, model, tokens[query["id"]]["prompt_ids"])

    exports = OUT / "exports"
    merged_dir = exports / "bf16"
    write(OUT / "live.json", {"stage": "merge_and_export_bf16"})
    unmerged = PeftModel.from_pretrained(load(base), ACTOR, is_trainable=False)
    model = unmerged.merge_and_unload(safe_merge=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    del unmerged, model
    clear()
    configs = {
        "int8": BitsAndBytesConfig(load_in_8bit=True),
        "nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16),
    }
    expected_layers = {"int8": bnb.nn.Linear8bitLt, "nf4": bnb.nn.Linear4bit}
    for mode in MODES:
        target = exports / mode
        write(OUT / "live.json", {"stage": "load_or_export", "mode": mode})
        if mode in configs:
            model = load(merged_dir, quantization_config=configs[mode])
            model.save_pretrained(target)
            tokenizer.save_pretrained(target)
            del model
            clear()
        model = load(target)
        quantized_layers = (
            sum(isinstance(layer, expected_layers[mode]) for layer in model.modules())
            if mode in expected_layers else 0
        )
        assert mode == "bf16" or quantized_layers > 0
        generate(model, queries[0])
        torch.cuda.reset_peak_memory_stats()
        for index, query in enumerate(queries):
            if (OUT / "STOP").exists():
                raise InterruptedError("Quantization stopped")
            result = generate(model, query)
            write(OUT / mode / "raw" / f"{index:04d}.json", {
                "id": query["id"], **result,
            })
            write(OUT / "live.json", {"stage": "evaluate", "mode": mode, "closed": index + 1})
        write(OUT / mode / "metrics.json", {
            "quantized_layers": quantized_layers,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "weights_bytes": sum(p.stat().st_size for p in target.glob("*.safetensors")),
        })
        del model
        clear()

    conditions = {mode: OUT / mode / "raw" for mode in MODES}
    conditions["historical_unmerged"] = REFERENCE / MODEL / "raw"
    grades = {}
    summaries = {}
    for mode, folder in conditions.items():
        rows = []
        for index, query in enumerate(queries):
            raw = read(folder / f"{index:04d}.json")
            assert raw["id"] == query["id"]
            label = answers[query["id"]]["answer"]
            parsed = classify(raw["raw_generated"], "off", "ust", query["metadata"], raw["finish_reason"])
            rows.append({
                "id": query["id"], "metric": query["metric"], "prediction": parsed["prediction"],
                "valid": bool(parsed["valid"]), "correct": bool(parsed["valid"] and parsed["prediction"] == label),
                "capped": raw["finish_reason"] == "length", "seconds": raw["request_wall_seconds"],
                "response_ids": raw["response_ids"],
            })
        grades[mode] = rows
        summaries[mode] = {
            "n": len(rows), "correct": sum(row["correct"] for row in rows),
            "valid": sum(row["valid"] for row in rows), "capped": sum(row["capped"] for row in rows),
        }
        if mode in MODES:
            summaries[mode].update(read(OUT / mode / "metrics.json"))
            summaries[mode]["median_seconds"] = statistics.median(row["seconds"] for row in rows)
    contrasts = {}
    for before, after in [("historical_unmerged", "bf16"), ("bf16", "int8"), ("bf16", "nf4")]:
        pair = list(zip(grades[before], grades[after], strict=True))
        contrasts[f"{before}_to_{after}"] = {
            "gains": sum(not a["correct"] and b["correct"] for a, b in pair),
            "losses": sum(a["correct"] and not b["correct"] for a, b in pair),
            "changed_answer": sum(a["prediction"] != b["prediction"] for a, b in pair),
            "changed_tokens": sum(a["response_ids"] != b["response_ids"] for a, b in pair),
        }
    assert all(digest(Path(path)) == value for path, value in pins.items())
    write(OUT / "paired-grades.local.json", grades)
    write(OUT / "results.json", {"status": "COMPLETE", "summaries": summaries, "contrasts": contrasts})
    write(OUT / "live.json", {"stage": "complete", "epoch": time.time()})


if __name__ == "__main__":
    main()
