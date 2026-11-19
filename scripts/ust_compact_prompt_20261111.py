"""Evaluate frozen UST prompts with deterministic compression of demonstrations."""
import re
from pathlib import Path
from statistics import median

from ust_quantization_20261111 import ROOT, OUT as QUANT, REFERENCE, digest, read, write

OUT = ROOT / "runs/2026-11-11-ust-efficient-inference-v1"
FIELDS = ("coverage", "demand", "distance", "waiting_time", "charging_time")
WRAPPERS = ("Task Description", "Data Schema", "Note")


def sections(text):
    return dict(re.findall(r"(?ms)^## ([^\n]+)\n(.*?)(?=^## |\Z)", text))


def compact_table(data):
    parts = re.split(r"(?m)^location ([1-4]):[ \t]*$", data)
    assert len(parts) == 9
    rows = []
    for location, block in zip(parts[1::2], parts[2::2], strict=True):
        pairs = re.findall(r"(?m)^- ([a-z_]+): ([^\n]+)$", block)
        assert tuple(field for field, _ in pairs) == FIELDS
        rows.append([location, *(value.strip() for _, value in pairs)])
    header = "location | " + " | ".join(FIELDS)
    table = "\n".join([header, *(" | ".join(row) for row in rows)])
    assert [line.split(" | ") for line in table.splitlines()[1:]] == rows
    return "\n" + table + "\n\n"


def prepare():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QUANT / "exports/bf16", local_files_only=True)
    original = read(REFERENCE / "fixed8.local.json")
    queries = read(REFERENCE / "queries.local.json")
    frozen = read(REFERENCE / "tokens.local.json")
    assert len(queries) == 72 and len(original) == 17
    all_sections = [sections(message["content"]) for message in original[1::2]]
    assert all(len({row[field] for row in all_sections}) == 1 for field in WRAPPERS)
    prefixes = {"full8": original, "dedup8": [original[0]], "table8": [original[0]]}
    for index, parts in enumerate(all_sections):
        for mode in ("dedup8", "table8"):
            compressed = dict(parts)
            if index:
                for field in WRAPPERS:
                    del compressed[field]
            if mode == "table8":
                compressed["Data"] = compact_table(parts["Data"])
            assert compressed["Question"] == parts["Question"]
            content = "".join(f"## {key}\n{value}" for key, value in compressed.items()).rstrip()
            prefixes[mode].extend([{"role": "user", "content": content}, original[2 + 2 * index]])
    output = {}
    for mode, prefix in prefixes.items():
        output[mode] = []
        for query in queries:
            prompt = tokenizer.apply_chat_template(
                [*prefix, {"role": "user", "content": query["question"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if mode == "full8":
                assert ids == frozen[query["id"]]["prompt_ids"]
            output[mode].append({"id": query["id"], "ids": ids})
    assert [median(len(row["ids"]) for row in output[mode]) for mode in prefixes] == [6000, 3851, 3323]
    write(OUT / "prompt-tokens.local.json", output)
    write(OUT / "prompt-protocol.json", {
        "source_checkpoint": str(QUANT / "exports/bf16"),
        "source_sha256": {name: digest(REFERENCE / name) for name in
                          ("fixed8.local.json", "queries.local.json", "answers.local.json", "tokens.local.json")},
        "prompt_tokens_sha256": digest(OUT / "prompt-tokens.local.json"),
        "cap": 1024, "api_calls": 0, "query_unchanged": True,
        "all_eight_examples_unchanged_in_count_order_answers": True,
    })


def summarize(raw, reference=None):
    from eval_ust_sg_qwen_modes_20261024 import classify

    questions = read(REFERENCE / "queries.local.json")
    labels = read(REFERENCE / "answers.local.json")

    def grade(folder):
        grades = []
        for index, query in enumerate(questions):
            row = read(folder / f"{index:04d}.json")
            assert row["id"] == query["id"]
            result = classify(row["raw_generated"], "off", "ust", query["metadata"], row["finish_reason"])
            prediction = result["prediction"]
            grades.append({
                **row, "prediction": prediction, "valid": bool(result["valid"]),
                "correct": bool(result["valid"] and prediction == labels[row["id"]]["answer"]),
            })
        return grades

    grades = grade(raw)
    summary = {
        "correct": sum(row["correct"] for row in grades),
        "valid": sum(row["valid"] for row in grades),
        "capped": sum(row["finish_reason"] == "length" for row in grades),
        "n": len(grades),
    }
    for key in ("prompt_tokens", "request_wall_seconds", "GPU_prefill_forward_seconds",
                "GPU_decode_forward_seconds", "cached_tokens", "ttft_seconds"):
        vals = [row.get(key) for row in grades if row.get(key) is not None]
        if vals:
            summary["median_" + key] = median(vals)
    if reference:
        pairs = list(zip(grade(reference), grades, strict=True))
        summary["paired"] = {
            "gains": sum(not a["correct"] and b["correct"] for a, b in pairs),
            "losses": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "changed_answers": sum(a["prediction"] != b["prediction"] for a, b in pairs),
            "changed_token_sequences": sum(a["response_ids"] != b["response_ids"] for a, b in pairs),
        }
    return summary


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from ust_fresh_zero_demo_student_cap1024_20261110 import decoder1024

    prepare()
    source = QUANT / "exports/bf16"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.28)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        source, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    decode = decoder1024()
    results = {}
    for mode, rows in read(OUT / "prompt-tokens.local.json").items():
        model.model.rope_deltas = None
        decode(torch, tokenizer, model, rows[0]["ids"])
        for index, row in enumerate(rows):
            model.model.rope_deltas = None
            answer = decode(torch, tokenizer, model, row["ids"])
            write(OUT / "hf" / mode / "raw" / f"{index:04d}.json", {"id": row["id"], **answer})
            write(OUT / "hf/live.json", {"mode": mode, "closed": index + 1})
        reference = (QUANT / "bf16/raw") if mode == "full8" else (OUT / "hf/full8/raw")
        results[mode] = summarize(OUT / "hf" / mode / "raw", reference)
    write(OUT / "hf/results.json", results)


if __name__ == "__main__":
    main()
