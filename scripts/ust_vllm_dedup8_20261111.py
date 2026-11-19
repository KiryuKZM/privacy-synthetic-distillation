"""Compare the frozen 3851-token UST prompt with and without vLLM prefix caching."""
import os
from pathlib import Path
import re
from statistics import median
import subprocess

from ust_compact_prompt_20261111 import OUT, REFERENCE, ROOT, digest, read, summarize, write

DEST = OUT / "vllm-dedup8-v1"
BASELINE = OUT / "vllm"
MODES = ("bf16", "bf16-cache")
PROMPTS = OUT / "prompt-tokens.local.json"


def main():
    assert not (DEST / "protocol.json").exists()
    assert read(OUT / "prompt-protocol.json")["prompt_tokens_sha256"] == digest(PROMPTS)
    prompts = read(PROMPTS)["dedup8"]
    assert len(prompts) == 72 and median(len(row["ids"]) for row in prompts) == 3851
    baseline = read(BASELINE / "results.json")
    assert all(baseline["statuses"][mode] == 0 for mode in MODES)
    worker = ROOT / "scripts/ust_serving_quantization_20261111.py"
    envdir = ROOT / "artifacts/ust-serve-env/venv"
    cuda = Path(read(OUT / "vllm-launch.json")["CUDA_HOME"])
    env = dict(
        os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="0", VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1", VLLM_USE_FLASHINFER_SAMPLER="0", CUDA_HOME=str(cuda),
        VLLM_CACHE_ROOT=str(envdir.parent / "cache"),
        TRITON_CACHE_DIR=str(envdir.parent / "triton"),
        PATH=f"{envdir / 'bin'}:{cuda / 'bin'}:{os.environ['PATH']}",
    )
    env.pop("PYTHONPATH", None)
    pins = {
        str(path): digest(path) for path in (
            PROMPTS, worker, Path(__file__), REFERENCE / "answers.local.json",
            REFERENCE / "queries.local.json", BASELINE / "results.json",
            *[BASELINE / mode / "engine-options.json" for mode in MODES],
            *[raw for mode in MODES for raw in (BASELINE / mode / "full8/raw").glob("*.json")],
        )
    }
    write(DEST / "protocol.json", {
        "checkpoint": "current_n68_s831 step136 merged BF16",
        "cohort": "same72", "new_prompts": ["dedup8"], "cache_states": list(MODES),
        "reference": "completed same-vLLM full8 at each cache setting",
        "pins": pins, "API": 0, "training": 0,
    })
    for mode in MODES:
        with (DEST / f"{mode}.log").open("x") as log:
            subprocess.run(
                [str(envdir / "bin/python"), "-u", str(worker), mode, "--prompts", "dedup8",
                 "--output-root", str(DEST)],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=480,
            )
        assert read(DEST / mode / "engine-options.json") == read(BASELINE / mode / "engine-options.json")

    answers = read(REFERENCE / "answers.local.json")
    output = {}
    for mode in MODES:
        folder = DEST / mode / "dedup8/raw"
        same_cache_reference = BASELINE / mode / "full8/raw"
        rows = [read(folder / f"{index:04d}.json") for index in range(72)]
        assert [row["id"] for row in rows] == [item["id"] for item in prompts]
        assert [row["prompt_tokens"] for row in rows] == [len(item["ids"]) for item in prompts]
        candidates = [re.findall(r"<Answer>([A-D])</Answer>", row["raw_generated"]) for row in rows]
        assert all(len(match) == 1 for match in candidates)
        assert all(row["finish_reason"] == "stop" for row in rows)
        count = sum(match[0] == answers[row["id"]]["answer"] for row, match in zip(rows, candidates, strict=True))
        metrics = summarize(folder, same_cache_reference)
        assert metrics["correct"] == count
        cached = [row for row in rows if row["cached_tokens"] > 0]
        output[mode] = {
            "dedup8": metrics,
            "same_cache_full8": baseline["results"][mode]["full8"],
            "cache_hits": len(cached),
            "warm_request_median_seconds": median(row["request_wall_seconds"] for row in cached) if cached else None,
        }
        assert bool(cached) == mode.endswith("-cache")
    output["cache_effect_on_dedup8"] = summarize(
        DEST / "bf16-cache/dedup8/raw", DEST / "bf16/dedup8/raw"
    )["paired"]
    assert all(digest(Path(path)) == value for path, value in pins.items())
    write(DEST / "results.json", {"status": "COMPLETE", "output": output, "independent_recount": 144})


if __name__ == "__main__":
    main()
