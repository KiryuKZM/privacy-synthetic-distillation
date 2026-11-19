"""Run the same frozen UST prompts on vLLM BF16, FP8, and GPTQ."""
import argparse
from pathlib import Path
from time import perf_counter

from ust_compact_prompt_20261111 import OUT, QUANT, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("bf16", "bf16-cache", "fp8", "gptq"))
    parser.add_argument("--prompts", nargs="+", choices=("full8", "dedup8", "table8"),
                        default=("full8", "table8"))
    parser.add_argument("--output-root", type=Path, default=OUT / "vllm")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    root = args.output_root / args.mode
    source = OUT / "exports/gptq" if args.mode == "gptq" else QUANT / "exports/bf16"
    use_cache = args.mode == "bf16-cache"
    options = {
        "model": str(source), "tokenizer": str(QUANT / "exports/bf16"),
        "dtype": "bfloat16", "max_model_len": 8192, "max_num_seqs": 1,
        "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.32,
        "enforce_eager": True, "enable_prefix_caching": use_cache,
        "language_model_only": True, "disable_log_stats": False,
    }
    if args.mode == "fp8":
        options["quantization"] = "fp8_per_tensor"
    write(root / "engine-options.json", options)
    engine = LLM(**options)
    tokenizer = AutoTokenizer.from_pretrained(options["tokenizer"], local_files_only=True)
    stops = [tokenizer.convert_tokens_to_ids(s) for s in ("<|im_end|>", "<|endoftext|>")]
    sampling = SamplingParams(temperature=0, max_tokens=1024, stop_token_ids=stops,
                              detokenize=False)

    def generate(ids, params):
        begin = perf_counter()
        response, = engine.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
        elapsed = perf_counter() - begin
        assert list(response.prompt_token_ids) == ids
        return response, elapsed

    data = read(OUT / "prompt-tokens.local.json")
    for prompt in args.prompts:
        rows = data[prompt]
        # Warm kernels with a short, different prompt. In cache mode the first
        # real request seeds the cache and is measured as cold.
        generate([tokenizer.bos_token_id or rows[0]["ids"][0]] * 32,
                 SamplingParams(temperature=0, max_tokens=1, detokenize=False))
        for index, row in enumerate(rows):
            response, elapsed = generate(row["ids"], sampling)
            completion, = response.outputs
            tokens = list(completion.token_ids)
            metrics = response.metrics
            cached = getattr(response, "num_cached_tokens", None)
            if cached is None and metrics is not None:
                cached = getattr(metrics, "num_cached_tokens", None)
            output = {
                "id": row["id"], "prompt_tokens": len(row["ids"]), "cached_tokens": cached,
                "response_ids": tokens, "raw_generated": tokenizer.decode(tokens, skip_special_tokens=False),
                "finish_reason": completion.finish_reason, "request_wall_seconds": elapsed,
                "ttft_seconds": getattr(metrics, "first_token_latency", None),
            }
            write(root / prompt / "raw" / f"{index:04d}.json", output)
            write(root / "live.json", {"prompt": prompt, "closed": index + 1})
    write(root / "terminal.json", {"status": "COMPLETE", "mode": args.mode})


if __name__ == "__main__":
    main()
