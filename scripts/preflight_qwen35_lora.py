#!/usr/bin/env python3
"""Bounded BF16 LoRA preflight for Qwen/Qwen3.5-2B. Text path only.

Uses a toy sentence, not Adult rows and not teacher labels.
Writes a JSON and Markdown report. Does not start Docker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Torch 2.14 routes a RoPE outer-product matmul through a Triton kernel. Triton
# then compiles a CUDA shim that includes Python.h, which this machine does not
# have. The eager matmul is the same operation. Do not treat this as QLoRA.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
TOY_USER = "Reply with the single token OK."
LORA_STEPS = 3
SEQ_CAP = 64
MAX_NEW_TOKENS = 64
SEED = 20261001


def gpu_query() -> dict[str, str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    used, free, util, name = (proc.stdout.strip().split(", ") + ["", "", "", ""])[:4]
    return {
        "name": name,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "utilization_gpu": util,
        "returncode": str(proc.returncode),
    }


def _finite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all().item())


def _text_lora_modules(model) -> list[str]:
    """Full module names on the text path. Leaf suffixes would also match vision."""
    import torch

    found: list[str] = []
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for name, module in model.named_modules():
        if "visual" in name:
            continue
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in wanted:
            found.append(name)
    return sorted(found)


def _kernel_status() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for name in ("causal_conv1d", "fla"):
        try:
            __import__(name)
        except Exception:  # noqa: BLE001 — absence is a recorded fact, not a failure
            status[name] = False
        else:
            status[name] = True
    return status


def _freeze_vision(model) -> dict[str, int]:
    frozen = 0
    trainable_vision = 0
    vision_params = 0
    for name, param in model.named_parameters():
        if "visual" in name:
            vision_params += param.numel()
            param.requires_grad = False
            frozen += param.numel()
            if param.requires_grad:
                trainable_vision += param.numel()
    for name, module in model.named_modules():
        if "visual" in name:
            module.eval()
    return {
        "vision_parameter_count": vision_params,
        "vision_parameters_frozen": frozen,
        "vision_parameters_still_trainable": trainable_vision,
    }


def _load_model(torch):
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    errors = []
    model = None
    loaded_as = None
    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
    ):
        try:
            import transformers

            cls = getattr(transformers, class_name)
        except AttributeError as exc:
            errors.append(f"{class_name}: {exc}")
            continue
        try:
            model = cls.from_pretrained(
                MODEL_ID,
                revision=REVISION,
                dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
            )
            loaded_as = class_name
            break
        except Exception as exc:  # noqa: BLE001 — record each loader and keep going
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError("no text-capable loader succeeded: " + " | ".join(errors))
    return config, tokenizer, model, loaded_as, errors


def _text_batch(tokenizer, torch, text: str):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_CAP)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded


def _forward_loss(model, batch):
    try:
        out = model(**batch)
        return out.loss, "full_model"
    except Exception as exc:  # noqa: BLE001
        language = None
        for attr in ("language_model", "model"):
            candidate = getattr(model, attr, None)
            if candidate is not None and hasattr(candidate, "language_model"):
                language = candidate.language_model
                break
            if attr == "language_model":
                language = candidate
        if language is None:
            raise
        out = language(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        return out.loss, f"language_model_after_{type(exc).__name__}"


def _language_model(model):
    current = model
    seen: set[int] = set()
    for _ in range(6):
        if id(current) in seen:
            break
        seen.add(id(current))
        for attr in ("language_model", "model", "base_model"):
            candidate = getattr(current, attr, None)
            if candidate is None or candidate is current:
                continue
            nested = getattr(candidate, "language_model", None)
            if nested is not None and hasattr(nested, "generate"):
                return nested
            if attr == "language_model" and hasattr(candidate, "generate"):
                return candidate
        nxt = getattr(current, "base_model", None)
        if nxt is None or nxt is current:
            nxt = getattr(current, "model", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return None


def _generate_once(module, tokenizer, torch, enable_thinking: bool, path: str) -> dict[str, object]:
    messages = [{"role": "user", "content": TOY_USER}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = int(inputs["input_ids"].shape[-1])
    module.eval()
    with torch.no_grad():
        output = module.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_len = int(output.shape[-1] - prompt_len)
    text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=False)
    return {
        "enable_thinking": enable_thinking,
        "generate_path": path,
        "prompt_tokens": prompt_len,
        "max_new_tokens": MAX_NEW_TOKENS,
        "new_tokens": new_len,
        "truncated": new_len >= MAX_NEW_TOKENS,
        "contains_think_tag": "<think>" in text or "</think>" in text,
        "text": text,
        "finite_ids": bool(torch.isfinite(output.float()).all().item())
        if output.dtype.is_floating_point
        else True,
    }


def _generate(model, tokenizer, torch, enable_thinking: bool) -> dict[str, object]:
    errors = []
    try:
        return _generate_once(model, tokenizer, torch, enable_thinking, "loaded_model")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"loaded_model: {type(exc).__name__}: {exc}")
    language = _language_model(model)
    if language is None:
        raise RuntimeError(" | ".join(errors))
    try:
        result = _generate_once(language, tokenizer, torch, enable_thinking, "language_model")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"language_model: {type(exc).__name__}: {exc}")
        raise RuntimeError(" | ".join(errors)) from exc
    result["fallback_errors"] = errors
    return result


def run() -> dict[str, object]:
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    started = time.perf_counter()
    gpu_before = gpu_query()
    free = float(gpu_before["memory_free_mib"] or 0)
    used = float(gpu_before["memory_used_mib"] or 0)
    if used > 2048 or free < 10000:
        raise RuntimeError(
            f"GPU is occupied or short of memory: used={used} MiB free={free} MiB"
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_t0 = time.perf_counter()
    config, tokenizer, model, loaded_as, load_errors = _load_model(torch)
    load_seconds = time.perf_counter() - load_t0
    vision = _freeze_vision(model)
    arch = getattr(config, "architectures", None)
    model_type = getattr(config, "model_type", None)
    has_vision = getattr(config, "vision_config", None) is not None
    targets = _text_lora_modules(model)
    if not targets:
        raise RuntimeError("no text-path Linear q/k/v/o projections found for LoRA")
    if any("visual" in name for name in targets):
        raise RuntimeError("LoRA target list includes a visual module")
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        exclude_modules=r".*visual.*",
    )
    model = get_peft_model(model, lora)
    # Vision stays frozen after PEFT wrapping.
    vision_after = _freeze_vision(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    batch = _text_batch(tokenizer, torch, TOY_USER)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    losses = []
    grad_finite = []
    step_seconds = []
    updated = False
    watched = next(p for p in model.parameters() if p.requires_grad)
    before = watched.detach().float().clone()
    model.train()
    for _ in range(LORA_STEPS):
        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, forward_path = _forward_loss(model, batch)
        if not _finite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        loss.backward()
        grads_ok = True
        for param in model.parameters():
            if param.grad is not None and not _finite(param.grad):
                grads_ok = False
                break
        grad_finite.append(grads_ok)
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))
        step_seconds.append(time.perf_counter() - step_t0)
    after = watched.detach().float()
    updated = bool((after - before).abs().sum().item() > 0)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    generations = []
    gen_errors = []
    for mode in (False, True):
        try:
            generations.append(_generate(model, tokenizer, torch, mode))
        except Exception as exc:  # noqa: BLE001
            gen_errors.append({"enable_thinking": mode, "error": f"{type(exc).__name__}: {exc}"})
    gen_error = " | ".join(item["error"] for item in gen_errors) if gen_errors else None
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "loaded_as": loaded_as,
        "load_errors": load_errors,
        "architectures": arch,
        "model_type": model_type,
        "has_vision_config": has_vision,
        "vision": vision,
        "vision_after_lora": vision_after,
        "text_path": "no image or video tensors; visual parameters frozen",
        "forward_path": forward_path,
        "lora_target_modules": targets,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "losses": losses,
        "grad_finite": grad_finite,
        "lora_weight_changed": updated,
        "step_seconds": step_seconds,
        "load_seconds": load_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_mib": peak,
        "gpu_before": gpu_before,
        "generations": generations,
        "generation_errors": gen_errors,
        "generation_error": gen_error,
        "optional_kernels": _kernel_status(),
        "precision": "bf16",
        "qlora": False,
        "seed": SEED,
        "lora_steps": LORA_STEPS,
        "seq_cap": SEQ_CAP,
        "toy_user": TOY_USER,
        "torch_disable_native_jit": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Qwen3.5-2B BF16 LoRA preflight",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Model: `{report.get('model_id')}` @ `{report.get('revision')}`",
        f"- Loader: `{report.get('loaded_as')}`",
        f"- model_type: `{report.get('model_type')}`",
        f"- Vision config present: {report.get('has_vision_config')}",
        f"- Text path: {report.get('text_path')}",
        f"- Forward path: `{report.get('forward_path')}`",
        f"- Precision: `{report.get('precision')}`",
        f"- QLoRA used: {report.get('qlora')}",
        f"- Optional kernels: `{report.get('optional_kernels')}`",
        f"- LoRA targets: `{report.get('lora_target_modules')}`",
        f"- Trainable / total parameters: {report.get('trainable_parameters')} / {report.get('total_parameters')}",
        f"- Losses: `{report.get('losses')}`",
        f"- Gradients finite: `{report.get('grad_finite')}`",
        f"- LoRA weight changed: {report.get('lora_weight_changed')}",
        f"- Peak allocated memory MiB: {report.get('peak_memory_mib')}",
        f"- Load seconds: {report.get('load_seconds')}",
        f"- Step seconds: `{report.get('step_seconds')}`",
        f"- Wall seconds: {report.get('wall_seconds')}",
        f"- TORCH_DISABLE_NATIVE_JIT: `{report.get('torch_disable_native_jit')}`",
        f"- torch: `{report.get('torch')}` cuda `{report.get('cuda')}`",
        f"- GPU: `{report.get('gpu_name')}`",
        "",
        "## Vision freeze",
        "",
        f"`{report.get('vision')}`",
        "",
        "## Generations",
        "",
    ]
    for item in report.get("generations") or []:
        lines.append(
            f"- enable_thinking={item.get('enable_thinking')} path={item.get('generate_path')} "
            f"prompt_tokens={item.get('prompt_tokens')} new_tokens={item.get('new_tokens')} "
            f"truncated={item.get('truncated')} think_tag={item.get('contains_think_tag')}"
        )
        lines.append("")
        lines.append("```")
        lines.append(str(item.get("text", ""))[:2000])
        lines.append("```")
        lines.append("")
    if report.get("generation_error"):
        lines.append(f"Generation error: `{report['generation_error']}`")
    if report.get("error"):
        lines.extend(["", "## Error", "", "```", str(report["error"]), "```"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "reports" / "2026-10-01-qwen35-lora-preflight.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "reports" / "2026-10-01-qwen35-lora-preflight.md",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "2026-10-01-qwen35-lora",
    )
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "hf-home"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        report = run()
        code = 0 if report.get("generation_error") is None else 2
    except Exception:
        report = {
            "status": "failed",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "qlora": False,
            "error": traceback.format_exc(),
            "gpu_before": gpu_query(),
        }
        code = 1
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.md_out.read_text(encoding="utf-8"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
