"""Qwen3.5-2B student interface.

Distillation training accepts only an approved teacher-label manifest.
Adult ground-truth labels and fabricated labels are refused. sealed_test
cannot select a setting or serve as the development report.
"""

from __future__ import annotations

import hashlib
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

from privacy_synth.adult import COLUMNS

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
FINAL_LABELS = ("<=50K", ">50K")
APPROVED_LABEL_SOURCE = "approved_teacher"
REJECTED_LABEL_SOURCES = (
    "adult_ground_truth",
    "fabricated",
    "local_income",
    "placeholder",
    "sealed_test",
)


class StudentDataError(ValueError):
    """Raised when a student run would use a forbidden label or split."""


@dataclass(frozen=True)
class StudentTrainConfig:
    model_id: str = MODEL_ID
    revision: str = REVISION
    seed: int = 20261001
    sample_order: str = "record_fp_ascending"
    max_steps: int = 200
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 1e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    max_seq_len: int = 512
    precision: str = "bf16"
    supervision_format: str = "teacher_final_label_only"
    inference_max_new_tokens: int = 256
    student_inference_mode: str = "unfrozen_pending_local_dev_comparison"
    teacher_thinking: str = "unfrozen"
    eval_split: str = "local_dev"

    def validate(self) -> None:
        if self.eval_split != "local_dev":
            raise StudentDataError("student development report uses local_dev only")
        if self.student_inference_mode == "thinking_off":
            raise StudentDataError("student inference mode is not preset to thinking off")
        if self.supervision_format != "teacher_final_label_only":
            raise StudentDataError("first-round supervision is the teacher final label only")
        if self.max_seq_len <= 0 or self.inference_max_new_tokens <= 0:
            raise StudentDataError("sequence and generation caps must be positive")


@dataclass(frozen=True)
class TeacherLabel:
    record_fp: str
    final_label: str
    source: str
    teacher_thinking_mode: str

    def validate(self) -> None:
        if self.source != APPROVED_LABEL_SOURCE:
            raise StudentDataError(
                f"label source {self.source!r} is not an approved teacher label"
            )
        if self.source in REJECTED_LABEL_SOURCES:
            raise StudentDataError(f"refusing label source {self.source}")
        if self.final_label not in FINAL_LABELS:
            raise StudentDataError(f"teacher final label {self.final_label!r} is not binary income")
        if not self.teacher_thinking_mode:
            raise StudentDataError("teacher thinking mode must be recorded on the label")


def load_student_config(path: Path) -> StudentTrainConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    student = raw["student"]
    train = raw["train"]
    inference = raw["inference"]
    teacher = raw["teacher"]
    evaluation = raw["eval"]
    config = StudentTrainConfig(
        model_id=student["model_id"],
        revision=student["revision"],
        seed=int(train["seed"]),
        sample_order=train["sample_order"],
        max_steps=int(train["max_steps"]),
        batch_size=int(train["batch_size"]),
        grad_accum=int(train["grad_accum"]),
        learning_rate=float(train["learning_rate"]),
        lora_r=int(train["lora_r"]),
        lora_alpha=int(train["lora_alpha"]),
        lora_dropout=float(train["lora_dropout"]),
        max_seq_len=int(train["max_seq_len"]),
        precision=student["precision"],
        supervision_format=train["supervision_format"],
        inference_max_new_tokens=int(inference["max_new_tokens"]),
        student_inference_mode=inference["student_inference_mode"],
        teacher_thinking=teacher["thinking"],
        eval_split=evaluation["allowed_split"],
    )
    config.validate()
    if "sealed_test" not in evaluation["forbidden_splits"]:
        raise StudentDataError("sealed_test must stay a forbidden eval split")
    if "arms" in raw and tuple(raw["arms"]["names"]) != SHARED_ARMS:
        raise StudentDataError("arm list does not match the shared four-arm contrast")
    return config


def order_record_fps(record_fps: list[str], sample_order: str = "record_fp_ascending") -> list[str]:
    if sample_order != "record_fp_ascending":
        raise StudentDataError(f"unknown sample order {sample_order}")
    return sorted(record_fps)


def render_row_prompt(mapping: dict[str, str]) -> str:
    """Text prompt for one Adult-like row. Income is omitted on purpose."""
    if "income" in mapping:
        raise StudentDataError("student prompts must not include the income field")
    missing = [column for column in COLUMNS if column != "income" and column not in mapping]
    if missing:
        raise StudentDataError(f"prompt row missing fields: {missing}")
    lines = [f"{column}: {mapping[column]}" for column in COLUMNS if column != "income"]
    body = "\n".join(lines)
    return (
        "Predict whether this census-style adult record has yearly income above 50,000 USD. "
        "Reply with exactly one of: <=50K or >50K.\n"
        f"{body}"
    )


def supervision_target(label: TeacherLabel) -> str:
    """Supervised text is the teacher final label, not a thinking trace."""
    label.validate()
    return label.final_label


def parse_final_label(text: str) -> str | None:
    stripped = text.strip()
    if stripped in FINAL_LABELS:
        return stripped
    for label in FINAL_LABELS:
        if stripped.endswith(label) and label in stripped.split():
            return label
    return None


def dev_report(
    *,
    split: str,
    labels: list[TeacherLabel],
    predictions: list[str],
    truncated: list[bool],
    train_seconds: float | None,
    peak_memory_mib: float | None,
    max_new_tokens: int,
) -> dict[str, object]:
    """Development metrics. Refuses sealed_test and unapproved labels."""
    if split != "local_dev":
        raise StudentDataError("development report is local_dev only; sealed_test is forbidden")
    if not labels:
        raise StudentDataError("no approved teacher labels; distillation metrics are not available")
    if len(labels) != len(predictions) or len(labels) != len(truncated):
        raise StudentDataError("label, prediction, and truncation lengths differ")
    for label in labels:
        label.validate()
    valid = 0
    correct = 0
    n_trunc = 0
    for label, prediction, was_truncated in zip(labels, predictions, truncated, strict=True):
        parsed = parse_final_label(prediction)
        if was_truncated:
            n_trunc += 1
        if parsed is None:
            continue
        valid += 1
        if parsed == label.final_label:
            correct += 1
    n = len(labels)
    return {
        "split": split,
        "n": n,
        "valid_output_rate": valid / n,
        "invalid_output_rate": (n - valid) / n,
        "accuracy_on_valid": (correct / valid) if valid else None,
        "truncation_rate": n_trunc / n,
        "max_new_tokens": max_new_tokens,
        "train_seconds": train_seconds,
        "peak_memory_mib": peak_memory_mib,
        "used_sealed_test": False,
    }


def synthetic_example_id(record_fp: str, arm: str) -> str:
    material = f"{arm}:{record_fp}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


# One generator/utility-filter contrast. All four arms use the same student settings.
SHARED_ARMS: tuple[str, ...] = (
    "generic_utility_off",
    "generic_utility_on",
    "task_aware_utility_off",
    "task_aware_utility_on",
)


def assert_main_experiment_ready(config: StudentTrainConfig) -> None:
    """Block a four-arm run until inference mode is a concrete shared setting."""
    config.validate()
    if config.student_inference_mode == "unfrozen_pending_local_dev_comparison":
        raise StudentDataError(
            "student inference mode is still unfrozen; main experiment training is blocked"
        )


def frozen_train_settings(config: StudentTrainConfig) -> dict[str, object]:
    """Settings that must be identical across the four arms before a main run."""
    config.validate()
    return {
        "model_id": config.model_id,
        "revision": config.revision,
        "seed": config.seed,
        "sample_order": config.sample_order,
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "learning_rate": config.learning_rate,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "max_seq_len": config.max_seq_len,
        "precision": config.precision,
        "supervision_format": config.supervision_format,
        "inference_max_new_tokens": config.inference_max_new_tokens,
        "student_inference_mode": config.student_inference_mode,
        "teacher_thinking": config.teacher_thinking,
        "eval_split": config.eval_split,
    }


@dataclass(frozen=True)
class DistillExample:
    example_id: str
    record_fp: str
    arm: str
    prompt: str
    supervision: str
    teacher_thinking_mode: str
    student_inference_mode: str


def _feature_mapping(row: dict[str, str]) -> dict[str, str]:
    if "income" in row:
        raise StudentDataError("synthetic student inputs must not include income")
    return {column: row[column] for column in COLUMNS if column != "income"}


def records_to_distill_examples(
    records: list[dict[str, object]],
    *,
    arm: str,
    config: StudentTrainConfig,
) -> list[DistillExample]:
    """Turn an approved-teacher manifest into ordered training examples."""
    rows: list[dict[str, str]] = []
    labels: list[TeacherLabel] = []
    for record in records:
        if record.get("split") == "sealed_test":
            raise StudentDataError("distillation manifest must not name sealed_test")
        if record.get("arm") not in (None, arm):
            raise StudentDataError("manifest arm does not match the requested arm")
        row = dict(record["row"])  # type: ignore[arg-type]
        row["record_fp"] = str(record["record_fp"])
        rows.append(row)
        labels.append(
            TeacherLabel(
                record_fp=str(record["record_fp"]),
                final_label=str(record["final_label"]),
                source=str(record["source"]),
                teacher_thinking_mode=str(record["teacher_thinking_mode"]),
            )
        )
    return build_distill_examples(rows, labels, arm=arm, config=config)


def build_distill_examples(
    rows: list[dict[str, str]],
    labels: list[TeacherLabel],
    *,
    arm: str,
    config: StudentTrainConfig,
) -> list[DistillExample]:
    """Pair synthetic inputs with teacher final labels. Does not invent labels."""
    if arm not in SHARED_ARMS:
        raise StudentDataError(f"unknown arm {arm}")
    config.validate()
    if not labels:
        raise StudentDataError("no approved teacher labels; distillation training is not available")
    by_fp: dict[str, dict[str, str]] = {}
    for row in rows:
        record_fp = row.get("record_fp", "")
        if not record_fp:
            raise StudentDataError("synthetic row is missing record_fp")
        if record_fp in by_fp:
            raise StudentDataError(f"duplicate synthetic row {record_fp}")
        by_fp[record_fp] = row
    label_by: dict[str, TeacherLabel] = {}
    for label in labels:
        label.validate()
        if label.record_fp in label_by:
            raise StudentDataError(f"duplicate teacher label {label.record_fp}")
        if label.record_fp not in by_fp:
            raise StudentDataError(f"teacher label {label.record_fp} has no synthetic row")
        label_by[label.record_fp] = label
    ordered = order_record_fps(list(label_by), config.sample_order)
    examples: list[DistillExample] = []
    for record_fp in ordered:
        label = label_by[record_fp]
        prompt = render_row_prompt(_feature_mapping(by_fp[record_fp]))
        examples.append(
            DistillExample(
                example_id=synthetic_example_id(record_fp, arm),
                record_fp=record_fp,
                arm=arm,
                prompt=prompt,
                supervision=supervision_target(label),
                teacher_thinking_mode=label.teacher_thinking_mode,
                student_inference_mode=config.student_inference_mode,
            )
        )
    return examples


def equal_label_schedule(
    examples: list[DistillExample],
    *,
    steps: int,
    grad_accum: int,
    seed: int,
) -> list[DistillExample]:
    """Repeat approved labels so each optimizer step sees both classes equally.

    Sampling is with replacement inside each class. No label is invented.
    """
    if steps <= 0:
        raise StudentDataError("equal label schedule needs a positive step count")
    if grad_accum <= 0 or grad_accum % 2 != 0:
        raise StudentDataError("equal label schedule needs a positive even grad accumulation")
    pools = {
        label: [example for example in examples if example.supervision == label]
        for label in FINAL_LABELS
    }
    if any(example.supervision not in FINAL_LABELS for example in examples):
        raise StudentDataError("equal label schedule received a non-binary supervision target")
    if not pools["<=50K"] or not pools[">50K"]:
        raise StudentDataError("equal label schedule needs both approved final labels")
    rng = random.Random(seed)
    half = grad_accum // 2
    scheduled: list[DistillExample] = []
    for step in range(steps):
        label = "<=50K" if (step % grad_accum) < half else ">50K"
        scheduled.append(rng.choice(pools[label]))
    return scheduled


def pack_supervised_ids(
    prompt_ids: list[int],
    target_ids: list[int],
    max_seq_len: int,
    ignore_index: int = -100,
) -> dict[str, object]:
    """Keep the teacher final-label tokens. Shorten the prompt if the cap requires it."""
    if max_seq_len <= 0:
        raise StudentDataError("sequence cap must be positive")
    if not target_ids:
        raise StudentDataError("supervision target tokenized to empty")
    if len(target_ids) > max_seq_len:
        raise StudentDataError("supervision target does not fit in max_seq_len")
    room = max_seq_len - len(target_ids)
    truncated = len(prompt_ids) > room
    kept_prompt = prompt_ids[-room:] if truncated else list(prompt_ids)
    return {
        "input_ids": kept_prompt + list(target_ids),
        "labels": [ignore_index] * len(kept_prompt) + list(target_ids),
        "truncated": truncated,
        "prompt_tokens_kept": len(kept_prompt),
        "max_seq_len": max_seq_len,
    }
