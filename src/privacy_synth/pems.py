"""PeMS-BAY pilot protocol.

The held-out cut, freeway labels, and teacher payload live here so a script
cannot send a real series or train the student on gold. This module does not
load a generator or a language model.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from privacy_synth.privacy_filter import refuse_sealed_path

PARTITION_ID = "pems-bay-cited-250x200-last10"
WINDOW_LENGTH = 200
N_WINDOWS = 250
TEST_WINDOW_COUNT = 10
TRAIN_WINDOW_COUNT = N_WINDOWS - TEST_WINDOW_COUNT
SERIES_STEPS_USED = WINDOW_LENGTH * N_WINDOWS
FREEWAY_LABELS: tuple[str, ...] = (
    "17",
    "85",
    "87",
    "101",
    "237",
    "280",
    "680",
    "880",
)
GENERATOR_COMMIT = "566307e6cf2d8095e58de4c6e3a6ae965b69b5b5"
TIMESTEPS = 100
MAX_EPOCHS = 8
WALL_SECONDS = 45 * 60
GENERATOR_BATCH = 64
GENERATOR_BATCH_IF_OOM = 32
SAMPLE_COUNT = 256
TRAIN_SEED = 20261005
SAMPLE_SEED = 20261009
TEACHER_MODEL = "deepseek-v4-flash"
TEACHER_PROVIDER = "deepseek"
TEACHER_MAX_TOKENS = 512
TEACHER_MAX_TOKENS_IF_FIRST_EIGHT_EMPTY = 1024
TEACHER_PLANNED_CALLS = SAMPLE_COUNT
TEACHER_RETRY_CEILING = SAMPLE_COUNT * 2
TEACHER_USD_CAP = 500.0
# Peak cache-miss Flash rates, USD per token. The alias is billed as V4.1 Flash.
INPUT_USD_PER_TOKEN_PEAK = 0.30 / 1_000_000
OUTPUT_USD_PER_TOKEN_PEAK = 1.20 / 1_000_000
ESTIMATED_INPUT_TOKENS_PER_CALL = 1500
STUDENT_MAX_NEW_TOKENS = 128
STUDENT_MAX_SEQ_LEN = 2048
STUDENT_MAX_STEPS = 200
STUDENT_SEED = 20261005
APPROVED_LABEL_SOURCE = "approved_teacher"
FORBIDDEN_RECORD_KEYS = (
    "sensor_id",
    "freeway",
    "gold",
    "gold_freeway",
    "generator_class",
    "latitude",
    "longitude",
    "neighbor",
    "neighbors",
)


class PemsProtocolError(ValueError):
    """Raised when a PeMS teacher or student path would break the pilot rules."""


def window_roles() -> dict[str, object]:
    """Cited cut: 250 non-overlapping windows, last 10 held out."""
    train = list(range(TRAIN_WINDOW_COUNT))
    test = list(range(TRAIN_WINDOW_COUNT, N_WINDOWS))
    if set(train) & set(test):
        raise PemsProtocolError("train and test windows overlap")
    if len(train) + len(test) != N_WINDOWS:
        raise PemsProtocolError("window roles do not cover the cited cut")
    return {
        "partition_id": PARTITION_ID,
        "window_length": WINDOW_LENGTH,
        "series_steps_used": SERIES_STEPS_USED,
        "train_windows": train,
        "test_windows": test,
    }


def question_id(window_index: int, sensor_id: str) -> str:
    return f"w{window_index:03d}-s{sensor_id}"


def question_list_sha256(question_ids: list[str]) -> str:
    material = "\n".join(question_ids).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def render_speed_prompt(speeds: list[float] | tuple[float, ...]) -> str:
    """One speed curve. The freeway of this curve is not in the text."""
    if len(speeds) != WINDOW_LENGTH:
        raise PemsProtocolError(
            f"a pilot question has {WINDOW_LENGTH} speeds, got {len(speeds)}"
        )
    if any(not _finite(value) for value in speeds):
        raise PemsProtocolError("speed prompt refuses a non-finite value")
    body = ",".join(f"{float(value):.3f}" for value in speeds)
    allowed = ", ".join(FREEWAY_LABELS)
    return (
        "Classify this Bay Area traffic speed series by freeway. "
        f"Reply with exactly one of: {allowed}.\n"
        f"speeds: {body}"
    )


def build_teacher_payload(record: dict[str, object], path: Path | None = None) -> dict[str, object]:
    """Synthetic speed window only. Real series and sealed paths are refused."""
    if path is not None:
        refuse_sealed_path(Path(path))
        name = Path(path).name
        if "test_questions" in name or name.startswith("test"):
            raise PemsProtocolError(f"teacher send must not read {path}")
    origin = record.get("origin")
    if origin != "synthetic":
        raise PemsProtocolError("teacher payload accepts synthetic windows only")
    role = str(record.get("role") or "")
    if role in {"pems_test", "sealed_test", "generator_train", "real"}:
        raise PemsProtocolError(f"teacher payload refuses role {role}")
    present = [key for key in FORBIDDEN_RECORD_KEYS if key in record]
    if present:
        raise PemsProtocolError(
            "synthetic teacher record must not carry a location, sensor, or gold field: "
            + ", ".join(present)
        )
    window_id = str(record["window_id"])
    if not window_id.startswith("syn-"):
        raise PemsProtocolError("teacher window id must be a synthetic id")
    speeds = record["speeds"]
    if not isinstance(speeds, (list, tuple)):
        raise PemsProtocolError("teacher speeds must be a list")
    prompt = render_speed_prompt([float(value) for value in speeds])
    payload = {
        "window_id": window_id,
        "prompt": prompt,
        "origin": "synthetic",
    }
    leaked = [key for key in FORBIDDEN_RECORD_KEYS if key in payload]
    if leaked:
        raise PemsProtocolError(f"payload leaked {leaked}")
    return payload


def finished_class_label(text: str) -> str | None:
    """One whole line equal to one freeway id. Padding after end-of-text is ignored."""
    cut = text.split("<|endoftext|>", 1)[0]
    cut = cut.split("<|im_end|>", 1)[0]
    lines = [line.strip() for line in cut.splitlines() if line.strip()]
    label_lines = [line for line in lines if line in FREEWAY_LABELS]
    if not label_lines:
        return None
    chosen = label_lines[-1]
    if lines[-1] != chosen:
        return None
    if any(line != chosen for line in label_lines):
        return None
    return chosen


@dataclass(frozen=True)
class PemsExample:
    window_id: str
    prompt: str
    supervision: str


def pems_distill_examples(records: list[dict[str, object]]) -> list[PemsExample]:
    """Student target is the teacher final label. Gold and generator class are refused."""
    if not records:
        raise PemsProtocolError("no approved teacher labels")
    examples: list[PemsExample] = []
    for record in records:
        path = record.get("path")
        if path is not None:
            refuse_sealed_path(Path(str(path)))
        if record.get("source") != APPROVED_LABEL_SOURCE:
            raise PemsProtocolError("distillation accepts approved teacher labels only")
        blocked = [key for key in ("gold", "gold_freeway", "generator_class") if key in record]
        if blocked:
            raise PemsProtocolError(
                "distillation target cannot be a generator class or a gold label"
            )
        label = record.get("final_label")
        if label not in FREEWAY_LABELS:
            raise PemsProtocolError(f"teacher final label {label!r} is not a freeway id")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.startswith("Classify this Bay Area"):
            raise PemsProtocolError("distillation prompt is not a speed-series question")
        if "\nsensor_id:" in prompt or prompt.startswith("sensor_id:"):
            raise PemsProtocolError("distillation prompt contains a sensor id field")
        examples.append(
            PemsExample(
                window_id=str(record["window_id"]),
                prompt=prompt,
                supervision=str(label),
            )
        )
    return examples


def score_heldout(
    question_ids: list[str],
    gold_by_id: dict[str, str],
    texts: list[str],
) -> dict[str, object]:
    """Count finished legal labels and how many of those match held-out gold."""
    if len(question_ids) != len(texts):
        raise PemsProtocolError("question list and generations differ in length")
    if len(set(question_ids)) != len(question_ids):
        raise PemsProtocolError("question list has a duplicate id")
    finished = 0
    gold_match = 0
    for question_id_value, text in zip(question_ids, texts, strict=True):
        if question_id_value not in gold_by_id:
            raise PemsProtocolError(f"missing gold for {question_id_value}")
        gold = gold_by_id[question_id_value]
        if gold not in FREEWAY_LABELS:
            raise PemsProtocolError(f"gold class {gold!r} is outside the frozen label set")
        parsed = finished_class_label(text)
        if parsed is None:
            continue
        finished += 1
        if parsed == gold:
            gold_match += 1
    return {
        "finished_label_count": finished,
        "gold_match_count": gold_match,
        "n": len(question_ids),
        "supervision": "teacher_final_label",
    }


def build_comparison_report(
    *,
    question_ids: list[str],
    gold_by_id: dict[str, str],
    untuned_texts: list[str],
    distilled_texts: list[str],
) -> dict[str, object]:
    """Same held-out list for both models. Does not call either model."""
    untuned = score_heldout(question_ids, gold_by_id, untuned_texts)
    distilled = score_heldout(question_ids, gold_by_id, distilled_texts)
    return {
        "test_partition_id": PARTITION_ID,
        "question_ids": list(question_ids),
        "question_list_sha256": question_list_sha256(question_ids),
        "n_questions": len(question_ids),
        "supervision": "teacher_final_label",
        "untuned": {
            "finished_label_count": untuned["finished_label_count"],
            "gold_match_count": untuned["gold_match_count"],
        },
        "teacher_distilled": {
            "finished_label_count": distilled["finished_label_count"],
            "gold_match_count": distilled["gold_match_count"],
        },
        "private": False,
        "feasible": False,
        "state_of_the_art": False,
    }


def teacher_estimate_document(written_unix: int) -> dict[str, object]:
    """Pre-send bill. Peak rates are the conservative cap, not an invoice."""
    planned = TEACHER_PLANNED_CALLS
    ceiling = TEACHER_RETRY_CEILING
    in_tok = ESTIMATED_INPUT_TOKENS_PER_CALL
    out_tok = TEACHER_MAX_TOKENS
    planned_usd = planned * in_tok * INPUT_USD_PER_TOKEN_PEAK + planned * out_tok * OUTPUT_USD_PER_TOKEN_PEAK
    ceiling_usd = ceiling * in_tok * INPUT_USD_PER_TOKEN_PEAK + ceiling * out_tok * OUTPUT_USD_PER_TOKEN_PEAK
    if ceiling_usd > TEACHER_USD_CAP:
        raise PemsProtocolError("retry ceiling exceeds the 500 USD cap")
    return {
        "written_unix": written_unix,
        "provider": TEACHER_PROVIDER,
        "model": TEACHER_MODEL,
        "planned_calls": planned,
        "retry_ceiling_calls": ceiling,
        "estimated_input_tokens": planned * in_tok,
        "estimated_output_tokens": planned * out_tok,
        "estimated_input_tokens_per_call": in_tok,
        "max_output_tokens_per_call": out_tok,
        "estimated_usd_planned_peak": round(planned_usd, 6),
        "estimated_usd_retry_ceiling_peak": round(ceiling_usd, 6),
        "usd_cap": TEACHER_USD_CAP,
        "payload": (
            "synthetic Diffusion-TS speed windows only, 200 numbers each; "
            "no sensor id, freeway label, neighbor, or real PeMS series"
        ),
        "cheaper_unused_alternative": (
            "score the untuned local student and do not call the teacher, cost 0 USD"
        ),
        "price_basis": (
            "peak cache-miss Flash rates 0.30 USD per 1M input tokens and "
            "1.20 USD per 1M output tokens"
        ),
    }


def teacher_request_body(model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    """Chat body for one synthetic window.

    The API default keeps thinking on and spends the completion cap before
    `message.content` exists. This pilot turns that switch off so the final
    freeway id is written in content. Reasoning text is not part of the body
    and is not a label.
    """
    if model != TEACHER_MODEL:
        raise PemsProtocolError(f"teacher model {model} is not the frozen pilot model")
    if max_tokens not in {TEACHER_MAX_TOKENS, TEACHER_MAX_TOKENS_IF_FIRST_EIGHT_EMPTY}:
        raise PemsProtocolError(f"teacher completion cap {max_tokens} is not frozen")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "thinking": {"type": "disabled"},
    }


def peak_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * INPUT_USD_PER_TOKEN_PEAK
        + completion_tokens * OUTPUT_USD_PER_TOKEN_PEAK
    )


def _finite(value: float) -> bool:
    number = float(value)
    return number == number and number not in {float("inf"), float("-inf")}


# Speed-forecast run. These names do not change the freeway-classification pilot.
FORECAST_PARTITION_ID = "pems-bay-chrono-70-10-20-h12"
FORECAST_N_STEPS = 52116
FORECAST_N_SENSORS = 325
FORECAST_HISTORY = 12
FORECAST_HORIZON = 12
FORECAST_SERIES_LENGTH = FORECAST_HISTORY + FORECAST_HORIZON
FORECAST_TIME_SLOTS = 8
FORECAST_STEP_MINUTES = 5
FORECAST_HORIZON_INDEX = {"15": 2, "30": 5, "60": 11}
FORECAST_SAMPLE_COUNT = 1024
FORECAST_TRAIN_SEED = 20261006
FORECAST_SAMPLE_SEED = 20261010
FORECAST_WALL_SECONDS = 60 * 60
FORECAST_TEACHER_USD_CAP = 200.0
FORECAST_TEACHER_MAX_TOKENS = 256
FORECAST_TEACHER_PLANNED_CALLS = FORECAST_SAMPLE_COUNT
FORECAST_TEACHER_RETRY_CEILING = FORECAST_SAMPLE_COUNT * 2
FORECAST_ESTIMATED_INPUT_TOKENS_PER_CALL = 400
FORECAST_STUDENT_MAX_NEW_TOKENS = 128
FORECAST_STUDENT_MAX_SEQ_LEN = 2048
FORECAST_STUDENT_MAX_STEPS = FORECAST_SAMPLE_COUNT
FORECAST_STUDENT_SEED = 20261006
FORECAST_EVAL_BATCH = 4
FORECAST_STUDENT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
FREEWAY_ADAPTER_MARK = "2026-10-05-pems-student"
FORECAST_NUMBER = re.compile(r"[+-]?(?:\d+\.\d+|\d+)")
FORECAST_DRAW_USE = (
    "After the fit on the chronological first 70% stops, draw 1024 independent "
    "length-24 series with new noise. These are post-fit samples, not mid-training "
    "checkpoints and not runs/2026-10-05-pems/synthetic.jsonl. The teacher receives "
    "only the first 12 speeds of each draw and is asked for the next 12. Student "
    "supervision is the teacher's parsed 12-speed text. The generator's own last 12 "
    "speeds are not supervision. Real futures, freeway ids, sensor ids, neighbors, "
    "and statistics of protected rows are absent from every teacher request."
)
_FORECAST_FORBIDDEN = FORBIDDEN_RECORD_KEYS + (
    "future",
    "gold_future",
    "generator_future",
    "sensor_index",
)


def chronological_bounds(n_steps: int = FORECAST_N_STEPS) -> dict[str, tuple[int, int]]:
    """Train, middle holdout, and test as counts of raw time steps.

    `int(n * 0.70)` is the first 70%. `int(n * 0.20)` is the last 20%.
    The middle slice is whatever remains and is neither fit nor reported.
    """
    if n_steps < FORECAST_SERIES_LENGTH * 3:
        raise PemsProtocolError(f"series length {n_steps} is too short to split")
    train_end = int(n_steps * 0.70)
    test_len = int(n_steps * 0.20)
    test_start = n_steps - test_len
    if not (0 < train_end < test_start < n_steps):
        raise PemsProtocolError("chronological bounds are not ordered")
    return {
        "train": (0, train_end),
        "val": (train_end, test_start),
        "test": (test_start, n_steps),
    }


def forecast_question_starts(n_steps: int = FORECAST_N_STEPS) -> list[int]:
    """Eight starts per sensor, evenly spaced, window fully inside the last 20%."""
    test_start, test_end = chronological_bounds(n_steps)["test"]
    n_valid = test_end - test_start - FORECAST_SERIES_LENGTH + 1
    if n_valid < FORECAST_TIME_SLOTS:
        raise PemsProtocolError("test span does not hold the frozen time slots")
    starts: list[int] = []
    for slot in range(FORECAST_TIME_SLOTS):
        offset = (slot * (n_valid - 1)) // (FORECAST_TIME_SLOTS - 1)
        start = test_start + offset
        if start < test_start or start + FORECAST_SERIES_LENGTH > test_end:
            raise PemsProtocolError(f"question start {start} leaves the last 20%")
        starts.append(start)
    if len(set(starts)) != len(starts):
        raise PemsProtocolError("forecast time slots are not unique")
    return starts


def forecast_question_id(start: int, sensor_index: int) -> str:
    return f"t{start:05d}-s{sensor_index:03d}"


def forecast_question_ids(
    n_steps: int = FORECAST_N_STEPS,
    n_sensors: int = FORECAST_N_SENSORS,
) -> list[str]:
    ids: list[str] = []
    for start in forecast_question_starts(n_steps):
        for sensor_index in range(n_sensors):
            ids.append(forecast_question_id(start, sensor_index))
    if len(set(ids)) != len(ids):
        raise PemsProtocolError("forecast question ids are not unique")
    return ids


def assert_forecast_question_id(question_id_value: str, n_steps: int = FORECAST_N_STEPS) -> None:
    test_start, test_end = chronological_bounds(n_steps)["test"]
    try:
        time_part, sensor_part = question_id_value.split("-")
        start = int(time_part[1:])
        sensor_index = int(sensor_part[1:])
    except (ValueError, IndexError) as exc:
        raise PemsProtocolError(f"question id {question_id_value} is not a forecast id") from exc
    if sensor_index < 0 or sensor_index >= FORECAST_N_SENSORS:
        raise PemsProtocolError(f"question id {question_id_value} is outside the sensor list")
    if start < test_start or start + FORECAST_SERIES_LENGTH > test_end:
        raise PemsProtocolError(f"question id {question_id_value} is outside the last 20%")


def render_forecast_prompt(history: list[float] | tuple[float, ...]) -> str:
    """Twelve past speeds. The next twelve, the freeway, and the sensor stay out."""
    if len(history) != FORECAST_HISTORY:
        raise PemsProtocolError(
            f"a forecast history has {FORECAST_HISTORY} speeds, got {len(history)}"
        )
    if any(not _finite(value) for value in history):
        raise PemsProtocolError("forecast prompt refuses a non-finite speed")
    body = ",".join(f"{float(value):.3f}" for value in history)
    return (
        "Forecast the next 12 Bay Area traffic speeds, one speed per 5-minute step. "
        "Reply with exactly 12 decimal speeds separated by commas.\n"
        f"speeds: {body}\n"
        "forecast:"
    )


def _history_from_record(speeds: object) -> list[float]:
    if not isinstance(speeds, (list, tuple)):
        raise PemsProtocolError("teacher speeds must be a list")
    values = [float(value) for value in speeds]
    if len(values) == FORECAST_SERIES_LENGTH:
        values = values[:FORECAST_HISTORY]
    if len(values) != FORECAST_HISTORY:
        raise PemsProtocolError(
            f"teacher history has {FORECAST_HISTORY} speeds, got {len(values)}"
        )
    return values


def build_forecast_teacher_payload(
    record: dict[str, object], path: Path | None = None
) -> dict[str, object]:
    """One synthetic history. Real series, futures, and freeway fields are refused."""
    if path is not None:
        refuse_sealed_path(Path(path))
        name = Path(path).name
        if "test_questions" in name or "sealed" in name:
            raise PemsProtocolError(f"teacher send must not read {path}")
    if record.get("origin") != "synthetic":
        raise PemsProtocolError("forecast teacher payload accepts synthetic windows only")
    role = str(record.get("role") or "")
    if role in {"pems_test", "sealed_test", "generator_train", "real"}:
        raise PemsProtocolError(f"forecast teacher payload refuses role {role}")
    present = [key for key in _FORECAST_FORBIDDEN if key in record]
    if present:
        raise PemsProtocolError(
            "synthetic forecast record must not carry a location, sensor, future, or gold field: "
            + ", ".join(present)
        )
    window_id = str(record["window_id"])
    if not window_id.startswith("syn-"):
        raise PemsProtocolError("forecast teacher window id must be a synthetic id")
    if window_id.startswith("t") or "-s" in window_id and window_id[1:2].isdigit():
        raise PemsProtocolError("forecast teacher window id must not be a real test id")
    prompt = render_forecast_prompt(_history_from_record(record.get("speeds")))
    if "freeway" in prompt.lower() or "sensor" in prompt.lower():
        raise PemsProtocolError("forecast prompt names a freeway or a sensor")
    return {"window_id": window_id, "prompt": prompt, "origin": "synthetic"}


def forecast_teacher_request_body(model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    """Thinking stays off so the 12 speeds are written in content."""
    if model != TEACHER_MODEL:
        raise PemsProtocolError(f"teacher model {model} is not the frozen pilot model")
    if max_tokens != FORECAST_TEACHER_MAX_TOKENS:
        raise PemsProtocolError(f"forecast completion cap {max_tokens} is not frozen")
    if not prompt.startswith("Forecast the next 12"):
        raise PemsProtocolError("forecast teacher prompt is not the frozen question")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "thinking": {"type": "disabled"},
    }


def _twelve_speeds(line: str) -> list[float] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != FORECAST_HORIZON:
        return None
    values: list[float] = []
    for part in parts:
        if FORECAST_NUMBER.fullmatch(part) is None:
            return None
        value = float(part)
        if not _finite(value):
            return None
        values.append(value)
    return values


def parse_forecast(text: str) -> list[float] | None:
    """Nearest complete line of 12 speeds. A cut-off trailing fragment is ignored."""
    cut = text.split("<|endoftext|>", 1)[0]
    cut = cut.split("<|im_end|>", 1)[0]
    lines = [line.strip() for line in cut.splitlines() if line.strip()]
    for line in reversed(lines):
        values = _twelve_speeds(line)
        if values is not None:
            return values
    return None


def canonical_forecast(values: list[float]) -> str:
    if len(values) != FORECAST_HORIZON or any(not _finite(value) for value in values):
        raise PemsProtocolError("canonical forecast requires 12 finite speeds")
    return ",".join(f"{float(value):.3f}" for value in values)


def forecast_distill_examples(records: list[dict[str, object]]) -> list[PemsExample]:
    """Supervision is the teacher's parsed forecast, not a real or generated future."""
    if not records:
        raise PemsProtocolError("no approved teacher forecasts")
    examples: list[PemsExample] = []
    for record in records:
        path = record.get("path")
        if path is not None:
            refuse_sealed_path(Path(str(path)))
        if record.get("source") != APPROVED_LABEL_SOURCE:
            raise PemsProtocolError("forecast distillation accepts approved teacher labels only")
        blocked = [
            key
            for key in ("gold", "gold_future", "future", "generator_future", "generator_class")
            if key in record
        ]
        if blocked:
            raise PemsProtocolError(
                "forecast supervision cannot be a real future or a generator future"
            )
        label = record.get("final_forecast")
        if not isinstance(label, str):
            raise PemsProtocolError("teacher final forecast is missing")
        parsed = parse_forecast(label)
        if parsed is None or canonical_forecast(parsed) != label:
            raise PemsProtocolError("teacher final forecast is not 12 canonical speeds")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.startswith("Forecast the next 12"):
            raise PemsProtocolError("distillation prompt is not a forecast question")
        if "freeway" in prompt.lower() or "sensor" in prompt.lower():
            raise PemsProtocolError("distillation prompt names a freeway or a sensor")
        examples.append(
            PemsExample(window_id=str(record["window_id"]), prompt=prompt, supervision=label)
        )
    return examples


def zero_masked_horizon_metrics(
    predictions: list[list[float]],
    truths: list[list[float]],
) -> dict[str, float | None]:
    """MAE, RMSE, and MAPE at 15, 30, and 60 minutes.

    A position whose real speed is 0 is left out of every average. MAPE is the
    mean absolute percentage as a ratio, so 0.10 means 10 percent. The 15, 30,
    and 60 minute scores use forecast steps 3, 6, and 12.
    """
    if len(predictions) != len(truths):
        raise PemsProtocolError("predictions and truths differ in length")
    scores: dict[str, float | None] = {}
    for name, index in FORECAST_HORIZON_INDEX.items():
        absolute: list[float] = []
        squared: list[float] = []
        percentage: list[float] = []
        for pred, truth in zip(predictions, truths, strict=True):
            if len(pred) != FORECAST_HORIZON or len(truth) != FORECAST_HORIZON:
                raise PemsProtocolError("a forecast metric row has 12 speeds")
            actual = float(truth[index])
            guessed = float(pred[index])
            if not _finite(actual) or not _finite(guessed):
                raise PemsProtocolError("forecast metric refuses a non-finite speed")
            if actual == 0.0:
                continue
            gap = guessed - actual
            absolute.append(abs(gap))
            squared.append(gap * gap)
            percentage.append(abs(gap) / abs(actual))
        if not absolute:
            scores[f"mae_{name}"] = None
            scores[f"rmse_{name}"] = None
            scores[f"mape_{name}"] = None
            continue
        count = len(absolute)
        scores[f"mae_{name}"] = sum(absolute) / count
        scores[f"rmse_{name}"] = math.sqrt(sum(squared) / count)
        scores[f"mape_{name}"] = sum(percentage) / count
    return scores


def assert_base_student_adapter(adapter: str | None) -> None:
    """The freeway adapter is not a starting point for either new score."""
    if adapter is None or adapter == "":
        return
    normalized = str(adapter).replace("\\", "/")
    if FREEWAY_ADAPTER_MARK in normalized:
        raise PemsProtocolError("freeway adapter is not loaded for this forecast run")


def build_forecast_comparison_report(
    *,
    question_ids: list[str],
    gold_by_id: dict[str, list[float]],
    untuned_texts: list[str],
    distilled_texts: list[str],
    distilled_adapter: str,
) -> dict[str, object]:
    """Same last-20% list for the base student and the teacher-distilled student."""
    expected = forecast_question_ids()
    if list(question_ids) != expected:
        raise PemsProtocolError("forecast comparison is not the frozen question list")
    if len(untuned_texts) != len(question_ids) or len(distilled_texts) != len(question_ids):
        raise PemsProtocolError("forecast generations do not match the question list")
    assert_base_student_adapter(None)
    assert_base_student_adapter(distilled_adapter)
    if not distilled_adapter or FREEWAY_ADAPTER_MARK in distilled_adapter:
        raise PemsProtocolError("distilled arm needs the new adapter path")
    for question_id_value in question_ids:
        assert_forecast_question_id(question_id_value)
        if question_id_value not in gold_by_id:
            raise PemsProtocolError(f"missing forecast gold for {question_id_value}")

    def arm(texts: list[str], adapter: str | None) -> dict[str, object]:
        parsed_rows: list[list[float]] = []
        gold_rows: list[list[float]] = []
        for question_id_value, text in zip(question_ids, texts, strict=True):
            parsed = parse_forecast(text)
            if parsed is None:
                continue
            parsed_rows.append(parsed)
            gold_rows.append([float(value) for value in gold_by_id[question_id_value]])
        metrics = zero_masked_horizon_metrics(parsed_rows, gold_rows)
        return {
            "student_revision": FORECAST_STUDENT_REVISION,
            "adapter": adapter,
            "parsed_count": len(parsed_rows),
            **metrics,
        }

    return {
        "test_partition_id": FORECAST_PARTITION_ID,
        "question_ids": list(question_ids),
        "question_list_sha256": question_list_sha256(question_ids),
        "n_questions": len(question_ids),
        "supervision": "teacher_forecast",
        "student_revision": FORECAST_STUDENT_REVISION,
        "untuned": arm(untuned_texts, None),
        "teacher_distilled": arm(distilled_texts, distilled_adapter),
        "sealed_test_read": False,
        "private": False,
        "feasible": False,
        "state_of_the_art": False,
    }


def forecast_teacher_estimate_document(written_unix: int) -> dict[str, object]:
    """Pre-send bill for the forecast ledger. Peak rates, cap 200 USD."""
    planned = FORECAST_TEACHER_PLANNED_CALLS
    ceiling = FORECAST_TEACHER_RETRY_CEILING
    in_tok = FORECAST_ESTIMATED_INPUT_TOKENS_PER_CALL
    out_tok = FORECAST_TEACHER_MAX_TOKENS
    planned_usd = planned * peak_cost_usd(in_tok, out_tok)
    ceiling_usd = ceiling * peak_cost_usd(in_tok, out_tok)
    if ceiling_usd > FORECAST_TEACHER_USD_CAP:
        raise PemsProtocolError("forecast retry ceiling exceeds the 200 USD cap")
    return {
        "written_unix": written_unix,
        "provider": TEACHER_PROVIDER,
        "model": TEACHER_MODEL,
        "planned_calls": planned,
        "retry_ceiling_calls": ceiling,
        "estimated_input_tokens": planned * in_tok,
        "estimated_output_tokens": planned * out_tok,
        "estimated_input_tokens_per_call": in_tok,
        "max_output_tokens_per_call": out_tok,
        "estimated_usd_planned_peak": round(planned_usd, 6),
        "estimated_usd_retry_ceiling_peak": round(ceiling_usd, 6),
        "usd_cap": FORECAST_TEACHER_USD_CAP,
        "payload": (
            "synthetic Diffusion-TS histories only, the first 12 speeds of each "
            "post-fit length-24 draw; no sensor id, freeway id, neighbor, real "
            "future, or real PeMS series"
        ),
        "draw_use": FORECAST_DRAW_USE,
        "cheaper_unused_alternative": (
            "score the untuned local student and do not call the teacher, cost 0 USD"
        ),
        "price_basis": (
            "peak cache-miss Flash rates 0.30 USD per 1M input tokens and "
            "1.20 USD per 1M output tokens"
        ),
    }
