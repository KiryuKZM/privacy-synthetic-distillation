"""MapEval-Textual roles, option labels, and fictional scene checks.

This module does not load a model, does not call a teacher, and does not read
a sealed split to choose a setting. Gold labels stay in the local record.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal

from privacy_synth.attack_protocol import refuse_screening_as_attack


class SpatialProtocolError(ValueError):
    """A MapEval split, label, or scene violated the frozen protocol."""


CLASSES = ("unanswerable", "trip", "routing", "poi", "nearby")
ROLES = ("public_calibration", "local_dev", "sealed_evaluation")
ROLE_CAPS = {"public_calibration": 64, "local_dev": 64, "sealed_evaluation": None}
MIN_SEALED = 150
MIN_CALIBRATION = 56
PRECHECK_N = 8
MIN_QUALIFICATION = 48
STUDENT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

# Documentary DeepSeek Flash prices fetched 2026-10-07T14:26:15Z.
# Peak is the hard gate. A call has not confirmed the live fingerprint.
REQUEST_MODEL = "deepseek-flash"
DOCUMENTED_MODEL_VERSION = "DeepSeek-V4.1-Flash"
MAX_OUTPUT_TOKENS = 131072
API_MAX_OUTPUT_TOKENS = 393216
PEAK_INPUT_CACHE_MISS_USD_PER_M = Decimal("0.30")
PEAK_INPUT_CACHE_HIT_USD_PER_M = Decimal("0.006")
PEAK_OUTPUT_USD_PER_M = Decimal("1.20")
STAGE_CALL_CAP = 800
STAGE_USD_CAP = Decimal("140")
PRECHECK_CALL_CAP = 16
PRECHECK_USD_CAP = Decimal("5")

ARMS = (
    "generic_utility_off",
    "generic_utility_on",
    "task_aware_utility_off",
    "task_aware_utility_on",
)
KINDS = (
    "poi_open",
    "poi_masked",
    "nearby_nearest",
    "nearby_filtered",
    "farthest_pair",
    "direction",
    "routing_next",
    "trip_count",
)
KIND_CLASS = {
    "poi_open": "poi",
    "poi_masked": "unanswerable",
    "nearby_nearest": "nearby",
    "nearby_filtered": "nearby",
    "farthest_pair": "poi",
    "direction": "poi",
    "routing_next": "routing",
    "trip_count": "trip",
}
_RETRY_REASONS = frozenset(
    {"timeout", "transport", "empty_final", "truncated_without_label"}
)
_ANSWER_LINE = re.compile(r"(?m)^answer:\s*([0-9]+)\s*$")
_JSON_OPTION = re.compile(r'"option_no"\s*:\s*([0-9]+)')
_POI = re.compile(r"Information of\s*<b>(.*?)</b>", re.IGNORECASE | re.DOTALL)
_DAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_PLACES = (
    ("Orin Quay", "cafe"),
    ("Bram Ferry", "library"),
    ("Lumen Court", "museum"),
    ("Pell Yard", "park"),
    ("Sable Market", "market"),
    ("Ivo Dock", "cafe"),
    ("Nera Hall", "museum"),
    ("Cinder Gate", "park"),
    ("Vellum Annex", "library"),
    ("Moss Landing", "market"),
    ("Quill Terrace", "cafe"),
    ("Harrow Bend", "library"),
)
_CARDINALS = ("North", "East", "South", "West")
_COUNT_WORDS = ("One", "Two", "Three", "Four")
TEXT_VERSION_HIDDEN = "20261007"
TEXT_VERSION_VISIBLE = "20261010"
CURRENT_TEXT_VERSION = TEXT_VERSION_VISIBLE
AXIS_SENTENCE = "North is the positive y direction and East is the positive x direction."
_DAY_RE = "|".join(_DAYS)
_PLACE_RE = r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"


@dataclass(frozen=True)
class TextualRecord:
    id: int
    context: str
    question: str
    options: tuple[str, ...]
    github_correct: int
    classification: str
    gold_label: str
    group_id: str = ""
    role: str = ""

    def render_prompt(self) -> str:
        return render_exam_prompt(self.context, self.question, self.options)


@dataclass(frozen=True)
class Scene:
    scene_id: str
    kind: str
    context: str
    question: str
    options: tuple[str, ...]
    claimed_label: str
    world: str

    def render_prompt(self) -> str:
        return render_exam_prompt(self.context, self.question, self.options)


@dataclass(frozen=True)
class RoleAssignment:
    records: tuple[TextualRecord, ...]
    precheck_ids: tuple[int, ...]
    qualification_ids: tuple[int, ...]

    def role_records(self, role: str) -> tuple[TextualRecord, ...]:
        if role not in ROLES:
            raise SpatialProtocolError(f"unknown role {role}")
        return tuple(record for record in self.records if record.role == role)


def refuse_sealed_path(path: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if "sealed_evaluation" in lowered or "sealed-evaluation" in lowered or "sealed_test" in lowered:
        raise SpatialProtocolError(f"sealed path is refused: {path}")


def content_options(options) -> tuple[str, ...]:
    """Drop trailing blank placeholders. A blank before later text is illegal."""
    kept: list[str] = []
    seen_blank = False
    for option in options:
        text = str(option).strip()
        if text == "":
            seen_blank = True
            continue
        if seen_blank:
            raise SpatialProtocolError("blank placeholder precedes a content option")
        kept.append(text)
    if not kept:
        raise SpatialProtocolError("question has no content option")
    return tuple(kept)


def gold_label(github_correct: int, options) -> str:
    """Map the GitHub `answer.correct` field onto the paper's option number.

    -1 is unanswerable and becomes label 0. A stored 0 is the first content
    option and becomes label 1. Blank placeholders are not labels.
    """
    content = content_options(options)
    if github_correct == -1:
        return "0"
    if not isinstance(github_correct, int) or isinstance(github_correct, bool):
        raise SpatialProtocolError("github correct is not an integer")
    if github_correct < 0 or github_correct >= len(options):
        raise SpatialProtocolError("github correct is outside the option list")
    if str(options[github_correct]).strip() == "":
        raise SpatialProtocolError("github correct points at a blank placeholder")
    if content[github_correct] != str(options[github_correct]).strip():
        raise SpatialProtocolError("a blank placeholder shifted the content index")
    return str(github_correct + 1)


def legal_labels(options) -> tuple[str, ...]:
    content = content_options(options)
    return tuple(str(index) for index in range(len(content) + 1))


def render_exam_prompt(context: str, question: str, options) -> str:
    content = content_options(options)
    lines = [
        "Answer using only the context. Do not use outside knowledge.",
        "Context:",
        str(context).strip(),
        "Question:",
        str(question).strip(),
        "Options:",
        "0. Unanswerable",
    ]
    for index, option in enumerate(content, start=1):
        lines.append(f"{index}. {option}")
    lines.append(
        'Reply with JSON {"option_no": <number>, "explanation": "<reason>"}. '
        "option_no 0 means Unanswerable, not the first content option. "
        "End with a final line of the form answer: <number>."
    )
    lines.append("answer:")
    return "\n".join(lines) + " "


def parse_teacher_label(text: str | None, legal) -> str | None:
    """Parse one final label. Missing, illegal, and truncated text stay missing."""
    if text is None:
        return None
    allowed = {str(label) for label in legal}
    lines = _ANSWER_LINE.findall(text)
    if lines:
        label = lines[-1]
        return label if label in allowed else None
    found = _JSON_OPTION.findall(text)
    if not found:
        return None
    label = found[-1]
    return label if label in allowed else None


def supervision_label(label: str) -> str:
    if not re.fullmatch(r"[0-9]+", label or ""):
        raise SpatialProtocolError("supervision must be a bare option label")
    return label


def retry_allowed(reason: str, attempts_already_made: int) -> bool:
    """One extra try is allowed only for a transport or empty-final failure."""
    if reason == "disagrees_with_gold":
        return False
    return attempts_already_made == 1 and reason in _RETRY_REASONS


def worst_case_usd(calls: int, input_tokens_per_call: int, output_tokens_per_call: int) -> Decimal:
    if calls < 0 or input_tokens_per_call < 0 or output_tokens_per_call < 0:
        raise SpatialProtocolError("token counts must be non-negative")
    if output_tokens_per_call > API_MAX_OUTPUT_TOKENS:
        raise SpatialProtocolError("output cap exceeds the documented API maximum")
    million = Decimal(1_000_000)
    prompt = Decimal(calls * input_tokens_per_call) / million
    generated = Decimal(calls * output_tokens_per_call) / million
    return prompt * PEAK_INPUT_CACHE_MISS_USD_PER_M + generated * PEAK_OUTPUT_USD_PER_M


def reservation_fits(spent: Decimal, reserved: Decimal, cap: Decimal) -> bool:
    return spent + reserved <= cap


def peak_spend_usd(
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
) -> Decimal:
    """Bill one call at the documented peak Flash rates.

    A missing cache split bills the whole prompt as a cache miss. Tokens that
    the split does not cover are also billed as misses, so the gate cannot
    under-count a reported prompt.
    """
    prompt = int(prompt_tokens)
    completion = int(completion_tokens)
    hit = int(cache_hit_tokens)
    miss = int(cache_miss_tokens)
    if min(prompt, completion, hit, miss) < 0:
        raise SpatialProtocolError("token counts must be non-negative")
    if hit + miss == 0:
        miss = prompt
    elif prompt > hit + miss:
        miss += prompt - (hit + miss)
    million = Decimal(1_000_000)
    input_cost = (
        Decimal(hit) * PEAK_INPUT_CACHE_HIT_USD_PER_M
        + Decimal(miss) * PEAK_INPUT_CACHE_MISS_USD_PER_M
    )
    output_cost = Decimal(completion) * PEAK_OUTPUT_USD_PER_M
    return (input_cost + output_cost) / million


def sha256_ids(ids) -> str:
    blob = "\n".join(str(item) for item in ids).encode()
    return hashlib.sha256(blob).hexdigest()


def _norm(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip().lower()


def _class_name(raw, github_correct: int) -> str:
    if github_correct == -1 or raw is None:
        if github_correct != -1:
            raise SpatialProtocolError("an answerable row has no classification")
        if raw not in (None, "unanswerable"):
            raise SpatialProtocolError("an unanswerable row carries another class")
        return "unanswerable"
    if raw not in {"poi", "nearby", "routing", "trip"}:
        raise SpatialProtocolError(f"unknown classification {raw}")
    return raw


def _record_from_github(row: dict) -> TextualRecord:
    answer = row["answer"]
    options = tuple(answer["options"])
    correct = answer["correct"]
    return TextualRecord(
        id=int(row["id"]),
        context=row["context"],
        question=row["question"],
        options=options,
        github_correct=int(correct),
        classification=_class_name(row.get("classification"), int(correct)),
        gold_label=gold_label(int(correct), options),
    )


def load_textual_records(github_rows: list[dict], hf_rows: list[dict]) -> tuple[TextualRecord, ...]:
    """Load GitHub rows and require the HF flat file to carry the same text."""
    if len(github_rows) != len(hf_rows):
        raise SpatialProtocolError("GitHub and HF row counts differ")
    hf_by_id = {int(row["id"]): row for row in hf_rows}
    if len(hf_by_id) != len(hf_rows):
        raise SpatialProtocolError("HF ids are not unique")
    records = []
    for row in github_rows:
        record = _record_from_github(row)
        hf_row = hf_by_id.get(record.id)
        if hf_row is None:
            raise SpatialProtocolError(f"HF file lacks id {record.id}")
        if hf_row["context"] != record.context or hf_row["question"] != record.question:
            raise SpatialProtocolError(f"HF text differs for id {record.id}")
        if tuple(hf_row["options"]) != record.options:
            raise SpatialProtocolError(f"HF options differ for id {record.id}")
        expected = 0 if record.github_correct == -1 else record.github_correct + 1
        if int(hf_row["answer"]) != expected:
            raise SpatialProtocolError(f"HF answer code differs for id {record.id}")
        hf_class = hf_row.get("classification")
        if record.classification == "unanswerable":
            if hf_class != "unanswerable":
                raise SpatialProtocolError(f"HF class differs for id {record.id}")
        elif hf_class != record.classification:
            raise SpatialProtocolError(f"HF class differs for id {record.id}")
        if record.gold_label not in legal_labels(record.options):
            raise SpatialProtocolError(f"gold label is illegal for id {record.id}")
        records.append(record)
    if len({record.id for record in records}) != len(records):
        raise SpatialProtocolError("GitHub ids are not unique")
    return tuple(sorted(records, key=lambda record: record.id))


def api_overlap(api_rows: list[dict], records: tuple[TextualRecord, ...]) -> dict:
    """MapEval-API repeats the textual questions and is not a second sample."""
    api_ids = {int(row["id"]) for row in api_rows}
    text_ids = {record.id for record in records}
    questions = {_norm(record.question) for record in records}
    api_questions = {_norm(row.get("question", "")) for row in api_rows}
    return {
        "api_rows": len(api_rows),
        "shared_ids": len(api_ids & text_ids),
        "api_only_ids": len(api_ids - text_ids),
        "textual_only_ids": len(text_ids - api_ids),
        "shared_questions": len(api_questions & questions),
    }


def _poi_names(context: str) -> set[str]:
    return {name for name in (_norm(match) for match in _POI.findall(context or "")) if name}


def _groups(records: tuple[TextualRecord, ...]) -> dict[int, str]:
    parent: dict[int, int] = {}

    def find(item: int) -> int:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_context: dict[str, list[int]] = defaultdict(list)
    by_question: dict[str, list[int]] = defaultdict(list)
    by_poi: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_context[record.context].append(record.id)
        by_question[_norm(record.question)].append(record.id)
        for name in _poi_names(record.context):
            by_poi[name].append(record.id)
    for bucket in (by_context, by_question, by_poi):
        for ids in bucket.values():
            if len(ids) > 1:
                for item in ids[1:]:
                    union(ids[0], item)
    members: dict[int, list[int]] = defaultdict(list)
    for record in records:
        members[find(record.id)].append(record.id)
    group_of = {}
    for ids in members.values():
        group_id = f"g{min(ids)}"
        for item in ids:
            group_of[item] = group_id
    return group_of


def _class_counts(records) -> Counter:
    return Counter(record.classification for record in records)


def assign_roles(records: tuple[TextualRecord, ...]) -> RoleAssignment:
    """Assign intact dependence groups. This function does not look at a score."""
    if len({record.id for record in records}) != len(records):
        raise SpatialProtocolError("duplicate ids")
    group_of = _groups(records)
    grouped: dict[str, list[TextualRecord]] = defaultdict(list)
    for record in records:
        grouped[group_of[record.id]].append(record)
    groups = [tuple(sorted(members, key=lambda record: record.id)) for members in grouped.values()]
    for group in groups:
        if ROLE_CAPS["public_calibration"] is not None and len(group) > ROLE_CAPS["public_calibration"]:
            raise SpatialProtocolError("a dependence group is larger than the calibration cap")
    unassigned = {group[0].group_id if False else group_of[group[0].id]: group for group in groups}
    placed: dict[str, list[TextualRecord]] = {role: [] for role in ROLES}
    used = {role: 0 for role in ROLES}

    def fits(role: str, size: int) -> bool:
        cap = ROLE_CAPS[role]
        return True if cap is None else used[role] + size <= cap

    def take(role: str, group: tuple[TextualRecord, ...]) -> None:
        key = group_of[group[0].id]
        if key not in unassigned:
            raise SpatialProtocolError("group assigned twice")
        del unassigned[key]
        placed[role].extend(group)
        used[role] += len(group)

    for role in ROLES:
        for class_name in CLASSES:
            if any(record.classification == class_name for record in placed[role]):
                continue
            candidates = [
                group
                for group in unassigned.values()
                if fits(role, len(group))
                and any(record.classification == class_name for record in group)
            ]
            if not candidates:
                raise SpatialProtocolError(f"{role} cannot receive class {class_name}")
            chosen = min(
                candidates,
                key=lambda group: (
                    len(group),
                    -sum(record.classification == class_name for record in group) / len(group),
                    group[0].id,
                ),
            )
            take(role, chosen)

    global_counts = _class_counts(records)
    total = len(records)
    targets = {
        "public_calibration": ROLE_CAPS["public_calibration"],
        "local_dev": ROLE_CAPS["local_dev"],
        "sealed_evaluation": total - ROLE_CAPS["public_calibration"] - ROLE_CAPS["local_dev"],
    }

    def error_after(role: str, group: tuple[TextualRecord, ...]) -> float:
        counts = _class_counts(placed[role])
        extra = _class_counts(group)
        error = 0.0
        for class_name, class_total in global_counts.items():
            want = targets[role] * class_total / total
            error += abs(counts[class_name] + extra[class_name] - want)
        return error

    ordered = sorted(unassigned.values(), key=lambda group: (-len(group), group[0].id))
    for group in ordered:
        choices = [role for role in ROLES if fits(role, len(group))]
        if not choices:
            raise SpatialProtocolError("no role can hold a remaining group")
        role = min(choices, key=lambda name: (error_after(name, group), name != "sealed_evaluation", name))
        take(role, group)
    if unassigned:
        raise SpatialProtocolError("groups remained unassigned")

    def removal_ok(group: tuple[TextualRecord, ...]) -> bool:
        if used["sealed_evaluation"] - len(group) < MIN_SEALED:
            return False
        sealed_counts = _class_counts(placed["sealed_evaluation"])
        group_counts = _class_counts(group)
        return all(sealed_counts[class_name] - group_counts[class_name] >= 1 for class_name in CLASSES)

    for role, target in (("public_calibration", targets["public_calibration"]), ("local_dev", targets["local_dev"])):
        while used[role] < target:
            gap = target - used[role]
            movable = []
            sealed_groups: dict[str, list[TextualRecord]] = defaultdict(list)
            for record in placed["sealed_evaluation"]:
                sealed_groups[group_of[record.id]].append(record)
            for group_list in sealed_groups.values():
                group = tuple(sorted(group_list, key=lambda record: record.id))
                if len(group) <= gap and removal_ok(group):
                    movable.append(group)
            if not movable:
                break
            # Prefer a group that fills the remaining gap, then the largest that still fits.
            chosen = min(movable, key=lambda group: (len(group) != gap, -len(group), group[0].id))
            placed["sealed_evaluation"] = [
                record for record in placed["sealed_evaluation"] if group_of[record.id] != group_of[chosen[0].id]
            ]
            used["sealed_evaluation"] -= len(chosen)
            placed[role].extend(chosen)
            used[role] += len(chosen)

    if used["public_calibration"] < MIN_CALIBRATION or used["public_calibration"] > ROLE_CAPS["public_calibration"]:
        raise SpatialProtocolError("public calibration size is outside the frozen bound")
    if used["local_dev"] > ROLE_CAPS["local_dev"]:
        raise SpatialProtocolError("local dev exceeds its cap")
    if used["sealed_evaluation"] < MIN_SEALED:
        raise SpatialProtocolError("sealed evaluation is below 150")
    if sum(used.values()) != len(records):
        raise SpatialProtocolError("role sizes do not cover the records")
    for role in ROLES:
        present = {record.classification for record in placed[role]}
        if present != set(CLASSES):
            raise SpatialProtocolError(f"{role} misses a class")

    stamped = []
    for role in ROLES:
        for record in sorted(placed[role], key=lambda item: item.id):
            stamped.append(
                TextualRecord(
                    id=record.id,
                    context=record.context,
                    question=record.question,
                    options=record.options,
                    github_correct=record.github_correct,
                    classification=record.classification,
                    gold_label=record.gold_label,
                    group_id=group_of[record.id],
                    role=role,
                )
            )
    assignment = RoleAssignment(tuple(stamped), (), ())
    precheck = _precheck_ids(assignment.role_records("public_calibration"))
    qualification = tuple(
        record.id
        for record in assignment.role_records("public_calibration")
        if record.id not in set(precheck)
    )
    if len(qualification) < MIN_QUALIFICATION:
        raise SpatialProtocolError("qualification set is below 48")
    remaining = [record for record in assignment.role_records("public_calibration") if record.id in set(qualification)]
    if {record.classification for record in remaining} != set(CLASSES):
        raise SpatialProtocolError("qualification set misses a class")
    return RoleAssignment(assignment.records, precheck, qualification)


def _precheck_ids(calibration: tuple[TextualRecord, ...]) -> tuple[int, ...]:
    counts = _class_counts(calibration)
    chosen = []
    for record in sorted(calibration, key=lambda item: item.id):
        if len(chosen) == PRECHECK_N:
            break
        if counts[record.classification] - 1 < 1:
            continue
        counts[record.classification] -= 1
        chosen.append(record.id)
    if len(chosen) != PRECHECK_N:
        raise SpatialProtocolError("format precheck could not keep every class in qualification")
    return tuple(chosen)


def assignment_digest(assignment: RoleAssignment) -> dict:
    roles = {}
    for role in ROLES:
        records = assignment.role_records(role)
        ids = [record.id for record in records]
        roles[role] = {
            "count": len(ids),
            "sha256_sorted_ids": sha256_ids(sorted(ids)),
            "first_id": ids[0] if ids else None,
            "last_id": ids[-1] if ids else None,
            "class_counts": dict(_class_counts(records)),
            "group_count": len({record.group_id for record in records}),
        }
    qualification = [
        record
        for record in assignment.role_records("public_calibration")
        if record.id in set(assignment.qualification_ids)
    ]
    return {
        "version": "mapeval-textual-v1",
        "student_revision": STUDENT_REVISION,
        "request_model": REQUEST_MODEL,
        "documented_model_version": DOCUMENTED_MODEL_VERSION,
        "calls_approved": False,
        "precheck_ids": list(assignment.precheck_ids),
        "precheck_sha256": sha256_ids(assignment.precheck_ids),
        "qualification_ids_sha256": sha256_ids(sorted(assignment.qualification_ids)),
        "qualification_count": len(assignment.qualification_ids),
        "qualification_class_counts": dict(_class_counts(qualification)),
        "roles": roles,
        "screening_reference_roles": ["public_calibration", "local_dev"],
        "utility_filter_rule": "unfrozen",
    }


def refuse_sealed_reference(records) -> None:
    for record in records:
        if record.role == "sealed_evaluation":
            raise SpatialProtocolError("sealed records cannot be a screen reference")


def copy_screen(scene: Scene, reference: tuple[TextualRecord, ...]) -> dict:
    """Exact question and POI-name screen. This is not a privacy proof or an attack."""
    refuse_sealed_reference(reference)
    question = _norm(scene.question)
    pois = _poi_names("\n".join(record.context for record in reference))
    question_hit = any(_norm(record.question) == question for record in reference)
    name_hits = sorted({name for name, _category in _PLACES if name.lower() in pois})
    screen = {
        "kind": "exact_overlap",
        "question_exact_hit": question_hit,
        "fictional_name_hits": name_hits,
        "privacy_proof": False,
        "membership_inference": False,
    }
    try:
        refuse_screening_as_attack(screen)
    except ValueError:
        return screen
    raise SpatialProtocolError("copy screen was accepted as an attack")


def paired_cluster_diff(groups, *, seed: int, n_boot: int = 4000) -> dict:
    """Micro-average paired gap with groups resampled together."""
    if n_boot < 1 or not groups or any(len(group) == 0 for group in groups):
        raise SpatialProtocolError("bootstrap needs non-empty groups")
    point = _mean_gap(groups)
    rng = random.Random(seed)
    draws = []
    width = len(groups)
    for _ in range(n_boot):
        sample = [groups[rng.randrange(width)] for _ in range(width)]
        draws.append(_mean_gap(sample))
    draws.sort()
    return {
        "point": point,
        "low": _percentile(draws, 0.025),
        "high": _percentile(draws, 0.975),
        "groups": width,
        "items": sum(len(group) for group in groups),
    }


def constant_label_accuracy(gold_labels) -> dict:
    labels = [str(label) for label in gold_labels]
    if not labels:
        raise SpatialProtocolError("constant baseline needs labels")
    counts = Counter(labels)
    best = sorted(counts, key=lambda label: (-counts[label], int(label)))[0]
    correct = sum(label == best for label in labels)
    return {"label": best, "correct": correct, "n": len(labels), "accuracy": correct / len(labels)}


def scenes_for_arm(arm: str, count: int, seed: int) -> tuple[Scene, ...]:
    if arm not in ARMS:
        raise SpatialProtocolError(f"unknown arm {arm}")
    if count < 1:
        raise SpatialProtocolError("scene count must be positive")
    if arm.endswith("utility_on"):
        raise SpatialProtocolError("utility filter rule is not frozen")
    mode = "task_aware" if arm.startswith("task_aware") else "generic"
    return iter_scenes(mode, count, seed)


def assert_request_prefix(short_ids, long_ids) -> None:
    if tuple(long_ids[: len(short_ids)]) != tuple(short_ids):
        raise SpatialProtocolError("the 64-request list is not a prefix of the 160-request list")


def iter_scenes(
    mode: str,
    count: int,
    seed: int,
    text_version: str = CURRENT_TEXT_VERSION,
) -> tuple[Scene, ...]:
    scenes, _attempts = _fill_scenes(mode, count, seed, text_version)
    return scenes


def _fill_scenes(
    mode: str,
    count: int,
    seed: int,
    text_version: str = CURRENT_TEXT_VERSION,
) -> tuple[tuple[Scene, ...], int]:
    if mode not in {"generic", "task_aware"}:
        raise SpatialProtocolError(f"unknown scene mode {mode}")
    rng = random.Random(seed)
    accepted: list[Scene] = []
    attempts = 0
    while len(accepted) < count:
        attempts += 1
        if attempts > count * 20:
            raise SpatialProtocolError("scene constructor could not fill the request")
        kind = KINDS[len(accepted) % len(KINDS)] if mode == "task_aware" else KINDS[rng.randrange(len(KINDS))]
        scene = propose_scene(kind, rng, f"{mode}-{seed}-{len(accepted)}", text_version)
        checked = accept_scene(scene)
        if checked is not None:
            accepted.append(checked)
    return tuple(accepted), attempts


def scene_audit(
    mode: str,
    count: int,
    seed: int,
    text_version: str = CURRENT_TEXT_VERSION,
) -> dict:
    """Recount an accepted batch. The accepted-set rate is the generation gate."""
    scenes, attempts = _fill_scenes(mode, count, seed, text_version)
    kinds = Counter(scene.kind for scene in scenes)
    classes = Counter(KIND_CLASS[scene.kind] for scene in scenes)
    rechecked = sum(accept_scene(scene) is not None for scene in scenes)
    return {
        "mode": mode,
        "seed": seed,
        "text_version": text_version,
        "requested": count,
        "proposals": attempts,
        "accepted": len(scenes),
        "rechecked_accepts": rechecked,
        "accepted_set_rate": (rechecked / len(scenes)) if scenes else 0.0,
        "proposal_accept_rate": (len(scenes) / attempts) if attempts else 0.0,
        "kind_counts": {kind: kinds[kind] for kind in KINDS},
        "class_counts": {name: classes[name] for name in CLASSES},
        "kinds_missing": [kind for kind in KINDS if kinds[kind] == 0],
        "scene_ids": [scene.scene_id for scene in scenes],
    }


def audit_visible_once(
    mode: str,
    count: int,
    seed: int,
    text_version: str = CURRENT_TEXT_VERSION,
) -> dict:
    """Score one proposal per draw. Rejections stay in the rate and are not redrawn."""
    if mode not in {"generic", "task_aware"}:
        raise SpatialProtocolError(f"unknown scene mode {mode}")
    if count < 1:
        raise SpatialProtocolError("scene count must be positive")
    rng = random.Random(seed)
    reasons: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    accepted = 0
    hidden_visible_conflict = 0
    for index in range(count):
        kind = KINDS[index % len(KINDS)] if mode == "task_aware" else KINDS[rng.randrange(len(KINDS))]
        scene = propose_scene(kind, rng, f"{mode}-{seed}-once-{index}", text_version)
        kind_counts[kind] += 1
        structural = basic_violations(scene)
        hidden = consistent_labels(scene)
        visible = labels_from_visible_text(scene.kind, scene.context, scene.question, scene.options)
        if hidden != visible:
            hidden_visible_conflict += 1
        if structural:
            reasons[structural[0]] += 1
        elif hidden != {scene.claimed_label}:
            reasons["hidden oracle disagrees with the claim"] += 1
        elif visible != {scene.claimed_label}:
            reasons["visible text does not entail the claim"] += 1
        else:
            accepted += 1
            reasons["accepted"] += 1
    return {
        "mode": mode,
        "seed": seed,
        "text_version": text_version,
        "proposals": count,
        "accepted": accepted,
        "rejected": count - accepted,
        "proposal_reject_rate": (count - accepted) / count,
        "hidden_visible_conflicts": hidden_visible_conflict,
        "kind_counts": {kind: kind_counts[kind] for kind in KINDS},
        "reasons": dict(reasons),
        "resampled": False,
    }


def accept_scene(scene: Scene) -> Scene | None:
    if basic_violations(scene):
        return None
    entailed = consistent_labels(scene)
    if entailed != {scene.claimed_label}:
        return None
    if scene.claimed_label not in legal_labels(scene.options):
        return None
    visible = labels_from_visible_text(scene.kind, scene.context, scene.question, scene.options)
    if visible != {scene.claimed_label}:
        return None
    return scene


def basic_violations(scene: Scene) -> tuple[str, ...]:
    problems = []
    if scene.kind not in KINDS:
        problems.append("unknown kind")
    if not scene.context.strip() or not scene.question.strip():
        problems.append("empty text")
    try:
        content = content_options(scene.options)
    except SpatialProtocolError as exc:
        return (str(exc),)
    if len(set(content)) != len(content):
        problems.append("duplicate options")
    if any(option.lower() == "unanswerable" for option in content):
        problems.append("content option repeats the unanswerable label")
    try:
        world = json.loads(scene.world)
    except json.JSONDecodeError:
        return ("world is not json",)
    names = [place["name"] for place in world.get("places", [])]
    if len(names) != len(set(names)) or any(not name for name in names):
        problems.append("place names are not unique")
    for place in world.get("places", []):
        if not isinstance(place["x_m"], int) or not isinstance(place["y_m"], int):
            problems.append("coordinates are not integers")
        rating = place.get("rating")
        if rating is not None and not 0 <= rating <= 5:
            problems.append("rating is outside 0 to 5")
    if scene.kind == "poi_masked":
        hidden = world.get("hidden_hours") or []
        rendered = scene.context
        for minute in hidden:
            if fmt_hhmm(int(minute)) in rendered:
                problems.append("masked hours appear in the context")
                break
        asked = world.get("asked_day")
        if asked and f"{asked}: not stated" not in rendered:
            problems.append("masked day is not marked not stated")
    return tuple(problems)


def consistent_labels(scene: Scene) -> set[str]:
    world = json.loads(scene.world)
    if scene.kind == "poi_open":
        return _open_labels(world, scene.options, masked=False)
    if scene.kind == "poi_masked":
        return _open_labels(world, scene.options, masked=True)
    if scene.kind == "nearby_nearest":
        return _nearest_labels(world, scene.options)
    if scene.kind == "nearby_filtered":
        return _filtered_labels(world, scene.options)
    if scene.kind == "farthest_pair":
        return _farthest_labels(world, scene.options)
    if scene.kind == "direction":
        return _direction_labels(world, scene.options)
    if scene.kind == "routing_next":
        return _route_labels(world, scene.options)
    if scene.kind == "trip_count":
        return _trip_labels(world, scene.options)
    return set()


def propose_scene(
    kind: str,
    rng: random.Random,
    scene_id: str,
    text_version: str = CURRENT_TEXT_VERSION,
) -> Scene:
    if text_version not in {TEXT_VERSION_HIDDEN, TEXT_VERSION_VISIBLE}:
        raise SpatialProtocolError(f"unknown text version {text_version}")
    if kind == "poi_open":
        scene = _propose_open(rng, scene_id, masked=False)
    elif kind == "poi_masked":
        scene = _propose_open(rng, scene_id, masked=True)
    elif kind == "nearby_nearest":
        scene = _propose_nearest(rng, scene_id)
    elif kind == "nearby_filtered":
        scene = _propose_filtered(rng, scene_id, text_version)
    elif kind == "farthest_pair":
        scene = _propose_farthest(rng, scene_id)
    elif kind == "direction":
        scene = _propose_direction(rng, scene_id, text_version)
    elif kind == "routing_next":
        scene = _propose_route(rng, scene_id)
    elif kind == "trip_count":
        scene = _propose_trip(rng, scene_id)
    else:
        raise SpatialProtocolError(f"unknown kind {kind}")
    if text_version == TEXT_VERSION_HIDDEN:
        return scene
    world = json.loads(scene.world)
    world["text_version"] = TEXT_VERSION_VISIBLE
    return Scene(
        scene.scene_id,
        scene.kind,
        scene.context,
        scene.question,
        scene.options,
        scene.claimed_label,
        _world_json(world),
    )


def fmt_hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _world_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _place(name: str, category: str, x_m: int, y_m: int, rating, hours, disclosed) -> dict:
    return {
        "name": name,
        "category": category,
        "x_m": x_m,
        "y_m": y_m,
        "rating": rating,
        "hours": hours,
        "hours_disclosed": disclosed,
    }


def _propose_open(rng: random.Random, scene_id: str, masked: bool) -> Scene:
    name, category = _PLACES[rng.randrange(len(_PLACES))]
    day = _DAYS[rng.randrange(len(_DAYS))]
    open_minute, close_minute = 8 * 60, 18 * 60
    shown_open, shown_close = (9 * 60, 17 * 60) if masked else (open_minute, close_minute)
    inside = bool(rng.randrange(2))
    minute = 10 * 60 if inside else 20 * 60
    hours = {}
    disclosed = {}
    for one_day in _DAYS:
        if masked and one_day == day:
            hours[one_day] = [open_minute, close_minute]
            disclosed[one_day] = False
        else:
            hours[one_day] = [shown_open, shown_close]
            disclosed[one_day] = True
    place = _place(name, category, 0, 0, 4.2, hours, disclosed)
    shown = []
    for one_day in _DAYS:
        if disclosed[one_day]:
            span = hours[one_day]
            shown.append(f"{one_day}: {fmt_hhmm(span[0])}-{fmt_hhmm(span[1])}")
        else:
            shown.append(f"{one_day}: not stated")
    context = f"Information of {name}. Category: {category}. Hours: " + "; ".join(shown) + "."
    question = f"Is {name} open at {fmt_hhmm(minute)} on {day}?"
    claimed = "0" if masked else ("1" if inside else "2")
    world = {
        "places": [place],
        "asked_place": name,
        "asked_day": day,
        "asked_minute": minute,
        "hidden_hours": [open_minute, close_minute] if masked else None,
    }
    return Scene(scene_id, "poi_masked" if masked else "poi_open", context, question, ("Yes", "No"), claimed, _world_json(world))


def _named_pool(excluded: str) -> tuple[tuple[str, str], ...]:
    return tuple(place for place in _PLACES if place[0] != excluded)


def _propose_nearest(rng: random.Random, scene_id: str) -> Scene:
    picks = rng.sample(_named_pool("Quill Terrace"), 4)
    origin = _place("Quill Terrace", "cafe", 0, 0, None, {}, {})
    places = [origin]
    distances = (300, 700, 1200, 1800)
    for (name, category), distance in zip(picks, distances):
        places.append(_place(name, category, distance, 0, None, {}, {}))
    options = tuple(place["name"] for place in places[1:])
    order = list(options)
    rng.shuffle(order)
    claimed = str(order.index(picks[0][0]) + 1)
    context = "Origin Quill Terrace is at (0 m, 0 m). " + " ".join(
        f"{place['name']} is at ({place['x_m']} m, {place['y_m']} m)." for place in places[1:]
    )
    question = "Which place is nearest to Quill Terrace?"
    world = {"places": places, "origin": "Quill Terrace"}
    return Scene(scene_id, "nearby_nearest", context, question, tuple(order), claimed, _world_json(world))


def _propose_filtered(rng: random.Random, scene_id: str, text_version: str) -> Scene:
    picks = rng.sample(_PLACES[:8], 4)
    day = "tuesday"
    minute = 9 * 60
    places = []
    winner = picks[0][0]
    for index, (name, category) in enumerate(picks):
        rating = 4.8 if index == 0 else 4.0
        if index == 1:
            hours = {"tuesday": [12 * 60, 18 * 60]}
        else:
            hours = {"tuesday": [8 * 60, 11 * 60]}
        disclosed = {"tuesday": True}
        places.append(_place(name, category, 200 * (index + 1), 0, rating, hours, disclosed))
    order = [place["name"] for place in places]
    rng.shuffle(order)
    claimed = str(order.index(winner) + 1)
    bits = []
    for place in places:
        span = place["hours"]["tuesday"]
        clock = f"{fmt_hhmm(span[0])}-{fmt_hhmm(span[1])}"
        if text_version == TEXT_VERSION_VISIBLE:
            bits.append(f"{place['name']} rating {place['rating']} hours {day} {clock}.")
        else:
            bits.append(f"{place['name']} rating {place['rating']} hours {clock}.")
    context = " ".join(bits)
    question = f"Which place is open at {fmt_hhmm(minute)} on {day} and has a rating of at least 4.5?"
    world = {
        "places": places,
        "asked_day": day,
        "asked_minute": minute,
        "min_rating": 4.5,
    }
    return Scene(scene_id, "nearby_filtered", context, question, tuple(order), claimed, _world_json(world))


def _propose_farthest(rng: random.Random, scene_id: str) -> Scene:
    picks = rng.sample(_PLACES, 4)
    places = []
    positions = (0, 400, 800, 3000)
    for (name, category), position in zip(picks, positions):
        places.append(_place(name, category, position, 0, None, {}, {}))
    true_pair = _pair_key(picks[0][0], picks[3][0])
    distractors = [
        _pair_key(picks[0][0], picks[1][0]),
        _pair_key(picks[1][0], picks[2][0]),
        _pair_key(picks[2][0], picks[3][0]),
    ]
    order = [true_pair, *distractors]
    rng.shuffle(order)
    claimed = str(order.index(true_pair) + 1)
    context = " ".join(f"{place['name']} is at ({place['x_m']} m, {place['y_m']} m)." for place in places)
    question = "Which pair of places is farthest apart?"
    world = {"places": places}
    return Scene(scene_id, "farthest_pair", context, question, tuple(order), claimed, _world_json(world))


def _propose_direction(rng: random.Random, scene_id: str, text_version: str) -> Scene:
    pool = _named_pool("Quill Terrace")
    name, category = pool[rng.randrange(len(pool))]
    cardinal = _CARDINALS[rng.randrange(len(_CARDINALS))]
    offset = {"North": (0, 1000), "East": (1000, 0), "South": (0, -1000), "West": (-1000, 0)}[cardinal]
    origin = _place("Quill Terrace", "cafe", 0, 0, None, {}, {})
    target = _place(name, category, offset[0], offset[1], None, {}, {})
    claimed = str(_CARDINALS.index(cardinal) + 1)
    context = f"Quill Terrace is at (0 m, 0 m). {name} is at ({offset[0]} m, {offset[1]} m)."
    if text_version == TEXT_VERSION_VISIBLE:
        context = f"{AXIS_SENTENCE} {context}"
    question = f"Which way is {name} from Quill Terrace?"
    world = {"places": [origin, target], "origin": "Quill Terrace", "target": name}
    return Scene(scene_id, "direction", context, question, _CARDINALS, claimed, _world_json(world))


def _propose_route(rng: random.Random, scene_id: str) -> Scene:
    picks = rng.sample(_PLACES, 4)
    steps = [
        f"Head north on {picks[0][0]}",
        f"Turn right onto {picks[1][0]}",
        f"Turn left onto {picks[2][0]}",
        f"Arrive at {picks[3][0]}",
    ]
    anchor = picks[1][0]
    nxt = steps[2]
    distractors = [steps[0], steps[1], steps[3]]
    order = [nxt, *distractors]
    rng.shuffle(order)
    claimed = str(order.index(nxt) + 1)
    context = "Route by foot: " + " ".join(f"{index}. {step}." for index, step in enumerate(steps, start=1))
    question = f"After reaching {anchor}, where should I go next?"
    world = {"places": [], "steps": steps, "anchor": anchor}
    return Scene(scene_id, "routing_next", context, question, tuple(order), claimed, _world_json(world))


def _propose_trip(rng: random.Random, scene_id: str) -> Scene:
    count = 1 + rng.randrange(4)
    picks = rng.sample(_named_pool("Quill Terrace"), 4)
    places = [_place(name, category, 500 * (index + 1), 0, None, {}, {}) for index, (name, category) in enumerate(picks)]
    visit = {place["name"]: 30 for place in places}
    names = ["Quill Terrace", *[place["name"] for place in places]]
    travel = {left: {right: 0 if left == right else 10 for right in names} for left in names}
    budget = count * 40
    context = (
        "Start at Quill Terrace. Each listed place takes 30 minutes on site. "
        "Travel between any two different places takes 10 minutes. "
        + " ".join(f"Place: {place['name']}." for place in places)
        + f" The time budget is {budget} minutes."
    )
    question = "What is the maximum number of listed places that can be visited?"
    world = {
        "places": places,
        "origin": "Quill Terrace",
        "visit_min": visit,
        "travel_min": travel,
        "budget_min": budget,
    }
    return Scene(scene_id, "trip_count", context, question, _COUNT_WORDS, str(count), _world_json(world))


def _by_name(world: dict) -> dict:
    return {place["name"]: place for place in world["places"]}


def _option_label(options, text: str) -> set[str]:
    hits = [str(index) for index, option in enumerate(options, start=1) if option == text]
    if len(hits) != 1:
        return set()
    return {hits[0]}


def _option_labels(options, names) -> set[str]:
    labels: set[str] = set()
    for name in names:
        hits = [str(index) for index, option in enumerate(options, start=1) if option == name]
        if len(hits) != 1:
            return set()
        labels.add(hits[0])
    return labels


def _open_labels(world: dict, options, masked: bool) -> set[str]:
    if tuple(options) != ("Yes", "No"):
        return set()
    place = _by_name(world)[world["asked_place"]]
    day = world["asked_day"]
    minute = world["asked_minute"]
    if masked or not place["hours_disclosed"].get(day, False):
        return {"0"}
    span = place["hours"].get(day)
    if span is not None and span[0] <= minute < span[1]:
        return {"1"}
    return {"2"}


def _squared(left: dict, right: dict) -> int:
    return (left["x_m"] - right["x_m"]) ** 2 + (left["y_m"] - right["y_m"]) ** 2


def _nearest_labels(world: dict, options) -> set[str]:
    places = _by_name(world)
    origin = places[world["origin"]]
    scored = []
    for option in options:
        if option not in places:
            return set()
        scored.append((_squared(origin, places[option]), option))
    best = min(distance for distance, _name in scored)
    winners = [name for distance, name in scored if distance == best]
    if len(winners) != 1:
        return set()
    return _option_label(options, winners[0])


def _filtered_labels(world: dict, options) -> set[str]:
    places = _by_name(world)
    day = world["asked_day"]
    minute = world["asked_minute"]
    minimum = world["min_rating"]
    winners = []
    for option in options:
        place = places.get(option)
        if place is None or place.get("rating") is None:
            return set()
        if not place["hours_disclosed"].get(day, False):
            continue
        span = place["hours"].get(day)
        if place["rating"] >= minimum and span is not None and span[0] <= minute < span[1]:
            winners.append(option)
    if len(winners) != 1:
        return {"0"} if not winners else set()
    return _option_label(options, winners[0])


def _pair_key(left: str, right: str) -> str:
    ordered = tuple(sorted((left, right)))
    return f"{ordered[0]} and {ordered[1]}"


def _farthest_labels(world: dict, options) -> set[str]:
    places = world["places"]
    scored = []
    for left, right in itertools.combinations(places, 2):
        scored.append((_squared(left, right), _pair_key(left["name"], right["name"])))
    best = max(distance for distance, _name in scored)
    winners = [name for distance, name in scored if distance == best]
    if len(winners) != 1:
        return set()
    return _option_label(options, winners[0])


def _direction_labels(world: dict, options) -> set[str]:
    places = _by_name(world)
    origin = places[world["origin"]]
    target = places[world["target"]]
    dx = target["x_m"] - origin["x_m"]
    dy = target["y_m"] - origin["y_m"]
    if dx == 0 and dy > 0:
        name = "North"
    elif dx > 0 and dy == 0:
        name = "East"
    elif dx == 0 and dy < 0:
        name = "South"
    elif dx < 0 and dy == 0:
        name = "West"
    else:
        return set()
    return _option_label(options, name)


def _route_labels(world: dict, options) -> set[str]:
    steps = world["steps"]
    hits = [index for index, step in enumerate(steps) if world["anchor"] in step]
    if len(hits) != 1 or hits[0] == len(steps) - 1:
        return set()
    return _option_label(options, steps[hits[0] + 1])


def _trip_labels(world: dict, options) -> set[str]:
    if tuple(options) != _COUNT_WORDS:
        return set()
    names = [place["name"] for place in world["places"]]
    origin = world["origin"]
    visit = world["visit_min"]
    travel = world["travel_min"]
    budget = world["budget_min"]
    best = 0
    for size in range(1, len(names) + 1):
        for sequence in itertools.permutations(names, size):
            elapsed = 0
            previous = origin
            feasible = True
            for name in sequence:
                try:
                    elapsed += int(travel[previous][name]) + int(visit[name])
                except KeyError:
                    return set()
                previous = name
                if elapsed > budget:
                    feasible = False
                    break
            if feasible:
                best = size
    if best < 1 or best > 4:
        return set()
    return {str(best)}


def _hhmm_to_minute(text: str) -> int:
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


def _axis_signs(context: str) -> tuple[int, int] | None:
    match = re.search(
        rf"(North|South) is the positive y direction and (East|West) is the positive x direction\.",
        context,
    )
    if match is None:
        return None
    y_name, x_name = match.group(1), match.group(2)
    return (1 if x_name == "East" else -1, 1 if y_name == "North" else -1)


def _coordinates(text: str) -> list[tuple[str, int, int]]:
    found = re.findall(
        rf"({_PLACE_RE}) is at \((-?\d+) m, (-?\d+) m\)\.",
        text,
    )
    return [(name, int(x_m), int(y_m)) for name, x_m, y_m in found]


NEAREST_TIE_SENTENCE = (
    "If several candidates are equally near, choose the name that comes first in alphabetical order."
)
FARTHEST_TIE_SENTENCE = (
    "If several pairs are equally far, choose the pair name that comes first in alphabetical order."
)
ANY_NEAREST_SENTENCE = (
    "If several candidates are equally near, every equally near name is a correct answer."
)
ANY_FARTHEST_SENTENCE = (
    "If several pairs are equally far, every equally far pair is a correct answer."
)


def labels_from_visible_text(kind: str, context: str, question: str, options) -> set[str]:
    """Entail a label from rendered text alone. This function does not read `world`."""
    if kind in {"poi_open", "poi_masked"}:
        return _visible_open(context, question, options)
    if kind == "nearby_nearest":
        return _visible_nearest(context, question, options)
    if kind == "nearby_filtered":
        return _visible_filtered(context, question, options)
    if kind == "farthest_pair":
        return _visible_farthest(context, question, options)
    if kind == "direction":
        return _visible_direction(context, question, options)
    if kind == "routing_next":
        return _visible_route(context, question, options)
    if kind == "trip_count":
        return _visible_trip(context, question, options)
    return set()


def _visible_open(context: str, question: str, options) -> set[str]:
    if tuple(options) != ("Yes", "No"):
        return set()
    header = re.fullmatch(
        rf"Information of ({_PLACE_RE})\. Category: ([^.]+)\. Hours: (.+)\.",
        context,
    )
    asked = re.fullmatch(
        rf"Is ({_PLACE_RE}) open at (\d{{2}}:\d{{2}}) on ({_DAY_RE})\?",
        question,
    )
    if header is None or asked is None or header.group(1) != asked.group(1):
        return set()
    hours = {}
    for piece in header.group(3).split("; "):
        match = re.fullmatch(rf"({_DAY_RE}): (not stated|\d{{2}}:\d{{2}}-\d{{2}}:\d{{2}})", piece)
        if match is None:
            return set()
        hours[match.group(1)] = match.group(2)
    stated = hours.get(asked.group(3))
    if stated is None or stated == "not stated":
        return {"0"}
    start, end = stated.split("-")
    minute = _hhmm_to_minute(asked.group(2))
    if _hhmm_to_minute(start) <= minute < _hhmm_to_minute(end):
        return {"1"}
    return {"2"}


def _visible_nearest(context: str, question: str, options) -> set[str]:
    origin = re.search(
        rf"Origin ({_PLACE_RE}) is at \((-?\d+) m, (-?\d+) m\)\.",
        context,
    )
    asked = re.fullmatch(
        rf"Which place is nearest to ({_PLACE_RE})\?(?: {re.escape(NEAREST_TIE_SENTENCE)}| {re.escape(ANY_NEAREST_SENTENCE)})?",
        question,
    )
    if origin is None or asked is None or origin.group(1) != asked.group(1):
        return set()
    rest = context[origin.end():]
    places = _coordinates(rest)
    origin_xy = (int(origin.group(2)), int(origin.group(3)))
    scored = []
    for option in options:
        hits = [place for place in places if place[0] == option]
        if len(hits) != 1:
            return set()
        distance = (hits[0][1] - origin_xy[0]) ** 2 + (hits[0][2] - origin_xy[1]) ** 2
        scored.append((distance, option))
    best = min(distance for distance, _name in scored)
    winners = [name for distance, name in scored if distance == best]
    if len(winners) != 1:
        if ANY_NEAREST_SENTENCE in question:
            return _option_labels(options, winners)
        if NEAREST_TIE_SENTENCE not in question:
            return set()
        winners = [min(winners)]
    return _option_label(options, winners[0])


def _visible_filtered(context: str, question: str, options) -> set[str]:
    asked = re.fullmatch(
        rf"Which place is open at (\d{{2}}:\d{{2}}) on ({_DAY_RE}) and has a rating of at least ([0-9]+(?:\.[0-9]+)?)\?",
        question,
    )
    if asked is None:
        return set()
    minute = _hhmm_to_minute(asked.group(1))
    day = asked.group(2)
    minimum = float(asked.group(3))
    scoped = re.findall(
        rf"({_PLACE_RE}) rating ([0-9]+(?:\.[0-9]+)?) hours ({_DAY_RE}) (\d{{2}}:\d{{2}})-(\d{{2}}:\d{{2}})\.",
        context,
    )
    by_name = {name: (float(rating), hour_day, start, end) for name, rating, hour_day, start, end in scoped}
    winners = []
    for option in options:
        row = by_name.get(option)
        if row is None:
            continue
        rating, hour_day, start, end = row
        if hour_day == day and rating >= minimum and _hhmm_to_minute(start) <= minute < _hhmm_to_minute(end):
            winners.append(option)
    if not winners:
        return {"0"}
    if len(winners) != 1:
        return set()
    return _option_label(options, winners[0])


def _visible_farthest(context: str, question: str, options) -> set[str]:
    base = "Which pair of places is farthest apart?"
    if question not in {base, f"{base} {FARTHEST_TIE_SENTENCE}", f"{base} {ANY_FARTHEST_SENTENCE}"}:
        return set()
    places = _coordinates(context)
    if len(places) < 2:
        return set()
    scored = []
    for left, right in itertools.combinations(places, 2):
        distance = (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2
        scored.append((distance, _pair_key(left[0], right[0])))
    best = max(distance for distance, _name in scored)
    winners = [name for distance, name in scored if distance == best]
    if len(winners) != 1:
        if ANY_FARTHEST_SENTENCE in question:
            return _option_labels(options, winners)
        if FARTHEST_TIE_SENTENCE not in question:
            return set()
        winners = [min(winners)]
    return _option_label(options, winners[0])


def _visible_direction(context: str, question: str, options) -> set[str]:
    asked = re.fullmatch(rf"Which way is ({_PLACE_RE}) from ({_PLACE_RE})\?", question)
    places = {name: (x_m, y_m) for name, x_m, y_m in _coordinates(context)}
    if asked is None or asked.group(1) not in places or asked.group(2) not in places:
        return set()
    signs = _axis_signs(context)
    if signs is None:
        return {"0"}
    x_sign, y_sign = signs
    target = places[asked.group(1)]
    origin = places[asked.group(2)]
    dx = x_sign * (target[0] - origin[0])
    dy = y_sign * (target[1] - origin[1])
    if dx == 0 and dy > 0:
        name = "North"
    elif dx > 0 and dy == 0:
        name = "East"
    elif dx == 0 and dy < 0:
        name = "South"
    elif dx < 0 and dy == 0:
        name = "West"
    else:
        return set()
    return _option_label(options, name)


def _visible_route(context: str, question: str, options) -> set[str]:
    asked = re.fullmatch(r"After reaching (.+), where should I go next\?", question)
    steps = re.findall(r"(\d+)\. (.+?)\.", context)
    if asked is None or not context.startswith("Route by foot:") or not steps:
        return set()
    ordered = [step for _number, step in steps]
    hits = [index for index, step in enumerate(ordered) if asked.group(1) in step]
    if len(hits) != 1 or hits[0] == len(ordered) - 1:
        return set()
    return _option_label(options, ordered[hits[0] + 1])


def _visible_trip(context: str, question: str, options) -> set[str]:
    if tuple(options) != _COUNT_WORDS or question != "What is the maximum number of listed places that can be visited?":
        return set()
    visit_match = re.search(r"Each listed place takes (\d+) minutes on site\.", context)
    travel_match = re.search(r"Travel between any two different places takes (\d+) minutes\.", context)
    budget_match = re.search(r"The time budget is (\d+) minutes\.", context)
    origin_match = re.search(rf"Start at ({_PLACE_RE})\.", context)
    names = re.findall(rf"Place: ({_PLACE_RE})\.", context)
    if not (visit_match and travel_match and budget_match and origin_match and names):
        return set()
    visit = int(visit_match.group(1))
    travel = int(travel_match.group(1))
    budget = int(budget_match.group(1))
    best = 0
    for size in range(1, len(names) + 1):
        elapsed = size * visit + size * travel
        if elapsed <= budget:
            best = size
    if best < 1 or best > 4:
        return set()
    return {str(best)}


def _mean_gap(groups) -> float:
    total = 0
    count = 0
    for group in groups:
        for left, right in group:
            total += int(bool(left)) - int(bool(right))
            count += 1
    if count == 0:
        raise SpatialProtocolError("bootstrap group has no pairs")
    return total / count


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise SpatialProtocolError("empty bootstrap")
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
