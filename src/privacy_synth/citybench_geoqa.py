"""CityBench GeoQA text roles, letter labels, and fictional scene checks.

This module does not load a model, does not call a teacher, and does not read
a sealed split to choose a setting. Gold letters stay in the local record.
The MapEval numeric prompt in spatial_scene.py is a different frozen contract.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from privacy_synth.attack_protocol import refuse_screening_as_attack
from privacy_synth.spatial_scene import (
    SpatialProtocolError,
    paired_cluster_diff,
    sha256_ids,
)

LETTERS = ("A", "B", "C", "D", "E", "F")
CATEGORIES = ("node", "landmark", "path", "districts", "boundary", "others")
ROLES = ("public_calibration", "local_dev", "sealed_evaluation")
ROLE_CAPS = {"public_calibration": 72, "local_dev": 256, "sealed_evaluation": None}
MIN_PUBLIC = 72
MIN_LOCAL_DEV = 200
MIN_SEALED = 500
PRECHECK_N = 8
MIN_QUALIFICATION = 64
STUDENT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
REQUEST_MODEL = "deepseek-flash"
STAGE_CALL_CAP = 800
STAGE_USD_CAP = 140
CALIBRATION_CALL_CAP = 160
ARM_CALL_CAP = 160
CITIES = (
    "Beijing",
    "Shanghai",
    "Mumbai",
    "Tokyo",
    "London",
    "Paris",
    "Moscow",
    "NewYork",
    "SanFrancisco",
    "SaoPaulo",
    "Nairobi",
    "CapeTown",
    "Sydney",
)
MAP_V1 = {
    "path": {
        "road_length": "eval_road_length.csv",
        "road_od": "eval_road_od.csv",
        "road_link": "eval_road_link.csv",
    },
    "node": {
        "poi2coor": "poi2coor.csv",
        "AOI_POI_road4": "AOI_POI_road4.csv",
        "aoi_near": "aoi_near.csv",
    },
    "landmark": {
        "landmark_env": "eval_landmark_env.csv",
        "landmark_path": "eval_landmark_path.csv",
    },
    "boundary": {
        "boundary_road": "eval_boundary_road.csv",
        "AOI_POI_road1": "AOI_POI_road1.csv",
        "AOI_POI_road2": "AOI_POI_road2.csv",
        "AOI_POI_road3": "AOI_POI_road3.csv",
    },
    "districts": {
        "aoi2type": "aoi2type.csv",
        "type2aoi": "type2aoi.csv",
        "aoi2addr": "aoi2addr.csv",
        "AOI_POI5": "AOI_POI5.csv",
        "AOI_POI6": "AOI_POI6.csv",
        "road_aoi": "eval_road_aoi.csv",
    },
    "others": {"AOI_POI3": "AOI_POI3.csv", "AOI_POI4": "AOI_POI4.csv"},
}
MAP_V2 = {
    "path": {
        "road_length": "road_length.csv",
        "road_od": "road_od.csv",
        "road_link": "road_link.csv",
        "road_arrived_pois": "road_arrived_pois.csv",
    },
    "node": {
        "poi2coor": "poi2coor.csv",
        "poi2addr": "poi2addr.csv",
        "poi2type": "poi2type.csv",
        "type2poi": "type2poi.csv",
        "AOI_POI_road4": "AOI_POI_road4.csv",
    },
    "landmark": {"landmark_env": "landmark_env.csv", "landmark_path": "landmark_path.csv"},
    "boundary": {
        "boundary_road": "boundary_road.csv",
        "aoi_boundary_poi": "aoi_boundary_poi.csv",
        "AOI_POI_road1": "AOI_POI_road1.csv",
        "AOI_POI_road2": "AOI_POI_road2.csv",
        "AOI_POI_road3": "AOI_POI_road3.csv",
    },
    "districts": {
        "aoi2type": "aoi2type.csv",
        "type2aoi": "type2aoi.csv",
        "aoi_poi": "aoi_poi.csv",
        "poi_aoi": "poi_aoi.csv",
        "aoi_group": "aoi_group.csv",
        "aoi2addr": "aoi2addr.csv",
        "districts_poi_type": "districts_poi_type.csv",
        "AOI_POI5": "AOI_POI5.csv",
        "AOI_POI6": "AOI_POI6.csv",
    },
    "others": {
        "AOI_POI": "AOI_POI.csv",
        "AOI_POI2": "AOI_POI2.csv",
        "AOI_POI3": "AOI_POI3.csv",
        "AOI_POI4": "AOI_POI4.csv",
    },
}


def _task_table(mapping: dict) -> dict[str, tuple[str, str]]:
    return {
        filename: (category, task)
        for category, tasks in mapping.items()
        for task, filename in tasks.items()
    }


TASK_OF = {"Beijing": _task_table(MAP_V2), "other": _task_table(MAP_V1)}
ARMS = (
    "generic_utility_off",
    "generic_utility_on",
    "task_aware_utility_off",
    "task_aware_utility_on",
)
KINDS = (
    "node_place",
    "landmark_type",
    "path_length",
    "district_member",
    "boundary_side",
    "other_open",
)
KIND_CATEGORY = {
    "node_place": "node",
    "landmark_type": "landmark",
    "path_length": "path",
    "district_member": "districts",
    "boundary_side": "boundary",
    "other_open": "others",
}
_ANSWER_LINE = re.compile(r"(?m)^answer:\s*([A-Fa-f])\s*$")
_JSON_OPTION = re.compile(r'"option"\s*:\s*"([A-Fa-f])"')
_PLACES = ("Orin Quay", "Bram Ferry", "Lumen Court", "Pell Yard", "Sable Market", "Ivo Dock")
_TYPES = ("cafe", "library", "museum", "park")
_ROADS = ("Quill Lane", "Harrow Bend", "Nera Way", "Cinder Road")
_DISTRICTS = ("North Ward", "South Ward", "East Ward", "West Ward")


@dataclass(frozen=True)
class GeoqaRecord:
    id: int
    city: str
    category: str
    task: str
    question: str
    options: tuple[tuple[str, str], ...]
    gold_label: str
    source: str
    group_id: str = ""
    role: str = ""

    def render_prompt(self) -> str:
        return render_exam_prompt(self.question, self.options)


@dataclass(frozen=True)
class Scene:
    scene_id: str
    kind: str
    context: str
    question: str
    options: tuple[tuple[str, str], ...]
    claimed_label: str
    world: str

    def render_prompt(self) -> str:
        return render_scene_prompt(self)


@dataclass(frozen=True)
class RoleAssignment:
    records: tuple[GeoqaRecord, ...]
    precheck_ids: tuple[int, ...]
    qualification_ids: tuple[int, ...]
    dropped: int

    def role_records(self, role: str) -> tuple[GeoqaRecord, ...]:
        if role not in ROLES:
            raise SpatialProtocolError(f"unknown role {role}")
        return tuple(record for record in self.records if record.role == role)


def refuse_sealed_path(path: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if "sealed_evaluation" in lowered or "sealed-evaluation" in lowered or "sealed_test" in lowered:
        raise SpatialProtocolError(f"sealed path is refused: {path}")


def file_task(city: str, filename: str) -> tuple[str, str]:
    table = TASK_OF["Beijing"] if city == "Beijing" else TASK_OF["other"]
    found = table.get(filename)
    if found is None:
        raise SpatialProtocolError(f"{city} file {filename} is outside the official GeoQA map")
    return found


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _group_id(city: str, category: str, task: str, question: str) -> str:
    blob = "\n".join((city, category, task, _norm(question))).encode()
    return hashlib.sha256(blob).hexdigest()


def options_from_row(row: dict) -> tuple[tuple[str, str], ...] | None:
    """Keep A–F in order. A blank before a later option drops the row."""
    kept: list[tuple[str, str]] = []
    seen_blank = False
    present = [letter for letter in LETTERS if letter in row]
    if not present:
        return None
    for letter in present:
        text = str(row.get(letter) or "").strip()
        if text == "":
            seen_blank = True
            continue
        if seen_blank:
            return None
        kept.append((letter, text))
    if len(kept) < 2:
        return None
    return tuple(kept)


def parse_csv_rows(city: str, filename: str, text: str) -> tuple[list[dict], int]:
    """Return valid row dicts and the number of dropped data rows."""
    category, task = file_task(city, filename)
    handle = io.StringIO(text)
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or "question" not in reader.fieldnames or "answer" not in reader.fieldnames:
        raise SpatialProtocolError(f"{city}/{filename} is missing question or answer")
    valid = []
    dropped = 0
    for source_row, row in enumerate(reader):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip().upper()
        options = options_from_row(row)
        letters = {letter for letter, _text in options} if options else set()
        if (
            not question
            or answer not in letters
            or options is None
            or _ANSWER_LINE.search(question)
        ):
            dropped += 1
            continue
        valid.append(
            {
                "city": city,
                "category": category,
                "task": task,
                "question": question,
                "options": options,
                "gold_label": answer,
                "source": f"{city}/v1/{filename}#{source_row}",
                "source_row": source_row,
            }
        )
    return valid, dropped


def load_extracted_tree(root: Path) -> tuple[tuple[GeoqaRecord, ...], int]:
    """Load the official mapped CSVs. `is_seen` is ignored and not stored."""
    pending = []
    dropped = 0
    for city in CITIES:
        mapping = MAP_V2 if city == "Beijing" else MAP_V1
        city_dir = root / city
        for _category, tasks in mapping.items():
            for filename in tasks.values():
                path = city_dir / filename
                if not path.is_file():
                    continue
                rows, lost = parse_csv_rows(city, filename, path.read_text(encoding="utf-8"))
                pending.extend(rows)
                dropped += lost
    pending.sort(key=lambda row: (row["city"], row["category"], row["task"], row["source_row"], row["question"]))
    records = tuple(
        GeoqaRecord(
            id=index,
            city=row["city"],
            category=row["category"],
            task=row["task"],
            question=row["question"],
            options=row["options"],
            gold_label=row["gold_label"],
            source=row["source"],
            group_id=_group_id(row["city"], row["category"], row["task"], row["question"]),
        )
        for index, row in enumerate(pending, start=1)
    )
    if len({record.id for record in records}) != len(records):
        raise SpatialProtocolError("duplicate ids")
    return records, dropped


def legal_labels(options) -> tuple[str, ...]:
    letters = tuple(letter for letter, _text in options)
    if letters != tuple(sorted(letters)):
        raise SpatialProtocolError("option letters are not in A–F order")
    if len(letters) < 2 or len(set(letters)) != len(letters):
        raise SpatialProtocolError("options are not distinct legal letters")
    if any(letter not in LETTERS for letter in letters):
        raise SpatialProtocolError("option letter is outside A–F")
    return letters


def render_exam_prompt(question: str, options) -> str:
    """Letter prompt for GeoQA. It does not offer an Unanswerable zero."""
    labels = legal_labels(options)
    by_letter = {letter: text for letter, text in options}
    lines = [
        "The following is a multiple-choice question about the geospatial knowledge of a city.",
        "Choose the most suitable option. The answer may require knowledge of the named places.",
        "Question:",
        str(question).strip(),
        "Options:",
    ]
    for letter in labels:
        lines.append(f"{letter}. {by_letter[letter].strip()}")
    lines.append(
        'Reply with JSON {"option": "<letter>", "explanation": "<reason>"}. '
        "End with a final line of the form answer: <letter>."
    )
    lines.append("answer:")
    return "\n".join(lines) + " "


def render_scene_prompt(scene: Scene) -> str:
    question = scene.context.strip() + "\n" + scene.question.strip()
    return render_exam_prompt(question, scene.options)


def parse_teacher_label(text: str | None, legal) -> str | None:
    """Last answer line wins. A missing or illegal letter stays missing."""
    if text is None:
        return None
    allowed = {str(label).upper() for label in legal}
    lines = [item.upper() for item in _ANSWER_LINE.findall(text)]
    if lines:
        label = lines[-1]
        return label if label in allowed else None
    found = [item.upper() for item in _JSON_OPTION.findall(text)]
    if not found:
        return None
    label = found[-1]
    return label if label in allowed else None


def supervision_label(label: str) -> str:
    if label not in LETTERS:
        raise SpatialProtocolError("supervision must be one option letter")
    return label


def _category_counts(records) -> Counter:
    return Counter(record.category for record in records)


def assign_roles(records: tuple[GeoqaRecord, ...], dropped: int = 0) -> RoleAssignment:
    """Assign intact question groups. This function does not look at a score."""
    if len(records) < MIN_PUBLIC + MIN_LOCAL_DEV + MIN_SEALED:
        raise SpatialProtocolError("GeoQA does not have enough valid rows for the frozen sizes")
    grouped: dict[str, list[GeoqaRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    groups = [tuple(sorted(members, key=lambda record: record.id)) for members in grouped.values()]
    unassigned = {group[0].group_id: group for group in groups}
    placed: dict[str, list[GeoqaRecord]] = {role: [] for role in ROLES}
    used = {role: 0 for role in ROLES}
    role_counts = {role: Counter() for role in ROLES}
    group_counts = {group[0].group_id: _category_counts(group) for group in groups}

    def fits(role: str, size: int) -> bool:
        cap = ROLE_CAPS[role]
        return True if cap is None else used[role] + size <= cap

    def take(role: str, group: tuple[GeoqaRecord, ...]) -> None:
        key = group[0].group_id
        if key not in unassigned:
            raise SpatialProtocolError("group assigned twice")
        del unassigned[key]
        placed[role].extend(group)
        used[role] += len(group)
        role_counts[role].update(group_counts[key])

    for role in ROLES:
        for category in CATEGORIES:
            if role_counts[role][category] > 0:
                continue
            candidates = [
                group
                for group in unassigned.values()
                if fits(role, len(group)) and any(record.category == category for record in group)
            ]
            if not candidates:
                raise SpatialProtocolError(f"{role} cannot receive category {category}")
            chosen = min(
                candidates,
                key=lambda group: (
                    len(group),
                    -sum(record.category == category for record in group) / len(group),
                    group[0].id,
                ),
            )
            take(role, chosen)
    print("citybench assign: categories seeded", file=sys.stderr, flush=True)

    global_counts = _category_counts(records)
    total = len(records)
    targets = {
        "public_calibration": ROLE_CAPS["public_calibration"],
        "local_dev": ROLE_CAPS["local_dev"],
        "sealed_evaluation": total - ROLE_CAPS["public_calibration"] - ROLE_CAPS["local_dev"],
    }

    def error_after(role: str, group: tuple[GeoqaRecord, ...]) -> float:
        counts = role_counts[role]
        extra = group_counts[group[0].group_id]
        error = 0.0
        for category, category_total in global_counts.items():
            want = targets[role] * category_total / total
            error += abs(counts[category] + extra[category] - want)
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
    print(
        "citybench assign: main pass "
        + " ".join(f"{role}={used[role]}" for role in ROLES),
        file=sys.stderr,
        flush=True,
    )

    def removal_ok(group: tuple[GeoqaRecord, ...]) -> bool:
        if used["sealed_evaluation"] - len(group) < MIN_SEALED:
            return False
        sealed_counts = role_counts["sealed_evaluation"]
        extra = group_counts[group[0].group_id]
        return all(sealed_counts[category] - extra[category] >= 1 for category in CATEGORIES)

    sealed_members: dict[str, list[GeoqaRecord]] = defaultdict(list)
    for record in placed["sealed_evaluation"]:
        sealed_members[record.group_id].append(record)
    sealed_index = {
        key: tuple(sorted(members, key=lambda record: record.id))
        for key, members in sealed_members.items()
    }
    by_size: dict[int, list[tuple[GeoqaRecord, ...]]] = defaultdict(list)
    for group in sealed_index.values():
        by_size[len(group)].append(group)
    for groups in by_size.values():
        groups.sort(key=lambda group: group[0].id)
    cursors = {size: 0 for size in by_size}

    def take_from(size: int) -> tuple[GeoqaRecord, ...] | None:
        groups = by_size.get(size)
        if not groups:
            return None
        index = cursors[size]
        while index < len(groups):
            group = groups[index]
            index += 1
            if group[0].group_id not in sealed_index or not removal_ok(group):
                continue
            cursors[size] = index
            return group
        cursors[size] = index
        return None

    for role, target in (("public_calibration", targets["public_calibration"]), ("local_dev", targets["local_dev"])):
        while used[role] < target:
            gap = target - used[role]
            chosen = take_from(gap)
            if chosen is None:
                for size in range(gap - 1, 0, -1):
                    chosen = take_from(size)
                    if chosen is not None:
                        break
            if chosen is None:
                break
            chosen_key = chosen[0].group_id
            del sealed_index[chosen_key]
            used["sealed_evaluation"] -= len(chosen)
            role_counts["sealed_evaluation"].subtract(group_counts[chosen_key])
            placed[role].extend(chosen)
            used[role] += len(chosen)
            role_counts[role].update(group_counts[chosen_key])
    placed["sealed_evaluation"] = [record for group in sealed_index.values() for record in group]
    print(
        "citybench assign: fill "
        + " ".join(f"{role}={used[role]}" for role in ROLES),
        file=sys.stderr,
        flush=True,
    )

    if used["public_calibration"] < MIN_PUBLIC or used["public_calibration"] > ROLE_CAPS["public_calibration"]:
        raise SpatialProtocolError("public calibration size is outside 72")
    if used["local_dev"] < MIN_LOCAL_DEV or used["local_dev"] > ROLE_CAPS["local_dev"]:
        raise SpatialProtocolError("local dev size is outside 200 to 256")
    if used["sealed_evaluation"] < MIN_SEALED:
        raise SpatialProtocolError("sealed evaluation is below 500")
    if sum(used.values()) != len(records):
        raise SpatialProtocolError("role sizes do not cover the records")
    for role in ROLES:
        present = {record.category for record in placed[role]}
        if present != set(CATEGORIES):
            raise SpatialProtocolError(f"{role} misses a category")

    stamped = []
    for role in ROLES:
        for record in sorted(placed[role], key=lambda item: item.id):
            stamped.append(
                GeoqaRecord(
                    id=record.id,
                    city=record.city,
                    category=record.category,
                    task=record.task,
                    question=record.question,
                    options=record.options,
                    gold_label=record.gold_label,
                    source=record.source,
                    group_id=record.group_id,
                    role=role,
                )
            )
    bare = RoleAssignment(tuple(stamped), (), (), dropped)
    precheck = _precheck_ids(bare.role_records("public_calibration"))
    qualification = tuple(
        record.id for record in bare.role_records("public_calibration") if record.id not in set(precheck)
    )
    if len(qualification) < MIN_QUALIFICATION:
        raise SpatialProtocolError("qualification set is below 64")
    remaining = [record for record in bare.role_records("public_calibration") if record.id in set(qualification)]
    if {record.category for record in remaining} != set(CATEGORIES):
        raise SpatialProtocolError("qualification set misses a category")
    return RoleAssignment(bare.records, precheck, qualification, dropped)


def _precheck_ids(calibration: tuple[GeoqaRecord, ...]) -> tuple[int, ...]:
    counts = _category_counts(calibration)
    chosen = []
    for record in sorted(calibration, key=lambda item: item.id):
        if len(chosen) == PRECHECK_N:
            break
        if counts[record.category] - 1 < 1:
            continue
        counts[record.category] -= 1
        chosen.append(record.id)
    if len(chosen) != PRECHECK_N:
        raise SpatialProtocolError("format precheck could not keep every category in qualification")
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
            "category_counts": dict(_category_counts(records)),
            "group_count": len({record.group_id for record in records}),
        }
    qualification = [
        record
        for record in assignment.role_records("public_calibration")
        if record.id in set(assignment.qualification_ids)
    ]
    return {
        "version": "citybench-geoqa-v1",
        "student_revision": STUDENT_REVISION,
        "request_model": REQUEST_MODEL,
        "documented_model_version": "DeepSeek-V4.1-Flash",
        "calls_approved": False,
        "precheck_ids": list(assignment.precheck_ids),
        "precheck_sha256": sha256_ids(assignment.precheck_ids),
        "qualification_ids_sha256": sha256_ids(sorted(assignment.qualification_ids)),
        "qualification_count": len(assignment.qualification_ids),
        "qualification_category_counts": dict(_category_counts(qualification)),
        "dropped_rows": assignment.dropped,
        "roles": roles,
        "screening_reference_roles": ["public_calibration", "local_dev"],
        "utility_filter_rule": "unfrozen",
        "images_extracted": False,
        "official_eval_executed": False,
    }


def constant_letter_accuracy(gold_labels) -> dict:
    labels = [str(label) for label in gold_labels]
    if not labels:
        raise SpatialProtocolError("constant baseline needs labels")
    counts = Counter(labels)
    best = sorted(counts, key=lambda label: (-counts[label], label))[0]
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


def iter_scenes(mode: str, count: int, seed: int) -> tuple[Scene, ...]:
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
        try:
            scene = propose_scene(kind, rng, f"{mode}-{seed}-{len(accepted)}")
            if entailed_label(scene) != scene.claimed_label:
                continue
        except SpatialProtocolError:
            continue
        if not scene.context.startswith("North is the positive y direction"):
            continue
        accepted.append(scene)
    return tuple(accepted)


def propose_scene(kind: str, rng: random.Random, scene_id: str) -> Scene:
    builders = {
        "node_place": _node_place,
        "landmark_type": _landmark_type,
        "path_length": _path_length,
        "district_member": _district_member,
        "boundary_side": _boundary_side,
        "other_open": _other_open,
    }
    if kind not in builders:
        raise SpatialProtocolError(f"unknown scene kind {kind}")
    return builders[kind](rng, scene_id)


def _axis() -> str:
    return "North is the positive y direction and East is the positive x direction."


def _letter_options(texts: tuple[str, ...], claimed_index: int) -> tuple[tuple[tuple[str, str], ...], str]:
    options = tuple((LETTERS[index], text) for index, text in enumerate(texts))
    return options, LETTERS[claimed_index]


def _pack(kind: str, scene_id: str, facts: dict, context: str, question: str, texts: tuple[str, ...], claimed_index: int) -> Scene:
    options, label = _letter_options(texts, claimed_index)
    return Scene(
        scene_id=scene_id,
        kind=kind,
        context=_axis() + " " + context,
        question=question,
        options=options,
        claimed_label=label,
        world=json.dumps({"kind": kind, "facts": facts, "claimed_index": claimed_index}, sort_keys=True),
    )


def _node_place(rng: random.Random, scene_id: str) -> Scene:
    places = rng.sample(_PLACES, 4)
    point = places[rng.randrange(4)]
    slots = [(x * 100, y * 100) for x in range(3) for y in range(3)]
    coords = dict(zip(places, rng.sample(slots, 4)))
    listed = "; ".join(f"{place} is at ({coords[place][0]} m, {coords[place][1]} m)" for place in places)
    texts = tuple(places)
    return _pack(
        "node_place",
        scene_id,
        {"coords": coords, "point": point},
        listed + ".",
        f"Which place is at ({coords[point][0]} m, {coords[point][1]} m)?",
        texts,
        texts.index(point),
    )


def _landmark_type(rng: random.Random, scene_id: str) -> Scene:
    places = rng.sample(_PLACES, 4)
    types = list(_TYPES)
    rng.shuffle(types)
    typed = dict(zip(places, types))
    museum = next(place for place, kind in typed.items() if kind == "museum")
    listed = "; ".join(f"{place} is a {typed[place]}" for place in places)
    texts = tuple(places)
    return _pack(
        "landmark_type",
        scene_id,
        {"types": typed, "museum": museum},
        listed + ".",
        "Which place is a museum?",
        texts,
        texts.index(museum),
    )


def _path_length(rng: random.Random, scene_id: str) -> Scene:
    roads = rng.sample(_ROADS, 4)
    lengths = [100, 250, 400, 800]
    rng.shuffle(lengths)
    named = dict(zip(roads, lengths))
    longest = max(named, key=named.get)
    listed = "; ".join(f"{road} is {named[road]} m long" for road in roads)
    texts = tuple(roads)
    return _pack(
        "path_length",
        scene_id,
        {"lengths": named, "longest": longest},
        listed + ".",
        "Which road is longest?",
        texts,
        texts.index(longest),
    )


def _district_member(rng: random.Random, scene_id: str) -> Scene:
    places = rng.sample(_PLACES, 4)
    districts = list(_DISTRICTS)
    home = districts[rng.randrange(4)]
    assigned = {places[0]: home}
    others = [name for name in districts if name != home]
    for place, district in zip(places[1:], others):
        assigned[place] = district
    listed = "; ".join(f"{place} is in {assigned[place]}" for place in places)
    texts = tuple(districts)
    return _pack(
        "district_member",
        scene_id,
        {"assigned": assigned, "home": home, "anchor": places[0]},
        listed + ".",
        f"Which district contains {places[0]}?",
        texts,
        texts.index(home),
    )


def _boundary_side(rng: random.Random, scene_id: str) -> Scene:
    places = rng.sample(_PLACES, 4)
    side = {place: ("inside" if index == 0 else "outside") for index, place in enumerate(places)}
    inside = places[0]
    listed = "; ".join(f"{place} is {side[place]} the boundary" for place in places)
    texts = tuple(places)
    return _pack(
        "boundary_side",
        scene_id,
        {"side": side, "inside": inside},
        "The boundary is a closed line. " + listed + ".",
        "Which place is inside the boundary?",
        texts,
        texts.index(inside),
    )


def _other_open(rng: random.Random, scene_id: str) -> Scene:
    places = rng.sample(_PLACES, 4)
    open_place = places[rng.randrange(4)]
    hours = {place: ("09:00-17:00" if place == open_place else "closed") for place in places}
    listed = "; ".join(f"{place} hours are {hours[place]}" for place in places)
    texts = tuple(places)
    return _pack(
        "other_open",
        scene_id,
        {"hours": hours, "open": open_place, "asked": "12:00"},
        listed + ".",
        "Which place is open at 12:00?",
        texts,
        texts.index(open_place),
    )


def entailed_label(scene: Scene) -> str:
    payload = json.loads(scene.world)
    facts = payload["facts"]
    kind = payload["kind"]
    if kind == "node_place":
        point = facts["point"]
        coords = {name: tuple(value) for name, value in facts["coords"].items()}
        winners = [name for name, value in coords.items() if tuple(value) == tuple(coords[point])]
        if winners != [point]:
            raise SpatialProtocolError("node scene coordinate is ambiguous")
        return _label_of(scene, point)
    if kind == "landmark_type":
        museums = [name for name, value in facts["types"].items() if value == "museum"]
        if museums != [facts["museum"]]:
            raise SpatialProtocolError("landmark scene museum is ambiguous")
        return _label_of(scene, facts["museum"])
    if kind == "path_length":
        longest = max(facts["lengths"], key=facts["lengths"].get)
        if longest != facts["longest"]:
            raise SpatialProtocolError("path scene longest road does not match")
        return _label_of(scene, longest)
    if kind == "district_member":
        return _label_of(scene, facts["home"])
    if kind == "boundary_side":
        insides = [name for name, value in facts["side"].items() if value == "inside"]
        if insides != [facts["inside"]]:
            raise SpatialProtocolError("boundary scene inside place is ambiguous")
        return _label_of(scene, facts["inside"])
    if kind == "other_open":
        opened = [name for name, value in facts["hours"].items() if value != "closed"]
        if opened != [facts["open"]]:
            raise SpatialProtocolError("open scene is ambiguous")
        return _label_of(scene, facts["open"])
    raise SpatialProtocolError(f"unknown scene kind {kind}")


def _label_of(scene: Scene, text: str) -> str:
    matches = [letter for letter, option in scene.options if option == text]
    if len(matches) != 1:
        raise SpatialProtocolError("scene option text is not unique")
    return matches[0]


def copy_screen(scene: Scene, reference: tuple[GeoqaRecord, ...]) -> dict:
    """Exact question screen. This is not a privacy proof or an attack."""
    for record in reference:
        if record.role == "sealed_evaluation":
            raise SpatialProtocolError("sealed records cannot be a screen reference")
    question = _norm(scene.question)
    question_hit = any(_norm(record.question) == question for record in reference)
    screen = {
        "kind": "exact_overlap",
        "question_exact_hit": question_hit,
        "privacy_proof": False,
        "membership_inference": False,
    }
    try:
        refuse_screening_as_attack(screen)
    except ValueError:
        return screen
    raise SpatialProtocolError("copy screen was accepted as an attack")


def assert_request_prefix(short_ids, long_ids) -> None:
    if tuple(long_ids[: len(short_ids)]) != tuple(short_ids):
        raise SpatialProtocolError("the 64-request list is not a prefix of the 160-request list")
