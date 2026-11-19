"""One place per row, and scenes read only from a released table.

The row builder and the scene builder do not call a generator, a teacher, or
a network service. A road-graph summary does not enter the scene text.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass

from privacy_synth.spatial_scene import (
    ANY_FARTHEST_SENTENCE,
    ANY_NEAREST_SENTENCE,
    AXIS_SENTENCE,
    FARTHEST_TIE_SENTENCE,
    NEAREST_TIE_SENTENCE,
    SpatialProtocolError,
    labels_from_visible_text,
)

BIN_M = 50
METERS_PER_DEGREE = 111_320.0
# Public category codes. A value outside this tuple is the code "other".
# The tuple is not a frequency cutoff computed from a protected table.
PUBLIC_CATEGORIES = (
    "restaurant",
    "cafe",
    "fast_food",
    "bar",
    "pub",
    "atm",
    "bank",
    "pharmacy",
    "parking",
    "fuel",
    "toilets",
    "bench",
    "park",
    "taxi",
    "food_court",
    "waste_basket",
    "shop",
)
OTHER = "other"
HOURS_ABSENT = "absent"
HOURS_PRESENT = "present"
HOURS_CODE = {HOURS_ABSENT: 0, HOURS_PRESENT: 1}
FIRST_ROUND_KINDS = (
    "nearby_nearest",
    "farthest_pair",
    "direction",
    "category_nearest",
)
TIE_RULES = ("skip", "lexicographic_display_name", "any_tied_answer")
_STEMS = (
    "Arbor",
    "Birch",
    "Cedar",
    "Daisy",
    "Elm",
    "Fern",
    "Grove",
    "Hazel",
    "Ivy",
    "Juniper",
)
_WORDS = (
    "Alpha",
    "Bay",
    "Cove",
    "Dale",
    "Echo",
    "Ford",
    "Glen",
    "Hill",
    "Isle",
    "Jetty",
    "Knoll",
    "Lane",
    "Mesa",
    "Nook",
    "Oak",
    "Pond",
    "Quarry",
    "Ridge",
    "Spring",
)
SCENE_TABLE_LIMIT = len(_STEMS) * len(_WORDS)
_CARDINALS = ("North", "East", "South", "West")
_CATEGORY_RE = "|".join(sorted(PUBLIC_CATEGORIES + (OTHER,), key=len, reverse=True))
_PLACE_RE = r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"


@dataclass(frozen=True)
class PlaceRow:
    """One protected place. `entity_key` is removed before a table is released."""

    entity_key: str
    lat_bin: int
    lon_bin: int
    category: str
    hours_state: str


@dataclass(frozen=True)
class ReleasedPlace:
    """A row that scene writing is allowed to read. It has no entity key."""

    lat_bin: int
    lon_bin: int
    category: str
    hours_state: str


def public_domain(bbox: tuple[float, float, float, float], bin_m: int = BIN_M) -> dict:
    """Bin counts from the public bbox only."""
    south, west, north, east = bbox
    if not (south < north and west < east):
        raise SpatialProtocolError("bbox is empty")
    if bin_m <= 0:
        raise SpatialProtocolError("bin size must be positive")
    mid = (south + north) / 2.0
    height_m = (north - south) * METERS_PER_DEGREE
    width_m = (east - west) * METERS_PER_DEGREE * math.cos(math.radians(mid))
    return {
        "lat_bin": max(1, math.ceil(height_m / bin_m - 1e-9)),
        "lon_bin": max(1, math.ceil(width_m / bin_m - 1e-9)),
        "category": len(PUBLIC_CATEGORIES) + 1,
        "hours_state": 2,
        "bin_m": bin_m,
        "bbox": {"south": south, "west": west, "north": north, "east": east},
    }


def category_code(category: str) -> int:
    if category == OTHER:
        return len(PUBLIC_CATEGORIES)
    try:
        return PUBLIC_CATEGORIES.index(category)
    except ValueError as exc:
        raise SpatialProtocolError(f"unknown category {category}") from exc


def category_name(code: int) -> str:
    if code == len(PUBLIC_CATEGORIES):
        return OTHER
    if 0 <= code < len(PUBLIC_CATEGORIES):
        return PUBLIC_CATEGORIES[code]
    raise SpatialProtocolError(f"category code {code} is outside the public domain")


def _category(tags: dict) -> str | None:
    amenity = str(tags.get("amenity", "")).strip()
    shop = str(tags.get("shop", "")).strip()
    if amenity:
        return amenity if amenity in PUBLIC_CATEGORIES else OTHER
    if shop:
        return "shop"
    return None


def _bin_index(meters: float, bins: int, span_m: float, bin_m: int) -> int:
    index = int(meters // bin_m)
    if index == bins and meters <= span_m + 1e-6:
        index = bins - 1
    if index < 0 or index >= bins:
        raise SpatialProtocolError("coordinate is outside the public bin grid")
    return index


def place_from_element(element: dict, bbox: tuple[float, float, float, float], bin_m: int = BIN_M) -> PlaceRow | None:
    """Map one OSM element to one place row, or return None when it is not a place."""
    if element.get("type") != "node":
        return None
    tags = element.get("tags") or {}
    category = _category(tags)
    if category is None:
        return None
    lat = element.get("lat")
    lon = element.get("lon")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise SpatialProtocolError("place node is missing coordinates")
    south, west, north, east = bbox
    if not (south <= float(lat) <= north and west <= float(lon) <= east):
        raise SpatialProtocolError("place node is outside the public bbox")
    domain = public_domain(bbox, bin_m)
    mid = (south + north) / 2.0
    height_m = (north - south) * METERS_PER_DEGREE
    width_m = (east - west) * METERS_PER_DEGREE * math.cos(math.radians(mid))
    y_m = (float(lat) - south) * METERS_PER_DEGREE
    x_m = (float(lon) - west) * METERS_PER_DEGREE * math.cos(math.radians(mid))
    hours = str(tags.get("opening_hours", "")).strip()
    return PlaceRow(
        entity_key=f"node:{int(element['id'])}",
        lat_bin=_bin_index(y_m, domain["lat_bin"], height_m, bin_m),
        lon_bin=_bin_index(x_m, domain["lon_bin"], width_m, bin_m),
        category=category,
        hours_state=HOURS_PRESENT if hours else HOURS_ABSENT,
    )


def place_rows(elements, bbox: tuple[float, float, float, float], bin_m: int = BIN_M) -> tuple[PlaceRow, ...]:
    rows = []
    for element in elements:
        row = place_from_element(element, bbox, bin_m)
        if row is not None:
            rows.append(row)
    keys = [row.entity_key for row in rows]
    if len(keys) != len(set(keys)):
        raise SpatialProtocolError("one place produced more than one row")
    return tuple(rows)


def _features(row: PlaceRow | ReleasedPlace) -> tuple:
    return (row.lat_bin, row.lon_bin, row.category, row.hours_state)


def deletion_effect(rows: tuple[PlaceRow, ...] | list[PlaceRow], entity_key: str) -> dict:
    """Deleting one place key removes one feature row and no other place key."""
    kept = [row for row in rows if row.entity_key != entity_key]
    if len(kept) != len(rows) - 1:
        raise SpatialProtocolError("entity was not exactly one row")
    before = Counter(_features(row) for row in rows)
    after = Counter(_features(row) for row in kept)
    changed = [(key, before[key] - after[key]) for key in set(before) | set(after) if before[key] != after[key]]
    if changed != [(_features(next(row for row in rows if row.entity_key == entity_key)), 1)]:
        raise SpatialProtocolError("deletion changed another feature row")
    removed = {row.entity_key for row in rows} - {row.entity_key for row in kept}
    if removed != {entity_key}:
        raise SpatialProtocolError("deletion removed another entity")
    return {
        "removed_entity": entity_key,
        "feature_rows_removed": 1,
        "other_entities_unchanged": True,
    }


def release_features(rows) -> tuple[ReleasedPlace, ...]:
    """Drop entity keys. The result is the only table scene writing may read."""
    return tuple(
        ReleasedPlace(row.lat_bin, row.lon_bin, row.category, row.hours_state)
        for row in rows
    )


def aim_columns(rows) -> dict[str, list[int]]:
    """Integer columns for an existing tabular mechanism. Entity keys are absent."""
    released = release_features(rows)
    return {
        "lat_bin": [row.lat_bin for row in released],
        "lon_bin": [row.lon_bin for row in released],
        "category": [category_code(row.category) for row in released],
        "hours_state": [HOURS_CODE[row.hours_state] for row in released],
    }


def released_from_codes(lat_bins, lon_bins, categories, hours_states) -> tuple[ReleasedPlace, ...]:
    """Rebuild released rows from a mechanism's integer table and public codes."""
    count = len(lat_bins)
    if not (len(lon_bins) == len(categories) == len(hours_states) == count):
        raise SpatialProtocolError("released code columns have different lengths")
    rows = []
    for index in range(count):
        hours_code = int(hours_states[index])
        if hours_code not in (0, 1):
            raise SpatialProtocolError("hours code is outside the public domain")
        rows.append(
            ReleasedPlace(
                int(lat_bins[index]),
                int(lon_bins[index]),
                category_name(int(categories[index])),
                HOURS_ABSENT if hours_code == 0 else HOURS_PRESENT,
            )
        )
    return tuple(rows)


def road_graph_report(elements) -> dict:
    """Summarize way member lists. This report is not an input to scene text."""
    ways = [
        element
        for element in elements
        if element.get("type") == "way" and "highway" in (element.get("tags") or {})
    ]
    adjacency: dict[int, set[int]] = {}
    two_refs = 0
    two_points = 0
    for way in ways:
        refs = [int(node_id) for node_id in (way.get("nodes") or [])]
        if len(refs) >= 2:
            two_refs += 1
        if len(way.get("geometry") or []) >= 2:
            two_points += 1
        for left, right in zip(refs, refs[1:]):
            if left == right:
                continue
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
    seen: set[int] = set()
    components = 0
    for start in adjacency:
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current] - seen)
    edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return {
        "highway_ways": len(ways),
        "ways_with_at_least_two_node_refs": two_refs,
        "ways_with_at_least_two_geometry_points": two_points,
        "graph_nodes": len(adjacency),
        "graph_undirected_edges": edges,
        "components": components,
        "used_by_scene_builder": False,
    }


def _display_name(index: int) -> str:
    span = len(_WORDS)
    if index < 0 or index >= len(_STEMS) * span:
        raise SpatialProtocolError("released table exceeds the public display names")
    return f"{_STEMS[index // span]} {_WORDS[index % span]}"


def _named(released: tuple[ReleasedPlace, ...]):
    if not released:
        raise SpatialProtocolError("released table is empty")
    y0 = min(row.lat_bin for row in released)
    x0 = min(row.lon_bin for row in released)
    named = []
    for index, row in enumerate(released):
        named.append(
            {
                "name": _display_name(index),
                "x_m": (row.lon_bin - x0) * BIN_M,
                "y_m": (row.lat_bin - y0) * BIN_M,
                "category": row.category,
                "hours_state": row.hours_state,
            }
        )
    return named


def _squared(left: dict, right: dict) -> int:
    return (left["x_m"] - right["x_m"]) ** 2 + (left["y_m"] - right["y_m"]) ** 2


def _pair_key(left: str, right: str) -> str:
    ordered = tuple(sorted((left, right)))
    return f"{ordered[0]} and {ordered[1]}"


def category_nearest_labels(context: str, question: str, options) -> set[str]:
    """Entail a category-nearest label from the rendered text alone."""
    if AXIS_SENTENCE not in context:
        return {"0"}
    asked = re.fullmatch(
        rf"Which ({_CATEGORY_RE}) is nearest to ({_PLACE_RE})\?(?: {re.escape(NEAREST_TIE_SENTENCE)}| {re.escape(ANY_NEAREST_SENTENCE)})?",
        question,
    )
    origin = re.search(
        rf"Origin ({_PLACE_RE}) is at \((-?\d+) m, (-?\d+) m\)\.",
        context,
    )
    if asked is None or origin is None or origin.group(1) != asked.group(2):
        return set()
    found = re.findall(
        rf"({_PLACE_RE}) is a ({_CATEGORY_RE}) at \((-?\d+) m, (-?\d+) m\)\.",
        context,
    )
    origin_xy = (int(origin.group(2)), int(origin.group(3)))
    scored = []
    for name, category, x_m, y_m in found:
        if category != asked.group(1):
            continue
        if name not in options:
            continue
        distance = (int(x_m) - origin_xy[0]) ** 2 + (int(y_m) - origin_xy[1]) ** 2
        scored.append((distance, name))
    if not scored:
        return {"0"}
    best = min(distance for distance, _name in scored)
    winners = [name for distance, name in scored if distance == best]
    if len(winners) != 1:
        if ANY_NEAREST_SENTENCE in question:
            labels: set[str] = set()
            for name in winners:
                hits = [str(index) for index, option in enumerate(options, start=1) if option == name]
                if len(hits) != 1:
                    return set()
                labels.add(hits[0])
            return labels
        if NEAREST_TIE_SENTENCE not in question:
            return set()
        winners = [min(winners)]
    hits = [str(index) for index, option in enumerate(options, start=1) if option == winners[0]]
    if len(hits) != 1:
        return set()
    return {hits[0]}


def _scene(kind: str, context: str, question: str, options: tuple[str, ...], claimed: str) -> dict:
    if kind == "category_nearest":
        visible = category_nearest_labels(context, question, options)
    else:
        visible = labels_from_visible_text(kind, context, question, options)
    text = f"{context} {question} {' '.join(options)}"
    return {
        "kind": kind,
        "context": context,
        "question": question,
        "options": list(options),
        "claimed_label": claimed,
        "visible_labels": sorted(visible),
        "agrees": visible == {claimed}
        or (
            claimed in visible
            and len(visible) > 1
            and (
                ANY_NEAREST_SENTENCE in question
                or ANY_FARTHEST_SENTENCE in question
            )
        ),
        "mentions_clock": bool(re.search(r"\d{2}:\d{2}", text)),
        "mentions_rating": "rating" in text.lower(),
    }


def nearest_population(released) -> dict:
    """Count origins whose nearest other released place is unique."""
    named = _named(tuple(released))
    unique = tied = short = 0
    for origin in named:
        others = [place for place in named if place["name"] != origin["name"]]
        if len(others) < 2:
            short += 1
            continue
        best = min(_squared(origin, place) for place in others)
        winners = sum(_squared(origin, place) == best for place in others)
        if winners == 1:
            unique += 1
        else:
            tied += 1
    scored = unique + tied
    return {
        "origins": unique + tied + short,
        "unique_nearest": unique,
        "tied_nearest": tied,
        "too_few_others": short,
        "tie_rate_among_scored": (tied / scored) if scored else None,
    }


def _ok(scene: dict) -> tuple[dict, dict]:
    return scene, {"reason": "formed"}


def _skip(reason: str, **detail) -> tuple[None, dict]:
    return None, {"reason": reason, **detail}


def _nearest(named, origin: dict, rng: random.Random, tie_rule: str) -> tuple[dict | None, dict]:
    others = [place for place in named if place["name"] != origin["name"]]
    if len(others) < 2:
        return _skip("fewer_than_two_others")
    best = min(_squared(origin, place) for place in others)
    winners = [place for place in others if _squared(origin, place) == best]
    if len(winners) != 1 and tie_rule == "skip":
        return _skip("nearest_tie", tied_count=len(winners))
    chosen = list(winners)
    farther = [place for place in others if _squared(origin, place) > best]
    rng.shuffle(farther)
    chosen.extend(farther[:3])
    if len(chosen) < 2:
        return _skip("fewer_than_two_options")
    rng.shuffle(chosen)
    options = tuple(place["name"] for place in chosen)
    claimed_name = min(winners, key=lambda place: place["name"])["name"]
    context = (
        f"{AXIS_SENTENCE} Origin {origin['name']} is at ({origin['x_m']} m, {origin['y_m']} m). "
        + " ".join(f"{place['name']} is at ({place['x_m']} m, {place['y_m']} m)." for place in chosen)
    )
    question = f"Which place is nearest to {origin['name']}?"
    if tie_rule == "lexicographic_display_name":
        question = f"{question} {NEAREST_TIE_SENTENCE}"
    elif tie_rule == "any_tied_answer":
        question = f"{question} {ANY_NEAREST_SENTENCE}"
    return _ok(_scene(kind="nearby_nearest", context=context, question=question, options=options, claimed=str(options.index(claimed_name) + 1)))


def _farthest(named, rng: random.Random, tie_rule: str) -> tuple[dict | None, dict]:
    if len(named) < 4:
        return _skip("fewer_than_four_places")
    picks = rng.sample(named, 4)
    scored = []
    for left_index, left in enumerate(picks):
        for right in picks[left_index + 1 :]:
            scored.append((_squared(left, right), _pair_key(left["name"], right["name"])))
    best = max(distance for distance, _name in scored)
    winners = [name for distance, name in scored if distance == best]
    if len(winners) != 1 and tie_rule == "skip":
        return _skip("farthest_tie", tied_count=len(winners))
    claimed = min(winners)
    distractors = [name for _distance, name in scored if name not in winners]
    rng.shuffle(distractors)
    options_list = [*winners, *distractors[:3]]
    rng.shuffle(options_list)
    options = tuple(options_list)
    context = f"{AXIS_SENTENCE} " + " ".join(
        f"{place['name']} is at ({place['x_m']} m, {place['y_m']} m)." for place in picks
    )
    question = "Which pair of places is farthest apart?"
    if tie_rule == "lexicographic_display_name":
        question = f"{question} {FARTHEST_TIE_SENTENCE}"
    elif tie_rule == "any_tied_answer":
        question = f"{question} {ANY_FARTHEST_SENTENCE}"
    return _ok(_scene("farthest_pair", context, question, options, str(options.index(claimed) + 1)))


def _direction(named, rng: random.Random, tie_rule: str) -> tuple[dict | None, dict]:
    del tie_rule
    pairs = []
    for origin in named:
        for target in named:
            if target["name"] == origin["name"]:
                continue
            dx = target["x_m"] - origin["x_m"]
            dy = target["y_m"] - origin["y_m"]
            if dx == 0 and dy == 0:
                continue
            if dx == 0 or dy == 0:
                pairs.append((origin, target))
    if not pairs:
        return _skip("no_axis_aligned_pair", places=len(named))
    origin, target = pairs[rng.randrange(len(pairs))]
    dx = target["x_m"] - origin["x_m"]
    dy = target["y_m"] - origin["y_m"]
    if dx == 0 and dy > 0:
        answer = "North"
    elif dx > 0 and dy == 0:
        answer = "East"
    elif dx == 0 and dy < 0:
        answer = "South"
    elif dx < 0 and dy == 0:
        answer = "West"
    else:
        return _skip("direction_not_cardinal")
    context = (
        f"{AXIS_SENTENCE} {target['name']} is at ({target['x_m']} m, {target['y_m']} m). "
        f"{origin['name']} is at ({origin['x_m']} m, {origin['y_m']} m)."
    )
    question = f"Which way is {target['name']} from {origin['name']}?"
    return _ok(_scene("direction", context, question, _CARDINALS, str(_CARDINALS.index(answer) + 1)))


def _category_nearest(named, rng: random.Random, tie_rule: str) -> tuple[dict | None, dict]:
    by_category: dict[str, list[dict]] = {}
    for place in named:
        by_category.setdefault(place["category"], []).append(place)
    candidates = []
    tied_pairs = 0
    for origin in named:
        for category, places in by_category.items():
            others = [place for place in places if place["name"] != origin["name"]]
            if len(others) < 2:
                continue
            best = min(_squared(origin, place) for place in others)
            if sum(_squared(origin, place) == best for place in others) != 1 and tie_rule == "skip":
                tied_pairs += 1
                continue
            candidates.append((origin, category, others, best))
    if not candidates:
        return _skip("no_unique_category_neighbor", tied_pairs=tied_pairs)
    origin, category, others, best = candidates[rng.randrange(len(candidates))]
    winners = [place for place in others if _squared(origin, place) == best]
    chosen = list(winners)
    farther = [place for place in others if _squared(origin, place) > best]
    rng.shuffle(farther)
    chosen.extend(farther[:3])
    # A distractor of another category makes the category constraint visible.
    outsiders = [place for place in named if place["category"] != category and place["name"] != origin["name"]]
    rng.shuffle(outsiders)
    if outsiders:
        chosen.append(outsiders[0])
    rng.shuffle(chosen)
    options = tuple(place["name"] for place in chosen)
    claimed_name = min(winners, key=lambda place: place["name"])["name"]
    context = (
        f"{AXIS_SENTENCE} Origin {origin['name']} is at ({origin['x_m']} m, {origin['y_m']} m). "
        + " ".join(
            f"{place['name']} is a {place['category']} at ({place['x_m']} m, {place['y_m']} m)."
            for place in chosen
        )
    )
    question = f"Which {category} is nearest to {origin['name']}?"
    if tie_rule == "lexicographic_display_name":
        question = f"{question} {NEAREST_TIE_SENTENCE}"
    elif tie_rule == "any_tied_answer":
        question = f"{question} {ANY_NEAREST_SENTENCE}"
    return _ok(
        _scene(
            "category_nearest",
            context,
            question,
            options,
            str(options.index(claimed_name) + 1),
        )
    )


def _build(kind: str, named, rng: random.Random, tie_rule: str):
    if kind == "nearby_nearest":
        return _nearest(named, named[rng.randrange(len(named))], rng, tie_rule)
    if kind == "farthest_pair":
        return _farthest(named, rng, tie_rule)
    if kind == "direction":
        return _direction(named, rng, tie_rule)
    if kind == "category_nearest":
        return _category_nearest(named, rng, tie_rule)
    raise SpatialProtocolError(f"unknown question kind {kind}")


def collect_kind_quotas(
    released,
    *,
    seed: int,
    quotas: dict[str, int],
    tie_rule: str = "skip",
    max_attempts: int | None = None,
) -> dict:
    """Draw until each kind reaches its quota, or the attempt cap stops the draw.

    A scene that disagrees with the visible-text solver is not kept. Duplicate
    visible text is not counted twice. This does not read an entity key.
    """
    rows = tuple(released)
    for row in rows:
        if not isinstance(row, ReleasedPlace):
            raise SpatialProtocolError("scene builder received a row that is not a released place")
    if tie_rule not in TIE_RULES:
        raise SpatialProtocolError(f"unknown tie rule {tie_rule}")
    unknown = set(quotas) - set(FIRST_ROUND_KINDS)
    if unknown:
        raise SpatialProtocolError(f"unknown question kind {sorted(unknown)}")
    wanted = {kind: int(quotas.get(kind, 0)) for kind in FIRST_ROUND_KINDS}
    if any(count < 0 for count in wanted.values()):
        raise SpatialProtocolError("question quota is negative")
    cap = sum(wanted.values()) * 20 if max_attempts is None else int(max_attempts)
    if cap < 0:
        raise SpatialProtocolError("attempt cap is negative")
    named = _named(rows)
    rng = random.Random(seed)
    formed = {kind: [] for kind in FIRST_ROUND_KINDS}
    seen: set[tuple] = set()
    attempts = {kind: 0 for kind in FIRST_ROUND_KINDS}
    skip_reasons = {kind: Counter() for kind in FIRST_ROUND_KINDS}
    used = 0
    while used < cap and any(len(formed[kind]) < wanted[kind] for kind in FIRST_ROUND_KINDS):
        pending = [kind for kind in FIRST_ROUND_KINDS if len(formed[kind]) < wanted[kind]]
        kind = pending[used % len(pending)]
        used += 1
        attempts[kind] += 1
        scene, detail = _build(kind, named, rng, tie_rule)
        if scene is None:
            skip_reasons[kind][str(detail["reason"])] += 1
            continue
        if not scene["agrees"]:
            skip_reasons[kind]["visible_disagreement"] += 1
            continue
        key = (scene["kind"], scene["context"], scene["question"], tuple(scene["options"]))
        if key in seen:
            skip_reasons[kind]["duplicate"] += 1
            continue
        seen.add(key)
        formed[kind].append(scene)
    return {
        "tie_rule": tie_rule,
        "seed": seed,
        "quotas": wanted,
        "attempts": attempts,
        "formed": {kind: len(formed[kind]) for kind in FIRST_ROUND_KINDS},
        "short": {
            kind: wanted[kind] - len(formed[kind])
            for kind in FIRST_ROUND_KINDS
            if len(formed[kind]) < wanted[kind]
        },
        "skip_reasons": {kind: dict(skip_reasons[kind]) for kind in FIRST_ROUND_KINDS if skip_reasons[kind]},
        "proposals": [scene for kind in FIRST_ROUND_KINDS for scene in formed[kind]],
        "reads_entity_keys": False,
    }


def sample_scene_table(released, *, size: int, seed: int) -> tuple[ReleasedPlace, ...]:
    """Draw a public-size scene table. Names exist only for this many rows."""
    rows = tuple(released)
    if size < 2 or size > SCENE_TABLE_LIMIT:
        raise SpatialProtocolError("scene table size is outside the public name limit")
    if len(rows) < size:
        raise SpatialProtocolError("released table is smaller than the scene table")
    for row in rows:
        if not isinstance(row, ReleasedPlace):
            raise SpatialProtocolError("scene sample received a row that is not a released place")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return tuple(rows[index] for index in order[:size])


def teacher_requests(report: dict) -> list[dict]:
    """Copy the visible question only. Labels and entity keys stay out."""
    requests = []
    for scene in report["proposals"]:
        payload = {
            "kind": scene["kind"],
            "context": scene["context"],
            "question": scene["question"],
            "options": list(scene["options"]),
        }
        blob = f"{payload['context']} {payload['question']} {' '.join(payload['options'])}"
        if "node:" in blob or "entity_key" in blob or "claimed_label" in payload:
            raise SpatialProtocolError("teacher request contains a closed field")
        requests.append(payload)
    return requests


def export_teacher_packet(
    released,
    *,
    seed: int,
    per_kind: int,
    tie_rule: str,
    scene_table_size: int,
    mode: str = "task_aware",
) -> dict:
    """Sample a scene table and copy only the visible teacher text.

    The caller passes released rows or codes that were already emitted.
    This function does not fit a generator and does not read a sealed list.
    """
    size = min(int(scene_table_size), len(tuple(released)))
    table = sample_scene_table(released, size=size, seed=seed)
    report = scenes_from_released(
        table,
        seed=seed,
        per_kind=per_kind,
        mode=mode,
        tie_rule=tie_rule,
    )
    requests = teacher_requests(report)
    proposals = int(report["proposal_count"])
    return {
        "scene_table_size": size,
        "mode": mode,
        "tie_rule": tie_rule,
        "seed": seed,
        "per_kind": per_kind,
        "proposal_count": proposals,
        "agreements": int(report["agreements"]),
        "disagreements": int(report["disagreements"]),
        "formation": report["formation"],
        "requests": requests,
    }


def scenes_from_released(
    released,
    *,
    seed: int,
    per_kind: int = 4,
    mode: str = "task_aware",
    utility_on: bool = False,
    tie_rule: str = "skip",
) -> dict:
    """Write scenes from released rows only. Utility filtering stays off."""
    if utility_on:
        raise SpatialProtocolError("utility filter is off until a rule is written")
    if mode not in {"generic", "task_aware"}:
        raise SpatialProtocolError(f"unknown scene mode {mode}")
    if tie_rule not in TIE_RULES:
        raise SpatialProtocolError(f"unknown tie rule {tie_rule}")
    for row in released:
        if not isinstance(row, ReleasedPlace):
            raise SpatialProtocolError("scene builder received a row that is not a released place")
    named = _named(tuple(released))
    rng = random.Random(seed)
    proposals = []
    attempt_log = []
    attempts = {kind: 0 for kind in FIRST_ROUND_KINDS}
    if mode == "task_aware":
        plan = [kind for kind in FIRST_ROUND_KINDS for _ in range(per_kind)]
    else:
        plan = [FIRST_ROUND_KINDS[rng.randrange(len(FIRST_ROUND_KINDS))] for _ in range(per_kind * len(FIRST_ROUND_KINDS))]
    for kind in plan:
        attempts[kind] += 1
        scene, detail = _build(kind, named, rng, tie_rule)
        record = {
            "index": len(attempt_log),
            "kind": kind,
            "formed": scene is not None,
            "reason": detail["reason"],
        }
        for key, value in detail.items():
            if key != "reason":
                record[key] = value
        attempt_log.append(record)
        if scene is not None:
            proposals.append(scene)
    formation = {}
    for kind in FIRST_ROUND_KINDS:
        rows = [row for row in attempt_log if row["kind"] == kind]
        formed = sum(1 for row in rows if row["formed"])
        formation[kind] = {
            "attempts": len(rows),
            "formed": formed,
            "retention": (formed / len(rows)) if rows else None,
            "skip_reasons": dict(Counter(row["reason"] for row in rows if not row["formed"])),
        }
    return {
        "mode": mode,
        "tie_rule": tie_rule,
        "utility_on": False,
        "reads_entity_keys": False,
        "reads_road_graph": False,
        "attempts": attempts,
        "proposal_count": len(proposals),
        "agreements": sum(1 for scene in proposals if scene["agrees"]),
        "disagreements": sum(1 for scene in proposals if not scene["agrees"]),
        "kinds": dict(Counter(scene["kind"] for scene in proposals)),
        "formation": formation,
        "attempt_log": attempt_log,
        "nearest_population": nearest_population(released),
        "proposals": proposals,
    }
