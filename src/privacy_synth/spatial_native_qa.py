"""Questions about native indoor point sets, without inferred units or room labels.

This extends the scene-data boundary for the native 3D export. The existing
``spatial_scene`` module handles a different, public 2D place schema. No model,
data loading, filtering, or training is performed here.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import math
import re

FAMILIES = ("category_majority", "nearest_category", "centroid_category")
SYSTEM = (
    "Answer from the supplied object list only. Coordinates use the source's "
    "native units. Do not infer objects, room labels, physical bounds or a vertical "
    "axis. Return exactly <final>X</final>, with X one of A,B,C,D,E. No explanation."
)
LETTERS = "ABCDE"


def _rank(seed: int, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{purpose}|{value}".encode()).hexdigest()


def displayed_scene(row: dict, vocabulary: list[str]) -> list[dict]:
    """The displayed, three-decimal coordinates define every question's facts.

    Rounding is a declared prompt encoding, not a modification of stored samples.
    Exact integer coordinates below prevent floating point near-tie disagreement.
    """
    obs = row["objects"]
    count = row.get("count", row.get("condition", {}).get("count"))
    if row.get("raw_valid", True) is not True or not obs or count != len(obs):
        raise ValueError("Invalid scene remains a failure; no refill")
    if len(vocabulary) < 4 or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("At least four distinct category names required")
    result = []
    for i, obj in enumerate(obs):
        category = obj["category_id"]
        xyz = obj["position_native"]
        if type(category) is not int or not 0 <= category < len(vocabulary):
            raise ValueError("Invalid category")
        if len(xyz) != 3 or not all(type(x) in (int, float) and math.isfinite(x) for x in xyz):
            raise ValueError("Nonfinite native point")
        text = [format(x, ".3f") for x in xyz]
        result.append(dict(id=f"p{i:03d}", category=vocabulary[category],
                           xyz=text, integer_xyz=[int(Decimal(x) * 1000) for x in text]))
    return result


def _distance(a: list, b: list):
    return sum((x-y)**2 for x, y in zip(a, b))


def build_question(row: dict, vocabulary: list[str], family: str, seed: int) -> dict:
    if family not in FAMILIES:
        raise ValueError("Unknown question family")
    objects = displayed_scene(row, vocabulary)
    anchor_index = int(_rank(seed, "anchor", family), 16) % len(objects)
    anchor = objects[anchor_index]
    eligible = objects if family == "category_majority" else [o for o in objects if o["id"] != anchor["id"]]
    present = {o["category"] for o in eligible}
    chosen = sorted(present, key=lambda c: _rank(seed, family+"-choose", c))[:4]
    absent = sorted(set(vocabulary)-set(chosen), key=lambda c: _rank(seed, family+"-fill", c))
    chosen += absent[:4-len(chosen)]
    chosen.sort(key=lambda c: _rank(seed, family+"-options", c))
    assert len(chosen) == len(set(chosen)) == 4
    considered = [o for o in eligible if o["category"] in chosen]
    if family == "category_majority":
        counts = Counter(o["category"] for o in considered)
        winner = min(chosen, key=lambda c: (-counts[c], c))
        question = ("Among the four listed categories, which has the largest object count? "
                    "Use all objects. If counts tie, choose the alphabetically first category.")
    elif family == "nearest_category":
        winner = None if not considered else min(considered, key=lambda o: (
            _distance(o["integer_xyz"], anchor["integer_xyz"]), o["id"]))["category"]
        question = (f"Excluding {anchor['id']}, among objects in the four listed categories, "
                    f"which category has the object nearest to {anchor['id']} by Euclidean distance? "
                    "For equal distances choose the object with the smallest ID. "
                    "Choose NONE only if there are no eligible objects.")
    else:
        centers = {}
        for c in chosen:
            subset = [o for o in considered if o["category"] == c]
            if subset:
                centers[c] = [Fraction(sum(o["integer_xyz"][j] for o in subset), len(subset)) for j in range(3)]
        winner = None if not centers else min(centers, key=lambda c: (
            _distance(centers[c], anchor["integer_xyz"]), c))
        question = (f"Excluding {anchor['id']}, calculate the arithmetic mean center of each "
                    f"listed category's objects. Which category's center is nearest to {anchor['id']} "
                    "by Euclidean distance? Ignore categories with no eligible objects. "
                    "For equal distances choose the alphabetically first category. "
                    "Choose NONE only if there are no eligible objects.")
    answer = "E" if winner is None else LETTERS[chosen.index(winner)]
    table = "\n".join(" ".join([o["id"], o["category"], *o["xyz"]]) for o in objects)
    options = "\n".join(f"{letter}. {c}" for letter, c in zip(LETTERS, [*chosen, "NONE"]))
    prompt = f"Object list (ID category x y z):\n{table}\n\nQuestion: {question}\n{options}"
    return dict(system=SYSTEM, prompt=prompt, answer=answer, target=f"<final>{answer}</final>",
                family=family, object_count=len(objects), anchor=anchor["id"],
                options=chosen, answer_category=winner, singleton=len(objects)==1,
                encoding="native center coordinates fixed3 decimals; answers from displayed facts only")


def parse_answer(text: str) -> str | None:
    """Strict output contract; malformed output counts as an incorrect answer."""
    match = re.fullmatch(r"\s*<final>([ABCDE])</final>\s*", text)
    return match.group(1) if match else None


def independent_answer(prompt: str, family: str, anchor_id: str) -> str:
    """Second implementation from serialized facts, without the generator row.

    Uses decimal/rational category distances and separate tie handling, so a
    hidden precision difference between the original sample and prompt is caught.
    """
    body = prompt.split("Object list (ID category x y z):\n", 1)[1].split("\n\nQuestion:", 1)[0]
    entries = []
    for line in body.splitlines():
        identifier, category, *xyz = line.split()
        if len(xyz) != 3:
            raise ValueError("Malformed table")
        entries.append((identifier, category, tuple(Fraction(Decimal(v)) for v in xyz)))
    options = dict(re.findall(r"^([ABCDE])\. (\S+)$", prompt, re.M))
    if set(options) != set(LETTERS) or options["E"] != "NONE":
        raise ValueError("Malformed options")
    names = [options[x] for x in "ABCD"]
    if family == "category_majority":
        counts = {c:sum(e[1] == c for e in entries) for c in names}
        winner = sorted(names, key=lambda c: (-counts[c], c))[0]
    else:
        anchor = next(e[2] for e in entries if e[0] == anchor_id)
        candidates = [e for e in entries if e[0] != anchor_id and e[1] in names]
        if not candidates:
            return "E"
        if family == "nearest_category":
            ordered = sorted(candidates, key=lambda e: e[0])
            smallest = min(sum((e[2][j]-anchor[j])**2 for j in range(3)) for e in ordered)
            winner = next(e[1] for e in ordered if sum((e[2][j]-anchor[j])**2 for j in range(3)) == smallest)
        elif family == "centroid_category":
            ds = {}
            for c in names:
                group = [e[2] for e in candidates if e[1] == c]
                if group:
                    mean = tuple(sum(v[j] for v in group)/len(group) for j in range(3))
                    ds[c] = sum((mean[j]-anchor[j])**2 for j in range(3))
            best = min(ds.values())
            winner = sorted(c for c, value in ds.items() if value == best)[0]
        else:
            raise ValueError("Unknown question family")
    return next(letter for letter in "ABCD" if options[letter] == winner)
