"""Fixed-denominator query diagnostics; incomplete output cannot score perfect."""
from __future__ import annotations

import math
import numpy as np


def semantic_nonempty(value, query):
    return value is not None and (query.get("empty_if") != "count_zero" or float(value) != 0)


def score_queries(reference_rows, other_rows, queries, *, min_reference_nonempty=5):
    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate planned query IDs")
    ref = {r["id"]: r for r in reference_rows}
    other = {r["id"]: r for r in other_rows}
    if len(ref) != len(reference_rows) or len(other) != len(other_rows):
        raise ValueError("Duplicate result IDs")
    families = {}
    for family in sorted({q["family"] for q in queries}):
        selected = [q for q in queries if q["family"] == family]
        report = dict(expected=len(selected), reference_ok=0, other_ok=0, reference_nonempty=0, other_nonempty=0,
                      numeric_pairs=0, null_matches=0, null_mismatches=0, missing_results=0, execution_failures=0)
        values = [float(ref[q["id"]]["value"]) for q in selected if q["id"] in ref and ref[q["id"]].get("ok")
                  and ref[q["id"]].get("value") is not None]
        scale = float(np.std(values)) if len(values) > 1 else 1.0
        if scale < 1e-8:
            scale = 1.0
        errors, complete_errors = [], []
        for q in selected:
            a, b = ref.get(q["id"]), other.get(q["id"])
            for row, prefix in [(a, "reference"), (b, "other")]:
                if row and row.get("ok"):
                    report[prefix + "_ok"] += 1
                    report[prefix + "_nonempty"] += int(semantic_nonempty(row.get("value"), q))
            if a is None or b is None:
                report["missing_results"] += 1
                continue
            if not a.get("ok") or not b.get("ok"):
                report["execution_failures"] += 1
                continue
            left, right = a.get("value"), b.get("value")
            if left is None and right is None:
                report["null_matches"] += 1
                complete_errors.append(0.0)
            elif left is None or right is None:
                report["null_mismatches"] += 1
            else:
                if not math.isfinite(float(left)) or not math.isfinite(float(right)):
                    raise ValueError("Non-finite query result")
                report["numeric_pairs"] += 1
                error = abs(float(left) - float(right)) / scale
                errors.append(error)
                complete_errors.append(error)
        report["supported"] = report["reference_nonempty"] >= min_reference_nonempty
        report["complete"] = len(complete_errors) == len(selected)
        report["paired_numeric_distance"] = float(np.mean(errors)) if errors else None
        report["distance"] = float(np.mean(complete_errors)) if report["complete"] and report["supported"] else None
        families[family] = report
    complete = all(r["distance"] is not None for r in families.values()) and bool(families)
    return {"scorer_version": 2, "query_denominator": len(ids), "family_denominator": len(families),
            "supported_families": sum(r["supported"] for r in families.values()),
            "complete_families": sum(r["complete"] for r in families.values()),
            "families": families, "macro": float(np.mean([r["distance"] for r in families.values()])) if complete else None,
            "status": "COMPLETE_COMPARISON" if complete else "INCOMPLETE_OR_UNSUPPORTED",
            "null_mismatches": sum(r["null_mismatches"] for r in families.values()),
            "execution_failures": sum(r["execution_failures"] for r in families.values()),
            "missing_results": sum(r["missing_results"] for r in families.values())}
