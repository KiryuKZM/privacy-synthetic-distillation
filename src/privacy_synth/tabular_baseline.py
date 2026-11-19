"""Auxiliary tabular diagnostic baseline.

This is not the first-round student. The first-round student is
Qwen/Qwen3.5-2B. Do not add another language model here, and do not
read sealed_test to choose this baseline.

The fitted ceiling is `scripts/score_xgboost_ceiling.py`. It stays
outside this module so the module does not import a model library.
"""

from __future__ import annotations

from privacy_synth.student import StudentDataError


def refuse_as_primary_student() -> None:
    raise StudentDataError(
        "the tree baseline is an auxiliary diagnostic, not the first-round student"
    )
