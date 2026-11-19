"""Utility filter for synthetic Adult rows.

Rules were fixed from the Adult education map and from generator_train
support only. This module does not read sealed_test and does not call a teacher.
"""

from __future__ import annotations

import math
from typing import Mapping

from privacy_synth.adult import COLUMNS, NUMERIC_COLUMNS

# Official Adult education map. This is the dataset coding, not a fitted model.
EDUCATION_TO_NUM: dict[str, str] = {
    "Preschool": "1",
    "1st-4th": "2",
    "5th-6th": "3",
    "7th-8th": "4",
    "9th": "5",
    "10th": "6",
    "11th": "7",
    "12th": "8",
    "HS-grad": "9",
    "Some-college": "10",
    "Assoc-voc": "11",
    "Assoc-acdm": "12",
    "Bachelors": "13",
    "Masters": "14",
    "Prof-school": "15",
    "Doctorate": "16",
}

# Minimum age observed for that education in generator_train on 2026-10-02.
# Measured before any synthetic sample. sealed_test was not used.
AGE_FLOOR_BY_EDUCATION: dict[str, int] = {
    "Preschool": 20,
    "1st-4th": 19,
    "5th-6th": 17,
    "7th-8th": 17,
    "9th": 17,
    "10th": 17,
    "11th": 17,
    "12th": 17,
    "HS-grad": 17,
    "Some-college": 17,
    "Assoc-voc": 19,
    "Assoc-acdm": 19,
    "Bachelors": 19,
    "Masters": 18,
    "Prof-school": 25,
    "Doctorate": 24,
}


def coerce_adult_row(mapping: Mapping[str, object]) -> dict[str, str]:
    """Shared numeric rounding. Both utility arms see these strings."""
    missing = [column for column in COLUMNS if column not in mapping]
    if missing:
        raise ValueError(f"row missing fields: {missing}")
    coerced: dict[str, str] = {}
    for column in COLUMNS:
        text = str(mapping[column]).strip()
        if column in NUMERIC_COLUMNS:
            number = float(text)
            if not math.isfinite(number):
                raise ValueError(f"{column} is not finite")
            coerced[column] = str(int(round(number)))
        else:
            coerced[column] = text
    return coerced


def utility_violations(row: Mapping[str, str]) -> list[str]:
    """Return rule names that a coerced row breaks. Empty means the row passes."""
    violations: list[str] = []
    education = row["education"]
    expected = EDUCATION_TO_NUM.get(education)
    if expected is None or row["education-num"] != expected:
        violations.append("education_num")
    age = int(row["age"])
    floor = AGE_FLOOR_BY_EDUCATION.get(education)
    if floor is None or age < floor:
        violations.append("age_education_floor")
    hours = int(row["hours-per-week"])
    if hours < 1 or hours > 99:
        violations.append("hours_range")
    gain = int(row["capital-gain"])
    loss = int(row["capital-loss"])
    if gain < 0 or loss < 0:
        violations.append("capital_nonnegative")
    if gain > 0 and loss > 0:
        violations.append("capital_exclusive")
    workclass = row["workclass"]
    occupation = row["occupation"]
    if occupation == "Never-worked":
        violations.append("occupation_never_worked")
    if workclass == "Never-worked" and (
        occupation != "?" or gain != 0 or loss != 0
    ):
        violations.append("never_worked_logic")
    if workclass == "Without-pay" and (gain != 0 or loss != 0):
        violations.append("without_pay_logic")
    return violations


def passes_utility(row: Mapping[str, str]) -> bool:
    return not utility_violations(row)
