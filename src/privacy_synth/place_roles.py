"""Assign public place rows to disjoint pilot roles by geographic block.

The window is the historical public center expanded by a fixed 4 km half-side.
Block edges use that same public meter scale. The assignment hash is public.
It is not a generator score, and it does not read a sealed question.
"""

from __future__ import annotations

import hashlib
import math

METERS_PER_DEGREE = 111_320.0
ROUND_ID = "2026-10-11-overnight-022356"
CENTER_LAT = 1.282
CENTER_LON = 103.842
HALF_M = 4000
BLOCK_M = 1000
ROLES = (
    "public_design",
    "public_calibration",
    "generator_train",
    "local_dev",
    "audit_nonmember",
    "sealed_test",
)
_RANGES = (
    (0, 20, "public_design"),
    (20, 35, "public_calibration"),
    (35, 60, "generator_train"),
    (60, 75, "local_dev"),
    (75, 85, "audit_nonmember"),
    (85, 100, "sealed_test"),
)


def pilot_bbox() -> tuple[float, float, float, float]:
    """South, west, north, east. The center is the historical public window."""
    dlat = HALF_M / METERS_PER_DEGREE
    dlon = HALF_M / (METERS_PER_DEGREE * math.cos(math.radians(CENTER_LAT)))
    return (CENTER_LAT - dlat, CENTER_LON - dlon, CENTER_LAT + dlat, CENTER_LON + dlon)


def block_of(lat: float, lon: float, bbox: tuple[float, float, float, float] | None = None) -> tuple[int, int]:
    south, west, north, east = bbox or pilot_bbox()
    mid = (south + north) / 2.0
    y_m = (lat - south) * METERS_PER_DEGREE
    x_m = (lon - west) * METERS_PER_DEGREE * math.cos(math.radians(mid))
    if y_m < 0 or x_m < 0:
        raise ValueError("place is outside the pilot window")
    return (int(y_m // BLOCK_M), int(x_m // BLOCK_M))


def role_for_block(block: tuple[int, int], round_id: str = ROUND_ID) -> str:
    digest = hashlib.sha256(f"{round_id}|{block[0]}|{block[1]}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    for start, stop, role in _RANGES:
        if start <= bucket < stop:
            return role
    raise ValueError(f"bucket {bucket} has no role")
