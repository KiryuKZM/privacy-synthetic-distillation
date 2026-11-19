"""Load one public place role for a local generator trial.

The sealed role has no loader. This module does not fit a generator and does
not call a teacher.
"""

from __future__ import annotations

import json
from pathlib import Path

from privacy_synth.place_roles import ROLES, pilot_bbox
from privacy_synth.spatial_entity import PlaceRow, SpatialProtocolError, place_from_element

SEALED_ROLE = "sealed_test"
SEALED_FILENAME = "sealed_test_ids.txt"


def load_place_rows(
    role: str,
    *,
    places_path: Path,
    ids_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[PlaceRow, ...]:
    """Return place rows for one non-sealed role, sorted by entity key."""
    if role == SEALED_ROLE or Path(ids_path).name == SEALED_FILENAME:
        raise SpatialProtocolError("sealed_test is not readable from the generator path")
    if role not in ROLES:
        raise SpatialProtocolError(f"unknown role {role}")
    wanted = {
        line.strip()
        for line in Path(ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not wanted:
        raise SpatialProtocolError("role id list is empty")
    rows: list[PlaceRow] = []
    seen: set[str] = set()
    with Path(places_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            element = json.loads(line)
            key = f"node:{int(element['id'])}"
            if key not in wanted:
                continue
            row = place_from_element(element, bbox or pilot_bbox())
            if row is None or row.entity_key != key:
                raise SpatialProtocolError("role id did not become one place row")
            rows.append(row)
            seen.add(key)
    missing = wanted - seen
    if missing:
        raise SpatialProtocolError(f"role list has {len(missing)} ids that were not loaded")
    rows.sort(key=lambda row: row.entity_key)
    return tuple(rows)
