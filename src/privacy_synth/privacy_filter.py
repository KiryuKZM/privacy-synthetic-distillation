"""Outbound privacy filter.

The reference set is an explicit allow-list. sealed_test is never a
reference. audit_nonmember is reserved for the attack and is not a
reference. Distance output is a screen, not a membership-inference result.
"""

from __future__ import annotations

from pathlib import Path

from privacy_synth.adult import QUASI_ID_COLUMNS, fingerprint

PRIVACY_REFERENCE_ROLES: tuple[str, ...] = (
    "generator_train",
    "local_dev",
    "public_calibration",
)
PRIVACY_FORBIDDEN_ROLES: tuple[str, ...] = (
    "sealed_test",
    "audit_nonmember",
    "excluded_cross_file",
    "excluded_label_conflict",
)


class PrivacyReferenceError(PermissionError):
    """Raised when a filter reference is outside the allow-list."""


def assert_reference_roles(roles: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    chosen = tuple(roles)
    refused = [role for role in chosen if role not in PRIVACY_REFERENCE_ROLES]
    if refused or not chosen:
        raise PrivacyReferenceError(
            "privacy filter reference must be a non-empty subset of "
            f"{PRIVACY_REFERENCE_ROLES}; refused {refused or chosen}"
        )
    return chosen


def exact_fingerprint(values: tuple[str, ...] | list[str]) -> str:
    return fingerprint(tuple(values))


def quasi_id_fingerprint(mapping: dict[str, str]) -> str:
    return fingerprint(tuple(mapping[column] for column in QUASI_ID_COLUMNS))


def screen_against_reference(
    candidate_fp: str,
    candidate_quasi_fp: str,
    *,
    reference_roles: tuple[str, ...] | list[str],
    exact_fps: dict[str, set[str]],
    quasi_fps: dict[str, set[str]],
) -> dict[str, object]:
    """Exact and quasi-id screen. Roles outside the allow-list raise."""
    roles = assert_reference_roles(reference_roles)
    exact_hit = []
    quasi_hit = []
    for role in roles:
        if candidate_fp in exact_fps.get(role, set()):
            exact_hit.append(role)
        if candidate_quasi_fp in quasi_fps.get(role, set()):
            quasi_hit.append(role)
    return {
        "kind": "screening",
        "reference_roles": list(roles),
        "exact_match_roles": exact_hit,
        "quasi_id_match_roles": quasi_hit,
        "block": bool(exact_hit or quasi_hit),
        "privacy_proof": False,
    }


def refuse_sealed_path(path: Path) -> None:
    """Stop if a caller points the filter at the sealed split on disk."""
    name = path.name
    if "sealed_test" in name or "sealed-test" in name:
        raise PrivacyReferenceError(f"privacy filter must not read {path}")
