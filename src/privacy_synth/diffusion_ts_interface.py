"""Sibling loader for the pinned Diffusion-TS checkout.

Does not edit vendor/diffusion-ts and does not import it until a fit or
sample call. The CTGAN module remains the only other vendor path insertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

from privacy_synth.pems import GENERATOR_COMMIT, PemsProtocolError

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_VENDOR = REPO_ROOT / "vendor" / "diffusion-ts"
TRAIN_ROLE = "pems_generator_train"


def assert_source_role(source_role: str) -> None:
    if source_role != TRAIN_ROLE:
        raise PemsProtocolError(
            f"Diffusion-TS fit accepts {TRAIN_ROLE} only, got {source_role!r}"
        )


def assert_checkout_commit() -> str:
    head = (DIFFUSION_VENDOR / ".git").exists() or (REPO_ROOT / ".git").exists()
    if not head:
        raise PemsProtocolError("Diffusion-TS checkout is missing")
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(DIFFUSION_VENDOR), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = proc.stdout.strip()
    if proc.returncode != 0 or commit != GENERATOR_COMMIT:
        raise PemsProtocolError(
            f"Diffusion-TS checkout {commit!r} is not pin {GENERATOR_COMMIT}"
        )
    return commit


def import_diffusion_ts():
    """Insert vendor/diffusion-ts on sys.path and return the model class."""
    assert_checkout_commit()
    vendor = str(DIFFUSION_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

    return Diffusion_TS
