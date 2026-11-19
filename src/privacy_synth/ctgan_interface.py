"""Local CTGAN interface over the pinned vendor checkout.

Does not edit vendor/ctgan. Does not train unless a caller explicitly
invokes fit on a non-sealed table. Generic and task-aware sampling are
two calls on the same fitted object.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CTGAN_VENDOR = REPO_ROOT / "vendor" / "ctgan"

DISCRETE_COLUMNS = (
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
)


def _import_ctgan():
    vendor = str(CTGAN_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from ctgan import CTGAN

    return CTGAN


def load_synthesizer(path: str):
    """Load a synthesizer saved by the official CTGAN.save. Does not fit."""
    ctgan_cls = _import_ctgan()
    return ctgan_cls.load(path)


def make_synthesizer(epochs: int = 300, batch_size: int = 500, seed: int = 20261001):
    """Construct the official CTGAN class. Does not fit."""
    ctgan_cls = _import_ctgan()
    return ctgan_cls(epochs=epochs, batch_size=batch_size, verbose=False, cuda=True)


def fit_generator(model, frame, *, source_role: str):
    if source_role != "generator_train":
        raise PermissionError("CTGAN fit accepts generator_train only")
    model.fit(frame, discrete_columns=list(DISCRETE_COLUMNS))
    return model


def sample_generic(model, n: int):
    """Official unconditional sample. Uses the training conditional distribution."""
    if n <= 0:
        raise ValueError("n must be positive")
    return model.sample(n)


def condition_bit_index(cond_starts, discrete_column_id: int, value_id: int) -> int:
    """Condition-vector column for one discrete category.

    The pinned CTGAN sampler leaves ``_discrete_column_matrix_st`` at zeros.
    The filled start array is ``_discrete_column_cond_st``.
    """
    starts = [int(item) for item in cond_starts]
    column_id = int(discrete_column_id)
    if column_id < 0 or column_id >= len(starts):
        raise ValueError(f"discrete column id {column_id} is outside the condition table")
    index = starts[column_id] + int(value_id)
    if index < starts[column_id]:
        raise ValueError(f"value id {value_id} is negative")
    return index


def corrected_condition_rows(sampler, condition_info: dict, batch: int) -> list[list[float]]:
    """One-hot rows using the discrete-column condition start. No vendor edit."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    n_categories = int(sampler._n_categories)
    index = condition_bit_index(
        sampler._discrete_column_cond_st,
        int(condition_info["discrete_column_id"]),
        int(condition_info["value_id"]),
    )
    if index >= n_categories:
        raise ValueError(f"condition bit {index} is outside 0..{n_categories - 1}")
    row = [0.0] * n_categories
    row[index] = 1.0
    return [row[:] for _ in range(batch)]


def _install_corrected_condition(model) -> None:
    """Point this loaded sampler at the corrected vector. Does not touch vendor files."""
    sampler = model._data_sampler

    def generate(condition_info, batch, _sampler=sampler):
        rows = corrected_condition_rows(_sampler, condition_info, int(batch))
        try:
            import numpy as np
        except ModuleNotFoundError:
            return rows
        return np.asarray(rows, dtype="float32")

    sampler.generate_cond_from_condition_column_info = generate


def sample_task_aware(model, n: int, condition_column: str, condition_value: str):
    """Same generator, with the condition bit on the requested discrete category."""
    if condition_column not in DISCRETE_COLUMNS:
        raise ValueError(f"{condition_column} is not a discrete CTGAN column")
    if n <= 0:
        raise ValueError("n must be positive")
    if hasattr(model, "_data_sampler") and hasattr(model, "_transformer"):
        _install_corrected_condition(model)
    generator = getattr(model, "_generator", None)
    if generator is None or not hasattr(generator, "eval"):
        return model.sample(
            n,
            condition_column=condition_column,
            condition_value=condition_value,
        )
    was_training = bool(generator.training)
    generator.eval()
    try:
        import torch

        with torch.no_grad():
            return model.sample(
                n,
                condition_column=condition_column,
                condition_value=condition_value,
            )
    finally:
        generator.train(was_training)
