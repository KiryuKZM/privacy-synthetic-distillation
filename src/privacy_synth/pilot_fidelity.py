"""Pilot distribution comparisons for one released table.

These functions report distances and intervals. They do not accept a generator
and they do not read a sealed question.
"""

from __future__ import annotations

import numpy as np

PAIRS = (
    ("lat_bin", "lon_bin"),
    ("lat_bin", "category"),
    ("lon_bin", "category"),
    ("category", "hours_state"),
)


def _ints(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def probability_tv(left_counts: np.ndarray, right_counts: np.ndarray) -> float:
    """Total variation between two histograms of the same shape."""
    left = np.asarray(left_counts, dtype=np.float64)
    right = np.asarray(right_counts, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("histograms differ in shape")
    if left.sum() <= 0 or right.sum() <= 0:
        raise ValueError("histogram is empty")
    return float(0.5 * np.abs((left / left.sum()) - (right / right.sum())).sum())


def joint_counts(xs, ys, x_bins: int, y_bins: int) -> np.ndarray:
    left = _ints(xs)
    right = _ints(ys)
    if len(left) != len(right):
        raise ValueError("joint columns differ in length")
    if len(left) == 0:
        raise ValueError("joint is empty")
    if left.min() < 0 or right.min() < 0 or left.max() >= x_bins or right.max() >= y_bins:
        raise ValueError("bin is outside the public domain")
    hist = np.zeros((x_bins, y_bins), dtype=np.float64)
    np.add.at(hist, (left, right), 1.0)
    return hist


def one_way_tv(real: dict, synthetic: dict, sizes: dict) -> dict[str, float]:
    scores = {}
    for name, bins in sizes.items():
        real_counts = np.bincount(_ints(real[name]), minlength=bins)
        synth_counts = np.bincount(_ints(synthetic[name]), minlength=bins)
        if real_counts.size != bins or synth_counts.size != bins:
            raise ValueError(f"{name} leaves the public domain")
        scores[name] = probability_tv(real_counts, synth_counts)
    return scores


def two_way_tv(real: dict, synthetic: dict, sizes: dict) -> dict[str, float]:
    scores = {}
    for left, right in PAIRS:
        real_hist = joint_counts(real[left], real[right], sizes[left], sizes[right])
        synth_hist = joint_counts(synthetic[left], synthetic[right], sizes[left], sizes[right])
        scores[f"{left}|{right}"] = probability_tv(real_hist, synth_hist)
    return scores


def dependence_tv(xs, ys, x_bins: int, y_bins: int) -> float:
    """Total variation between a joint and the product of its own margins."""
    joint = joint_counts(xs, ys, x_bins, y_bins)
    prob = joint / joint.sum()
    product = np.outer(prob.sum(axis=1), prob.sum(axis=0))
    return float(0.5 * np.abs(prob - product).sum())


def block_id(lat_bin: int, lon_bin: int, bins_per_block: int = 20) -> int:
    if bins_per_block <= 0:
        raise ValueError("block width must be positive")
    return (int(lat_bin) // bins_per_block) * 1000 + (int(lon_bin) // bins_per_block)


def block_ids(columns: dict, bins_per_block: int = 20) -> np.ndarray:
    return np.asarray(
        [block_id(lat, lon, bins_per_block) for lat, lon in zip(columns["lat_bin"], columns["lon_bin"])],
        dtype=np.int64,
    )


def _take(columns: dict, index: np.ndarray) -> dict:
    return {name: _ints(values)[index] for name, values in columns.items()}


def _reference_tv(columns: dict, groups: np.ndarray, sizes: dict, pair: tuple[str, str]) -> float:
    unique = np.array(sorted(set(int(item) for item in groups)))
    if len(unique) < 2:
        raise ValueError("real-real reference needs two groups")
    left_groups = set(unique[0::2].tolist())
    right_groups = set(unique[1::2].tolist())
    left_index = np.flatnonzero(np.isin(groups, list(left_groups)))
    right_index = np.flatnonzero(np.isin(groups, list(right_groups)))
    left = _take(columns, left_index)
    right = _take(columns, right_index)
    left_name, right_name = pair
    return probability_tv(
        joint_counts(left[left_name], left[right_name], sizes[left_name], sizes[right_name]),
        joint_counts(right[left_name], right[right_name], sizes[left_name], sizes[right_name]),
    )


def excess_interval(
    real: dict,
    synthetic: dict,
    sizes: dict,
    pair: tuple[str, str],
    *,
    bins_per_block: int = 20,
    draws: int = 200,
    seed: int = 20261011,
) -> dict:
    """Block-bootstrap the excess of candidate TV over a real-real split.

    The synthetic table stays fixed. The interval is over real geographic
    blocks, so it is not a multi-seed generator interval.
    """
    if draws < 20:
        raise ValueError("bootstrap needs at least 20 draws")
    groups = block_ids(real, bins_per_block)
    unique = np.array(sorted(set(int(item) for item in groups)))
    left_name, right_name = pair
    candidate = probability_tv(
        joint_counts(real[left_name], real[right_name], sizes[left_name], sizes[right_name]),
        joint_counts(synthetic[left_name], synthetic[right_name], sizes[left_name], sizes[right_name]),
    )
    reference = _reference_tv(real, groups, sizes, pair)
    rng = np.random.default_rng(seed)
    excesses = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(groups == item) for item in chosen])
        resampled = _take(real, index)
        resampled_groups = np.concatenate([np.full(np.count_nonzero(groups == item), int(item)) for item in chosen])
        resampled_candidate = probability_tv(
            joint_counts(resampled[left_name], resampled[right_name], sizes[left_name], sizes[right_name]),
            joint_counts(synthetic[left_name], synthetic[right_name], sizes[left_name], sizes[right_name]),
        )
        try:
            resampled_reference = _reference_tv(resampled, resampled_groups, sizes, pair)
        except ValueError:
            excesses[draw] = candidate - reference
            continue
        excesses[draw] = resampled_candidate - resampled_reference
    low, high = np.quantile(excesses, [0.025, 0.975])
    return {
        "pair": f"{left_name}|{right_name}",
        "candidate_tv": candidate,
        "real_real_tv": reference,
        "excess": candidate - reference,
        "interval_low": float(low),
        "interval_high": float(high),
        "draws": draws,
        "groups": int(len(unique)),
        "synthetic_table": "held_fixed",
        "interval": "real block bootstrap, 95 percent percentile",
    }


def permutation_null(columns: dict, sizes: dict, pair: tuple[str, str], *, draws: int, seed: int) -> dict:
    """Dependence TV after permuting the second column inside one table."""
    left_name, right_name = pair
    observed = dependence_tv(columns[left_name], columns[right_name], sizes[left_name], sizes[right_name])
    rng = np.random.default_rng(seed)
    right = _ints(columns[right_name])
    nulls = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        nulls[draw] = dependence_tv(columns[left_name], rng.permutation(right), sizes[left_name], sizes[right_name])
    return {
        "pair": f"{left_name}|{right_name}",
        "observed_dependence_tv": observed,
        "null_max": float(nulls.max()),
        "null_mean": float(nulls.mean()),
        "draws": draws,
    }


def _matrix(columns: dict, index: np.ndarray) -> np.ndarray:
    picked = _take(columns, index)
    return np.column_stack([picked[name] for name in ("lat_bin", "lon_bin", "category", "hours_state")])


def _balanced_accuracy(classifier, features: np.ndarray, labels: np.ndarray) -> float:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if min(len(positive), len(negative)) < 5:
        raise ValueError("C2ST test split has fewer than 5 rows in a class")
    # Average the two class recalls without randomly discarding test rows.
    predicted = classifier.predict(features)
    return float(0.5 * (
        np.mean(predicted[positive] == 1) + np.mean(predicted[negative] == 0)
    ))


def c2st_block_accuracy(
    real: dict,
    synthetic: dict,
    *,
    bins_per_block: int = 20,
    draws: int = 30,
    seed: int = 20261011,
    permute_latitude: bool = False,
    shift_latitude: int = 0,
) -> dict:
    """Balanced accuracy of a forest on held-out real geographic blocks.

    The point and bootstrap use the same shared block population. Synthetic
    mass outside all real blocks is reported separately, not hidden in an
    accuracy whose bootstrap excludes it. Neither score is an equivalence test.
    """
    from sklearn.ensemble import RandomForestClassifier

    synth = {name: _ints(values).copy() for name, values in synthetic.items()}
    real_groups = block_ids(real, bins_per_block)
    synth_groups = block_ids(synth, bins_per_block)
    if permute_latitude:
        # Corrupt the feature after the block split is fixed. The split stays
        # the known-difference control, not a new assignment of rows.
        synth["lat_bin"] = np.random.default_rng(seed + 1).permutation(synth["lat_bin"])
    if shift_latitude:
        synth["lat_bin"] = synth["lat_bin"] + int(shift_latitude)
    unique = np.array(sorted(set(int(item) for item in real_groups)))
    if len(unique) < 4:
        raise ValueError("C2ST needs at least four real groups")
    train_groups = set(unique[0::2].tolist())
    test_groups = set(unique[1::2].tolist())
    real_index = np.arange(len(real_groups))
    synth_index = np.arange(len(synth_groups))
    train_real = real_index[np.isin(real_groups, list(train_groups))]
    test_real = real_index[np.isin(real_groups, list(test_groups))]
    train_synth = synth_index[np.isin(synth_groups, list(train_groups))]
    test_synth = synth_index[np.isin(synth_groups, list(test_groups))]
    outside_synth = synth_index[~np.isin(synth_groups, unique)]
    if min(len(train_real), len(train_synth)) < 5:
        raise ValueError("C2ST training split has fewer than 5 rows in a class")
    rng = np.random.default_rng(seed)
    train_features = np.vstack([_matrix(real, train_real), _matrix(synth, train_synth)])
    train_labels = np.concatenate([
        np.zeros(len(train_real), dtype=np.int64),
        np.ones(len(train_synth), dtype=np.int64),
    ])
    test_features = np.vstack([_matrix(real, test_real), _matrix(synth, test_synth)])
    test_labels = np.concatenate([
        np.zeros(len(test_real), dtype=np.int64),
        np.ones(len(test_synth), dtype=np.int64),
    ])
    classifier = RandomForestClassifier(
        n_estimators=40,
        max_depth=6,
        random_state=seed,
        n_jobs=1,
    )
    classifier.fit(train_features, train_labels)
    point = _balanced_accuracy(classifier, test_features, test_labels)
    test_group_ids = np.array(sorted(test_groups))
    scores = []
    for draw in range(draws):
        chosen = rng.choice(test_group_ids, size=len(test_group_ids), replace=True)
        real_pick = np.concatenate([test_real[real_groups[test_real] == item] for item in chosen])
        synth_pick = np.concatenate([
            test_synth[synth_groups[test_synth] == item] for item in chosen if np.any(synth_groups[test_synth] == item)
        ]) if any(np.any(synth_groups[test_synth] == item) for item in chosen) else np.array([], dtype=np.int64)
        if len(synth_pick) < 5 or len(real_pick) < 5:
            continue
        folded_features = np.vstack([_matrix(real, real_pick), _matrix(synth, synth_pick)])
        folded_labels = np.concatenate([
            np.zeros(len(real_pick), dtype=np.int64),
            np.ones(len(synth_pick), dtype=np.int64),
        ])
        scores.append(_balanced_accuracy(classifier, folded_features, folded_labels))
    low, high = np.quantile(scores, [0.025, 0.975]) if scores else (None, None)
    return {
        "accuracy": point,
        "interval_low": None if low is None else float(low),
        "interval_high": None if high is None else float(high),
        "draws": len(scores),
        "requested_draws": draws,
        "invalid_draws": draws - len(scores),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "test_real_rows": len(test_real),
        "test_synthetic_rows": len(test_synth),
        "synthetic_outside_real_blocks": len(outside_synth),
        "synthetic_outside_real_block_fraction": float(len(outside_synth) / len(synth_groups)),
        "permute_latitude": permute_latitude,
        "shift_latitude": int(shift_latitude),
        "classifier": "RandomForestClassifier depth 6, 40 trees",
        "accuracy_definition": "mean class recall on shared held-out real blocks; outside mass reported separately",
        "interval_scope": "fixed classifier and tables; held-out shared block bootstrap",
        "not_generator_acceptance": True,
    }


def within_block_excess_interval(
    real: dict,
    synthetic: dict,
    sizes: dict,
    pair: tuple[str, str],
    *,
    bins_per_block: int = 20,
    draws: int = 200,
    seed: int = 20261011,
) -> dict:
    """Paired block sensitivity with an immutable within-block reference split.

    Split original source rows once, before resampling. A block's multiplicity
    then weights both original halves and the corresponding synthetic rows.
    Synthetic rows outside the observed real blocks stay as a fixed stratum
    and are reported separately. No generator is refit. These percentile
    ranges are diagnostics conditional on one table, not acceptance intervals.
    """
    if draws < 20:
        raise ValueError("block sensitivity needs at least 20 draws")
    groups = block_ids(real, bins_per_block)
    unique = np.array(sorted(set(int(item) for item in groups)))
    synth_groups = block_ids(synthetic, bins_per_block)
    splits = _within_block_splits(groups, seed=seed)
    if len(splits) < 2:
        raise ValueError("within-block reference needs two blocks")
    left_name, right_name = pair
    shape = (sizes[left_name], sizes[right_name])

    def hist(columns: dict, index: np.ndarray) -> np.ndarray:
        if not len(index):
            return np.zeros(shape, dtype=np.float64)
        return joint_counts(
            _ints(columns[left_name])[index], _ints(columns[right_name])[index], *shape,
        )

    real_histograms = []
    synth_histograms = []
    left_histograms = []
    right_histograms = []
    for group in unique:
        real_histograms.append(hist(real, np.flatnonzero(groups == group)))
        synth_histograms.append(hist(synthetic, np.flatnonzero(synth_groups == group)))
        left, right = splits.get(int(group), (np.array([], dtype=np.int64),) * 2)
        left_histograms.append(hist(real, left))
        right_histograms.append(hist(real, right))
    arrays = [np.stack(parts) for parts in (
        real_histograms, synth_histograms, left_histograms, right_histograms,
    )]
    outside_index = np.flatnonzero(~np.isin(synth_groups, unique))
    outside_histogram = hist(synthetic, outside_index)

    def components(weights: np.ndarray) -> tuple[float, float]:
        real_counts, synth_counts, left_counts, right_counts = [
            np.tensordot(weights, array, axes=1) for array in arrays
        ]
        return (
            probability_tv(real_counts, synth_counts + outside_histogram),
            probability_tv(left_counts, right_counts),
        )

    candidate, reference = components(np.ones(len(unique)))
    key = f"{pair[0]}|{pair[1]}"
    rng = np.random.default_rng(seed + 1)
    excesses = []
    for _draw in range(draws):
        chosen = rng.choice(len(unique), size=len(unique), replace=True)
        weights = np.bincount(chosen, minlength=len(unique))
        try:
            cand, ref = components(weights)
        except ValueError:
            continue
        excesses.append(cand - ref)
    if not excesses:
        raise ValueError("no valid block sensitivity draws")
    low, high = np.quantile(excesses, [0.025, 0.975])
    return {
        "pair": key,
        "candidate_tv": candidate,
        "within_block_real_real_tv": reference,
        "excess": candidate - reference,
        "interval_low": float(low),
        "interval_high": float(high),
        "draws": len(excesses),
        "requested_draws": draws,
        "invalid_draws": draws - len(excesses),
        "real_groups": len(unique),
        "reference_groups": len(splits),
        "reference_split_seed": seed,
        "bootstrap_seed": seed + 1,
        "synthetic_outside_real_blocks": len(outside_index),
        "synthetic_outside_real_block_fraction": float(len(outside_index) / len(synth_groups)),
        "source_halves": "split once before bootstrap; copied blocks never repartitioned",
        "resampling": "same block multiplicities for real, synthetic, and both reference halves",
        "outside_synthetic_stratum": "held fixed; its sampling uncertainty is not estimated",
        "interval": "2.5/97.5 percentiles of paired block sensitivity draws; conditional on one table and one split",
        "not_generator_acceptance": True,
    }


def _within_block_splits(groups: np.ndarray, *, seed: int) -> dict:
    """Source row indices assigned once; cell duplicates remain distinct rows."""
    rng = np.random.default_rng(seed)
    splits = {}
    for group in sorted(set(int(item) for item in groups)):
        index = np.flatnonzero(groups == group)
        if len(index) < 2:
            continue
        order = rng.permutation(index)
        cut = len(order) // 2
        splits[group] = (order[:cut], order[cut:])
    return splits


def within_block_tv(
    columns: dict,
    sizes: dict,
    pair: tuple[str, str],
    *,
    bins_per_block: int = 20,
    seed: int = 20261011,
) -> dict:
    """Total variation between two halves of each geographic block.

    This is a same-neighborhood reference. It is not the even/odd block
    split, and it is not a generator acceptance gate.
    """
    groups = block_ids(columns, bins_per_block)
    splits = _within_block_splits(groups, seed=seed)
    if len(splits) < 2:
        raise ValueError("within-block reference needs two blocks")
    left = _take(columns, np.concatenate([parts[0] for parts in splits.values()]))
    right = _take(columns, np.concatenate([parts[1] for parts in splits.values()]))
    left_name, right_name = pair
    distance = probability_tv(
        joint_counts(left[left_name], left[right_name], sizes[left_name], sizes[right_name]),
        joint_counts(right[left_name], right[right_name], sizes[left_name], sizes[right_name]),
    )
    return {"pair": f"{left_name}|{right_name}", "real_real_tv": distance, "blocks_used": len(splits), "seed": seed}


def dependence_gap(dependence: dict) -> float:
    """How far an observed dependence sits above its column-permutation maximum."""
    return float(dependence["observed_dependence_tv"]) - float(dependence["null_max"])


def spatial_resolution_diagnostics(
    real: dict,
    synthetic: dict,
    sizes: dict,
    *,
    factors: tuple[int, ...] = (1, 2, 4, 10),
    bins_per_block: int = 20,
    seed: int = 20261011,
) -> list[dict]:
    """Measure saved tables at coarser resolutions without changing any rows.

    This distinguishes fine-cell sparsity from errors visible even at a coarse
    scale. The release representation and acceptance tolerance are unchanged.
    Empty observed cells are an empirical diagnostic, not a forbidden domain
    or a privacy signal.
    """
    records = []
    for factor in factors:
        if factor <= 0 or bins_per_block % factor:
            raise ValueError("resolution factor must divide the public block width")
        tables = []
        for source in (real, synthetic):
            table = {name: _ints(values).copy() for name, values in source.items()}
            table["lat_bin"] //= factor
            table["lon_bin"] //= factor
            tables.append(table)
        real_table, synth_table = tables
        lat_size = (sizes["lat_bin"] + factor - 1) // factor
        lon_size = (sizes["lon_bin"] + factor - 1) // factor
        real_hist = joint_counts(real_table["lat_bin"], real_table["lon_bin"], lat_size, lon_size)
        synth_hist = joint_counts(synth_table["lat_bin"], synth_table["lon_bin"], lat_size, lon_size)
        reference = within_block_tv(
            real_table, {**sizes, "lat_bin": lat_size, "lon_bin": lon_size},
            ("lat_bin", "lon_bin"), bins_per_block=bins_per_block // factor, seed=seed,
        )
        candidate = probability_tv(real_hist, synth_hist)
        records.append({
            "measurement_coarsening_factor": factor,
            "domain_cells": lat_size * lon_size,
            "real_occupied_cells": int(np.count_nonzero(real_hist)),
            "synthetic_occupied_cells": int(np.count_nonzero(synth_hist)),
            "synthetic_mass_in_cells_unobserved_in_real": float(synth_hist[real_hist == 0].sum() / synth_hist.sum()),
            "candidate_tv": candidate,
            "within_block_real_real_tv": reference["real_real_tv"],
            "descriptive_excess": candidate - reference["real_real_tv"],
            "not_a_new_generator_configuration": True,
            "not_generator_acceptance": True,
        })
    return records


def reference_is_saturated(real_real_tv: float) -> bool:
    """A total-variation reference of 1 cannot detect a missing joint.

    Every candidate total variation is at most 1, so the excess over a
    saturated real-real split is never positive. That split is two
    geographic halves, not two draws from one neighborhood.
    """
    return float(real_real_tv) >= 1.0 - 1e-9


def research_pair_call(
    *,
    real_real_tv: float,
    interval_low: float,
    interval_high: float,
    real_dependence: dict,
    synthetic_dependence: dict,
    tolerance: float,
) -> dict:
    """Apply the frozen excess rule, and refuse a pass from a saturated reference.

    The collapse branch is the one already written for an interval that covers
    the tolerance: real dependence exceeds its null by more than the tolerance
    and the synthetic dependence does not. A saturated reference skips the
    excess pass and uses that branch. It does not change the tolerance.
    """
    real_gap = dependence_gap(real_dependence)
    synthetic_gap = dependence_gap(synthetic_dependence)
    saturated = reference_is_saturated(real_real_tv)
    collapsed = real_gap > tolerance and synthetic_gap <= 0.0
    if saturated:
        call = "FAIL" if collapsed else "INCONCLUSIVE"
        reason = "saturated_real_real_reference"
    elif interval_high <= tolerance:
        call = "PASS"
        reason = "excess_upper_bound"
    elif interval_low > tolerance:
        call = "FAIL"
        reason = "excess_lower_bound"
    elif collapsed:
        call = "FAIL"
        reason = "joint_collapsed"
    else:
        call = "INCONCLUSIVE"
        reason = "interval_covers_tolerance"
    return {
        "call": call,
        "reason": reason,
        "saturated_reference": saturated,
        "real_gap": real_gap,
        "synthetic_gap": synthetic_gap,
        "collapsed": collapsed,
    }
