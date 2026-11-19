"""Road-graph speed forecasts for a sensor network.

Pure functions only. Training stays in ``scripts/fit_metr_graph.py``.
"""

from __future__ import annotations

import math

GRAPH_DISTANCE_THRESHOLD = 0.1
GRAPH_WEEK_STEPS = 2016
GRAPH_SEASONAL_WEEKS = 4
GRAPH_HISTORY = 12
GRAPH_HORIZON = 12
HORIZON_INDEX = {"15": 2, "30": 5, "60": 11}


class GraphForecastError(ValueError):
    """A road-graph forecast input broke a frozen rule."""


def road_adjacency(
    sensor_ids: list[str],
    edges: list[tuple[str, str, float]],
    threshold: float = GRAPH_DISTANCE_THRESHOLD,
) -> list[list[float]]:
    """Row-normalized Gaussian weights from road distances.

    ``sigma`` is the population standard deviation of the kept costs, including
    zeros. Weights below ``threshold`` are dropped. A one-sensor graph and a
    graph with no off-diagonal weight are refused.
    """
    if len(sensor_ids) < 2 or len(set(sensor_ids)) != len(sensor_ids):
        raise GraphForecastError("a road graph needs more than one sensor")
    index = {sensor_id: position for position, sensor_id in enumerate(sensor_ids)}
    kept: list[tuple[int, int, float]] = []
    for source, dest, cost in edges:
        if source not in index or dest not in index:
            continue
        kept.append((index[source], index[dest], float(cost)))
    if len(kept) < 2:
        raise GraphForecastError("not enough road distances")
    mean = sum(cost for _src, _dst, cost in kept) / len(kept)
    variance = sum((cost - mean) ** 2 for _src, _dst, cost in kept) / len(kept)
    std = math.sqrt(variance)
    if std == 0.0:
        raise GraphForecastError("road distances have no spread")
    size = len(sensor_ids)
    adjacency = [[0.0] * size for _ in range(size)]
    for src, dst, cost in kept:
        weight = math.exp(-((cost / std) ** 2))
        if weight < threshold:
            continue
        adjacency[src][dst] = weight
    off_diagonal = any(
        adjacency[row][col] > 0.0
        for row in range(size)
        for col in range(size)
        if row != col
    )
    if not off_diagonal:
        raise GraphForecastError("adjacency has no off-diagonal weight")
    for row in range(size):
        total = sum(adjacency[row])
        if total > 0.0:
            adjacency[row] = [value / total for value in adjacency[row]]
    return adjacency


def network_window(
    speeds: list[list[float]],
    start: int,
    length: int = GRAPH_HISTORY,
) -> list[list[float]]:
    """History for every sensor. A single-sensor window is refused."""
    if length <= 0:
        raise GraphForecastError("window length must be positive")
    if not speeds or start < 0 or start + length > len(speeds):
        raise GraphForecastError("speed window leaves the series")
    window: list[list[float]] = []
    width = None
    for time in range(start, start + length):
        row = [float(value) for value in speeds[time]]
        if width is None:
            width = len(row)
        if len(row) != width:
            raise GraphForecastError("speed rows differ in width")
        window.append(row)
    if width is None or width < 2:
        raise GraphForecastError("forecast input has one sensor")
    return window


def prepare_forecast_inputs(
    sensor_ids: list[str],
    edges: list[tuple[str, str, float]],
    speeds: list[list[float]],
    start: int,
    length: int = GRAPH_HISTORY,
) -> dict[str, object]:
    """Adjacency from the road distances and a multi-sensor history."""
    adjacency = road_adjacency(sensor_ids, edges)
    history = network_window(speeds, start, length)
    return {"adjacency": adjacency, "history": history}


def neighbor_mix(
    history: list[list[float]],
    adjacency: list[list[float]],
) -> list[list[float]]:
    """One row-normalized hop. Column ``i`` uses every sensor with a nonzero weight."""
    if len(history[0]) < 2:
        raise GraphForecastError("forecast input has one sensor")
    if len(adjacency) != len(history[0]):
        raise GraphForecastError("adjacency does not match the sensors")
    mixed: list[list[float]] = []
    for row in history:
        mixed.append(
            [
                sum(weight * value for weight, value in zip(neighbors, row, strict=True))
                for neighbors in adjacency
            ]
        )
    return mixed


def seasonal_value(
    series: list[float],
    target: int,
    train_end: int,
    fallback: float,
    period: int = GRAPH_WEEK_STEPS,
    weeks: int = GRAPH_SEASONAL_WEEKS,
) -> float:
    """Mean of up to ``weeks`` earlier periods that land inside the train span.

    A lookup whose speed is 0 is skipped. If none remain, ``fallback`` is used.
    """
    if period <= 0 or weeks <= 0:
        raise GraphForecastError("seasonal period must be positive")
    total = 0.0
    count = 0
    for week in range(1, weeks + 1):
        index = target - week * period
        if index < 0 or index >= train_end or index >= len(series):
            continue
        value = float(series[index])
        if value == 0.0:
            continue
        total += value
        count += 1
    if count == 0:
        return float(fallback)
    return total / count


def new_horizon_totals() -> dict[str, dict[str, float]]:
    return {
        name: {"abs": 0.0, "sq": 0.0, "pct": 0.0, "n": 0.0, "zeros": 0.0}
        for name in HORIZON_INDEX
    }


def accumulate_horizon(
    predictions: list[list[float]],
    truths: list[list[float]],
    totals: dict[str, dict[str, float]],
) -> None:
    """Same skip rule as ``zero_masked_horizon_metrics``: a true speed of 0 is left out."""
    if len(predictions) != len(truths):
        raise GraphForecastError("predictions and truths differ in length")
    for predicted, truth in zip(predictions, truths, strict=True):
        if len(predicted) != GRAPH_HORIZON or len(truth) != GRAPH_HORIZON:
            raise GraphForecastError("a forecast metric row has 12 speeds")
        for name, index in HORIZON_INDEX.items():
            actual = float(truth[index])
            guessed = float(predicted[index])
            if actual == 0.0:
                totals[name]["zeros"] += 1.0
                continue
            gap = guessed - actual
            totals[name]["abs"] += abs(gap)
            totals[name]["sq"] += gap * gap
            totals[name]["pct"] += abs(gap) / abs(actual)
            totals[name]["n"] += 1.0


def metrics_from_totals(totals: dict[str, dict[str, float]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, bucket in totals.items():
        count = bucket["n"]
        if count <= 0.0:
            raise GraphForecastError(f"horizon {name} has no nonzero truths")
        scores[f"mae_{name}"] = bucket["abs"] / count
        scores[f"rmse_{name}"] = math.sqrt(bucket["sq"] / count)
        scores[f"mape_{name}"] = bucket["pct"] / count
    return scores


def score_speed_cubes(predicted: object, actual: object) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score arrays shaped ``[windows, 12, sensors]`` in window-then-sensor order."""
    windows = len(predicted)  # type: ignore[arg-type]
    if windows == 0:
        raise GraphForecastError("no forecast windows")
    horizon = len(predicted[0])  # type: ignore[index]
    sensors = len(predicted[0][0])  # type: ignore[index]
    if horizon != GRAPH_HORIZON or sensors < 2:
        raise GraphForecastError("a scored cube needs 12 steps and more than one sensor")
    totals = new_horizon_totals()
    for window in range(windows):
        batch_pred = [
            [float(predicted[window][step][sensor]) for step in range(horizon)]  # type: ignore[index]
            for sensor in range(sensors)
        ]
        batch_true = [
            [float(actual[window][step][sensor]) for step in range(horizon)]  # type: ignore[index]
            for sensor in range(sensors)
        ]
        accumulate_horizon(batch_pred, batch_true, totals)
    return metrics_from_totals(totals), totals


GRAPH_PROMPT_NEIGHBORS = 4
GRAPH_PROMPT_HEAD = (
    "Forecast the next 12 traffic speeds, one speed per 5-minute step. "
    "Reply with exactly 12 decimal speeds separated by commas.\n"
)


def nearest_neighbor_indexes(
    adjacency: list[list[float]],
    sensor: int,
    k: int = GRAPH_PROMPT_NEIGHBORS,
) -> list[int]:
    """Off-diagonal neighbors, highest weight first. Ties break toward the smaller index."""
    if k <= 0:
        raise GraphForecastError("neighbor count must be positive")
    if sensor < 0 or sensor >= len(adjacency):
        raise GraphForecastError("sensor is outside the graph")
    ranked = sorted(
        (
            (adjacency[sensor][other], other)
            for other in range(len(adjacency))
            if other != sensor and adjacency[sensor][other] > 0.0
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked:
        raise GraphForecastError("sensor has no neighbor")
    return [other for _weight, other in ranked[:k]]


def render_neighbor_prompt(
    target: list[float],
    neighbors: list[list[float]],
) -> str:
    """Local prompt: the target history plus up to four neighbor histories."""
    if not neighbors or len(neighbors) > GRAPH_PROMPT_NEIGHBORS:
        raise GraphForecastError("prompt needs one to four neighbor histories")

    def line(values: list[float]) -> str:
        if len(values) != GRAPH_HISTORY:
            raise GraphForecastError("prompt history is not 12 speeds")
        return ",".join(f"{float(value):.3f}" for value in values)

    rows = [f"target: {line(target)}"]
    rows.extend(
        f"neighbor_{index}: {line(history)}"
        for index, history in enumerate(neighbors, start=1)
    )
    prompt = GRAPH_PROMPT_HEAD + "\n".join(rows) + "\nforecast:"
    lowered = prompt.lower()
    if "freeway" in lowered or "sensor" in lowered or "bay area" in lowered:
        raise GraphForecastError("prompt names a place or a sensor")
    return prompt


