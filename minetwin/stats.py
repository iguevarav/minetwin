from dataclasses import asdict, dataclass
from math import erfc, isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class WilcoxonResult:
    pairs: int
    statistic: float
    p_value: float
    rank_biserial: float
    method: str

    def as_dict(self) -> dict:
        return asdict(self)


def wilcoxon_signed_rank(differences) -> WilcoxonResult:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("differences debe ser un vector finito.")
    values = values[values != 0]
    if not len(values):
        return WilcoxonResult(0, 0.0, 1.0, 0.0, "all_zero")
    ranks, has_ties, tie_counts = _average_ranks(np.abs(values))
    positive = float(np.sum(ranks[values > 0]))
    negative = float(np.sum(ranks[values < 0]))
    statistic = min(positive, negative)
    total = positive + negative
    effect = (positive - negative) / total if total else 0.0
    if len(values) <= 25 and not has_ties:
        p_value = _exact_two_sided(round(statistic), len(values))
        method = "exact"
    else:
        mean = len(values) * (len(values) + 1) / 4
        variance = len(values) * (len(values) + 1) * (2 * len(values) + 1) / 24
        variance -= sum(count**3 - count for count in tie_counts) / 48
        distance = max(0.0, abs(positive - mean) - 0.5)
        z_score = distance / sqrt(variance) if variance > 0 else 0.0
        p_value = erfc(z_score / sqrt(2))
        method = "normal"
    return WilcoxonResult(
        len(values), statistic, min(1.0, p_value), effect, method
    )


def bootstrap_mean_interval(
    values,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("values debe ser un vector finito no vacío.")
    if not 0 < confidence < 1:
        raise ValueError("confidence debe estar entre cero y uno.")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples debe ser un entero positivo.")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    block = max(1, min(samples, 1_000_000 // len(data)))
    for start in range(0, samples, block):
        stop = min(samples, start + block)
        indexes = generator.integers(0, len(data), size=(stop - start, len(data)))
        means[start:stop] = np.mean(data[indexes], axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, (alpha, 1 - alpha))
    return float(lower), float(upper)


def holm_adjust(p_values) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Los valores p deben estar entre cero y uno.")
    count = len(values)
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [0.0] * count
    previous = 0.0
    for position, index in enumerate(order):
        current = min(1.0, (count - position) * values[index])
        previous = max(previous, current)
        adjusted[index] = previous
    return adjusted


def _average_ranks(values: np.ndarray) -> tuple[np.ndarray, bool, list[int]]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    tie_counts = []
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        tie_counts.append(stop - start)
        start = stop
    return ranks, any(count > 1 for count in tie_counts), tie_counts


def _exact_two_sided(observed: int, count: int) -> float:
    total = count * (count + 1) // 2
    combinations = [0] * (total + 1)
    combinations[0] = 1
    reachable = 0
    for rank in range(1, count + 1):
        for value in range(reachable, -1, -1):
            combinations[value + rank] += combinations[value]
        reachable += rank
    tail = sum(combinations[: observed + 1])
    return min(1.0, 2 * tail / (2**count))
