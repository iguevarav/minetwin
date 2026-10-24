from dataclasses import dataclass
from math import exp, log

import numpy as np


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("La temperatura debe ser positiva y finita.")

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = _probabilities(probabilities)
        logits = np.log(np.clip(values, 1e-15, 1)) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        calibrated = np.exp(logits)
        return calibrated / calibrated.sum(axis=1, keepdims=True)


def fit_temperature_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> TemperatureCalibrator:
    values = _probabilities(probabilities)
    targets = _labels(labels, len(values))
    lower = log(0.05)
    upper = log(20.0)
    ratio = (5**0.5 - 1) / 2
    first = upper - ratio * (upper - lower)
    second = lower + ratio * (upper - lower)
    for _ in range(80):
        first_loss = _negative_log_likelihood(values, targets, exp(first))
        second_loss = _negative_log_likelihood(values, targets, exp(second))
        if first_loss <= second_loss:
            upper = second
            second = first
            first = upper - ratio * (upper - lower)
        else:
            lower = first
            first = second
            second = lower + ratio * (upper - lower)
    return TemperatureCalibrator(exp((lower + upper) / 2))


def _negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    calibrated = TemperatureCalibrator(temperature).transform(probabilities)
    return float(-np.mean(np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-15, 1))))


def _probabilities(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 5 or not len(result):
        raise ValueError("Se esperan probabilidades para cinco clases.")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Las probabilidades deben ser finitas y no negativas.")
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Cada predicción requiere masa de probabilidad positiva.")
    return result / totals


def _labels(values: np.ndarray, expected: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != expected:
        raise ValueError("Etiquetas y probabilidades no coinciden.")
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError("Las etiquetas deben estar entre cero y cuatro.")
    return labels
