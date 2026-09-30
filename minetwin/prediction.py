import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from minetwin.domain import (
    ENGINE_SIGNALS,
    AssetStatus,
    EnginePrediction,
    ForecastStatus,
    Quality,
    TelemetryPacket,
    TruckProfile,
)


@dataclass(frozen=True)
class PredictorConfig:
    window_samples: int = 120
    min_samples: int = 20
    min_operating_span_hours: float = 0.25
    min_slope_per_hour: float = 0.01
    min_r_squared: float = 0.8
    critical_indicator: float = 1.0
    max_horizon_hours: float = 12.0
    stale_after_seconds: float = 180.0
    temperature_scale: float = 25.0
    vibration_scale: float = 2.5

    def __post_init__(self) -> None:
        if type(self.min_samples) is not int or type(self.window_samples) is not int:
            raise TypeError("El tamaño de ventana debe ser entero.")
        if not 3 <= self.min_samples <= self.window_samples:
            raise ValueError(
                "Se necesitan al menos tres muestras y una ventana suficiente."
            )
        for name in (
            "min_operating_span_hours",
            "min_slope_per_hour",
            "critical_indicator",
            "max_horizon_hours",
            "stale_after_seconds",
            "temperature_scale",
            "vibration_scale",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo y finito.")
        if not math.isfinite(self.min_r_squared) or not 0 <= self.min_r_squared <= 1:
            raise ValueError("El criterio de estabilidad debe estar entre 0 y 1.")


class EnginePredictor:
    def __init__(self, profile: TruckProfile, config: PredictorConfig) -> None:
        self.profile = profile
        self.config = config
        self._samples: deque[tuple[float, float]] = deque(maxlen=config.window_samples)
        self._last_time: datetime | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._last_time = None

    def evaluate(self, packet: TelemetryPacket, now: datetime) -> EnginePrediction:
        age = (now - packet.source_time).total_seconds()
        reason = None
        if age < 0 or age > self.config.stale_after_seconds:
            reason = "Observación desactualizada o con fecha futura."
        elif packet.asset_status != AssetStatus.AVAILABLE:
            reason = "Camión detenido por falla o mantenimiento."
        elif any(
            reading.quality != Quality.VALID or reading.value is None
            for reading in (packet.reading(name) for name in ENGINE_SIGNALS)
        ):
            reason = "Faltan señales esenciales válidas."
        if reason:
            self.reset()
            return EnginePrediction(
                packet.source_time, ForecastStatus.NOT_ESTIMABLE, reason
            )

        hours = packet.reading("operating_hours").value
        temperature = packet.reading("engine_temperature").value
        vibration = packet.reading("engine_vibration").value
        assert hours is not None and temperature is not None and vibration is not None
        if (
            self._last_time
            and (packet.source_time - self._last_time).total_seconds()
            > self.config.stale_after_seconds
        ):
            self.reset()
        if self._samples and hours < self._samples[-1][0]:
            self.reset()
            return EnginePrediction(
                packet.source_time,
                ForecastStatus.NOT_ESTIMABLE,
                "El horómetro retrocedió; se requiere una nueva ventana.",
                operating_hours=hours,
            )
        self._last_time = packet.source_time
        baseline_temperature, baseline_vibration = self.profile.engine_baseline(
            packet.operating_state
        )
        indicator = 0.5 * (
            (temperature - baseline_temperature) / self.config.temperature_scale
            + (vibration - baseline_vibration) / self.config.vibration_scale
        )
        if not self._samples or hours > self._samples[-1][0]:
            self._samples.append((hours, indicator))
        common = {
            "timestamp": packet.source_time,
            "operating_hours": hours,
            "indicator": indicator,
            "sample_count": len(self._samples),
        }
        if indicator >= self.config.critical_indicator:
            return EnginePrediction(
                **common,
                status=ForecastStatus.ESTIMABLE,
                reason="El indicador alcanzó el umbral configurado.",
                hours_to_threshold=0.0,
            )
        if len(self._samples) < self.config.min_samples or (
            self._samples[-1][0] - self._samples[0][0]
            < self.config.min_operating_span_hours
        ):
            return EnginePrediction(
                **common,
                status=ForecastStatus.NOT_ESTIMABLE,
                reason="Historial operativo insuficiente para estimar una tendencia.",
            )
        samples = np.asarray(self._samples)
        x = samples[:, 0] - samples[0, 0]
        y = samples[:, 1]
        centered_x = x - x.mean()
        slope = float(np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x))
        fitted = y.mean() + slope * centered_x
        total_variance = float(np.sum((y - y.mean()) ** 2))
        r_squared = (
            max(0.0, 1 - float(np.sum((y - fitted) ** 2)) / total_variance)
            if total_variance > 1e-15
            else 0.0
        )
        common.update(slope_per_hour=slope, r_squared=r_squared)
        if (
            slope < self.config.min_slope_per_hour
            or r_squared < self.config.min_r_squared
        ):
            return EnginePrediction(
                **common,
                status=ForecastStatus.NOT_ESTIMABLE,
                reason="La tendencia no es creciente o suficientemente estable.",
            )
        remaining = (self.config.critical_indicator - indicator) / slope
        if remaining > self.config.max_horizon_hours:
            return EnginePrediction(
                **common,
                status=ForecastStatus.OUTSIDE_HORIZON,
                reason="La extrapolación excede el horizonte útil configurado.",
            )
        return EnginePrediction(
            **common,
            status=ForecastStatus.ESTIMABLE,
            hours_to_threshold=remaining,
            reason="Estimación bajo continuidad del régimen, en horas de operación.",
        )
