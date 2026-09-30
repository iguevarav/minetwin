from datetime import datetime, timedelta

from minetwin.domain import (
    ENGINE_SIGNALS,
    AssetStatus,
    Diagnosis,
    EngineStatus,
    Quality,
    TelemetryPacket,
    TruckProfile,
)


class EngineMonitor:
    def __init__(self, profile: TruckProfile, step_seconds: int) -> None:
        self._profile = profile
        self._max_gap = timedelta(seconds=step_seconds * 1.5)
        self._persistence = timedelta(minutes=3)
        self._high_since: datetime | None = None
        self._low_since: datetime | None = None
        self._last_observation: datetime | None = None
        self._alert_active = False

    def evaluate(self, packet: TelemetryPacket) -> Diagnosis:
        now = packet.source_time
        if packet.asset_status == AssetStatus.FAILED:
            self._high_since = self._low_since = None
            self._alert_active = True
            return Diagnosis(
                EngineStatus.ALERT,
                "Falla de motor reportada. El camión requiere reparación correctiva.",
            )
        if packet.asset_status == AssetStatus.MAINTENANCE:
            self._high_since = self._low_since = None
            return Diagnosis(
                EngineStatus.UNKNOWN, "Motor detenido durante la intervención."
            )
        if self._last_observation and now - self._last_observation > self._max_gap:
            self._high_since = self._low_since = None
        self._last_observation = now
        unavailable = [
            reading.name
            for reading in (packet.reading(name) for name in ENGINE_SIGNALS)
            if reading.quality != Quality.VALID or reading.value is None
        ]
        if unavailable:
            self._high_since = self._low_since = None
            return Diagnosis(
                EngineStatus.UNKNOWN,
                "No estimable: faltan lecturas válidas de "
                + ", ".join(unavailable)
                + ".",
            )
        temperature = packet.reading("engine_temperature").value
        vibration = packet.reading("engine_vibration").value
        assert temperature is not None and vibration is not None
        baseline_temperature, baseline_vibration = self._profile.engine_baseline(
            packet.operating_state
        )
        temperature_residual = temperature - baseline_temperature
        vibration_residual = vibration - baseline_vibration
        elevated = temperature_residual >= 8.0 and vibration_residual >= 0.8
        recovered = temperature_residual <= 6.0 and vibration_residual <= 0.6

        if elevated:
            self._high_since = self._high_since or now
            self._low_since = None
            if now - self._high_since >= self._persistence:
                self._alert_active = True
        elif recovered:
            self._high_since = None
            self._low_since = self._low_since or now
            if now - self._low_since >= self._persistence:
                self._alert_active = False
        else:
            self._high_since = self._low_since = None

        if self._alert_active:
            status = EngineStatus.ALERT
            message = (
                "Alerta de motor: desviaciones persistentes respecto al régimen. "
                "Apertura: temperatura ≥ 8 °C y vibración ≥ 0.8 mm/s durante "
                "3 minutos. Cierre: ambas ≤ 6 °C y ≤ 0.6 mm/s durante 3 minutos. "
                "Se recomienda inspeccionar el motor."
            )
        elif elevated or not recovered:
            status = EngineStatus.WATCH
            message = "Desviación en observación; aún no cumple la regla persistente."
        else:
            status = EngineStatus.NORMAL
            message = "Sin anomalía detectada respecto al régimen operacional."
        explanation = (
            f"{message} Desviación actual: {temperature_residual:+.2f} °C y "
            f"{vibration_residual:+.2f} mm/s. Referencias: "
            f"{baseline_temperature:.1f} °C y {baseline_vibration:.1f} mm/s."
        )
        return Diagnosis(status, explanation, temperature_residual, vibration_residual)
