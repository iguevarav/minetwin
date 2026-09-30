from collections import deque
from datetime import timedelta
from math import ceil

from minetwin.domain import (
    AssetStatus,
    Diagnosis,
    EngineStatus,
    Quality,
    TelemetryPacket,
    TruckProfile,
)


class ComponentMonitor:
    def __init__(
        self, component: str, profile: TruckProfile, step_seconds: int
    ) -> None:
        self.component = component
        self.profile = profile
        self._gap = timedelta(seconds=step_seconds * 1.5)
        self._pressure_window = max(30, ceil(1800 / step_seconds) + 1)
        self.reset()

    def reset(self) -> None:
        self._high_since = self._low_since = self._last_time = None
        self._active = False
        self._pressures = deque(maxlen=self._pressure_window)

    def evaluate(self, packet: TelemetryPacket) -> Diagnosis:
        tire = self.component.startswith("tire:")
        rule = "SIM-TIRE-001" if tire else "SIM-BRAKE-001"
        names = (
            (f"tire_pressure_{self.component.split(':')[1]}",)
            if tire
            else (
                "brake_temperature",
                "load_ratio",
                "slope_percent",
            )
        )
        readings = [packet.reading(name) for name in names]
        if self._last_time and packet.source_time - self._last_time > self._gap:
            self._high_since = self._low_since = None
            self._pressures.clear()
        self._last_time = packet.source_time
        if packet.asset_status == AssetStatus.MAINTENANCE or any(
            r.quality != Quality.VALID or r.value is None for r in readings
        ):
            self._high_since = self._low_since = None
            self._pressures.clear()
            return Diagnosis(
                EngineStatus.UNKNOWN,
                "Se requieren observaciones válidas del componente fuera del mantenimiento.",
                rule_id=rule,
                component=self.component,
            )
        if tire:
            pressure = readings[0].value
            self._pressures.append((packet.source_time, pressure))
            start, first = self._pressures[0]
            falling = (
                packet.source_time - start >= timedelta(minutes=10)
                and first - pressure >= self.profile.tire_pressure_kpa * 0.03
            )
            elevated = pressure < 0.85 * self.profile.tire_pressure_kpa or falling
            recovered = pressure > 0.92 * self.profile.tire_pressure_kpa and not falling
            explanation = f"Neumático {self.component.split(':')[1]}: {pressure:.1f} kPa; referencia {self.profile.tire_pressure_kpa:.1f} kPa. Se evalúan presión baja y pérdida sostenida."
        else:
            temperature, load, slope = (reading.value for reading in readings)
            reference = self.profile.brake_baseline(load, slope)
            elevated, recovered = (
                temperature - reference >= 30,
                temperature - reference <= 20,
            )
            explanation = f"Frenos: {temperature:.1f} °C; referencia {reference:.1f} °C para carga {load:.0%} y pendiente {slope:.1f} %. Desviación {temperature - reference:+.1f} °C."
        now = packet.source_time
        if elevated:
            self._high_since = self._high_since or now
            self._low_since = None
            if now - self._high_since >= timedelta(minutes=3):
                self._active = True
        elif recovered:
            self._low_since = self._low_since or now
            self._high_since = None
            if now - self._low_since >= timedelta(minutes=3):
                self._active = False
        else:
            self._high_since = self._low_since = None
        status = (
            EngineStatus.ALERT
            if self._active
            else (EngineStatus.WATCH if elevated else EngineStatus.NORMAL)
        )
        return Diagnosis(status, explanation, rule_id=rule, component=self.component)
