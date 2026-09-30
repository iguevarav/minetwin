import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from minetwin.domain import (
    WHEELS,
    AssetStatus,
    Intervention,
    OperatingState,
    Scenario,
    TruckProfile,
)


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    scenario: Scenario = Scenario.NORMAL
    step_seconds: int = 60
    history_limit: int = 720
    alert_limit: int = 100
    event_limit: int = 1000
    engine_wear_per_hour: float = 0.18
    failure_wear: float = 1.0
    start_time: datetime = datetime(2026, 1, 1, tzinfo=UTC)

    def __post_init__(self) -> None:
        for name in ("step_seconds", "history_limit", "alert_limit", "event_limit"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} debe ser un entero positivo.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed debe ser un entero no negativo.")
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario debe ser un escenario válido.")
        if self.start_time.utcoffset() is None:
            raise ValueError("start_time debe incluir zona horaria.")
        for name in ("engine_wear_per_hour", "failure_wear"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo y finito.")


CYCLE = (
    (OperatingState.WAITING, 120),
    (OperatingState.LOADING, 180),
    (OperatingState.HAULING, 600),
    (OperatingState.UNLOADING, 120),
    (OperatingState.RETURNING, 480),
)


class SimulatedTruck:
    def __init__(
        self,
        config: SimulationConfig,
        profile: TruckProfile,
        truck_id: str = "TRUCK-001",
        node_id: str = "alpha",
    ) -> None:
        self.config = config
        self.profile = profile
        self.truck_id = truck_id
        self.node_id = node_id
        self._time = config.start_time
        self._sequence = 0
        self._cycle_index = 0
        self._state_elapsed = 0
        self._operating_seconds = 0
        self._cycles = 0
        self._wear = 0.0
        self._brake_excess = 0.0
        self._pressure_loss = {wheel: 0.0 for wheel in WHEELS}
        self._engine_failed = False
        self._status = AssetStatus.AVAILABLE
        self._available_seconds = 0.0
        self._failed_seconds = 0.0
        self._maintenance_seconds = 0.0
        self._failure_count = 0
        self._pending_failures: list[tuple[datetime, float]] = []

    @property
    def time(self) -> datetime:
        return self._time

    @property
    def status(self) -> AssetStatus:
        return self._status

    @property
    def operating_hours(self) -> float:
        return self._operating_seconds / 3600

    def statistics(self) -> dict[str, float | int | None]:
        elapsed = (self.time - self.config.start_time).total_seconds()
        return {
            "scheduled_seconds": elapsed,
            "available_seconds": self._available_seconds,
            "failed_seconds": self._failed_seconds,
            "maintenance_seconds": self._maintenance_seconds,
            "operating_hours": self.operating_hours,
            "failures": self._failure_count,
            "availability": self._available_seconds / elapsed if elapsed else None,
        }

    def drain_failures(self) -> tuple[tuple[datetime, float], ...]:
        failures = tuple(self._pending_failures)
        self._pending_failures.clear()
        return failures

    def begin_maintenance(self) -> None:
        self._status = AssetStatus.MAINTENANCE

    def _engine_wear_rate(self, state: OperatingState) -> float:
        if state == OperatingState.WAITING:
            return 0.0
        base = self.config.engine_wear_per_hour / 3600
        if self.config.scenario == Scenario.ENGINE_DEGRADATION:
            return base
        if self.config.scenario != Scenario.ENGINE_VARIABLE_DEGRADATION:
            return 0.0
        progression = 0.65 + 0.7 * (1 - math.exp(-self.operating_hours / 2))
        regime = {
            OperatingState.LOADING: 1.1,
            OperatingState.HAULING: 1.25,
            OperatingState.UNLOADING: 0.85,
            OperatingState.RETURNING: 0.75,
        }[state]
        return base * progression * regime

    def repair(
        self,
        intervention: Intervention,
        remaining_fraction: float,
        component: str = "engine",
    ) -> None:
        if self._status != AssetStatus.MAINTENANCE:
            raise ValueError("El activo no está en mantenimiento.")
        factor = (
            remaining_fraction if intervention == Intervention.PARTIAL_REPAIR else 0.0
        )
        if component == "engine":
            self._wear *= factor
            self._engine_failed = self._wear >= self.config.failure_wear
        elif component == "brakes":
            self._brake_excess *= factor
        elif component in self.profile.components:
            self._pressure_loss[component.split(":")[1]] *= factor
        else:
            raise ValueError("Componente no admitido.")
        self._status = (
            AssetStatus.FAILED if self._engine_failed else AssetStatus.AVAILABLE
        )

    def advance(self, seconds: float | None = None) -> str:
        remaining = self.config.step_seconds if seconds is None else seconds
        if not math.isfinite(remaining) or remaining <= 0:
            raise ValueError("El avance debe ser positivo y finito.")
        end_time = self.time + timedelta(seconds=remaining)
        while remaining > 1e-9:
            if self._status != AssetStatus.AVAILABLE:
                if self._status == AssetStatus.FAILED:
                    self._failed_seconds += remaining
                else:
                    self._maintenance_seconds += remaining
                self._time += timedelta(seconds=remaining)
                break
            state, duration = CYCLE[self._cycle_index]
            elapsed = min(remaining, duration - self._state_elapsed)
            wear_rate = self._engine_wear_rate(state)
            if wear_rate:
                elapsed = min(
                    elapsed,
                    max(0.0, (self.config.failure_wear - self._wear) / wear_rate),
                )
            if state != OperatingState.WAITING:
                self._operating_seconds += elapsed
                if self.config.scenario == Scenario.BRAKE_STRESS:
                    self._brake_excess = min(
                        100.0, self._brake_excess + elapsed / 60 * 0.6
                    )
                if self.config.scenario == Scenario.TIRE_LEAK:
                    self._pressure_loss["FL"] = min(
                        self.profile.tire_pressure_kpa,
                        self._pressure_loss["FL"] + elapsed / 60 * 1.8,
                    )
            self._wear += elapsed * wear_rate
            self._available_seconds += elapsed
            self._time += timedelta(seconds=elapsed)
            self._state_elapsed += elapsed
            remaining -= elapsed
            if self._wear >= self.config.failure_wear - 1e-12:
                self._status = AssetStatus.FAILED
                self._engine_failed = True
                self._failure_count += 1
                self._pending_failures.append((self.time, self.operating_hours))
            if self._state_elapsed >= duration - 1e-9:
                self._state_elapsed = 0
                self._cycle_index = (self._cycle_index + 1) % len(CYCLE)
                if self._cycle_index == 0:
                    self._cycles += 1
        self._time = end_time
        self._sequence += 1
        return self.snapshot_json()

    def snapshot_json(self) -> str:
        state, duration = CYCLE[self._cycle_index]
        fraction = self._state_elapsed / duration
        load_ratio = {
            OperatingState.WAITING: 0.0,
            OperatingState.LOADING: fraction,
            OperatingState.HAULING: 1.0,
            OperatingState.UNLOADING: 1.0 - fraction,
            OperatingState.RETURNING: 0.0,
        }[state]
        temperature, vibration = self.profile.engine_baseline(state)
        timestamp_key = round(
            (self.time - self.config.start_time).total_seconds() * 1_000_000
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, timestamp_key])
        )
        temperature += 25 * self._wear + rng.normal(0, 0.25)
        vibration += 2.5 * self._wear + rng.normal(0, 0.03)
        available = self._status == AssetStatus.AVAILABLE
        slope = (
            -8.0
            if state == OperatingState.HAULING
            else (4.0 if state == OperatingState.RETURNING else 0.0)
        )
        brake_temperature = (
            self.profile.brake_baseline(load_ratio, slope)
            + self._brake_excess
            + rng.normal(0, 0.3)
        )
        return json.dumps(
            {
                "schema_version": 1,
                "event_id": f"{self.node_id}-{self.truck_id}-{self._sequence:08d}",
                "node_id": self.node_id,
                "truck_id": self.truck_id,
                "source_time": self._time.isoformat(),
                "operating_state": state.value,
                "asset_status": self._status.value,
                "cycles": self._cycles,
                "load_ratio": load_ratio,
                "engine_temperature_f": temperature * 9 / 5 + 32 if available else None,
                "engine_vibration_mm_s": max(0.0, vibration) if available else None,
                "operating_hours": self._operating_seconds / 3600,
                "brake_temperature_c": brake_temperature if available else None,
                "slope_percent": slope,
                **{
                    f"tire_{wheel}_kpa": max(
                        0.0,
                        self.profile.tire_pressure_kpa
                        - self._pressure_loss[wheel]
                        + rng.normal(0, 0.4),
                    )
                    for wheel in WHEELS
                },
            },
            allow_nan=False,
        )
