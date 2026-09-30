from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class OperatingState(StrEnum):
    WAITING = "waiting"
    LOADING = "loading"
    HAULING = "hauling"
    UNLOADING = "unloading"
    RETURNING = "returning"


class Scenario(StrEnum):
    NORMAL = "normal"
    ENGINE_DEGRADATION = "engine_degradation"
    ENGINE_VARIABLE_DEGRADATION = "engine_variable_degradation"
    BRAKE_STRESS = "brake_stress"
    TIRE_LEAK = "tire_leak"


class WireFormat(StrEnum):
    JSON = "json"
    XML = "xml"
    CSV = "csv"


WHEELS = ("FL", "FR", "RL1", "RR1", "RL2", "RR2")
ENGINE_SIGNALS = (
    "load_ratio",
    "engine_temperature",
    "engine_vibration",
    "operating_hours",
)


class Quality(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"


class EngineStatus(StrEnum):
    NORMAL = "normal"
    WATCH = "watch"
    ALERT = "alert"
    UNKNOWN = "unknown"


class AssetStatus(StrEnum):
    AVAILABLE = "available"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class ForecastStatus(StrEnum):
    ESTIMABLE = "estimable"
    NOT_ESTIMABLE = "not_estimable"
    OUTSIDE_HORIZON = "outside_horizon"


class Intervention(StrEnum):
    PARTIAL_REPAIR = "partial_repair"
    REPLACEMENT = "replacement"


class OrderStatus(StrEnum):
    OPEN = "open"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MaintenancePolicy(StrEnum):
    NONE = "none"
    THRESHOLD = "threshold"
    PREDICTIVE = "predictive"


@dataclass(frozen=True)
class TruckProfile:
    id: str = "SIM-ALPHA"
    capacity_tonnes: float = 150.0
    traction: str = "mechanical"
    temperature_offset: float = 0.0
    vibration_factor: float = 1.0
    brake_reference: float = 55.0
    tire_pressure_kpa: float = 750.0

    def engine_baseline(self, state: OperatingState) -> tuple[float, float]:
        temperature, vibration = {
            OperatingState.WAITING: (65.0, 1.0),
            OperatingState.LOADING: (78.0, 1.8),
            OperatingState.HAULING: (88.0, 2.8),
            OperatingState.UNLOADING: (80.0, 2.0),
            OperatingState.RETURNING: (72.0, 1.5),
        }[state]
        return temperature + self.temperature_offset, vibration * self.vibration_factor

    def brake_baseline(self, load_ratio: float, slope_percent: float) -> float:
        return self.brake_reference + max(0.0, -slope_percent) * load_ratio * 3.5

    @property
    def components(self) -> tuple[str, ...]:
        return ("engine", "brakes", *(f"tire:{wheel}" for wheel in WHEELS))


@dataclass(frozen=True)
class SensorReading:
    name: str
    value: float | None
    unit: str
    quality: Quality
    reason: str | None = None


@dataclass(frozen=True)
class TelemetryPacket:
    event_id: str
    node_id: str
    truck_id: str
    source_time: datetime
    received_time: datetime
    operating_state: OperatingState
    cycles: int
    readings: tuple[SensorReading, ...]
    schema_version: int = 1
    asset_status: AssetStatus = AssetStatus.AVAILABLE

    def reading(self, name: str) -> SensorReading:
        return next(
            (reading for reading in self.readings if reading.name == name),
            SensorReading(name, None, "", Quality.MISSING, "Sensor ausente."),
        )


@dataclass(frozen=True)
class Diagnosis:
    status: EngineStatus
    explanation: str
    temperature_residual: float | None = None
    vibration_residual: float | None = None
    rule_id: str = "SIM-ENGINE-001"
    rules_version: str = "1"
    component: str = "engine"


@dataclass(frozen=True)
class Alert:
    id: str
    truck_id: str
    opened_at: datetime
    explanation: str
    closed_at: datetime | None = None
    rule_id: str = "SIM-ENGINE-001"
    component: str = "engine"


@dataclass(frozen=True)
class TwinState:
    observation: TelemetryPacket
    diagnosis: Diagnosis
    version: int
    component_diagnoses: tuple[Diagnosis, ...] = ()


@dataclass(frozen=True)
class EnginePrediction:
    timestamp: datetime
    status: ForecastStatus
    reason: str
    operating_hours: float | None = None
    indicator: float | None = None
    hours_to_threshold: float | None = None
    slope_per_hour: float | None = None
    r_squared: float | None = None
    sample_count: int = 0
    rules_version: str = "engine-trend-1"


@dataclass(frozen=True)
class WorkOrder:
    id: str
    truck_id: str
    component: str
    intervention: Intervention
    reason: str
    created_at: datetime
    status: OrderStatus = OrderStatus.OPEN
    queued_at: datetime | None = None
    corrective: bool = False
    duration_seconds: int | None = None
    started_at: datetime | None = None
    due_at: datetime | None = None
    closed_at: datetime | None = None


@dataclass(frozen=True)
class SimulationEvent:
    sequence: int
    timestamp: datetime
    kind: str
    operating_hours: float
    component: str = "engine"
    order_id: str | None = None
    alert_id: str | None = None
    description: str = ""
