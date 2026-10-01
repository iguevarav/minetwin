from dataclasses import dataclass
from enum import StrEnum
from math import isclose, isfinite

from minetwin.domain import AssetStatus, EngineStatus, ForecastStatus


class PublishedCondition(StrEnum):
    NORMAL = "normal"
    WATCH = "watch"
    ALERT = "alert"


@dataclass(frozen=True)
class RiskAssessment:
    predicted_class: int
    probabilities: tuple[float, ...]
    condition: PublishedCondition
    recommendation: str
    model_id: str

    def __post_init__(self) -> None:
        if self.predicted_class not in range(5):
            raise ValueError("predicted_class debe estar entre cero y cuatro.")
        if len(self.probabilities) != 5 or any(
            not isfinite(value) or value < 0 for value in self.probabilities
        ):
            raise ValueError("Se requieren cinco probabilidades validas.")
        if not isclose(sum(self.probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("Las probabilidades deben sumar uno.")
        if not self.recommendation or not self.model_id:
            raise ValueError("La evaluacion requiere recomendacion y modelo.")


@dataclass(frozen=True)
class PublishedTwinState:
    asset_id: str
    node_id: str
    component: str
    source: str
    split: str
    sequence: int
    risk: RiskAssessment
    quality: float | None
    training_id: str
    data_version: str
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not all((self.asset_id, self.node_id, self.component, self.source, self.split)):
            raise ValueError("El estado publicado requiere identidad completa.")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence debe ser un entero positivo.")
        if self.quality is not None and (
            not isfinite(self.quality) or not 0 <= self.quality <= 1
        ):
            raise ValueError("quality debe estar entre cero y uno.")
        if not self.training_id or not self.data_version:
            raise ValueError("El estado requiere versiones de datos y entrenamiento.")
        if self.schema_version != 2:
            raise ValueError("Version de estado publicado no admitida.")


@dataclass(frozen=True)
class PublishedOperationalState:
    truck_id: str
    node_id: str
    version: int
    source_time: str
    asset_status: AssetStatus
    engine_status: EngineStatus
    forecast_status: ForecastStatus | None
    hours_to_threshold: float | None
    cycles: int
    capacity_tonnes: float
    active_alert: bool
    open_order: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.truck_id or not self.node_id or not self.source_time:
            raise ValueError("El estado operativo requiere identidad y tiempo.")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version debe ser un entero positivo.")
        if type(self.cycles) is not int or self.cycles < 0:
            raise ValueError("cycles debe ser un entero no negativo.")
        if not isfinite(self.capacity_tonnes) or self.capacity_tonnes <= 0:
            raise ValueError("capacity_tonnes debe ser positiva y finita.")
        if self.hours_to_threshold is not None and (
            not isfinite(self.hours_to_threshold) or self.hours_to_threshold < 0
        ):
            raise ValueError("hours_to_threshold debe ser no negativo y finito.")
        if self.schema_version != 1:
            raise ValueError("Version de estado operativo no admitida.")
