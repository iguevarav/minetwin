from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


SCANIA_CLASS_WINDOWS = (
    (0, 48.0, None),
    (1, 24.0, 48.0),
    (2, 12.0, 24.0),
    (3, 6.0, 12.0),
    (4, 0.0, 6.0),
)

SCANIA_COST_MATRIX = (
    (0, 7, 8, 9, 10),
    (200, 0, 7, 8, 9),
    (300, 200, 0, 7, 8),
    (400, 300, 200, 0, 7),
    (500, 400, 300, 200, 0),
)


@dataclass(frozen=True)
class ScaniaSchema:
    feature_names: tuple[str, ...]
    specification_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("El esquema requiere variables operativas unicas.")
        if len(set(self.specification_names)) != len(self.specification_names):
            raise ValueError("Las especificaciones deben ser unicas.")


@dataclass(frozen=True)
class FeatureReadout:
    split: DatasetSplit
    vehicle_id: str
    time_step: float
    feature_names: tuple[str, ...]
    values: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if not self.vehicle_id:
            raise ValueError("vehicle_id no puede estar vacio.")
        if not isfinite(self.time_step) or self.time_step < 0:
            raise ValueError("time_step debe ser finito y no negativo.")
        if not self.feature_names or len(self.feature_names) != len(self.values):
            raise ValueError("Las variables y sus valores no coinciden.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Las variables deben ser unicas.")
        if any(value is not None and not isfinite(value) for value in self.values):
            raise ValueError("Las variables solo admiten numeros finitos o ausentes.")


@dataclass(frozen=True)
class ImputedReadout:
    source: FeatureReadout
    values: tuple[float, ...]
    missing: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.source.values) or len(self.missing) != len(
            self.source.values
        ):
            raise ValueError("La imputacion no coincide con el esquema de origen.")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("La imputacion debe producir valores finitos.")


@dataclass(frozen=True)
class FeatureWindow:
    split: DatasetSplit
    node_id: str
    vehicle_id: str
    feature_names: tuple[str, ...]
    time_steps: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    missing: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        row_count = len(self.time_steps)
        feature_count = len(self.feature_names)
        if not self.node_id or not self.vehicle_id or not row_count:
            raise ValueError("La ventana requiere nodo, vehiculo y observaciones.")
        if len(self.values) != row_count or len(self.missing) != row_count:
            raise ValueError("Las dimensiones temporales de la ventana no coinciden.")
        if any(len(row) != feature_count for row in (*self.values, *self.missing)):
            raise ValueError("Las dimensiones de variables de la ventana no coinciden.")
        if any(
            current <= previous
            for previous, current in zip(self.time_steps, self.time_steps[1:])
        ):
            raise ValueError("Los tiempos de la ventana deben ser crecientes.")

    @property
    def end_time_step(self) -> float:
        return self.time_steps[-1]


@dataclass(frozen=True)
class VehicleSpecifications:
    split: DatasetSplit
    vehicle_id: str
    names: tuple[str, ...]
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.vehicle_id or len(self.names) != len(self.values):
            raise ValueError("Especificaciones de vehiculo invalidas.")
        if any(not value for value in self.values):
            raise ValueError("Las especificaciones no admiten valores ausentes.")


@dataclass(frozen=True)
class VehicleOutcome:
    split: DatasetSplit
    vehicle_id: str
    class_label: int | None = None
    study_end_time_step: float | None = None
    in_study_repair: bool | None = None

    def __post_init__(self) -> None:
        if not self.vehicle_id:
            raise ValueError("vehicle_id no puede estar vacio.")
        if self.class_label is not None and self.class_label not in range(5):
            raise ValueError("class_label debe estar entre 0 y 4.")
        if self.study_end_time_step is not None and (
            not isfinite(self.study_end_time_step) or self.study_end_time_step < 0
        ):
            raise ValueError("El fin del estudio debe ser finito y no negativo.")
        if self.split == DatasetSplit.TRAIN:
            if self.class_label is not None or self.study_end_time_step is None:
                raise ValueError("El resultado de entrenamiento requiere tiempo a evento.")
            if self.in_study_repair is None:
                raise ValueError("El resultado de entrenamiento requiere estado de reparacion.")
        elif self.class_label is None or any(
            value is not None
            for value in (self.study_end_time_step, self.in_study_repair)
        ):
            raise ValueError("Validacion y prueba requieren una clase observada.")

    def class_at(self, time_step: float) -> int | None:
        if self.split != DatasetSplit.TRAIN:
            raise ValueError(
                "La clase de validacion o prueba solo corresponde al ultimo readout."
            )
        if self.study_end_time_step is None or self.in_study_repair is None:
            return None
        if not self.in_study_repair:
            return 0
        remaining = self.study_end_time_step - time_step
        if remaining < 0:
            raise ValueError("El readout ocurre despues del fin del estudio.")
        if remaining > 48:
            return 0
        if remaining > 24:
            return 1
        if remaining > 12:
            return 2
        if remaining > 6:
            return 3
        return 4


@dataclass(frozen=True)
class WindowConfig:
    size: int = 12
    min_size: int | None = None
    stride: int = 1
    initial_value: float = 0.0

    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size < 1:
            raise ValueError("size debe ser un entero positivo.")
        if type(self.stride) is not int or self.stride < 1:
            raise ValueError("stride debe ser un entero positivo.")
        minimum = self.size if self.min_size is None else self.min_size
        if type(minimum) is not int or minimum < 1 or minimum > self.size:
            raise ValueError("min_size debe estar entre uno y size.")
        if not isfinite(self.initial_value):
            raise ValueError("initial_value debe ser finito.")

    @property
    def effective_min_size(self) -> int:
        return self.size if self.min_size is None else self.min_size


@dataclass(frozen=True)
class SplitSummary:
    split: DatasetSplit
    vehicles: int
    readouts: int
    features: int
    missing_values: int
    specification_rows: int
    outcome_rows: int
