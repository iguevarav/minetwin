from collections import deque
from collections.abc import Iterable, Iterator

from minetwin.data.scania.contracts import (
    FeatureReadout,
    FeatureWindow,
    ImputedReadout,
    VehicleSpecifications,
    WindowConfig,
)
from minetwin.data.scania.partition import CategoricalNodePartitioner


class CausalImputer:
    def __init__(self, initial_value: float = 0.0) -> None:
        self.initial_value = initial_value
        self._vehicle_id: str | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._last: list[float | None] = []

    def transform(self, readout: FeatureReadout) -> ImputedReadout:
        identity = (readout.vehicle_id, readout.feature_names)
        if identity != (self._vehicle_id, self._feature_names):
            self._vehicle_id = readout.vehicle_id
            self._feature_names = readout.feature_names
            self._last = [None] * len(readout.values)
        values = []
        missing = []
        for index, value in enumerate(readout.values):
            absent = value is None
            if absent:
                value = self._last[index]
            if value is None:
                value = self.initial_value
            else:
                self._last[index] = value
            values.append(value)
            missing.append(absent)
        return ImputedReadout(readout, tuple(values), tuple(missing))


def build_windows(
    readouts: Iterable[FeatureReadout],
    specifications: dict[str, VehicleSpecifications],
    partitioner: CategoricalNodePartitioner,
    config: WindowConfig | None = None,
) -> Iterator[FeatureWindow]:
    effective = config or WindowConfig()
    imputer = CausalImputer(effective.initial_value)
    window: deque[ImputedReadout] = deque(maxlen=effective.size)
    current_vehicle: str | None = None
    completed: set[str] = set()
    observations = 0
    node_id: str | None = None
    for raw in readouts:
        if raw.vehicle_id not in specifications:
            raise ValueError(f"Faltan especificaciones para {raw.vehicle_id}.")
        if raw.vehicle_id != current_vehicle:
            if raw.vehicle_id in completed:
                raise ValueError("Los readouts de cada vehiculo deben ser contiguos.")
            if current_vehicle is not None:
                completed.add(current_vehicle)
            current_vehicle = raw.vehicle_id
            window.clear()
            observations = 0
            node_id = partitioner.assign(specifications[raw.vehicle_id])
        window.append(imputer.transform(raw))
        observations += 1
        if len(window) < effective.effective_min_size:
            continue
        if (observations - effective.effective_min_size) % effective.stride:
            continue
        rows = tuple(window)
        yield FeatureWindow(
            split=raw.split,
            node_id=node_id,
            vehicle_id=raw.vehicle_id,
            feature_names=raw.feature_names,
            time_steps=tuple(row.source.time_step for row in rows),
            values=tuple(row.values for row in rows),
            missing=tuple(row.missing for row in rows),
        )
