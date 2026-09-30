import csv
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from minetwin.data.scania.contracts import (
    DatasetSplit,
    FeatureReadout,
    FeatureWindow,
    VehicleOutcome,
)
from minetwin.data.scania.loader import (
    READOUT_FILES,
    ScaniaDataset,
    feature_readout_from_row,
)
from minetwin.data.scania.manifest import ScaniaManifest
from minetwin.data.scania.preprocessing import CausalImputer


@dataclass(frozen=True)
class ReplayVehicle:
    split: DatasetSplit
    vehicle_id: str
    node_id: str
    start_byte: int
    end_byte: int
    readouts: int
    first_time_step: float
    last_time_step: float
    class_label: int | None
    study_end_time_step: float | None
    in_study_repair: bool | None


@dataclass(frozen=True)
class ScaniaReplayView:
    split: DatasetSplit
    vehicle_id: str
    node_id: str
    position: int
    readout_count: int
    current: FeatureReadout
    history: tuple[FeatureReadout, ...]
    observed_class: int | None
    window: FeatureWindow
    source: str = "SCANIA Component X"
    evidence: str = "observed"


def prepare_replay_store(
    dataset_root: Path,
    phase_one_output: Path,
    output: Path,
) -> dict:
    dataset = ScaniaDataset(dataset_root)
    phase_one_output = Path(phase_one_output)
    manifest = ScaniaManifest.read(phase_one_output / "dataset_manifest.json")
    manifest.verify(dataset.root)
    assignments = _assignments(phase_one_output / "partitions.csv")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        vehicles = []
        for split in DatasetSplit:
            indexed = _index_split(dataset, split, assignments)
            outcomes = dataset.load_outcomes(split)
            vehicles.extend(_with_outcomes(indexed, outcomes))
        fingerprints = {item.name: asdict(item) for item in manifest.files}
        payload = {
            "schema_version": 1,
            "dataset": "SCANIA Component X",
            "dataset_version": manifest.version,
            "feature_names": dataset.schema.feature_names,
            "source_files": {
                split: fingerprints[READOUT_FILES[split]] for split in DatasetSplit
            },
            "vehicles": [asdict(vehicle) for vehicle in vehicles],
        }
        (output / "index.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        summary = {
            "dataset": payload["dataset"],
            "version": payload["dataset_version"],
            "features": len(dataset.schema.feature_names),
            "vehicles": len(vehicles),
            "readouts": sum(vehicle.readouts for vehicle in vehicles),
            "splits": {
                split: {
                    "vehicles": sum(vehicle.split == split for vehicle in vehicles),
                    "readouts": sum(
                        vehicle.readouts
                        for vehicle in vehicles
                        if vehicle.split == split
                    ),
                }
                for split in DatasetSplit
            },
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


class ScaniaReplayStore:
    def __init__(self, dataset_root: Path, index: Path) -> None:
        self.dataset_root = Path(dataset_root)
        payload = json.loads(Path(index).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Versión de índice SCANIA no admitida.")
        self.feature_names = tuple(payload["feature_names"])
        self._sources = {
            DatasetSplit(split): values
            for split, values in payload["source_files"].items()
        }
        self._vehicles = tuple(
            ReplayVehicle(
                split=DatasetSplit(row["split"]),
                vehicle_id=row["vehicle_id"],
                node_id=row["node_id"],
                start_byte=row["start_byte"],
                end_byte=row["end_byte"],
                readouts=row["readouts"],
                first_time_step=row["first_time_step"],
                last_time_step=row["last_time_step"],
                class_label=row["class_label"],
                study_end_time_step=row["study_end_time_step"],
                in_study_repair=row["in_study_repair"],
            )
            for row in payload["vehicles"]
        )
        self._by_identity = {
            (vehicle.split, vehicle.vehicle_id): vehicle for vehicle in self._vehicles
        }
        self._validate_sources()

    def nodes(self, split: DatasetSplit | None = None) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    vehicle.node_id
                    for vehicle in self._vehicles
                    if split is None or vehicle.split == split
                }
            )
        )

    def vehicles(
        self,
        split: DatasetSplit,
        node_id: str | None = None,
    ) -> tuple[ReplayVehicle, ...]:
        return tuple(
            vehicle
            for vehicle in self._vehicles
            if vehicle.split == split
            and (node_id is None or vehicle.node_id == node_id)
        )

    def vehicle(self, split: DatasetSplit, vehicle_id: str) -> ReplayVehicle:
        try:
            return self._by_identity[(split, vehicle_id)]
        except KeyError as error:
            raise ValueError(
                "El vehículo no pertenece al split seleccionado."
            ) from error

    def readouts(
        self,
        split: DatasetSplit,
        vehicle_id: str,
    ) -> tuple[FeatureReadout, ...]:
        vehicle = self.vehicle(split, vehicle_id)
        path = self.dataset_root / READOUT_FILES[split]
        fields = ("vehicle_id", "time_step", *self.feature_names)
        rows = []
        with path.open("rb") as source:
            source.seek(vehicle.start_byte)
            while source.tell() < vehicle.end_byte:
                line = source.readline()
                if not line:
                    break
                values = next(csv.reader([line.decode("utf-8")]))
                if len(values) != len(fields):
                    raise ValueError("Fila SCANIA incompatible con el índice.")
                rows.append(
                    feature_readout_from_row(
                        split,
                        dict(zip(fields, values, strict=True)),
                        self.feature_names,
                    )
                )
        if len(rows) != vehicle.readouts:
            raise ValueError("El índice SCANIA no coincide con el archivo de origen.")
        return tuple(rows)

    def _validate_sources(self) -> None:
        for split, expected in self._sources.items():
            path = self.dataset_root / READOUT_FILES[split]
            if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
                raise ValueError(
                    f"El archivo de {split.value} no coincide con el índice."
                )


class ScaniaReplaySession:
    def __init__(
        self,
        store: ScaniaReplayStore,
        split: DatasetSplit = DatasetSplit.VALIDATION,
        node_id: str | None = None,
        vehicle_id: str | None = None,
        window_size: int = 12,
    ) -> None:
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size debe ser un entero positivo.")
        self.store = store
        self.window_size = window_size
        self._position = 0
        effective_node = node_id or store.nodes(split)[0]
        candidates = store.vehicles(split, effective_node)
        if not candidates:
            raise ValueError(
                "El nodo no contiene vehículos para el split seleccionado."
            )
        self.select(split, vehicle_id or candidates[0].vehicle_id)

    @property
    def split(self) -> DatasetSplit:
        return self._vehicle.split

    @property
    def node_id(self) -> str:
        return self._vehicle.node_id

    @property
    def vehicle_id(self) -> str:
        return self._vehicle.vehicle_id

    @property
    def position(self) -> int:
        return self._position

    @property
    def readout_count(self) -> int:
        return len(self._readouts)

    @property
    def view(self) -> ScaniaReplayView:
        return ScaniaReplayView(
            split=self.split,
            vehicle_id=self.vehicle_id,
            node_id=self.node_id,
            position=self.position,
            readout_count=self.readout_count,
            current=self._readouts[self.position],
            history=self._readouts[: self.position + 1],
            observed_class=self._observed_class(),
            window=self._window(),
        )

    def select(self, split: DatasetSplit, vehicle_id: str) -> None:
        if not isinstance(split, DatasetSplit):
            raise TypeError("split debe ser DatasetSplit.")
        self._vehicle = self.store.vehicle(split, vehicle_id)
        self._readouts = self.store.readouts(split, vehicle_id)
        self._position = 0

    def advance(self, steps: int = 1) -> ScaniaReplayView:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps debe ser un entero positivo.")
        self._position = min(len(self._readouts) - 1, self._position + steps)
        return self.view

    def retreat(self, steps: int = 1) -> ScaniaReplayView:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps debe ser un entero positivo.")
        self._position = max(0, self._position - steps)
        return self.view

    def reset(self) -> ScaniaReplayView:
        self._position = 0
        return self.view

    def _window(self) -> FeatureWindow:
        imputer = CausalImputer()
        transformed = [
            imputer.transform(readout)
            for readout in self._readouts[: self._position + 1]
        ]
        selected = transformed[-self.window_size :]
        return FeatureWindow(
            split=self.split,
            node_id=self.node_id,
            vehicle_id=self.vehicle_id,
            feature_names=self.store.feature_names,
            time_steps=tuple(item.source.time_step for item in selected),
            values=tuple(item.values for item in selected),
            missing=tuple(item.missing for item in selected),
        )

    def _observed_class(self) -> int | None:
        if self.split != DatasetSplit.TRAIN:
            return (
                self._vehicle.class_label
                if self.position == self.readout_count - 1
                else None
            )
        if (
            self._vehicle.study_end_time_step is None
            or self._vehicle.in_study_repair is None
        ):
            return None
        outcome = VehicleOutcome(
            split=self.split,
            vehicle_id=self.vehicle_id,
            study_end_time_step=self._vehicle.study_end_time_step,
            in_study_repair=self._vehicle.in_study_repair,
        )
        return outcome.class_at(self._readouts[self.position].time_step)


def _index_split(
    dataset: ScaniaDataset,
    split: DatasetSplit,
    assignments: dict[tuple[DatasetSplit, str], str],
) -> list[ReplayVehicle]:
    path = dataset.root / READOUT_FILES[split]
    rows = []
    with path.open("rb") as source:
        source.readline()
        current_id = None
        start = source.tell()
        count = 0
        first_time = 0.0
        last_time = 0.0
        while True:
            line_start = source.tell()
            line = source.readline()
            if not line:
                if current_id is not None:
                    rows.append(
                        _indexed_vehicle(
                            split,
                            current_id,
                            assignments,
                            start,
                            line_start,
                            count,
                            first_time,
                            last_time,
                        )
                    )
                break
            values = next(csv.reader([line.decode("utf-8")]))
            vehicle_id = values[0].strip()
            time_step = float(values[1])
            if current_id is not None and vehicle_id != current_id:
                rows.append(
                    _indexed_vehicle(
                        split,
                        current_id,
                        assignments,
                        start,
                        line_start,
                        count,
                        first_time,
                        last_time,
                    )
                )
                start = line_start
                count = 0
                first_time = time_step
            elif current_id is None:
                first_time = time_step
            current_id = vehicle_id
            last_time = time_step
            count += 1
    return rows


def _indexed_vehicle(
    split: DatasetSplit,
    vehicle_id: str,
    assignments: dict[tuple[DatasetSplit, str], str],
    start: int,
    end: int,
    count: int,
    first_time: float,
    last_time: float,
) -> ReplayVehicle:
    try:
        node_id = assignments[(split, vehicle_id)]
    except KeyError as error:
        raise ValueError(f"Falta partición para {split.value}/{vehicle_id}.") from error
    return ReplayVehicle(
        split,
        vehicle_id,
        node_id,
        start,
        end,
        count,
        first_time,
        last_time,
        None,
        None,
        None,
    )


def _with_outcomes(
    indexed: list[ReplayVehicle],
    outcomes: dict[str, VehicleOutcome],
) -> list[ReplayVehicle]:
    result = []
    for vehicle in indexed:
        outcome = outcomes.get(vehicle.vehicle_id)
        if outcome is None:
            result.append(vehicle)
            continue
        result.append(
            replace(
                vehicle,
                class_label=outcome.class_label,
                study_end_time_step=outcome.study_end_time_step,
                in_study_repair=outcome.in_study_repair,
            )
        )
    return result


def _assignments(path: Path) -> dict[tuple[DatasetSplit, str], str]:
    with Path(path).open(encoding="utf-8", newline="") as source:
        rows = tuple(csv.DictReader(source))
    return {
        (DatasetSplit(row["split"]), row["vehicle_id"]): row["node_id"]
        for row in rows
    }


