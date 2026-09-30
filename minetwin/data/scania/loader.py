import csv
import math
from pathlib import Path

from minetwin.data.scania.contracts import (
    DatasetSplit,
    FeatureReadout,
    ScaniaSchema,
    SplitSummary,
    VehicleOutcome,
    VehicleSpecifications,
)

READOUT_FILES = {
    split: f"{split.value}_operational_readouts.csv" for split in DatasetSplit
}
SPECIFICATION_FILES = {
    split: f"{split.value}_specifications.csv" for split in DatasetSplit
}
OUTCOME_FILES = {
    DatasetSplit.TRAIN: "train_tte.csv",
    DatasetSplit.VALIDATION: "validation_labels.csv",
    DatasetSplit.TEST: "test_labels.csv",
}
MISSING_VALUES = frozenset(("", "na", "nan", "null", "none"))


class ScaniaDataset:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._schema: ScaniaSchema | None = None

    @property
    def schema(self) -> ScaniaSchema:
        if self._schema is None:
            self._schema = self._inspect_schema()
        return self._schema

    def iter_readouts(self, split: DatasetSplit):
        path = self.root / READOUT_FILES[split]
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            expected = ("vehicle_id", "time_step", *self.schema.feature_names)
            _require_header(reader.fieldnames, expected, path.name)
            for line, row in enumerate(reader, 2):
                _require_complete_row(row, path.name, line)
                try:
                    yield feature_readout_from_row(
                        split, row, self.schema.feature_names
                    )
                except ValueError as error:
                    raise ValueError(f"{path.name}, fila {line}: {error}") from error

    def load_specifications(
        self, split: DatasetSplit
    ) -> dict[str, VehicleSpecifications]:
        path = self.root / SPECIFICATION_FILES[split]
        rows: dict[str, VehicleSpecifications] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            expected = ("vehicle_id", *self.schema.specification_names)
            _require_header(reader.fieldnames, expected, path.name)
            for line, row in enumerate(reader, 2):
                _require_complete_row(row, path.name, line)
                vehicle_id = _identity(row["vehicle_id"])
                if vehicle_id in rows:
                    raise ValueError(f"{path.name}: vehicle_id duplicado: {vehicle_id}.")
                rows[vehicle_id] = VehicleSpecifications(
                    split=split,
                    vehicle_id=vehicle_id,
                    names=self.schema.specification_names,
                    values=tuple(
                        _identity(row[name]) for name in self.schema.specification_names
                    ),
                )
        return rows

    def load_outcomes(self, split: DatasetSplit) -> dict[str, VehicleOutcome]:
        path = self.root / OUTCOME_FILES[split]
        if split == DatasetSplit.TEST and not path.is_file():
            return {}
        expected = {
            DatasetSplit.TRAIN: (
                "vehicle_id",
                "length_of_study_time_step",
                "in_study_repair",
            ),
            DatasetSplit.VALIDATION: ("vehicle_id", "class_label"),
            DatasetSplit.TEST: ("vehicle_id", "class_label"),
        }[split]
        outcomes: dict[str, VehicleOutcome] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            _require_header(reader.fieldnames, expected, path.name)
            for line, row in enumerate(reader, 2):
                _require_complete_row(row, path.name, line)
                try:
                    vehicle_id = _identity(row["vehicle_id"])
                    if vehicle_id in outcomes:
                        raise ValueError(f"vehicle_id duplicado: {vehicle_id}")
                    outcomes[vehicle_id] = (
                        VehicleOutcome(
                            split=split,
                            vehicle_id=vehicle_id,
                            study_end_time_step=_required_number(
                                row["length_of_study_time_step"]
                            ),
                            in_study_repair=_binary(row["in_study_repair"]),
                        )
                        if split == DatasetSplit.TRAIN
                        else VehicleOutcome(
                            split=split,
                            vehicle_id=vehicle_id,
                            class_label=_class_label(row["class_label"]),
                        )
                    )
                except ValueError as error:
                    raise ValueError(f"{path.name}, fila {line}: {error}") from error
        return outcomes

    def validate(self) -> tuple[SplitSummary, ...]:
        split_vehicles: dict[DatasetSplit, set[str]] = {}
        summaries = []
        for split in DatasetSplit:
            specifications = self.load_specifications(split)
            outcomes = self.load_outcomes(split)
            if outcomes and set(outcomes) != set(specifications):
                raise ValueError(
                    f"Los resultados y especificaciones de {split.value} no coinciden."
                )
            vehicles, readouts, missing = self._validate_readouts(
                split, set(specifications)
            )
            split_vehicles[split] = vehicles
            summaries.append(
                SplitSummary(
                    split=split,
                    vehicles=len(vehicles),
                    readouts=readouts,
                    features=len(self.schema.feature_names),
                    missing_values=missing,
                    specification_rows=len(specifications),
                    outcome_rows=len(outcomes),
                )
            )
        for left_index, left in enumerate(DatasetSplit):
            for right in tuple(DatasetSplit)[left_index + 1 :]:
                overlap = split_vehicles[left] & split_vehicles[right]
                if overlap:
                    raise ValueError(
                        f"Los splits {left.value} y {right.value} comparten vehiculos."
                    )
        return tuple(summaries)

    def _inspect_schema(self) -> ScaniaSchema:
        readout_headers = {
            split: _header(self.root / filename)
            for split, filename in READOUT_FILES.items()
        }
        reference = readout_headers[DatasetSplit.TRAIN]
        if len(reference) < 3 or reference[:2] != ("vehicle_id", "time_step"):
            raise ValueError("El esquema operativo no inicia con vehicle_id,time_step.")
        if any(header != reference for header in readout_headers.values()):
            raise ValueError("Los splits no comparten el mismo esquema operativo.")
        specification_headers = {
            split: _header(self.root / filename)
            for split, filename in SPECIFICATION_FILES.items()
        }
        spec_reference = specification_headers[DatasetSplit.TRAIN]
        if len(spec_reference) < 2 or spec_reference[0] != "vehicle_id":
            raise ValueError("El esquema de especificaciones no inicia con vehicle_id.")
        if any(header != spec_reference for header in specification_headers.values()):
            raise ValueError("Los splits no comparten las mismas especificaciones.")
        return ScaniaSchema(reference[2:], spec_reference[1:])

    def _validate_readouts(
        self, split: DatasetSplit, expected_vehicles: set[str]
    ) -> tuple[set[str], int, int]:
        vehicles: set[str] = set()
        completed: set[str] = set()
        current: str | None = None
        previous_time: float | None = None
        readouts = 0
        missing = 0
        for readout in self.iter_readouts(split):
            if readout.vehicle_id not in expected_vehicles:
                raise ValueError(
                    f"{split.value}: faltan especificaciones para {readout.vehicle_id}."
                )
            if readout.vehicle_id != current:
                if readout.vehicle_id in completed:
                    raise ValueError(
                        f"{split.value}: los readouts de {readout.vehicle_id} no son contiguos."
                    )
                if current is not None:
                    completed.add(current)
                current = readout.vehicle_id
                previous_time = None
                vehicles.add(current)
            if previous_time is not None and readout.time_step <= previous_time:
                raise ValueError(
                    f"{split.value}: time_step no crece para {readout.vehicle_id}."
                )
            previous_time = readout.time_step
            readouts += 1
            missing += sum(value is None for value in readout.values)
        if not readouts:
            raise ValueError(f"{split.value}: no contiene readouts.")
        if vehicles != expected_vehicles:
            raise ValueError(
                f"Los readouts y especificaciones de {split.value} no coinciden."
            )
        return vehicles, readouts, missing


def _header(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}.")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)
    if not header or len(header) != len(set(header)) or any(not name for name in header):
        raise ValueError(f"Cabecera invalida en {path.name}.")
    return tuple(header)


def feature_readout_from_row(
    split: DatasetSplit,
    row: dict[str, str],
    feature_names: tuple[str, ...],
) -> FeatureReadout:
    expected = {"vehicle_id", "time_step", *feature_names}
    if set(row) != expected or any(value is None for value in row.values()):
        raise ValueError("Fila operativa incompleta o con esquema desconocido.")
    return FeatureReadout(
        split=split,
        vehicle_id=_identity(row["vehicle_id"]),
        time_step=_required_number(row["time_step"]),
        feature_names=feature_names,
        values=tuple(_optional_number(row[name]) for name in feature_names),
    )


def _require_header(
    actual: list[str] | None, expected: tuple[str, ...], filename: str
) -> None:
    if tuple(actual or ()) != expected:
        raise ValueError(f"Cabecera inesperada en {filename}.")


def _require_complete_row(row: dict, filename: str, line: int) -> None:
    if None in row or any(value is None for value in row.values()):
        raise ValueError(f"Fila incompleta en {filename}, linea {line}.")


def _identity(value: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError("Un identificador esta vacio.")
    return result


def _required_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Se esperaba un numero finito no negativo.")
    return number


def _optional_number(value: str) -> float | None:
    if value.strip().lower() in MISSING_VALUES:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Las variables deben ser finitas.")
    return number


def _binary(value: str) -> bool:
    if value not in ("0", "1"):
        raise ValueError("in_study_repair debe ser 0 o 1.")
    return value == "1"


def _class_label(value: str) -> int:
    label = int(value)
    if str(label) != value.strip() or label not in range(5):
        raise ValueError("class_label debe ser un entero entre 0 y 4.")
    return label
