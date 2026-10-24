import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from minetwin.data.scania import (
    SCANIA_CLASS_DESCRIPTIONS,
    CausalImputer,
    DatasetSplit,
    ScaniaReplayStore,
)
from minetwin.learning.data import WindowVectorizer
from minetwin.paths import DATASET_ROOT, result_path

CASES_PATH = result_path("11_mejora", "07_demostracion", "demo_cases.csv")
REPLAY_INDEX = result_path("02_reproduccion", "index.json")
POLICY_ROOT = result_path("11_mejora", "05_politica_prioridad_v2")
SELECTION_PATH = result_path("11_mejora", "04_comparacion_v2", "selection.json")
FEDERATION_PATH = result_path(
    "11_mejora", "06_evaluacion_final_v2", "federation", "summary.json"
)
SENSOR_CODES = (("A", "Contador A", "171_0"), ("B", "Contador B", "666_0"), ("C", "Contador C", "427_0"))
STATISTIC_NAMES = {
    "last": "\u00daltimo",
    "mean": "Promedio",
    "std": "Variaci\u00f3n",
    "trend": "Tendencia",
    "missing_rate": "Faltantes",
}


def construir_payload_demo(ids: list[str] | None = None) -> dict:
    cases = _read_csv(CASES_PATH)
    queue = _read_csv(POLICY_ROOT / "oof_priority_queue.csv")
    policy = _read_json(POLICY_ROOT / "policy.json")
    selection = _read_json(SELECTION_PATH)
    store = ScaniaReplayStore(DATASET_ROOT, REPLAY_INDEX)
    selected_cases = _select_cases(cases, queue, store, ids)
    return _native(
        {
            "version": 2,
            "modo": "real",
            "fuente": "SCANIA Component X \u00b7 camiones de entrenamiento evaluados por grupos",
            "modelo": _model(selection),
            "clases": list(SCANIA_CLASS_DESCRIPTIONS),
            "sensores": [
                {"clave": key, "nombre": name, "codigo": code}
                for key, name, code in (*SENSOR_CODES, ("perfil", "Perfil de carga", "397"))
            ],
            "estadisticas": [
                {"clave": name, "nombre": STATISTIC_NAMES[name]}
                for name in WindowVectorizer.STATISTICS
            ],
            "capacidad": int(policy["selected_capacity"]),
            "grupos": int(selection["folds"]),
            "camiones": [
                _truck_payload(
                    store,
                    row,
                    queue,
                    int(policy["selected_capacity"]),
                    case,
                )
                for case, row in selected_cases
            ],
            "flota": _fleet(queue),
            "federacion": _federation(),
        }
    )


def validar_payload(payload: dict) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["El payload no es un objeto."]
    capacity = payload.get("capacidad")
    if type(capacity) is not int or capacity < 1:
        errors.append("capacidad invalida")
    sensors = payload.get("sensores")
    if not isinstance(sensors, list) or not any(
        sensor.get("clave") == "perfil" for sensor in sensors
    ):
        errors.append("sensores invalidos")
    keys = [sensor["clave"] for sensor in sensors or [] if sensor.get("clave") != "perfil"]
    if not isinstance(payload.get("estadisticas"), list) or not payload["estadisticas"]:
        errors.append("estadisticas invalidas")
    if not isinstance(payload.get("modelo", {}).get("capas"), list):
        errors.append("modelo invalido")
    if not isinstance(payload.get("clases"), list) or len(payload["clases"]) != 5:
        errors.append("clases invalidas")
    fleet = payload.get("flota")
    if not isinstance(fleet, dict) or not any(
        item.get("revisiones") == capacity for item in fleet.get("capacidades", [])
    ):
        errors.append("flota invalida")
    trucks = payload.get("camiones")
    if not isinstance(trucks, list) or not trucks:
        return [*errors, "camiones invalidos"]
    for index, truck in enumerate(trucks):
        prefix = f"camiones[{index}]"
        readings = truck.get("lecturas")
        if truck.get("id") is None:
            errors.append(f"{prefix}.id falta")
        if not isinstance(readings, list) or len(readings) < 2:
            errors.append(f"{prefix}.lecturas invalidas")
            continue
        for reading_index, reading in enumerate(readings):
            if not _number(reading.get("t")) or (
                reading_index and reading["t"] < readings[reading_index - 1]["t"]
            ):
                errors.append(f"{prefix}.lecturas invalidas")
            if not all(
                reading.get("valores", {}).get(key) is None
                or _number(reading["valores"].get(key))
                for key in keys
            ):
                errors.append(f"{prefix}.valores invalidos")
        window = truck.get("ventana", {})
        if not (0 <= window.get("inicio", -1) <= window.get("fin", -1) < len(readings)):
            errors.append(f"{prefix}.ventana invalida")
        if not all(key in truck.get("resumen", {}) for key in keys):
            errors.append(f"{prefix}.resumen invalido")
        probabilities = truck.get("probabilidades")
        if not (
            isinstance(probabilities, list)
            and len(probabilities) == 5
            and all(_number(value) for value in probabilities)
            and math.isclose(sum(probabilities), 1.0, abs_tol=0.01)
        ):
            errors.append(f"{prefix}.probabilidades invalidas")
        total = truck.get("total_grupo")
        rank = truck.get("puesto")
        if type(total) is not int or total < 1 or type(rank) is not int or not 1 <= rank <= total:
            errors.append(f"{prefix}.puesto invalido")
        if truck.get("prioritario") is not (rank <= capacity):
            errors.append(f"{prefix}.prioritario invalido")
        if not _number(truck.get("umbral")):
            errors.append(f"{prefix}.umbral invalido")
        row = truck.get("fila")
        if not isinstance(row, list) or len(row) < 10 or not all(_number(value) for value in row):
            errors.append(f"{prefix}.fila invalida")
    return errors


def _select_cases(
    rows: list[dict[str, str]],
    queue: list[dict[str, str]],
    store: ScaniaReplayStore,
    ids: list[str] | None,
) -> list[tuple[str, dict[str, str]]]:
    eligible = [
        row
        for row in rows
        if int(row["window_observations"]) == 12
        and _readouts_until_cut(store, row) >= 20
    ]
    if ids is not None:
        by_vehicle = {row["vehicle_id"]: row for row in eligible}
        missing = [vehicle_id for vehicle_id in ids if vehicle_id not in by_vehicle]
        if missing:
            raise ValueError(f"No hay casos configurados para: {', '.join(missing)}.")
        names = ("Requiere revisi\u00f3n", "Punto medio", "No requiere")
        return list(zip(names, (by_vehicle[vehicle_id] for vehicle_id in ids), strict=True))
    totals = _fold_totals(queue)
    return [
        (
            "Requiere revisi\u00f3n",
            _select(
                eligible,
                queue,
                store,
                lambda row: row["selected"] == "True",
                lambda row: (int(row["rank"]), int(row["vehicle_id"])),
            ),
        ),
        (
            "Punto medio",
            _select(
                eligible,
                queue,
                store,
                lambda row: 20 < int(row["rank"]) <= 250,
                lambda row: (abs(int(row["rank"]) - 50), int(row["vehicle_id"])),
            ),
        ),
        (
            "No requiere",
            _select(
                eligible,
                queue,
                store,
                lambda row: True,
                lambda row: (
                    abs(int(row["rank"]) - round(totals[row["fold"]] / 2)),
                    int(row["vehicle_id"]),
                ),
            ),
        ),
    ]


def _select(
    preferred: list[dict[str, str]],
    fallback: list[dict[str, str]],
    store: ScaniaReplayStore,
    predicate,
    key,
) -> dict[str, str]:
    candidates = [row for row in preferred if predicate(row)]
    if not candidates:
        candidates = [
            row
            for row in fallback
            if int(row["window_observations"]) == 12
            and predicate(row)
            and _readouts_until_cut(store, row) >= 20
        ]
    if not candidates:
        raise ValueError("No existe un caso de demostracion elegible.")
    return min(candidates, key=key)


def _truck_payload(
    store: ScaniaReplayStore,
    row: dict[str, str],
    queue: list[dict[str, str]],
    capacity: int,
    case: str,
) -> dict:
    vehicle_id = row["vehicle_id"]
    readouts = store.readouts(DatasetSplit.TRAIN, vehicle_id)
    cutoff = _cutoff_index(readouts, float(row["time_step"]))
    window_size = int(row["window_observations"])
    if cutoff + 1 < max(20, window_size):
        raise ValueError(f"El vehiculo {vehicle_id} no tiene historia suficiente.")
    shown_start = cutoff + 1 - max(20, window_size)
    shown = readouts[shown_start : cutoff + 1]
    window_start = cutoff - window_size + 1
    fold_rows = sorted(
        (item for item in queue if item["fold"] == row["fold"]),
        key=lambda item: int(item["rank"]),
    )
    if len(fold_rows) < capacity:
        raise ValueError("La cola no alcanza la capacidad publicada.")
    rank = int(row["rank"])
    probabilities = [float(row[f"probability_{label}"]) for label in range(5)]
    selected = rank <= capacity
    if selected != _boolean(row["selected"]):
        raise ValueError("La selección del caso no coincide con su puesto.")
    return {
        "id": vehicle_id,
        "nodo": row["node_id"].capitalize(),
        "caso": case,
        "grupo": int(row["fold"]),
        "sensores_totales": len(store.feature_names),
        "lecturas_totales": cutoff + 1,
        "lecturas": [
            _reading_payload(readouts, shown_start + offset)
            for offset in range(len(shown))
        ],
        "ventana": {
            "inicio": window_start - shown_start,
            "fin": len(shown) - 1,
        },
        "resumen": _summary(store, readouts, cutoff, window_size),
        "probabilidades": probabilities,
        "puesto": rank,
        "total_grupo": len(fold_rows),
        "prioritario": selected,
        "umbral": float(fold_rows[capacity - 1]["calibrated_probability_4"]),
        "fila": _queue_row(fold_rows),
    }


def _reading_payload(readouts: tuple, index: int) -> dict:
    reading = readouts[index]
    values = dict(zip(reading.feature_names, reading.values, strict=True))
    return {
        "n": index + 1,
        "t": reading.time_step,
        "valores": {key: values[code] for key, _, code in SENSOR_CODES},
        "perfil": _profile(readouts, index),
        "sin_dato": sum(value is None for value in reading.values),
    }


def _profile(readouts: tuple, index: int) -> list[float] | None:
    if index == 0:
        return None
    current = dict(zip(readouts[index].feature_names, readouts[index].values, strict=True))
    previous = dict(zip(readouts[index - 1].feature_names, readouts[index - 1].values, strict=True))
    names = tuple(f"397_{bin_index}" for bin_index in range(36))
    if any(current[name] is None or previous[name] is None for name in names):
        return None
    increments = np.asarray([current[name] - previous[name] for name in names])
    if np.any(increments < 0):
        return None
    totals = np.asarray([chunk.sum() for chunk in np.array_split(increments, 6)])
    total = float(totals.sum())
    if total <= 0:
        return None
    return [float(value / total) for value in totals]


def _summary(
    store: ScaniaReplayStore,
    readouts: tuple,
    cutoff: int,
    window_size: int,
) -> dict:
    imputer = CausalImputer()
    transformed = [imputer.transform(readout) for readout in readouts[: cutoff + 1]]
    window = transformed[-window_size:]
    values = np.asarray([item.values for item in window], dtype=float)
    missing = np.asarray([item.missing for item in window], dtype=float)
    elapsed = window[-1].source.time_step - window[0].source.time_step
    duration = elapsed if elapsed > 0 else 1.0
    statistics = {
        "last": values[-1],
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "trend": (values[-1] - values[0]) / duration,
        "missing_rate": missing.mean(axis=0),
    }
    indexes = {name: index for index, name in enumerate(store.feature_names)}
    return {
        key: {statistic: float(values[indexes[code]]) for statistic, values in statistics.items()}
        for key, _, code in SENSOR_CODES
    }


def _queue_row(rows: list[dict[str, str]]) -> list[float]:
    values = np.asarray([float(row["calibrated_probability_4"]) for row in rows])
    return [float(chunk.mean()) for chunk in np.array_split(values, 100)]


def _fleet(queue: list[dict[str, str]]) -> dict:
    summary = _read_csv(POLICY_ROOT / "policy_summary.csv")
    folds = _fold_totals(queue)
    capacities = []
    for row in summary:
        capacity = int(row["capacity"])
        precision = float(row["precision_at_capacity_mean"])
        lift = float(row["priority_lift_mean"])
        capacities.append(
            {
                "revisiones": capacity,
                "encontrados": round(precision * capacity, 2),
                "azar": round(precision / lift * capacity, 2) if lift else 0.0,
                "veces": round(lift, 2),
                "cobertura": float(row["recall_at_capacity_mean"]),
            }
        )
    return {
        "vehiculos_por_grupo": round(len(queue) / len(folds)),
        "urgentes_por_grupo": round(
            sum(row["label"] == "4" for row in queue) / len(folds)
        ),
        "promedio_de_grupos": len(folds),
        "capacidades": capacities,
    }


def _federation() -> dict:
    summary = _read_json(FEDERATION_PATH)
    return {
        "reportes_enviados": int(summary["transfer"]["raw_records"]),
        "mb_compartidos": round(summary["published_state_bytes"] / 1_000_000, 2),
        "mb_centralizado": round(
            summary["centralized_feature_baseline_bytes"] / 1_000_000,
            2,
        ),
        "nodos": [node.capitalize() for node in summary["nodes"]],
    }


def _model(selection: dict) -> dict:
    feature_count = selection["representations"][selection["selected_representation"]]["features"]
    return {
        "capas": [feature_count, *selection["mlp_config"]["hidden_sizes"], 5],
        "extra_ventana": 2,
    }


def _cutoff_index(readouts: tuple, time_step: float) -> int:
    for index, readout in enumerate(readouts):
        if math.isclose(readout.time_step, time_step, abs_tol=1e-9):
            return index
    raise ValueError(f"No existe el corte temporal {time_step:g}.")


def _fold_totals(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        fold: sum(row["fold"] == fold for row in rows)
        for fold in {row["fold"] for row in rows}
    }


def _readouts_until_cut(store: ScaniaReplayStore, row: dict[str, str]) -> int:
    readouts = store.readouts(DatasetSplit.TRAIN, row["vehicle_id"])
    return _cutoff_index(readouts, float(row["time_step"])) + 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Booleano invalido: {value}.")


def _number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _native(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_native(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
