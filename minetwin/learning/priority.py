import numpy as np

from minetwin.learning.metrics import mean_misclassification_cost

DEFAULT_CAPACITIES = (20, 50, 100, 250)


def priority_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    capacity: int,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not len(labels) or probabilities.shape != (len(labels), 5):
        raise ValueError("Las probabilidades deben contener cinco clases por ejemplo.")
    if type(capacity) is not int or capacity < 1:
        raise ValueError("La capacidad debe ser un entero positivo.")
    reviewed = min(capacity, len(labels))
    selected = np.argsort(-probabilities[:, 4], kind="stable")[:reviewed]
    positives = int(np.sum(labels == 4))
    found = int(np.sum(labels[selected] == 4))
    prevalence = positives / len(labels)
    return {
        "capacity": capacity,
        "examples": len(labels),
        "class_4_examples": positives,
        "class_4_found": found,
        "recall_at_capacity": found / positives if positives else 0.0,
        "precision_at_capacity": found / reviewed if reviewed else 0.0,
        "random_recall": reviewed / len(labels),
        "random_precision": prevalence,
        "priority_lift": found / reviewed / prevalence if prevalence else 0.0,
        "queue_mean_cost": mean_misclassification_cost(
            labels[selected], probabilities[selected]
        ),
    }


def priority_baseline(
    labels: np.ndarray,
    probabilities: np.ndarray,
    vehicle_ids: np.ndarray,
    nodes: np.ndarray,
    time_steps: np.ndarray,
    capacities: tuple[int, ...] = DEFAULT_CAPACITIES,
) -> tuple[list[dict], list[dict]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    vehicle_ids = np.asarray(vehicle_ids)
    nodes = np.asarray(nodes)
    time_steps = np.asarray(time_steps, dtype=np.float64)
    _validate_inputs(
        labels,
        probabilities,
        vehicle_ids,
        nodes,
        time_steps,
        capacities,
    )
    indexes = _latest_indexes(vehicle_ids, time_steps)
    ordered = sorted(
        indexes,
        key=lambda index: (-probabilities[index, 4], str(vehicle_ids[index])),
    )
    rows = [
        {
            "rank": rank,
            "vehicle_id": str(vehicle_ids[index]),
            "node_id": str(nodes[index]),
            "time_step": float(time_steps[index]),
            "label": int(labels[index]),
            **{
                f"probability_{label}": float(probabilities[index, label])
                for label in range(5)
            },
        }
        for rank, index in enumerate(ordered, 1)
    ]
    return rows, [
        _baseline_metrics(rows, capacity)
        for capacity in capacities
    ]


def _latest_indexes(vehicle_ids: np.ndarray, time_steps: np.ndarray) -> list[int]:
    latest: dict[str, int] = {}
    for index, vehicle_id in enumerate(vehicle_ids):
        key = str(vehicle_id)
        previous = latest.get(key)
        if previous is None or time_steps[index] > time_steps[previous]:
            latest[key] = index
    return list(latest.values())


def _baseline_metrics(rows: list[dict], capacity: int) -> dict:
    metrics = priority_metrics(
        np.asarray([row["label"] for row in rows]),
        np.asarray(
            [[row[f"probability_{label}"] for label in range(5)] for row in rows]
        ),
        capacity,
    )
    return {key: value for key, value in metrics.items() if key != "examples"} | {
        "vehicles": metrics["examples"]
    }


def _validate_inputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
    vehicle_ids: np.ndarray,
    nodes: np.ndarray,
    time_steps: np.ndarray,
    capacities: tuple[int, ...],
) -> None:
    size = len(labels)
    if not size or probabilities.shape != (size, 5):
        raise ValueError("Las probabilidades deben contener cinco clases por ejemplo.")
    if any(len(values) != size for values in (vehicle_ids, nodes, time_steps)):
        raise ValueError("Los metadatos de prioridad no coinciden.")
    if not np.isfinite(time_steps).all():
        raise ValueError("La línea base requiere pasos temporales de la caché.")
    if any(type(capacity) is not int or capacity < 1 for capacity in capacities):
        raise ValueError("Las capacidades deben ser enteros positivos.")
