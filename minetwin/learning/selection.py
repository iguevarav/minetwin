import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.data import CachedSplit
from minetwin.learning.metrics import classification_metrics
from minetwin.learning.scaling import FederatedStandardScaler
from minetwin.learning.training import fit_centralized_candidate


def _default_candidates() -> tuple[TrainingConfig, ...]:
    return (
        TrainingConfig(hidden_sizes=(64,), epochs=10, learning_rate=0.001),
        TrainingConfig(hidden_sizes=(128, 64), epochs=10, learning_rate=0.001),
        TrainingConfig(hidden_sizes=(64,), epochs=10, learning_rate=0.0005),
        TrainingConfig(hidden_sizes=(128, 64), epochs=10, learning_rate=0.0005),
    )


@dataclass(frozen=True)
class SelectionConfig:
    folds: int = 5
    seed: int = 42
    candidates: tuple[TrainingConfig, ...] = field(
        default_factory=_default_candidates
    )

    def __post_init__(self) -> None:
        if type(self.folds) is not int or self.folds < 2:
            raise ValueError("folds debe ser un entero mayor que uno.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed debe ser un entero no negativo.")
        if not self.candidates:
            raise ValueError("Se requiere al menos un candidato.")


def run_model_selection(
    cache: Path,
    output: Path,
    config: SelectionConfig | None = None,
) -> dict:
    effective = config or SelectionConfig()
    cache = Path(cache)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "cache": str(cache.resolve()),
        "config": {
            "folds": effective.folds,
            "seed": effective.seed,
            "candidates": [asdict(candidate) for candidate in effective.candidates],
        },
        "selection_rule": (
            "minimum mean validation cost; ties use higher macro F1, "
            "higher balanced accuracy and fewer parameters"
        ),
    }
    _write_json(output / "manifest.json", manifest)
    try:
        train = CachedSplit.load(cache / "train.npz")
        folds = grouped_vehicle_folds(
            train.vehicle_ids,
            train.labels,
            train.nodes,
            effective.folds,
            effective.seed,
        )
        assignments = _fold_assignments(train, folds)
        fold_rows = []
        history_rows = []
        for candidate_index, candidate in enumerate(effective.candidates, 1):
            candidate_id = f"MLP-{candidate_index:02d}"
            for fold in range(effective.folds):
                training = folds != fold
                validation = folds == fold
                scaler = FederatedStandardScaler.fit(
                    train.features[training], train.nodes[training]
                )
                probabilities, history = fit_centralized_candidate(
                    scaler.transform(train.features[training]),
                    train.labels[training],
                    scaler.transform(train.features[validation]),
                    replace(candidate, seed=effective.seed + fold),
                )
                metrics = classification_metrics(
                    train.labels[validation], probabilities
                )
                fold_rows.append(
                    {
                        "candidate": candidate_id,
                        "fold": fold + 1,
                        "vehicles": len(set(train.vehicle_ids[validation])),
                        **_candidate_fields(candidate, train.features.shape[1]),
                        **_scalar_metrics(metrics),
                    }
                )
                history_rows.extend(
                    {
                        "candidate": candidate_id,
                        "fold": fold + 1,
                        **point,
                    }
                    for point in history
                )
        candidates = _candidate_summary(fold_rows)
        selected = candidates[0]
        selection = {
            "selected_candidate": selected["candidate"],
            "selection_rule": manifest["selection_rule"],
            "best_config": asdict(
                effective.candidates[int(selected["candidate"].split("-")[1]) - 1]
            ),
            "cross_validation": {
                "folds": effective.folds,
                "group": "vehicle_id",
                "stratification": "node_id and maximum observed risk class",
                "train_validation_vehicle_overlap": 0,
            },
            "best_metrics": selected,
        }
        _write_csv(output / "fold_assignments.csv", assignments)
        _write_csv(output / "fold_metrics.csv", fold_rows)
        _write_csv(output / "training_history.csv", history_rows)
        _write_csv(output / "candidates.csv", candidates)
        _write_json(output / "eda.json", describe_cache(cache))
        _write_json(output / "selection.json", selection)
        manifest.update(
            status="complete",
            counts={
                "candidates": len(effective.candidates),
                "folds": effective.folds,
                "fits": len(fold_rows),
                "vehicles": len(assignments),
            },
        )
        _write_json(output / "manifest.json", manifest)
        return selection
    except BaseException:
        manifest["status"] = "incomplete"
        _write_json(output / "manifest.json", manifest)
        raise


def grouped_vehicle_folds(
    vehicle_ids: np.ndarray,
    labels: np.ndarray,
    nodes: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    vehicle_ids = np.asarray(vehicle_ids)
    labels = np.asarray(labels)
    nodes = np.asarray(nodes)
    if not len(vehicle_ids) or not (
        len(vehicle_ids) == len(labels) == len(nodes)
    ):
        raise ValueError("Los vectores de ejemplos no coinciden.")
    vehicles = np.unique(vehicle_ids)
    if len(vehicles) < folds:
        raise ValueError("No hay suficientes vehículos para los folds solicitados.")
    strata: dict[tuple[str, int], list[str]] = {}
    for vehicle in vehicles:
        selected = vehicle_ids == vehicle
        vehicle_nodes = np.unique(nodes[selected])
        if len(vehicle_nodes) != 1:
            raise ValueError("Cada vehículo debe pertenecer a un único nodo.")
        key = str(vehicle_nodes[0]), int(np.max(labels[selected]))
        strata.setdefault(key, []).append(str(vehicle))
    assignment = {}
    for key, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda vehicle: _digest(seed, key, vehicle),
        )
        for position, vehicle in enumerate(ordered):
            assignment[vehicle] = position % folds
    return np.asarray([assignment[str(vehicle)] for vehicle in vehicle_ids])


def describe_cache(cache: Path) -> dict:
    cache = Path(cache)
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    splits = []
    for name in ("train", "validation", "test"):
        cached = CachedSplit.load(cache / f"{name}.npz")
        splits.append(
            {
                "split": name,
                "examples": len(cached.labels),
                "vehicles": len(np.unique(cached.vehicle_ids)),
                "nodes": _counts(cached.nodes),
                "classes": _counts(cached.labels) if np.all(cached.labels >= 0) else {},
                "node_classes": _node_classes(cached),
            }
        )
    train = CachedSplit.load(cache / "train.npz")
    source_summary = _source_summary(metadata)
    return {
        "feature_count": len(train.feature_names),
        "feature_config": metadata["feature_config"],
        "sampling": metadata["sampling"],
        "splits": splits,
        "raw_missing_rate": source_summary,
        "top_class_separation_features": _class_separation(train, 12),
        "class_meanings": {
            "0": "más de 48 pasos hasta reparación o sin reparación",
            "1": "entre 24 y 48 pasos",
            "2": "entre 12 y 24 pasos",
            "3": "entre 6 y 12 pasos",
            "4": "hasta 6 pasos",
        },
    }


def load_selected_training(path: Path) -> TrainingConfig:
    payload = json.loads((Path(path) / "selection.json").read_text(encoding="utf-8"))
    config = payload["best_config"]
    config["hidden_sizes"] = tuple(config["hidden_sizes"])
    return TrainingConfig(**config)


def _candidate_summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for candidate in sorted({row["candidate"] for row in rows}):
        selected = [row for row in rows if row["candidate"] == candidate]
        first = selected[0]
        summary = {
            key: first[key]
            for key in (
                "candidate",
                "hidden_sizes",
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "parameters",
            )
        }
        for metric in (
            "mean_cost",
            "macro_f1",
            "balanced_accuracy",
            "macro_pr_auc",
        ):
            values = np.asarray([row[metric] for row in selected])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1))
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row["mean_cost_mean"],
            -row["macro_f1_mean"],
            -row["balanced_accuracy_mean"],
            row["parameters"],
        )
    )
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
        row["selected"] = rank == 1
    return summaries


def _fold_assignments(cached: CachedSplit, folds: np.ndarray) -> list[dict]:
    rows = []
    for vehicle in np.unique(cached.vehicle_ids):
        selected = cached.vehicle_ids == vehicle
        rows.append(
            {
                "vehicle_id": vehicle,
                "node_id": cached.nodes[selected][0],
                "maximum_class": int(np.max(cached.labels[selected])),
                "fold": int(folds[selected][0]) + 1,
            }
        )
    return rows


def _candidate_fields(config: TrainingConfig, input_size: int) -> dict:
    return {
        "hidden_sizes": "x".join(map(str, config.hidden_sizes)),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "parameters": _parameter_count(input_size, config.hidden_sizes, 5),
    }


def _parameter_count(input_size: int, hidden: tuple[int, ...], output: int) -> int:
    sizes = (input_size, *hidden, output)
    return sum(
        (source + 1) * target
        for source, target in pairwise(sizes)
    )


def _scalar_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=True)}


def _node_classes(cached: CachedSplit) -> list[dict]:
    if np.any(cached.labels < 0):
        return []
    return [
        {
            "node": str(node),
            "class": label,
            "examples": int(np.sum((cached.nodes == node) & (cached.labels == label))),
        }
        for node in np.unique(cached.nodes)
        for label in range(5)
    ]


def _source_summary(metadata: dict) -> dict:
    manifest = Path(metadata["source_manifest"])
    summary = manifest.with_name("summary.json")
    if not summary.is_file():
        return {}
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return {
        row["split"]: row["missing_values"] / (row["readouts"] * row["features"])
        for row in payload["splits"]
    }


def _class_separation(cached: CachedSplit, limit: int) -> list[dict]:
    features = cached.features.astype(np.float64)
    overall = np.mean(features, axis=0)
    total = np.sum((features - overall) ** 2, axis=0)
    between = np.zeros(features.shape[1], dtype=np.float64)
    for label in range(5):
        selected = cached.labels == label
        if np.any(selected):
            between += np.sum(selected) * (np.mean(features[selected], axis=0) - overall) ** 2
    score = np.divide(
        between,
        total,
        out=np.zeros_like(between),
        where=total > 0,
    )
    indexes = np.argsort(-score, kind="stable")[:limit]
    return [
        {"feature": cached.feature_names[index], "eta_squared": float(score[index])}
        for index in indexes
    ]


def _digest(seed: int, key: tuple[str, int], vehicle: str) -> str:
    return hashlib.sha256(f"{seed}:{key[0]}:{key[1]}:{vehicle}".encode()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
