import json
import shutil
from pathlib import Path

import numpy as np

from minetwin.learning.data import CachedSplit
from minetwin.learning.inference import TorchRiskPrognosticator
from minetwin.learning.metrics import (
    classification_metrics,
    grouped_classification_metrics,
)
from minetwin.learning.provenance import file_sha256

REGIMES = ("local", "centralized", "fedavg", "fedprox")


def run_model_diagnostics(cache: Path, models: Path, output: Path) -> dict:
    cache = Path(cache)
    models = Path(models)
    train = CachedSplit.load(cache / "train.npz")
    validation = CachedSplit.load(cache / "validation.npz")
    if train.feature_names != validation.feature_names:
        raise ValueError("Entrenamiento y validación no comparten variables.")
    training = json.loads((models / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if training["features"] != len(validation.feature_names):
        raise ValueError("El modelo y la caché no comparten variables.")
    if (
        training["training_examples"] != len(train.labels)
        or training["validation_examples"] != len(validation.labels)
    ):
        raise ValueError("El modelo y la caché no comparten tamaños de muestra.")
    counts = np.bincount(train.labels, minlength=5)
    prior = counts / counts.sum()
    nodes = sorted(str(node) for node in np.unique(validation.nodes))
    model_paths = [models / f"local_{node}.pt" for node in nodes]
    model_paths.extend(models / f"{regime}.pt" for regime in REGIMES[1:])
    baselines = {
        "always_0": _constant_baseline(validation, 0),
        "always_4": _constant_baseline(validation, 4),
        "train_prior": grouped_classification_metrics(
            validation.labels,
            np.broadcast_to(prior, (len(validation.labels), 5)),
            validation.nodes,
        ),
    }
    results = {}
    argmax_results = {}
    model_ids = {}
    for regime in REGIMES:
        probabilities, ids = _probabilities(validation, models, regime)
        results[regime] = grouped_classification_metrics(
            validation.labels, probabilities, validation.nodes
        )
        argmax_results[regime] = classification_metrics(
            validation.labels, probabilities, decision="argmax"
        )
        model_ids[regime] = ids
    report = {
        "split": "validation",
        "role": "diagnostic_only_no_model_selection",
        "training_class_counts": counts.tolist(),
        "validation_class_counts": np.bincount(
            validation.labels, minlength=5
        ).tolist(),
        "nodes": nodes,
        "feature_count": len(train.feature_names),
        "feature_config": metadata["feature_config"],
        "sampling": metadata["sampling"],
        "training_config": training["config"],
        "torch": training["torch"],
        "model_ids": model_ids,
        "sha256": {
            "train_cache": file_sha256(cache / "train.npz"),
            "validation_cache": file_sha256(cache / "validation.npz"),
            "scaler": file_sha256(models / "scaler.npz"),
            "cache_metadata": file_sha256(cache / "metadata.json"),
            "models": {path.name: file_sha256(path) for path in model_paths},
        },
        "baselines": baselines,
        "results": results,
        "argmax_results": argmax_results,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "diagnostics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise
    return {
        "split": report["split"],
        "examples": len(validation.labels),
        "regimes": list(results),
        "output": str(output),
    }


def _constant_baseline(cached: CachedSplit, label: int) -> dict:
    probabilities = np.broadcast_to(np.eye(5)[label], (len(cached.labels), 5))
    return grouped_classification_metrics(
        cached.labels, probabilities, cached.nodes
    )


def _probabilities(
    cached: CachedSplit, models: Path, regime: str
) -> tuple[np.ndarray, dict[str, str]]:
    if regime != "local":
        predictor = TorchRiskPrognosticator.load(
            models / f"{regime}.pt", models / "scaler.npz"
        )
        return predictor.predict_probabilities(cached.features), {
            "global": predictor.model_id
        }
    probabilities = np.empty((len(cached.labels), 5), dtype=np.float32)
    ids = {}
    for node in np.unique(cached.nodes):
        selected = cached.nodes == node
        predictor = TorchRiskPrognosticator.load(
            models / f"local_{node}.pt", models / "scaler.npz"
        )
        probabilities[selected] = predictor.predict_probabilities(
            cached.features[selected]
        )
        ids[str(node)] = predictor.model_id
    return probabilities, ids


