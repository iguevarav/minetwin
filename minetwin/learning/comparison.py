import csv
import json
from dataclasses import asdict
from math import ceil
from pathlib import Path

import numpy as np

from minetwin.learning.candidates import candidate_probabilities
from minetwin.learning.data import CachedSplit
from minetwin.learning.metrics import class_average_precision, classification_metrics
from minetwin.learning.priority import priority_metrics
from minetwin.learning.scaling import FederatedStandardScaler
from minetwin.learning.selection import (
    grouped_vehicle_folds,
    latest_vehicle_mask,
    load_selected_training,
    vehicle_fold_assignments,
)


def run_representation_comparison(
    baseline_cache: Path,
    usage_cache: Path,
    baseline_selection: Path,
    output: Path,
    folds: int = 5,
    seed: int = 42,
    review_fraction: float = 0.01,
) -> dict:
    if type(folds) is not int or folds < 2:
        raise ValueError("folds debe ser un entero mayor que uno.")
    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction debe estar entre cero y uno.")
    baseline = CachedSplit.load(Path(baseline_cache) / "train.npz")
    usage = CachedSplit.load(Path(usage_cache) / "train.npz")
    _validate_representations(baseline, usage)
    assignments = grouped_vehicle_folds(
        baseline.vehicle_ids,
        baseline.labels,
        baseline.nodes,
        folds,
        seed,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        mlp = load_selected_training(baseline_selection)
        fold_rows = []
        prevalence_rows = []
        for fold in range(folds):
            training = assignments != fold
            evaluation = ~training
            evaluation &= latest_vehicle_mask(
                baseline.vehicle_ids,
                baseline.time_steps,
                evaluation,
            )
            prevalence_rows.append(_prevalence_row(baseline, evaluation, fold))
            for representation, cached in (("baseline", baseline), ("usage", usage)):
                scaler = FederatedStandardScaler.fit(
                    cached.features[training],
                    cached.nodes[training],
                )
                train_features = scaler.transform(cached.features[training])
                evaluation_features = scaler.transform(cached.features[evaluation])
                for model in (
                    "mlp",
                    "logistic_regression",
                    "hist_gradient_boosting",
                ):
                    values = candidate_probabilities(
                        model,
                        train_features,
                        cached.labels[training],
                        evaluation_features,
                        mlp,
                        seed + fold,
                    )
                    fold_rows.append(
                        _metrics_row(
                            representation,
                            model,
                            fold,
                            cached.labels[evaluation],
                            values,
                            review_fraction,
                        )
                    )
        summary = _summary(fold_rows)
        selected = summary[0]
        result = {
            "selected_representation": selected["representation"],
            "selected_model": selected["model"],
            "review_fraction": review_fraction,
            "folds": folds,
            "seed": seed,
            "selection_rule": (
                "higher class 4 average precision, higher recall at review "
                "fraction, then lower mean cost"
            ),
            "evaluation": "latest causal window per vehicle in held-out train fold",
            "validation_used": False,
            "calibration": "not applied; selected separately in the priority policy",
            "representations": {
                "baseline": _representation_details(baseline_cache, baseline),
                "usage": _representation_details(usage_cache, usage),
            },
            "mlp_config": asdict(mlp),
            "candidates": {
                "logistic_regression": {"epochs": 30},
                "hist_gradient_boosting": {
                    "max_depth": 6,
                    "max_iter": 300,
                    "early_stopping": True,
                },
            },
        }
        _write_csv(output / "comparison_folds.csv", fold_rows)
        _write_csv(output / "comparison_summary.csv", summary)
        _write_csv(
            output / "fold_assignments.csv",
            vehicle_fold_assignments(baseline, assignments),
        )
        _write_csv(output / "fold_prevalence.csv", prevalence_rows)
        _write_json(output / "selection.json", result)
        return result
    except (Exception, KeyboardInterrupt):
        if output.exists():
            for path in output.iterdir():
                path.unlink()
            output.rmdir()
        raise


def _representation_details(cache: Path, cached: CachedSplit) -> dict:
    return {
        "cache": str(Path(cache).resolve()),
        "features": int(cached.features.shape[1]),
    }


def _validate_representations(baseline: CachedSplit, usage: CachedSplit) -> None:
    if baseline.features.shape[0] != usage.features.shape[0]:
        raise ValueError(
            "Las representaciones no tienen el mismo número de ejemplos."
        )
    for attribute in ("labels", "nodes", "vehicle_ids", "time_steps"):
        if not np.array_equal(
            getattr(baseline, attribute),
            getattr(usage, attribute),
        ):
            raise ValueError(
                "Las representaciones no describen los mismos ejemplos."
            )
    if baseline.time_steps is None or not np.isfinite(baseline.time_steps).all():
        raise ValueError("La comparación requiere pasos temporales en la caché.")


def _metrics_row(
    representation: str,
    model: str,
    fold: int,
    labels: np.ndarray,
    probabilities: np.ndarray,
    review_fraction: float,
) -> dict:
    metrics = classification_metrics(labels, probabilities)
    capacity = max(1, ceil(len(labels) * review_fraction))
    priority = priority_metrics(labels, probabilities, capacity)
    return {
        "representation": representation,
        "model": model,
        "fold": fold + 1,
        "examples": len(labels),
        "class_4_prevalence": float(np.mean(labels == 4)),
        "class_4_average_precision": class_average_precision(
            labels,
            probabilities,
            4,
        ),
        "review_fraction": review_fraction,
        "review_capacity": capacity,
        "recall_at_review_fraction": priority["recall_at_capacity"],
        "precision_at_review_fraction": priority["precision_at_capacity"],
        "random_recall": priority["random_recall"],
        "mean_cost": metrics["mean_cost"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_pr_auc": metrics["macro_pr_auc"],
    }


def _prevalence_row(cached: CachedSplit, selected: np.ndarray, fold: int) -> dict:
    labels = cached.labels[selected]
    return {
        "fold": fold + 1,
        "examples": len(labels),
        **{
            f"class_{label}_prevalence": float(np.mean(labels == label))
            for label in range(5)
        },
    }


def _summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for representation, model in sorted(
        {(row["representation"], row["model"]) for row in rows}
    ):
        selected = [
            row
            for row in rows
            if (row["representation"], row["model"]) == (representation, model)
        ]
        summary = {
            "representation": representation,
            "model": model,
            "folds": len(selected),
        }
        for metric in (
            "class_4_average_precision",
            "recall_at_review_fraction",
            "precision_at_review_fraction",
            "mean_cost",
            "balanced_accuracy",
            "macro_pr_auc",
        ):
            values = np.asarray([row[metric] for row in selected])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1))
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            -row["class_4_average_precision_mean"],
            -row["recall_at_review_fraction_mean"],
            row["mean_cost_mean"],
        )
    )
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
        row["selected"] = rank == 1
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
