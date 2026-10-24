import csv
import json
import shutil
from pathlib import Path

import numpy as np

from minetwin.learning.calibration import (
    TemperatureCalibrator,
    fit_temperature_calibrator,
)
from minetwin.learning.candidates import candidate_probabilities
from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.data import CachedSplit
from minetwin.learning.metrics import (
    classification_metrics,
    expected_decision_costs,
)
from minetwin.learning.priority import DEFAULT_CAPACITIES, priority_metrics
from minetwin.learning.scaling import FederatedStandardScaler
from minetwin.learning.selection import grouped_vehicle_folds, latest_vehicle_mask

CALIBRATION_FOLDS = 5


def run_priority_policy(cache: Path, comparison: Path, output: Path) -> dict:
    selection = _selection(comparison)
    cached = CachedSplit.load(Path(cache) / "train.npz")
    _validate_cache(cached, selection)
    assignments = grouped_vehicle_folds(
        cached.vehicle_ids,
        cached.labels,
        cached.nodes,
        selection["folds"],
        selection["seed"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        policy_rows, calibration_rows, queue_rows, random_rows = _evaluate_folds(
            cached,
            assignments,
            selection,
        )
        summary = _policy_summary(policy_rows)
        selected = summary[0]
        _mark_selected(queue_rows, selected["capacity"])
        result = {
            "selected_capacity": selected["capacity"],
            "selected_representation": selection["selected_representation"],
            "selected_model": selection["selected_model"],
            "priority_score": "calibrated_probability_4",
            "calibration": {
                "method": "multiclass_temperature_scaling",
                "folds": CALIBRATION_FOLDS,
                "source": "outer training fold only",
            },
            "selection_rule": (
                "higher mean precision at capacity, higher mean recall at capacity, "
                "lower queue mean cost, then lower capacity"
            ),
            "evaluation": "latest causal window per vehicle in held-out train fold",
            "validation_used": False,
            "model_config": selection["mlp_config"],
            "comparison": str(Path(comparison).resolve()),
            "cache": str(Path(cache).resolve()),
        }
        _write_csv(output / "policy_folds.csv", policy_rows)
        _write_csv(output / "policy_summary.csv", summary)
        _write_csv(output / "calibration_folds.csv", calibration_rows)
        _write_csv(output / "random_baseline.csv", random_rows)
        _write_csv(output / "oof_priority_queue.csv", queue_rows)
        _write_json(output / "policy.json", result)
        return result
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _selection(comparison: Path) -> dict:
    path = Path(comparison) / "selection.json"
    if not path.is_file():
        raise ValueError("No existe la selección de representación.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "selected_representation",
        "selected_model",
        "folds",
        "seed",
        "representations",
        "mlp_config",
    }
    if missing := required.difference(selection):
        raise ValueError(f"La selección no contiene: {', '.join(sorted(missing))}.")
    return selection


def _validate_cache(cached: CachedSplit, selection: dict) -> None:
    if selection["selected_model"] not in {
        "mlp",
        "logistic_regression",
        "hist_gradient_boosting",
    }:
        raise ValueError("El modelo seleccionado no es compatible con la política.")
    selected = selection["representations"][selection["selected_representation"]]
    if cached.features.shape[1] != selected["features"]:
        raise ValueError("La caché no coincide con la representación seleccionada.")
    if cached.time_steps is None or not np.isfinite(cached.time_steps).all():
        raise ValueError("La política requiere pasos temporales en la caché.")


def _evaluate_folds(
    cached: CachedSplit,
    assignments: np.ndarray,
    selection: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    policy_rows = []
    calibration_rows = []
    queue_rows = []
    random_rows = []
    model = selection["selected_model"]
    config = _training_config(selection["mlp_config"])
    for fold in range(selection["folds"]):
        training = assignments != fold
        calibration_vehicles = _calibration_mask(
            cached,
            training,
            selection["seed"] + fold,
        )
        fitting = training & ~calibration_vehicles
        calibration = calibration_vehicles & latest_vehicle_mask(
            cached.vehicle_ids,
            cached.time_steps,
            calibration_vehicles,
        )
        evaluation = ~training
        evaluation &= latest_vehicle_mask(
            cached.vehicle_ids,
            cached.time_steps,
            evaluation,
        )
        scaler = FederatedStandardScaler.fit(
            cached.features[fitting],
            cached.nodes[fitting],
        )
        calibration_features = scaler.transform(cached.features[calibration])
        evaluation_features = scaler.transform(cached.features[evaluation])
        raw_probabilities = candidate_probabilities(
            model,
            scaler.transform(cached.features[fitting]),
            cached.labels[fitting],
            np.concatenate((calibration_features, evaluation_features)),
            config,
            selection["seed"] + fold,
        )
        calibration_size = len(calibration_features)
        raw_calibration = raw_probabilities[:calibration_size]
        raw_evaluation = raw_probabilities[calibration_size:]
        calibrator = fit_temperature_calibrator(
            raw_calibration,
            cached.labels[calibration],
        )
        calibrated = calibrator.transform(raw_evaluation)
        calibration_rows.append(
            _calibration_row(
                fold,
                calibrator,
                cached.labels[calibration],
                raw_calibration,
            )
        )
        labels = cached.labels[evaluation]
        metrics = classification_metrics(labels, calibrated)
        for capacity in DEFAULT_CAPACITIES:
            policy = priority_metrics(labels, calibrated, capacity)
            policy_rows.append(
                {
                    "fold": fold + 1,
                    **policy,
                    "mean_cost": metrics["mean_cost"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )
            random_rows.append(_random_row(fold, capacity, policy, metrics))
        queue_rows.extend(_queue_rows(cached, evaluation, calibrated, fold))
    return policy_rows, calibration_rows, queue_rows, random_rows


def _training_config(value: dict) -> TrainingConfig:
    config = dict(value)
    config["hidden_sizes"] = tuple(config["hidden_sizes"])
    return TrainingConfig(**config)


def _calibration_mask(
    cached: CachedSplit,
    training: np.ndarray,
    seed: int,
) -> np.ndarray:
    assignments = grouped_vehicle_folds(
        cached.vehicle_ids[training],
        cached.labels[training],
        cached.nodes[training],
        CALIBRATION_FOLDS,
        seed,
    )
    mask = np.zeros(len(training), dtype=bool)
    mask[np.flatnonzero(training)[assignments == 0]] = True
    return mask


def _calibration_row(
    fold: int,
    calibrator: TemperatureCalibrator,
    labels: np.ndarray,
    raw: np.ndarray,
) -> dict:
    calibrated = calibrator.transform(raw)
    raw_metrics = classification_metrics(labels, raw)
    calibrated_metrics = classification_metrics(labels, calibrated)
    return {
        "fold": fold + 1,
        "examples": len(labels),
        "temperature": calibrator.temperature,
        "raw_log_loss": raw_metrics["log_loss"],
        "calibrated_log_loss": calibrated_metrics["log_loss"],
        "raw_brier_score": raw_metrics["brier_score"],
        "calibrated_brier_score": calibrated_metrics["brier_score"],
        "raw_argmax_ece": raw_metrics["argmax_ece"],
        "calibrated_argmax_ece": calibrated_metrics["argmax_ece"],
    }


def _random_row(fold: int, capacity: int, policy: dict, metrics: dict) -> dict:
    return {
        "fold": fold + 1,
        "capacity": capacity,
        "examples": policy["examples"],
        "class_4_examples": policy["class_4_examples"],
        "recall_at_capacity": policy["random_recall"],
        "precision_at_capacity": policy["random_precision"],
        "expected_queue_mean_cost": metrics["mean_cost"],
    }


def _queue_rows(
    cached: CachedSplit,
    evaluation: np.ndarray,
    probabilities: np.ndarray,
    fold: int,
) -> list[dict]:
    indexes = np.flatnonzero(evaluation)
    order = np.argsort(-probabilities[:, 4], kind="stable")
    costs = expected_decision_costs(probabilities)
    observations = _window_observations(cached, indexes)
    return [
        {
            "fold": fold + 1,
            "rank": rank,
            "vehicle_id": str(cached.vehicle_ids[indexes[position]]),
            "node_id": str(cached.nodes[indexes[position]]),
            "time_step": float(cached.time_steps[indexes[position]]),
            "window_observations": int(observations[position]),
            "label": int(cached.labels[indexes[position]]),
            "most_likely_class": int(np.argmax(probabilities[position])),
            "maximum_probability": float(np.max(probabilities[position])),
            "published_decision": int(np.argmin(costs[position])),
            "published_expected_cost": float(
                costs[position, np.argmin(costs[position])]
            ),
            "calibrated_probability_4": float(probabilities[position, 4]),
            **{
                f"probability_{label}": float(probabilities[position, label])
                for label in range(5)
            },
            **{
                f"expected_cost_{label}": float(costs[position, label])
                for label in range(5)
            },
            "selected": False,
        }
        for rank, position in enumerate(order, 1)
    ]


def _window_observations(cached: CachedSplit, indexes: np.ndarray) -> np.ndarray:
    try:
        position = cached.feature_names.index("window__observations")
    except ValueError as error:
        raise ValueError(
            "La caché no contiene el tamaño de la ventana causal."
        ) from error
    return np.rint(np.expm1(cached.features[indexes, position])).astype(int)


def _mark_selected(rows: list[dict], capacity: int) -> None:
    for row in rows:
        row["selected"] = row["rank"] <= capacity


def _policy_summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for capacity in DEFAULT_CAPACITIES:
        selected = [row for row in rows if row["capacity"] == capacity]
        summary = {"capacity": capacity, "folds": len(selected)}
        for metric in (
            "recall_at_capacity",
            "precision_at_capacity",
            "priority_lift",
            "queue_mean_cost",
            "mean_cost",
            "balanced_accuracy",
        ):
            values = np.asarray([row[metric] for row in selected])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1))
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            -row["precision_at_capacity_mean"],
            -row["recall_at_capacity_mean"],
            row["queue_mean_cost_mean"],
            row["capacity"],
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
