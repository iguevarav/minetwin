import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.data import CachedSplit
from minetwin.learning.metrics import classification_metrics
from minetwin.learning.scaling import FederatedStandardScaler
from minetwin.learning.selection import grouped_vehicle_folds
from minetwin.learning.training import fit_centralized_candidate, train_learning_regimes
from minetwin.stats import bootstrap_mean_interval, holm_adjust, wilcoxon_signed_rank


@dataclass(frozen=True)
class DecisionVariant:
    name: str
    weighted_loss: bool
    decision: str
    prior_correction: bool


VARIANTS = (
    DecisionVariant("a_weighted_cost", True, "cost", False),
    DecisionVariant("b_unweighted_cost", False, "cost", False),
    DecisionVariant("c_weighted_argmax", True, "argmax", False),
    DecisionVariant("d_unweighted_cost_prior", False, "cost", True),
)
METRICS = ("mean_cost", "macro_pr_auc", "balanced_accuracy", "accuracy")
COMPARISONS = (
    ("local", "fedavg"),
    ("local", "fedprox"),
    ("centralized", "fedavg"),
    ("centralized", "fedprox"),
    ("fedavg", "fedprox"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico y corrección SCANIA.")
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv")
    cv.add_argument("--cache", type=Path, required=True)
    cv.add_argument("--output", type=Path, required=True)
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--seed", type=int, default=42)
    references = commands.add_parser("references")
    references.add_argument("--cache", type=Path, required=True)
    references.add_argument("--output", type=Path, required=True)
    references.add_argument("--folds", type=int, default=5)
    references.add_argument("--seed", type=int, default=42)
    study = commands.add_parser("study")
    study.add_argument("--cache", type=Path, required=True)
    study.add_argument("--selection", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "cv":
            result = run_decision_cv(args.cache, args.output, args.folds, args.seed)
        elif args.command == "references":
            result = run_reference_cv(args.cache, args.output, args.folds, args.seed)
        else:
            result = run_multiseed_study(args.cache, args.selection, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_decision_cv(cache: Path, output: Path, folds: int = 5, seed: int = 42) -> dict:
    cache = Path(cache)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train = CachedSplit.load(cache / "train.npz")
    assignments = grouped_vehicle_folds(
        train.vehicle_ids, train.labels, train.nodes, folds, seed
    )
    config = TrainingConfig(hidden_sizes=(64,), epochs=10, seed=seed)
    variant_rows = []
    for fold in range(folds):
        training = assignments != fold
        validation = assignments == fold
        evaluation = validation & _last_per_vehicle(train.vehicle_ids, validation)
        scaler = FederatedStandardScaler.fit(
            train.features[training], train.nodes[training]
        )
        train_features = scaler.transform(train.features[training])
        evaluation_features = scaler.transform(train.features[evaluation])
        correction = _prior_correction(
            train.labels[training], train.vehicle_ids[training]
        )
        for variant in VARIANTS:
            probabilities, _ = fit_centralized_candidate(
                train_features,
                train.labels[training],
                evaluation_features,
                TrainingConfig(**{**asdict(config), "seed": seed + fold}),
                variant.weighted_loss,
            )
            metrics = classification_metrics(
                train.labels[evaluation],
                _correct(
                    probabilities, correction if variant.prior_correction else None
                ),
                variant.decision,
            )
            variant_rows.append(_metric_row(variant.name, fold, metrics, evaluation))
    summaries = _summary(variant_rows, "variant")
    winner = min(
        summaries,
        key=lambda row: (
            row["mean_cost_mean"],
            -row["macro_pr_auc_mean"],
            -row["balanced_accuracy_mean"],
        ),
    )
    _write_csv(output / "variant_folds.csv", variant_rows)
    _write_csv(output / "variant_summary.csv", summaries)
    _write_csv(output / "fold_assignments.csv", _fold_rows(train, assignments))
    selection = {
        "winner": winner["variant"],
        "winner_config": asdict(
            next(item for item in VARIANTS if item.name == winner["variant"])
        ),
        "folds": folds,
        "seed": seed,
        "model": asdict(config),
        "evaluation": "last cached window per vehicle in held-out train fold",
        "prior_correction": (
            "training prior adjusted to the class frequency of the last cached "
            "train window per vehicle"
        ),
        "validation_used": False,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def run_reference_cv(cache: Path, output: Path, folds: int = 5, seed: int = 42) -> dict:
    cache = Path(cache)
    output = Path(output)
    if not output.is_dir():
        raise ValueError("La carpeta de selección de variante no existe.")
    train = CachedSplit.load(cache / "train.npz")
    assignments = grouped_vehicle_folds(
        train.vehicle_ids, train.labels, train.nodes, folds, seed
    )
    rows = []
    for fold in range(folds):
        training = assignments != fold
        validation = assignments == fold
        evaluation = validation & _last_per_vehicle(train.vehicle_ids, validation)
        scaler = FederatedStandardScaler.fit(
            train.features[training], train.nodes[training]
        )
        train_features = scaler.transform(train.features[training])
        evaluation_features = scaler.transform(train.features[evaluation])
        labels = train.labels[training]
        evaluation_labels = train.labels[evaluation]
        logistic, _ = fit_centralized_candidate(
            train_features,
            labels,
            evaluation_features,
            TrainingConfig(hidden_sizes=(), epochs=30, seed=seed + fold),
            False,
        )
        boosting = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=300,
            early_stopping=True,
            random_state=seed + fold,
        )
        boosting.fit(train_features, labels)
        probabilities = _aligned_probabilities(
            boosting.predict_proba(evaluation_features), boosting.classes_
        )
        for name, predicted in (
            ("logistic_regression", logistic),
            ("hist_gradient_boosting", probabilities),
        ):
            rows.append(
                _metric_row(
                    name,
                    fold,
                    classification_metrics(evaluation_labels, predicted, "cost"),
                    evaluation,
                    "model",
                )
            )
    _write_csv(output / "reference_folds.csv", rows)
    summary = _summary(rows, "model")
    _write_csv(output / "reference_summary.csv", summary)
    return {
        "output": str(output),
        "models": [row["model"] for row in summary],
        "validation_used": False,
    }


def run_multiseed_study(cache: Path, selection: Path, output: Path) -> dict:
    cache = Path(cache)
    selected = json.loads(
        (Path(selection) / "selection.json").read_text(encoding="utf-8")
    )
    variant = DecisionVariant(**selected["winner_config"])
    base = TrainingConfig(**_training_config(selected["model"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    histories = []
    for seed in range(100, 110):
        run_started = time.perf_counter()
        report = train_learning_regimes(
            cache,
            output / f"seed_{seed}",
            TrainingConfig(**{**asdict(base), "seed": seed}),
            variant.weighted_loss,
            variant.decision,
            variant.prior_correction,
        )
        for regime, result in report["results"].items():
            rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    **{key: result["global"][key] for key in METRICS},
                    "examples": result["global"]["examples"],
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )
        histories.extend(_client_loss_rows(seed, report["training_history"]))
    primary = _paired_statistics(rows, ("mean_cost",), holm=True)
    secondary = _paired_statistics(rows, METRICS[1:], holm=False)
    _write_csv(output / "metrics.csv", rows)
    _write_csv(output / "client_round_losses.csv", histories)
    _write_csv(output / "primary_statistics.csv", primary)
    _write_csv(output / "secondary_statistics.csv", secondary)
    manifest = {
        "selection": selected,
        "runtime": _runtime(),
        "elapsed_seconds": time.perf_counter() - started,
        "seeds": list(range(100, 110)),
        "validation_evaluations": 40,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"winner": variant.name, "output": str(output), "runs": len(rows)}


def _aligned_probabilities(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(values), 5), dtype=np.float64)
    probabilities[:, np.asarray(classes, dtype=int)] = values
    return probabilities


def _metric_row(name, fold, metrics, mask, field="variant") -> dict:
    return {
        field: name,
        "fold": fold + 1,
        "examples": int(np.sum(mask)),
        **{metric: metrics[metric] for metric in METRICS},
    }


def _last_per_vehicle(vehicle_ids: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    indexes = {}
    for index, vehicle in enumerate(vehicle_ids):
        if allowed[index]:
            indexes[str(vehicle)] = index
    result = np.zeros(len(vehicle_ids), dtype=bool)
    result[list(indexes.values())] = True
    return result


def _prior_correction(
    labels: np.ndarray, vehicle_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    training_prior = np.bincount(labels, minlength=5) / len(labels)
    last = _last_per_vehicle(vehicle_ids, np.ones(len(vehicle_ids), dtype=bool))
    deployment_prior = np.bincount(labels[last], minlength=5)
    return training_prior, deployment_prior / deployment_prior.sum()


def _correct(probabilities, correction) -> np.ndarray:
    if correction is None:
        return probabilities
    training_prior, deployment_prior = correction
    adjusted = probabilities * deployment_prior / training_prior
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _summary(rows: list[dict], field: str) -> list[dict]:
    result = []
    for name in sorted({row[field] for row in rows}):
        selected = [row for row in rows if row[field] == name]
        result.append(
            {
                field: name,
                **{
                    f"{metric}_{statistic}": float(
                        function([row[metric] for row in selected])
                    )
                    for metric in METRICS
                    for statistic, function in (
                        ("mean", np.mean),
                        ("sd", lambda values: np.std(values, ddof=1)),
                    )
                },
                "folds": len(selected),
            }
        )
    return result


def _fold_rows(cached: CachedSplit, assignments: np.ndarray) -> list[dict]:
    rows = []
    for vehicle in sorted(set(cached.vehicle_ids.astype(str))):
        index = np.flatnonzero(cached.vehicle_ids == vehicle)[-1]
        rows.append(
            {
                "vehicle_id": vehicle,
                "fold": int(assignments[index]) + 1,
                "last_window_label": int(cached.labels[index]),
            }
        )
    return rows


def _client_loss_rows(seed: int, history: dict) -> list[dict]:
    rows = []
    for regime in ("fedavg", "fedprox"):
        for point in history[regime]:
            rows.extend(
                {
                    "seed": seed,
                    "regime": regime,
                    "round": point["round"],
                    "node": node,
                    "training_loss": loss,
                }
                for node, loss in point["client_losses"].items()
            )
    return rows


def _paired_statistics(rows: list[dict], metrics, holm: bool) -> list[dict]:
    indexed = {(row["seed"], row["regime"]): row for row in rows}
    output = []
    for reference, candidate in COMPARISONS:
        for metric in metrics:
            differences = np.asarray(
                [
                    indexed[(seed, candidate)][metric]
                    - indexed[(seed, reference)][metric]
                    for seed in range(100, 110)
                ]
            )
            test = wilcoxon_signed_rank(differences)
            interval = bootstrap_mean_interval(
                differences, samples=10_000, seed=_seed(reference, candidate, metric)
            )
            output.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "mean_difference": float(np.mean(differences)),
                    "ci_95_lower": interval[0],
                    "ci_95_upper": interval[1],
                    "wilcoxon_statistic": test.statistic,
                    "p_value": test.p_value,
                    "rank_biserial": test.rank_biserial,
                    "nonzero_pairs": test.pairs,
                    "test_method": test.method,
                }
            )
    if holm:
        for row, value in zip(
            output, holm_adjust(row["p_value"] for row in output), strict=True
        ):
            row["p_value_holm"] = value
    return output


def _training_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key in TrainingConfig.__dataclass_fields__
    }


def _runtime() -> dict:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except OSError:
        commit = "unavailable"
    return {
        "commit": commit,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _seed(*parts: str) -> int:
    payload = "|".join(parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
