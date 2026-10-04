import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from minetwin.learning.data import CachedSplit
from minetwin.learning.inference import TorchRiskPrognosticator
from minetwin.learning.metrics import mean_misclassification_cost
from minetwin.learning.provenance import verified_evaluation_provenance


@dataclass(frozen=True)
class PermutationConfig:
    repeats: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("repeats debe ser un entero positivo.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed debe ser un entero no negativo.")


def run_permutation_importance(
    cache: Path,
    models: Path,
    output: Path,
    split: str = "validation",
    regime: str = "fedprox",
    config: PermutationConfig | None = None,
) -> dict:
    if split not in ("train", "validation"):
        raise ValueError("La importancia requiere etiquetas de train o validation.")
    if regime not in ("centralized", "fedavg", "fedprox"):
        raise ValueError("La importancia requiere un modelo global.")
    effective = config or PermutationConfig()
    cache = Path(cache)
    models = Path(models)
    training = json.loads((models / "metrics.json").read_text(encoding="utf-8"))
    _, provenance = verified_evaluation_provenance(cache, models, training, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        cached = CachedSplit.load(cache / f"{split}.npz")
        predictor = TorchRiskPrognosticator.load(
            models / f"{regime}.pt",
            models / "scaler.npz",
        )
        baseline = mean_misclassification_cost(
            cached.labels,
            predictor.predict_probabilities(cached.features),
        )
        rows = _importance_rows(cached, predictor, baseline, effective)
        result = {
            "split": split,
            "regime": regime,
            "model_id": predictor.model_id,
            "examples": len(cached.labels),
            "baseline_mean_cost": baseline,
            "config": asdict(effective),
            "provenance": {
                **provenance,
                "model_ids": {regime: predictor.model_id},
            },
            "groups": {
                dimension: sum(row["dimension"] == dimension for row in rows)
                for dimension in ("source_variable", "temporal_statistic")
            },
        }
        _write_csv(output / "permutation_importance.csv", rows)
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _importance_rows(
    cached: CachedSplit,
    predictor: TorchRiskPrognosticator,
    baseline: float,
    config: PermutationConfig,
) -> list[dict]:
    generator = np.random.default_rng(config.seed)
    rows = []
    for dimension, groups in feature_groups(cached.feature_names).items():
        for group, indexes in groups.items():
            increases = []
            for _ in range(config.repeats):
                order = generator.permutation(len(cached.features))
                permuted = cached.features.copy()
                permuted[:, indexes] = cached.features[order][:, indexes]
                cost = mean_misclassification_cost(
                    cached.labels,
                    predictor.predict_probabilities(permuted),
                )
                increases.append(cost - baseline)
            rows.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "features": len(indexes),
                    "mean_cost_increase": float(np.mean(increases)),
                    "sd_cost_increase": float(
                        np.std(increases, ddof=1) if len(increases) > 1 else 0
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["dimension"],
            -row["mean_cost_increase"],
            row["group"],
        ),
    )


def feature_groups(
    feature_names: tuple[str, ...],
) -> dict[str, dict[str, tuple[int, ...]]]:
    source: dict[str, list[int]] = {}
    statistic: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        if "__" in name:
            variable, temporal = name.rsplit("__", 1)
        else:
            variable, temporal = name, "derived"
        source.setdefault(variable, []).append(index)
        statistic.setdefault(temporal, []).append(index)
    return {
        "source_variable": {
            name: tuple(indexes) for name, indexes in source.items()
        },
        "temporal_statistic": {
            name: tuple(indexes) for name, indexes in statistic.items()
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
