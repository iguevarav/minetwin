import argparse
import csv
import hashlib
import html
import json
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from minetwin.domain import Scenario
from minetwin.langflow_client import FLOW_INSTRUCTIONS, PROMPT_VERSION
from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.provenance import (
    file_sha256,
    value_sha256,
    verify_training_provenance,
)
from minetwin.learning.selection import load_selected_training
from minetwin.learning.training import train_learning_regimes
from minetwin.maintenance import MaintenanceConfig
from minetwin.operations import CoordinatedFleet, FleetPolicy, WorkshopConfig
from minetwin.simulation import SimulationConfig
from minetwin.stats import (
    bootstrap_mean_interval,
    holm_adjust,
    wilcoxon_signed_rank,
)


@dataclass(frozen=True)
class LearningStudyConfig:
    seeds: tuple[int, ...] = tuple(range(100, 110))
    bootstrap_samples: int = 10_000
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selection_source: str | None = None

    def __post_init__(self) -> None:
        _validate_unique_integers(self.seeds, "seeds")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples debe ser positivo.")


@dataclass(frozen=True)
class WorkshopStudyConfig:
    seeds: tuple[int, ...] = tuple(range(100, 110))
    scenarios: tuple[Scenario, ...] = (
        Scenario.NORMAL,
        Scenario.ENGINE_DEGRADATION,
        Scenario.ENGINE_VARIABLE_DEGRADATION,
    )
    bays: tuple[int, ...] = (1, 2, 3)
    cost_ratios: tuple[int, ...] = (3, 5, 10)
    steps: int = 450
    minimum_available: int = 6
    preventive_cost: float = 1_000.0
    queue_cost_per_hour: float = 50.0
    noninferiority_margin: float = 0.02
    bootstrap_samples: int = 10_000

    def __post_init__(self) -> None:
        _validate_unique_integers(self.seeds, "seeds")
        _validate_unique_integers(self.bays, "bays", 1)
        _validate_unique_integers(self.cost_ratios, "cost_ratios", 1)
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps debe ser positivo.")
        if type(self.minimum_available) is not int or self.minimum_available < 0:
            raise ValueError("minimum_available debe ser no negativo.")
        if self.preventive_cost <= 0:
            raise ValueError("preventive_cost debe ser positivo.")
        if self.queue_cost_per_hour < 0:
            raise ValueError("queue_cost_per_hour debe ser no negativo.")
        if not 0 < self.noninferiority_margin < 1:
            raise ValueError("noninferiority_margin debe estar entre cero y uno.")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples debe ser positivo.")
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios debe contener valores únicos.")


def run_learning_study(
    cache: Path,
    output: Path,
    config: LearningStudyConfig | None = None,
) -> dict:
    effective = config or LearningStudyConfig()
    output = _start_run(output, "learning", _learning_config(effective, cache))
    rows = []
    try:
        for seed in effective.seeds:
            report = train_learning_regimes(
                cache,
                output / f"seed_{seed}",
                replace(effective.training, seed=seed),
            )
            for regime, result in report["results"].items():
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        **{
                            key: value
                            for key, value in result["global"].items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
        statistics = _learning_statistics(rows, effective)
        _write_csv(output / "metrics.csv", rows)
        _write_csv(output / "statistics.csv", statistics)
        summary = {
            "runs": len(rows),
            "seeds": len(effective.seeds),
            "regimes": sorted({row["regime"] for row in rows}),
            "aggregates": _aggregate(rows, "regime"),
        }
        _write_json(output / "summary.json", summary)
        _complete_run(output, {"metrics": len(rows), "comparisons": len(statistics)})
        return summary
    except BaseException:
        _fail_run(output)
        raise


def run_workshop_study(
    output: Path,
    config: WorkshopStudyConfig | None = None,
) -> dict:
    effective = config or WorkshopStudyConfig()
    output = _start_run(output, "workshop", _workshop_config(effective))
    rows = []
    try:
        for scenario in effective.scenarios:
            for bays in effective.bays:
                for seed in effective.seeds:
                    base_rows = _run_workshop_case(effective, scenario, bays, seed)
                    for base in base_rows:
                        for ratio in effective.cost_ratios:
                            rows.append(_apply_cost_ratio(base, ratio, effective))
        statistics = _workshop_statistics(rows, effective)
        noninferiority = _production_noninferiority(rows, effective)
        _write_csv(output / "metrics.csv", rows)
        _write_csv(output / "statistics.csv", statistics)
        _write_json(output / "noninferiority.json", noninferiority)
        reference_rows = [
            row
            for row in rows
            if row["cost_ratio"] == _reference_ratio(effective.cost_ratios)
        ]
        summary = {
            "simulations": len(reference_rows),
            "cost_evaluations": len(rows),
            "aggregates": _aggregate(reference_rows, "policy"),
            "noninferiority": noninferiority,
        }
        _write_json(output / "summary.json", summary)
        _complete_run(
            output,
            {
                "simulations": len(reference_rows),
                "cost_evaluations": len(rows),
                "comparisons": len(statistics),
            },
        )
        return summary
    except BaseException:
        _fail_run(output)
        raise


def build_study_report(
    output: Path,
    cache: Path,
    learning: Path,
    scania: Path,
    selection: Path,
    interpretability: Path,
    federation: Path,
    langflow: Path,
) -> dict:
    cache = Path(cache)
    learning = Path(learning)
    scania = Path(scania)
    selection = Path(selection)
    interpretability = Path(interpretability)
    federation = Path(federation)
    langflow = Path(langflow)
    sources = {
        "learning": _read_complete(learning),
        "selection": _read_complete(selection),
    }
    summary = {
        "scope": "SCANIA Component X; public historical heavy-truck data",
        "dataset": _read_json(scania / "summary.json"),
        "learning": _read_json(learning / "summary.json"),
        "federation": _read_json(federation / "summary.json"),
        "langflow": _langflow_summary(langflow),
        "selection": _read_json(selection / "selection.json"),
        "eda": _read_json(selection / "eda.json"),
        "interpretability": _read_json(interpretability / "summary.json"),
        "sources": sources,
    }
    learning_stats = _read_csv(learning / "statistics.csv")
    candidates = _read_csv(selection / "candidates.csv")
    folds = _read_csv(selection / "fold_metrics.csv")
    importance = _read_csv(interpretability / "permutation_importance.csv")
    if not all((learning_stats, candidates, folds, importance)):
        raise ValueError("El reporte requiere resultados SCANIA no vacíos.")
    _verify_real_report_sources(
        summary, cache, learning, scania, federation, langflow
    )
    summary["learning_statistics"] = learning_stats
    summary["interpretability"]["importance"] = importance
    summary["sources"]["sha256"] = {
        "dataset_manifest": file_sha256(scania / "dataset_manifest.json"),
        "cache_metadata": file_sha256(cache / "metadata.json"),
        "selection": file_sha256(selection / "selection.json"),
        "learning_statistics": file_sha256(learning / "statistics.csv"),
        "interpretability": file_sha256(
            interpretability / "permutation_importance.csv"
        ),
        "federation": file_sha256(federation / "summary.json"),
        "published_states": file_sha256(federation / "published_states.jsonl"),
        "langflow": file_sha256(langflow / "manifest.json"),
        "langflow_cases": file_sha256(langflow / "cases.jsonl"),
        "langflow_flow": file_sha256(langflow / "flow_definition.json"),
        "langflow_prompt": file_sha256(langflow / "prompt.txt"),
        "langflow_review_csv": file_sha256(langflow / "review.csv"),
        "langflow_review": file_sha256(langflow / "review_summary.json"),
    }
    output = _start_run(
        output, "scania_report", {"sources": summary["sources"]["sha256"]}
    )
    try:
        _write_json(output / "study_summary.json", summary)
        (output / "report.html").write_text(
            _report_html(summary, candidates, folds), encoding="utf-8"
        )
        _complete_run(
            output,
            {
                "learning_seeds": summary["learning"]["seeds"],
                "validation_vehicles": next(
                    row["vehicles"]
                    for row in summary["dataset"]["splits"]
                    if row["split"] == "validation"
                ),
                "langflow_responses": summary["langflow"]["manifest"]["counts"][
                    "responses"
                ],
            },
        )
    except BaseException:
        _fail_run(output)
        raise
    return summary


def _run_workshop_case(config, scenario, bays, seed) -> list[dict]:
    rows = []
    workshop = WorkshopConfig(
        bays=bays,
        minimum_available=config.minimum_available,
        preventive_cost=config.preventive_cost,
        corrective_cost=config.preventive_cost * config.cost_ratios[0],
        queue_cost_per_hour=config.queue_cost_per_hour,
    )
    for policy in FleetPolicy:
        fleet = CoordinatedFleet(
            SimulationConfig(seed=seed, scenario=scenario),
            workshop_config=workshop,
            policy=policy,
        )
        fleet.advance(config.steps)
        rows.append(
            {
                "scenario": scenario.value,
                "bays": bays,
                "seed": seed,
                **fleet.metrics,
            }
        )
    return rows


def _apply_cost_ratio(row: dict, ratio: int, config) -> dict:
    preventive = row["preventive_starts"] * config.preventive_cost
    corrective = max(row["failures"], row["corrective_starts"])
    corrective *= config.preventive_cost * ratio
    return {
        **row,
        "cost_ratio": ratio,
        "preventive_cost": preventive,
        "corrective_cost": corrective,
        "maintenance_cost": preventive + corrective,
        "total_cost": preventive + corrective + row["queue_cost"],
    }


def _learning_statistics(rows: list[dict], config) -> list[dict]:
    comparisons = (
        ("local", "fedavg"),
        ("local", "fedprox"),
        ("centralized", "fedavg"),
        ("centralized", "fedprox"),
    )
    metrics = (
        ("total_cost", False),
        ("macro_f1", True),
        ("balanced_accuracy", True),
        ("macro_pr_auc", True),
    )
    result = []
    for reference, candidate in comparisons:
        for metric, higher_is_better in metrics:
            result.append(
                _paired_result(
                    rows,
                    ("seed",),
                    "regime",
                    reference,
                    candidate,
                    metric,
                    higher_is_better,
                    config.bootstrap_samples,
                    "learning",
                )
            )
    return _adjust_statistics(result)


def _workshop_statistics(rows: list[dict], config) -> list[dict]:
    result = []
    operational = [
        row
        for row in rows
        if row["cost_ratio"] == _reference_ratio(config.cost_ratios)
    ]
    comparisons = (
        (FleetPolicy.P0_NONE.value, FleetPolicy.P1_THRESHOLD.value),
        (FleetPolicy.P0_NONE.value, FleetPolicy.P2_LOCAL_PREDICTIVE.value),
        (FleetPolicy.P0_NONE.value, FleetPolicy.P3_COORDINATED.value),
        (FleetPolicy.P2_LOCAL_PREDICTIVE.value, FleetPolicy.P3_COORDINATED.value),
    )
    metrics = (
        ("failures", False),
        ("tonnes_delivered", True),
        ("availability", True),
        ("downtime_seconds", False),
        ("queue_seconds", False),
    )
    for reference, candidate in comparisons:
        for metric, higher_is_better in metrics:
            result.append(
                _paired_result(
                    operational,
                    ("scenario", "bays", "seed"),
                    "policy",
                    reference,
                    candidate,
                    metric,
                    higher_is_better,
                    config.bootstrap_samples,
                    "workshop",
                )
            )
        for ratio in config.cost_ratios:
            selected = [row for row in rows if row["cost_ratio"] == ratio]
            comparison = _paired_result(
                selected,
                ("scenario", "bays", "seed"),
                "policy",
                reference,
                candidate,
                "total_cost",
                False,
                config.bootstrap_samples,
                "workshop",
            )
            comparison["cost_ratio"] = ratio
            result.append(comparison)
    return _adjust_statistics(result)


def _paired_result(
    rows,
    keys,
    group,
    reference,
    candidate,
    metric,
    higher_is_better,
    bootstrap_samples,
    study,
) -> dict:
    selected = {(row[group], *(row[key] for key in keys)): row for row in rows}
    shared = sorted(
        {
            tuple(row[key] for key in keys)
            for row in rows
            if row[group] == reference
        }
        & {
            tuple(row[key] for key in keys)
            for row in rows
            if row[group] == candidate
        },
        key=str,
    )
    reference_values = np.asarray(
        [selected[(reference, *key)][metric] for key in shared], dtype=np.float64
    )
    candidate_values = np.asarray(
        [selected[(candidate, *key)][metric] for key in shared], dtype=np.float64
    )
    if not len(shared):
        raise ValueError(f"No existen pares para {reference} y {candidate}.")
    difference = candidate_values - reference_values
    improvement = difference if higher_is_better else -difference
    interval = bootstrap_mean_interval(
        improvement,
        samples=bootstrap_samples,
        seed=_stable_seed(study, reference, candidate, metric),
    )
    test = wilcoxon_signed_rank(improvement)
    return {
        "study": study,
        "reference": reference,
        "candidate": candidate,
        "metric": metric,
        "direction": "higher" if higher_is_better else "lower",
        "pairs": len(shared),
        "nonzero_pairs": test.pairs,
        "reference_mean": float(np.mean(reference_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "mean_difference": float(np.mean(difference)),
        "mean_improvement": float(np.mean(improvement)),
        "improvement_ci_lower": interval[0],
        "improvement_ci_upper": interval[1],
        "wilcoxon_statistic": test.statistic,
        "p_value": test.p_value,
        "rank_biserial": test.rank_biserial,
        "test_method": test.method,
    }


def _production_noninferiority(rows, config) -> dict:
    selected = [
        row
        for row in rows
        if row["cost_ratio"] == _reference_ratio(config.cost_ratios)
    ]
    indexed = {
        (row["policy"], row["scenario"], row["bays"], row["seed"]): row
        for row in selected
    }
    ratios = []
    for scenario in config.scenarios:
        for bays in config.bays:
            for seed in config.seeds:
                reference = indexed[
                    (FleetPolicy.P2_LOCAL_PREDICTIVE.value, scenario.value, bays, seed)
                ]["tonnes_delivered"]
                candidate = indexed[
                    (FleetPolicy.P3_COORDINATED.value, scenario.value, bays, seed)
                ]["tonnes_delivered"]
                if reference:
                    ratios.append((candidate - reference) / reference)
    lower, upper = bootstrap_mean_interval(
        ratios,
        confidence=0.90,
        samples=config.bootstrap_samples,
        seed=42,
    )
    return {
        "reference": FleetPolicy.P2_LOCAL_PREDICTIVE.value,
        "candidate": FleetPolicy.P3_COORDINATED.value,
        "metric": "tonnes_delivered",
        "pairs": len(ratios),
        "margin": config.noninferiority_margin,
        "mean_relative_difference": float(np.mean(ratios)),
        "confidence": 0.90,
        "ci_lower": lower,
        "ci_upper": upper,
        "noninferior": lower > -config.noninferiority_margin,
    }


def _adjust_statistics(rows: list[dict]) -> list[dict]:
    adjusted = holm_adjust(row["p_value"] for row in rows)
    for row, value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = value
    return rows


def _reference_ratio(ratios: tuple[int, ...]) -> int:
    return 5 if 5 in ratios else ratios[0]


def _aggregate(rows: list[dict], group: str) -> dict:
    excluded = {group, "seed", "scenario", "bays", "cost_ratio"}
    result = {}
    for value in sorted({str(row[group]) for row in rows}):
        selected = [row for row in rows if str(row[group]) == value]
        numeric = {
            key
            for row in selected
            for key, item in row.items()
            if key not in excluded and isinstance(item, (int, float))
        }
        result[value] = {
            key: {
                "mean": float(np.mean([row[key] for row in selected])),
                "sd": float(np.std([row[key] for row in selected], ddof=1))
                if len(selected) > 1
                else 0.0,
            }
            for key in sorted(numeric)
            if all(key in row for row in selected)
        }
    return result


def _start_run(output: Path, study: str, config: dict) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "running",
            "study": study,
            "config": config,
            "code_sha256": _code_fingerprint(),
            "dependencies": _versions(),
        },
    )
    return output


def _complete_run(output: Path, counts: dict) -> None:
    manifest = _read_json(output / "manifest.json")
    manifest.update(status="complete", counts=counts)
    _write_json(output / "manifest.json", manifest)


def _fail_run(output: Path) -> None:
    manifest = _read_json(output / "manifest.json")
    manifest["status"] = "incomplete"
    _write_json(output / "manifest.json", manifest)


def _read_complete(path: Path) -> dict:
    manifest = _read_json(Path(path) / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"La ejecución {path} no está completa.")
    return manifest


def _learning_config(config, cache: Path) -> dict:
    return {
        "cache": str(Path(cache).resolve()),
        "seeds": config.seeds,
        "bootstrap_samples": config.bootstrap_samples,
        "training": asdict(config.training),
        "selection_source": config.selection_source,
    }


def _workshop_config(config) -> dict:
    result = asdict(config)
    result["scenarios"] = tuple(item.value for item in config.scenarios)
    result["maintenance"] = asdict(MaintenanceConfig())
    result["simulation"] = asdict(SimulationConfig())
    result["simulation"]["scenario"] = "varies_by_run"
    result["simulation"]["start_time"] = result["simulation"][
        "start_time"
    ].isoformat()
    return result


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions() -> dict:
    result = {}
    for package in ("numpy", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No hay filas para exportar.")
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(*values) -> int:
    digest = hashlib.sha256(":".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _validate_unique_integers(values, name, minimum: int = 0) -> None:
    if not values or any(
        type(value) is not int or value < minimum for value in values
    ):
        raise ValueError(f"{name} debe contener enteros desde {minimum}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} no puede contener repetidos.")


def _langflow_summary(path: Path):
    path = Path(path)
    manifest = _read_json(path / "manifest.json")
    return {
        "manifest": manifest,
        "human_review": _read_json(path / "review_summary.json"),
    }


def _verify_real_report_sources(
    summary: dict,
    cache: Path,
    learning: Path,
    scania: Path,
    federation: Path,
    langflow: Path,
) -> None:
    dataset = summary["dataset"]
    metadata = _read_json(cache / "metadata.json")
    selection = summary["selection"]
    eda = summary["eda"]
    learning_manifest = summary["sources"]["learning"]
    federation_summary = summary["federation"]
    provenance = federation_summary.get("provenance", {})
    interpretation = summary["interpretability"]
    flow = summary["langflow"]["manifest"]
    review = summary["langflow"]["human_review"]
    if dataset.get("dataset") != "SCANIA Component X":
        raise ValueError("El reporte solo admite el dataset SCANIA Component X.")
    if metadata["source_manifest_sha256"] != file_sha256(
        scania / "dataset_manifest.json"
    ):
        raise ValueError("La caché no procede del dataset indicado.")
    if (
        eda["feature_config"] != metadata["feature_config"]
        or eda["feature_count"] != metadata["feature_count"]
        or selection["best_config"] != learning_manifest["config"]["training"]
        or selection["cross_validation"]["train_validation_vehicle_overlap"] != 0
        or summary["learning"]["seeds"] < 2
        or len(learning_manifest["config"]["seeds"]) != summary["learning"]["seeds"]
        or 100 not in learning_manifest["config"]["seeds"]
    ):
        raise ValueError("La selección, la caché y el aprendizaje no coinciden.")
    validation = next(
        row for row in dataset["splits"] if row["split"] == "validation"
    )
    eda_validation = next(
        row for row in eda["splits"] if row["split"] == "validation"
    )
    available_classes = sorted(
        int(label)
        for label, count in eda_validation["classes"].items()
        if count
    )
    if (
        not provenance.get("verified")
        or federation_summary["split"] != "validation"
        or federation_summary["regime"] not in ("fedavg", "fedprox")
        or federation_summary["coordinator_states"] != validation["vehicles"]
        or eda_validation["vehicles"] != validation["vehicles"]
        or federation_summary["coordinator_raw_features"] != 0
        or federation_summary["transfer"]["raw_records"] != 0
        or file_sha256(federation / "published_states.jsonl")
        != federation_summary.get("published_states_sha256")
    ):
        raise ValueError("La publicación federada no es verificable o compatible.")
    model_root = learning / "seed_100"
    model_report = _read_json(model_root / "metrics.json")
    training = verify_training_provenance(cache, model_root, model_report)
    split_hash = file_sha256(cache / "validation.npz")
    data_version = value_sha256(
        {
            "training_id": training["run_id"],
            "split": "validation",
            "sha256": split_hash,
        }
    )
    regime = federation_summary["regime"]
    model_id = f"{regime}:{file_sha256(model_root / f'{regime}.pt')[:12]}"
    if (
        provenance["training_id"] != training["run_id"]
        or provenance["data_version"] != data_version
        or provenance["split_sha256"] != split_hash
        or set(provenance["model_ids"].values()) != {model_id}
        or interpretation["model_id"] != model_id
        or interpretation["split"] != "validation"
        or interpretation["regime"] != regime
        or interpretation["examples"] != validation["vehicles"]
    ):
        raise ValueError("La federación y la interpretación usan modelos distintos.")
    if (
        flow.get("status") != "complete"
        or flow.get("source") != "SCANIA Component X validation"
        or flow.get("prompt_version") != PROMPT_VERSION
        or flow.get("training_id") != training["run_id"]
        or flow.get("data_version") != data_version
        or flow.get("predictive_models") != [model_id]
        or flow.get("observed_classes_evaluated") != available_classes
        or not flow.get("language_model_id")
        or not flow.get("flow_version")
        or not flow.get("counts", {}).get("responses")
        or review["reviewed_responses"] != flow["counts"]["responses"]
        or review["total_cases"] != flow["counts"]["cases"]
        or file_sha256(langflow / "cases.jsonl") != flow.get("cases_sha256")
        or file_sha256(langflow / "review.csv") != review.get("review_sha256")
        or file_sha256(langflow / "flow_definition.json")
        != flow.get("flow_definition_sha256")
        or file_sha256(langflow / "prompt.txt") != flow.get("prompt_sha256")
        or hashlib.sha256(FLOW_INSTRUCTIONS.encode()).hexdigest()
        != flow["prompt_sha256"]
    ):
        raise ValueError("La evaluación Langflow no es trazable a SCANIA.")


def _report_html(summary, candidates, folds) -> str:
    dataset = summary["dataset"]
    federation = summary["federation"]
    transfer = federation["transfer"]
    interpretation = summary["interpretability"]
    flow = summary["langflow"]["manifest"]
    review = summary["langflow"]["human_review"]
    statistical_fields = (
        "reference",
        "candidate",
        "metric",
        "pairs",
        "mean_improvement",
        "improvement_ci_lower",
        "improvement_ci_upper",
        "p_value_holm",
        "rank_biserial",
    )
    sections = [
        _html_table(
            "Dataset oficial y particiones",
            dataset["splits"],
            ("split", "vehicles", "readouts", "features", "missing_values"),
            "Los tres splits proceden del dataset público SCANIA Component X. "
            "Las variables son anónimas y no describen operaciones mineras.",
        ),
        _html_table(
            "EDA y ventanas causales",
            [
                {
                    "split": row["split"],
                    "examples": row["examples"],
                    "vehicles": row["vehicles"],
                    "nodes": len(row["nodes"]),
                    **{
                        f"class_{label}": row["classes"].get(str(label), 0)
                        for label in range(5)
                    },
                }
                for row in summary["eda"]["splits"]
            ],
            (
                "split",
                "examples",
                "vehicles",
                "nodes",
                "class_0",
                "class_1",
                "class_2",
                "class_3",
                "class_4",
            ),
            "Las clases muestran desbalance; las ventanas de entrenamiento pueden "
            "aportar varios ejemplos por vehículo. La validación permanece separada.",
        ),
        _html_table(
            "Selección de hiperparámetros",
            candidates,
            (
                "candidate",
                "hidden_sizes",
                "epochs",
                "learning_rate",
                "parameters",
                "mean_cost_mean",
                "mean_cost_sd",
                "macro_f1_mean",
                "selected",
            ),
            "El menor costo medio decide; F1 macro, exactitud balanceada y tamaño "
            "del modelo resuelven empates.",
        ),
        _html_table(
            "Validación cruzada por vehículo",
            folds,
            (
                "candidate",
                "fold",
                "vehicles",
                "mean_cost",
                "macro_f1",
                "balanced_accuracy",
                "macro_pr_auc",
            ),
            "Ningún vehículo debe aparecer en entrenamiento y validación del mismo "
            "fold; los folds pertenecen a la selección, no a la prueba final.",
        ),
        _html_table(
            "Aprendizaje por régimen",
            [
                {
                    "regime": regime,
                    "mean_cost": metrics["mean_cost"]["mean"],
                    "mean_cost_sd": metrics["mean_cost"]["sd"],
                    "macro_f1": metrics["macro_f1"]["mean"],
                    "balanced_accuracy": metrics["balanced_accuracy"]["mean"],
                    "macro_pr_auc": metrics["macro_pr_auc"]["mean"],
                }
                for regime, metrics in summary["learning"]["aggregates"].items()
            ],
            (
                "regime",
                "mean_cost",
                "mean_cost_sd",
                "macro_f1",
                "balanced_accuracy",
                "macro_pr_auc",
            ),
            "El costo medio menor es mejor. F1 macro, exactitud balanceada y PR-AUC "
            "macro muestran rendimiento en clases poco frecuentes.",
        ),
        _html_table(
            "Pruebas estadísticas pareadas",
            summary["learning_statistics"],
            statistical_fields,
            "Mejora positiva favorece al candidato. El intervalo bootstrap, Wilcoxon "
            "pareado, Holm y la biserial de rangos describen incertidumbre, "
            "significancia y tamaño del efecto.",
        ),
        _html_table(
            "Interpretabilidad por permutación",
            [
                row
                for row in interpretation["importance"]
                if row["dimension"] == "source_variable"
            ][:10],
            ("group", "features", "mean_cost_increase", "sd_cost_increase"),
            "Un aumento de costo al permutar indica asociación predictiva en "
            "validación; no identifica una pieza física ni demuestra causalidad.",
        ),
        _html_table(
            "Publicación federada por partición",
            [
                {"node": node, **values}
                for node, values in federation["nodes"].items()
            ],
            (
                "node",
                "private_records",
                "published_states",
                "mean_observation_quality",
            ),
            "Los nodos son particiones lógicas de un mismo dataset. La calidad "
            "mide cobertura de lecturas, no precisión del modelo.",
        ),
        _html_table(
            "Evaluación Langflow y revisión humana",
            [
                {"measure": "Casos", "value": flow["counts"]["cases"]},
                {"measure": "Respuestas", "value": flow["counts"]["responses"]},
                {"measure": "Errores", "value": flow["counts"]["errors"]},
                {
                    "measure": "Clases observadas cubiertas",
                    "value": flow["observed_classes_evaluated"],
                },
                {
                    "measure": "Respuestas revisadas",
                    "value": review["reviewed_responses"],
                },
                {"measure": "Puntuación media", "value": review["overall_mean_score"]},
                *(
                    {"measure": metric, "value": value}
                    for metric, value in review["mean_scores"].items()
                ),
            ],
            ("measure", "value"),
            "Las puntuaciones son revisión humana de explicaciones; no sustituyen "
            "las métricas del modelo predictivo ni validan un despliegue minero.",
        ),
    ]
    sources = summary["sources"]["sha256"]
    sections.append(
        _html_table(
            "Trazabilidad",
            [{"artifact": name, "sha256": digest} for name, digest in sources.items()],
            ("artifact", "sha256"),
            "Las huellas identifican los archivos usados por este reporte. El modelo, "
            "el flujo y el prompt se vinculan a la publicación verificada.",
        )
    )
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MineTwin · Evaluación final</title><style>
body{{background:#f4f6f8;color:#1b2430;font:14px system-ui;margin:0;padding:32px}}main{{max-width:1500px;margin:auto}}
h1,h2{{font-weight:600}}p{{color:#5f6b7a}}section{{background:#fff;border:1px solid #d5dbe3;border-radius:12px;margin:20px 0;padding:20px;overflow:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d5dbe3;padding:9px;text-align:right;white-space:nowrap}}th{{color:#9a6200}}th:first-child,td:first-child{{text-align:left}}
</style></head><body><main><h1>MineTwin · Evaluación final</h1>
<p>SCANIA Component X v{html.escape(str(dataset['version']))}: {dataset['features']} variables. El coordinador recibió {federation['coordinator_states']} estados publicados y {federation['coordinator_raw_features']} variables crudas.</p>
<p>Transferencia federada: {transfer['messages']} mensajes, {transfer['payload_bytes']} bytes medidos y estimados, y {transfer['raw_records']} registros crudos publicados.</p>
<p>Reducción del tamaño de estados publicados frente a centralizar las variables: {federation['publication_reduction_ratio']:.2%}. Es una comparación del tamaño publicado, no una medición de red.</p>
<p>Modelo {html.escape(interpretation['model_id'])} · flujo {html.escape(flow['flow_id'])} versión {html.escape(flow['flow_version'])} · modelo lingüístico {html.escape(flow['language_model_id'])}.</p>
<p>Estudio retrospectivo en camiones pesados. Las particiones federadas son experimentales sobre una sola fuente pública; no se validó en mina, entre contratistas ni en operación en tiempo real.</p>
{''.join(sections)}</main></body></html>"""


def _html_table(
    title: str,
    rows: list[dict],
    fields: tuple[str, ...],
    explanation: str,
) -> str:
    headers = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        + "</tr>"
        for row in rows
    )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<p>Cómo interpretarlo: {html.escape(explanation)}</p>"
        f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></section>"
    )


def _integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Se requieren enteros separados por comas.") from error
    try:
        _validate_unique_integers(result, "values")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return result


def _scenarios(value: str) -> tuple[Scenario, ...]:
    try:
        result = tuple(Scenario(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Escenario desconocido.") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Los escenarios deben ser únicos.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación final de MineTwin.")
    modes = parser.add_subparsers(dest="mode", required=True)
    learning = modes.add_parser("learning")
    learning.add_argument("--cache", type=Path, required=True)
    learning.add_argument("--output", type=Path, required=True)
    learning.add_argument("--seeds", type=_integers, default=tuple(range(100, 110)))
    learning.add_argument("--selection", type=Path)
    workshop = modes.add_parser("workshop")
    workshop.add_argument("--output", type=Path, required=True)
    workshop.add_argument("--steps", type=int, default=450)
    workshop.add_argument("--seeds", type=_integers, default=tuple(range(100, 110)))
    workshop.add_argument("--bays", type=_integers, default=(1, 2, 3))
    workshop.add_argument("--cost-ratios", type=_integers, default=(3, 5, 10))
    workshop.add_argument(
        "--scenarios",
        type=_scenarios,
        default=(
            Scenario.NORMAL,
            Scenario.ENGINE_DEGRADATION,
            Scenario.ENGINE_VARIABLE_DEGRADATION,
        ),
    )
    report = modes.add_parser("report")
    report.add_argument("--cache", type=Path, required=True)
    report.add_argument("--learning", type=Path, required=True)
    report.add_argument("--scania", type=Path, required=True)
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--interpretability", type=Path, required=True)
    report.add_argument("--federation", type=Path, required=True)
    report.add_argument("--langflow", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "learning":
            training = (
                load_selected_training(args.selection)
                if args.selection is not None
                else TrainingConfig()
            )
            result = run_learning_study(
                args.cache,
                args.output,
                LearningStudyConfig(
                    seeds=args.seeds,
                    training=training,
                    selection_source=(
                        str(args.selection.resolve())
                        if args.selection is not None
                        else None
                    ),
                ),
            )
        elif args.mode == "workshop":
            result = run_workshop_study(
                args.output,
                WorkshopStudyConfig(
                    seeds=args.seeds,
                    scenarios=args.scenarios,
                    bays=args.bays,
                    cost_ratios=args.cost_ratios,
                    steps=args.steps,
                ),
            )
        else:
            result = build_study_report(
                args.output,
                args.cache,
                args.learning,
                args.scania,
                args.selection,
                args.interpretability,
                args.federation,
                args.langflow,
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
