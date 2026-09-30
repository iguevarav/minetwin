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
from minetwin.learning.contracts import TrainingConfig
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
    learning: Path,
    workshop: Path,
    scania: Path,
    federation: Path,
    langflow: Path | None = None,
    selection: Path | None = None,
) -> dict:
    sources = {
        "learning": _read_complete(learning),
        "workshop": _read_complete(workshop),
    }
    summary = {
        "dataset": _read_json(Path(scania) / "summary.json"),
        "learning": _read_json(Path(learning) / "summary.json"),
        "workshop": _read_json(Path(workshop) / "summary.json"),
        "federation": _read_json(Path(federation) / "summary.json"),
        "langflow": _langflow_summary(langflow),
        "selection": (
            _read_json(Path(selection) / "selection.json")
            if selection is not None
            else None
        ),
        "eda": (
            _read_json(Path(selection) / "eda.json")
            if selection is not None
            else None
        ),
        "sources": sources,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "study_summary.json", summary)
    learning_stats = _read_csv(Path(learning) / "statistics.csv")
    workshop_stats = _read_csv(Path(workshop) / "statistics.csv")
    candidates = (
        _read_csv(Path(selection) / "candidates.csv")
        if selection is not None
        else []
    )
    folds = (
        _read_csv(Path(selection) / "fold_metrics.csv")
        if selection is not None
        else []
    )
    (output / "report.html").write_text(
        _report_html(summary, candidates, folds, learning_stats, workshop_stats),
        encoding="utf-8",
    )
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


def _langflow_summary(path: Path | None):
    if path is None:
        return None
    path = Path(path)
    manifest = _read_json(path / "manifest.json")
    review = path / "review_summary.json"
    return {
        "manifest": manifest,
        "human_review": _read_json(review) if review.is_file() else None,
    }


def _report_html(summary, candidates, folds, learning_stats, workshop_stats) -> str:
    dataset = summary["dataset"]
    federation = summary["federation"]
    transfer = federation["transfer"]
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
            "Aprendizaje federado",
            learning_stats,
            statistical_fields,
            "Mejora positiva favorece al candidato. El intervalo bootstrap, Wilcoxon "
            "pareado, Holm y la biserial de rangos describen incertidumbre, significancia "
            "y tamaño del efecto.",
        ),
        _html_table(
            "Taller y coordinación",
            workshop_stats,
            statistical_fields,
            "Las políticas se comparan con la misma semilla y escenario. Producción, "
            "disponibilidad, fallas, cola y costo deben interpretarse conjuntamente.",
        ),
    ]
    if summary["eda"]:
        sections.insert(
            0,
            _html_table(
                "EDA y particiones",
                summary["eda"]["splits"],
                ("split", "examples", "vehicles", "nodes", "classes"),
                "Las particiones contienen vehículos distintos. La distribución de "
                "clases evidencia desbalance y la distribución por nodo muestra la "
                "heterogeneidad usada por el aprendizaje federado.",
            ),
        )
    if candidates:
        sections.insert(
            0,
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
                "Se selecciona el menor costo medio entre folds; los empates priorizan "
                "F1 macro, exactitud balanceada y menor número de parámetros. La "
                "arquitectura elegida se aplica luego a los cuatro regímenes.",
            ),
        )
    if folds:
        sections.insert(
            1,
            _html_table(
                "Validación cruzada agrupada",
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
                "Los folds se agrupan por vehículo para impedir que ventanas del mismo "
                "camión aparezcan a ambos lados de una partición.",
            ),
        )
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MineTwin · Evaluación final</title><style>
body{{background:#f4f6f8;color:#1b2430;font:14px system-ui;margin:0;padding:32px}}main{{max-width:1500px;margin:auto}}
h1,h2{{font-weight:600}}p{{color:#5f6b7a}}section{{background:#fff;border:1px solid #d5dbe3;border-radius:12px;margin:20px 0;padding:20px;overflow:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d5dbe3;padding:9px;text-align:right;white-space:nowrap}}th{{color:#9a6200}}th:first-child,td:first-child{{text-align:left}}
</style></head><body><main><h1>MineTwin · Evaluación final</h1>
<p>SCANIA Component X v{html.escape(str(dataset['version']))}: {dataset['features']} variables. El coordinador recibió {federation['coordinator_states']} estados publicados y {federation['coordinator_raw_features']} variables crudas.</p>
<p>Transferencia federada: {transfer['messages']} mensajes, {transfer['payload_bytes']} bytes y {transfer['raw_records']} registros crudos.</p>
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
    report.add_argument("--learning", type=Path, required=True)
    report.add_argument("--workshop", type=Path, required=True)
    report.add_argument("--scania", type=Path, required=True)
    report.add_argument("--federation", type=Path, required=True)
    report.add_argument("--langflow", type=Path)
    report.add_argument("--selection", type=Path)
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
                args.learning,
                args.workshop,
                args.scania,
                args.federation,
                args.langflow,
                args.selection,
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
