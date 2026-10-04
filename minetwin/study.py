import argparse
import csv
import hashlib
import html
import json
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from minetwin.langflow_client import FLOW_INSTRUCTIONS, PROMPT_VERSION
from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.provenance import (
    file_sha256,
    verified_evaluation_provenance,
    verify_training_provenance,
)
from minetwin.learning.selection import load_selected_training
from minetwin.learning.training import train_learning_regimes
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


def run_learning_study(
    cache: Path,
    output: Path,
    config: LearningStudyConfig | None = None,
) -> dict:
    effective = config or LearningStudyConfig()
    output = _start_run(output, "learning", _learning_config(effective, cache))
    rows = []
    runs = {}
    try:
        for seed in effective.seeds:
            report = train_learning_regimes(
                cache,
                output / f"seed_{seed}",
                replace(effective.training, seed=seed),
            )
            runs[str(seed)] = verify_training_provenance(
                cache, output / f"seed_{seed}", report
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
            "provenance": _study_provenance(runs),
        }
        _write_json(output / "summary.json", summary)
        _complete_run(output, {"metrics": len(rows), "comparisons": len(statistics)})
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
        raise ValueError("El reporte requiere resultados SCANIA no vacÃ­os.")
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


def _adjust_statistics(rows: list[dict]) -> list[dict]:
    adjusted = holm_adjust(row["p_value"] for row in rows)
    for row, value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = value
    return rows


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
        raise ValueError(f"La ejecuciÃ³n {path} no estÃ¡ completa.")
    return manifest


def _learning_config(config, cache: Path) -> dict:
    return {
        "cache": str(Path(cache).resolve()),
        "seeds": config.seeds,
        "bootstrap_samples": config.bootstrap_samples,
        "training": asdict(config.training),
        "selection_source": config.selection_source,
    }


def _study_provenance(runs: dict[str, dict]) -> dict:
    cache = next(iter(runs.values()))["cache"]
    if any(run["cache"] != cache for run in runs.values()):
        raise ValueError("Las semillas del estudio no comparten la misma cache.")
    return {
        "verified": True,
        "cache_sha256": cache,
        "training_ids": {seed: run["run_id"] for seed, run in runs.items()},
    }


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
    learning_provenance = summary["learning"].get("provenance", {})
    federation_summary = summary["federation"]
    provenance = federation_summary.get("provenance", {})
    interpretation = summary["interpretability"]
    interpretation_provenance = interpretation.get("provenance", {})
    flow = summary["langflow"]["manifest"]
    review = summary["langflow"]["human_review"]
    if dataset.get("dataset") != "SCANIA Component X":
        raise ValueError("El reporte solo admite el dataset SCANIA Component X.")
    if metadata["source_manifest_sha256"] != file_sha256(
        scania / "dataset_manifest.json"
    ):
        raise ValueError("La cachÃ© no procede del dataset indicado.")
    if (
        eda["feature_config"] != metadata["feature_config"]
        or eda["feature_count"] != metadata["feature_count"]
        or selection["best_config"] != learning_manifest["config"]["training"]
        or selection["cross_validation"]["train_validation_vehicle_overlap"] != 0
        or summary["learning"]["seeds"] < 2
        or len(learning_manifest["config"]["seeds"]) != summary["learning"]["seeds"]
        or 100 not in learning_manifest["config"]["seeds"]
    ):
        raise ValueError("La selecciÃ³n, la cachÃ© y el aprendizaje no coinciden.")
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
        raise ValueError("La publicaciÃ³n federada no es verificable o compatible.")
    model_root = learning / "seed_100"
    model_report = _read_json(model_root / "metrics.json")
    training, evaluation_provenance = verified_evaluation_provenance(
        cache, model_root, model_report, "validation"
    )
    regime = federation_summary["regime"]
    model_id = f"{regime}:{file_sha256(model_root / f'{regime}.pt')[:12]}"
    if (
        not learning_provenance.get("verified")
        or learning_provenance.get("cache_sha256") != training["cache"]
        or learning_provenance.get("training_ids", {}).get("100")
        != training["run_id"]
        or not _matches_provenance(provenance, evaluation_provenance)
        or set(provenance["model_ids"].values()) != {model_id}
        or interpretation["model_id"] != model_id
        or interpretation["split"] != "validation"
        or interpretation["regime"] != regime
        or interpretation["examples"] != validation["vehicles"]
        or not _matches_provenance(interpretation_provenance, evaluation_provenance)
        or interpretation_provenance.get("model_ids") != {regime: model_id}
    ):
        raise ValueError("La federaciÃ³n y la interpretaciÃ³n usan modelos distintos.")
    if (
        flow.get("status") != "complete"
        or flow.get("source") != "SCANIA Component X validation"
        or flow.get("prompt_version") != PROMPT_VERSION
        or flow.get("training_id") != training["run_id"]
        or flow.get("data_version") != evaluation_provenance["data_version"]
        or flow.get("predictive_models") != [model_id]
        or flow.get("observed_classes_evaluated") != available_classes
        or not flow.get("language_model_id")
        or not flow.get("langflow_url")
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
        raise ValueError("La evaluaciÃ³n Langflow no es trazable a SCANIA.")


def _matches_provenance(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


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
            "Los tres splits proceden del dataset pÃºblico SCANIA Component X. "
            "Las variables son anÃ³nimas y no describen operaciones mineras.",
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
            "aportar varios ejemplos por vehÃ­culo. La validaciÃ³n permanece separada.",
        ),
        _html_table(
            "SelecciÃ³n de hiperparÃ¡metros",
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
            "El menor costo medio decide; F1 macro, exactitud balanceada y tamaÃ±o "
            "del modelo resuelven empates.",
        ),
        _html_table(
            "ValidaciÃ³n cruzada por vehÃ­culo",
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
            "NingÃºn vehÃ­culo debe aparecer en entrenamiento y validaciÃ³n del mismo "
            "fold; los folds pertenecen a la selecciÃ³n, no a la prueba final.",
        ),
        _html_table(
            "Aprendizaje por rÃ©gimen",
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
            "Pruebas estadÃ­sticas pareadas",
            summary["learning_statistics"],
            statistical_fields,
            "Mejora positiva favorece al candidato. El intervalo bootstrap, Wilcoxon "
            "pareado, Holm y la biserial de rangos describen incertidumbre, "
            "significancia y tamaÃ±o del efecto.",
        ),
        _html_table(
            "Interpretabilidad por permutaciÃ³n",
            [
                row
                for row in interpretation["importance"]
                if row["dimension"] == "source_variable"
            ][:10],
            ("group", "features", "mean_cost_increase", "sd_cost_increase"),
            "Un aumento de costo al permutar indica asociaciÃ³n predictiva en "
            "validaciÃ³n; no identifica una pieza fÃ­sica ni demuestra causalidad.",
        ),
        _html_table(
            "PublicaciÃ³n federada por particiÃ³n",
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
            "Los nodos son particiones lÃ³gicas de un mismo dataset. La calidad "
            "mide cobertura de lecturas, no precisiÃ³n del modelo.",
        ),
        _html_table(
            "EvaluaciÃ³n Langflow y revisiÃ³n humana",
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
                {"measure": "PuntuaciÃ³n media", "value": review["overall_mean_score"]},
                *(
                    {"measure": metric, "value": value}
                    for metric, value in review["mean_scores"].items()
                ),
            ],
            ("measure", "value"),
            "Las puntuaciones son revisiÃ³n humana de explicaciones; no sustituyen "
            "las mÃ©tricas del modelo predictivo ni validan un despliegue minero.",
        ),
    ]
    sources = summary["sources"]["sha256"]
    sections.append(
        _html_table(
            "Trazabilidad",
            [{"artifact": name, "sha256": digest} for name, digest in sources.items()],
            ("artifact", "sha256"),
            "Las huellas identifican los archivos usados por este reporte. El modelo, "
            "el flujo y el prompt se vinculan a la publicaciÃ³n verificada.",
        )
    )
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MineTwin Â· EvaluaciÃ³n final</title><style>
body{{background:#f4f6f8;color:#1b2430;font:14px system-ui;margin:0;padding:32px}}main{{max-width:1500px;margin:auto}}
h1,h2{{font-weight:600}}p{{color:#5f6b7a}}section{{background:#fff;border:1px solid #d5dbe3;border-radius:12px;margin:20px 0;padding:20px;overflow:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d5dbe3;padding:9px;text-align:right;white-space:nowrap}}th{{color:#9a6200}}th:first-child,td:first-child{{text-align:left}}
</style></head><body><main><h1>MineTwin Â· EvaluaciÃ³n final</h1>
<p>SCANIA Component X v{html.escape(str(dataset['version']))}: {dataset['features']} variables. El coordinador recibiÃ³ {federation['coordinator_states']} estados publicados y {federation['coordinator_raw_features']} variables crudas.</p>
<p>Transferencia federada: {transfer['messages']} mensajes, {transfer['payload_bytes']} bytes medidos y estimados, y {transfer['raw_records']} registros crudos publicados.</p>
<p>ReducciÃ³n del tamaÃ±o de estados publicados frente a centralizar las variables: {federation['publication_reduction_ratio']:.2%}. Es una comparaciÃ³n del tamaÃ±o publicado, no una mediciÃ³n de red.</p>
<p>Modelo {html.escape(interpretation['model_id'])} Â· flujo {html.escape(flow['flow_id'])} versiÃ³n {html.escape(flow['flow_version'])} Â· modelo lingÃ¼Ã­stico {html.escape(flow['language_model_id'])}.</p>
<p>Estudio retrospectivo en camiones pesados. Las particiones federadas son experimentales sobre una sola fuente pÃºblica; no se validÃ³ en mina, entre contratistas ni en operaciÃ³n en tiempo real.</p>
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
        f"<p>CÃ³mo interpretarlo: {html.escape(explanation)}</p>"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="EvaluaciÃ³n final de MineTwin.")
    modes = parser.add_subparsers(dest="mode", required=True)
    learning = modes.add_parser("learning")
    learning.add_argument("--cache", type=Path, required=True)
    learning.add_argument("--output", type=Path, required=True)
    learning.add_argument("--seeds", type=_integers, default=tuple(range(100, 110)))
    learning.add_argument("--selection", type=Path)
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
