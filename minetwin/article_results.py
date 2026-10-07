import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minetwin.data.scania.loader import MISSING_VALUES, READOUT_FILES
from minetwin.stats import bootstrap_mean_interval, holm_adjust, wilcoxon_signed_rank

METRICS = (
    "mean_cost",
    "macro_f1",
    "balanced_accuracy",
    "macro_pr_auc",
    "accuracy",
)
COMPARISONS = (
    ("local", "fedavg"),
    ("local", "fedprox"),
    ("centralized", "fedavg"),
    ("centralized", "fedprox"),
    ("fedavg", "fedprox"),
)
REGIMES = ("local", "centralized", "fedavg", "fedprox")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolida resultados reproducibles para el articulo SCANIA."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--learning", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--interpretability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_article_results(**vars(args))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_article_results(
    dataset: Path,
    phase1: Path,
    cache: Path,
    selection: Path,
    learning: Path,
    diagnostics: Path,
    interpretability: Path,
    output: Path,
) -> dict:
    paths = _paths(
        dataset,
        phase1,
        cache,
        selection,
        learning,
        diagnostics,
        interpretability,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    phase_one = _read_json(paths["phase1"] / "summary.json")
    cache_metadata = _read_json(paths["cache"] / "metadata.json")
    eda = _read_json(paths["selection"] / "eda.json")
    learning_rows = _read_csv(paths["learning"] / "metrics.csv")
    diagnostics_data = _read_json(paths["diagnostics"] / "diagnostics.json")
    importance_rows = _read_csv(
        paths["interpretability"] / "permutation_importance.csv"
    )
    sources = _source_names(paths)
    tables = {}
    tables["eda_partitions"] = _write_csv(
        output / "eda_partitions.csv",
        _eda_partitions(phase_one, eda, sources),
    )
    tables["eda_classes"] = _write_csv(
        output / "eda_classes.csv", _eda_classes(eda, sources["eda"])
    )
    tables["eda_nodes"] = _write_csv(
        output / "eda_nodes.csv", _node_distribution(paths["cache"], sources["cache"])
    )
    tables["eda_missingness"] = _write_csv(
        output / "eda_missingness.csv",
        _missingness(paths["dataset"], sources["dataset"]),
    )
    tables["eda_eta_squared"] = _write_csv(
        output / "eda_eta_squared.csv", _eta_squared(eda, sources["eda"])
    )
    tables["model_features"] = _write_csv(
        output / "model_features.csv",
        [
            {
                "model_features": cache_metadata["feature_count"],
                "source_file": sources["cache"],
            }
        ],
    )
    candidates = _with_source(
        _read_csv(paths["selection"] / "candidates.csv"), sources["candidates"]
    )
    folds = _with_source(
        _read_csv(paths["selection"] / "fold_metrics.csv"), sources["folds"]
    )
    tables["selection_candidates"] = _write_csv(
        output / "selection_candidates.csv", candidates
    )
    tables["selection_folds"] = _write_csv(output / "selection_folds.csv", folds)
    metrics = _with_source(learning_rows, sources["learning_metrics"])
    tables["multiseed_metrics"] = _write_csv(output / "multiseed_metrics.csv", metrics)
    multiseed_summary = _multiseed_summary(learning_rows, sources["learning_metrics"])
    tables["multiseed_summary"] = _write_csv(
        output / "multiseed_summary.csv", multiseed_summary
    )
    node_metrics = _node_metrics(paths["learning"], sources["seed_metrics"])
    tables["local_node_metrics"] = _write_csv(
        output / "local_node_metrics.csv", node_metrics
    )
    tables["local_node_summary"] = _write_csv(
        output / "local_node_summary.csv",
        _group_metric_summary(node_metrics, ("node",), "source_file"),
    )
    statistics = _statistics(learning_rows, sources["learning_metrics"])
    tables["statistics"] = _write_csv(output / "statistics.csv", statistics)
    baselines = _baselines(diagnostics_data, sources["diagnostics"])
    tables["baselines"] = _write_csv(output / "baselines.csv", baselines)
    confusion, class_metrics = _diagnostic_tables(
        diagnostics_data, sources["diagnostics"]
    )
    tables["confusion_matrices"] = _write_csv(
        output / "confusion_matrices.csv", confusion
    )
    tables["class_metrics"] = _write_csv(output / "class_metrics.csv", class_metrics)
    rounds, local_losses = _training_curves(paths["learning"], sources["seed_metrics"])
    tables["federated_rounds"] = _write_csv(output / "federated_rounds.csv", rounds)
    tables["federated_round_summary"] = _write_csv(
        output / "federated_round_summary.csv",
        _curve_summary(rounds, ("regime", "round")),
    )
    tables["local_node_losses"] = _write_csv(
        output / "local_node_losses.csv", local_losses
    )
    tables["local_node_loss_summary"] = _write_csv(
        output / "local_node_loss_summary.csv",
        _curve_summary(local_losses, ("node", "epoch")),
    )
    importance = _importance(importance_rows, sources["importance"])
    tables["permutation_importance"] = _write_csv(
        output / "permutation_importance_top15.csv", importance
    )
    provenance = _provenance(paths, diagnostics_data, sources)
    tables["provenance"] = _write_csv(output / "provenance.csv", provenance)
    figures = _figures(
        output,
        eda,
        multiseed_summary,
        rounds,
        confusion,
        importance,
    )
    manifest = {
        "bootstrap": {"samples": 10_000, "confidence": 0.95},
        "validation_split": "validation",
        "sources": sources,
        "tables": tables,
        "figures": figures,
        "runtime": _runtime(),
    }
    (output / "article_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(
        output / "resultados.md",
        phase_one,
        eda,
        candidates,
        multiseed_summary,
        statistics,
        baselines,
        importance,
        sources,
    )
    return {"output": str(output), "tables": len(tables), "figures": len(figures)}


def _paths(
    dataset: Path,
    phase1: Path,
    cache: Path,
    selection: Path,
    learning: Path,
    diagnostics: Path,
    interpretability: Path,
) -> dict[str, Path]:
    paths = {
        "dataset": Path(dataset),
        "phase1": Path(phase1),
        "cache": Path(cache),
        "selection": Path(selection),
        "learning": Path(learning),
        "diagnostics": Path(diagnostics),
        "interpretability": Path(interpretability),
    }
    required = (
        "phase1/summary.json",
        "cache/metadata.json",
        "selection/eda.json",
        "selection/candidates.csv",
        "selection/fold_metrics.csv",
        "learning/metrics.csv",
        "diagnostics/diagnostics.json",
        "interpretability/permutation_importance.csv",
    )
    for item in required:
        name, relative = item.split("/", 1)
        if not (paths[name] / relative).is_file():
            raise ValueError(f"Falta el artefacto requerido: {paths[name] / relative}")
    return paths


def _source_names(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "dataset": str(paths["dataset"]),
        "phase1": str(paths["phase1"] / "summary.json"),
        "cache": str(paths["cache"] / "metadata.json"),
        "eda": str(paths["selection"] / "eda.json"),
        "candidates": str(paths["selection"] / "candidates.csv"),
        "folds": str(paths["selection"] / "fold_metrics.csv"),
        "learning_metrics": str(paths["learning"] / "metrics.csv"),
        "seed_metrics": str(paths["learning"] / "seed_*/metrics.json"),
        "diagnostics": str(paths["diagnostics"] / "diagnostics.json"),
        "importance": str(paths["interpretability"] / "permutation_importance.csv"),
    }


def _eda_partitions(phase_one: dict, eda: dict, sources: dict[str, str]) -> list[dict]:
    prepared = {row["split"]: row for row in eda["splits"]}
    rows = []
    for row in phase_one["splits"]:
        selected = prepared[row["split"]]
        rows.append(
            {
                **row,
                "model_examples": selected["examples"],
                "model_vehicles": selected["vehicles"],
                "raw_missing_rate": eda["raw_missing_rate"][row["split"]],
                "source_file": f"{sources['phase1']}; {sources['eda']}",
            }
        )
    return rows


def _eda_classes(eda: dict, source: str) -> list[dict]:
    rows = []
    for split in eda["splits"]:
        for label, examples in split["classes"].items():
            rows.append(
                {
                    "split": split["split"],
                    "node": "global",
                    "class": label,
                    "examples": examples,
                    "source_file": source,
                }
            )
        rows.extend(
            {"split": split["split"], **row, "source_file": source}
            for row in split["node_classes"]
        )
    return rows


def _node_distribution(cache: Path, source: str) -> list[dict]:
    rows = []
    for path in sorted(cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as values:
            nodes = values["nodes"].astype(str)
            vehicles = values["vehicle_ids"].astype(str)
        for node in sorted(set(nodes)):
            selected = nodes == node
            rows.append(
                {
                    "split": path.stem,
                    "node": node,
                    "examples": int(np.sum(selected)),
                    "vehicles": len(set(vehicles[selected])),
                    "source_file": source,
                }
            )
    return rows


def _missingness(dataset: Path, source: str) -> list[dict]:
    rows = []
    for split, filename in READOUT_FILES.items():
        path = dataset / filename
        missing = defaultdict(int)
        total = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            features = tuple(reader.fieldnames or ())[2:]
            for row in reader:
                total += 1
                for feature in features:
                    if row[feature].strip().lower() in MISSING_VALUES:
                        missing[feature] += 1
        rows.extend(
            {
                "split": split.value,
                "variable": feature,
                "missing_values": missing[feature],
                "readouts": total,
                "missing_rate": missing[feature] / total,
                "source_file": source,
            }
            for feature in features
        )
    return rows


def _eta_squared(eda: dict, source: str) -> list[dict]:
    return [
        {"rank": index, **row, "source_file": source}
        for index, row in enumerate(eda["top_class_separation_features"], 1)
    ]


def _multiseed_summary(rows: list[dict], source: str) -> list[dict]:
    return _group_metric_summary(rows, ("regime",), "source_file", source)


def _group_metric_summary(
    rows: list[dict],
    grouping: tuple[str, ...],
    source_key: str,
    source: str | None = None,
) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in grouping)].append(row)
    result = []
    for key, selected in sorted(groups.items()):
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            interval = bootstrap_mean_interval(
                values, samples=10_000, seed=_seed(*key, metric)
            )
            result.append(
                {
                    **dict(zip(grouping, key, strict=True)),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "ci_95_lower": interval[0],
                    "ci_95_upper": interval[1],
                    "n": len(values),
                    source_key: source or selected[0][source_key],
                }
            )
    return result


def _node_metrics(learning: Path, source: str) -> list[dict]:
    rows = []
    for path in sorted(learning.glob("seed_*/metrics.json")):
        report = _read_json(path)
        seed = report["config"]["seed"]
        for node, metrics in report["results"]["local"]["nodes"].items():
            rows.append(
                {
                    "seed": seed,
                    "node": node,
                    **{metric: metrics[metric] for metric in METRICS},
                    "examples": metrics["examples"],
                    "source_file": str(path),
                }
            )
    return rows


def _statistics(rows: list[dict], source: str) -> list[dict]:
    indexed = {(row["seed"], row["regime"]): row for row in rows}
    result = []
    for reference, candidate in COMPARISONS:
        shared = sorted(
            seed
            for seed, regime in indexed
            if regime == reference
            if (seed, candidate) in indexed
        )
        for metric in METRICS:
            differences = np.asarray(
                [
                    float(indexed[(seed, candidate)][metric])
                    - float(indexed[(seed, reference)][metric])
                    for seed in shared
                ]
            )
            test = wilcoxon_signed_rank(differences)
            interval = bootstrap_mean_interval(
                differences,
                samples=10_000,
                seed=_seed(reference, candidate, metric),
            )
            result.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "mean_difference": float(np.mean(differences)),
                    "difference_ci_95_lower": interval[0],
                    "difference_ci_95_upper": interval[1],
                    "wilcoxon_statistic": test.statistic,
                    "p_value": test.p_value,
                    "rank_biserial": test.rank_biserial,
                    "nonzero_pairs": test.pairs,
                    "pairs": len(shared),
                    "test_method": test.method,
                    "source_file": source,
                }
            )
    adjusted = holm_adjust(row["p_value"] for row in result)
    for row, p_value in zip(result, adjusted, strict=True):
        row["p_value_holm"] = p_value
    return result


def _baselines(diagnostics: dict, source: str) -> list[dict]:
    rows = []
    labels = {"always_0": "always_class_0_train_majority", "always_4": "always_class_4"}
    for name, report in diagnostics["baselines"].items():
        metrics = report["global"]
        rows.append(
            {
                "baseline": labels.get(name, name),
                **{metric: metrics[metric] for metric in METRICS},
                "examples": metrics["examples"],
                "source_file": source,
            }
        )
    return rows


def _diagnostic_tables(diagnostics: dict, source: str) -> tuple[list[dict], list[dict]]:
    confusion = []
    class_metrics = []
    for regime, report in diagnostics["results"].items():
        metrics = report["global"]
        for actual, values in enumerate(metrics["confusion_matrix"]):
            for predicted, examples in enumerate(values):
                confusion.append(
                    {
                        "regime": regime,
                        "actual_class": actual,
                        "predicted_class": predicted,
                        "examples": examples,
                        "source_file": source,
                    }
                )
        for label in range(5):
            class_metrics.append(
                {
                    "regime": regime,
                    "class": label,
                    "precision": metrics["precision"][label],
                    "recall": metrics["recall"][label],
                    "f1": metrics["f1"][label],
                    "support": metrics["support"][label],
                    "source_file": source,
                }
            )
    return confusion, class_metrics


def _training_curves(learning: Path, source: str) -> tuple[list[dict], list[dict]]:
    rounds = []
    local_losses = []
    for path in sorted(learning.glob("seed_*/metrics.json")):
        report = _read_json(path)
        seed = report["config"]["seed"]
        history = report["training_history"]
        for regime in ("fedavg", "fedprox"):
            rounds.extend(
                {
                    "seed": seed,
                    "regime": regime,
                    **row,
                    "source_file": str(path),
                }
                for row in history[regime]
            )
        for node, values in history["local"].items():
            local_losses.extend(
                {
                    "seed": seed,
                    "node": node,
                    **row,
                    "source_file": str(path),
                }
                for row in values
            )
    return rounds, local_losses


def _curve_summary(rows: list[dict], grouping: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in grouping)].append(
            float(row["training_loss"])
        )
    return [
        {
            **dict(zip(grouping, key, strict=True)),
            "mean_training_loss": float(np.mean(values)),
            "sd_training_loss": float(np.std(values, ddof=1)),
            "n": len(values),
            "source_file": "results/final_learning_verified/seed_*/metrics.json",
        }
        for key, values in sorted(groups.items())
    ]


def _importance(rows: list[dict], source: str) -> list[dict]:
    selected = [row for row in rows if row["dimension"] == "source_variable"]
    selected.sort(key=lambda row: float(row["mean_cost_increase"]), reverse=True)
    return [
        {"rank": index, **row, "source_file": source}
        for index, row in enumerate(selected[:15], 1)
    ]


def _provenance(paths: dict[str, Path], diagnostics: dict, sources: dict) -> list[dict]:
    rows = [
        _hash_row("dataset_manifest", paths["phase1"] / "dataset_manifest.json"),
        _hash_row("cache_metadata", paths["cache"] / "metadata.json"),
        _hash_row("train_cache", paths["cache"] / "train.npz"),
        _hash_row("validation_cache", paths["cache"] / "validation.npz"),
        _hash_row("selection", paths["selection"] / "selection.json"),
        _hash_row("learning_summary", paths["learning"] / "summary.json"),
        _hash_row("diagnostics", paths["diagnostics"] / "diagnostics.json"),
        _hash_row(
            "permutation_importance",
            paths["interpretability"] / "permutation_importance.csv",
        ),
    ]
    for name, digest in diagnostics["sha256"]["models"].items():
        rows.append(
            {
                "artifact": f"model_{name}",
                "sha256": digest,
                "source_file": sources["diagnostics"],
            }
        )
    rows.append(
        {
            "artifact": "scaler",
            "sha256": diagnostics["sha256"]["scaler"],
            "source_file": sources["diagnostics"],
        }
    )
    return rows


def _figures(
    output: Path,
    eda: dict,
    summary: list[dict],
    rounds: list[dict],
    confusion: list[dict],
    importance: list[dict],
) -> list[str]:
    figures = []
    figures.append(_plot_classes(output, eda))
    figures.append(_plot_cost(output, summary))
    figures.append(_plot_rounds(output, rounds))
    figures.append(_plot_confusion(output, confusion))
    figures.append(_plot_importance(output, importance))
    return figures


def _plot_classes(output: Path, eda: dict) -> str:
    splits = [item for item in eda["splits"] if item["classes"]]
    labels = np.arange(5)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    for index, item in enumerate(splits):
        values = [item["classes"].get(str(label), 0) for label in labels]
        axis.bar(labels + (index - 0.5) * width, values, width, label=item["split"])
    axis.set_xticks(labels)
    axis.set_xlabel("Clase")
    axis.set_ylabel("Ejemplos")
    axis.legend()
    return _save_figure(figure, output / "class_distribution.png")


def _plot_cost(output: Path, summary: list[dict]) -> str:
    rows = [row for row in summary if row["metric"] == "mean_cost"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    names = [row["regime"] for row in rows]
    means = [row["mean"] for row in rows]
    errors = [row["sd"] for row in rows]
    axis.bar(names, means, yerr=errors, capsize=4)
    axis.set_ylabel("Costo medio SCANIA")
    return _save_figure(figure, output / "cost_by_regime.png")


def _plot_rounds(output: Path, rounds: list[dict]) -> str:
    groups = defaultdict(list)
    for row in rounds:
        groups[(row["regime"], int(row["round"]))].append(float(row["training_loss"]))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for regime in ("fedavg", "fedprox"):
        selected = sorted(
            (round_, values)
            for (name, round_), values in groups.items()
            if name == regime
        )
        axis.plot(
            [item[0] for item in selected],
            [float(np.mean(item[1])) for item in selected],
            marker="o",
            label=regime,
        )
    axis.set_xlabel("Ronda")
    axis.set_ylabel("Pérdida de entrenamiento")
    axis.legend()
    return _save_figure(figure, output / "federated_rounds.png")


def _plot_confusion(output: Path, rows: list[dict]) -> str:
    figure, axes = plt.subplots(2, 2, figsize=(8, 7))
    for axis, regime in zip(axes.flat, REGIMES, strict=True):
        matrix = np.zeros((5, 5), dtype=int)
        for row in rows:
            if row["regime"] == regime:
                matrix[int(row["actual_class"]), int(row["predicted_class"])] = int(
                    row["examples"]
                )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(regime)
        axis.set_xlabel("Predicha")
        axis.set_ylabel("Real")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save_figure(figure, output / "confusion_matrices.png")


def _plot_importance(output: Path, rows: list[dict]) -> str:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    labels = [row["group"] for row in reversed(rows)]
    means = [float(row["mean_cost_increase"]) for row in reversed(rows)]
    errors = [float(row["sd_cost_increase"]) for row in reversed(rows)]
    axis.barh(labels, means, xerr=errors, capsize=3)
    axis.set_xlabel("Incremento del costo medio")
    return _save_figure(figure, output / "permutation_importance.png")


def _save_figure(figure, path: Path) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return str(path)


def _write_markdown(
    path: Path,
    phase_one: dict,
    eda: dict,
    candidates: list[dict],
    summary: list[dict],
    statistics: list[dict],
    baselines: list[dict],
    importance: list[dict],
    sources: dict[str, str],
) -> None:
    partitions = [
        {
            "partición": row["split"],
            "vehículos": row["vehicles"],
            "lecturas": row["readouts"],
            "variables": row["features"],
            "ausencias": row["missing_values"],
        }
        for row in phase_one["splits"]
    ]
    lines = [
        "# Resultados reproducibles de MineTwin",
        "",
        (
            "Todas las métricas supervisadas usan la partición `validation`. "
            "El conjunto `test` no tiene etiquetas y no se usa para estas métricas."
        ),
        "",
        "## EDA",
        "",
        _markdown_table(partitions),
        "",
        f"Fuente: `{sources['phase1']}`.",
        "",
        (
            "La distribución de clases global y por nodo está en "
            "`eda_classes.csv`; las ausencias por variable y partición están en "
            "`eda_missingness.csv`; los vehículos y ejemplos por nodo están en "
            "`eda_nodes.csv`; las 12 variables por eta cuadrado están en "
            "`eda_eta_squared.csv`."
        ),
        "",
        "## Selección del modelo",
        "",
        _markdown_table(candidates),
        "",
        (
            f"Fuente: `{sources['candidates']}`. Las 20 métricas por fold están en "
            "`selection_folds.csv`."
        ),
        "",
        "## Estudio multisemilla",
        "",
        _markdown_table(summary),
        "",
        (
            "Fuente: `results/final_learning_verified/metrics.csv`. Cada intervalo "
            "es bootstrap percentil 95 % con 10 000 remuestras y n=10 semillas."
        ),
        "",
        (
            "Las métricas por semilla y régimen están en `multiseed_metrics.csv`; "
            "las métricas locales por nodo y sus agregados están en "
            "`local_node_metrics.csv` y `local_node_summary.csv`."
        ),
        "",
        "## Baselines",
        "",
        _markdown_table(baselines),
        "",
        (
            f"Fuente: `{sources['diagnostics']}`. La clase 0 es la mayoritaria en "
            "entrenamiento; por ello `always_class_0_train_majority` cubre ambos "
            "baselines solicitados."
        ),
        "",
        "## Estadística pareada",
        "",
        _markdown_table(statistics),
        "",
        (
            "Fuente: `results/final_learning_verified/metrics.csv`. `mean_difference` "
            "es candidato menos referencia; para costo, un valor negativo favorece al "
            "candidato. Holm ajusta las 25 pruebas."
        ),
        "",
        "## Federación e interpretabilidad",
        "",
        (
            "Las curvas por ronda y las pérdidas locales por nodo están en "
            "`federated_rounds.csv`, `federated_round_summary.csv`, "
            "`local_node_losses.csv` y `local_node_loss_summary.csv`. La pérdida por "
            "nodo en cada ronda FedAvg/FedProx es **NO DISPONIBLE**: los artefactos "
            "originales solo guardan la pérdida global por ronda. Para generarla, se "
            "debe registrar la pérdida cliente-ronda durante `train_learning_regimes` "
            "y ejecutar de nuevo el estudio multisemilla."
        ),
        "",
        _markdown_table(importance),
        "",
        (
            f"Fuente: `{sources['importance']}`. Corresponde a FedProx, semilla 100, "
            "validación y tres permutaciones por variable."
        ),
        "",
        "## Proveniencia y límites",
        "",
        (
            "Los hashes de caché, escalador y modelos están en `provenance.csv`. "
            "La configuración está en los artefactos `selection.json`, "
            "`seed_*/metrics.json` y `diagnostics.json` de las rutas indicadas."
        ),
        "",
        (
            "El commit, hardware y tiempo de cómputo de los entrenamientos históricos "
            "son **NO DISPONIBLE** en los artefactos originales. Para registrarlos en "
            "una repetición: `Get-CimInstance Win32_Processor,Win32_ComputerSystem | "
            "ConvertTo-Json > results\\hardware.json`; `Measure-Command { python -m "
            "minetwin.study learning --cache results\\phase2_cache --selection "
            "results\\final_selection --output results\\learning_with_timing }`; "
            "`git rev-parse HEAD > results\\commit.txt`."
        ),
        "",
        (
            "Las figuras PNG a 300 dpi son `class_distribution.png`, "
            "`cost_by_regime.png`, `federated_rounds.png`, "
            "`confusion_matrices.png` y `permutation_importance.png`."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "NO DISPONIBLE"
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    values = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *values))


def _runtime() -> dict:
    packages = {}
    for package in ("numpy", "matplotlib", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not_installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _hash_row(name: str, path: Path) -> dict:
    return {
        "artifact": name,
        "sha256": _sha256(path),
        "source_file": str(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(*parts: object) -> int:
    text = "|".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _with_source(rows: list[dict], source: str) -> list[dict]:
    return [{**row, "source_file": source} for row in rows]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> str:
    if not rows:
        raise ValueError(f"No hay filas para {path.name}.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


if __name__ == "__main__":
    main()
