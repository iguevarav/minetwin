import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minetwin.learning.data import CachedSplit

REGIMES = ("local", "centralized", "fedavg", "fedprox")
METRICS = ("mean_cost", "macro_f1", "macro_pr_auc", "balanced_accuracy", "accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el reporte del artículo.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.results)
    output = args.output or root / "10_reporte" / "articulo"
    print(
        json.dumps(
            build_report(root, output, overwrite=args.overwrite),
            ensure_ascii=False,
            indent=2,
        )
    )


def build_report(root: Path, output: Path, overwrite: bool = False) -> dict:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=overwrite)
    paths = _paths(root)
    eda = _json(paths["selection"] / "eda.json")
    diagnostics = _json(paths["diagnostics"] / "diagnostics.json")
    partitions = _partition_rows(_json(paths["data"] / "summary.json"))
    classes, nodes = _eda_rows(eda, paths["cache"])
    seed_rows, local_rows = _seed_rows(paths["study"])
    confusion, class_metrics = _diagnostic_rows(diagnostics)
    tables = {
        "eda_particiones.csv": partitions,
        "eda_clases.csv": classes,
        "eda_nodos.csv": nodes,
        "eda_eta_cuadrado.csv": _eta_rows(eda),
        "seleccion_modelo.csv": _csv(paths["selection"] / "candidates.csv"),
        "seleccion_variante.csv": _csv(paths["variants"] / "variant_summary.csv"),
        "referencias_cv.csv": _csv(paths["variants"] / "reference_summary.csv"),
        "baselines_validation.csv": _baseline_rows(diagnostics),
        "multisemilla_resumen.csv": _study_summary(seed_rows),
        "multisemilla_semillas.csv": seed_rows,
        "local_por_nodo.csv": local_rows,
        "estadistica_primaria.csv": _csv(paths["study"] / "primary_statistics.csv"),
        "estadistica_secundaria.csv": _csv(paths["study"] / "secondary_statistics.csv"),
        "confusion.csv": confusion,
        "metricas_por_clase.csv": class_metrics,
        "perdidas_cliente_ronda.csv": _csv(paths["study"] / "client_round_losses.csv"),
        "importancia_permutacion.csv": _importance_rows(paths["interpretability"]),
    }
    for name, rows in tables.items():
        _write_csv(output / name, rows)
    provenance = _provenance(paths)
    (output / "procedencia.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figures = _figures(output, tables)
    _markdown(output / "resultados.md", tables, provenance, figures)
    return {"output": str(output), "tables": len(tables) + 2, "figures": len(figures)}


def _paths(root: Path) -> dict[str, Path]:
    return {
        "data": root / "01_datos",
        "cache": root / "03_cache",
        "selection": root / "04_seleccion_modelo",
        "variants": root / "05_seleccion_variante",
        "study": root / "06_aprendizaje",
        "diagnostics": root / "08_diagnostico",
        "interpretability": root / "09_interpretabilidad",
    }


def _partition_rows(summary: dict) -> list[dict]:
    return [
        {
            "partition": row["split"],
            "vehicles": row["vehicles"],
            "readouts": row["readouts"],
            "variables": row["features"],
            "missing_values": row["missing_values"],
            "source_file": "results/01_datos/summary.json",
        }
        for row in summary["splits"]
    ]


def _eda_rows(eda: dict, cache: Path) -> tuple[list[dict], list[dict]]:
    classes, nodes = [], []
    for split in ("train", "validation"):
        source = next(row for row in eda["splits"] if row["split"] == split)
        cached = CachedSplit.load(cache / f"{split}.npz")
        classes.extend(
            {
                "partition": split,
                "scope": "global",
                "node": "",
                "class": label,
                "examples": examples,
                "source_file": "results/04_seleccion_modelo/eda.json",
            }
            for label, examples in source["classes"].items()
        )
        classes.extend(
            {
                "partition": split,
                "scope": "node",
                "node": row["node"],
                "class": row["class"],
                "examples": row["examples"],
                "source_file": "results/04_seleccion_modelo/eda.json",
            }
            for row in source["node_classes"]
        )
        nodes.extend(
            {
                "partition": split,
                "node": node,
                "vehicles": len(np.unique(cached.vehicle_ids[cached.nodes == node])),
                "examples": examples,
                "source_file": f"results/03_cache/{split}.npz",
            }
            for node, examples in source["nodes"].items()
        )
    return classes, nodes


def _eta_rows(eda: dict) -> list[dict]:
    return [
        {**row, "rank": rank, "source_file": "results/04_seleccion_modelo/eda.json"}
        for rank, row in enumerate(eda["top_class_separation_features"], 1)
    ]


def _baseline_rows(diagnostics: dict) -> list[dict]:
    return [
        {
            "baseline": name,
            **_metrics(report["global"]),
            "source_file": "results/08_diagnostico/diagnostics.json",
        }
        for name, report in diagnostics["baselines"].items()
    ]


def _seed_rows(study: Path) -> tuple[list[dict], list[dict]]:
    rows, local = [], []
    for seed in range(100, 110):
        report = _json(study / f"seed_{seed}" / "metrics.json")
        source = f"results/06_aprendizaje/seed_{seed}/metrics.json"
        for regime in REGIMES:
            rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    **_metrics(report["results"][regime]["global"]),
                    "source_file": source,
                }
            )
        for node, metrics in report["results"]["local"]["nodes"].items():
            local.append(
                {"seed": seed, "node": node, **_metrics(metrics), "source_file": source}
            )
    return rows, local


def _metrics(metrics: dict) -> dict:
    return {metric: metrics[metric] for metric in METRICS} | {
        "examples": metrics["examples"]
    }


def _study_summary(rows: list[dict]) -> list[dict]:
    generator, output = np.random.default_rng(42), []
    for regime in REGIMES:
        selected = [row for row in rows if row["regime"] == regime]
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            resampled = np.mean(generator.choice(values, (10_000, len(values))), axis=1)
            output.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "ci_95_lower": float(np.quantile(resampled, 0.025)),
                    "ci_95_upper": float(np.quantile(resampled, 0.975)),
                    "n": len(values),
                    "source_file": "results/06_aprendizaje/seed_*/metrics.json",
                }
            )
    return output


def _diagnostic_rows(diagnostics: dict) -> tuple[list[dict], list[dict]]:
    confusion, metrics = [], []
    for regime in REGIMES:
        global_metrics = diagnostics["results"][regime]["global"]
        for actual, row in enumerate(global_metrics["confusion_matrix"]):
            confusion.extend(
                {
                    "regime": regime,
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "examples": examples,
                    "source_file": "results/08_diagnostico/diagnostics.json",
                }
                for predicted, examples in enumerate(row)
            )
        metrics.extend(
            {
                "regime": regime,
                "class": label,
                "precision": global_metrics["precision"][label],
                "recall": global_metrics["recall"][label],
                "f1": global_metrics["f1"][label],
                "support": global_metrics["support"][label],
                "source_file": "results/08_diagnostico/diagnostics.json",
            }
            for label in range(5)
        )
    return confusion, metrics


def _importance_rows(path: Path) -> list[dict]:
    rows = sorted(
        _csv(path / "permutation_importance.csv"),
        key=lambda row: float(row["mean_cost_increase"]),
        reverse=True,
    )
    return [
        {
            **row,
            "rank": rank,
            "source_file": "results/09_interpretabilidad/permutation_importance.csv",
        }
        for rank, row in enumerate(rows[:15], 1)
    ]


def _provenance(paths: dict[str, Path]) -> dict:
    manifest = _json(paths["study"] / "manifest.json")
    selected = _json(paths["variants"] / "selection.json")
    files = {
        "cache_train": paths["cache"] / "train.npz",
        "cache_validation": paths["cache"] / "validation.npz",
        "cache_metadata": paths["cache"] / "metadata.json",
        "seed_100_metrics": paths["study"] / "seed_100" / "metrics.json",
        "diagnostics": paths["diagnostics"] / "diagnostics.json",
        "interpretability": paths["interpretability"] / "permutation_importance.csv",
    }
    hardware = {
        key: value for key, value in manifest["runtime"].items() if key != "commit"
    }
    return {
        "commit": manifest["runtime"]["commit"],
        "hardware": hardware,
        "versions": {"python": sys.version, "numpy": np.__version__},
        "timing_seconds": manifest["elapsed_seconds"],
        "configuration": selected["winner_config"],
        "hashes": {name: _sha256(path) for name, path in files.items()},
    }


def _figures(output: Path, tables: dict[str, list[dict]]) -> list[str]:
    classes, summary = tables["eda_clases.csv"], tables["multisemilla_resumen.csv"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for split in ("train", "validation"):
        rows = [
            row
            for row in classes
            if row["partition"] == split and row["scope"] == "global"
        ]
        axis.plot(
            [row["class"] for row in rows],
            [row["examples"] for row in rows],
            marker="o",
            label=split,
        )
    axis.set(xlabel="Clase", ylabel="Ejemplos")
    axis.legend()
    class_path = _save(figure, output / "distribucion_clases.png")
    rows = [row for row in summary if row["metric"] == "mean_cost"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [row["regime"] for row in rows],
        [row["mean"] for row in rows],
        yerr=[row["sd"] for row in rows],
        capsize=4,
    )
    axis.set_ylabel("Costo medio SCANIA")
    cost_path = _save(figure, output / "costo_por_regimen.png")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for regime in ("fedavg", "fedprox"):
        for node in ("alpha", "beta", "gamma"):
            rows = [
                row
                for row in tables["perdidas_cliente_ronda.csv"]
                if row["regime"] == regime and row["node"] == node
            ]
            rounds = sorted({int(row["round"]) for row in rows})
            values = [
                np.mean(
                    [
                        float(row["training_loss"])
                        for row in rows
                        if int(row["round"]) == round_
                    ]
                )
                for round_ in rounds
            ]
            axis.plot(rounds, values, label=f"{regime}-{node}")
    axis.set(xlabel="Ronda", ylabel="Pérdida de cliente")
    axis.legend(ncol=2)
    loss_path = _save(figure, output / "perdidas_federadas.png")
    figure, axes = plt.subplots(2, 2, figsize=(8, 7))
    for axis, regime in zip(axes.flat, REGIMES, strict=True):
        values = np.zeros((5, 5), dtype=int)
        for row in tables["confusion.csv"]:
            if row["regime"] == regime:
                values[int(row["actual_class"]), int(row["predicted_class"])] = int(
                    row["examples"]
                )
        image = axis.imshow(values, cmap="Blues")
        axis.set(title=regime, xlabel="Predicha", ylabel="Real")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    confusion_path = _save(figure, output / "matrices_confusion.png")
    rows = tables["importancia_permutacion.csv"]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(
        [row["group"] for row in rows][::-1],
        [float(row["mean_cost_increase"]) for row in rows][::-1],
        xerr=[float(row["sd_cost_increase"]) for row in rows][::-1],
        capsize=3,
    )
    axis.set_xlabel("Aumento de costo por permutación")
    importance_path = _save(figure, output / "importancia_permutacion.png")
    rows = tables["seleccion_variante.csv"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [row["variant"] for row in rows],
        [float(row["mean_cost_mean"]) for row in rows],
        yerr=[float(row["mean_cost_sd"]) for row in rows],
        capsize=4,
    )
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylabel("Costo medio SCANIA en CV")
    variant_path = _save(figure, output / "seleccion_variante.png")
    return [
        str(path)
        for path in (
            class_path,
            cost_path,
            loss_path,
            confusion_path,
            importance_path,
            variant_path,
        )
    ]


def _save(figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _markdown(
    path: Path, tables: dict[str, list[dict]], provenance: dict, figures: list[str]
) -> None:
    lines = ["# Resultados reproducibles de MineTwin", ""]
    for name, rows in tables.items():
        lines.extend(
            (
                f"## {name.removesuffix('.csv').replace('_', ' ').title()}",
                "",
                _table(rows),
                "",
            )
        )
    lines.extend(
        (
            "## Procedencia",
            "",
            "```json",
            json.dumps(provenance, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Figuras PNG a 300 dpi",
            "",
            *[f"- {Path(figure).name}" for figure in figures],
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _table(rows: list[dict]) -> str:
    if not rows:
        return "NO DISPONIBLE"
    fields = list(rows[0])
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    return "\n".join(
        (
            header,
            divider,
            *[
                "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
                for row in rows
            ],
        )
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
