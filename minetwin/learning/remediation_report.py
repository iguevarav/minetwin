import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el informe de remediación.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--cv", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_report(
    cache: Path, diagnostics: Path, cv: Path, study: Path, output: Path
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    diagnostics_data = _read_json(Path(diagnostics) / "diagnostics.json")
    cv = Path(cv)
    study = Path(study)
    pre_matrix, pre_distribution = _pre_diagnostic(diagnostics_data)
    distributions = _class_distributions(Path(cache))
    variants = _read_csv(cv / "variant_summary.csv")
    baselines = _read_csv(cv / "baseline_summary.csv")
    study_rows = _read_csv(study / "metrics.csv")
    primary = _read_csv(study / "primary_statistics.csv")
    secondary = _read_csv(study / "secondary_statistics.csv")
    client_losses = _read_csv(study / "client_round_losses.csv")
    final_summary = _study_summary(study_rows)
    sources = {
        "diagnostics": str(Path(diagnostics) / "diagnostics.json"),
        "cache": str(Path(cache)),
        "variants": str(cv / "variant_summary.csv"),
        "baselines": str(cv / "baseline_summary.csv"),
        "study": str(study / "metrics.csv"),
        "primary": str(study / "primary_statistics.csv"),
        "secondary": str(study / "secondary_statistics.csv"),
        "client_losses": str(study / "client_round_losses.csv"),
    }
    tables = {
        "pre_confusion": _write_csv(output / "pre_confusion.csv", pre_matrix),
        "pre_predicted_distribution": _write_csv(
            output / "pre_predicted_distribution.csv", pre_distribution
        ),
        "class_distributions": _write_csv(
            output / "train_validation_class_distribution.csv", distributions
        ),
        "variant_folds": _copy_csv(cv / "variant_folds.csv", output),
        "variant_summary": _copy_csv(cv / "variant_summary.csv", output),
        "baseline_folds": _copy_csv(cv / "baseline_folds.csv", output),
        "baseline_summary": _copy_csv(cv / "baseline_summary.csv", output),
        "multiseed_metrics": _write_csv(output / "multiseed_metrics.csv", study_rows),
        "multiseed_summary": _write_csv(
            output / "multiseed_summary.csv", final_summary
        ),
        "primary_statistics": _write_csv(output / "primary_statistics.csv", primary),
        "secondary_statistics": _write_csv(
            output / "secondary_statistics.csv", secondary
        ),
        "client_round_losses": _write_csv(
            output / "client_round_losses.csv", client_losses
        ),
    }
    figures = _figures(
        output,
        pre_matrix,
        distributions,
        variants,
        baselines,
        final_summary,
        client_losses,
    )
    _write_markdown(
        output / "resultados.md",
        diagnostics_data,
        pre_distribution,
        distributions,
        variants,
        baselines,
        final_summary,
        primary,
        secondary,
        sources,
    )
    return {"output": str(output), "tables": len(tables), "figures": len(figures)}


def _pre_diagnostic(data: dict) -> tuple[list[dict], list[dict]]:
    matrix = []
    distributions = []
    for regime, report in data["results"].items():
        metrics = report["global"]
        for label, examples in enumerate(metrics["support"]):
            distributions.append(
                {
                    "regime": regime,
                    "kind": "actual",
                    "class": label,
                    "examples": examples,
                    "source_file": "results/final_diagnostics_verified/diagnostics.json",
                }
            )
        for label, examples in enumerate(metrics["predicted_support"]):
            distributions.append(
                {
                    "regime": regime,
                    "kind": "predicted_cost_rule",
                    "class": label,
                    "examples": examples,
                    "source_file": "results/final_diagnostics_verified/diagnostics.json",
                }
            )
        for actual, row in enumerate(metrics["confusion_matrix"]):
            for predicted, examples in enumerate(row):
                matrix.append(
                    {
                        "regime": regime,
                        "actual_class": actual,
                        "predicted_class": predicted,
                        "examples": examples,
                        "source_file": "results/final_diagnostics_verified/diagnostics.json",
                    }
                )
    return matrix, distributions


def _class_distributions(cache: Path) -> list[dict]:
    rows = []
    for split in ("train", "validation"):
        with np.load(cache / f"{split}.npz", allow_pickle=False) as values:
            labels = values["labels"]
        for label, examples in enumerate(np.bincount(labels, minlength=5)):
            rows.append(
                {
                    "split": split,
                    "class": label,
                    "examples": int(examples),
                    "share": float(examples / len(labels)),
                    "source_file": str(cache / f"{split}.npz"),
                }
            )
    return rows


def _study_summary(rows: list[dict]) -> list[dict]:
    metrics = ("mean_cost", "macro_pr_auc", "balanced_accuracy", "accuracy")
    output = []
    for regime in sorted({row["regime"] for row in rows}):
        selected = [row for row in rows if row["regime"] == regime]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            output.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "n": len(values),
                    "source_file": "results/remediation_multiseed/metrics.csv",
                }
            )
    return output


def _figures(output, matrix, distributions, variants, baselines, summary, losses):
    return [
        _plot_pre_confusion(output, matrix),
        _plot_distribution(output, distributions),
        _plot_cv(output, variants, "variant", "cv_variants.png"),
        _plot_cv(output, baselines, "model", "cv_baselines.png"),
        _plot_study(output, summary),
        _plot_client_losses(output, losses),
    ]


def _plot_pre_confusion(output: Path, rows: list[dict]) -> str:
    figure, axes = plt.subplots(2, 2, figsize=(8, 7))
    for axis, regime in zip(
        axes.flat, ("local", "centralized", "fedavg", "fedprox"), strict=True
    ):
        values = np.zeros((5, 5), dtype=int)
        for row in rows:
            if row["regime"] == regime:
                values[int(row["actual_class"]), int(row["predicted_class"])] = int(
                    row["examples"]
                )
        image = axis.imshow(values, cmap="Blues")
        axis.set_title(regime)
        axis.set_xlabel("Predicha por costo esperado")
        axis.set_ylabel("Real")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save(figure, output / "pre_confusion_matrices.png")


def _plot_distribution(output: Path, rows: list[dict]) -> str:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        axis.plot(
            [row["class"] for row in selected],
            [row["share"] for row in selected],
            marker="o",
            label=split,
        )
    axis.set_xlabel("Clase")
    axis.set_ylabel("Proporción")
    axis.legend()
    return _save(figure, output / "train_validation_distribution.png")


def _plot_cv(output: Path, rows: list[dict], field: str, name: str) -> str:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    names = [row[field] for row in rows]
    values = [float(row["mean_cost_mean"]) for row in rows]
    errors = [float(row["mean_cost_sd"]) for row in rows]
    axis.bar(names, values, yerr=errors, capsize=4)
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel("Costo medio SCANIA en CV")
    return _save(figure, output / name)


def _plot_study(output: Path, rows: list[dict]) -> str:
    selected = [row for row in rows if row["metric"] == "mean_cost"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [row["regime"] for row in selected],
        [row["mean"] for row in selected],
        yerr=[row["sd"] for row in selected],
        capsize=4,
    )
    axis.set_ylabel("Costo medio SCANIA")
    return _save(figure, output / "multiseed_cost.png")


def _plot_client_losses(output: Path, rows: list[dict]) -> str:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for regime in ("fedavg", "fedprox"):
        for node in ("alpha", "beta", "gamma"):
            selected = [
                row for row in rows if row["regime"] == regime and row["node"] == node
            ]
            rounds = sorted({int(row["round"]) for row in selected})
            means = [
                np.mean(
                    [
                        float(row["training_loss"])
                        for row in selected
                        if int(row["round"]) == round_
                    ]
                )
                for round_ in rounds
            ]
            axis.plot(rounds, means, label=f"{regime}-{node}")
    axis.set_xlabel("Ronda")
    axis.set_ylabel("Pérdida cliente")
    axis.legend(ncol=2)
    return _save(figure, output / "client_round_losses.png")


def _save(figure, path: Path) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return str(path)


def _write_markdown(
    path,
    diagnostics,
    pre_distribution,
    distributions,
    variants,
    baselines,
    summary,
    primary,
    secondary,
    sources,
):
    weights = np.sqrt(
        sum(diagnostics["training_class_counts"])
        / (5 * np.asarray(diagnostics["training_class_counts"]))
    )
    mechanism = [
        {
            "field": "decision_rule",
            "value": "minimum_expected_cost",
            "source_file": sources["diagnostics"],
        },
        {
            "field": "loss_weighting",
            "value": "sqrt_inverse_frequency",
            "source_file": sources["diagnostics"],
        },
        {
            "field": "class_weights",
            "value": list(map(float, weights)),
            "source_file": sources["diagnostics"],
        },
        {
            "field": "double_correction",
            "value": True,
            "source_file": sources["diagnostics"],
        },
    ]
    lines = [
        "# Diagnóstico y remediación de desbalance",
        "",
        "## Diagnóstico sin reentrenamiento",
        "",
        _table(mechanism),
        "",
        "La regla histórica es mínimo costo esperado. La pérdida usa pesos de raíz del inverso de frecuencia; junto con la matriz de costos en la decisión, existe doble corrección del desbalance.",
        "",
        "Las matrices y distribuciones predichas están en `pre_confusion.csv` y `pre_predicted_distribution.csv`.",
        "",
        _table(pre_distribution),
        "",
        _table(distributions),
        "",
        "## Selección interna agrupada",
        "",
        _table(variants),
        "",
        "Cada fold retiene vehículos completos y evalúa solamente su última ventana. `validation` no se usó para seleccionar la variante.",
        "",
        "La corrección de la variante d ajusta el prior de entrenamiento a la frecuencia de clase de la última ventana almacenada por vehículo dentro del fold de entrenamiento.",
        "",
        "## Modelos de referencia",
        "",
        _table(baselines),
        "",
        "## Evaluación multisemilla de la variante elegida",
        "",
        _table(summary),
        "",
        "## Estadística primaria: costo medio",
        "",
        _table(primary),
        "",
        "Holm se aplica únicamente a los cinco contrastes de costo medio. Las métricas secundarias se reportan sin ajuste en `secondary_statistics.csv`.",
        "",
        "## Estadísticas secundarias",
        "",
        _table(secondary),
        "",
        "La procedencia, commit, hardware y tiempos están en `results/remediation_multiseed/manifest.json`. Las figuras PNG a 300 dpi están adjuntas en este directorio.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(rows: list[dict]) -> str:
    if not rows:
        return "NO DISPONIBLE"
    fields = list(rows[0])
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    values = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *values))


def _copy_csv(source: Path, output: Path) -> str:
    target = output / source.name
    target.write_bytes(source.read_bytes())
    return str(target)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> str:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


if __name__ == "__main__":
    main()
