"""Regenerate the historical diagnosis figures from saved CSVs, without training."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from minetwin.explanations import describe
from minetwin.exports import explained_rows, matplotlib_jpg, table_xlsx

TABLES = {
    "baseline_folds": "folds",
    "baseline_summary": "selection",
    "client_round_losses": "training",
    "multiseed_metrics": "performance",
    "multiseed_summary": "performance",
    "pre_confusion": "confusion",
    "pre_predicted_distribution": "classes",
    "primary_statistics": "statistics",
    "secondary_statistics": "statistics",
    "train_validation_class_distribution": "classes",
    "variant_folds": "folds",
    "variant_summary": "selection",
}
FIGURES = {
    "train_validation_distribution": "train_validation_class_distribution",
    "cv_variants": "variant_summary",
    "cv_baselines": "baseline_summary",
    "multiseed_cost": "multiseed_summary",
    "client_round_losses": "client_round_losses",
    "pre_confusion_matrices": "pre_confusion",
}
START = "<!-- MINETWIN_EXPORT_START -->"
END = "<!-- MINETWIN_EXPORT_END -->"


def refresh_legacy_report(output: Path) -> dict:
    output = Path(output)
    tables = {}
    for name in TABLES:
        with (output / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    notes = {
        name: explained_rows(
            describe(
                kind,
                name.replace("_", " ").title(),
                source=f"Reporte histórico de diagnóstico · {name}.csv; se conservan los valores y las referencias de origen.",
            ),
            tables[name],
        )
        for name, kind in TABLES.items()
    }
    for name, rows in tables.items():
        (output / f"{name}.xlsx").write_bytes(
            table_xlsx(rows, notes[name], parse_numbers=True)
        )
    images = []
    for name, table in FIGURES.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        rows = tables[table]
        if name == "train_validation_distribution":
            for split in ("train", "validation"):
                selected = [row for row in rows if row["split"] == split]
                axis.plot(
                    [int(row["class"]) for row in selected],
                    [int(row["examples"]) for row in selected],
                    marker="o",
                    label=split,
                )
            axis.set(xlabel="Clase", ylabel="Ejemplos")
            axis.legend()
        elif name in {"cv_variants", "cv_baselines"}:
            label = "variant" if name == "cv_variants" else "model"
            axis.bar(
                [row[label] for row in rows],
                [float(row["mean_cost_mean"]) for row in rows],
                yerr=[float(row["mean_cost_sd"]) for row in rows],
                capsize=4,
            )
            axis.tick_params(axis="x", rotation=20)
            axis.set_ylabel("Costo medio SCANIA en CV (± DE)")
        elif name == "multiseed_cost":
            selected = [row for row in rows if row["metric"] == "mean_cost"]
            axis.bar(
                [row["regime"] for row in selected],
                [float(row["mean"]) for row in selected],
                yerr=[float(row["sd"]) for row in selected],
                capsize=4,
            )
            axis.set_ylabel("Costo medio SCANIA (± DE entre semillas)")
        elif name == "client_round_losses":
            for regime in ("fedavg", "fedprox"):
                for node in ("alpha", "beta", "gamma"):
                    selected = [
                        row
                        for row in rows
                        if row["regime"] == regime and row["node"] == node
                    ]
                    rounds = sorted({int(row["round"]) for row in selected})
                    values = [
                        np.mean(
                            [
                                float(row["training_loss"])
                                for row in selected
                                if int(row["round"]) == round_
                            ]
                        )
                        for round_ in rounds
                    ]
                    axis.plot(rounds, values, label=f"{regime}-{node}")
            axis.set(xlabel="Ronda", ylabel="Pérdida de entrenamiento")
            axis.legend(ncol=2)
        else:
            plt.close(figure)
            figure, axes = plt.subplots(2, 2, figsize=(8, 7))
            for axis, regime in zip(
                axes.flat, ("local", "centralized", "fedavg", "fedprox"), strict=True
            ):
                values = np.zeros((5, 5), dtype=int)
                for row in rows:
                    if row["regime"] == regime:
                        values[
                            int(row["actual_class"]), int(row["predicted_class"])
                        ] = int(row["examples"])
                image = axis.imshow(values, cmap="Blues")
                axis.set(title=regime, xlabel="Predicha", ylabel="Real")
                figure.colorbar(image, ax=axis)
        try:
            target = output / f"{name}.jpg"
            target.write_bytes(matplotlib_jpg(figure, notes[table]))
            with Image.open(target) as image:
                images.append(
                    {
                        "file": target.name,
                        "pixels": list(image.size),
                        "dpi": list(image.info["dpi"]),
                    }
                )
        finally:
            plt.close(figure)
    markdown = output / "resultados.md"
    original = markdown.read_text(encoding="utf-8")
    original = re.sub(
        re.escape(START) + r".*?" + re.escape(END), "", original, flags=re.DOTALL
    )
    lines, enriched = original.splitlines(), []
    index, extra_tables = 0, 0
    while index < len(lines):
        line = lines[index]
        if (
            line.startswith("|")
            and index + 1 < len(lines)
            and "---" in lines[index + 1]
        ):
            headers = [part.strip() for part in line.strip("|").split("|")]
            matched = next(
                (
                    name
                    for name, rows in tables.items()
                    if rows and list(rows[0]) == headers
                ),
                None,
            )
            if matched:
                explanation = notes[matched]
                excel_name = matched
            else:
                extra_tables += 1
                excel_name = f"metadatos_historicos_{extra_tables}"
                values, cursor = [], index + 2
                while cursor < len(lines) and lines[cursor].startswith("|"):
                    cells = [
                        part.strip() for part in lines[cursor].strip("|").split("|")
                    ]
                    values.append(dict(zip(headers, cells, strict=True)))
                    cursor += 1
                explanation = explained_rows(
                    describe(
                        "context",
                        "Metadatos del diagnóstico histórico",
                        source="Reporte histórico · tabla de metadatos de resultados.md",
                    ),
                    values,
                )
                (output / f"{excel_name}.xlsx").write_bytes(
                    table_xlsx(values, explanation, parse_numbers=True)
                )
            enriched.extend([START, explanation.markdown(), ""])
            enriched.append(f"[Descargar tabla en Excel]({excel_name}.xlsx)")
            enriched.extend([END, ""])
        enriched.append(line)
        index += 1
    enriched.extend(["", START, "## Descargas del diagnóstico histórico", ""])
    for name in tables:
        enriched.extend(
            [
                f"### {notes[name].title}",
                notes[name].markdown(),
                f"[Descargar tabla en Excel]({name}.xlsx)",
                "",
            ]
        )
    for name, table in FIGURES.items():
        enriched.extend(
            [
                f"![{name}]({name}.jpg)",
                notes[table].markdown(),
                f"[Descargar imagen JPG · 600 DPI]({name}.jpg)",
                "",
            ]
        )
    enriched.append(END)
    markdown.write_text("\n".join(enriched), encoding="utf-8")
    (output / "exportaciones.json").write_text(
        json.dumps({"dpi": 600, "figures": images}, indent=2), encoding="utf-8"
    )
    for name in FIGURES:
        (output / f"{name}.png").unlink(missing_ok=True)
    return {
        "output": str(output),
        "figures": len(images),
        "excel": len(tables) + extra_tables,
        "dpi": 600,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/10_reporte"))
    args = parser.parse_args()
    print(json.dumps(refresh_legacy_report(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
