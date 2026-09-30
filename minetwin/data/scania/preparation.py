import csv
import json
from dataclasses import asdict
from pathlib import Path

from minetwin.data.scania.contracts import (
    SCANIA_CLASS_WINDOWS,
    SCANIA_COST_MATRIX,
    DatasetSplit,
)
from minetwin.data.scania.loader import ScaniaDataset
from minetwin.data.scania.manifest import ScaniaManifest
from minetwin.data.scania.partition import CategoricalNodePartitioner


def prepare_scania_dataset(
    dataset_root: Path,
    output: Path,
    version: str = "3",
    partitioner: CategoricalNodePartitioner | None = None,
) -> dict:
    dataset = ScaniaDataset(dataset_root)
    effective_partitioner = partitioner or CategoricalNodePartitioner()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = ScaniaManifest.build(dataset.root, version)
        summaries = dataset.validate()
        assignments = _assignments(dataset, effective_partitioner)
        manifest.write(output / "dataset_manifest.json")
        _write_assignments(output / "partitions.csv", assignments)
        experiment = {
            "objective": "five_class_imminent_component_failure",
            "primary_metric": "total_misclassification_cost",
            "reporting_metrics": (
                "macro_f1",
                "per_class_precision",
                "per_class_recall",
                "balanced_accuracy",
                "one_vs_rest_pr_auc",
            ),
            "class_windows": [
                {"class_label": label, "lower": lower, "upper": upper}
                for label, lower, upper in SCANIA_CLASS_WINDOWS
            ],
            "cost_matrix": SCANIA_COST_MATRIX,
            "partition": asdict(effective_partitioner),
            "leakage_controls": {
                "official_splits_preserved": True,
                "partition_uses_labels": False,
                "imputation": "causal_locf_with_missing_indicators",
                "windows_cross_vehicle_boundaries": False,
            },
        }
        (output / "experiment.json").write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = {
            "dataset": "SCANIA Component X",
            "version": manifest.version,
            "features": len(dataset.schema.feature_names),
            "specifications": len(dataset.schema.specification_names),
            "splits": [asdict(summary) for summary in summaries],
            "nodes": {
                node_id: sum(row[2] == node_id for row in assignments)
                for node_id in effective_partitioner.node_ids
            },
        }
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    except (Exception, KeyboardInterrupt):
        for path in output.iterdir():
            if path.is_file():
                path.unlink()
        output.rmdir()
        raise


def _assignments(
    dataset: ScaniaDataset, partitioner: CategoricalNodePartitioner
) -> list[tuple[str, str, str]]:
    rows = []
    for split in DatasetSplit:
        for vehicle_id, specifications in dataset.load_specifications(split).items():
            rows.append((split, vehicle_id, partitioner.assign(specifications)))
    return rows


def _write_assignments(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("split", "vehicle_id", "node_id"))
        writer.writerows(rows)
