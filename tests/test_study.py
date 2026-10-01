import csv
import json

import pytest

from minetwin.domain import Scenario
from minetwin.langflow_client import FLOW_INSTRUCTIONS, PROMPT_VERSION
from minetwin.learning import TrainingConfig
from minetwin.learning.provenance import (
    file_sha256,
    training_provenance,
    value_sha256,
)
from minetwin.study import (
    LearningStudyConfig,
    WorkshopStudyConfig,
    build_study_report,
    run_learning_study,
    run_workshop_study,
)


def test_learning_study_exports_paired_statistics(monkeypatch, tmp_path):
    def train(cache, output, config):
        output.mkdir()
        return {
            "results": {
                regime: {
                    "global": {
                        "total_cost": base + config.seed,
                        "macro_f1": 1 / base,
                        "balanced_accuracy": 1 / base,
                        "macro_pr_auc": 1 / base,
                    }
                }
                for regime, base in {
                    "local": 4,
                    "centralized": 3,
                    "fedavg": 2,
                    "fedprox": 1,
                }.items()
            }
        }

    monkeypatch.setattr("minetwin.study.train_learning_regimes", train)
    output = tmp_path / "learning"
    summary = run_learning_study(
        tmp_path,
        output,
        LearningStudyConfig(
            seeds=(100, 101),
            bootstrap_samples=100,
            training=TrainingConfig(epochs=1, federated_rounds=1),
        ),
    )
    assert summary["runs"] == 8
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    with (output / "statistics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 16
    assert all(row["p_value_holm"] for row in rows)


def test_workshop_study_exports_cost_sensitivity_and_noninferiority(tmp_path):
    output = tmp_path / "workshop"
    summary = run_workshop_study(
        output,
        WorkshopStudyConfig(
            seeds=(100,),
            scenarios=(Scenario.NORMAL,),
            bays=(1,),
            cost_ratios=(3, 5),
            steps=30,
            bootstrap_samples=100,
        ),
    )
    assert summary["simulations"] == 4
    assert summary["cost_evaluations"] == 8
    assert summary["noninferiority"]["pairs"] == 1
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    with (output / "metrics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert {row["cost_ratio"] for row in rows} == {"3", "5"}


def test_real_report_excludes_simulation_and_rejects_old_langflow(tmp_path):
    paths = {
        name: tmp_path / name
        for name in (
            "cache",
            "learning",
            "scania",
            "selection",
            "interpretability",
            "federation",
            "langflow",
        )
    }
    for path in paths.values():
        path.mkdir()
    cache = paths["cache"]
    scania = paths["scania"]
    (scania / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    _json(
        scania / "summary.json",
        {
            "dataset": "SCANIA Component X",
            "version": "3",
            "features": 2,
            "splits": [{"split": "validation", "vehicles": 1, "readouts": 2}],
        },
    )
    for name in ("train.npz", "validation.npz"):
        (cache / name).write_bytes(name.encode())
    _json(
        cache / "metadata.json",
        {
            "source_manifest_sha256": file_sha256(
                scania / "dataset_manifest.json"
            ),
            "feature_config": {"window_size": 12},
            "feature_count": 2,
        },
    )
    training_config = {"epochs": 1}
    selection = paths["selection"]
    _json(selection / "manifest.json", {"status": "complete"})
    _json(
        selection / "selection.json",
        {
            "best_config": training_config,
            "cross_validation": {"train_validation_vehicle_overlap": 0},
        },
    )
    _json(
        selection / "eda.json",
        {
            "feature_config": {"window_size": 12},
            "feature_count": 2,
            "splits": [
                {
                    "split": "train",
                    "examples": 1,
                    "vehicles": 1,
                    "nodes": {"alpha": 1},
                    "classes": {"0": 1},
                },
                {
                    "split": "validation",
                    "examples": 1,
                    "vehicles": 1,
                    "nodes": {"alpha": 1},
                    "classes": {"0": 1},
                },
            ],
        },
    )
    (selection / "candidates.csv").write_text(
        "candidate,selected\nMLP-01,True\n", encoding="utf-8"
    )
    (selection / "fold_metrics.csv").write_text(
        "candidate,fold\nMLP-01,1\n", encoding="utf-8"
    )
    learning = paths["learning"]
    _json(
        learning / "manifest.json",
        {
            "status": "complete",
            "config": {"training": training_config, "seeds": [100, 101]},
        },
    )
    _json(
        learning / "summary.json",
        {
            "seeds": 2,
            "aggregates": {
                "fedprox": {
                    name: {"mean": 0.5, "sd": 0.0}
                    for name in (
                        "mean_cost",
                        "macro_f1",
                        "balanced_accuracy",
                        "macro_pr_auc",
                    )
                }
            },
        },
    )
    (learning / "statistics.csv").write_text(
        "reference,candidate,metric,pairs\nlocal,fedprox,mean_cost,1\n",
        encoding="utf-8",
    )
    model_root = learning / "seed_100"
    model_root.mkdir()
    (model_root / "scaler.npz").write_bytes(b"scaler")
    (model_root / "fedprox.pt").write_bytes(b"model")
    model_report = {"config": training_config, "torch": "test"}
    model_report["provenance"] = training_provenance(
        cache, model_root, training_config, "test"
    )
    _json(model_root / "metrics.json", model_report)
    training_id = model_report["provenance"]["run_id"]
    split_hash = file_sha256(cache / "validation.npz")
    data_version = value_sha256(
        {"training_id": training_id, "split": "validation", "sha256": split_hash}
    )
    model_id = f"fedprox:{file_sha256(model_root / 'fedprox.pt')[:12]}"
    federation = paths["federation"]
    (federation / "published_states.jsonl").write_text("{}\n", encoding="utf-8")
    _json(
        federation / "summary.json",
        {
            "split": "validation",
            "regime": "fedprox",
            "coordinator_states": 1,
            "coordinator_raw_features": 0,
            "publication_reduction_ratio": 0.5,
            "published_states_sha256": file_sha256(
                federation / "published_states.jsonl"
            ),
            "transfer": {"messages": 1, "payload_bytes": 10, "raw_records": 0},
            "nodes": {"alpha": {"private_records": 1, "published_states": 1}},
            "provenance": {
                "verified": True,
                "training_id": training_id,
                "data_version": data_version,
                "split_sha256": split_hash,
                "model_ids": {"alpha": model_id},
            },
        },
    )
    interpretation = paths["interpretability"]
    _json(
        interpretation / "summary.json",
        {
            "model_id": model_id,
            "split": "validation",
            "regime": "fedprox",
            "examples": 1,
        },
    )
    (interpretation / "permutation_importance.csv").write_text(
        "dimension,group,mean_cost_increase\nsource_variable,feature_0,0.1\n",
        encoding="utf-8",
    )
    langflow = paths["langflow"]
    (langflow / "flow_definition.json").write_text("{}", encoding="utf-8")
    (langflow / "prompt.txt").write_text(FLOW_INSTRUCTIONS, encoding="utf-8")
    (langflow / "cases.jsonl").write_text("{}\n", encoding="utf-8")
    (langflow / "review.csv").write_text(
        "case_id,response_received,reviewer\nclass_0,True,reviewer\n",
        encoding="utf-8",
    )
    flow_manifest = {
        "status": "complete",
        "source": "SCANIA Component X validation",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": file_sha256(langflow / "prompt.txt"),
        "training_id": training_id,
        "data_version": data_version,
        "predictive_models": [model_id],
        "language_model_id": "ollama-model",
        "flow_id": "flow-1",
        "flow_version": "1",
        "flow_definition_sha256": file_sha256(langflow / "flow_definition.json"),
        "cases_sha256": file_sha256(langflow / "cases.jsonl"),
        "observed_classes_evaluated": [0],
        "counts": {"cases": 1, "responses": 1, "errors": 0},
    }
    _json(langflow / "manifest.json", flow_manifest)
    _json(
        langflow / "review_summary.json",
        {
            "reviewed_responses": 1,
            "total_cases": 1,
            "review_sha256": file_sha256(langflow / "review.csv"),
            "overall_mean_score": 2,
            "mean_scores": {"factual_accuracy": 2},
        },
    )
    output = tmp_path / "report"
    summary = build_study_report(output, **paths)
    assert "workshop" not in summary
    assert "Taller" not in (output / "report.html").read_text(encoding="utf-8")
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))[
        "status"
    ] == "complete"
    flow_manifest["source"] = "Synthetic academic simulation"
    _json(langflow / "manifest.json", flow_manifest)
    with pytest.raises(ValueError, match="Langflow"):
        build_study_report(tmp_path / "rejected", **paths)
    assert not (tmp_path / "rejected").exists()


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
