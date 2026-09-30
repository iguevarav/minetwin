import csv
import json

from minetwin.domain import Scenario
from minetwin.learning import TrainingConfig
from minetwin.study import (
    LearningStudyConfig,
    WorkshopStudyConfig,
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
