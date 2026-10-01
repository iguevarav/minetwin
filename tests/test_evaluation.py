import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from minetwin.domain import MaintenancePolicy, Scenario
from minetwin.evaluation import EvaluationConfig, run_fleet_evaluation
from minetwin.langflow_client import LangflowConfig, ScaniaExplanationObservation
from minetwin.langflow_evaluation import (
    REVIEW_FIELDS,
    LangflowCase,
    run_langflow_evaluation,
    summarize_langflow_review,
)
from minetwin.learning.inference import risk_assessment
from minetwin.learning.provenance import file_sha256
from minetwin.maintenance import MaintenanceConfig
from minetwin.publication import PublishedTwinState
from minetwin.services import MineTwin
from minetwin.simulation import SimulationConfig


class ExplanationClient:
    def __init__(self):
        self.config = LangflowConfig("http://localhost:7860", "evaluation-flow", "key")
        self.calls = []

    def explain(self, observation):
        self.calls.append(observation)
        return (
            f"Clase estimada: {observation.state.risk.predicted_class}. "
            "La explicación se limita a la evidencia proporcionada."
        )


def _langflow_cases():
    cases = []
    for label in range(5):
        state = PublishedTwinState(
            asset_id=f"V-{label}",
            node_id="alpha",
            component="Component X",
            source="SCANIA Component X",
            split="validation",
            sequence=label + 1,
            risk=risk_assessment(label, np.eye(5)[label], f"model-{label}"),
            quality=0.8,
            training_id="training-1",
            data_version="data-1",
        )
        observation = ScaniaExplanationObservation(
            state=state,
            vehicle_id=state.asset_id,
            node_id="alpha",
            split="validation",
            time_step=10.0 + label,
            readout_index=2,
            readout_count=2,
        )
        cases.append(LangflowCase(f"class_{label}", observation, label))
    reference = cases[0].observation
    cases.extend(
        (
            LangflowCase(
                "missing_readings",
                replace(
                    reference,
                    unavailable_readings=("feature_1",),
                    unavailable_count=1,
                ),
                0,
            ),
            LangflowCase("missing_state", replace(reference, state=None), 0),
            LangflowCase("stale_observation", replace(reference, stale=True), 0),
            LangflowCase(
                "disconnected_node", replace(reference, connected=False), 0
            ),
            LangflowCase(
                "connection_error", reference, 0, transport_fault="connection_error"
            ),
            LangflowCase("timeout", reference, 0, transport_fault="timeout"),
        )
    )
    return tuple(cases)


def test_fleet_evaluation_exports_paired_reproducible_evidence(tmp_path):
    output = tmp_path / "evaluation"
    result = run_fleet_evaluation(
        output,
        EvaluationConfig(
            steps=3,
            seeds=(101,),
            scenarios=(Scenario.NORMAL,),
            dataset_role="evaluation",
        ),
    )
    assert len(result) == 9
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {
        "trajectories": 27,
        "paired_differences": 18,
        "paired_summary_groups": 6,
        "summary_groups": 9,
        "interoperability_cases": 9,
        "continuity_cases": 1,
    }
    expected = {
        "manifest.json",
        "events.csv",
        "predictions.csv",
        "trajectory_metrics.csv",
        "paired_differences.csv",
        "paired_summary.csv",
        "summary.csv",
        "interoperability.csv",
        "continuity.csv",
        "report.html",
    }
    assert {path.name for path in output.iterdir()} == expected
    with (output / "trajectory_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        trajectories = list(csv.DictReader(stream))
    assert len(trajectories) == 27
    assert {row["truck_id"] for row in trajectories} == {
        f"TRUCK-{index:03d}" for index in range(1, 10)
    }
    assert {row["policy"] for row in trajectories} == {
        policy.value for policy in MaintenancePolicy
    }
    with (output / "paired_differences.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 18
    with (output / "interoperability.csv").open(encoding="utf-8", newline="") as stream:
        interoperability = list(csv.DictReader(stream))
    assert all(row["canonical_equivalent"] == "True" for row in interoperability)
    assert all(row["malformed_packet_rejected"] == "True" for row in interoperability)
    with (output / "continuity.csv").open(encoding="utf-8", newline="") as stream:
        continuity = list(csv.DictReader(stream))
    assert all(
        continuity[0][field] == "True"
        for field in (
            "central_view_frozen",
            "local_node_continued",
            "other_nodes_continued",
            "offline_view_stale",
            "reconnected_to_current_state",
            "reconnected_view_fresh",
        )
    )


def test_evaluation_requires_unique_seeds_reference_policy_and_new_output(tmp_path):
    with pytest.raises(ValueError):
        EvaluationConfig(seeds=(1, 1))
    with pytest.raises(ValueError):
        EvaluationConfig(policies=(MaintenancePolicy.PREDICTIVE,))
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_fleet_evaluation(output, EvaluationConfig(steps=1, seeds=(1,)))
    assert marker.read_text(encoding="utf-8") == "keep"


def test_incomplete_fleet_evaluation_is_marked(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr("minetwin.evaluation._failure_reference", fail)
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError):
        run_fleet_evaluation(
            output,
            EvaluationConfig(
                steps=1,
                seeds=(1,),
                scenarios=(Scenario.NORMAL,),
            ),
        )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"


def test_external_noise_path_is_equal_at_the_same_simulated_time():
    reference = MineTwin(SimulationConfig(seed=7))
    delayed = MineTwin(
        SimulationConfig(seed=7),
        maintenance_config=MaintenanceConfig(brake_service_seconds=60),
    )
    reference.advance(11)
    delayed.advance(5)
    order = delayed.create_order(component="brakes")
    delayed.start_order(order.id)
    delayed.advance()
    delayed.advance(5)
    assert reference.time == delayed.time
    for signal, baseline_index in (
        ("engine_temperature", 0),
        ("engine_vibration", 1),
    ):
        residuals = []
        for simulation in (reference, delayed):
            packet = simulation.twin.observation
            baseline = simulation.profile.engine_baseline(packet.operating_state)[
                baseline_index
            ]
            residuals.append(packet.reading(signal).value - baseline)
        assert residuals[0] == pytest.approx(residuals[1])


def test_variable_degradation_differs_from_linear_trajectory():
    linear = MineTwin(SimulationConfig(seed=9, scenario=Scenario.ENGINE_DEGRADATION))
    variable = MineTwin(
        SimulationConfig(seed=9, scenario=Scenario.ENGINE_VARIABLE_DEGRADATION)
    )
    linear.advance(180)
    variable.advance(180)
    assert (
        linear.twin.diagnosis.temperature_residual
        != variable.twin.diagnosis.temperature_residual
    )
    assert (
        linear.twin.diagnosis.vibration_residual
        != variable.twin.diagnosis.vibration_residual
    )


def test_langflow_evaluation_exports_evidence_and_blank_human_review(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MINETWIN_LANGFLOW_FLOW_VERSION", raising=False)
    client = ExplanationClient()
    output = tmp_path / "langflow"
    flow = tmp_path / "flow.json"
    flow.write_text('{"id":"evaluation-flow"}', encoding="utf-8")
    records = run_langflow_evaluation(
        output, client, _langflow_cases(), flow_definition=flow
    )
    assert len(records) == 11
    assert len(client.calls) == 6
    assert records[-1]["case_id"] == "timeout"
    assert records[-1]["error"]
    assert sum(record["blocked_locally"] for record in records) == 3
    assert all(record["state_unchanged"] for record in records)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert (output / "flow_definition.json").read_bytes() == flow.read_bytes()
    assert manifest["flow_definition_sha256"]
    assert manifest["flow_version"] == manifest["flow_definition_sha256"][:12]
    assert manifest["cases_sha256"] == file_sha256(output / "cases.jsonl")
    assert manifest["counts"] == {
        "cases": 11,
        "responses": 6,
        "errors": 5,
        "locally_blocked": 3,
    }
    with (output / "review.csv").open(encoding="utf-8", newline="") as stream:
        review = list(csv.DictReader(stream))
    assert len(review) == 11
    assert all(not row["factual_accuracy_0_to_2"] for row in review)
    cases = [
        json.loads(line)
        for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [case["expected"]["observed_class"] for case in cases[:5]] == list(
        range(5)
    )
    assert cases[5]["context"]["observation"]["unavailable_reading_count"] == 1
    assert all(case["context"] is None for case in cases[6:9])


def test_langflow_human_review_is_validated_and_summarized(tmp_path):
    output = tmp_path / "langflow-review"
    run_langflow_evaluation(output, ExplanationClient(), _langflow_cases())
    with (output / "review.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["response_received"] == "True":
            row.update(
                factual_accuracy_0_to_2="2",
                class_and_scope_0_to_2="2",
                uncertainty_handling_0_to_2="1",
                no_unsupported_action_claim_0_to_2="2",
                reviewer="reviewer-1",
            )
    with (output / "review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_langflow_review(output)
    assert summary["reviewed_responses"] == 6
    assert summary["overall_mean_score"] == pytest.approx(1.75)
    assert summary["reviewers"] == ["reviewer-1"]
    assert summary["review_sha256"] == file_sha256(output / "review.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_review"] == summary
    assert (
        json.loads((output / "review_summary.json").read_text(encoding="utf-8"))
        == summary
    )
