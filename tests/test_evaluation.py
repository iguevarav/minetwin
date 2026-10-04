import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from minetwin.langflow_client import LangflowConfig, ScaniaExplanationObservation
from minetwin.langflow_evaluation import (
    REVIEW_FIELDS,
    LangflowCase,
    run_langflow_evaluation,
    summarize_langflow_review,
)
from minetwin.learning.inference import risk_assessment
from minetwin.learning.provenance import file_sha256
from minetwin.publication import PublishedTwinState


class ExplanationClient:
    def __init__(self):
        self.config = LangflowConfig("http://localhost:7860", "evaluation-flow", "key")
        self.calls = []

    def explain(self, observation):
        self.calls.append(observation)
        return (
            f"Clase estimada: {observation.state.risk.predicted_class}. "
            "La explicaciÃ³n se limita a la evidencia proporcionada."
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
    assert manifest["langflow_url"] == "http://localhost:7860"
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


def test_scania_langflow_evaluation_requires_the_selected_model(tmp_path, monkeypatch):
    monkeypatch.delenv("MINETWIN_LANGFLOW_MODEL_ID", raising=False)
    flow = tmp_path / "flow.json"
    flow.write_text('{"id":"evaluation-flow"}', encoding="utf-8")

    with pytest.raises(ValueError, match="MINETWIN_LANGFLOW_MODEL_ID"):
        run_langflow_evaluation(tmp_path / "langflow", flow_definition=flow)


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
