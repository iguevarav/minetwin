import json

import numpy as np

from minetwin.learning.data import CachedSplit
from minetwin.learning.federation import ScaniaFederatedInference
from minetwin.learning.inference import risk_assessment
from minetwin.publication import PublishedCondition
from minetwin.transfer import MessageType, TransferDirection, TransferLedger


class FakePrognosticator:
    model_id = "risk:test"
    parameter_bytes = 120

    def predict(self, features):
        classes = features[:, 0].astype(int)
        probabilities = np.eye(5, dtype=np.float64)[classes]
        return classes, probabilities


def cached_split():
    return CachedSplit(
        features=np.asarray(((0, 2), (3, 4), (4, 6)), dtype=np.float32),
        labels=np.asarray((0, 3, 4), dtype=np.int8),
        nodes=np.asarray(("alpha", "beta", "alpha")),
        vehicle_ids=np.asarray(("A-1", "B-1", "A-2")),
        feature_names=("feature_0", "feature_1"),
    )


def test_nodes_publish_only_risk_state_and_keep_features_private():
    predictors = {"alpha": FakePrognosticator(), "beta": FakePrognosticator()}
    federation = ScaniaFederatedInference(
        cached_split(), "validation", predictors, "fedprox"
    )
    federation.publish()
    assert len(federation.states) == 3
    assert federation.state("A-1").risk.condition == PublishedCondition.NORMAL
    assert federation.state("B-1").risk.condition == PublishedCondition.ALERT
    assert not hasattr(federation.coordinator, "features")
    assert not hasattr(federation.state("A-1"), "features")
    assert all(record.raw_records == 0 for record in federation.ledger.records)
    assert all(
        record.message_type == MessageType.PUBLISHED_STATE
        for record in federation.ledger.records
    )


def test_risk_assessment_normalizes_probabilities_and_limits_the_recommendation():
    assessment = risk_assessment(
        3,
        np.asarray((0.1, 0.2, 0.3, 0.4, 0.0)),
        "risk:test",
    )
    assert assessment.condition == PublishedCondition.ALERT
    assert np.isclose(sum(assessment.probabilities), 1)
    assert assessment.recommendation == "Priorizar la inspección de Component X."


def test_disconnected_node_keeps_local_data_without_publishing():
    predictors = {"alpha": FakePrognosticator(), "beta": FakePrognosticator()}
    federation = ScaniaFederatedInference(
        cached_split(), "validation", predictors, "fedavg"
    )
    federation.set_connected("alpha", False)
    federation.publish()
    assert [state.asset_id for state in federation.states] == ["B-1"]
    assert federation.nodes["alpha"].private_records == 2
    assert federation.nodes["alpha"].private_feature_bytes == 16


def test_transfer_ledger_measures_canonical_payload_bytes(tmp_path):
    ledger = TransferLedger()
    payload = {"risk": 4, "node": "alpha"}
    record = ledger.record_payload(
        "alpha",
        TransferDirection.NODE_TO_COORDINATOR,
        MessageType.PUBLISHED_STATE,
        payload,
    )
    expected = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert record.payload_bytes == len(expected)
    assert ledger.summary()["raw_records"] == 0
    ledger.write_csv(tmp_path / "ledger.csv")
    assert (tmp_path / "ledger.csv").read_text(encoding="utf-8").count("\n") == 2
