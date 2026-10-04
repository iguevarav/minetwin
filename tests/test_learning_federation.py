import json
from dataclasses import replace

import numpy as np
import pytest

from minetwin.learning.data import CachedSplit
from minetwin.learning.diagnostics import run_model_diagnostics
from minetwin.learning.federation import ScaniaFederatedInference
from minetwin.learning.inference import risk_assessment
from minetwin.learning.provenance import (
    training_provenance,
    verified_evaluation_provenance,
    verify_training_provenance,
)
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
    assert federation.state("A-1").quality is None
    assert not hasattr(federation.coordinator, "features")
    assert not hasattr(federation.state("A-1"), "features")
    assert all(record.raw_records == 0 for record in federation.ledger.records)
    assert all(
        record.message_type == MessageType.PUBLISHED_STATE
        for record in federation.ledger.records
    )


def test_published_quality_uses_only_the_window_missingness_summary():
    cached = CachedSplit(
        features=np.asarray(((0, 2, 0.25),), dtype=np.float32),
        labels=np.asarray((0,), dtype=np.int8),
        nodes=np.asarray(("alpha",)),
        vehicle_ids=np.asarray(("A-1",)),
        feature_names=("a__last", "b__last", "a__missing_rate"),
    )
    federation = ScaniaFederatedInference(
        cached, "validation", {"alpha": FakePrognosticator()}, "local"
    )
    federation.publish()
    state = federation.state("A-1")
    assert state.quality == pytest.approx(0.75)
    assert not hasattr(state, "features")


def test_coordinator_rejects_incompatible_model_and_data_versions():
    predictors = {"alpha": FakePrognosticator(), "beta": FakePrognosticator()}
    federation = ScaniaFederatedInference(
        cached_split(), "validation", predictors, "fedprox"
    )
    federation.publish()
    state = federation.state("A-1")
    with pytest.raises(ValueError, match="no coincide"):
        federation.coordinator.receive(replace(state, data_version="other"))
    with pytest.raises(ValueError, match="no coincide"):
        federation.coordinator.receive(
            replace(state, risk=replace(state.risk, model_id="other"))
        )
    with pytest.raises(ValueError, match="no coincide"):
        federation.coordinator.receive(replace(state, asset_id="unknown"))
    federation.coordinator.receive(state)
    assert len(federation.ledger.records) == 3


def test_training_provenance_detects_cache_or_model_changes(tmp_path):
    cache = tmp_path / "cache"
    models = tmp_path / "models"
    cache.mkdir()
    models.mkdir()
    for name in ("train.npz", "validation.npz", "metadata.json"):
        (cache / name).write_bytes(name.encode())
    (models / "scaler.npz").write_bytes(b"scaler")
    (models / "fedprox.pt").write_bytes(b"model")
    report = {
        "config": {"seed": 100},
        "torch": "2.14.0",
        "provenance": training_provenance(cache, models, {"seed": 100}, "2.14.0"),
    }
    assert verify_training_provenance(cache, models, report) == report["provenance"]
    _, provenance = verified_evaluation_provenance(
        cache, models, report, "validation"
    )
    assert provenance["verified"]
    assert provenance["training_id"] == report["provenance"]["run_id"]
    (models / "fedprox.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="no coinciden"):
        verify_training_provenance(cache, models, report)
    (models / "fedprox.pt").write_bytes(b"model")
    (cache / "train.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="no coinciden"):
        verify_training_provenance(cache, models, report)


@pytest.mark.parametrize("changed_artifact", ("cache", "model"))
def test_diagnostics_rejects_artifacts_without_matching_provenance(
    tmp_path, changed_artifact
):
    cache = tmp_path / "cache"
    models = tmp_path / "models"
    cache.mkdir()
    models.mkdir()
    for name in ("train.npz", "validation.npz", "metadata.json"):
        (cache / name).write_bytes(name.encode())
    (models / "scaler.npz").write_bytes(b"scaler")
    model = models / "fedprox.pt"
    model.write_bytes(b"model")
    config = {"seed": 100}
    report = {
        "config": config,
        "torch": "2.14.0",
        "provenance": training_provenance(cache, models, config, "2.14.0"),
    }
    (models / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    target = cache / "train.npz" if changed_artifact == "cache" else model
    target.write_bytes(b"changed")

    output = tmp_path / "diagnostics"
    with pytest.raises(ValueError, match="no coinciden"):
        run_model_diagnostics(cache, models, output)

    assert not output.exists()


def test_legacy_models_cannot_be_published_without_provenance(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "metrics.json").write_text(
        json.dumps({"config": {}, "torch": "2.14.0"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="procedencia verificable"):
        ScaniaFederatedInference.from_artifacts(tmp_path, models)


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
