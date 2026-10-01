import io
import json
from dataclasses import replace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from minetwin.data.scania import (
    DatasetSplit,
    FeatureReadout,
    FeatureWindow,
    ScaniaReplayView,
)
from minetwin.langflow_client import (
    LangflowClient,
    LangflowConfig,
    LangflowError,
    ScaniaExplanationObservation,
    explanation_context,
)
from minetwin.learning import FeatureConfig, WindowVectorizer
from minetwin.learning.data import CachedSplit
from minetwin.learning.inference import risk_assessment
from minetwin.learning.scaling import FederatedStandardScaler
from minetwin.publication import PublishedTwinState
from minetwin.scania_explanation import ScaniaExplanationSource


@pytest.fixture
def observation():
    state = PublishedTwinState(
        asset_id="V-001",
        node_id="alpha",
        component="Component X",
        source="SCANIA Component X",
        split="validation",
        sequence=1,
        risk=risk_assessment(4, np.eye(5)[4], "fedprox:abc"),
        quality=0.75,
        training_id="training-1",
        data_version="data-1",
    )
    return ScaniaExplanationObservation(
        state=state,
        vehicle_id="V-001",
        node_id="alpha",
        split="validation",
        time_step=24.0,
        readout_index=4,
        readout_count=4,
        relevant_features=("feature_12", "feature_19"),
        unavailable_readings=("feature_19",),
        unavailable_count=1,
    )


def client_with_response(body):
    opener = Mock()
    opener.open.return_value = io.BytesIO(body)
    config = LangflowConfig("http://127.0.0.1:7860", "flow-123", "secret-key")
    return LangflowClient(config, opener), opener


def test_langflow_request_contains_only_published_scania_evidence(observation):
    text = "Component X presenta clase 4 estimada."
    response = {"outputs": [{"outputs": [{"results": {"message": {"text": text}}}]}]}
    client, opener = client_with_response(json.dumps(response).encode())
    prior = explanation_context(observation)
    assert client.explain(observation) == text
    request = opener.open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:7860/api/v1/run/flow-123"
    assert request.get_method() == "POST"
    assert request.get_header("X-api-key") == "secret-key"
    payload = json.loads(request.data)
    evidence = json.loads(payload["input_value"].split("EVIDENCIA:\n")[1])
    assert evidence["source"] == "SCANIA Component X"
    assert evidence["observation"]["vehicle_id"] == "V-001"
    assert evidence["observation"]["time_step"] == 24.0
    assert evidence["published_risk"]["predicted_class"] == 4
    assert evidence["published_risk"]["class_meaning"] == "Hasta 6 pasos"
    assert evidence["published_risk"]["quality"] == 0.75
    assert evidence["observation"]["relevant_anonymized_features"] == [
        "feature_12",
        "feature_19",
    ]
    assert "readings" not in evidence
    assert "work_orders" not in evidence
    assert "secret-key" not in payload["input_value"]
    assert payload["input_type"] == payload["output_type"] == "chat"
    assert opener.open.call_args.kwargs["timeout"] == 15
    assert explanation_context(observation) == prior
    assert "secret-key" not in repr(client.config)


@pytest.mark.parametrize(
    "body",
    [b"invalid-json", b"[]", b"{}", b'{"outputs": null}', b"x" * 1_000_001],
)
def test_malformed_empty_or_oversized_responses_fail_cleanly(observation, body):
    client, _ = client_with_response(body)
    with pytest.raises(LangflowError):
        client.explain(observation)


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("secret-key"),
        URLError("secret-key"),
        HTTPError("http://localhost", 401, "secret-key", {}, None),
    ],
)
def test_network_failures_do_not_expose_credentials(observation, error):
    client, opener = client_with_response(b"")
    opener.open.side_effect = error
    with pytest.raises(LangflowError) as failure:
        client.explain(observation)
    assert "secret-key" not in str(failure.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"connected": False},
        {"stale": True},
        {"state": None},
        {"readout_index": 2},
        {"vehicle_id": "other"},
    ],
)
def test_unavailable_observations_never_call_langflow(observation, changes):
    client, opener = client_with_response(b"{}")
    with pytest.raises(LangflowError):
        client.explain(replace(observation, **changes))
    opener.open.assert_not_called()


def test_scania_source_rejects_readouts_not_matching_published_window(observation):
    names = ("feature_0", "feature_1")
    readout = FeatureReadout(
        DatasetSplit.VALIDATION, "V-001", 24.0, names, (None, 5.0)
    )
    window = FeatureWindow(
        DatasetSplit.VALIDATION,
        "alpha",
        "V-001",
        names,
        (24.0,),
        ((0.0, 5.0),),
        ((True, False),),
    )
    view = ScaniaReplayView(
        DatasetSplit.VALIDATION,
        "V-001",
        "alpha",
        0,
        1,
        readout,
        (readout,),
        4,
        window,
    )
    source = ScaniaExplanationSource.__new__(ScaniaExplanationSource)
    source.split = "validation"
    source.regime = "fedprox"
    source.vectorizer = WindowVectorizer(FeatureConfig(window_size=1))
    features = source.vectorizer.transform(window)
    source.cached = CachedSplit(
        features[None, :],
        np.asarray([4]),
        np.asarray(["alpha"]),
        np.asarray(["V-001"]),
        source.vectorizer.feature_names(names),
    )
    source.scaler = FederatedStandardScaler(
        np.zeros_like(features), np.ones_like(features)
    )
    source._positions = {"V-001": 0}
    source.states = {"V-001": replace(observation.state, quality=0.5)}
    source.provenance = {
        "model_ids": {"alpha": observation.state.risk.model_id},
        "training_id": "training-1",
        "data_version": "data-1",
    }
    current = source.observation(view, "fedprox")
    assert not current.stale
    assert current.unavailable_count == 1
    assert current.relevant_features[0] == "feature_1"
    assert source.observation(replace(view, readout_count=2), "fedprox").stale
    assert source.observation(
        replace(view, current=replace(readout, time_step=25.0)), "fedprox"
    ).stale
    assert source.observation(
        replace(view, window=replace(window, values=((0.0, 6.0),))), "fedprox"
    ).stale


def test_configuration_reads_environment_without_changing_it(monkeypatch):
    names = (
        "MINETWIN_LANGFLOW_URL",
        "MINETWIN_LANGFLOW_FLOW_ID",
        "MINETWIN_LANGFLOW_API_KEY",
        "MINETWIN_LANGFLOW_TIMEOUT_SECONDS",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    assert LangflowConfig.from_environment() is None
    for name, value in zip(
        names, ("http://localhost:7860", "flow-id", "key", "300"), strict=True
    ):
        monkeypatch.setenv(name, value)
    configuration = LangflowConfig.from_environment()
    assert configuration.flow_id == "flow-id"
    assert configuration.timeout_seconds == 300


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example",
        "https://user:password@example.com",
        "https://example.com?key=secret",
        "file:///tmp",
    ],
)
def test_invalid_connections_are_rejected(url):
    with pytest.raises(ValueError):
        LangflowConfig(url, "flow", "key")
