import io
import json
from dataclasses import replace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from minetwin.federation import FederatedFleet
from minetwin.langflow_client import (
    LangflowClient,
    LangflowConfig,
    LangflowError,
    explanation_context,
)


@pytest.fixture
def observation():
    fleet = FederatedFleet()
    fleet.advance(5)
    return fleet.view("TRUCK-001")


def client_with_response(body):
    opener = Mock()
    opener.open.return_value = io.BytesIO(body)
    config = LangflowConfig("http://127.0.0.1:7860", "flow-123", "secret-key")
    return LangflowClient(config, opener), opener


def test_langflow_request_contains_only_observed_evidence_and_does_not_mutate_state(
    observation,
):
    text = "El motor opera dentro de la referencia simulada."
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
    assert evidence["truck_id"] == "TRUCK-001"
    assert len(evidence["component_statuses"]) == 7
    assert evidence["attention_diagnoses"] == []
    assert evidence["deterministic_summary"]["active_alert_components"] == []
    assert len(evidence["deterministic_summary"]["normal_components"]) == 7
    assert (
        evidence["observation"]["source_time"]
        == observation.twin.observation.source_time.isoformat()
    )
    assert "wear" not in evidence
    assert "secret-key" not in payload["input_value"]
    assert payload["input_type"] == payload["output_type"] == "chat"
    assert opener.open.call_args.kwargs["timeout"] == 15
    assert explanation_context(observation) == prior
    assert "secret-key" not in repr(client.config)


@pytest.mark.parametrize(
    "body",
    [b"invalid-json", b"[]", b"{}", b'{"outputs": null}', b"x" * 1_000_001],
    ids=["invalid-json", "list", "empty", "null-outputs", "oversized"],
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
    "changes", [{"connected": False}, {"stale": True}, {"twin": None}]
)
def test_unavailable_observations_never_call_langflow(observation, changes):
    client, opener = client_with_response(b"{}")
    with pytest.raises(LangflowError):
        client.explain(replace(observation, **changes))
    opener.open.assert_not_called()


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
