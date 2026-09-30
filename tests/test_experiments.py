import csv
import io
import json
from zipfile import ZipFile

import pytest

from minetwin import MineTwin, SimulationConfig
from minetwin.domain import MaintenancePolicy, Scenario
from minetwin.experiments import export_snapshot, run_comparison, run_experiment


def test_comparison_exports_reproducible_metrics_and_observed_failure_reference(
    tmp_path,
):
    config = SimulationConfig(
        scenario=Scenario.ENGINE_DEGRADATION, history_limit=2, event_limit=1
    )
    output = tmp_path / "comparison"
    results = run_comparison(output, config, 450)
    assert len(results) == 3
    reference, threshold, predictive = results
    assert reference["policy"] == MaintenancePolicy.NONE
    assert reference["failures"] == 1
    assert reference["failure_reference_observed"]
    assert reference["forecast_mae_operating_hours"] < 0.15
    assert 0 < reference["forecast_coverage"] < 1
    assert threshold["interventions"] > 0
    assert predictive["interventions"] > 0
    assert threshold["scheduled_seconds"] == predictive["scheduled_seconds"] == 450 * 60
    assert threshold["forecast_mae_operating_hours"] is None
    for policy in MaintenancePolicy:
        directory = output / policy.value
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["config"]["seed"] == 42
        assert len(manifest["code_sha256"]) == 64
        with (directory / "telemetry.csv").open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 450
    with (output / "threshold" / "events.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        events = list(csv.DictReader(stream))
    assert len(events) > config.event_limit
    assert any(
        event["kind"] == "alert_opened" and event["alert_id"] for event in events
    )
    assert any(
        event["kind"] == "alert_closed" and event["alert_id"] for event in events
    )
    assert (
        sum(event["kind"] == "maintenance_completed" for event in events)
        == threshold["interventions"]
    )
    repeat = run_experiment(tmp_path / "repeat", config, 450)
    assert repeat == reference


def test_unobserved_failure_is_not_given_a_reference_error(tmp_path):
    metrics = run_experiment(tmp_path / "normal", SimulationConfig(), 60)
    assert metrics["failures"] == 0
    assert not metrics["failure_reference_observed"]
    assert metrics["forecast_mae_operating_hours"] is None


def test_snapshot_exports_only_retained_history_without_mutating_simulation():
    simulation = MineTwin(SimulationConfig(history_limit=3))
    simulation.advance(10)
    order = simulation.create_order()
    simulation.start_order(order.id)
    simulation.advance()
    before = (simulation.time, simulation.twin, simulation.orders, simulation.events)
    with ZipFile(io.BytesIO(export_snapshot(simulation))) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "metrics.json",
            "telemetry.csv",
            "predictions.csv",
            "events.csv",
            "orders.csv",
        }
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["scope"] == "retained_session_history"
        rows = list(
            csv.DictReader(io.StringIO(archive.read("telemetry.csv").decode("utf-8")))
        )
        assert len(rows) == 3
        assert rows[-1]["asset_status"] == "maintenance"
        assert rows[-1]["engine_temperature"] == ""
    assert (
        simulation.time,
        simulation.twin,
        simulation.orders,
        simulation.events,
    ) == before


def test_export_does_not_overwrite_an_existing_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "important.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_experiment(output, SimulationConfig(), 1)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert list(output.iterdir()) == [marker]


def test_interrupted_experiment_is_marked_incomplete(tmp_path, monkeypatch):
    def fail_advance(self, steps=1):
        raise RuntimeError("Interrupted run")

    monkeypatch.setattr(MineTwin, "advance", fail_advance)
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError):
        run_experiment(output, SimulationConfig(), 10)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
