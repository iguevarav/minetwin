import json
from datetime import timedelta

import pytest

from minetwin.adapters import adapt_telemetry, encode_payload
from minetwin.domain import (
    AssetStatus,
    EngineStatus,
    ForecastStatus,
    OrderStatus,
    Quality,
    Scenario,
    WireFormat,
)
from minetwin.federation import PROFILES, FederatedFleet
from minetwin.prediction import PredictorConfig
from minetwin.services import MineTwin
from minetwin.simulation import SimulationConfig
from minetwin.telemetry import TelemetryError


@pytest.mark.parametrize("profile", PROFILES)
def test_formats_preserve_canonical_values_and_missing_readings(profile):
    source = MineTwin(profile=profile)
    source.advance(8)
    payload = json.loads(source.raw_history[-1])
    payload["tire_FL_kpa"] = None
    payload["tire_FR_kpa"] = "defective"
    packets = [
        adapt_telemetry(
            encode_payload(payload, fmt, profile), source.time, fmt, profile
        )
        for fmt in WireFormat
    ]
    for packet in packets:
        assert packet.event_id == packets[0].event_id
        assert packet.source_time == packets[0].source_time
        assert packet.operating_state == packets[0].operating_state
        for reading, reference in zip(
            packet.readings, packets[0].readings, strict=True
        ):
            assert (reading.name, reading.unit, reading.quality) == (
                reference.name,
                reference.unit,
                reference.quality,
            )
            assert reading.value == (
                pytest.approx(reference.value, abs=1e-8)
                if reference.value is not None
                else None
            )


@pytest.mark.parametrize("wire_format", [WireFormat.XML, WireFormat.CSV])
def test_unknown_units_and_malformed_documents_are_rejected(wire_format):
    simulation = MineTwin()
    simulation.advance()
    raw = encode_payload(
        json.loads(simulation.raw_history[-1]), wire_format, simulation.profile
    )
    bad_unit = (
        raw.replace("temperature_kelvin", "temperature_c")
        if wire_format == WireFormat.XML
        else raw.replace("pressure_FL_bar", "pressure_FL_psi")
    )
    for defective in (bad_unit, raw + raw, "invalid"):
        with pytest.raises(TelemetryError):
            adapt_telemetry(defective, simulation.time, wire_format, simulation.profile)


def test_fleet_profiles_are_distinct_and_normal_operation_has_no_false_alerts():
    fleet = FederatedFleet()
    fleet.advance(180)
    assert len(fleet.truck_ids) == 9
    assert {node.wire_format for node in fleet.nodes.values()} == set(WireFormat)
    assert {fleet.view(truck).profile.capacity_tonnes for truck in fleet.truck_ids} == {
        150,
        180,
        220,
    }
    for truck_id in fleet.truck_ids:
        view = fleet.view(truck_id)
        assert view.twin.observation.truck_id == truck_id
        assert view.twin.diagnosis.status == EngineStatus.NORMAL
        assert all(
            item.status == EngineStatus.NORMAL for item in view.twin.component_diagnoses
        )
        assert not view.alerts
        assert view.age_seconds == 0


def test_offline_nodes_continue_locally_without_publishing_and_reconnect_current_state():
    fleet = FederatedFleet(SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION))
    fleet.advance(90)
    connection = fleet.access("TRUCK-001")
    prior = fleet.view("TRUCK-001")
    quality = fleet.quality("alpha")
    bound_create = connection.create_order
    fleet.set_connected("alpha", False)
    fleet.advance(10)
    stale = fleet.view("TRUCK-001")
    assert stale.twin is prior.twin
    assert stale.age_seconds == 600
    assert stale.stale and not stale.connected
    assert stale.prediction.status == ForecastStatus.NOT_ESTIMABLE
    assert stale.prediction.hours_to_threshold is None
    assert fleet.quality("alpha") == quality
    assert fleet.nodes["alpha"].trucks["TRUCK-001"].twin.version == 100
    assert fleet.view("TRUCK-004").twin.version == 100
    with pytest.raises(ConnectionError):
        _ = connection.history
    with pytest.raises(ConnectionError):
        bound_create()
    fleet.set_connected("alpha", True)
    current = fleet.view("TRUCK-001")
    assert current.twin.version == 100
    assert not current.stale
    assert current.age_seconds == 0
    assert len(connection.history) == 100
    assert fleet.quality("alpha").accepted == 300


def test_started_maintenance_completes_offline_and_other_remote_actions_are_blocked():
    fleet = FederatedFleet()
    fleet.advance(10)
    connection = fleet.access("TRUCK-001")
    order = connection.create_order(component="tire:FL")
    start, cancel = connection.start_order, connection.cancel_order
    fleet.set_connected("alpha", False)
    for callback in (start, cancel):
        with pytest.raises(ConnectionError):
            callback(order.id)
    fleet.set_connected("alpha", True)
    connection.start_order(order.id)
    fleet.set_connected("alpha", False)
    fleet.advance(10)
    local = fleet.nodes["alpha"].trucks["TRUCK-001"]
    assert local.orders[-1].status == OrderStatus.COMPLETED
    assert fleet.view("TRUCK-001").orders[-1].status == OrderStatus.IN_PROGRESS
    fleet.set_connected("alpha", True)
    assert connection.orders[-1].status == OrderStatus.COMPLETED
    assert connection.asset_status == AssetStatus.AVAILABLE


def test_packet_quality_counts_duplicates_late_rejected_and_missing_without_overwriting():
    fleet = FederatedFleet()
    fleet.advance(2)
    node = fleet.nodes["alpha"]
    truck = node.trucks["TRUCK-001"]
    previous = truck.twin
    assert not node.receive(truck.truck_id, truck.raw_history[-1], fleet.time)
    old = json.loads(truck.raw_history[0])
    old["event_id"] = "late-new-id"
    assert not node.receive(truck.truck_id, json.dumps(old), fleet.time)
    assert not node.receive(truck.truck_id, "broken", fleet.time)
    wrong = json.loads(truck.raw_history[-1])
    wrong["truck_id"] = "TRUCK-009"
    assert not node.receive(truck.truck_id, json.dumps(wrong), fleet.time)
    assert truck.twin is previous
    quality = truck.quality.snapshot()
    assert (quality.accepted, quality.duplicates, quality.late, quality.rejected) == (
        2,
        1,
        1,
        2,
    )
    assert (quality.required, quality.present, quality.valid) == (24, 24, 24)
    payload = json.loads(truck.raw_history[-1])
    payload.update(
        event_id="new",
        source_time=(fleet.time + timedelta(minutes=1)).isoformat(),
        load_ratio=1,
    )
    payload["tire_FL_kpa"] = None
    payload["tire_FR_kpa"] = "broken"
    assert node.receive(
        truck.truck_id, json.dumps(payload), fleet.time + timedelta(minutes=1)
    )
    quality = truck.quality.snapshot()
    assert (quality.required, quality.present, quality.valid) == (36, 35, 34)
    assert quality.completeness == pytest.approx(35 / 36)
    assert quality.validity == pytest.approx(34 / 35)
    assert truck.twin.observation.reading("tire_FL_kpa").quality == Quality.MISSING


def test_event_identity_is_unique_within_node_and_bad_packet_does_not_stop_fleet():
    fleet = FederatedFleet()
    fleet.advance()
    node = fleet.nodes["alpha"]
    second = node.trucks["TRUCK-002"]
    payload = json.loads(second.raw_history[-1])
    payload["event_id"] = node.trucks["TRUCK-001"].twin.observation.event_id
    previous = second.twin
    assert not node.receive(second.truck_id, json.dumps(payload), fleet.time)
    assert second.quality.duplicates == 1
    assert second.twin is previous
    fleet.demonstrate_bad_packet("beta")
    assert fleet.quality("beta").rejected == 1
    fleet.advance()
    assert all(fleet.view(truck).twin.version == 2 for truck in fleet.truck_ids)


def test_caducity_uses_observation_time_even_if_a_view_is_published():
    fleet = FederatedFleet(predictor_config=PredictorConfig(stale_after_seconds=90))
    fleet.advance()
    fleet.time += timedelta(seconds=91)
    fleet.publish()
    assert fleet.view("TRUCK-001").stale


def test_retention_is_bounded_without_losing_cumulative_quality_counts():
    fleet = FederatedFleet(SimulationConfig(history_limit=5))
    fleet.advance(20)
    truck = fleet.local("TRUCK-001")
    assert (
        len(truck.history)
        == len(truck.raw_history)
        == len(truck.prediction_history)
        == 5
    )
    assert len(truck.quality._seen) <= 5
    assert len(fleet.nodes["alpha"]._seen) <= 15
    assert fleet.quality("alpha").accepted == 60
    assert len(fleet._views) == 9


@pytest.mark.parametrize(
    ("scenario", "component"),
    [
        (Scenario.BRAKE_STRESS, "brakes"),
        (Scenario.TIRE_LEAK, "tire:FL"),
    ],
)
def test_component_anomalies_and_repairs_are_scoped(scenario, component):
    simulation = MineTwin(SimulationConfig(scenario=scenario))
    simulation.advance(100)
    diagnoses = {
        item.component: item.status for item in simulation.twin.component_diagnoses
    }
    assert diagnoses[component] == EngineStatus.ALERT
    assert all(
        status == EngineStatus.NORMAL
        for name, status in diagnoses.items()
        if name != component
    )
    assert simulation.twin.diagnosis.status == EngineStatus.NORMAL
    assert {alert.component for alert in simulation.alerts} == {component}
    before_hours = simulation.metrics["operating_hours"]
    wrong_component = "tire:FR"
    order = simulation.create_order(component=wrong_component)
    simulation.start_order(order.id)
    simulation.advance(10)
    assert simulation.metrics["operating_hours"] == before_hours
    simulation.advance(4)
    assert (
        next(
            item
            for item in simulation.twin.component_diagnoses
            if item.component == component
        ).status
        == EngineStatus.ALERT
    )
    order = simulation.create_order(component=component)
    started = simulation.start_order(order.id)
    simulation.advance(started.duration_seconds // simulation.config.step_seconds)
    assert simulation.orders[-1].status == OrderStatus.COMPLETED
    assert (
        next(
            item
            for item in simulation.twin.component_diagnoses
            if item.component == component
        ).status
        == EngineStatus.NORMAL
    )
    assert all(
        alert.closed_at for alert in simulation.alerts if alert.component == component
    )


def test_repairing_brakes_does_not_repair_a_failed_engine():
    simulation = MineTwin(SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION))
    simulation.advance(400)
    assert simulation.asset_status == AssetStatus.FAILED
    order = simulation.create_order(component="brakes")
    simulation.start_order(order.id)
    simulation.advance(15)
    assert simulation.asset_status == AssetStatus.FAILED
    assert simulation.metrics["failures"] == 1


def test_missing_auxiliary_sensor_preserves_engine_forecast_and_missing_engine_preserves_brakes():
    engine = MineTwin(SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION))
    engine.advance(100)
    payload = json.loads(engine.raw_history[-1])
    payload.update(
        event_id="missing-tire",
        source_time=(engine.time + timedelta(minutes=1)).isoformat(),
        tire_FL_kpa=None,
    )
    engine.ingest(json.dumps(payload), engine.time + timedelta(minutes=1))
    assert engine.prediction.status == ForecastStatus.ESTIMABLE
    assert (
        next(
            item
            for item in engine.twin.component_diagnoses
            if item.component == "tire:FL"
        ).status
        == EngineStatus.UNKNOWN
    )
    brakes = MineTwin(SimulationConfig(scenario=Scenario.BRAKE_STRESS))
    brakes.advance(100)
    payload = json.loads(brakes.raw_history[-1])
    payload.update(
        event_id="missing-engine",
        source_time=(brakes.time + timedelta(minutes=1)).isoformat(),
        engine_temperature_f=None,
    )
    brakes.ingest(json.dumps(payload), brakes.time + timedelta(minutes=1))
    assert brakes.prediction.status == ForecastStatus.NOT_ESTIMABLE
    assert brakes.twin.component_diagnoses[0].status == EngineStatus.ALERT


def test_wrong_identity_and_future_observations_cannot_replace_the_twin():
    truck = MineTwin()
    truck.advance()
    prior = truck.twin
    for changes in (
        {"truck_id": "TRUCK-008"},
        {"node_id": "beta"},
        {"source_time": (truck.time + timedelta(seconds=1)).isoformat()},
    ):
        payload = json.loads(truck.raw_history[-1])
        payload.update(changes)
        with pytest.raises(TelemetryError):
            truck.ingest(json.dumps(payload), truck.time)
        assert truck.twin is prior


def test_inconsistent_load_is_counted_separately_from_numeric_validity():
    simulation = MineTwin()
    simulation.advance()
    payload = json.loads(simulation.raw_history[-1])
    payload.update(
        event_id="contradiction",
        load_ratio=1,
        source_time=(simulation.time + timedelta(minutes=1)).isoformat(),
    )
    simulation.ingest(json.dumps(payload), simulation.time + timedelta(minutes=1))
    quality = simulation.quality.snapshot()
    assert quality.inconsistent == 1
    assert quality.validity == 1
    assert quality.completeness == 1
    assert any("Carga incompatible" in issue for issue in simulation.quality.issues)


def test_chart_markers_follow_the_signal_component():
    from minetwin.visuals import signal_chart

    simulation = MineTwin(SimulationConfig(scenario=Scenario.BRAKE_STRESS))
    simulation.advance(100)
    assert simulation.alerts
    engine_chart = signal_chart(simulation, "engine_temperature", "Motor", "°C")
    brakes_chart = signal_chart(simulation, "brake_temperature", "Frenos", "°C")
    assert not engine_chart.layout.shapes
    assert len(brakes_chart.layout.shapes) == 1
