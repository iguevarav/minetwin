import json
from datetime import timedelta

import pytest

from minetwin import MineTwin, SimulationConfig
from minetwin.domain import EngineStatus, OperatingState, Quality, Scenario
from minetwin.telemetry import TelemetryError, adapt_alpha_json


def test_same_seed_and_scenario_reproduce_observations_and_alerts():
    config = SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION)
    first, second = MineTwin(config), MineTwin(config)
    first.advance(180)
    second.advance(180)
    assert first.history == second.history
    assert first.raw_history == second.raw_history
    assert first.alerts == second.alerts
    assert first.alert_count == 1


def test_seed_changes_noise_without_changing_operational_cycle():
    first = MineTwin(SimulationConfig(seed=1))
    second = MineTwin(SimulationConfig(seed=2))
    first.advance(50)
    second.advance(50)
    assert first.history != second.history
    assert [p.operating_state for p in first.history] == [
        p.operating_state for p in second.history
    ]


def test_full_cycle_unloads_and_counts_only_operating_time():
    simulation = MineTwin()
    simulation.advance(25)
    assert simulation.twin.observation.cycles == 1
    assert simulation.twin.observation.operating_state == OperatingState.WAITING
    assert simulation.twin.observation.reading(
        "operating_hours"
    ).value == pytest.approx(23 / 60)
    assert {packet.operating_state for packet in simulation.history} == set(
        OperatingState
    )
    for packet in simulation.history:
        load = packet.reading("load_ratio").value
        if packet.operating_state == OperatingState.HAULING:
            assert load == 1
        elif packet.operating_state in (
            OperatingState.WAITING,
            OperatingState.RETURNING,
        ):
            assert load == 0


def test_large_step_integrates_operating_time_across_multiple_cycles():
    simulation = MineTwin(SimulationConfig(step_seconds=3_000))
    twin = simulation.advance()
    assert twin.observation.cycles == 2
    assert twin.observation.reading("operating_hours").value == pytest.approx(46 / 60)


@pytest.mark.parametrize("seed", range(10))
def test_normal_operation_does_not_raise_engine_alerts(seed):
    simulation = MineTwin(SimulationConfig(seed=seed))
    simulation.advance(300)
    assert simulation.alert_count == 0
    assert simulation.twin.diagnosis.status == EngineStatus.NORMAL


def test_histories_are_bounded_and_queries_do_not_advance_time():
    simulation = MineTwin(SimulationConfig(history_limit=5))
    simulation.advance(100)
    observation = simulation.twin
    now = simulation.time
    for _ in range(10):
        assert len(simulation.history) == len(simulation.raw_history) == 5
        assert simulation.twin is observation
        assert simulation.time == now
        assert simulation.alerts == ()


@pytest.fixture
def raw_packet():
    simulation = MineTwin()
    simulation.advance()
    return json.loads(simulation.raw_history[-1])


def ingest_sample(simulation, raw_packet, minute, temperature_f=230, vibration=3.0):
    packet = dict(raw_packet)
    capture = simulation.config.start_time + timedelta(minutes=minute)
    packet.update(
        source_time=capture.isoformat(),
        event_id=f"test-{minute}",
        operating_state="waiting",
        engine_temperature_f=temperature_f,
        engine_vibration_mm_s=vibration,
    )
    return simulation.ingest(json.dumps(packet), capture)


def test_adapter_converts_fahrenheit_and_preserves_missing_value(raw_packet):
    raw_packet["engine_temperature_f"] = 212
    raw_packet.pop("engine_vibration_mm_s")
    received = SimulationConfig().start_time + timedelta(minutes=1)
    packet = adapt_alpha_json(json.dumps(raw_packet), received)
    assert packet.reading("engine_temperature").value == pytest.approx(100)
    missing = packet.reading("engine_vibration")
    assert missing.value is None
    assert missing.quality == Quality.MISSING


@pytest.mark.parametrize("invalid", ["hot", True, -1_000, [], {"value": 90}])
def test_invalid_sensor_values_do_not_become_zero(raw_packet, invalid):
    simulation = MineTwin()
    twin = ingest_sample(simulation, raw_packet, 1, temperature_f=invalid)
    assert twin.observation.reading("engine_temperature").quality == Quality.INVALID
    assert twin.observation.reading("engine_temperature").value is None
    assert twin.diagnosis.status == EngineStatus.UNKNOWN


def test_alert_requires_persistence_and_recovery_uses_hysteresis(raw_packet):
    simulation = MineTwin()
    for minute in (1, 2, 3):
        twin = ingest_sample(simulation, raw_packet, minute)
        assert twin.diagnosis.status == EngineStatus.WATCH
    twin = ingest_sample(simulation, raw_packet, 4)
    assert twin.diagnosis.status == EngineStatus.ALERT
    for minute in range(5, 10):
        ingest_sample(simulation, raw_packet, minute)
    assert simulation.alert_count == 1
    for minute in (10, 11, 12):
        twin = ingest_sample(simulation, raw_packet, minute, 149, 1.0)
        assert twin.diagnosis.status == EngineStatus.ALERT
    twin = ingest_sample(simulation, raw_packet, 13, 149, 1.0)
    assert twin.diagnosis.status == EngineStatus.NORMAL
    assert simulation.alerts[0].closed_at == twin.observation.source_time


def test_missing_observation_breaks_persistence_without_clearing_active_alert(
    raw_packet,
):
    simulation = MineTwin()
    for minute in (1, 2):
        ingest_sample(simulation, raw_packet, minute)
    twin = ingest_sample(simulation, raw_packet, 3, temperature_f=None)
    assert twin.diagnosis.status == EngineStatus.UNKNOWN
    for minute in (4, 5, 6):
        assert ingest_sample(simulation, raw_packet, minute).diagnosis.status != (
            EngineStatus.ALERT
        )
    ingest_sample(simulation, raw_packet, 7)
    twin = ingest_sample(simulation, raw_packet, 8, temperature_f=None)
    assert twin.diagnosis.status == EngineStatus.UNKNOWN
    assert simulation.alerts[-1].closed_at is None


def test_gap_between_observations_does_not_count_as_persistent_evidence(raw_packet):
    simulation = MineTwin()
    ingest_sample(simulation, raw_packet, 1)
    twin = ingest_sample(simulation, raw_packet, 20)
    assert twin.diagnosis.status == EngineStatus.WATCH
    assert simulation.alert_count == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", "1"),
        ("truck_id", "unknown"),
    ],
)
def test_invalid_packet_identity_does_not_change_twin(raw_packet, field, value):
    simulation = MineTwin()
    simulation.advance()
    original = simulation.twin
    raw_packet[field] = value
    with pytest.raises(TelemetryError):
        simulation.ingest(json.dumps(raw_packet), simulation.time)
    assert simulation.twin is original
    assert len(simulation.history) == 1


def test_old_or_repeated_observation_does_not_replace_current_twin():
    simulation = MineTwin()
    simulation.advance(2)
    original = simulation.twin
    for raw in simulation.raw_history:
        with pytest.raises(TelemetryError):
            simulation.ingest(raw, simulation.time)
    assert simulation.twin is original
    assert len(simulation.history) == 2


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_step_count_does_not_advance_simulation(steps):
    simulation = MineTwin()
    with pytest.raises(ValueError):
        simulation.advance(steps)
    assert simulation.time == simulation.config.start_time
    assert simulation.twin is None


def test_closed_alert_history_is_bounded(raw_packet):
    simulation = MineTwin(SimulationConfig(alert_limit=2))
    for episode in range(4):
        start = episode * 8 + 1
        for minute in range(start, start + 4):
            ingest_sample(simulation, raw_packet, minute)
        for minute in range(start + 4, start + 8):
            ingest_sample(simulation, raw_packet, minute, 149, 1.0)
    assert simulation.alert_count == 4
    assert len(simulation.alerts) == 2
    assert all(alert.closed_at is not None for alert in simulation.alerts)
