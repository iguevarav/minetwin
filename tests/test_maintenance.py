from datetime import timedelta

import pytest

from minetwin import MineTwin, SimulationConfig
from minetwin.domain import (
    AssetStatus,
    ForecastStatus,
    Intervention,
    MaintenancePolicy,
    OrderStatus,
    Scenario,
)
from minetwin.maintenance import MaintenanceConfig


def degraded(**kwargs):
    return MineTwin(SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION), **kwargs)


def test_open_order_does_not_stop_operation_and_replacement_consumes_time():
    simulation = degraded()
    simulation.advance(180)
    order = simulation.create_order()
    assert simulation.asset_status == AssetStatus.AVAILABLE
    simulation.advance()
    hours = simulation.twin.observation.reading("operating_hours").value
    cycles = simulation.twin.observation.cycles
    history_before = simulation.history
    started = simulation.start_order(order.id)
    assert started.status == OrderStatus.IN_PROGRESS
    simulation.advance(19)
    assert simulation.asset_status == AssetStatus.MAINTENANCE
    assert simulation.twin.observation.reading("operating_hours").value == hours
    assert simulation.twin.observation.cycles == cycles
    assert simulation.prediction.status == ForecastStatus.NOT_ESTIMABLE
    simulation.advance()
    assert simulation.asset_status == AssetStatus.AVAILABLE
    assert simulation.current_order is None
    assert simulation.orders[-1].status == OrderStatus.COMPLETED
    assert simulation.metrics["maintenance_seconds"] == 1200
    assert simulation.twin.observation.reading("operating_hours").value == hours
    assert simulation.history[: len(history_before)] == history_before
    assert simulation.prediction.sample_count == 1
    assert simulation.prediction.status == ForecastStatus.NOT_ESTIMABLE
    assert abs(simulation.prediction.indicator) < 0.03
    simulation.advance(25)
    assert simulation.twin.observation.reading("operating_hours").value > hours
    assert simulation.metrics["interventions"] == 1


def test_repeated_actions_cannot_duplicate_orders_or_repair_effects():
    simulation = degraded()
    simulation.advance(180)
    order = simulation.create_order()
    assert simulation.create_order() == order
    started = simulation.start_order(order.id)
    assert simulation.start_order(order.id) == started
    assert sum(event.kind == "maintenance_started" for event in simulation.events) == 1
    with pytest.raises(ValueError, match="duración"):
        simulation.complete_order(order.id)
    assert simulation.current_order == started
    with pytest.raises(ValueError):
        simulation.cancel_order(order.id)
    simulation.advance(30)
    before = (
        simulation.prediction,
        simulation.metrics,
        simulation.events,
        simulation.time,
    )
    completed = simulation.complete_order(order.id)
    assert completed.status == OrderStatus.COMPLETED
    assert (
        simulation.prediction,
        simulation.metrics,
        simulation.events,
        simulation.time,
    ) == before


def test_cancellation_is_terminal_and_does_not_change_operation():
    simulation = MineTwin()
    simulation.advance(10)
    before = (simulation.time, simulation.twin, simulation.metrics)
    order = simulation.create_order()
    cancelled = simulation.cancel_order(order.id)
    assert cancelled.status == OrderStatus.CANCELLED
    assert simulation.cancel_order(order.id) == cancelled
    assert (simulation.time, simulation.twin, simulation.metrics) == before
    with pytest.raises(ValueError):
        simulation.start_order(order.id)


def test_unsupported_component_is_rejected_before_any_state_change():
    simulation = MineTwin()
    with pytest.raises(ValueError, match="motor"):
        simulation.create_order(component="hydraulics")
    assert simulation.orders == simulation.events == ()
    assert simulation.asset_status == AssetStatus.AVAILABLE


def test_completion_inside_a_step_preserves_exact_downtime_and_remaining_operation():
    simulation = MineTwin(maintenance_config=MaintenanceConfig(replacement_seconds=90))
    simulation.advance(10)
    hours = simulation.twin.observation.reading("operating_hours").value
    start = simulation.time
    order = simulation.create_order()
    simulation.start_order(order.id)
    simulation.advance(2)
    assert simulation.time == start + timedelta(minutes=2)
    assert simulation.orders[-1].closed_at == start + timedelta(seconds=90)
    assert simulation.metrics["maintenance_seconds"] == 90
    assert simulation.twin.observation.reading(
        "operating_hours"
    ).value == pytest.approx(hours + 30 / 3600)


def test_failure_stops_operation_once_and_corrective_repair_restores_it():
    simulation = degraded()
    simulation.advance(420)
    assert simulation.asset_status == AssetStatus.FAILED
    assert simulation.metrics["failures"] == 1
    assert simulation.twin.observation.reading(
        "operating_hours"
    ).value == pytest.approx(1 / 0.18)
    failure = next(event for event in simulation.events if event.kind == "failure")
    assert failure.timestamp < simulation.time
    hours = simulation.twin.observation.reading("operating_hours").value
    cycles = simulation.twin.observation.cycles
    simulation.advance(10)
    assert simulation.metrics["failures"] == 1
    assert simulation.twin.observation.cycles == cycles
    assert simulation.twin.observation.reading("operating_hours").value == hours
    order = simulation.create_order()
    started = simulation.start_order(order.id)
    assert started.corrective
    assert started.duration_seconds == 3600
    simulation.advance(60)
    assert simulation.asset_status == AssetStatus.AVAILABLE
    assert simulation.metrics["maintenance_seconds"] == 3600
    simulation.advance(25)
    assert simulation.twin.observation.reading("operating_hours").value > hours


def test_partial_repair_reduces_observed_degradation_without_resetting_operating_history():
    simulation = degraded(
        maintenance_config=MaintenanceConfig(partial_repair_seconds=60)
    )
    simulation.advance(180)
    before = simulation.prediction.indicator
    hours = simulation.twin.observation.reading("operating_hours").value
    order = simulation.create_order(Intervention.PARTIAL_REPAIR)
    simulation.start_order(order.id)
    simulation.advance()
    assert simulation.prediction.indicator == pytest.approx(before * 0.25, abs=0.03)
    assert simulation.prediction.indicator > 0.05
    assert simulation.twin.observation.reading("operating_hours").value == hours


def test_time_accounting_balances_after_failures_and_repair():
    simulation = degraded()
    simulation.advance(420)
    order = simulation.create_order()
    simulation.start_order(order.id)
    simulation.advance(90)
    metrics = simulation.metrics
    assert sum(
        metrics[key]
        for key in ("available_seconds", "failed_seconds", "maintenance_seconds")
    ) == pytest.approx(metrics["scheduled_seconds"])
    assert 0 < metrics["availability"] < 1


def test_sensor_noise_at_a_given_time_is_independent_of_repair_sampling():
    uninterrupted = MineTwin()
    repaired = MineTwin(maintenance_config=MaintenanceConfig(replacement_seconds=90))
    uninterrupted.advance(12)
    repaired.advance(10)
    order = repaired.create_order()
    repaired.start_order(order.id)
    repaired.advance(2)
    for signal, baseline_index in (("engine_temperature", 0), ("engine_vibration", 1)):
        residuals = []
        for simulation in (uninterrupted, repaired):
            packet = simulation.twin.observation
            baseline = simulation.profile.engine_baseline(packet.operating_state)[
                baseline_index
            ]
            residuals.append(packet.reading(signal).value - baseline)
        assert residuals[0] == pytest.approx(residuals[1])


@pytest.mark.parametrize(
    "policy", [MaintenancePolicy.THRESHOLD, MaintenancePolicy.PREDICTIVE]
)
def test_automatic_policies_use_order_lifecycle_and_operational_downtime(policy):
    simulation = degraded(policy=policy)
    simulation.advance(450)
    assert simulation.metrics["interventions"] > 0
    assert simulation.metrics["maintenance_seconds"] > 0
    assert any(event.kind == "maintenance_completed" for event in simulation.events)
    assert all(
        order.duration_seconds == 1200
        for order in simulation.orders
        if order.status == OrderStatus.COMPLETED
    )
