import pytest

from minetwin.domain import AssetStatus, OrderStatus, Scenario
from minetwin.maintenance import MaintenanceConfig
from minetwin.operations import (
    CoordinatedFleet,
    FleetPolicy,
    WorkshopConfig,
    run_workshop_comparison,
)
from minetwin.simulation import SimulationConfig


def test_queued_trucks_keep_operating_until_a_bay_is_available():
    fleet = CoordinatedFleet(
        SimulationConfig(),
        MaintenanceConfig(replacement_seconds=120),
        WorkshopConfig(bays=1, minimum_available=8),
        FleetPolicy.P0_NONE,
    )
    fleet.advance()
    first = fleet.trucks["TRUCK-001"]
    second = fleet.trucks["TRUCK-002"]
    fleet.request_maintenance(first.truck_id, urgency=2)
    fleet.request_maintenance(second.truck_id, urgency=1)
    second_hours = second.metrics["operating_hours"]
    fleet.advance()
    assert first.asset_status == AssetStatus.MAINTENANCE
    assert second.current_order.status == OrderStatus.QUEUED
    assert second.asset_status == AssetStatus.AVAILABLE
    fleet.advance()
    assert second.metrics["operating_hours"] > second_hours
    assert fleet.metrics["max_active_bays"] == 1
    assert first.orders[-1].status == OrderStatus.COMPLETED
    assert second.current_order.status == OrderStatus.QUEUED
    fleet.advance()
    assert second.asset_status == AssetStatus.MAINTENANCE


def test_production_uses_completed_cycles_and_profile_capacity():
    fleet = CoordinatedFleet(
        SimulationConfig(),
        policy=FleetPolicy.P0_NONE,
    )
    fleet.advance(25)
    expected = 3 * (150 + 220 + 180)
    assert fleet.metrics["tonnes_delivered"] == pytest.approx(expected)


def test_coordinated_policy_publishes_summaries_without_raw_records():
    fleet = CoordinatedFleet(
        SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION),
        policy=FleetPolicy.P3_COORDINATED,
    )
    fleet.set_connected("alpha", False)
    fleet.advance(2)
    assert len(fleet.published_states) == 6
    assert {state.node_id for state in fleet.published_states} == {"beta", "gamma"}
    assert all(not hasattr(state, "readings") for state in fleet.published_states)
    assert fleet.metrics["published_raw_records"] == 0


def test_workshop_never_exceeds_bays_or_starts_preventive_below_minimum():
    fleet = CoordinatedFleet(
        SimulationConfig(),
        MaintenanceConfig(replacement_seconds=60),
        WorkshopConfig(bays=2, minimum_available=7),
        FleetPolicy.P0_NONE,
    )
    fleet.advance()
    for truck_id in fleet.fleet.truck_ids[:4]:
        fleet.request_maintenance(truck_id)
    fleet.advance(5)
    assert fleet.metrics["max_active_bays"] <= 2
    assert fleet.metrics["minimum_available_observed"] >= 7


def test_comparison_exports_all_policies_with_paired_configuration(tmp_path):
    output = tmp_path / "workshop"
    rows = run_workshop_comparison(
        output,
        SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION),
        2,
    )
    assert [row["policy"] for row in rows] == list(FleetPolicy)
    assert (output / "comparison.csv").is_file()
    for policy in FleetPolicy:
        assert (output / policy / "metrics.json").is_file()
        assert (output / policy / "requests.csv").is_file()
        assert (output / policy / "transfer_ledger.csv").is_file()
