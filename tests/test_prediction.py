from dataclasses import replace
from datetime import timedelta

import pytest

from minetwin.domain import (
    AssetStatus,
    ForecastStatus,
    OperatingState,
    Quality,
    SensorReading,
    TelemetryPacket,
    TruckProfile,
)
from minetwin.prediction import EnginePredictor, PredictorConfig
from minetwin.simulation import SimulationConfig


def sample(minute, indicator, hours=None):
    timestamp = SimulationConfig().start_time + timedelta(minutes=minute)
    return TelemetryPacket(
        event_id=f"sample-{minute}",
        node_id="alpha",
        truck_id="TRUCK-001",
        source_time=timestamp,
        received_time=timestamp,
        operating_state=OperatingState.WAITING,
        cycles=0,
        readings=(
            SensorReading("load_ratio", 0, "1", Quality.VALID),
            SensorReading(
                "engine_temperature", 65 + 25 * indicator, "°C", Quality.VALID
            ),
            SensorReading(
                "engine_vibration", 1 + 2.5 * indicator, "mm/s", Quality.VALID
            ),
            SensorReading(
                "operating_hours",
                minute / 60 if hours is None else hours,
                "h",
                Quality.VALID,
            ),
        ),
    )


def predictor(**overrides):
    return EnginePredictor(
        TruckProfile(),
        PredictorConfig(
            min_samples=4,
            min_operating_span_hours=0.04,
            **overrides,
        ),
    )


def test_prediction_extrapolates_observable_indicator_in_operating_hours():
    monitor = predictor()
    for minute in range(1, 21):
        packet = sample(minute, 0.2 + 0.18 * minute / 60)
        result = monitor.evaluate(packet, packet.source_time)
    assert result.status == ForecastStatus.ESTIMABLE
    assert result.slope_per_hour == pytest.approx(0.18)
    assert result.r_squared == pytest.approx(1)
    assert result.hours_to_threshold == pytest.approx((1 - 0.26) / 0.18)


@pytest.mark.parametrize("slope", [0.0, -0.18])
def test_flat_or_negative_trend_does_not_produce_a_forecast(slope):
    monitor = predictor()
    for minute in range(1, 21):
        packet = sample(minute, 0.5 + slope * minute / 60)
        result = monitor.evaluate(packet, packet.source_time)
    assert result.status == ForecastStatus.NOT_ESTIMABLE
    assert result.hours_to_threshold is None


def test_noisy_unstable_trend_is_not_extrapolated():
    monitor = predictor()
    for minute in range(1, 21):
        packet = sample(minute, 0.1 if minute % 2 else 0.8)
        result = monitor.evaluate(packet, packet.source_time)
    assert result.status == ForecastStatus.NOT_ESTIMABLE
    assert result.r_squared < 0.8


def test_outside_horizon_is_not_clamped_to_a_fake_estimate():
    monitor = predictor(max_horizon_hours=1)
    for minute in range(1, 21):
        packet = sample(minute, 0.2 + 0.18 * minute / 60)
        result = monitor.evaluate(packet, packet.source_time)
    assert result.status == ForecastStatus.OUTSIDE_HORIZON
    assert result.hours_to_threshold is None


def test_valid_observation_at_threshold_returns_zero_without_a_trend():
    packet = sample(1, 1.1)
    result = predictor().evaluate(packet, packet.source_time)
    assert result.status == ForecastStatus.ESTIMABLE
    assert result.hours_to_threshold == 0


def test_waiting_does_not_inflate_the_number_of_operating_samples():
    monitor = predictor()
    for minute in range(1, 21):
        packet = sample(minute, 0.2, hours=0)
        result = monitor.evaluate(packet, packet.source_time)
    assert result.sample_count == 1
    assert result.status == ForecastStatus.NOT_ESTIMABLE


@pytest.mark.parametrize(
    "condition", ["stale", "missing", "maintenance", "failed", "future"]
)
def test_unusable_observations_clear_the_window(condition):
    monitor = predictor()
    for minute in range(1, 10):
        packet = sample(minute, 0.2 + minute / 300)
        monitor.evaluate(packet, packet.source_time)
    packet = sample(10, 0.4)
    now = packet.source_time
    if condition == "stale":
        now += timedelta(seconds=181)
    elif condition == "future":
        now -= timedelta(seconds=1)
    elif condition == "missing":
        packet = replace(
            packet,
            readings=tuple(
                replace(reading, value=None, quality=Quality.MISSING)
                if reading.name == "engine_temperature"
                else reading
                for reading in packet.readings
            ),
        )
    else:
        packet = replace(packet, asset_status=AssetStatus(condition))
    result = monitor.evaluate(packet, now)
    assert result.status == ForecastStatus.NOT_ESTIMABLE
    assert result.hours_to_threshold is None
    next_packet = sample(11, 0.4)
    assert monitor.evaluate(next_packet, next_packet.source_time).sample_count == 1


def test_operating_clock_regression_requires_a_new_window():
    monitor = predictor()
    first = sample(1, 0.1, hours=2)
    monitor.evaluate(first, first.source_time)
    second = sample(2, 0.2, hours=1)
    result = monitor.evaluate(second, second.source_time)
    assert result.status == ForecastStatus.NOT_ESTIMABLE
    assert "retrocedió" in result.reason


def test_large_observation_gap_does_not_bridge_old_samples():
    monitor = predictor()
    first = sample(1, 0.1)
    monitor.evaluate(first, first.source_time)
    later = sample(30, 0.5)
    assert monitor.evaluate(later, later.source_time).sample_count == 1
