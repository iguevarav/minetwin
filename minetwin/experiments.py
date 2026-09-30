import csv
import hashlib
import io
import json
from contextlib import ExitStack
from dataclasses import asdict, fields
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from minetwin.domain import (
    WHEELS,
    AssetStatus,
    EnginePrediction,
    ForecastStatus,
    MaintenancePolicy,
    SimulationEvent,
    TelemetryPacket,
    WorkOrder,
)
from minetwin.maintenance import MaintenanceConfig
from minetwin.prediction import PredictorConfig
from minetwin.services import MineTwin
from minetwin.simulation import SimulationConfig

TELEMETRY_FIELDS = (
    "event_id",
    "source_time",
    "truck_id",
    "asset_status",
    "operating_state",
    "cycles",
    "load_ratio",
    "engine_temperature",
    "engine_vibration",
    "operating_hours",
    "brake_temperature",
    "slope_percent",
    *(f"tire_pressure_{wheel}" for wheel in WHEELS),
)
PREDICTION_FIELDS = tuple(field.name for field in fields(EnginePrediction))
EVENT_FIELDS = tuple(field.name for field in fields(SimulationEvent))
ORDER_FIELDS = tuple(field.name for field in fields(WorkOrder))


def _json_text(value: object) -> str:
    return json.dumps(
        value, default=lambda item: item.isoformat(), indent=2, allow_nan=False
    )


def _csv_row(values: dict) -> dict:
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in values.items()
    }


def _telemetry_row(packet: TelemetryPacket) -> dict:
    return {
        "event_id": packet.event_id,
        "source_time": packet.source_time.isoformat(),
        "truck_id": packet.truck_id,
        "asset_status": packet.asset_status,
        "operating_state": packet.operating_state,
        "cycles": packet.cycles,
        **{reading.name: reading.value for reading in packet.readings},
    }


def _manifest(simulation: MineTwin) -> dict:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "config": asdict(simulation.config),
        "maintenance": asdict(simulation.maintenance_config),
        "predictor": asdict(simulation.predictor_config),
        "policy": simulation.policy,
        "profile": asdict(simulation.profile),
        "truck_id": simulation.truck_id,
        "node_id": simulation.node_id,
        "wire_format": simulation.wire_format,
        "code_sha256": digest.hexdigest(),
        "dependencies": {name: version(name) for name in ("numpy", "pydantic")},
        "assumptions": [
            "Synthetic engine degradation; no industrial calibration.",
            "Sensor noise is keyed by seed and simulated timestamp.",
            "Forecast extrapolates an observable indicator to a configured threshold.",
            "Failure-reference errors apply only to the trajectory without intervention.",
        ],
    }


def _csv_text(rows, fieldnames) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def export_snapshot(simulation: MineTwin) -> bytes:
    manifest = {
        **_manifest(simulation),
        "scope": "retained_session_history",
        "time": simulation.time,
    }
    contents = {
        "manifest.json": _json_text(manifest),
        "metrics.json": _json_text(simulation.metrics),
        "telemetry.csv": _csv_text(
            (_telemetry_row(packet) for packet in simulation.history), TELEMETRY_FIELDS
        ),
        "predictions.csv": _csv_text(
            (_csv_row(asdict(item)) for item in simulation.prediction_history),
            PREDICTION_FIELDS,
        ),
        "events.csv": _csv_text(
            (_csv_row(asdict(item)) for item in simulation.events), EVENT_FIELDS
        ),
        "orders.csv": _csv_text(
            (_csv_row(asdict(item)) for item in simulation.orders), ORDER_FIELDS
        ),
    }
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content.encode("utf-8"))
    return buffer.getvalue()


def _score_forecasts(output: Path, policy: MaintenancePolicy) -> dict:
    failure_hours = None
    if policy == MaintenancePolicy.NONE:
        with (output / "events.csv").open(encoding="utf-8", newline="") as stream:
            for event in csv.DictReader(stream):
                if event["kind"] == "failure":
                    failure_hours = float(event["operating_hours"])
                    break
    eligible = estimated = scored = 0
    absolute_error = 0.0
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        for prediction in csv.DictReader(stream):
            if prediction["evaluation_point"] != "True":
                continue
            eligible += 1
            if prediction["status"] != ForecastStatus.ESTIMABLE:
                continue
            estimated += 1
            if failure_hours is not None:
                true_remaining = failure_hours - float(prediction["operating_hours"])
                if true_remaining >= 0:
                    absolute_error += abs(
                        float(prediction["hours_to_threshold"]) - true_remaining
                    )
                    scored += 1
    return {
        "forecast_evaluation_points": eligible,
        "estimable_forecasts": estimated,
        "forecast_coverage": estimated / eligible if eligible else None,
        "failure_reference_observed": failure_hours is not None,
        "scored_forecasts": scored,
        "forecast_mae_operating_hours": absolute_error / scored if scored else None,
    }


def run_experiment(
    output: Path,
    config: SimulationConfig,
    steps: int,
    policy: MaintenancePolicy = MaintenancePolicy.NONE,
    maintenance_config: MaintenanceConfig | None = None,
    predictor_config: PredictorConfig | None = None,
) -> dict:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps debe ser un entero positivo.")
    simulation = MineTwin(config, maintenance_config, predictor_config, policy)
    manifest = {**_manifest(simulation), "steps": steps, "status": "running"}
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    try:
        with ExitStack() as stack:
            writers = {}
            for name, columns in (
                ("telemetry", TELEMETRY_FIELDS),
                ("predictions", (*PREDICTION_FIELDS, "evaluation_point")),
                ("events", EVENT_FIELDS),
            ):
                stream = stack.enter_context(
                    (output / f"{name}.csv").open("w", encoding="utf-8", newline="")
                )
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writers[name] = writer
            simulation = MineTwin(
                config,
                maintenance_config,
                predictor_config,
                policy,
                event_sink=lambda event: writers["events"].writerow(
                    _csv_row(asdict(event))
                ),
            )
            previous_hours = 0.0
            for _ in range(steps):
                twin = simulation.advance()
                writers["telemetry"].writerow(_telemetry_row(twin.observation))
                prediction = simulation.prediction
                assert prediction is not None
                hours = twin.observation.reading("operating_hours").value
                evaluation_point = (
                    twin.observation.asset_status == AssetStatus.AVAILABLE
                    and hours is not None
                    and hours > previous_hours
                )
                writers["predictions"].writerow(
                    {
                        **_csv_row(asdict(prediction)),
                        "evaluation_point": evaluation_point,
                    }
                )
                if hours is not None:
                    previous_hours = hours
        metrics = {
            "policy": policy,
            **simulation.metrics,
            **_score_forecasts(output, policy),
        }
        (output / "metrics.csv").write_text(
            _csv_text([metrics], tuple(metrics)), encoding="utf-8"
        )
        manifest["status"] = "complete"
        manifest_path.write_text(_json_text(manifest), encoding="utf-8")
        return metrics
    except (Exception, KeyboardInterrupt):
        manifest["status"] = "incomplete"
        manifest_path.write_text(_json_text(manifest), encoding="utf-8")
        raise


def run_comparison(output: Path, config: SimulationConfig, steps: int) -> list[dict]:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps debe ser un entero positivo.")
    output.mkdir(parents=True, exist_ok=False)
    results = [
        run_experiment(output / policy.value, config, steps, policy)
        for policy in MaintenancePolicy
    ]
    (output / "comparison.csv").write_text(
        _csv_text(results, tuple(results[0])), encoding="utf-8"
    )
    return results
