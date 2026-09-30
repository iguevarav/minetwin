import csv
import hashlib
import html
import io
import json
import statistics
from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from minetwin.adapters import adapt_telemetry, encode_payload
from minetwin.domain import (
    AssetStatus,
    EnginePrediction,
    ForecastStatus,
    MaintenancePolicy,
    Scenario,
    SimulationEvent,
    TruckProfile,
    WireFormat,
)
from minetwin.federation import PROFILES, FederatedFleet
from minetwin.maintenance import MaintenanceConfig
from minetwin.prediction import PredictorConfig
from minetwin.services import MineTwin
from minetwin.simulation import SimulationConfig
from minetwin.telemetry import TelemetryError

IDENTITY_FIELDS = (
    "dataset_role",
    "scenario",
    "repetition_seed",
    "simulation_seed",
    "policy",
    "truck_id",
    "node_id",
    "profile_id",
    "wire_format",
)
EVENT_FIELDS = (*IDENTITY_FIELDS, *(field.name for field in fields(SimulationEvent)))
PREDICTION_FIELDS = (
    *IDENTITY_FIELDS,
    *(field.name for field in fields(EnginePrediction)),
    "true_hours_to_failure",
    "absolute_error_hours",
)
TRAJECTORY_FIELDS = (
    *IDENTITY_FIELDS,
    "steps",
    "scheduled_seconds",
    "available_seconds",
    "failed_seconds",
    "maintenance_seconds",
    "operating_hours",
    "availability",
    "failures",
    "interventions",
    "alerts",
    "engine_alerts",
    "false_engine_alerts",
    "false_alerts_per_operating_hour",
    "failure_reference_observed",
    "failure_reference_hours",
    "first_engine_alert_hours",
    "detection_anticipation_hours",
    "forecast_evaluation_points",
    "estimable_forecasts",
    "forecast_coverage",
    "scored_forecasts",
    "forecast_absolute_error_sum",
    "forecast_mae_operating_hours",
)


@dataclass(frozen=True)
class EvaluationConfig:
    steps: int = 450
    seeds: tuple[int, ...] = tuple(range(100, 110))
    scenarios: tuple[Scenario, ...] = (
        Scenario.NORMAL,
        Scenario.ENGINE_DEGRADATION,
        Scenario.ENGINE_VARIABLE_DEGRADATION,
    )
    policies: tuple[MaintenancePolicy, ...] = tuple(MaintenancePolicy)
    dataset_role: str = "evaluation"
    continuity_offline_steps: int = 5

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps debe ser un entero positivo.")
        if not self.seeds or any(
            type(seed) is not int or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds debe contener enteros no negativos.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Las semillas no pueden repetirse.")
        if not self.scenarios or any(
            not isinstance(item, Scenario) for item in self.scenarios
        ):
            raise TypeError("Se requiere al menos un escenario válido.")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("Los escenarios no pueden repetirse.")
        if not self.policies or any(
            not isinstance(item, MaintenancePolicy) for item in self.policies
        ):
            raise TypeError("Se requiere al menos una política válida.")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError("Las políticas no pueden repetirse.")
        if MaintenancePolicy.NONE not in self.policies:
            raise ValueError("La política none es necesaria como referencia pareada.")
        if self.dataset_role not in ("calibration", "evaluation"):
            raise ValueError("dataset_role debe ser calibration o evaluation.")
        if (
            type(self.continuity_offline_steps) is not int
            or self.continuity_offline_steps < 1
        ):
            raise ValueError("continuity_offline_steps debe ser positivo.")


@dataclass(frozen=True)
class TruckSpec:
    truck_id: str
    node_id: str
    profile: TruckProfile
    wire_format: WireFormat


def fleet_specs() -> tuple[TruckSpec, ...]:
    specifications = []
    for node_index, (node_id, wire_format, profile) in enumerate(
        zip(("alpha", "beta", "gamma"), WireFormat, PROFILES, strict=True)
    ):
        for unit in range(3):
            index = node_index * 3 + unit
            specifications.append(
                TruckSpec(
                    f"TRUCK-{index + 1:03d}",
                    node_id,
                    profile,
                    wire_format,
                )
            )
    return tuple(specifications)


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        default=lambda item: item.isoformat(),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def _csv_value(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _csv_row(values: dict) -> dict:
    return {key: _csv_value(value) for key, value in values.items()}


def _csv_text(rows: list[dict], fieldnames: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(_csv_row(row) for row in rows)
    return buffer.getvalue()


def _display(value, digits: int = 4) -> str:
    if value is None:
        return "No estimable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _html_report(summary: list[dict], paired: list[dict]) -> str:
    summary_rows = "".join(
        "<tr>"
        f"<td>{_display(row['scenario'])}</td>"
        f"<td>{_display(row['profile_id'])}</td>"
        f"<td>{_display(row['policy'])}</td>"
        f"<td>{row['trajectories']}</td>"
        f"<td>{row['availability_mean']:.2%}</td>"
        f"<td>{row['availability_population_sd']:.4f}</td>"
        f"<td>{row['failures_total']}</td>"
        f"<td>{row['interventions_total']}</td>"
        f"<td>{_display(row['forecast_coverage'])}</td>"
        f"<td>{_display(row['forecast_mae_operating_hours'])}</td>"
        "</tr>"
        for row in summary
    )
    paired_rows = "".join(
        "<tr>"
        f"<td>{_display(row['scenario'])}</td>"
        f"<td>{_display(row['profile_id'])}</td>"
        f"<td>{_display(row['policy'])}</td>"
        f"<td>{row['pairs']}</td>"
        f"<td>{row['delta_availability_mean']:+.4f}</td>"
        f"<td>{row['delta_failures_mean']:+.4f}</td>"
        f"<td>{row['delta_interventions_mean']:+.4f}</td>"
        f"<td>{row['delta_downtime_seconds_mean']:+.1f}</td>"
        "</tr>"
        for row in paired
    )
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MineTwin · Evaluación</title>
<style>
body{{background:#15181c;color:#edf0f4;font:14px system-ui;margin:0;padding:32px}}
main{{max-width:1400px;margin:auto}}h1,h2{{font-weight:600}}p{{color:#a7afbb}}
section{{background:#1c1f24;border:1px solid #30353c;border-radius:12px;margin:20px 0;padding:20px;overflow:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #30353c;padding:10px;text-align:right;white-space:nowrap}}
th{{color:#f2b544}}th:first-child,td:first-child{{text-align:left}}
</style>
</head>
<body><main>
<h1>MineTwin · Evaluación reproducible</h1>
<p>Disponibilidad como proporción; MAE y anticipación en horas de operación; dispersión poblacional. Las comparaciones pareadas usan la misma trayectoria sin mantenimiento como referencia.</p>
<section><h2>Resultados por perfil y política</h2><table>
<thead><tr><th>Escenario</th><th>Perfil</th><th>Política</th><th>n</th><th>Disponibilidad</th><th>DE</th><th>Fallas</th><th>Intervenciones</th><th>Cobertura</th><th>MAE (h)</th></tr></thead>
<tbody>{summary_rows}</tbody></table></section>
<section><h2>Diferencias pareadas frente a none</h2><table>
<thead><tr><th>Escenario</th><th>Perfil</th><th>Política</th><th>Pares</th><th>Δ disponibilidad</th><th>Δ fallas</th><th>Δ intervenciones</th><th>Δ parada (s)</th></tr></thead>
<tbody>{paired_rows}</tbody></table></section>
</main></body></html>"""


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    dependencies = {}
    for name in ("numpy", "pydantic"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "not-installed"
    return dependencies


def _derived_seed(repetition_seed: int, truck_id: str) -> int:
    digest = hashlib.sha256(f"{repetition_seed}:{truck_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _identity(
    config: EvaluationConfig,
    scenario: Scenario,
    repetition_seed: int,
    simulation_seed: int,
    policy: MaintenancePolicy,
    spec: TruckSpec,
) -> dict:
    return {
        "dataset_role": config.dataset_role,
        "scenario": scenario,
        "repetition_seed": repetition_seed,
        "simulation_seed": simulation_seed,
        "policy": policy,
        "truck_id": spec.truck_id,
        "node_id": spec.node_id,
        "profile_id": spec.profile.id,
        "wire_format": spec.wire_format,
    }


def _failure_reference(
    simulation_config: SimulationConfig,
    spec: TruckSpec,
    steps: int,
    predictor_config: PredictorConfig,
    maintenance_config: MaintenanceConfig,
) -> float | None:
    simulation = MineTwin(
        simulation_config,
        maintenance_config,
        predictor_config,
        MaintenancePolicy.NONE,
        profile=spec.profile,
        truck_id=spec.truck_id,
        node_id=spec.node_id,
        wire_format=spec.wire_format,
    )
    for _ in range(steps):
        simulation.advance()
        failure = next(
            (event for event in simulation.events if event.kind == "failure"), None
        )
        if failure:
            return failure.operating_hours
    return None


def _run_trajectory(
    evaluation_config: EvaluationConfig,
    simulation_config: SimulationConfig,
    spec: TruckSpec,
    policy: MaintenancePolicy,
    repetition_seed: int,
    failure_reference: float | None,
    event_writer: csv.DictWriter,
    prediction_writer: csv.DictWriter,
    maintenance_config: MaintenanceConfig,
    predictor_config: PredictorConfig,
) -> dict:
    identity = _identity(
        evaluation_config,
        simulation_config.scenario,
        repetition_seed,
        simulation_config.seed,
        policy,
        spec,
    )
    engine_alerts = 0
    first_alert_hours = None
    intervention_started = False

    def record_event(event: SimulationEvent) -> None:
        nonlocal engine_alerts, first_alert_hours, intervention_started
        event_writer.writerow(_csv_row({**identity, **asdict(event)}))
        if event.kind == "alert_opened" and event.component == "engine":
            engine_alerts += 1
            if first_alert_hours is None:
                first_alert_hours = event.operating_hours
        if event.kind == "maintenance_started" and event.component == "engine":
            intervention_started = True

    simulation = MineTwin(
        simulation_config,
        maintenance_config,
        predictor_config,
        policy,
        record_event,
        spec.profile,
        spec.truck_id,
        spec.node_id,
        spec.wire_format,
    )
    evaluation_points = estimable = scored = 0
    absolute_error_sum = 0.0
    previous_hours = 0.0
    for _ in range(evaluation_config.steps):
        twin = simulation.advance()
        prediction = simulation.prediction
        hours = twin.observation.reading("operating_hours").value
        is_evaluation_point = (
            twin.observation.asset_status == AssetStatus.AVAILABLE
            and hours is not None
            and hours > previous_hours
            and not intervention_started
        )
        if is_evaluation_point and prediction:
            evaluation_points += 1
            true_remaining = (
                failure_reference - hours
                if failure_reference is not None and hours <= failure_reference
                else None
            )
            absolute_error = None
            if prediction.status == ForecastStatus.ESTIMABLE:
                estimable += 1
                if (
                    true_remaining is not None
                    and prediction.hours_to_threshold is not None
                ):
                    absolute_error = abs(prediction.hours_to_threshold - true_remaining)
                    absolute_error_sum += absolute_error
                    scored += 1
            prediction_writer.writerow(
                _csv_row(
                    {
                        **identity,
                        **asdict(prediction),
                        "true_hours_to_failure": true_remaining,
                        "absolute_error_hours": absolute_error,
                    }
                )
            )
        if hours is not None:
            previous_hours = hours
    metrics = simulation.metrics
    operating_hours = metrics["operating_hours"]
    false_alerts = engine_alerts if simulation_config.scenario == Scenario.NORMAL else 0
    anticipation = (
        failure_reference - first_alert_hours
        if failure_reference is not None
        and first_alert_hours is not None
        and first_alert_hours <= failure_reference
        else None
    )
    return {
        **identity,
        "steps": evaluation_config.steps,
        **metrics,
        "engine_alerts": engine_alerts,
        "false_engine_alerts": false_alerts,
        "false_alerts_per_operating_hour": (
            false_alerts / operating_hours if operating_hours else None
        ),
        "failure_reference_observed": failure_reference is not None,
        "failure_reference_hours": failure_reference,
        "first_engine_alert_hours": first_alert_hours,
        "detection_anticipation_hours": anticipation,
        "forecast_evaluation_points": evaluation_points,
        "estimable_forecasts": estimable,
        "forecast_coverage": (
            estimable / evaluation_points if evaluation_points else None
        ),
        "scored_forecasts": scored,
        "forecast_absolute_error_sum": absolute_error_sum,
        "forecast_mae_operating_hours": (
            absolute_error_sum / scored if scored else None
        ),
    }


def _paired_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (
            row["dataset_role"],
            row["scenario"],
            row["repetition_seed"],
            row["truck_id"],
        )
        grouped.setdefault(key, {})[row["policy"]] = row
    results = []
    for key, policies in grouped.items():
        reference = policies.get(MaintenancePolicy.NONE)
        if reference is None:
            continue
        for policy, candidate in policies.items():
            if policy == MaintenancePolicy.NONE:
                continue
            results.append(
                {
                    "dataset_role": key[0],
                    "scenario": key[1],
                    "repetition_seed": key[2],
                    "truck_id": key[3],
                    "profile_id": candidate["profile_id"],
                    "policy": policy,
                    "delta_availability": (
                        candidate["availability"] - reference["availability"]
                    ),
                    "delta_failures": candidate["failures"] - reference["failures"],
                    "delta_interventions": (
                        candidate["interventions"] - reference["interventions"]
                    ),
                    "delta_downtime_seconds": (
                        candidate["failed_seconds"]
                        + candidate["maintenance_seconds"]
                        - reference["failed_seconds"]
                        - reference["maintenance_seconds"]
                    ),
                }
            )
    return results


def _summary_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (
            row["dataset_role"],
            row["scenario"],
            row["profile_id"],
            row["policy"],
        )
        groups.setdefault(key, []).append(row)
    results = []
    for key, members in groups.items():
        availability = [row["availability"] for row in members]
        operating_hours = sum(row["operating_hours"] for row in members)
        false_alerts = sum(row["false_engine_alerts"] for row in members)
        evaluation_points = sum(row["forecast_evaluation_points"] for row in members)
        estimable = sum(row["estimable_forecasts"] for row in members)
        scored = sum(row["scored_forecasts"] for row in members)
        absolute_error = sum(row["forecast_absolute_error_sum"] for row in members)
        detectable = [row for row in members if row["failure_reference_observed"]]
        detected = [
            row for row in detectable if row["first_engine_alert_hours"] is not None
        ]
        anticipations = [
            row["detection_anticipation_hours"]
            for row in detected
            if row["detection_anticipation_hours"] is not None
        ]
        results.append(
            {
                "dataset_role": key[0],
                "scenario": key[1],
                "profile_id": key[2],
                "policy": key[3],
                "trajectories": len(members),
                "availability_mean": statistics.fmean(availability),
                "availability_population_sd": (
                    statistics.pstdev(availability) if len(availability) > 1 else 0
                ),
                "failures_total": sum(row["failures"] for row in members),
                "interventions_total": sum(row["interventions"] for row in members),
                "false_engine_alerts_total": false_alerts,
                "false_alerts_per_operating_hour": (
                    false_alerts / operating_hours if operating_hours else None
                ),
                "detection_rate": (
                    len(detected) / len(detectable) if detectable else None
                ),
                "anticipation_hours_mean": (
                    statistics.fmean(anticipations) if anticipations else None
                ),
                "forecast_coverage": (
                    estimable / evaluation_points if evaluation_points else None
                ),
                "forecast_mae_operating_hours": (
                    absolute_error / scored if scored else None
                ),
                "scored_forecasts": scored,
            }
        )
    return results


def _paired_summary_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (
            row["dataset_role"],
            row["scenario"],
            row["profile_id"],
            row["policy"],
        )
        groups.setdefault(key, []).append(row)
    results = []
    metrics = (
        "delta_availability",
        "delta_failures",
        "delta_interventions",
        "delta_downtime_seconds",
    )
    for key, members in groups.items():
        result = {
            "dataset_role": key[0],
            "scenario": key[1],
            "profile_id": key[2],
            "policy": key[3],
            "pairs": len(members),
        }
        for metric in metrics:
            values = [row[metric] for row in members]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_population_sd"] = (
                statistics.pstdev(values) if len(values) > 1 else 0
            )
        results.append(result)
    return results


def _interoperability_rows() -> list[dict]:
    results = []
    for profile_index, profile in enumerate(PROFILES):
        source = MineTwin(
            SimulationConfig(seed=9000 + profile_index),
            profile=profile,
        )
        source.advance(8)
        payload = json.loads(source.raw_history[-1])
        reference = adapt_telemetry(
            encode_payload(payload, WireFormat.JSON, profile),
            source.time,
            WireFormat.JSON,
            profile,
        )
        for wire_format in WireFormat:
            packet = adapt_telemetry(
                encode_payload(payload, wire_format, profile),
                source.time,
                wire_format,
                profile,
            )
            differences = [
                abs(reading.value - expected.value)
                for reading, expected in zip(
                    packet.readings, reference.readings, strict=True
                )
                if reading.value is not None and expected.value is not None
            ]
            max_difference = max(differences, default=0)
            defective = {
                WireFormat.JSON: "{}",
                WireFormat.XML: "<telemetry/>",
                WireFormat.CSV: "unknown\nvalue",
            }[wire_format]
            rejected = False
            try:
                adapt_telemetry(defective, source.time, wire_format, profile)
            except TelemetryError:
                rejected = True
            results.append(
                {
                    "profile_id": profile.id,
                    "wire_format": wire_format,
                    "canonical_equivalent": (
                        packet.event_id == reference.event_id
                        and packet.operating_state == reference.operating_state
                        and max_difference <= 1e-8
                        and all(
                            (reading.name, reading.unit, reading.quality)
                            == (expected.name, expected.unit, expected.quality)
                            for reading, expected in zip(
                                packet.readings, reference.readings, strict=True
                            )
                        )
                    ),
                    "max_absolute_numeric_difference": max_difference,
                    "malformed_packet_rejected": rejected,
                }
            )
    return results


def _continuity_rows(config: EvaluationConfig) -> list[dict]:
    rows = []
    for repetition_seed in config.seeds:
        fleet = FederatedFleet(SimulationConfig(seed=repetition_seed))
        fleet.advance(5)
        selected = "TRUCK-001"
        other = "TRUCK-004"
        central_before = fleet.view(selected).twin.version
        other_before = fleet.view(other).twin.version
        fleet.set_connected("alpha", False)
        fleet.advance(config.continuity_offline_steps)
        central_offline = fleet.view(selected)
        local_offline = fleet.nodes["alpha"].trucks[selected].twin.version
        other_offline = fleet.view(other).twin.version
        fleet.set_connected("alpha", True)
        central_after = fleet.view(selected)
        rows.append(
            {
                "dataset_role": config.dataset_role,
                "repetition_seed": repetition_seed,
                "offline_steps": config.continuity_offline_steps,
                "central_view_frozen": central_offline.twin.version == central_before,
                "local_node_continued": local_offline
                == central_before + config.continuity_offline_steps,
                "other_nodes_continued": other_offline
                == other_before + config.continuity_offline_steps,
                "offline_view_stale": central_offline.stale,
                "offline_age_seconds": central_offline.age_seconds,
                "reconnected_to_current_state": central_after.twin.version
                == local_offline,
                "reconnected_view_fresh": not central_after.stale,
            }
        )
    return rows


def run_fleet_evaluation(
    output: Path,
    config: EvaluationConfig | None = None,
    maintenance_config: MaintenanceConfig | None = None,
    predictor_config: PredictorConfig | None = None,
) -> list[dict]:
    effective = config or EvaluationConfig()
    maintenance = maintenance_config or MaintenanceConfig()
    predictor = predictor_config or PredictorConfig()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = {
        "status": "running",
        "evaluation": asdict(effective),
        "maintenance": asdict(maintenance),
        "predictor": asdict(predictor),
        "profiles": [asdict(profile) for profile in PROFILES],
        "code_sha256": _code_fingerprint(),
        "dependencies": _dependency_versions(),
        "design": {
            "pairing": "Same scenario, repetition seed, truck and external noise path across policies.",
            "seed_derivation": "SHA-256 of repetition seed and truck id, truncated to 32 bits.",
            "forecast_reference": "Observed first failure without intervention for the paired trajectory.",
            "forecast_scoring": "Operational observations before first engine intervention.",
            "censoring": "No forecast error when the reference trajectory does not fail.",
            "language_model": "Excluded from maintenance policies and quantitative metrics.",
        },
    }
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    trajectory_rows = []
    try:
        with (
            (output / "events.csv").open("w", encoding="utf-8", newline="") as events,
            (output / "predictions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as predictions,
        ):
            event_writer = csv.DictWriter(events, fieldnames=EVENT_FIELDS)
            prediction_writer = csv.DictWriter(
                predictions, fieldnames=PREDICTION_FIELDS
            )
            event_writer.writeheader()
            prediction_writer.writeheader()
            for scenario in effective.scenarios:
                for repetition_seed in effective.seeds:
                    for spec in fleet_specs():
                        simulation_seed = _derived_seed(repetition_seed, spec.truck_id)
                        simulation_config = SimulationConfig(
                            seed=simulation_seed,
                            scenario=scenario,
                        )
                        reference = _failure_reference(
                            simulation_config,
                            spec,
                            effective.steps,
                            predictor,
                            maintenance,
                        )
                        for policy in effective.policies:
                            trajectory_rows.append(
                                _run_trajectory(
                                    effective,
                                    simulation_config,
                                    spec,
                                    policy,
                                    repetition_seed,
                                    reference,
                                    event_writer,
                                    prediction_writer,
                                    maintenance,
                                    predictor,
                                )
                            )
        paired = _paired_rows(trajectory_rows)
        paired_summary = _paired_summary_rows(paired)
        summary = _summary_rows(trajectory_rows)
        interoperability = _interoperability_rows()
        continuity = _continuity_rows(effective)
        (output / "trajectory_metrics.csv").write_text(
            _csv_text(trajectory_rows, TRAJECTORY_FIELDS), encoding="utf-8"
        )
        if paired:
            (output / "paired_differences.csv").write_text(
                _csv_text(paired, tuple(paired[0])), encoding="utf-8"
            )
            (output / "paired_summary.csv").write_text(
                _csv_text(paired_summary, tuple(paired_summary[0])),
                encoding="utf-8",
            )
        (output / "summary.csv").write_text(
            _csv_text(summary, tuple(summary[0])), encoding="utf-8"
        )
        (output / "interoperability.csv").write_text(
            _csv_text(interoperability, tuple(interoperability[0])),
            encoding="utf-8",
        )
        (output / "continuity.csv").write_text(
            _csv_text(continuity, tuple(continuity[0])), encoding="utf-8"
        )
        (output / "report.html").write_text(
            _html_report(summary, paired_summary), encoding="utf-8"
        )
        manifest["status"] = "complete"
        manifest["counts"] = {
            "trajectories": len(trajectory_rows),
            "paired_differences": len(paired),
            "paired_summary_groups": len(paired_summary),
            "summary_groups": len(summary),
            "interoperability_cases": len(interoperability),
            "continuity_cases": len(continuity),
        }
        manifest_path.write_text(_json_text(manifest), encoding="utf-8")
        return summary
    except (Exception, KeyboardInterrupt):
        manifest["status"] = "incomplete"
        manifest_path.write_text(_json_text(manifest), encoding="utf-8")
        raise
