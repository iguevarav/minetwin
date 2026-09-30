import csv
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from minetwin.domain import (
    Diagnosis,
    EnginePrediction,
    EngineStatus,
    ForecastStatus,
    Quality,
    Scenario,
)
from minetwin.federation import FederatedFleet, TruckView
from minetwin.langflow_client import (
    FLOW_INSTRUCTIONS,
    LangflowClient,
    LangflowConfig,
    LangflowError,
    explanation_context,
)
from minetwin.simulation import SimulationConfig

REVIEW_FIELDS = (
    "case_id",
    "response_received",
    "latency_seconds",
    "word_count",
    "within_word_limit",
    "state_unchanged",
    "factual_accuracy_0_to_2",
    "component_identification_0_to_2",
    "uncertainty_handling_0_to_2",
    "no_unsupported_action_claim_0_to_2",
    "reviewer",
    "review_notes",
)
SCORE_FIELDS = (
    "factual_accuracy_0_to_2",
    "component_identification_0_to_2",
    "uncertainty_handling_0_to_2",
    "no_unsupported_action_claim_0_to_2",
)


def _json_value(value: object) -> str:
    return json.dumps(
        value,
        default=lambda item: item.isoformat(),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json_value(value).encode()).hexdigest()


def _scenario_view(scenario: Scenario, steps: int) -> TruckView:
    fleet = FederatedFleet(SimulationConfig(seed=42, scenario=scenario))
    fleet.advance(steps)
    return fleet.view("TRUCK-001")


def _missing_engine_view() -> TruckView:
    view = _scenario_view(Scenario.NORMAL, 30)
    observation = view.twin.observation
    readings = tuple(
        replace(
            reading,
            value=None,
            quality=Quality.MISSING,
            reason="Lectura ausente para el caso de evaluación.",
        )
        if reading.name == "engine_temperature"
        else reading
        for reading in observation.readings
    )
    diagnosis = Diagnosis(
        EngineStatus.UNKNOWN,
        "No estimable: falta una lectura válida de engine_temperature.",
    )
    prediction = EnginePrediction(
        observation.source_time,
        ForecastStatus.NOT_ESTIMABLE,
        "Faltan señales esenciales válidas.",
    )
    twin = replace(
        view.twin,
        observation=replace(observation, readings=readings),
        diagnosis=diagnosis,
    )
    return replace(view, twin=twin, prediction=prediction)


def _maintenance_view() -> TruckView:
    fleet = FederatedFleet(SimulationConfig(seed=42))
    fleet.advance(30)
    truck = fleet.local("TRUCK-001")
    order = truck.create_order(reason="Caso de evaluación de mantenimiento.")
    truck.start_order(order.id)
    fleet.advance()
    return fleet.view("TRUCK-001")


def _disconnected_view() -> TruckView:
    fleet = FederatedFleet(SimulationConfig(seed=42))
    fleet.advance(30)
    fleet.set_connected("alpha", False)
    return fleet.view("TRUCK-001")


def langflow_cases() -> tuple[tuple[str, TruckView], ...]:
    return (
        ("normal", _scenario_view(Scenario.NORMAL, 30)),
        (
            "engine_degradation",
            _scenario_view(Scenario.ENGINE_DEGRADATION, 180),
        ),
        (
            "engine_variable_degradation",
            _scenario_view(Scenario.ENGINE_VARIABLE_DEGRADATION, 180),
        ),
        ("brake_stress", _scenario_view(Scenario.BRAKE_STRESS, 100)),
        ("tire_leak", _scenario_view(Scenario.TIRE_LEAK, 100)),
        ("missing_engine_signal", _missing_engine_view()),
        ("maintenance", _maintenance_view()),
        ("disconnected", _disconnected_view()),
    )


def _expected(view: TruckView) -> dict:
    return {
        "asset_status": view.asset_status,
        "engine_status": view.twin.diagnosis.status if view.twin else None,
        "forecast_status": view.prediction.status if view.prediction else None,
        "open_alert_components": sorted(
            alert.component for alert in view.alerts if alert.closed_at is None
        ),
        "unknown_components": sorted(
            diagnosis.component
            for diagnosis in view.twin.component_diagnoses
            if diagnosis.status == EngineStatus.UNKNOWN
        )
        if view.twin
        else [],
        "stale": view.stale,
        "connected": view.connected,
    }


def run_langflow_evaluation(
    output: Path,
    client: LangflowClient | None = None,
) -> list[dict]:
    active_client = client
    if active_client is None:
        configuration = LangflowConfig.from_environment()
        if configuration is None:
            raise ValueError(
                "Faltan MINETWIN_LANGFLOW_URL, MINETWIN_LANGFLOW_FLOW_ID y "
                "MINETWIN_LANGFLOW_API_KEY."
            )
        active_client = LangflowClient(configuration)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = {
        "status": "running",
        "flow_id": active_client.config.flow_id,
        "flow_version": os.environ.get("MINETWIN_LANGFLOW_FLOW_VERSION", ""),
        "prompt_version": os.environ.get(
            "MINETWIN_LANGFLOW_PROMPT_VERSION", "minetwin-explanation-2"
        ),
        "model_id": os.environ.get("MINETWIN_LANGFLOW_MODEL_ID", ""),
        "prompt_sha256": hashlib.sha256(FLOW_INSTRUCTIONS.encode()).hexdigest(),
        "review_scale": {
            "0": "Incorrecto o ausente",
            "1": "Parcialmente correcto",
            "2": "Correcto según la evidencia",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records = []
    review_rows = []
    try:
        for case_id, view in langflow_cases():
            expected = _expected(view)
            before = _fingerprint(asdict(view))
            context = None
            response = None
            error = None
            blocked_locally = False
            started = perf_counter()
            try:
                context = explanation_context(view)
                response = active_client.explain(view)
            except LangflowError as failure:
                error = str(failure)
                blocked_locally = view.stale or not view.connected or view.twin is None
            latency = perf_counter() - started
            after = _fingerprint(asdict(view))
            words = len(response.split()) if response else 0
            record = {
                "case_id": case_id,
                "expected": expected,
                "context": context,
                "context_sha256": _fingerprint(context) if context else None,
                "response": response,
                "response_sha256": _fingerprint(response) if response else None,
                "error": error,
                "blocked_locally": blocked_locally,
                "latency_seconds": latency,
                "word_count": words,
                "within_word_limit": words <= 250 if response else None,
                "state_unchanged": before == after,
            }
            records.append(record)
            review_rows.append(
                {
                    "case_id": case_id,
                    "response_received": response is not None,
                    "latency_seconds": latency,
                    "word_count": words,
                    "within_word_limit": words <= 250 if response else "",
                    "state_unchanged": before == after,
                    "factual_accuracy_0_to_2": "",
                    "component_identification_0_to_2": "",
                    "uncertainty_handling_0_to_2": "",
                    "no_unsupported_action_claim_0_to_2": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
        with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(_json_value(record) + "\n")
        with (output / "review.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(review_rows)
        manifest["status"] = "complete"
        manifest["counts"] = {
            "cases": len(records),
            "responses": sum(record["response"] is not None for record in records),
            "errors": sum(record["error"] is not None for record in records),
            "locally_blocked": sum(record["blocked_locally"] for record in records),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return records
    except (Exception, KeyboardInterrupt):
        manifest["status"] = "incomplete"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


def summarize_langflow_review(output: Path) -> dict:
    review_path = output / "review.csv"
    manifest_path = output / "manifest.json"
    if not review_path.is_file() or not manifest_path.is_file():
        raise ValueError("El directorio no contiene una evaluación Langflow completa.")
    with review_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(REVIEW_FIELDS).issubset(
            reader.fieldnames
        ):
            raise ValueError("review.csv no contiene las columnas requeridas.")
        rows = list(reader)
    reviewed = [row for row in rows if row["response_received"] == "True"]
    if not reviewed:
        raise ValueError("No hay respuestas de Langflow para revisar.")
    scores = {field: [] for field in SCORE_FIELDS}
    for row in reviewed:
        if not row["reviewer"].strip():
            raise ValueError(f"Falta el revisor del caso {row['case_id']}.")
        for field in SCORE_FIELDS:
            try:
                score = int(row[field])
            except ValueError as error:
                raise ValueError(
                    f"El caso {row['case_id']} necesita una puntuación entera en {field}."
                ) from error
            if score not in (0, 1, 2):
                raise ValueError(f"La puntuación de {field} debe estar entre 0 y 2.")
            scores[field].append(score)
    summary = {
        "reviewed_responses": len(reviewed),
        "total_cases": len(rows),
        "reviewers": sorted({row["reviewer"].strip() for row in reviewed}),
        "mean_scores": {
            field.removesuffix("_0_to_2"): sum(values) / len(values)
            for field, values in scores.items()
        },
    }
    summary["overall_mean_score"] = sum(summary["mean_scores"].values()) / len(
        summary["mean_scores"]
    )
    (output / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["human_review"] = summary
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
