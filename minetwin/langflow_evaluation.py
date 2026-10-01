import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from urllib.error import URLError

import numpy as np

from minetwin.data.scania import DatasetSplit, ScaniaReplaySession, ScaniaReplayStore
from minetwin.langflow_client import (
    FLOW_INSTRUCTIONS,
    MAX_RESPONSE_WORDS,
    PROMPT_VERSION,
    LangflowClient,
    LangflowConfig,
    LangflowError,
    ScaniaExplanationObservation,
    explanation_context,
)
from minetwin.learning.provenance import file_sha256
from minetwin.paths import DATASET_ROOT, RESULTS_ROOT
from minetwin.scania_explanation import ScaniaExplanationSource

REVIEW_FIELDS = (
    "case_id",
    "response_received",
    "latency_seconds",
    "word_count",
    "within_word_limit",
    "state_unchanged",
    "factual_accuracy_0_to_2",
    "class_and_scope_0_to_2",
    "uncertainty_handling_0_to_2",
    "no_unsupported_action_claim_0_to_2",
    "reviewer",
    "review_notes",
)
SCORE_FIELDS = REVIEW_FIELDS[6:10]


@dataclass(frozen=True)
class LangflowCase:
    case_id: str
    observation: ScaniaExplanationObservation
    observed_class: int | None
    kind: str = "dataset"
    transport_fault: str | None = None


class _FailingOpener:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def open(self, request, timeout):
        raise self.error


def _json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json_value(value).encode("utf-8")).hexdigest()


def _latest_observation(
    source: ScaniaExplanationSource,
    store: ScaniaReplayStore,
    vehicle_id: str,
) -> ScaniaExplanationObservation:
    session = ScaniaReplaySession(
        store,
        DatasetSplit(source.split),
        vehicle_id=vehicle_id,
        window_size=source.vectorizer.config.window_size,
    )
    session.advance(session.readout_count)
    return source.observation(session.view, source.regime)


def langflow_cases(
    source: ScaniaExplanationSource, store: ScaniaReplayStore
) -> tuple[LangflowCase, ...]:
    cases = []
    for label in range(5):
        for index in np.flatnonzero(source.cached.labels == label):
            vehicle_id = str(source.cached.vehicle_ids[index])
            if vehicle_id not in source.states:
                continue
            observation = _latest_observation(source, store, vehicle_id)
            if not observation.stale:
                cases.append(LangflowCase(f"class_{label}", observation, label))
                break
    if not cases:
        raise ValueError("No hay estados SCANIA vigentes para evaluar.")
    missing = next(
        (case for case in cases if case.observation.unavailable_count), None
    )
    if missing is None:
        for vehicle_id in source.states:
            observation = _latest_observation(source, store, vehicle_id)
            if observation.unavailable_count and not observation.stale:
                missing = LangflowCase(
                    "missing_readings",
                    observation,
                    source.observed_class(vehicle_id),
                )
                break
    else:
        missing = replace(missing, case_id="missing_readings")
    if missing is None:
        raise ValueError("No se encontró un readout SCANIA con valores ausentes.")
    cases.append(missing)
    reference = cases[0]
    for case_id, change in (
        ("missing_state", {"state": None}),
        ("stale_observation", {"stale": True}),
        ("disconnected_node", {"connected": False}),
    ):
        cases.append(
            LangflowCase(
                case_id,
                replace(reference.observation, **change),
                reference.observed_class,
                kind="local_fault_injection",
            )
        )
    for fault in ("connection_error", "timeout"):
        cases.append(
            LangflowCase(
                fault,
                reference.observation,
                reference.observed_class,
                kind="transport_fault_injection",
                transport_fault=fault,
            )
        )
    return tuple(cases)


def run_langflow_evaluation(
    output: Path,
    client: LangflowClient | None = None,
    cases: tuple[LangflowCase, ...] | None = None,
    flow_definition: Path | None = None,
) -> list[dict]:
    if cases is None and flow_definition is None:
        raise ValueError(
            "Exporta el flujo Langflow y pásalo con --flow-definition."
        )
    active_client = client
    if active_client is None:
        configuration = LangflowConfig.from_environment()
        if configuration is None:
            raise ValueError(
                "Faltan MINETWIN_LANGFLOW_URL, MINETWIN_LANGFLOW_FLOW_ID y "
                "MINETWIN_LANGFLOW_API_KEY."
            )
        active_client = LangflowClient(configuration)
    source = None
    if cases is None:
        source = ScaniaExplanationSource()
        if source.split != "validation":
            raise ValueError("La evaluación requiere el split validation etiquetado.")
        store = ScaniaReplayStore(
            DATASET_ROOT, RESULTS_ROOT / "phase1_replay" / "index.json"
        )
        cases = langflow_cases(source, store)
    definition = None
    if flow_definition is not None:
        definition = Path(flow_definition).read_bytes()
        json.loads(definition)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    if definition is not None:
        (output / "flow_definition.json").write_bytes(definition)
    (output / "prompt.txt").write_text(FLOW_INSTRUCTIONS, encoding="utf-8")
    flow_hash = hashlib.sha256(definition).hexdigest() if definition else None
    manifest = {
        "status": "running",
        "source": "SCANIA Component X validation",
        "flow_id": active_client.config.flow_id,
        "flow_version": os.environ.get("MINETWIN_LANGFLOW_FLOW_VERSION")
        or (flow_hash[:12] if flow_hash else None),
        "flow_definition_sha256": flow_hash,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(FLOW_INSTRUCTIONS.encode()).hexdigest(),
        "language_model_id": os.environ.get("MINETWIN_LANGFLOW_MODEL_ID", ""),
        "predictive_models": sorted(
            {
                case.observation.state.risk.model_id
                for case in cases
                if case.observation.state is not None
            }
        ),
        "observed_classes_evaluated": sorted(
            {
                case.observed_class
                for case in cases
                if case.kind == "dataset" and case.observed_class is not None
            }
        ),
        "training_id": source.provenance["training_id"] if source else None,
        "data_version": source.provenance["data_version"] if source else None,
        "word_limit": MAX_RESPONSE_WORDS,
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
        for case in cases:
            observation = case.observation
            before = _fingerprint(asdict(observation))
            context = None
            response = None
            error = None
            started = perf_counter()
            try:
                context = explanation_context(observation)
                if case.transport_fault:
                    failure = (
                        URLError("simulated connection failure")
                        if case.transport_fault == "connection_error"
                        else TimeoutError("simulated timeout")
                    )
                    response = LangflowClient(
                        active_client.config, _FailingOpener(failure)
                    ).explain(observation)
                else:
                    response = active_client.explain(observation)
            except LangflowError as failure:
                error = str(failure)
            latency = perf_counter() - started
            words = len(response.split()) if response else 0
            blocked_locally = context is None and error is not None
            record = {
                "case_id": case.case_id,
                "kind": case.kind,
                "expected": {
                    "observed_class": case.observed_class,
                    "published_class": (
                        observation.state.risk.predicted_class
                        if observation.state is not None
                        else None
                    ),
                    "node_id": observation.node_id,
                    "split": observation.split,
                    "time_step": observation.time_step,
                    "unavailable_reading_count": observation.unavailable_count,
                },
                "context": context,
                "context_sha256": _fingerprint(context) if context else None,
                "response": response,
                "response_sha256": _fingerprint(response) if response else None,
                "error": error,
                "blocked_locally": blocked_locally,
                "latency_seconds": latency,
                "word_count": words,
                "within_word_limit": (
                    words <= MAX_RESPONSE_WORDS if response else None
                ),
                "state_unchanged": before == _fingerprint(asdict(observation)),
            }
            records.append(record)
            review_rows.append(
                {
                    "case_id": case.case_id,
                    "response_received": response is not None,
                    "latency_seconds": latency,
                    "word_count": words,
                    "within_word_limit": (
                        words <= MAX_RESPONSE_WORDS if response else ""
                    ),
                    "state_unchanged": record["state_unchanged"],
                    **{field: "" for field in SCORE_FIELDS},
                    "reviewer": "",
                    "review_notes": "",
                }
            )
        with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(_json_value(record) + "\n")
        with (output / "review.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
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
        manifest["cases_sha256"] = file_sha256(output / "cases.jsonl")
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
    output = Path(output)
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
                    f"El caso {row['case_id']} necesita una puntuación en {field}."
                ) from error
            if score not in (0, 1, 2):
                raise ValueError(f"La puntuación de {field} debe estar entre 0 y 2.")
            scores[field].append(score)
    summary = {
        "reviewed_responses": len(reviewed),
        "total_cases": len(rows),
        "review_sha256": file_sha256(review_path),
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
