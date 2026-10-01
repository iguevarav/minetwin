import json
import os
import re
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from minetwin.data.scania import SCANIA_CLASS_DESCRIPTIONS
from minetwin.publication import PublishedTwinState

PROMPT_VERSION = "scania-component-x-1"
MAX_RESPONSE_WORDS = 180
FLOW_INSTRUCTIONS = (
    "Eres el asistente explicativo de MineTwin. Responde en español usando solo "
    "el contexto JSON de una observación histórica SCANIA Component X. La clase y "
    "las cinco probabilidades proceden del modelo publicado; no las cambies. La "
    "calidad mide cobertura de lecturas, no exactitud. Las variables anónimas se "
    "seleccionaron por desviación estandarizada, no prueban causalidad ni explican "
    "por sí solas la predicción. No atribuyas Component X a una pieza física, no "
    "afirmes que los datos proceden de una mina ni que son datos en tiempo real. "
    "No inventes lecturas, fallas, reparación, pronóstico de vida útil ni acciones "
    "ejecutadas. Sugiere solo revisión humana. Organiza la respuesta en Condición, "
    "Evidencia, Limitaciones e Inspección sugerida. Máximo 180 palabras."
)


class LangflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class LangflowConfig:
    base_url: str
    flow_id: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or (parsed.scheme == "http" and not local)
        ):
            raise ValueError("Langflow requiere HTTPS o HTTP local.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("La URL base no admite credenciales ni parámetros.")
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", self.flow_id)
            or not self.api_key.strip()
        ):
            raise ValueError("Se requieren el identificador de flujo y la clave API.")
        if not 0 < self.timeout_seconds <= 500:
            raise ValueError("El tiempo de espera debe estar entre 0 y 500 segundos.")

    @classmethod
    def from_environment(cls):
        values = [
            os.environ.get(name, "").strip()
            for name in (
                "MINETWIN_LANGFLOW_URL",
                "MINETWIN_LANGFLOW_FLOW_ID",
                "MINETWIN_LANGFLOW_API_KEY",
            )
        ]
        if not all(values):
            return None
        timeout_value = os.environ.get(
            "MINETWIN_LANGFLOW_TIMEOUT_SECONDS", ""
        ).strip()
        if not timeout_value:
            return cls(*values)
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as error:
            raise ValueError(
                "MINETWIN_LANGFLOW_TIMEOUT_SECONDS debe ser numérico."
            ) from error
        return cls(*values, timeout_seconds)


@dataclass(frozen=True)
class ScaniaExplanationObservation:
    state: PublishedTwinState | None
    vehicle_id: str
    node_id: str
    split: str
    time_step: float
    readout_index: int
    readout_count: int
    relevant_features: tuple[str, ...] = ()
    unavailable_readings: tuple[str, ...] = ()
    unavailable_count: int = 0
    connected: bool = True
    stale: bool = False


def explanation_context(observation: ScaniaExplanationObservation) -> dict:
    state = observation.state
    if state is None:
        raise LangflowError("No existe un estado publicado para esta observación.")
    if not observation.connected:
        raise LangflowError("La partición no está disponible para consulta.")
    if observation.stale or observation.readout_index != observation.readout_count:
        raise LangflowError(
            "El estado publicado no corresponde al readout seleccionado."
        )
    if (
        state.asset_id != observation.vehicle_id
        or state.node_id != observation.node_id
        or state.component != "Component X"
        or state.source != "SCANIA Component X"
        or state.split != observation.split
        or state.schema_version != 2
        or state.quality is None
        or state.training_id == "unverified"
        or state.data_version == "unverified"
    ):
        raise LangflowError("La identidad del estado publicado no coincide.")
    return {
        "prompt_version": PROMPT_VERSION,
        "source": "SCANIA Component X",
        "scope": "historical_public_dataset_experimental_partitions",
        "observation": {
            "vehicle_id": observation.vehicle_id,
            "node_id": observation.node_id,
            "split": observation.split,
            "time_step": observation.time_step,
            "readout_index": observation.readout_index,
            "readout_count": observation.readout_count,
            "unavailable_reading_count": observation.unavailable_count,
            "unavailable_anonymized_readings": observation.unavailable_readings,
            "relevant_anonymized_features": observation.relevant_features,
            "feature_selection": "largest_absolute_training_standardized_deviation",
        },
        "published_risk": {
            "predicted_class": state.risk.predicted_class,
            "class_meaning": SCANIA_CLASS_DESCRIPTIONS[state.risk.predicted_class],
            "probabilities": state.risk.probabilities,
            "condition": state.risk.condition,
            "quality": state.quality,
            "recommendation": state.risk.recommendation,
            "model_id": state.risk.model_id,
            "training_id": state.training_id,
            "data_version": state.data_version,
        },
    }


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LangflowClient:
    def __init__(self, config: LangflowConfig, opener=None) -> None:
        self.config = config
        self._opener = opener or build_opener(_NoRedirect())

    def explain(self, observation: ScaniaExplanationObservation) -> str:
        context = explanation_context(observation)
        payload = {
            "input_value": FLOW_INSTRUCTIONS
            + "\nEVIDENCIA:\n"
            + json.dumps(
                context,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "input_type": "chat",
            "output_type": "chat",
            "session_id": "minetwin-" + uuid4().hex,
        }
        request = Request(
            f"{self.config.base_url.rstrip('/')}/api/v1/run/{self.config.flow_id}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
            },
        )
        try:
            with self._opener.open(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = response.read(1_000_001)
            if len(body) > 1_000_000:
                raise LangflowError("La respuesta excede el tamaño permitido.")
            result = json.loads(body)
            for output in result.get("outputs", []):
                for item in output.get("outputs", []):
                    text = item.get("results", {}).get("message", {}).get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()[:12_000]
            raise LangflowError("El flujo no devolvió una respuesta de Chat Output.")
        except HTTPError as error:
            raise LangflowError(f"Langflow respondió con HTTP {error.code}.") from None
        except (URLError, TimeoutError, OSError):
            raise LangflowError(
                "No se pudo contactar con Langflow dentro del tiempo permitido."
            ) from None
        except (ValueError, TypeError, AttributeError):
            raise LangflowError(
                "Langflow devolvió una respuesta incompatible."
            ) from None
