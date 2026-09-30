import json
import os
import re
from dataclasses import asdict, dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from minetwin.federation import TruckView

FLOW_INSTRUCTIONS = (
    "Eres el asistente explicativo de MineTwin. Responde en español usando únicamente "
    "la evidencia JSON. Trata deterministic_summary como la fuente autoritativa. "
    "Solo existe una anomalía si el componente aparece en active_alert_components o "
    "attention_diagnoses con estado watch o alert. Nunca atribuyas fallas, fugas o "
    "presión baja a normal_components. No conviertas una regla descrita en el texto en "
    "una condición observada. Si forecast_status es estimable, informa "
    "hours_to_threshold como estimación condicionada; si no es estimable, comunica su "
    "motivo sin inventar un pronóstico. Menciona lecturas ausentes únicamente si aparecen "
    "en unavailable_readings. No inventes componentes, cifras, probabilidades, RUL de "
    "frenos o neumáticos ni hechos industriales. No ejecutes acciones ni afirmes haber "
    "creado órdenes. Estructura la respuesta como Condición, Evidencia, Limitaciones e "
    "Inspección sugerida. Usa como máximo 180 palabras."
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


def explanation_context(view: TruckView) -> dict:
    if view.stale or not view.connected or view.twin is None:
        raise LangflowError("Se necesita una observación vigente del nodo conectado.")
    packet = view.twin.observation
    component_diagnoses = view.twin.component_diagnoses
    prediction = view.prediction
    return {
        "truck_id": view.truck_id,
        "node_id": view.node_id,
        "deterministic_summary": {
            "asset_status": view.asset_status,
            "engine_status": view.twin.diagnosis.status,
            "forecast_status": prediction.status if prediction else None,
            "hours_to_threshold": (
                prediction.hours_to_threshold if prediction else None
            ),
            "active_alert_components": sorted(
                alert.component for alert in view.alerts if alert.closed_at is None
            ),
            "normal_components": sorted(
                diagnosis.component
                for diagnosis in component_diagnoses
                if diagnosis.status == "normal"
            ),
            "unknown_components": sorted(
                diagnosis.component
                for diagnosis in component_diagnoses
                if diagnosis.status == "unknown"
            ),
            "unavailable_readings": [
                reading.name for reading in packet.readings if reading.quality != "valid"
            ],
        },
        "profile": {
            "id": view.profile.id,
            "capacity_tonnes": view.profile.capacity_tonnes,
            "traction": view.profile.traction,
        },
        "work_orders": [asdict(order) for order in view.orders],
        "observation": {
            "event_id": packet.event_id,
            "source_time": packet.source_time,
            "operating_state": packet.operating_state,
            "cycles": packet.cycles,
            "readings": [asdict(reading) for reading in packet.readings],
        },
        "engine_diagnosis": asdict(view.twin.diagnosis),
        "component_statuses": [
            {"component": diagnosis.component, "status": diagnosis.status}
            for diagnosis in component_diagnoses
        ],
        "attention_diagnoses": [
            asdict(diagnosis)
            for diagnosis in component_diagnoses
            if diagnosis.status != "normal"
        ],
        "forecast": asdict(prediction) if prediction else None,
        "open_alerts": [
            asdict(alert) for alert in view.alerts if alert.closed_at is None
        ],
        "scope": "Synthetic academic simulation; explanatory support only.",
    }


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LangflowClient:
    def __init__(self, config: LangflowConfig, opener=None) -> None:
        self.config = config
        self._opener = opener or build_opener(_NoRedirect())

    def explain(self, view: TruckView) -> str:
        context = explanation_context(view)
        payload = {
            "input_value": FLOW_INSTRUCTIONS
            + "\nEVIDENCIA:\n"
            + json.dumps(
                context,
                default=lambda value: value.isoformat(),
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
