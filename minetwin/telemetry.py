import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from minetwin.domain import (
    WHEELS,
    AssetStatus,
    OperatingState,
    Quality,
    SensorReading,
    TelemetryPacket,
)


class TelemetryError(ValueError):
    pass


class AlphaPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    event_id: Annotated[str, Field(strict=True, min_length=1)]
    node_id: Literal["alpha", "beta", "gamma"]
    truck_id: Annotated[str, Field(strict=True, pattern=r"^TRUCK-\d{3}$")]
    source_time: AwareDatetime
    operating_state: OperatingState
    asset_status: AssetStatus = AssetStatus.AVAILABLE
    cycles: Annotated[int, Field(strict=True, ge=0)]
    load_ratio: Any = None
    engine_temperature_f: Any = None
    engine_vibration_mm_s: Any = None
    operating_hours: Any = None
    brake_temperature_c: Any = None
    slope_percent: Any = None
    tire_FL_kpa: Any = None
    tire_FR_kpa: Any = None
    tire_RL1_kpa: Any = None
    tire_RR1_kpa: Any = None
    tire_RL2_kpa: Any = None
    tire_RR2_kpa: Any = None


def _reading(
    name: str,
    value: object,
    unit: str,
    bounds: tuple[float, float],
    scale: float = 1.0,
    offset: float = 0.0,
) -> SensorReading:
    if value is None:
        return SensorReading(name, None, unit, Quality.MISSING, "Lectura ausente.")
    if type(value) not in (int, float):
        return SensorReading(name, None, unit, Quality.INVALID, "Valor no numérico.")
    try:
        canonical = float(value) * scale + offset
    except OverflowError:
        canonical = math.inf
    if not math.isfinite(canonical) or not bounds[0] <= canonical <= bounds[1]:
        return SensorReading(
            name, None, unit, Quality.INVALID, "Fuera del rango simulado admisible."
        )
    return SensorReading(name, canonical, unit, Quality.VALID)


def adapt_alpha_json(raw: str, received_time: datetime) -> TelemetryPacket:
    if len(raw) > 1_000_000:
        raise TelemetryError("Paquete demasiado grande.")
    try:
        payload = AlphaPayload.model_validate_json(raw)
    except ValidationError as error:
        raise TelemetryError(f"Paquete Alpha rechazado: {error}") from error
    if received_time.utcoffset() is None or received_time < payload.source_time:
        raise TelemetryError("Recepción inválida respecto al instante de captura.")
    return TelemetryPacket(
        event_id=payload.event_id,
        node_id=payload.node_id,
        truck_id=payload.truck_id,
        source_time=payload.source_time,
        received_time=received_time,
        operating_state=payload.operating_state,
        cycles=payload.cycles,
        asset_status=payload.asset_status,
        readings=(
            _reading("load_ratio", payload.load_ratio, "1", (0, 1.5)),
            _reading(
                "engine_temperature",
                payload.engine_temperature_f,
                "°C",
                (-40, 180),
                5 / 9,
                -32 * 5 / 9,
            ),
            _reading(
                "engine_vibration", payload.engine_vibration_mm_s, "mm/s", (0, 50)
            ),
            _reading("operating_hours", payload.operating_hours, "h", (0, 1_000_000)),
            _reading(
                "brake_temperature", payload.brake_temperature_c, "°C", (-40, 1000)
            ),
            _reading("slope_percent", payload.slope_percent, "%", (-30, 30)),
            *(
                _reading(
                    f"tire_pressure_{wheel}",
                    getattr(payload, f"tire_{wheel}_kpa"),
                    "kPa",
                    (0, 2000),
                )
                for wheel in WHEELS
            ),
        ),
    )
