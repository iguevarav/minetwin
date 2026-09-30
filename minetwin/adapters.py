import csv
import io
import json
from datetime import datetime
from xml.etree import ElementTree as ET

from minetwin.domain import WHEELS, TelemetryPacket, TruckProfile, WireFormat
from minetwin.telemetry import TelemetryError, adapt_alpha_json

CSV_FIELDS = (
    "version",
    "event",
    "node",
    "asset",
    "captured_at",
    "mode",
    "status",
    "cycles",
    "load_pct",
    "engine_c",
    "vibration_in_s",
    "hours",
    "brake_f",
    "grade_ratio",
    *(f"pressure_{wheel}_bar" for wheel in WHEELS),
)


def _number(value, scale=1.0, offset=0.0):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "Valor booleano no admisible."
    try:
        return float(value) * scale + offset
    except (TypeError, ValueError, OverflowError):
        return "Valor numérico inválido."


def encode_payload(
    payload: dict, wire_format: WireFormat, profile: TruckProfile
) -> str:
    if wire_format == WireFormat.JSON:
        return json.dumps(payload, allow_nan=False)
    temperature_c = _number(payload.get("engine_temperature_f"), 5 / 9, -32 * 5 / 9)
    if wire_format == WireFormat.CSV:
        values = {
            "version": payload["schema_version"],
            "event": payload["event_id"],
            "node": payload["node_id"],
            "asset": payload["truck_id"],
            "captured_at": payload["source_time"],
            "mode": payload["operating_state"],
            "status": payload["asset_status"],
            "cycles": payload["cycles"],
            "load_pct": _number(payload.get("load_ratio"), 100),
            "engine_c": temperature_c,
            "vibration_in_s": _number(payload.get("engine_vibration_mm_s"), 1 / 25.4),
            "hours": payload.get("operating_hours"),
            "brake_f": _number(payload.get("brake_temperature_c"), 1.8, 32),
            "grade_ratio": _number(payload.get("slope_percent"), 0.01),
            **{
                f"pressure_{wheel}_bar": _number(payload.get(f"tire_{wheel}_kpa"), 0.01)
                for wheel in WHEELS
            },
        }
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(values)
        return buffer.getvalue()
    if wire_format != WireFormat.XML:
        raise TelemetryError("Formato no admitido.")
    root = ET.Element(
        "telemetry", version=str(payload["schema_version"]), event=payload["event_id"]
    )
    ET.SubElement(root, "origin", node=payload["node_id"], truck=payload["truck_id"])
    ET.SubElement(root, "time", captured=payload["source_time"])
    groups = {
        "operation": {
            "state": payload["operating_state"],
            "status": payload["asset_status"],
            "cycles": payload["cycles"],
            "load_tonnes": _number(payload.get("load_ratio"), profile.capacity_tonnes),
            "hours": payload.get("operating_hours"),
        },
        "engine": {
            "temperature_kelvin": _number(temperature_c, 1, 273.15),
            "vibration_m_s": _number(payload.get("engine_vibration_mm_s"), 0.001),
        },
        "brakes": {
            "temperature_kelvin": _number(
                payload.get("brake_temperature_c"), 1, 273.15
            ),
            "grade_percent": payload.get("slope_percent"),
        },
    }
    for tag, values in groups.items():
        ET.SubElement(
            root,
            tag,
            {key: str(value) for key, value in values.items() if value is not None},
        )
    tires = ET.SubElement(root, "tires")
    for wheel in WHEELS:
        pressure = _number(payload.get(f"tire_{wheel}_kpa"), 1 / 6.894757293168)
        ET.SubElement(
            tires,
            "wheel",
            id=wheel,
            **({"pressure_psi": str(pressure)} if pressure is not None else {}),
        )
    return ET.tostring(root, encoding="unicode")


def _csv_payload(raw: str) -> dict:
    reader = csv.DictReader(io.StringIO(raw))
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(set(reader.fieldnames))
        or set(reader.fieldnames) != set(CSV_FIELDS)
    ):
        raise TelemetryError("Cabecera CSV desconocida o con unidades incompatibles.")
    rows = list(reader)
    if (
        len(rows) != 1
        or None in rows[0]
        or any(value is None for value in rows[0].values())
    ):
        raise TelemetryError(
            "Cada paquete CSV debe contener exactamente una fila completa."
        )
    row = rows[0]
    return {
        "schema_version": int(row["version"]),
        "event_id": row["event"],
        "node_id": row["node"],
        "truck_id": row["asset"],
        "source_time": row["captured_at"],
        "operating_state": row["mode"],
        "asset_status": row["status"],
        "cycles": int(row["cycles"]),
        "load_ratio": _number(row["load_pct"], 0.01),
        "engine_temperature_f": _number(row["engine_c"], 1.8, 32),
        "engine_vibration_mm_s": _number(row["vibration_in_s"], 25.4),
        "operating_hours": _number(row["hours"]),
        "brake_temperature_c": _number(row["brake_f"], 5 / 9, -32 * 5 / 9),
        "slope_percent": _number(row["grade_ratio"], 100),
        **{
            f"tire_{wheel}_kpa": _number(row[f"pressure_{wheel}_bar"], 100)
            for wheel in WHEELS
        },
    }


def _xml_payload(raw: str, profile: TruckProfile) -> dict:
    if "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
        raise TelemetryError("No se admiten declaraciones de entidades XML.")
    root = ET.fromstring(raw)
    expected = {"origin", "time", "operation", "engine", "brakes", "tires"}
    if (
        root.tag != "telemetry"
        or {child.tag for child in root} != expected
        or len(root) != len(expected)
    ):
        raise TelemetryError("Estructura XML desconocida.")
    groups = {child.tag: child for child in root}
    origin, operation = groups["origin"].attrib, groups["operation"].attrib
    engine, brakes = groups["engine"].attrib, groups["brakes"].attrib
    allowed = {
        "origin": {"node", "truck"},
        "time": {"captured"},
        "operation": {"state", "status", "cycles", "load_tonnes", "hours"},
        "engine": {"temperature_kelvin", "vibration_m_s"},
        "brakes": {"temperature_kelvin", "grade_percent"},
        "tires": set(),
    }
    if (
        set(root.attrib) != {"version", "event"}
        or any(set(groups[tag].attrib) - names for tag, names in allowed.items())
        or any(len(groups[tag]) for tag in allowed if tag != "tires")
    ):
        raise TelemetryError("Campos XML o unidades desconocidos.")
    pressures = {}
    for wheel in groups["tires"]:
        identity = wheel.get("id")
        if (
            wheel.tag != "wheel"
            or identity not in WHEELS
            or identity in pressures
            or set(wheel.attrib) - {"id", "pressure_psi"}
        ):
            raise TelemetryError("Identidad o unidad de neumático inválida.")
        pressures[identity] = _number(wheel.get("pressure_psi"), 6.894757293168)
    return {
        "schema_version": int(root.attrib["version"]),
        "event_id": root.attrib["event"],
        "node_id": origin["node"],
        "truck_id": origin["truck"],
        "source_time": groups["time"].attrib["captured"],
        "operating_state": operation["state"],
        "asset_status": operation["status"],
        "cycles": int(operation["cycles"]),
        "load_ratio": _number(
            operation.get("load_tonnes"), 1 / profile.capacity_tonnes
        ),
        "engine_temperature_f": _number(
            engine.get("temperature_kelvin"), 1.8, 32 - 273.15 * 1.8
        ),
        "engine_vibration_mm_s": _number(engine.get("vibration_m_s"), 1000),
        "operating_hours": _number(operation.get("hours")),
        "brake_temperature_c": _number(brakes.get("temperature_kelvin"), 1, -273.15),
        "slope_percent": _number(brakes.get("grade_percent")),
        **{f"tire_{wheel}_kpa": pressures.get(wheel) for wheel in WHEELS},
    }


def adapt_telemetry(
    raw: str, received_time: datetime, wire_format: WireFormat, profile: TruckProfile
) -> TelemetryPacket:
    if len(raw) > 1_000_000:
        raise TelemetryError("Paquete demasiado grande.")
    if wire_format == WireFormat.JSON:
        return adapt_alpha_json(raw, received_time)
    try:
        if wire_format == WireFormat.XML:
            payload = _xml_payload(raw, profile)
        elif wire_format == WireFormat.CSV:
            payload = _csv_payload(raw)
        else:
            raise TelemetryError("Formato no admitido.")
        return adapt_alpha_json(json.dumps(payload), received_time)
    except (ET.ParseError, csv.Error, ValueError, KeyError, TypeError) as error:
        raise TelemetryError(
            f"Paquete {wire_format.value.upper()} rechazado: {error}"
        ) from error
