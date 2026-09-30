import csv
import io
import json
from xml.etree import ElementTree as ET

from minetwin.data.scania.contracts import DatasetSplit, FeatureReadout
from minetwin.domain import WireFormat

IDENTITY_FIELDS = ("split", "vehicle_id", "time_step")


def encode_readout(readout: FeatureReadout, wire_format: WireFormat) -> str:
    if wire_format == WireFormat.JSON:
        return json.dumps(
            {
                "split": readout.split,
                "vehicle_id": readout.vehicle_id,
                "time_step": readout.time_step,
                "features": dict(zip(readout.feature_names, readout.values, strict=True)),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    if wire_format == WireFormat.CSV:
        buffer = io.StringIO(newline="")
        fields = (*IDENTITY_FIELDS, *readout.feature_names)
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "split": readout.split,
                "vehicle_id": readout.vehicle_id,
                "time_step": readout.time_step,
                **dict(zip(readout.feature_names, readout.values, strict=True)),
            }
        )
        return buffer.getvalue()
    if wire_format != WireFormat.XML:
        raise ValueError("Formato no admitido.")
    root = ET.Element(
        "readout",
        split=readout.split,
        vehicle_id=readout.vehicle_id,
        time_step=str(readout.time_step),
    )
    for name, value in zip(readout.feature_names, readout.values, strict=True):
        attributes = {"name": name}
        if value is not None:
            attributes["value"] = str(value)
        ET.SubElement(root, "feature", attributes)
    return ET.tostring(root, encoding="unicode")


def decode_readout(
    raw: str,
    wire_format: WireFormat,
    feature_names: tuple[str, ...],
) -> FeatureReadout:
    if len(raw) > 1_000_000:
        raise ValueError("Readout demasiado grande.")
    if wire_format == WireFormat.JSON:
        payload = json.loads(raw)
        if set(payload) != {*IDENTITY_FIELDS, "features"}:
            raise ValueError("Estructura JSON desconocida.")
        features = payload["features"]
        if not isinstance(features, dict) or tuple(features) != feature_names:
            raise ValueError("El esquema JSON no coincide.")
        values = tuple(_number(features[name]) for name in feature_names)
    elif wire_format == WireFormat.CSV:
        reader = csv.DictReader(io.StringIO(raw))
        if tuple(reader.fieldnames or ()) != (*IDENTITY_FIELDS, *feature_names):
            raise ValueError("El esquema CSV no coincide.")
        rows = list(reader)
        if len(rows) != 1 or None in rows[0]:
            raise ValueError("El CSV debe contener un unico readout completo.")
        payload = rows[0]
        values = tuple(_number(payload[name]) for name in feature_names)
    elif wire_format == WireFormat.XML:
        if "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
            raise ValueError("No se admiten entidades XML.")
        root = ET.fromstring(raw)
        if root.tag != "readout" or set(root.attrib) != set(IDENTITY_FIELDS):
            raise ValueError("Estructura XML desconocida.")
        features = list(root)
        if any(
            child.tag != "feature"
            or set(child.attrib) - {"name", "value"}
            or len(child)
            for child in features
        ):
            raise ValueError("Variable XML invalida.")
        if tuple(child.get("name") for child in features) != feature_names:
            raise ValueError("El esquema XML no coincide.")
        payload = root.attrib
        values = tuple(_number(child.get("value")) for child in features)
    else:
        raise ValueError("Formato no admitido.")
    return FeatureReadout(
        split=DatasetSplit(payload["split"]),
        vehicle_id=payload["vehicle_id"],
        time_step=float(payload["time_step"]),
        feature_names=feature_names,
        values=values,
    )


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise TypeError("Los booleanos no son valores operativos.")
    return float(value)
