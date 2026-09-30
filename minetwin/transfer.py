import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from pathlib import Path


class TransferDirection(StrEnum):
    NODE_TO_COORDINATOR = "node_to_coordinator"
    COORDINATOR_TO_NODE = "coordinator_to_node"


class MessageType(StrEnum):
    PUBLISHED_STATE = "published_state"
    MODEL_PARAMETERS = "model_parameters"
    MODEL_DISTRIBUTION = "model_distribution"
    COMMAND = "command"


@dataclass(frozen=True)
class TransferRecord:
    sequence: int
    node_id: str
    direction: TransferDirection
    message_type: MessageType
    payload_bytes: int
    raw_records: int
    round_id: int | None = None
    asset_id: str | None = None
    model_id: str | None = None


class TransferLedger:
    def __init__(self) -> None:
        self._records: list[TransferRecord] = []

    @property
    def records(self) -> tuple[TransferRecord, ...]:
        return tuple(self._records)

    def record_payload(
        self,
        node_id: str,
        direction: TransferDirection,
        message_type: MessageType,
        payload,
        raw_records: int = 0,
        round_id: int | None = None,
        asset_id: str | None = None,
        model_id: str | None = None,
    ) -> TransferRecord:
        serialized = json.dumps(
            payload,
            default=_json_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return self.record_bytes(
            node_id,
            direction,
            message_type,
            len(serialized),
            raw_records,
            round_id,
            asset_id,
            model_id,
        )

    def record_bytes(
        self,
        node_id: str,
        direction: TransferDirection,
        message_type: MessageType,
        payload_bytes: int,
        raw_records: int = 0,
        round_id: int | None = None,
        asset_id: str | None = None,
        model_id: str | None = None,
    ) -> TransferRecord:
        if not node_id:
            raise ValueError("node_id no puede estar vacio.")
        if type(payload_bytes) is not int or payload_bytes < 0:
            raise ValueError("payload_bytes debe ser un entero no negativo.")
        if type(raw_records) is not int or raw_records < 0:
            raise ValueError("raw_records debe ser un entero no negativo.")
        record = TransferRecord(
            sequence=len(self._records) + 1,
            node_id=node_id,
            direction=direction,
            message_type=message_type,
            payload_bytes=payload_bytes,
            raw_records=raw_records,
            round_id=round_id,
            asset_id=asset_id,
            model_id=model_id,
        )
        self._records.append(record)
        return record

    def summary(self) -> dict:
        return {
            "messages": len(self._records),
            "payload_bytes": sum(record.payload_bytes for record in self._records),
            "raw_records": sum(record.raw_records for record in self._records),
            "by_type": {
                message_type: {
                    "messages": sum(
                        record.message_type == message_type for record in self._records
                    ),
                    "payload_bytes": sum(
                        record.payload_bytes
                        for record in self._records
                        if record.message_type == message_type
                    ),
                }
                for message_type in MessageType
            },
        }

    def write_csv(self, path: Path) -> None:
        fields = tuple(TransferRecord.__dataclass_fields__)
        with Path(path).open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(record) for record in self._records)


def _json_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Tipo no serializable: {type(value).__name__}.")
