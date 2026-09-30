from collections import OrderedDict, deque
from dataclasses import dataclass

from minetwin.domain import OperatingState, Quality, TelemetryPacket


@dataclass(frozen=True)
class QualitySnapshot:
    accepted: int
    rejected: int
    duplicates: int
    late: int
    required: int
    present: int
    valid: int
    inconsistent: int

    @property
    def completeness(self) -> float | None:
        return self.present / self.required if self.required else None

    @property
    def validity(self) -> float | None:
        return self.valid / self.present if self.present else None


class QualityTracker:
    def __init__(self, retention: int) -> None:
        self.accepted = self.rejected = self.duplicates = self.late = 0
        self.required = self.present = self.valid = self.inconsistent = 0
        self._retention = retention
        self._seen: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._issues: deque[str] = deque(maxlen=min(retention, 100))

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(self._issues)

    def snapshot(self) -> QualitySnapshot:
        return QualitySnapshot(
            self.accepted,
            self.rejected,
            self.duplicates,
            self.late,
            self.required,
            self.present,
            self.valid,
            self.inconsistent,
        )

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self._issues.append(reason[:500])

    def check(
        self, packet: TelemetryPacket, last: TelemetryPacket | None
    ) -> str | None:
        key = (packet.node_id, packet.event_id)
        if key in self._seen:
            self.duplicates += 1
            self._issues.append("Evento duplicado: " + packet.event_id)
            return "Evento duplicado."
        self._seen[key] = None
        if len(self._seen) > self._retention:
            self._seen.popitem(last=False)
        if last and packet.source_time <= last.source_time:
            self.late += 1
            self._issues.append("Observación tardía: " + packet.event_id)
            return "La observación no es posterior al estado actual."
        return None

    def record(self, packet: TelemetryPacket) -> None:
        self.accepted += 1
        self.required += len(packet.readings)
        self.present += sum(
            reading.quality != Quality.MISSING for reading in packet.readings
        )
        self.valid += sum(
            reading.quality == Quality.VALID for reading in packet.readings
        )
        load = packet.reading("load_ratio").value
        if load is not None and (
            (
                packet.operating_state
                in (OperatingState.WAITING, OperatingState.RETURNING)
                and load > 0.05
            )
            or (packet.operating_state == OperatingState.HAULING and load < 0.5)
        ):
            self.inconsistent += 1
            self._issues.append("Carga incompatible con el régimen: " + packet.event_id)
