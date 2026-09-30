import math
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from minetwin.domain import Intervention, OrderStatus, WorkOrder


@dataclass(frozen=True)
class MaintenanceConfig:
    partial_repair_seconds: int = 600
    replacement_seconds: int = 1200
    corrective_seconds: int = 3600
    remaining_wear_fraction: float = 0.25
    order_history_limit: int = 100
    predictive_margin_hours: float = 4.5
    brake_service_seconds: int = 900
    tire_service_seconds: int = 600

    def __post_init__(self) -> None:
        for field in (
            "partial_repair_seconds",
            "replacement_seconds",
            "corrective_seconds",
            "order_history_limit",
            "brake_service_seconds",
            "tire_service_seconds",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} debe ser un entero positivo.")
        if (
            not math.isfinite(self.remaining_wear_fraction)
            or not 0 <= self.remaining_wear_fraction < 1
        ):
            raise ValueError("La fracción de desgaste restante debe estar entre 0 y 1.")
        if (
            not math.isfinite(self.predictive_margin_hours)
            or self.predictive_margin_hours <= 0
        ):
            raise ValueError("El margen predictivo debe ser positivo y finito.")


class WorkOrderBook:
    def __init__(self, config: MaintenanceConfig, truck_id: str = "TRUCK-001") -> None:
        self.config = config
        self.truck_id = truck_id
        self._current: WorkOrder | None = None
        self._history: deque[WorkOrder] = deque(maxlen=config.order_history_limit)
        self._sequence = 0
        self.completed_count = 0

    @property
    def current(self) -> WorkOrder | None:
        return self._current

    @property
    def orders(self) -> tuple[WorkOrder, ...]:
        return (*self._history, *((self._current,) if self._current else ()))

    def get(self, order_id: str) -> WorkOrder:
        for order in self.orders:
            if order.id == order_id:
                return order
        raise ValueError("La orden no existe en el historial disponible.")

    def create(
        self,
        intervention: Intervention,
        reason: str,
        now: datetime,
        component: str = "engine",
    ) -> WorkOrder:
        if not isinstance(intervention, Intervention):
            raise TypeError("Intervención no válida.")
        if not reason.strip():
            raise ValueError("La orden necesita un motivo.")
        if self._current:
            if self._current.component != component:
                raise ValueError(
                    "El camión ya tiene una orden pendiente para otro componente."
                )
            return self._current
        self._sequence += 1
        self._current = WorkOrder(
            id=f"WO-{self._sequence:06d}",
            truck_id=self.truck_id,
            component=component,
            intervention=intervention,
            reason=reason.strip(),
            created_at=now,
        )
        return self._current

    def start(self, order_id: str, now: datetime, corrective: bool) -> WorkOrder:
        order = self.get(order_id)
        if order.status == OrderStatus.IN_PROGRESS:
            return order
        if order.status not in (OrderStatus.OPEN, OrderStatus.QUEUED):
            raise ValueError("Solo se puede iniciar una orden abierta o en cola.")
        duration = (
            self.config.corrective_seconds
            if corrective
            else (
                self.config.partial_repair_seconds
                if order.intervention == Intervention.PARTIAL_REPAIR
                else self.config.replacement_seconds
            )
        )
        if order.component == "brakes":
            duration = self.config.brake_service_seconds
        elif order.component.startswith("tire:"):
            duration = self.config.tire_service_seconds
        self._current = replace(
            order,
            status=OrderStatus.IN_PROGRESS,
            started_at=now,
            due_at=now + timedelta(seconds=duration),
            duration_seconds=duration,
            corrective=corrective,
        )
        return self._current

    def queue(self, order_id: str, now: datetime) -> WorkOrder:
        order = self.get(order_id)
        if order.status == OrderStatus.QUEUED:
            return order
        if order.status != OrderStatus.OPEN:
            raise ValueError("Solo se puede encolar una orden abierta.")
        self._current = replace(
            order,
            status=OrderStatus.QUEUED,
            queued_at=now,
        )
        return self._current

    def complete(self, order_id: str, now: datetime) -> WorkOrder:
        order = self.get(order_id)
        if order.status == OrderStatus.COMPLETED:
            return order
        if order.status != OrderStatus.IN_PROGRESS or order.due_at is None:
            raise ValueError("La orden no está en progreso.")
        if now < order.due_at:
            raise ValueError("La intervención todavía no ha cumplido su duración.")
        completed = replace(order, status=OrderStatus.COMPLETED, closed_at=now)
        self._history.append(completed)
        self._current = None
        self.completed_count += 1
        return completed

    def cancel(self, order_id: str, now: datetime) -> WorkOrder:
        order = self.get(order_id)
        if order.status == OrderStatus.CANCELLED:
            return order
        if order.status not in (OrderStatus.OPEN, OrderStatus.QUEUED):
            raise ValueError("Solo se puede cancelar una orden que no ha comenzado.")
        cancelled = replace(order, status=OrderStatus.CANCELLED, closed_at=now)
        self._history.append(cancelled)
        self._current = None
        return cancelled
