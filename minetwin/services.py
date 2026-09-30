import json
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

from minetwin.adapters import adapt_telemetry, encode_payload
from minetwin.analytics import EngineMonitor
from minetwin.components import ComponentMonitor
from minetwin.domain import (
    Alert,
    AssetStatus,
    EnginePrediction,
    EngineStatus,
    ForecastStatus,
    Intervention,
    MaintenancePolicy,
    OrderStatus,
    SimulationEvent,
    TelemetryPacket,
    TruckProfile,
    TwinState,
    WireFormat,
    WorkOrder,
)
from minetwin.maintenance import MaintenanceConfig, WorkOrderBook
from minetwin.prediction import EnginePredictor, PredictorConfig
from minetwin.quality import QualityTracker
from minetwin.simulation import SimulatedTruck, SimulationConfig
from minetwin.telemetry import TelemetryError


class MineTwin:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        maintenance_config: MaintenanceConfig | None = None,
        predictor_config: PredictorConfig | None = None,
        policy: MaintenancePolicy = MaintenancePolicy.NONE,
        event_sink: Callable[[SimulationEvent], None] | None = None,
        profile: TruckProfile | None = None,
        truck_id: str = "TRUCK-001",
        node_id: str = "alpha",
        wire_format: WireFormat = WireFormat.JSON,
    ) -> None:
        if not isinstance(policy, MaintenancePolicy):
            raise TypeError("Política de mantenimiento no válida.")
        self.config = config or SimulationConfig()
        self.maintenance_config = maintenance_config or MaintenanceConfig()
        self.predictor_config = predictor_config or PredictorConfig()
        self.policy = policy
        self._event_sink = event_sink
        self.profile = profile or TruckProfile()
        self.truck_id = truck_id
        self.node_id = node_id
        self.wire_format = wire_format
        self.quality = QualityTracker(self.config.history_limit)
        self._truck = SimulatedTruck(self.config, self.profile, truck_id, node_id)
        self._monitor = EngineMonitor(self.profile, self.config.step_seconds)
        self._predictor = EnginePredictor(self.profile, self.predictor_config)
        self._orders = WorkOrderBook(self.maintenance_config, truck_id)
        self._component_monitors = {
            component: ComponentMonitor(
                component, self.profile, self.config.step_seconds
            )
            for component in self.profile.components
            if component != "engine"
        }
        self._prediction: EnginePrediction | None = None
        self._prediction_history: deque[EnginePrediction] = deque(
            maxlen=self.config.history_limit
        )
        self._events: deque[SimulationEvent] = deque(maxlen=self.config.event_limit)
        self._event_sequence = 0
        self._history: deque[TelemetryPacket] = deque(maxlen=self.config.history_limit)
        self._raw_history: deque[str] = deque(maxlen=self.config.history_limit)
        self._closed_alerts: deque[Alert] = deque(maxlen=self.config.alert_limit)
        self._twin: TwinState | None = None
        self._active_alerts: dict[str, Alert] = {}
        self._alert_count = 0
        self._running = False

    @property
    def twin(self) -> TwinState | None:
        return self._twin

    @property
    def time(self) -> datetime:
        return self._truck.time

    @property
    def history(self) -> tuple[TelemetryPacket, ...]:
        return tuple(self._history)

    @property
    def raw_history(self) -> tuple[str, ...]:
        return tuple(self._raw_history)

    @property
    def alerts(self) -> tuple[Alert, ...]:
        return (*self._closed_alerts, *self._active_alerts.values())

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def running(self) -> bool:
        return self._running

    @property
    def asset_status(self) -> AssetStatus:
        return self._truck.status

    @property
    def prediction(self) -> EnginePrediction | None:
        return self._prediction

    @property
    def prediction_history(self) -> tuple[EnginePrediction, ...]:
        return tuple(self._prediction_history)

    @property
    def orders(self) -> tuple[WorkOrder, ...]:
        return self._orders.orders

    @property
    def current_order(self) -> WorkOrder | None:
        return self._orders.current

    @property
    def events(self) -> tuple[SimulationEvent, ...]:
        return tuple(self._events)

    @property
    def metrics(self) -> dict[str, float | int | None]:
        return {
            **self._truck.statistics(),
            "interventions": self._orders.completed_count,
            "alerts": self.alert_count,
        }

    def _record_event(
        self,
        kind: str,
        order_id: str | None = None,
        timestamp: datetime | None = None,
        operating_hours: float | None = None,
        alert_id: str | None = None,
        description: str = "",
        component: str = "engine",
    ) -> None:
        self._event_sequence += 1
        event = SimulationEvent(
            self._event_sequence,
            timestamp or self.time,
            kind,
            self._truck.operating_hours if operating_hours is None else operating_hours,
            order_id=order_id,
            alert_id=alert_id,
            description=description,
            component=component,
        )
        self._events.append(event)
        if self._event_sink:
            self._event_sink(event)

    def create_order(
        self,
        intervention: Intervention = Intervention.REPLACEMENT,
        reason: str = "Inspección e intervención manual del motor.",
        component: str = "engine",
    ) -> WorkOrder:
        if component not in self.profile.components:
            raise ValueError(
                "Componente no admitido; se dispone de motor, frenos y neumáticos."
            )
        previous = self._orders.current
        order = self._orders.create(intervention, reason, self.time, component)
        if previous is None:
            self._record_event(
                "order_created",
                order.id,
                description=f"{order.intervention.value}: {order.reason}",
                component=order.component,
            )
        return order

    def start_order(self, order_id: str) -> WorkOrder:
        previous = self._orders.get(order_id)
        order = self._orders.start(
            order_id,
            self.time,
            self.asset_status == AssetStatus.FAILED and previous.component == "engine",
        )
        if previous.status != OrderStatus.IN_PROGRESS:
            self._truck.begin_maintenance()
            self._record_event(
                "maintenance_started",
                order.id,
                description=f"{order.intervention.value}; duration_seconds={order.duration_seconds}; corrective={order.corrective}",
                component=order.component,
            )
        return order

    def queue_order(self, order_id: str) -> WorkOrder:
        previous = self._orders.get(order_id)
        order = self._orders.queue(order_id, self.time)
        if previous.status != OrderStatus.QUEUED:
            self._record_event("maintenance_queued", order.id, component=order.component)
        return order

    def cancel_order(self, order_id: str) -> WorkOrder:
        previous = self._orders.get(order_id)
        order = self._orders.cancel(order_id, self.time)
        if previous.status != OrderStatus.CANCELLED:
            self._record_event("order_cancelled", order.id, component=order.component)
        return order

    def complete_order(self, order_id: str) -> WorkOrder:
        previous = self._orders.get(order_id)
        order = self._orders.complete(order_id, self.time)
        if previous.status != OrderStatus.COMPLETED:
            self._truck.repair(
                order.intervention,
                self.maintenance_config.remaining_wear_fraction,
                order.component,
            )
            if order.component == "engine":
                self._predictor.reset()
                self._monitor = EngineMonitor(self.profile, self.config.step_seconds)
            else:
                self._component_monitors[order.component].reset()
            self._record_event(
                "maintenance_completed", order.id, component=order.component
            )
        return order

    def _apply_policy(self) -> None:
        if self.policy == MaintenancePolicy.NONE:
            return
        order = self.current_order
        if order and order.status == OrderStatus.IN_PROGRESS:
            return
        reason = None
        if self.asset_status == AssetStatus.FAILED:
            reason = "Reparación correctiva tras falla observada."
        elif self.twin and self.twin.diagnosis.status == EngineStatus.ALERT:
            reason = "Intervención por umbral persistente del motor."
        elif (
            self.policy == MaintenancePolicy.PREDICTIVE
            and self.prediction
            and self.prediction.status == ForecastStatus.ESTIMABLE
            and self.prediction.hours_to_threshold is not None
            and self.prediction.hours_to_threshold
            <= self.maintenance_config.predictive_margin_hours
        ):
            reason = "Intervención por proximidad pronosticada al umbral."
        if reason or order:
            order = order or self.create_order(
                reason=reason or "Mantenimiento programado."
            )
            self.start_order(order.id)

    def start(self) -> None:
        self._running = True

    def pause(self) -> None:
        self._running = False

    def advance(self, steps: int = 1) -> TwinState:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps debe ser un entero positivo.")
        for _ in range(steps):
            self._apply_policy()
            target_time = self.time + timedelta(seconds=self.config.step_seconds)
            while self.time < target_time:
                order = self.current_order
                boundary = (
                    min(target_time, order.due_at)
                    if order and order.due_at
                    else target_time
                )
                raw = self._truck.advance((boundary - self.time).total_seconds())
                for timestamp, hours in self._truck.drain_failures():
                    self._record_event(
                        "failure", timestamp=timestamp, operating_hours=hours
                    )
                if (
                    order
                    and order.status == OrderStatus.IN_PROGRESS
                    and order.due_at == self.time
                ):
                    self.complete_order(order.id)
                    raw = self._truck.snapshot_json()
                self.ingest(
                    encode_payload(json.loads(raw), self.wire_format, self.profile),
                    self.time,
                )
        assert self._twin is not None
        return self._twin

    def ingest(self, raw: str, received_time: datetime) -> TwinState:
        try:
            packet = adapt_telemetry(raw, received_time, self.wire_format, self.profile)
            if packet.truck_id != self.truck_id or packet.node_id != self.node_id:
                raise TelemetryError("La identidad no corresponde al activo local.")
        except TelemetryError as error:
            self.quality.reject(str(error))
            raise
        reason = self.quality.check(
            packet, self.twin.observation if self.twin else None
        )
        if reason:
            raise TelemetryError(reason)
        self.quality.record(packet)
        diagnosis = self._monitor.evaluate(packet)
        component_diagnoses = tuple(
            monitor.evaluate(packet) for monitor in self._component_monitors.values()
        )
        for result in (diagnosis, *component_diagnoses):
            self._update_alert(result, packet)
        version = self._twin.version + 1 if self._twin else 1
        self._twin = TwinState(packet, diagnosis, version, component_diagnoses)
        self._prediction = self._predictor.evaluate(packet, received_time)
        self._prediction_history.append(self._prediction)
        self._history.append(packet)
        self._raw_history.append(raw)
        return self._twin

    def _update_alert(self, diagnosis, packet: TelemetryPacket) -> None:
        component = diagnosis.component
        active = self._active_alerts.get(component)
        if diagnosis.status == EngineStatus.ALERT and active is None:
            self._alert_count += 1
            alert = Alert(
                id=f"{component}-{self._alert_count:06d}",
                truck_id=packet.truck_id,
                opened_at=packet.source_time,
                explanation=diagnosis.explanation,
                rule_id=diagnosis.rule_id,
                component=component,
            )
            self._active_alerts[component] = alert
            self._record_event(
                "alert_opened",
                timestamp=packet.source_time,
                operating_hours=packet.reading("operating_hours").value,
                alert_id=alert.id,
                description=diagnosis.explanation,
                component=component,
            )
        elif diagnosis.status == EngineStatus.NORMAL and active:
            self._record_event(
                "alert_closed",
                timestamp=packet.source_time,
                operating_hours=packet.reading("operating_hours").value,
                alert_id=active.id,
                description=diagnosis.explanation,
                component=component,
            )
            self._closed_alerts.append(replace(active, closed_at=packet.source_time))
            del self._active_alerts[component]
