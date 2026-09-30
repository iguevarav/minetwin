from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from minetwin.adapters import adapt_telemetry
from minetwin.domain import (
    Alert,
    AssetStatus,
    EnginePrediction,
    ForecastStatus,
    Intervention,
    TruckProfile,
    TwinState,
    WireFormat,
    WorkOrder,
)
from minetwin.maintenance import MaintenanceConfig
from minetwin.prediction import PredictorConfig
from minetwin.quality import QualitySnapshot
from minetwin.services import MineTwin
from minetwin.simulation import SimulationConfig
from minetwin.telemetry import TelemetryError

PROFILES = (
    TruckProfile(),
    TruckProfile("SIM-BETA", 220, "electric", 9, 0.8, 65, 800),
    TruckProfile("SIM-GAMMA", 180, "mechanical", -5, 1.2, 50, 700),
)


@dataclass(frozen=True)
class TruckView:
    truck_id: str
    node_id: str
    profile: TruckProfile
    twin: TwinState | None
    prediction: EnginePrediction | None
    asset_status: AssetStatus
    alerts: tuple[Alert, ...]
    orders: tuple[WorkOrder, ...]
    metrics: tuple[tuple[str, float | int | None], ...]
    published_at: datetime
    connected: bool = True
    stale: bool = False
    age_seconds: float | None = None


class FederatedNode:
    def __init__(
        self, node_id: str, wire_format: WireFormat, trucks: dict[str, MineTwin]
    ) -> None:
        self.id = node_id
        self.wire_format = wire_format
        self.trucks = trucks
        self.connected = True
        self.rejections = 0
        self.issues: deque[str] = deque(maxlen=100)
        self._seen: OrderedDict[str, str] = OrderedDict()
        self._retention = sum(truck.config.history_limit for truck in trucks.values())

    def _remember(self, event_id: str, truck_id: str) -> None:
        self._seen[event_id] = truck_id
        if len(self._seen) > self._retention:
            self._seen.popitem(last=False)

    def advance(self) -> None:
        for truck in self.trucks.values():
            try:
                truck.advance()
                if truck.twin:
                    self._remember(truck.twin.observation.event_id, truck.truck_id)
            except (TelemetryError, ValueError) as error:
                self.issues.append(f"{truck.truck_id}: {error}"[:500])

    def receive(self, truck_id: str, raw: str, received_time: datetime) -> bool:
        truck = self.trucks.get(truck_id)
        if truck is None:
            self.rejections += 1
            self.issues.append("Camión ajeno al nodo: " + truck_id)
            return False
        try:
            packet = adapt_telemetry(
                raw, received_time, self.wire_format, truck.profile
            )
        except TelemetryError as error:
            truck.quality.reject(str(error))
            self.issues.append(str(error)[:500])
            return False
        if packet.event_id in self._seen and self._seen[packet.event_id] != truck_id:
            truck.quality.duplicates += 1
            self.issues.append("Identificador de evento repetido en el nodo.")
            return False
        try:
            truck.ingest(raw, received_time)
        except TelemetryError as error:
            self.issues.append(str(error)[:500])
            return False
        self._remember(packet.event_id, truck_id)
        return True

    def quality(self) -> QualitySnapshot:
        snapshots = [truck.quality.snapshot() for truck in self.trucks.values()]
        values = {
            name: sum(getattr(snapshot, name) for snapshot in snapshots)
            for name in QualitySnapshot.__dataclass_fields__
        }
        values["rejected"] += self.rejections
        return QualitySnapshot(**values)


class FederatedFleet:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        first_truck: MineTwin | None = None,
        predictor_config: PredictorConfig | None = None,
        maintenance_config: MaintenanceConfig | None = None,
    ) -> None:
        self.config = config or (
            first_truck.config if first_truck else SimulationConfig()
        )
        self.time = first_truck.time if first_truck else self.config.start_time
        self.predictor_config = predictor_config or (
            first_truck.predictor_config if first_truck else PredictorConfig()
        )
        self.maintenance_config = maintenance_config or (
            first_truck.maintenance_config if first_truck else MaintenanceConfig()
        )
        self.running = False
        self.nodes: dict[str, FederatedNode] = {}
        self._views: dict[str, TruckView] = {}
        self._membership: dict[str, str] = {}
        self._quality_views: dict[str, QualitySnapshot] = {}
        self._issue_views: dict[str, tuple[str, ...]] = {}
        for node_index, (node_id, wire_format, profile) in enumerate(
            zip(
                ("alpha", "beta", "gamma"),
                WireFormat,
                PROFILES,
                strict=True,
            )
        ):
            trucks = {}
            for unit in range(3):
                index = node_index * 3 + unit
                truck_id = f"TRUCK-{index + 1:03d}"
                if index == 0 and first_truck:
                    truck = first_truck
                else:
                    truck = MineTwin(
                        replace(self.config, seed=self.config.seed + index),
                        maintenance_config=self.maintenance_config,
                        predictor_config=self.predictor_config,
                        profile=profile,
                        truck_id=truck_id,
                        node_id=node_id,
                        wire_format=wire_format,
                    )
                    elapsed = (self.time - self.config.start_time).total_seconds()
                    if elapsed:
                        truck.advance(round(elapsed / self.config.step_seconds))
                trucks[truck_id] = truck
                self._membership[truck_id] = node_id
            self.nodes[node_id] = FederatedNode(node_id, wire_format, trucks)
        self.publish()

    @property
    def truck_ids(self) -> tuple[str, ...]:
        return tuple(self._membership)

    def start(self) -> None:
        self.running = True

    def pause(self) -> None:
        self.running = False

    def node_for(self, truck_id: str) -> FederatedNode:
        if truck_id not in self._membership:
            raise ValueError("Camión no registrado.")
        return self.nodes[self._membership[truck_id]]

    def set_connected(self, node_id: str, connected: bool) -> None:
        self.nodes[node_id].connected = connected
        if connected:
            self.publish(node_id)

    def advance(self, steps: int = 1) -> None:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps debe ser un entero positivo.")
        for _ in range(steps):
            for node in self.nodes.values():
                node.advance()
            self.time += timedelta(seconds=self.config.step_seconds)
            self.publish()

    def publish(self, node_id: str | None = None) -> None:
        for node in self.nodes.values():
            if not node.connected or (node_id and node.id != node_id):
                continue
            self._quality_views[node.id] = node.quality()
            self._issue_views[node.id] = tuple(node.issues)
            for truck in node.trucks.values():
                self._views[truck.truck_id] = TruckView(
                    truck.truck_id,
                    node.id,
                    truck.profile,
                    truck.twin,
                    truck.prediction,
                    truck.asset_status,
                    truck.alerts,
                    truck.orders,
                    tuple(truck.metrics.items()),
                    self.time,
                )

    def view(self, truck_id: str) -> TruckView:
        node = self.node_for(truck_id)
        view = self._views[truck_id]
        age = (
            (self.time - view.twin.observation.source_time).total_seconds()
            if view.twin
            else None
        )
        stale = not node.connected or (
            age is not None and age > self.predictor_config.stale_after_seconds
        )
        prediction = view.prediction
        if stale and prediction:
            prediction = replace(
                prediction,
                status=ForecastStatus.NOT_ESTIMABLE,
                hours_to_threshold=None,
                reason="Vista central desactualizada; no hay un pronóstico vigente.",
            )
        return replace(
            view,
            prediction=prediction,
            connected=node.connected,
            stale=stale,
            age_seconds=age,
        )

    def access(self, truck_id: str):
        self.node_for(truck_id)
        return TruckConnection(self, truck_id)

    def quality(self, node_id: str) -> QualitySnapshot:
        return self._quality_views[node_id]

    def issues(self, node_id: str) -> tuple[str, ...]:
        return self._issue_views[node_id]

    def demonstrate_bad_packet(self, node_id: str, duplicate: bool = False) -> None:
        node = self.nodes[node_id]
        if not node.connected:
            raise ConnectionError("El nodo está desconectado.")
        truck = next(iter(node.trucks.values()))
        raw = (
            truck.raw_history[-1]
            if duplicate and truck.raw_history
            else "invalid-packet"
        )
        node.receive(truck.truck_id, raw, self.time)
        self.publish(node_id)

    def local(self, truck_id: str) -> MineTwin:
        node = self.node_for(truck_id)
        if not node.connected:
            raise ConnectionError(
                "El nodo está desconectado. No se permiten consultas ni órdenes remotas."
            )
        return node.trucks[truck_id]


class TruckConnection:
    def __init__(self, fleet: FederatedFleet, truck_id: str) -> None:
        self._fleet = fleet
        self.truck_id = truck_id

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        value = getattr(self._fleet.local(self.truck_id), name)
        if not callable(value):
            return value
        raise AttributeError("Operación remota no disponible: " + name)

    @property
    def twin(self):
        return self._fleet.view(self.truck_id).twin

    @property
    def prediction(self):
        return self._fleet.view(self.truck_id).prediction

    @property
    def time(self):
        return self._fleet.time

    @property
    def running(self):
        return self._fleet.running

    def start(self):
        self._fleet.start()

    def pause(self):
        self._fleet.pause()

    def advance(self, steps=1):
        self._fleet.advance(steps)

    def create_order(
        self,
        intervention=Intervention.REPLACEMENT,
        reason="Intervención manual.",
        component="engine",
    ):
        result = self._fleet.local(self.truck_id).create_order(
            intervention, reason, component
        )
        self._fleet.publish(self._fleet.node_for(self.truck_id).id)
        return result

    def start_order(self, order_id):
        result = self._fleet.local(self.truck_id).start_order(order_id)
        self._fleet.publish(self._fleet.node_for(self.truck_id).id)
        return result

    def cancel_order(self, order_id):
        result = self._fleet.local(self.truck_id).cancel_order(order_id)
        self._fleet.publish(self._fleet.node_for(self.truck_id).id)
        return result
