import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from minetwin.domain import (
    AssetStatus,
    EngineStatus,
    ForecastStatus,
    Intervention,
    OrderStatus,
    Scenario,
)
from minetwin.federation import FederatedFleet
from minetwin.maintenance import MaintenanceConfig
from minetwin.publication import PublishedOperationalState
from minetwin.simulation import SimulationConfig
from minetwin.transfer import MessageType, TransferDirection, TransferLedger


class FleetPolicy(StrEnum):
    P0_NONE = "p0_none"
    P1_THRESHOLD = "p1_threshold"
    P2_LOCAL_PREDICTIVE = "p2_local_predictive"
    P3_COORDINATED = "p3_coordinated"


@dataclass(frozen=True)
class WorkshopConfig:
    bays: int = 2
    minimum_available: int = 6
    preventive_cost: float = 1_000.0
    corrective_cost: float = 5_000.0
    queue_cost_per_hour: float = 50.0

    def __post_init__(self) -> None:
        if type(self.bays) is not int or self.bays < 1:
            raise ValueError("bays debe ser un entero positivo.")
        if type(self.minimum_available) is not int or self.minimum_available < 0:
            raise ValueError("minimum_available debe ser un entero no negativo.")
        for name in ("preventive_cost", "corrective_cost", "queue_cost_per_hour"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} debe ser finito y no negativo.")


@dataclass(frozen=True)
class WorkshopRequest:
    id: str
    truck_id: str
    node_id: str
    order_id: str
    reason: str
    queued_at: datetime
    urgency: float
    corrective: bool
    status: OrderStatus = OrderStatus.QUEUED
    started_at: datetime | None = None
    completed_at: datetime | None = None


class OperationalStatePublisher:
    @staticmethod
    def publish(truck) -> PublishedOperationalState:
        if truck.twin is None:
            raise ValueError("El camion todavia no tiene una observacion.")
        observation = truck.twin.observation
        prediction = truck.prediction
        return PublishedOperationalState(
            truck_id=truck.truck_id,
            node_id=truck.node_id,
            version=truck.twin.version,
            source_time=observation.source_time.isoformat(),
            asset_status=truck.asset_status,
            engine_status=truck.twin.diagnosis.status,
            forecast_status=prediction.status if prediction else None,
            hours_to_threshold=(
                prediction.hours_to_threshold if prediction else None
            ),
            cycles=observation.cycles,
            capacity_tonnes=truck.profile.capacity_tonnes,
            active_alert=any(
                alert.component == "engine" and alert.closed_at is None
                for alert in truck.alerts
            ),
            open_order=truck.current_order is not None,
        )


class SharedWorkshop:
    def __init__(self, config: WorkshopConfig, fleet_size: int) -> None:
        if config.minimum_available > fleet_size:
            raise ValueError("minimum_available no puede superar el tamano de flota.")
        self.config = config
        self._requests: dict[str, WorkshopRequest] = {}
        self.preventive_starts = 0
        self.corrective_starts = 0

    @property
    def requests(self) -> tuple[WorkshopRequest, ...]:
        return tuple(self._requests.values())

    @property
    def queued(self) -> tuple[WorkshopRequest, ...]:
        return tuple(
            request
            for request in self._requests.values()
            if request.status == OrderStatus.QUEUED
        )

    @property
    def has_idle_bay(self) -> bool:
        return self._active_count() < self.config.bays

    def enqueue(
        self,
        truck,
        reason: str,
        urgency: float,
        now: datetime,
    ) -> WorkshopRequest:
        existing = next(
            (
                request
                for request in self._requests.values()
                if request.truck_id == truck.truck_id
                and request.status in (OrderStatus.QUEUED, OrderStatus.IN_PROGRESS)
            ),
            None,
        )
        if existing:
            return existing
        order = truck.current_order or truck.create_order(
            Intervention.REPLACEMENT, reason, "engine"
        )
        truck.queue_order(order.id)
        request = WorkshopRequest(
            id=f"{truck.truck_id}:{order.id}",
            truck_id=truck.truck_id,
            node_id=truck.node_id,
            order_id=order.id,
            reason=reason,
            queued_at=now,
            urgency=urgency,
            corrective=truck.asset_status == AssetStatus.FAILED,
        )
        self._requests[request.id] = request
        return request

    def synchronize(self, trucks: dict[str, object], now: datetime) -> None:
        for request in self._requests.values():
            if request.status != OrderStatus.IN_PROGRESS:
                continue
            truck = trucks[request.truck_id]
            order = next(
                (item for item in truck.orders if item.id == request.order_id), None
            )
            if order and order.status == OrderStatus.COMPLETED:
                self._requests[request.id] = replace(
                    request,
                    status=OrderStatus.COMPLETED,
                    completed_at=order.closed_at or now,
                )

    def admit(self, trucks: dict[str, object], now: datetime) -> None:
        while self._active_count() < self.config.bays:
            available = sum(
                truck.asset_status == AssetStatus.AVAILABLE for truck in trucks.values()
            )
            candidate = next(
                (
                    request
                    for request in sorted(
                        self.queued,
                        key=lambda item: (
                            not item.corrective,
                            -item.urgency,
                            item.queued_at,
                            item.truck_id,
                        ),
                    )
                    if self._can_admit(request, trucks, available)
                ),
                None,
            )
            if candidate is None:
                return
            truck = trucks[candidate.truck_id]
            order = truck.start_order(candidate.order_id)
            self._requests[candidate.id] = replace(
                candidate,
                corrective=order.corrective,
                status=OrderStatus.IN_PROGRESS,
                started_at=now,
            )
            if order.corrective:
                self.corrective_starts += 1
            else:
                self.preventive_starts += 1

    def queue_seconds(self, now: datetime) -> float:
        return sum(
            (
                (request.started_at or now) - request.queued_at
            ).total_seconds()
            for request in self._requests.values()
        )

    def _active_count(self) -> int:
        return sum(
            request.status == OrderStatus.IN_PROGRESS
            for request in self._requests.values()
        )

    def _can_admit(
        self,
        request: WorkshopRequest,
        trucks: dict[str, object],
        available: int,
    ) -> bool:
        truck = trucks[request.truck_id]
        if truck.asset_status == AssetStatus.FAILED:
            return True
        return available - 1 >= self.config.minimum_available


class CoordinatedFleet:
    def __init__(
        self,
        simulation_config: SimulationConfig | None = None,
        maintenance_config: MaintenanceConfig | None = None,
        workshop_config: WorkshopConfig | None = None,
        policy: FleetPolicy = FleetPolicy.P3_COORDINATED,
    ) -> None:
        if not isinstance(policy, FleetPolicy):
            raise TypeError("Politica de flota no valida.")
        self.policy = policy
        self.maintenance_config = maintenance_config or MaintenanceConfig()
        self.fleet = FederatedFleet(
            simulation_config,
            maintenance_config=self.maintenance_config,
        )
        self.workshop = SharedWorkshop(
            workshop_config or WorkshopConfig(), len(self.fleet.truck_ids)
        )
        self.ledger = TransferLedger()
        self._published: dict[str, PublishedOperationalState] = {}
        self._previous_cycles = {truck_id: 0 for truck_id in self.fleet.truck_ids}
        self._tonnes_delivered = 0.0
        self._max_active_bays = 0
        self._minimum_available_observed = len(self.fleet.truck_ids)

    @property
    def time(self) -> datetime:
        return self.fleet.time

    @property
    def trucks(self) -> dict[str, object]:
        return {
            truck_id: truck
            for node in self.fleet.nodes.values()
            for truck_id, truck in node.trucks.items()
        }

    @property
    def published_states(self) -> tuple[PublishedOperationalState, ...]:
        return tuple(self._published.values())

    def set_connected(self, node_id: str, connected: bool) -> None:
        self.fleet.nodes[node_id].connected = connected

    def request_maintenance(
        self, truck_id: str, reason: str = "Solicitud manual.", urgency: float = 0
    ) -> WorkshopRequest:
        return self.workshop.enqueue(
            self.trucks[truck_id], reason, urgency, self.time
        )

    def advance(self, steps: int = 1) -> None:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps debe ser un entero positivo.")
        for _ in range(steps):
            self.workshop.synchronize(self.trucks, self.time)
            if self.policy == FleetPolicy.P3_COORDINATED:
                self._publish_connected()
            self._enqueue_policy_requests()
            self.workshop.admit(self.trucks, self.time)
            self._observe_constraints()
            for node in self.fleet.nodes.values():
                node.advance()
            self.fleet.time += timedelta(seconds=self.fleet.config.step_seconds)
            self._record_production()
            self.workshop.synchronize(self.trucks, self.time)
            self._observe_constraints()

    @property
    def metrics(self) -> dict:
        trucks = tuple(self.trucks.values())
        scheduled = sum(truck.metrics["scheduled_seconds"] for truck in trucks)
        available = sum(truck.metrics["available_seconds"] for truck in trucks)
        maintenance_seconds = sum(
            truck.metrics["maintenance_seconds"] for truck in trucks
        )
        failed_seconds = sum(truck.metrics["failed_seconds"] for truck in trucks)
        queue_seconds = self.workshop.queue_seconds(self.time)
        failures = sum(truck.metrics["failures"] for truck in trucks)
        preventive_cost = (
            self.workshop.preventive_starts * self.workshop.config.preventive_cost
        )
        corrective_cost = (
            max(failures, self.workshop.corrective_starts)
            * self.workshop.config.corrective_cost
        )
        queue_cost = (
            queue_seconds / 3600 * self.workshop.config.queue_cost_per_hour
        )
        return {
            "policy": self.policy,
            "scheduled_seconds": scheduled,
            "tonnes_delivered": self._tonnes_delivered,
            "availability": available / scheduled if scheduled else None,
            "failed_seconds": failed_seconds,
            "maintenance_seconds": maintenance_seconds,
            "downtime_seconds": failed_seconds + maintenance_seconds,
            "failures": failures,
            "interventions": sum(
                truck.metrics["interventions"] for truck in trucks
            ),
            "queued_requests": len(self.workshop.queued),
            "queue_seconds": queue_seconds,
            "preventive_starts": self.workshop.preventive_starts,
            "corrective_starts": self.workshop.corrective_starts,
            "maintenance_cost": preventive_cost + corrective_cost,
            "preventive_cost": preventive_cost,
            "corrective_cost": corrective_cost,
            "queue_cost": queue_cost,
            "total_cost": preventive_cost + corrective_cost + queue_cost,
            "workshop_utilization": (
                maintenance_seconds
                / (self.workshop.config.bays * scheduled / len(trucks))
                if scheduled
                else None
            ),
            "max_active_bays": self._max_active_bays,
            "workshop_bays": self.workshop.config.bays,
            "required_minimum_available": self.workshop.config.minimum_available,
            "minimum_available_observed": self._minimum_available_observed,
            "published_state_bytes": sum(
                record.payload_bytes
                for record in self.ledger.records
                if record.message_type == MessageType.PUBLISHED_STATE
            ),
            "published_raw_records": sum(
                record.raw_records for record in self.ledger.records
            ),
        }

    def _publish_connected(self) -> None:
        for node in self.fleet.nodes.values():
            if not node.connected:
                continue
            for truck in node.trucks.values():
                if truck.twin is None:
                    continue
                state = OperationalStatePublisher.publish(truck)
                self._published[truck.truck_id] = state
                self.ledger.record_payload(
                    node.id,
                    TransferDirection.NODE_TO_COORDINATOR,
                    MessageType.PUBLISHED_STATE,
                    state,
                    asset_id=truck.truck_id,
                )

    def _enqueue_policy_requests(self) -> None:
        if self.policy == FleetPolicy.P0_NONE:
            return
        for node in self.fleet.nodes.values():
            for truck in node.trucks.values():
                coordinated = (
                    self.policy == FleetPolicy.P3_COORDINATED and node.connected
                )
                if coordinated:
                    decision = _published_decision(
                        self._published.get(truck.truck_id),
                        self.maintenance_config,
                        self.workshop.has_idle_bay and not self.workshop.queued,
                    )
                else:
                    if truck.current_order is not None or truck.twin is None:
                        continue
                    decision = _local_decision(
                        truck,
                        self.policy != FleetPolicy.P1_THRESHOLD,
                        self.maintenance_config,
                    )
                if decision:
                    reason, urgency = decision
                    if not coordinated:
                        urgency = 0.0
                    request_count = len(self.workshop.requests)
                    request = self.workshop.enqueue(
                        truck, reason, urgency, self.time
                    )
                    if coordinated and len(self.workshop.requests) > request_count:
                        self.ledger.record_payload(
                            node.id,
                            TransferDirection.COORDINATOR_TO_NODE,
                            MessageType.COMMAND,
                            {
                                "action": "queue_maintenance",
                                "truck_id": truck.truck_id,
                                "request_id": request.id,
                                "urgency": urgency,
                            },
                            asset_id=truck.truck_id,
                        )

    def _record_production(self) -> None:
        for truck_id, truck in self.trucks.items():
            cycles = truck.twin.observation.cycles
            completed = cycles - self._previous_cycles[truck_id]
            if completed < 0:
                raise ValueError("El contador de ciclos no puede retroceder.")
            self._tonnes_delivered += completed * truck.profile.capacity_tonnes
            self._previous_cycles[truck_id] = cycles

    def _observe_constraints(self) -> None:
        active = sum(
            truck.asset_status == AssetStatus.MAINTENANCE
            for truck in self.trucks.values()
        )
        available = sum(
            truck.asset_status == AssetStatus.AVAILABLE
            for truck in self.trucks.values()
        )
        self._max_active_bays = max(self._max_active_bays, active)
        self._minimum_available_observed = min(
            self._minimum_available_observed, available
        )
        if active > self.workshop.config.bays:
            raise RuntimeError("Se excedio la capacidad del taller.")


def _local_decision(truck, predictive: bool, config: MaintenanceConfig):
    if truck.asset_status == AssetStatus.FAILED:
        return "Reparacion correctiva tras falla observada.", 1_000.0
    if truck.twin.diagnosis.status == EngineStatus.ALERT:
        return "Intervencion por umbral persistente del motor.", 500.0
    prediction = truck.prediction
    if (
        predictive
        and prediction
        and prediction.status == ForecastStatus.ESTIMABLE
        and prediction.hours_to_threshold is not None
        and prediction.hours_to_threshold <= config.predictive_margin_hours
    ):
        urgency = config.predictive_margin_hours - prediction.hours_to_threshold
        return "Intervencion por riesgo predictivo local.", 100.0 + urgency
    return None


def _published_decision(
    state: PublishedOperationalState | None,
    config: MaintenanceConfig,
    allow_opportunistic: bool = False,
):
    if state is None or state.open_order:
        return None
    if state.asset_status == AssetStatus.FAILED:
        return "Reparacion correctiva solicitada por estado publicado.", 1_000.0
    if state.engine_status == EngineStatus.ALERT:
        return "Intervencion por alerta publicada del motor.", 500.0
    if (
        state.forecast_status == ForecastStatus.ESTIMABLE
        and state.hours_to_threshold is not None
        and state.hours_to_threshold <= config.predictive_margin_hours
    ):
        urgency = config.predictive_margin_hours - state.hours_to_threshold
        return "Intervencion coordinada por riesgo publicado.", 100.0 + urgency
    if (
        allow_opportunistic
        and state.forecast_status == ForecastStatus.ESTIMABLE
        and state.hours_to_threshold is not None
        and state.hours_to_threshold <= 2 * config.predictive_margin_hours
    ):
        urgency = 2 * config.predictive_margin_hours - state.hours_to_threshold
        return "Intervencion adelantada por capacidad ociosa.", 10.0 + urgency
    return None


def run_workshop_comparison(
    output: Path,
    simulation_config: SimulationConfig,
    steps: int,
    workshop_config: WorkshopConfig | None = None,
    maintenance_config: MaintenanceConfig | None = None,
) -> list[dict]:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps debe ser un entero positivo.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for policy in FleetPolicy:
            fleet = CoordinatedFleet(
                simulation_config,
                maintenance_config,
                workshop_config,
                policy,
            )
            fleet.advance(steps)
            directory = output / policy.value
            directory.mkdir()
            metrics = fleet.metrics
            (directory / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _write_requests(directory / "requests.csv", fleet.workshop.requests)
            fleet.ledger.write_csv(directory / "transfer_ledger.csv")
            results.append(metrics)
        _write_comparison(output / "comparison.csv", results)
        return results
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _write_requests(path: Path, requests: tuple[WorkshopRequest, ...]) -> None:
    fields = tuple(WorkshopRequest.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(request) for request in requests)


def _write_comparison(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coordinacion de mantenimiento de la flota MineTwin."
    )
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=list(Scenario), default=Scenario.ENGINE_DEGRADATION)
    parser.add_argument("--bays", type=int, default=2)
    parser.add_argument("--minimum-available", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_workshop_comparison(
            args.output,
            SimulationConfig(seed=args.seed, scenario=Scenario(args.scenario)),
            args.steps,
            WorkshopConfig(
                bays=args.bays,
                minimum_available=args.minimum_available,
            ),
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
