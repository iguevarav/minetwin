import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from minetwin.learning.data import CachedSplit
from minetwin.learning.inference import TorchRiskPrognosticator, risk_assessment
from minetwin.publication import PublishedTwinState
from minetwin.transfer import (
    MessageType,
    TransferDirection,
    TransferLedger,
)


class ScaniaInferenceNode:
    def __init__(
        self,
        node_id: str,
        split: str,
        vehicle_ids: np.ndarray,
        features: np.ndarray,
        predictor: TorchRiskPrognosticator,
    ) -> None:
        self.id = node_id
        self.split = split
        self.vehicle_ids = vehicle_ids
        self._features = features
        self.predictor = predictor
        self.connected = True

    @property
    def private_records(self) -> int:
        return len(self._features)

    @property
    def private_feature_bytes(self) -> int:
        return self._features.nbytes

    def published_states(self) -> tuple[PublishedTwinState, ...]:
        classes, probabilities = self.predictor.predict(self._features)
        return tuple(
            PublishedTwinState(
                asset_id=str(vehicle_id),
                node_id=self.id,
                component="Component X",
                source="SCANIA Component X",
                split=self.split,
                sequence=index,
                risk=risk_assessment(
                    int(predicted), probability, self.predictor.model_id
                ),
            )
            for index, (vehicle_id, predicted, probability) in enumerate(
                zip(self.vehicle_ids, classes, probabilities, strict=True), 1
            )
        )


class PublishedStateCoordinator:
    def __init__(self, ledger: TransferLedger) -> None:
        self.ledger = ledger
        self._states: dict[str, PublishedTwinState] = {}

    @property
    def states(self) -> tuple[PublishedTwinState, ...]:
        return tuple(self._states.values())

    def receive(self, state: PublishedTwinState) -> None:
        self._states[state.asset_id] = state
        self.ledger.record_payload(
            state.node_id,
            TransferDirection.NODE_TO_COORDINATOR,
            MessageType.PUBLISHED_STATE,
            state,
            asset_id=state.asset_id,
            model_id=state.risk.model_id,
        )

    def state(self, asset_id: str) -> PublishedTwinState:
        return self._states[asset_id]


class ScaniaFederatedInference:
    def __init__(
        self,
        cached: CachedSplit,
        split: str,
        predictors: dict[str, TorchRiskPrognosticator],
        regime: str,
    ) -> None:
        self.split = split
        self.regime = regime
        self.ledger = TransferLedger()
        self.coordinator = PublishedStateCoordinator(self.ledger)
        self.nodes = {
            str(node): ScaniaInferenceNode(
                str(node),
                split,
                cached.vehicle_ids[cached.nodes == node],
                cached.features[cached.nodes == node],
                predictors[str(node)],
            )
            for node in np.unique(cached.nodes)
        }

    @classmethod
    def from_artifacts(
        cls,
        cache: Path,
        models: Path,
        split: str = "validation",
        regime: str = "fedprox",
    ) -> "ScaniaFederatedInference":
        if split not in ("validation", "test"):
            raise ValueError("La inferencia solo admite validation o test.")
        if regime not in ("local", "centralized", "fedavg", "fedprox"):
            raise ValueError("Regimen de aprendizaje desconocido.")
        cached = CachedSplit.load(Path(cache) / f"{split}.npz")
        models = Path(models)
        node_ids = tuple(str(node) for node in np.unique(cached.nodes))
        if regime == "local":
            predictors = {
                node: TorchRiskPrognosticator.load(
                    models / f"local_{node}.pt", models / "scaler.npz"
                )
                for node in node_ids
            }
        else:
            predictor = TorchRiskPrognosticator.load(
                models / f"{regime}.pt", models / "scaler.npz"
            )
            predictors = {node: predictor for node in node_ids}
        result = cls(cached, split, predictors, regime)
        result._register_parameters(models)
        return result

    @property
    def states(self) -> tuple[PublishedTwinState, ...]:
        return self.coordinator.states

    def set_connected(self, node_id: str, connected: bool) -> None:
        self.nodes[node_id].connected = connected

    def publish(self, node_id: str | None = None) -> None:
        selected = self.nodes.values() if node_id is None else (self.nodes[node_id],)
        for node in selected:
            if not node.connected:
                continue
            for state in node.published_states():
                self.coordinator.receive(state)

    def state(self, asset_id: str) -> PublishedTwinState:
        return self.coordinator.state(asset_id)

    def _register_parameters(self, models: Path) -> None:
        parameter_bytes = next(iter(self.nodes.values())).predictor.parameter_bytes
        if self.regime in ("fedavg", "fedprox"):
            metrics = json.loads((models / "metrics.json").read_text(encoding="utf-8"))
            rounds = metrics["config"]["federated_rounds"]
            for round_id in range(1, rounds + 1):
                for node in self.nodes.values():
                    self.ledger.record_bytes(
                        node.id,
                        TransferDirection.COORDINATOR_TO_NODE,
                        MessageType.MODEL_DISTRIBUTION,
                        parameter_bytes,
                        round_id=round_id,
                        model_id=node.predictor.model_id,
                    )
                    self.ledger.record_bytes(
                        node.id,
                        TransferDirection.NODE_TO_COORDINATOR,
                        MessageType.MODEL_PARAMETERS,
                        parameter_bytes,
                        round_id=round_id,
                        model_id=node.predictor.model_id,
                    )
        elif self.regime == "centralized":
            for node in self.nodes.values():
                self.ledger.record_bytes(
                    node.id,
                    TransferDirection.COORDINATOR_TO_NODE,
                    MessageType.MODEL_DISTRIBUTION,
                    parameter_bytes,
                    model_id=node.predictor.model_id,
                )


def export_federated_inference(
    cache: Path,
    models: Path,
    output: Path,
    split: str = "validation",
    regime: str = "fedprox",
) -> dict:
    federation = ScaniaFederatedInference.from_artifacts(
        cache, models, split, regime
    )
    federation.publish()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with (output / "published_states.jsonl").open(
            "w", encoding="utf-8"
        ) as target:
            for state in federation.states:
                target.write(
                    json.dumps(asdict(state), ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
        federation.ledger.write_csv(output / "transfer_ledger.csv")
        centralized_baseline = sum(
            node.private_feature_bytes for node in federation.nodes.values()
        )
        published_bytes = sum(
            record.payload_bytes
            for record in federation.ledger.records
            if record.message_type == MessageType.PUBLISHED_STATE
        )
        result = {
            "split": split,
            "regime": regime,
            "nodes": {
                node.id: {
                    "private_records": node.private_records,
                    "published_states": sum(
                        state.node_id == node.id for state in federation.states
                    ),
                }
                for node in federation.nodes.values()
            },
            "coordinator_states": len(federation.states),
            "coordinator_raw_features": 0,
            "centralized_feature_baseline_bytes": centralized_baseline,
            "published_state_bytes": published_bytes,
            "publication_reduction_ratio": (
                1 - published_bytes / centralized_baseline
                if centralized_baseline
                else None
            ),
            "transfer": federation.ledger.summary(),
        }
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise
