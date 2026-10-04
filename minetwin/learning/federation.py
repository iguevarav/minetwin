import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from minetwin.learning.data import CachedSplit
from minetwin.learning.inference import (
    RECOMMENDATIONS,
    TorchRiskPrognosticator,
    risk_assessment,
)
from minetwin.learning.provenance import (
    file_sha256,
    verified_evaluation_provenance,
)
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
        feature_names: tuple[str, ...],
        predictor: TorchRiskPrognosticator,
        training_id: str,
        data_version: str,
    ) -> None:
        self.id = node_id
        self.split = split
        self.vehicle_ids = vehicle_ids
        self._features = features
        self._quality = _observation_quality(features, feature_names)
        self.predictor = predictor
        self.training_id = training_id
        self.data_version = data_version
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
                quality=quality,
                training_id=self.training_id,
                data_version=self.data_version,
            )
            for index, (vehicle_id, predicted, probability, quality) in enumerate(
                zip(
                    self.vehicle_ids,
                    classes,
                    probabilities,
                    self._quality,
                    strict=True,
                ),
                1,
            )
        )


def _observation_quality(
    features: np.ndarray, feature_names: tuple[str, ...]
) -> tuple[float | None, ...]:
    indexes = [
        index
        for index, name in enumerate(feature_names)
        if name.endswith("__missing_rate")
    ]
    if not indexes:
        return (None,) * len(features)
    missing = features[:, indexes]
    if not np.isfinite(missing).all() or np.any((missing < 0) | (missing > 1)):
        raise ValueError("Las tasas de ausencia deben estar entre cero y uno.")
    return tuple(float(1 - rate) for rate in np.mean(missing, axis=1))


class PublishedStateCoordinator:
    def __init__(
        self,
        ledger: TransferLedger,
        expected_models: dict[str, str],
        expected_assets: dict[str, frozenset[str]],
        split: str,
        training_id: str,
        data_version: str,
    ) -> None:
        self.ledger = ledger
        self.expected_models = expected_models
        self.expected_assets = expected_assets
        self.split = split
        self.training_id = training_id
        self.data_version = data_version
        self._states: dict[str, PublishedTwinState] = {}

    @property
    def states(self) -> tuple[PublishedTwinState, ...]:
        return tuple(self._states.values())

    def receive(self, state: PublishedTwinState) -> None:
        if type(state) is not PublishedTwinState:
            raise TypeError("El coordinador solo admite estados publicados.")
        recommendation = RECOMMENDATIONS[state.risk.predicted_class]
        if (
            self.expected_models.get(state.node_id) != state.risk.model_id
            or state.asset_id not in self.expected_assets.get(state.node_id, ())
            or state.split != self.split
            or state.source != "SCANIA Component X"
            or state.component != "Component X"
            or state.risk.recommendation != recommendation
            or state.training_id != self.training_id
            or state.data_version != self.data_version
        ):
            raise ValueError("El estado no coincide con el modelo y los datos activos.")
        previous = self._states.get(state.asset_id)
        if previous is not None:
            if previous == state:
                return
            raise ValueError("Un vehículo no puede publicar estados incompatibles.")
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
        provenance: dict | None = None,
    ) -> None:
        self.split = split
        self.regime = regime
        self.provenance = provenance or {
            "verified": False,
            "training_id": "unverified",
            "data_version": "unverified",
        }
        self.ledger = TransferLedger()
        self.coordinator = PublishedStateCoordinator(
            self.ledger,
            {node: predictor.model_id for node, predictor in predictors.items()},
            {
                str(node): frozenset(
                    str(asset_id)
                    for asset_id in cached.vehicle_ids[cached.nodes == node]
                )
                for node in np.unique(cached.nodes)
            },
            split,
            self.provenance["training_id"],
            self.provenance["data_version"],
        )
        self.nodes = {
            str(node): ScaniaInferenceNode(
                str(node),
                split,
                cached.vehicle_ids[cached.nodes == node],
                cached.features[cached.nodes == node],
                cached.feature_names,
                predictors[str(node)],
                self.provenance["training_id"],
                self.provenance["data_version"],
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
        cache = Path(cache)
        models = Path(models)
        report = json.loads((models / "metrics.json").read_text(encoding="utf-8"))
        _, provenance = verified_evaluation_provenance(
            cache, models, report, split
        )
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        cached = CachedSplit.load(cache / f"{split}.npz")
        if cached.features.shape[1] != report["features"]:
            raise ValueError("El split publicado no coincide con el modelo entrenado.")
        with np.load(cache / "train.npz", allow_pickle=False) as source:
            training_features = tuple(source["feature_names"].tolist())
        if cached.feature_names != training_features:
            raise ValueError(
                "El split publicado no comparte el esquema de entrenamiento."
            )
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
        result = cls(
            cached,
            split,
            predictors,
            regime,
            {
                **provenance,
                "feature_config": metadata["feature_config"],
                "partition_config": metadata.get("partition_config"),
                "source_manifest_sha256": metadata.get("source_manifest_sha256"),
                "model_ids": {
                    node: predictor.model_id for node, predictor in predictors.items()
                },
            },
        )
        result._register_parameters(report)
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

    def _register_parameters(self, report: dict) -> None:
        parameter_bytes = next(iter(self.nodes.values())).predictor.parameter_bytes
        if self.regime in ("fedavg", "fedprox"):
            rounds = report["config"]["federated_rounds"]
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
                    json.dumps(
                        asdict(state),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
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
        transfer = federation.ledger.summary()
        by_type = transfer["by_type"]
        states_by_node = {node: [] for node in federation.nodes}
        for state in federation.states:
            states_by_node[state.node_id].append(state)
        result = {
            "split": split,
            "regime": regime,
            "nodes": {
                node.id: {
                    "private_records": node.private_records,
                    "published_states": len(states_by_node[node.id]),
                    "mean_observation_quality": _mean_quality(
                        states_by_node[node.id]
                    ),
                }
                for node in federation.nodes.values()
            },
            "coordinator_states": len(federation.states),
            "mean_observation_quality": _mean_quality(federation.states),
            "coordinator_raw_features": 0,
            "coordinator_raw_records": transfer["raw_records"],
            "centralized_feature_baseline_bytes": centralized_baseline,
            "published_state_bytes": published_bytes,
            "published_states_sha256": file_sha256(
                output / "published_states.jsonl"
            ),
            "transfer_ledger_sha256": file_sha256(
                output / "transfer_ledger.csv"
            ),
            "publication_reduction_ratio": (
                1 - published_bytes / centralized_baseline
                if centralized_baseline
                else None
            ),
            "communication": {
                "published_state_bytes_serialized": published_bytes,
                "training_parameter_bytes_estimated": by_type[
                    MessageType.MODEL_PARAMETERS
                ]["payload_bytes"],
                "model_distribution_bytes_estimated": by_type[
                    MessageType.MODEL_DISTRIBUTION
                ]["payload_bytes"],
            },
            "provenance": federation.provenance,
            "partition_scope": "logical_partitions_of_one_public_dataset",
            "deployment_scope": "single_process_experiment",
            "centralized_training_uses_raw_data": regime == "centralized",
            "transfer": transfer,
        }
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _mean_quality(states) -> float | None:
    values = [state.quality for state in states if state.quality is not None]
    return sum(values) / len(values) if values else None
