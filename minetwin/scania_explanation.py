import json
from pathlib import Path

import numpy as np

from minetwin.data.scania import ScaniaReplayView
from minetwin.langflow_client import LangflowError, ScaniaExplanationObservation
from minetwin.learning import FeatureConfig, WindowVectorizer
from minetwin.learning.data import CachedSplit
from minetwin.learning.provenance import (
    file_sha256,
    value_sha256,
    verify_training_provenance,
)
from minetwin.learning.scaling import FederatedStandardScaler
from minetwin.paths import RESULTS_ROOT
from minetwin.publication import (
    PublishedCondition,
    PublishedTwinState,
    RiskAssessment,
)


class ScaniaExplanationSource:
    def __init__(
        self,
        cache: Path = RESULTS_ROOT / "phase2_cache",
        models: Path = RESULTS_ROOT / "final_learning_verified" / "seed_100",
        publication: Path = RESULTS_ROOT / "final_federation_verified",
    ) -> None:
        cache = Path(cache)
        models = Path(models)
        publication = Path(publication)
        self.summary = json.loads(
            (publication / "summary.json").read_text(encoding="utf-8")
        )
        provenance = self.summary.get("provenance", {})
        if not provenance.get("verified"):
            raise LangflowError(
                "La publicación SCANIA no tiene procedencia verificada."
            )
        self.split = self.summary["split"]
        self.regime = self.summary["regime"]
        report = json.loads((models / "metrics.json").read_text(encoding="utf-8"))
        training = verify_training_provenance(cache, models, report)
        split_hash = file_sha256(cache / f"{self.split}.npz")
        data_version = value_sha256(
            {
                "training_id": training["run_id"],
                "split": self.split,
                "sha256": split_hash,
            }
        )
        if (
            provenance["training_id"] != training["run_id"]
            or provenance["split_sha256"] != split_hash
            or provenance["data_version"] != data_version
        ):
            raise LangflowError(
                "La publicación no corresponde a la caché y los modelos."
            )
        self.provenance = provenance
        self.cached = CachedSplit.load(cache / f"{self.split}.npz")
        self._positions = {
            str(asset_id): index
            for index, asset_id in enumerate(self.cached.vehicle_ids)
        }
        if len(self._positions) != len(self.cached.vehicle_ids):
            raise ValueError("La partición publicada requiere un ejemplo por vehículo.")
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        self.vectorizer = WindowVectorizer(FeatureConfig(**metadata["feature_config"]))
        self.scaler = FederatedStandardScaler.load(models / "scaler.npz")
        if file_sha256(publication / "published_states.jsonl") != self.summary.get(
            "published_states_sha256"
        ):
            raise LangflowError("Los estados publicados no coinciden con su resumen.")
        self.states = self._load_states(publication / "published_states.jsonl")
        if len(self.states) != self.summary["coordinator_states"]:
            raise ValueError("El número de estados no coincide con el resumen.")

    def observation(
        self,
        view: ScaniaReplayView,
        regime: str,
        connected: bool = True,
    ) -> ScaniaExplanationObservation:
        if view.split.value != self.split or regime != self.regime:
            raise LangflowError("Selecciona el split y el modelo de la publicación.")
        state = self.states.get(view.vehicle_id)
        missing = tuple(
            name
            for name, value in zip(
                view.current.feature_names, view.current.values, strict=True
            )
            if value is None
        )
        latest = view.position == view.readout_count - 1
        stale = not latest or not self._matches(view, state)
        features = self._relevant_features(view) if not stale else ()
        return ScaniaExplanationObservation(
            state=state,
            vehicle_id=view.vehicle_id,
            node_id=view.node_id,
            split=view.split.value,
            time_step=view.current.time_step,
            readout_index=view.position + 1,
            readout_count=view.readout_count,
            relevant_features=features,
            unavailable_readings=missing[:5],
            unavailable_count=len(missing),
            connected=connected,
            stale=stale,
        )

    def observed_class(self, vehicle_id: str) -> int:
        return int(self.cached.labels[self._positions[vehicle_id]])

    def _matches(
        self, view: ScaniaReplayView, state: PublishedTwinState | None
    ) -> bool:
        if state is None:
            return False
        position = self._positions.get(view.vehicle_id)
        if position is None:
            return False
        expected_model = self.provenance["model_ids"].get(view.node_id)
        if (
            state.asset_id != view.vehicle_id
            or state.node_id != view.node_id
            or state.component != "Component X"
            or state.source != "SCANIA Component X"
            or state.split != self.split
            or view.window.vehicle_id != view.vehicle_id
            or view.window.node_id != view.node_id
            or view.window.split != view.split
            or view.window.end_time_step != view.current.time_step
            or state.risk.model_id != expected_model
            or state.training_id != self.provenance["training_id"]
            or state.data_version != self.provenance["data_version"]
            or self.cached.nodes[position] != view.node_id
        ):
            return False
        names = self.vectorizer.feature_names(view.window.feature_names)
        if names != self.cached.feature_names:
            return False
        values = self.vectorizer.transform(view.window)
        if not np.allclose(
            values, self.cached.features[position], rtol=1e-6, atol=1e-6
        ):
            return False
        missing = np.asarray(view.window.missing)
        quality = 1 - float(np.mean(missing))
        return state.quality is not None and abs(state.quality - quality) <= 1e-6

    def _relevant_features(self, view: ScaniaReplayView) -> tuple[str, ...]:
        names = self.cached.feature_names
        values = self.vectorizer.transform(view.window)
        standardized = self.scaler.transform(values[None, :])[0]
        selected = []
        for index in np.argsort(-np.abs(standardized), kind="stable"):
            name = names[index]
            if name.endswith("__missing_rate") or name.startswith("window__"):
                continue
            source = name.rsplit("__", 1)[0]
            if source not in selected:
                selected.append(source)
            if len(selected) == 5:
                break
        return tuple(selected)

    def _load_states(self, path: Path) -> dict[str, PublishedTwinState]:
        states = {}
        with path.open(encoding="utf-8") as source:
            for line in source:
                payload = json.loads(line)
                risk = payload.pop("risk")
                risk["condition"] = PublishedCondition(risk["condition"])
                risk["probabilities"] = tuple(risk["probabilities"])
                state = PublishedTwinState(**payload, risk=RiskAssessment(**risk))
                position = self._positions.get(state.asset_id)
                expected_model = self.provenance["model_ids"].get(state.node_id)
                if (
                    position is None
                    or state.node_id != self.cached.nodes[position]
                    or state.component != "Component X"
                    or state.source != "SCANIA Component X"
                    or state.split != self.split
                    or state.training_id != self.provenance["training_id"]
                    or state.data_version != self.provenance["data_version"]
                    or state.risk.model_id != expected_model
                ):
                    raise ValueError("Un estado publicado no coincide con su origen.")
                if state.asset_id in states:
                    raise ValueError("La publicación contiene vehículos duplicados.")
                states[state.asset_id] = state
        return states
