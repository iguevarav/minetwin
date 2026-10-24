import json
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from minetwin.data.scania import (
    CategoricalNodePartitioner,
    DatasetSplit,
    FeatureWindow,
    ScaniaDataset,
    ScaniaManifest,
    WindowConfig,
    build_windows,
)
from minetwin.learning.contracts import FeatureConfig
from minetwin.learning.provenance import file_sha256


@dataclass(frozen=True)
class LearningExample:
    vehicle_id: str
    node_id: str
    label: int
    time_step: float
    values: np.ndarray


@dataclass(frozen=True)
class CachedSplit:
    features: np.ndarray
    labels: np.ndarray
    nodes: np.ndarray
    vehicle_ids: np.ndarray
    feature_names: tuple[str, ...]
    time_steps: np.ndarray | None = None

    @classmethod
    def load(cls, path: Path) -> "CachedSplit":
        with np.load(path, allow_pickle=False) as source:
            return cls(
                features=source["features"],
                labels=source["labels"],
                nodes=source["nodes"],
                vehicle_ids=source["vehicle_ids"],
                feature_names=tuple(source["feature_names"].tolist()),
                time_steps=(
                    source["time_steps"]
                    if "time_steps" in source.files
                    else np.full(len(source["labels"]), np.nan)
                ),
            )


class WindowVectorizer:
    STATISTICS = ("last", "mean", "std", "trend", "missing_rate")

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    @property
    def schema_version(self) -> int:
        return 2 if self.config.usage_features else 1

    def feature_names(self, source_names: tuple[str, ...]) -> tuple[str, ...]:
        names = (
            *(
                f"{name}__{statistic}"
                for statistic in self.STATISTICS
                for name in source_names
            ),
            "window__observations",
            "window__elapsed",
        )
        if not self.config.usage_features:
            return names
        families = _indexed_families(source_names)
        return (
            *names,
            *(f"{name}__increment" for name in source_names),
            *(
                f"{source_names[index]}__increment_share"
                for _, indexes in families
                for index in indexes
            ),
            *(f"family_{name}__increment_total" for name, _ in families),
            *(f"family_{name}__zero_increment" for name, _ in families),
            *(f"family_{name}__decrease" for name, _ in families),
        )

    def transform(self, window: FeatureWindow) -> np.ndarray:
        values = np.asarray(window.values, dtype=np.float64)
        missing = np.asarray(window.missing, dtype=np.float64)
        elapsed = window.time_steps[-1] - window.time_steps[0]
        duration = elapsed if elapsed > 0 else 1.0
        statistics = (
            values[-1],
            values.mean(axis=0),
            values.std(axis=0),
            (values[-1] - values[0]) / duration,
        )
        if self.config.signed_log:
            statistics = tuple(_signed_log(item) for item in statistics)
        result = np.concatenate(
            (
                *statistics,
                missing.mean(axis=0),
                np.asarray((np.log1p(len(values)), np.log1p(elapsed))),
            )
        )
        if self.config.usage_features:
            result = np.concatenate(
                (result, self._usage_features(values, window.feature_names))
            )
        return result.astype(np.float32)

    def _usage_features(
        self, values: np.ndarray, source_names: tuple[str, ...]
    ) -> np.ndarray:
        increments = values[-1] - values[0]
        encoded = _signed_log(increments) if self.config.signed_log else increments
        shares = []
        totals = []
        zeros = []
        decreases = []
        for _, indexes in _indexed_families(source_names):
            family = increments[np.asarray(indexes)]
            positive = np.maximum(family, 0.0)
            total = float(positive.sum())
            shares.append(positive / total if total else np.zeros(len(family)))
            totals.append(np.log1p(total) if self.config.signed_log else total)
            zeros.append(float(total == 0.0))
            decreases.append(float(np.any(family < 0.0)))
        return np.concatenate(
            (
                encoded,
                *shares,
                np.asarray(totals),
                np.asarray(zeros),
                np.asarray(decreases),
            )
        )


def prepare_learning_cache(
    dataset_root: Path,
    phase_one_output: Path,
    output: Path,
    config: FeatureConfig | None = None,
) -> dict:
    effective = config or FeatureConfig()
    dataset = ScaniaDataset(dataset_root)
    phase_one_output = Path(phase_one_output)
    manifest = ScaniaManifest.read(phase_one_output / "dataset_manifest.json")
    manifest.verify(dataset.root)
    partitioner = _partitioner(phase_one_output / "experiment.json")
    vectorizer = WindowVectorizer(effective)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    try:
        for split in DatasetSplit:
            examples = tuple(
                _examples(dataset, split, partitioner, vectorizer, effective)
            )
            if not examples:
                raise ValueError(f"No se generaron ejemplos para {split.value}.")
            feature_names = vectorizer.feature_names(dataset.schema.feature_names)
            np.savez_compressed(
                output / f"{split.value}.npz",
                features=np.stack([item.values for item in examples]),
                labels=np.asarray([item.label for item in examples], dtype=np.int8),
                nodes=np.asarray([item.node_id for item in examples]),
                vehicle_ids=np.asarray([item.vehicle_id for item in examples]),
                time_steps=np.asarray(
                    [item.time_step for item in examples], dtype=np.float64
                ),
                feature_names=np.asarray(feature_names),
            )
            labels = np.asarray([item.label for item in examples])
            summaries.append(
                {
                    "split": split,
                    "examples": len(examples),
                    "vehicles": len({item.vehicle_id for item in examples}),
                    "nodes": {
                        node: sum(item.node_id == node for item in examples)
                        for node in partitioner.node_ids
                    },
                    "classes": {
                        str(label): int(np.sum(labels == label))
                        for label in range(5)
                    },
                }
            )
        metadata = {
            "feature_config": asdict(effective),
            "feature_schema_version": vectorizer.schema_version,
            "usage_feature_families": [
                name for name, _ in _indexed_families(dataset.schema.feature_names)
            ]
            if effective.usage_features
            else [],
            "partition_config": asdict(partitioner),
            "source_manifest": str(
                (phase_one_output / "dataset_manifest.json").resolve()
            ),
            "source_manifest_sha256": file_sha256(
                phase_one_output / "dataset_manifest.json"
            ),
            "feature_count": len(vectorizer.feature_names(dataset.schema.feature_names)),
            "sampling": "last_causal_window_per_observed_class_and_vehicle",
            "splits": summaries,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metadata
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _examples(
    dataset: ScaniaDataset,
    split: DatasetSplit,
    partitioner: CategoricalNodePartitioner,
    vectorizer: WindowVectorizer,
    config: FeatureConfig,
) -> Iterator[LearningExample]:
    outcomes = dataset.load_outcomes(split)
    specifications = dataset.load_specifications(split)
    windows = build_windows(
        dataset.iter_readouts(split),
        specifications,
        partitioner,
        WindowConfig(
            size=config.window_size,
            min_size=1,
            stride=config.stride,
        ),
    )
    current_vehicle: str | None = None
    selected: dict[int, FeatureWindow] = {}
    final: FeatureWindow | None = None
    for window in windows:
        if current_vehicle is not None and window.vehicle_id != current_vehicle:
            yield from _selected_examples(split, selected, final, outcomes, vectorizer)
            selected = {}
            final = None
        current_vehicle = window.vehicle_id
        if split == DatasetSplit.TRAIN:
            label = outcomes[window.vehicle_id].class_at(window.end_time_step)
            selected[label] = window
        else:
            final = window
    if current_vehicle is not None:
        yield from _selected_examples(split, selected, final, outcomes, vectorizer)


def _selected_examples(
    split: DatasetSplit,
    selected: dict[int, FeatureWindow],
    final: FeatureWindow | None,
    outcomes,
    vectorizer: WindowVectorizer,
) -> Iterator[LearningExample]:
    candidates = (
        tuple(sorted(selected.items()))
        if split == DatasetSplit.TRAIN
        else ((outcomes[final.vehicle_id].class_label if outcomes else -1, final),)
    )
    for label, window in candidates:
        yield LearningExample(
            vehicle_id=window.vehicle_id,
            node_id=window.node_id,
            label=label,
            time_step=window.end_time_step,
            values=vectorizer.transform(window),
        )


def _partitioner(path: Path) -> CategoricalNodePartitioner:
    payload = json.loads(path.read_text(encoding="utf-8"))["partition"]
    return CategoricalNodePartitioner(
        node_ids=tuple(payload["node_ids"]),
        concentration=payload["concentration"],
        seed=payload["seed"],
    )


def _indexed_families(
    source_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    families: dict[str, list[int]] = {}
    for index, name in enumerate(source_names):
        prefix, separator, suffix = name.rpartition("_")
        if separator and prefix and suffix.isdigit():
            families.setdefault(prefix, []).append(index)
    return tuple(
        (name, tuple(indexes))
        for name, indexes in families.items()
        if len(indexes) > 1
    )


def _signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))
