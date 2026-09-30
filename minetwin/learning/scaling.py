from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SufficientStatistics:
    count: int
    total: np.ndarray
    squared_total: np.ndarray

    @classmethod
    def from_features(cls, features: np.ndarray) -> "SufficientStatistics":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not len(values):
            raise ValueError("Se requiere una matriz no vacia de variables.")
        return cls(len(values), values.sum(axis=0), np.square(values).sum(axis=0))

    @classmethod
    def aggregate(
        cls, statistics: tuple["SufficientStatistics", ...]
    ) -> "SufficientStatistics":
        if not statistics:
            raise ValueError("No existen estadisticas para agregar.")
        return cls(
            sum(item.count for item in statistics),
            sum((item.total for item in statistics), np.zeros_like(statistics[0].total)),
            sum(
                (item.squared_total for item in statistics),
                np.zeros_like(statistics[0].squared_total),
            ),
        )


@dataclass(frozen=True)
class FederatedStandardScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, nodes: np.ndarray) -> "FederatedStandardScaler":
        node_statistics = tuple(
            SufficientStatistics.from_features(features[nodes == node])
            for node in np.unique(nodes)
        )
        combined = SufficientStatistics.aggregate(node_statistics)
        mean = combined.total / combined.count
        variance = np.maximum(combined.squared_total / combined.count - mean**2, 0)
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        if not np.isfinite(values).all():
            raise ValueError("El escalado produjo valores no finitos.")
        return values.astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> "FederatedStandardScaler":
        with np.load(path, allow_pickle=False) as source:
            return cls(source["mean"], source["scale"])
