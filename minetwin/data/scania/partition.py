import hashlib
import math
import random
from dataclasses import dataclass

from minetwin.data.scania.contracts import VehicleSpecifications


@dataclass(frozen=True)
class CategoricalNodePartitioner:
    node_ids: tuple[str, ...] = ("alpha", "beta", "gamma")
    concentration: float = 0.3
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.node_ids) < 2 or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("Se requieren al menos dos nodos unicos.")
        if not math.isfinite(self.concentration) or self.concentration <= 0:
            raise ValueError("concentration debe ser positiva.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed debe ser un entero no negativo.")

    def assign(self, specifications: VehicleSpecifications) -> str:
        signature = "\x1f".join(specifications.values)
        weights = self._weights(signature)
        position = self._uniform(f"{signature}\x1e{specifications.vehicle_id}")
        cumulative = 0.0
        for node_id, weight in zip(self.node_ids, weights, strict=True):
            cumulative += weight
            if position <= cumulative:
                return node_id
        return self.node_ids[-1]

    def _weights(self, signature: str) -> tuple[float, ...]:
        generator = random.Random(self._integer(f"weights\x1e{signature}"))
        samples = tuple(
            generator.gammavariate(self.concentration, 1.0) for _ in self.node_ids
        )
        total = sum(samples)
        return tuple(sample / total for sample in samples)

    def _uniform(self, value: str) -> float:
        return self._integer(f"vehicle\x1e{value}") / (2**256 - 1)

    def _integer(self, value: str) -> int:
        payload = f"{self.seed}\x1e{value}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest(), "big")
