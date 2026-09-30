from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class FeatureConfig:
    window_size: int = 12
    stride: int = 1
    signed_log: bool = True

    def __post_init__(self) -> None:
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError("window_size debe ser un entero positivo.")
        if type(self.stride) is not int or self.stride < 1:
            raise ValueError("stride debe ser un entero positivo.")


@dataclass(frozen=True)
class TrainingConfig:
    hidden_sizes: tuple[int, ...] = (128, 64)
    epochs: int = 20
    federated_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    proximal_mu: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        integers = {
            "epochs": self.epochs,
            "federated_rounds": self.federated_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
        }
        if any(type(value) is not int or value < 1 for value in integers.values()):
            raise ValueError("Epocas, rondas y batch_size deben ser positivos.")
        if not self.hidden_sizes or any(
            type(size) is not int or size < 1 for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes debe contener enteros positivos.")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("proximal_mu", self.proximal_mu),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} debe ser finito y no negativo.")
        if not self.learning_rate:
            raise ValueError("learning_rate debe ser mayor que cero.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed debe ser un entero no negativo.")
