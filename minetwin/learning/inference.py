import hashlib
from pathlib import Path

import numpy as np

from minetwin.learning.metrics import cost_sensitive_predictions
from minetwin.learning.model import build_risk_mlp, require_torch
from minetwin.learning.scaling import FederatedStandardScaler
from minetwin.publication import PublishedCondition, RiskAssessment

RECOMMENDATIONS = (
    "Sin prioridad adicional según esta estimación histórica.",
    "Aumentar el seguimiento de Component X.",
    "Programar una inspección de Component X.",
    "Priorizar la inspección de Component X.",
    "Priorizar una revisión de Component X.",
)


class TorchRiskPrognosticator:
    def __init__(self, model, scaler, model_id: str, parameter_bytes: int) -> None:
        self.model = model
        self.scaler = scaler
        self.model_id = model_id
        self.parameter_bytes = parameter_bytes

    @classmethod
    def load(
        cls, model_path: Path, scaler_path: Path
    ) -> "TorchRiskPrognosticator":
        torch = require_torch()
        model_path = Path(model_path)
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        model = build_risk_mlp(
            payload["input_size"],
            tuple(payload["hidden_sizes"]),
            payload["classes"],
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        return cls(
            model,
            FederatedStandardScaler.load(scaler_path),
            f"{model_path.stem}:{digest}",
            parameter_bytes,
        )

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        torch = require_torch()
        values = self.scaler.transform(features)
        tensor = torch.from_numpy(values)
        with torch.no_grad():
            return torch.softmax(self.model(tensor), dim=1).numpy()

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_probabilities(features)
        return cost_sensitive_predictions(probabilities), probabilities

    def predict_classes(self, features: np.ndarray) -> np.ndarray:
        return cost_sensitive_predictions(self.predict_probabilities(features))

    def assess(self, features: np.ndarray) -> tuple[RiskAssessment, ...]:
        classes, probabilities = self.predict(features)
        return tuple(
            risk_assessment(int(predicted), probability, self.model_id)
            for predicted, probability in zip(
                classes, probabilities, strict=True
            )
        )


def risk_assessment(
    predicted_class: int,
    probabilities: np.ndarray,
    model_id: str,
) -> RiskAssessment:
    condition = (
        PublishedCondition.NORMAL
        if predicted_class == 0
        else PublishedCondition.WATCH
        if predicted_class < 3
        else PublishedCondition.ALERT
    )
    normalized = probabilities / probabilities.sum()
    return RiskAssessment(
        predicted_class=predicted_class,
        probabilities=tuple(float(value) for value in normalized),
        condition=condition,
        recommendation=RECOMMENDATIONS[predicted_class],
        model_id=model_id,
    )
