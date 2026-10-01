import numpy as np

from minetwin.data.scania import SCANIA_COST_MATRIX


def cost_sensitive_predictions(probabilities: np.ndarray) -> np.ndarray:
    probabilities = _probabilities(probabilities)
    return _cost_sensitive_predictions(probabilities)


def _cost_sensitive_predictions(probabilities: np.ndarray) -> np.ndarray:
    costs = np.asarray(SCANIA_COST_MATRIX, dtype=np.float64)
    return np.argmin(probabilities @ costs, axis=1)


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, decision: str = "cost"
) -> dict:
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, len(probabilities))
    if decision not in ("cost", "argmax"):
        raise ValueError("La decisión debe ser cost o argmax.")
    top_class = np.argmax(probabilities, axis=1)
    predictions = (
        _cost_sensitive_predictions(probabilities) if decision == "cost" else top_class
    )
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(
        diagonal,
        true_counts,
        out=np.zeros(5, dtype=np.float64),
        where=true_counts != 0,
    )
    precision = np.divide(
        diagonal,
        predicted_counts,
        out=np.zeros(5, dtype=np.float64),
        where=predicted_counts != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5, dtype=np.float64),
        where=precision + recall != 0,
    )
    present = true_counts > 0
    cost_matrix = np.asarray(SCANIA_COST_MATRIX, dtype=np.int64)
    confidence = np.max(probabilities, axis=1)
    calibration = _calibration_bins(confidence, top_class == labels)
    return {
        "examples": len(labels),
        "total_cost": int(np.sum(confusion * cost_matrix)),
        "mean_cost": float(np.sum(confusion * cost_matrix) / len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(recall[present])),
        "macro_f1": float(np.mean(f1[present])),
        "macro_pr_auc": float(
            np.mean(
                [
                    _average_precision(labels == label, probabilities[:, label])
                    for label in range(5)
                    if np.any(labels == label)
                ]
            )
        ),
        "brier_score": float(
            np.mean(np.sum((probabilities - np.eye(5)[labels]) ** 2, axis=1))
        ),
        "log_loss": float(
            -np.mean(
                np.log(
                    np.clip(
                        probabilities[np.arange(len(labels)), labels], 1e-15, 1
                    )
                )
            )
        ),
        "argmax_accuracy": float(np.mean(top_class == labels)),
        "argmax_ece": float(
            sum(
                bin_["examples"] / len(labels)
                * abs(bin_["mean_confidence"] - bin_["accuracy"])
                for bin_ in calibration
            )
        ),
        "calibration_bins": calibration,
        "support": true_counts.tolist(),
        "predicted_support": predicted_counts.tolist(),
        "mean_cost_by_class": [
            float(np.sum(confusion[index] * cost_matrix[index]) / true_counts[index])
            if true_counts[index]
            else None
            for index in range(5)
        ],
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def _calibration_bins(confidence: np.ndarray, correct: np.ndarray) -> list[dict]:
    indexes = np.minimum((confidence * 10).astype(int), 9)
    return [
        {
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "examples": int(np.sum(selected)),
            "mean_confidence": (
                float(np.mean(confidence[selected])) if np.any(selected) else 0.0
            ),
            "accuracy": float(np.mean(correct[selected])) if np.any(selected) else 0.0,
        }
        for index in range(10)
        for selected in (indexes == index,)
    ]


def mean_misclassification_cost(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, len(probabilities))
    predictions = _cost_sensitive_predictions(probabilities)
    costs = np.asarray(SCANIA_COST_MATRIX, dtype=np.float64)
    return float(np.mean(costs[labels, predictions]))


def grouped_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    nodes: np.ndarray,
    decision: str = "cost",
) -> dict:
    return {
        "global": classification_metrics(labels, probabilities, decision),
        "nodes": {
            str(node): classification_metrics(
                labels[nodes == node], probabilities[nodes == node], decision
            )
            for node in np.unique(nodes)
        },
    }


def _probabilities(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 5 or not len(result):
        raise ValueError("Se esperan probabilidades para cinco clases.")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Las probabilidades deben ser finitas y no negativas.")
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Cada prediccion requiere masa de probabilidad positiva.")
    return result / totals


def _labels(values: np.ndarray, expected: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != expected:
        raise ValueError("Etiquetas y probabilidades no coinciden.")
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError("Las etiquetas deben estar entre cero y cuatro.")
    return labels


def _average_precision(relevant: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(relevant))
    if not positives:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ordered = relevant[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)
