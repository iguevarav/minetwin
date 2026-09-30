import numpy as np

from minetwin.data.scania import SCANIA_COST_MATRIX


def cost_sensitive_predictions(probabilities: np.ndarray) -> np.ndarray:
    probabilities = _probabilities(probabilities)
    return _cost_sensitive_predictions(probabilities)


def _cost_sensitive_predictions(probabilities: np.ndarray) -> np.ndarray:
    costs = np.asarray(SCANIA_COST_MATRIX, dtype=np.float64)
    return np.argmin(probabilities @ costs, axis=1)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, len(probabilities))
    predictions = _cost_sensitive_predictions(probabilities)
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
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def mean_misclassification_cost(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, len(probabilities))
    predictions = _cost_sensitive_predictions(probabilities)
    costs = np.asarray(SCANIA_COST_MATRIX, dtype=np.float64)
    return float(np.mean(costs[labels, predictions]))


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
