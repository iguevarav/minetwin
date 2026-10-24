from dataclasses import replace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.training import fit_centralized_candidate


def candidate_probabilities(
    model: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: np.ndarray,
    mlp: TrainingConfig,
    seed: int,
) -> np.ndarray:
    if model == "mlp":
        probabilities, _ = fit_centralized_candidate(
            train_features,
            train_labels,
            evaluation_features,
            replace(mlp, seed=seed),
        )
        return probabilities
    if model == "logistic_regression":
        probabilities, _ = fit_centralized_candidate(
            train_features,
            train_labels,
            evaluation_features,
            TrainingConfig(hidden_sizes=(), epochs=30, seed=seed),
        )
        return probabilities
    if model == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        )
        estimator.fit(train_features, train_labels)
        return _aligned_probabilities(
            estimator.predict_proba(evaluation_features),
            estimator.classes_,
        )
    raise ValueError(f"Modelo no disponible: {model}.")


def _aligned_probabilities(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(values), 5), dtype=np.float64)
    probabilities[:, np.asarray(classes, dtype=int)] = values
    return probabilities
