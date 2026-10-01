import numpy as np
import pytest

from minetwin.data.scania import SCANIA_COST_MATRIX, DatasetSplit, FeatureWindow
from minetwin.learning import (
    FederatedStandardScaler,
    WindowVectorizer,
    classification_metrics,
    cost_sensitive_predictions,
    feature_groups,
    grouped_vehicle_folds,
    mean_misclassification_cost,
)


def test_federated_scaler_matches_statistics_from_combined_training_data():
    features = np.asarray(
        ((1, 10), (3, 20), (5, 30), (7, 40)), dtype=np.float32
    )
    nodes = np.asarray(("alpha", "alpha", "beta", "gamma"))
    scaler = FederatedStandardScaler.fit(features, nodes)
    assert scaler.mean == pytest.approx(features.mean(axis=0))
    assert scaler.scale == pytest.approx(features.std(axis=0))
    transformed = scaler.transform(features)
    assert transformed.mean(axis=0) == pytest.approx((0, 0), abs=1e-6)
    assert transformed.std(axis=0) == pytest.approx((1, 1), abs=1e-6)


def test_window_vectorizer_has_fixed_shape_and_preserves_missingness():
    window = FeatureWindow(
        split=DatasetSplit.TRAIN,
        node_id="alpha",
        vehicle_id="T-1",
        feature_names=("a", "b"),
        time_steps=(2, 5),
        values=((1, 4), (3, 8)),
        missing=((False, True), (False, False)),
    )
    vectorizer = WindowVectorizer()
    values = vectorizer.transform(window)
    assert len(values) == len(vectorizer.feature_names(window.feature_names)) == 12
    assert values[8:10] == pytest.approx((0, 0.5))
    assert np.isfinite(values).all()


def test_feature_groups_preserve_source_variables_and_temporal_statistics():
    groups = feature_groups(
        ("a__last", "b__last", "a__mean", "b__mean", "window__elapsed")
    )
    assert groups["source_variable"] == {
        "a": (0, 2),
        "b": (1, 3),
        "window": (4,),
    }
    assert groups["temporal_statistic"] == {
        "last": (0, 1),
        "mean": (2, 3),
        "elapsed": (4,),
    }


def test_cost_sensitive_decision_and_metrics_use_official_cost_matrix():
    probabilities = np.eye(5, dtype=np.float64)
    predictions = cost_sensitive_predictions(probabilities)
    assert predictions.tolist() == list(range(5))
    metrics = classification_metrics(np.arange(5), probabilities)
    assert metrics["total_cost"] == 0
    assert metrics["macro_f1"] == 1
    assert metrics["confusion_matrix"] == np.eye(5, dtype=int).tolist()
    assert mean_misclassification_cost(np.arange(5), probabilities) == 0
    assert SCANIA_COST_MATRIX[4][0] == 500


def test_calibration_and_class_diagnostics_keep_argmax_separate_from_cost_decision():
    labels = np.arange(5)
    perfect = classification_metrics(labels, np.eye(5))
    assert perfect["brier_score"] == 0
    assert perfect["log_loss"] == 0
    assert perfect["argmax_ece"] == 0
    assert perfect["support"] == [1] * 5
    assert perfect["mean_cost_by_class"] == [0] * 5

    probabilities = np.broadcast_to(
        np.asarray((0.8, 0.05, 0.05, 0.05, 0.05)), (5, 5)
    )
    metrics = classification_metrics(labels, probabilities)
    argmax = classification_metrics(labels, probabilities, decision="argmax")
    assert metrics["argmax_accuracy"] == pytest.approx(0.2)
    assert metrics["argmax_ece"] == pytest.approx(0.6)
    assert metrics["predicted_support"] != [5, 0, 0, 0, 0]
    assert argmax["predicted_support"] == [5, 0, 0, 0, 0]
    assert argmax["total_cost"] != metrics["total_cost"]


def test_cross_validation_keeps_every_vehicle_in_one_fold():
    vehicles = np.asarray(("A", "A", "B", "C", "C", "D", "E", "F"))
    labels = np.asarray((0, 1, 0, 2, 3, 4, 1, 2))
    nodes = np.asarray(
        ("alpha", "alpha", "alpha", "beta", "beta", "gamma", "alpha", "beta")
    )
    folds = grouped_vehicle_folds(vehicles, labels, nodes, folds=3, seed=42)
    assert set(folds) <= {0, 1, 2}
    for vehicle in np.unique(vehicles):
        assert len(np.unique(folds[vehicles == vehicle])) == 1
