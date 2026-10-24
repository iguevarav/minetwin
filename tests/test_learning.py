import numpy as np
import pytest

from minetwin.data.scania import SCANIA_COST_MATRIX, DatasetSplit, FeatureWindow
from minetwin.learning import (
    FeatureConfig,
    FederatedStandardScaler,
    WindowVectorizer,
    class_average_precision,
    classification_metrics,
    cost_sensitive_predictions,
    expected_decision_costs,
    feature_groups,
    grouped_vehicle_folds,
    mean_misclassification_cost,
)
from minetwin.learning.calibration import fit_temperature_calibrator
from minetwin.learning.demonstration import _stratified_cases
from minetwin.learning.priority import priority_baseline, priority_metrics
from minetwin.learning.selection import latest_vehicle_mask


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


def test_usage_features_use_causal_increments_and_family_shares():
    window = FeatureWindow(
        split=DatasetSplit.TRAIN,
        node_id="alpha",
        vehicle_id="T-1",
        feature_names=("167_0", "167_1", "sensor"),
        time_steps=(2, 5),
        values=((2, 3, 4), (4, 7, 3)),
        missing=((False, False, False), (False, False, False)),
    )
    vectorizer = WindowVectorizer(FeatureConfig(usage_features=True))
    values = vectorizer.transform(window)
    names = vectorizer.feature_names(window.feature_names)
    assert len(values) == len(names) == 25
    assert values[names.index("167_0__increment_share")] == pytest.approx(1 / 3)
    assert values[names.index("167_1__increment_share")] == pytest.approx(2 / 3)
    assert values[names.index("family_167__increment_total")] == pytest.approx(
        np.log1p(6)
    )
    assert values[names.index("family_167__decrease")] == 0


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


def test_expected_costs_use_true_class_rows_and_published_class_columns():
    probabilities = np.asarray(
        ((0.9542948603630066, 0.012896033935248852, 0.012458737008273602,
          0.00330547709017992, 0.017044849693775177),)
    )
    costs = expected_decision_costs(probabilities)
    assert costs[0] == pytest.approx(
        (16.161443572491407, 16.981394428759813, 13.499181446619332,
         12.188003112562, 9.781821144744754)
    )
    assert cost_sensitive_predictions(probabilities).tolist() == [4]


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


def test_priority_baseline_uses_the_latest_window_of_each_vehicle():
    rows, summary = priority_baseline(
        np.asarray((0, 4, 4)),
        np.asarray(
            (
                (0.8, 0.05, 0.05, 0.05, 0.05),
                (0.1, 0.05, 0.05, 0.1, 0.7),
                (0.2, 0.05, 0.05, 0.1, 0.6),
            )
        ),
        np.asarray(("A", "A", "B")),
        np.asarray(("alpha", "alpha", "beta")),
        np.asarray((1.0, 2.0, 1.0)),
        capacities=(1,),
    )
    assert [row["vehicle_id"] for row in rows] == ["A", "B"]
    assert summary[0]["capacity"] == 1
    assert summary[0]["vehicles"] == 2
    assert summary[0]["class_4_examples"] == 2
    assert summary[0]["class_4_found"] == 1
    assert summary[0]["recall_at_capacity"] == 0.5
    assert summary[0]["precision_at_capacity"] == 1
    assert summary[0]["random_recall"] == 0.5
    assert summary[0]["random_precision"] == 1
    assert summary[0]["priority_lift"] == 1


def test_priority_metrics_and_class_four_average_precision_use_ranked_scores():
    labels = np.asarray((4, 0, 4, 0))
    probabilities = np.asarray(
        (
            (0.1, 0.1, 0.1, 0.1, 0.6),
            (0.2, 0.2, 0.2, 0.2, 0.2),
            (0.2, 0.1, 0.1, 0.1, 0.5),
            (0.5, 0.1, 0.1, 0.1, 0.2),
        )
    )
    metrics = priority_metrics(labels, probabilities, capacity=2)
    assert metrics["recall_at_capacity"] == 1
    assert metrics["precision_at_capacity"] == 1
    assert class_average_precision(labels, probabilities, 4) == 1


def test_comparison_uses_latest_time_step_for_each_held_out_vehicle():
    selected = latest_vehicle_mask(
        np.asarray(("A", "B", "A", "B")),
        np.asarray((4.0, 8.0, 9.0, 5.0)),
        np.asarray((True, True, True, True)),
    )
    assert selected.tolist() == [False, True, True, False]


def test_temperature_calibration_normalizes_probabilities_and_reduces_fit_loss():
    probabilities = np.asarray(
        (
            (0.96, 0.01, 0.01, 0.01, 0.01),
            (0.96, 0.01, 0.01, 0.01, 0.01),
            (0.96, 0.01, 0.01, 0.01, 0.01),
            (0.96, 0.01, 0.01, 0.01, 0.01),
            (0.96, 0.01, 0.01, 0.01, 0.01),
        )
    )
    labels = np.asarray((0, 1, 2, 3, 4))
    calibrator = fit_temperature_calibrator(probabilities, labels)
    calibrated = calibrator.transform(probabilities)
    assert calibrator.temperature > 1
    assert np.allclose(calibrated.sum(axis=1), 1)
    calibrated_loss = classification_metrics(labels, calibrated)["log_loss"]
    raw_loss = classification_metrics(labels, probabilities)["log_loss"]
    assert calibrated_loss < raw_loss


def test_demonstration_cases_are_stratified_and_prefer_selected_rows():
    rows = [
        {
            "node_id": "alpha",
            "label": "4",
            "selected": "False",
            "calibrated_probability_4": "0.9",
            "vehicle_id": "A",
        },
        {
            "node_id": "alpha",
            "label": "4",
            "selected": "True",
            "calibrated_probability_4": "0.1",
            "vehicle_id": "B",
        },
        {
            "node_id": "beta",
            "label": "0",
            "selected": "False",
            "calibrated_probability_4": "0.2",
            "vehicle_id": "C",
        },
    ]
    cases = _stratified_cases(rows)
    assert [case["vehicle_id"] for case in cases] == ["B", "A", "C"]
    assert cases[0]["stratum"] == "alpha: clase 4"
    assert cases[0]["priority"] == "Revisión humana prioritaria"
