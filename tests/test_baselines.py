import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    directional_accuracy,
    drop_constant_features,
    mae_skill_score,
    majority_class_accuracy,
    majority_class_label,
    persistence_predictions,
    regression_error_scores,
)
from src.evaluator import regression_metrics


def test_persistence_forecast_is_the_last_close():
    current = np.array([10.0, 11.0, 12.5])
    np.testing.assert_array_equal(persistence_predictions(current), current)
    assert persistence_predictions(current) is not current


def test_regression_error_scores_match_printed_metrics():
    y_true = np.array([100.0, 110.0, 90.0])
    y_pred = np.array([101.0, 108.0, 93.0])

    quiet = regression_error_scores(y_true, y_pred)
    printed = regression_metrics(y_true, y_pred, verbose=False)

    assert quiet == printed


def test_mae_skill_score_is_positive_when_model_is_closer():
    assert mae_skill_score(1.0, 2.0) == pytest.approx(0.5)
    assert mae_skill_score(2.0, 2.0) == pytest.approx(0.0)
    assert mae_skill_score(3.0, 2.0) == pytest.approx(-0.5)


def test_directional_accuracy_counts_up_and_down_agreement():
    future = np.array([11.0, 9.0, 10.0])
    current = np.array([10.0, 10.0, 10.0])
    predicted = np.array([12.0, 8.0, 13.0])

    # Up, down, and a flat actual counted as not-up. Third forecast is a miss.
    assert directional_accuracy(future, predicted, current) == pytest.approx(2 / 3)


def test_majority_class_tie_resolves_to_down():
    assert majority_class_label(np.array([0, 1, 0, 1])) == 0


def test_majority_class_accuracy_uses_training_label_only():
    accuracy, label = majority_class_accuracy(
        np.array([1, 1, 0, 1]),
        np.array([1, 1, 1, 0]),
    )

    assert label == 1
    assert accuracy == pytest.approx(0.75)


def test_drop_constant_features_reports_neutral_sentiment():
    frame = pd.DataFrame(
        {
            "Close_Lag_1": [1.0, 2.0, 3.0],
            "Sentiment": [0.0, 0.0, 0.0],
        }
    )

    kept, dropped = drop_constant_features(frame, ["Close_Lag_1", "Sentiment"])

    assert kept == ["Close_Lag_1"]
    assert dropped == ["Sentiment"]
