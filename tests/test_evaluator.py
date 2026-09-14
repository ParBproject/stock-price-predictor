import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.evaluator import (
    classification_metrics,
    max_drawdown_from_returns,
    plot_confusion_matrix,
    regression_metrics,
)


def test_regression_metrics_treats_series_positionally_despite_different_indexes():
    y_true = pd.Series([100.0, 200.0], index=[10, 11])
    y_pred = pd.Series([110.0, 180.0], index=[20, 21])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(15.0)
    assert metrics["MSE"] == pytest.approx(250.0)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(250.0))
    assert metrics["MAPE%"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "y_true,y_pred,match",
    [
        (np.array([1.0, 2.0]), np.array([1.0]), "matching shapes"),
        (np.array([[1.0, 2.0]]), np.array([1.0, 2.0]), "one-dimensional"),
        (np.array([]), np.array([]), "must not be empty"),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), "finite values"),
        (np.array([1.0, 2.0]), np.array([1.0, np.inf]), "finite values"),
    ],
)
def test_regression_metrics_rejects_invalid_inputs(y_true, y_pred, match):
    with pytest.raises(ValueError, match=match):
        regression_metrics(y_true, y_pred)


def test_classification_metrics_handles_all_up_holdout(capsys):
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])

    metrics = classification_metrics(y_true, y_pred, label="All Up")
    output = capsys.readouterr().out

    assert metrics == {
        "Accuracy": 1.0,
        "Precision": 1.0,
        "Recall": 1.0,
        "F1": 1.0,
    }
    assert "Down" in output
    assert "Up" in output


def test_classification_metrics_handles_all_down_holdout(capsys):
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    metrics = classification_metrics(y_true, y_pred, label="All Down")
    output = capsys.readouterr().out

    assert metrics == {
        "Accuracy": 1.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }
    assert "Down" in output
    assert "Up" in output


def test_max_drawdown_from_returns_includes_first_period_loss():
    result = max_drawdown_from_returns(np.array([-0.20, 0.10]))

    assert result == pytest.approx(-0.20)


def test_max_drawdown_from_returns_empty_series_has_no_drawdown():
    result = max_drawdown_from_returns(np.array([], dtype=float))

    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "returns,initial_equity,match",
    [
        (np.array([[0.1]]), 1.0, "one-dimensional"),
        (np.array([np.nan]), 1.0, "finite values"),
        (np.array([0.1]), 0.0, "finite and positive"),
    ],
)
def test_max_drawdown_from_returns_rejects_invalid_inputs(
    returns, initial_equity, match
):
    with pytest.raises(ValueError, match=match):
        max_drawdown_from_returns(returns, initial_equity=initial_equity)


def test_plot_confusion_matrix_keeps_two_by_two_binary_shape(monkeypatch):
    captured = {}

    def fake_heatmap(data, *args, **kwargs):
        captured["shape"] = data.shape
        captured["xticklabels"] = kwargs["xticklabels"]
        captured["yticklabels"] = kwargs["yticklabels"]
        return kwargs["ax"]

    monkeypatch.setattr("src.evaluator.sns.heatmap", fake_heatmap)
    monkeypatch.setattr("src.evaluator.plt.show", lambda: None)

    plot_confusion_matrix(
        np.array([1, 1, 1]),
        np.array([1, 1, 1]),
        save=False,
    )

    assert captured["shape"] == (2, 2)
    assert captured["xticklabels"] == ["Down", "Up"]
    assert captured["yticklabels"] == ["Down", "Up"]
