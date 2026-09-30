"""Naive forecast baselines used to keep model scores honest."""

import numpy as np

from src.evaluator import regression_metrics


def persistence_predictions(current_close: np.ndarray) -> np.ndarray:
    """Forecast the next close as the last observed close.

    This is the standard no-change baseline for a one-step price-level model.
    A low error against the actual next close is not evidence of skill until
    it beats this forecast.
    """
    current = np.asarray(current_close, dtype=float)
    if current.ndim != 1:
        raise ValueError("current_close must be one-dimensional")
    if len(current) == 0:
        raise ValueError("current_close must not be empty")
    if not np.isfinite(current).all():
        raise ValueError("current_close must contain only finite values")
    return current.copy()


def regression_error_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, MSE, RMSE, and MAPE without printing."""
    return regression_metrics(y_true, y_pred, verbose=False)


def mae_skill_score(model_mae: float, baseline_mae: float) -> float:
    """Return ``1 - model_mae / baseline_mae``.

    Positive values mean the model has a lower MAE than the baseline. Zero
    means the errors match. Negative values mean the model is worse.
    """
    model_mae = float(model_mae)
    baseline_mae = float(baseline_mae)
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae):
        raise ValueError("MAE values must be finite")
    if model_mae < 0 or baseline_mae < 0:
        raise ValueError("MAE values must be non-negative")
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("-inf")
    return 1.0 - model_mae / baseline_mae


def direction_labels(future_close: np.ndarray, current_close: np.ndarray) -> np.ndarray:
    """Return 1 when the future close is above the current close, else 0."""
    future = np.asarray(future_close, dtype=float)
    current = np.asarray(current_close, dtype=float)
    if future.shape != current.shape or future.ndim != 1:
        raise ValueError("future_close and current_close must be 1-D and aligned")
    if not np.isfinite(future).all() or not np.isfinite(current).all():
        raise ValueError("close arrays must be finite")
    return (future > current).astype(np.int8)


def directional_accuracy(
    future_close: np.ndarray,
    predicted_close: np.ndarray,
    current_close: np.ndarray,
) -> float:
    """Share of days where the forecast and the market agree on up versus down."""
    actual = direction_labels(future_close, current_close)
    predicted = direction_labels(predicted_close, current_close)
    return float((actual == predicted).mean())


def majority_class_label(y_train: np.ndarray) -> int:
    """Return the most common binary training label. Ties resolve to 0."""
    labels = np.asarray(y_train)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("y_train must be a non-empty 1-D array")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_train must contain only 0 and 1")
    counts = np.bincount(labels.astype(int), minlength=2)
    return int(np.argmax(counts))


def majority_class_accuracy(y_true: np.ndarray, y_train: np.ndarray) -> tuple[float, int]:
    """Accuracy of predicting the training-set majority label on ``y_true``."""
    label = majority_class_label(y_train)
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty 1-D array")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    accuracy = float((y_true.astype(int) == label).mean())
    return accuracy, label


def drop_constant_features(
    values,
    columns: list[str],
) -> tuple[list[str], list[str]]:
    """Split columns into varying features and constant features.

    A constant column, such as neutral sentiment when no headlines are
    available, cannot carry information and is reported separately so it is
    not mistaken for a model input.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for column in columns:
        series = values[column]
        if series.nunique(dropna=False) <= 1:
            dropped.append(column)
        else:
            kept.append(column)
    if not kept:
        raise ValueError("Every candidate feature is constant")
    return kept, dropped
