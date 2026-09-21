"""Shared forecast scores for the chronological holdout.

Persistence, the LSTM, and the Random Forest are compared with one MAE and
one directional hit rate. The same-day helper is the leakage alignment the
README describes: features at ``t`` paired with ``Close[t]``.
"""

import numpy as np
import pandas as pd

from src.backtesting import next_day_direction_signals
from src.evaluator import regression_metrics


def persistence_forecast(current_close: np.ndarray) -> np.ndarray:
    """Forecast the next close as equal to the current close.

    For a target ``Close[t + 1]``, the prediction is ``Close[t]``.
    """
    current = np.asarray(current_close, dtype=float)
    if current.ndim != 1:
        raise ValueError("current_close must be one-dimensional")
    if len(current) == 0:
        raise ValueError("current_close must not be empty")
    if not np.isfinite(current).all():
        raise ValueError("current_close must contain only finite values")
    return current.copy()


def directional_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_close: np.ndarray,
) -> float:
    """Fraction of days whose predicted up/down call matches the realized move.

    Up means strictly above ``previous_close``. That is the same rule as
    ``next_day_direction_signals`` and the Random Forest notebook's direction
    labels (``y > current_close``). A persistence forecast equals the previous
    close, so it never calls up and hits only on down or unchanged days.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    previous_close = np.asarray(previous_close, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1 or previous_close.ndim != 1:
        raise ValueError("y_true, y_pred, and previous_close must be one-dimensional")
    if not (y_true.shape == y_pred.shape == previous_close.shape):
        raise ValueError("y_true, y_pred, and previous_close must have matching shapes")
    if len(y_true) == 0:
        raise ValueError("y_true, y_pred, and previous_close must not be empty")
    if (
        not np.isfinite(y_true).all()
        or not np.isfinite(y_pred).all()
        or not np.isfinite(previous_close).all()
    ):
        raise ValueError("forecast inputs must contain only finite values")

    predicted_up = next_day_direction_signals(y_pred, previous_close)
    actual_up = next_day_direction_signals(y_true, previous_close)
    return float(np.mean(predicted_up == actual_up))


def score_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_close: np.ndarray,
    label: str = "Forecast",
) -> dict:
    """MAE and directional hit rate for one forecast on one aligned sample.

    MAE is ``regression_metrics`` (sklearn ``mean_absolute_error``). Hit rate
    uses :func:`directional_hit_rate`.
    """
    metrics = regression_metrics(y_true, y_pred, label=label)
    hit_rate = directional_hit_rate(y_true, y_pred, previous_close)
    print(f"  Directional hit rate: {hit_rate:.4f}")
    return {"MAE": float(metrics["MAE"]), "directional_hit_rate": hit_rate}


def same_day_feature_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    dates: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pair features at each date with that same date's close.

    This is the leakage case: technical features on row ``t`` are computed
    from ``Close[t]``, so the target is already visible. Callers must label
    scores from this alignment as leaky. It is not a next-day forecast.
    """
    if "Close" not in df.columns:
        raise KeyError("Missing required columns: ['Close']")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    dates = pd.Index(dates)
    if dates.empty:
        raise ValueError("dates must not be empty")
    if dates.has_duplicates:
        raise ValueError("dates must be unique")
    missing_dates = dates.difference(df.index)
    if len(missing_dates):
        raise KeyError("dates are not all present in the frame index")

    features = df.loc[dates, list(feature_cols)].copy()
    target = df.loc[dates, "Close"].astype(float).copy()
    target.name = "Close_same_day_leaky"
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("same-day close targets must be finite")
    return features, target
