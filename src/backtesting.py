"""Utilities for turning next-day forecasts into backtest trading signals."""

import numpy as np


def next_day_direction_signals(
    predicted_next_close: np.ndarray,
    current_close: np.ndarray,
) -> np.ndarray:
    """Return long/exit signals from next-day close forecasts.

    A prediction generated from features observed at date ``t`` estimates
    ``Close[t+1]``. The actionable direction at the end of date ``t`` is
    therefore determined by comparing that forecast with the currently known
    ``Close[t]``.

    Returns ``1`` when the forecast is above the current close and ``0``
    otherwise.
    """
    predicted = np.asarray(predicted_next_close, dtype=float)
    current = np.asarray(current_close, dtype=float)

    if predicted.ndim != 1 or current.ndim != 1:
        raise ValueError("predicted_next_close and current_close must be 1-D")
    if predicted.shape != current.shape:
        raise ValueError("predicted_next_close and current_close must have matching shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(current).all():
        raise ValueError("forecast and current-close values must be finite")

    return (predicted > current).astype(np.int8)
