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


def commission_aware_position_size(
    cash: float,
    price: float,
    commission_rate: float = 0.0,
) -> int:
    """Return the largest whole-share position affordable after commission.

    ``commission_rate`` is expressed as a fraction of notional value, e.g.
    ``0.001`` for 0.1%. The calculation uses the supplied reference price and
    does not assume future execution prices.
    """
    cash = float(cash)
    price = float(price)
    commission_rate = float(commission_rate)

    if not np.isfinite(cash) or not np.isfinite(price) or not np.isfinite(commission_rate):
        raise ValueError("cash, price, and commission_rate must be finite")
    if cash < 0:
        raise ValueError("cash must be non-negative")
    if price <= 0:
        raise ValueError("price must be positive")
    if commission_rate < 0:
        raise ValueError("commission_rate must be non-negative")

    cost_per_share = price * (1.0 + commission_rate)
    return int(np.floor(cash / cost_per_share))
