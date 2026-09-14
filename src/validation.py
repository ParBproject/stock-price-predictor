"""Time-series validation helpers for horizon-shifted forecasting samples."""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def make_forecast_time_series_split(
    n_splits: int = 5,
    forecast_horizon: int = 1,
) -> TimeSeriesSplit:
    """Create chronological CV folds with a horizon-sized embargo.

    For samples where features at row ``t`` predict a target at
    ``t + forecast_horizon``, the same number of rows must be excluded between
    each training fold and validation fold. This prevents the final training
    labels from containing outcomes that occur inside the validation feature
    period.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise TypeError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if isinstance(forecast_horizon, bool) or not isinstance(
        forecast_horizon, (int, np.integer)
    ):
        raise TypeError("forecast_horizon must be an integer")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1")

    return TimeSeriesSplit(n_splits=int(n_splits), gap=int(forecast_horizon))
