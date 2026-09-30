"""Time-series validation helpers for horizon-shifted forecasting samples."""

import numpy as np
import pandas as pd
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


def iter_expanding_forecast_splits(
    n_samples: int,
    n_splits: int = 4,
    gap: int = 1,
    min_train_size: int | None = None,
) -> list[tuple[int, int, int]]:
    """Return expanding-window ``(train_end, test_start, test_end)`` bounds.

    Each pair describes supervised rows already aligned so that row ``i`` holds
    features at time ``t`` and a target at ``t + horizon``. ``gap`` rows are
    left unused between the training block ``[0, train_end)`` and the test
    block ``[test_start, test_end)``. For a one-step target, ``gap=1`` keeps
    the last training label from falling on the first test feature date.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be an integer")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise TypeError("n_splits must be an integer")
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)):
        raise TypeError("gap must be an integer")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    n_samples = int(n_samples)
    n_splits = int(n_splits)
    gap = int(gap)
    if min_train_size is None:
        min_train_size = n_samples // 2
    elif isinstance(min_train_size, bool) or not isinstance(min_train_size, (int, np.integer)):
        raise TypeError("min_train_size must be an integer")
    min_train_size = int(min_train_size)
    if min_train_size < 1:
        raise ValueError("min_train_size must be at least 1")

    first_test_start = min_train_size + gap
    if first_test_start >= n_samples:
        raise ValueError("Not enough rows for the requested train size and gap")

    test_size = (n_samples - first_test_start) // n_splits
    if test_size < 1:
        raise ValueError("Not enough rows to build the requested walk-forward splits")

    start = n_samples - test_size * n_splits
    splits: list[tuple[int, int, int]] = []
    for fold in range(n_splits):
        test_start = start + fold * test_size
        test_end = test_start + test_size
        train_end = test_start - gap
        splits.append((train_end, test_start, test_end))
    return splits


def forecast_split_leaks(
    feature_dates: pd.Index,
    target_dates: pd.Index,
    train_end: int,
    test_start: int,
) -> bool:
    """Return whether the last training target overlaps the first test features."""
    if train_end < 1 or test_start >= len(feature_dates):
        raise ValueError("Split bounds are outside the supervised sample")
    if len(feature_dates) != len(target_dates):
        raise ValueError("feature_dates and target_dates must have the same length")

    last_train_target = pd.Timestamp(target_dates[train_end - 1])
    first_test_feature = pd.Timestamp(feature_dates[test_start])
    return last_train_target >= first_test_feature
