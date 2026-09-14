"""Leakage-safe feature scaling helpers for chronological model partitions."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_time_series_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Close",
):
    """Scale chronological partitions with a scaler fit only on ``train``.

    Validation and test rows are transformed with the training-fitted scaler,
    so future extrema cannot influence the representation used to fit the model.
    Values outside the training range are intentionally allowed to transform
    below 0 or above 1.
    """
    cols = list(dict.fromkeys([*feature_cols, target_col]))
    if not cols:
        raise ValueError("At least one feature or target column is required")

    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    for name, frame in partitions.items():
        missing = [col for col in cols if col not in frame.columns]
        if missing:
            raise KeyError(f"{name} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} partition must not be empty")
        values = frame[cols].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} partition must contain only finite values")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train[cols])
    validation_scaled = scaler.transform(validation[cols])
    test_scaled = scaler.transform(test[cols])

    return train_scaled, validation_scaled, test_scaled, scaler
