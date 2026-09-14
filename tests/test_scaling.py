import numpy as np
import pandas as pd
import pytest

from src.scaling import scale_time_series_partitions


def test_scaler_fits_only_training_partition():
    train = pd.DataFrame(
        {
            "feature": [0.0, 10.0],
            "Close": [100.0, 110.0],
        }
    )
    validation = pd.DataFrame({"feature": [20.0], "Close": [120.0]})
    test = pd.DataFrame({"feature": [30.0], "Close": [130.0]})

    train_scaled, validation_scaled, test_scaled, scaler = (
        scale_time_series_partitions(
            train, validation, test, ["feature"], target_col="Close"
        )
    )

    np.testing.assert_allclose(train_scaled, [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(validation_scaled, [[2.0, 2.0]])
    np.testing.assert_allclose(test_scaled, [[3.0, 3.0]])
    np.testing.assert_allclose(scaler.data_max_, [10.0, 110.0])


def test_scaler_rejects_missing_required_columns():
    train = pd.DataFrame({"feature": [1.0], "Close": [100.0]})
    validation = pd.DataFrame({"feature": [2.0]})
    test = pd.DataFrame({"feature": [3.0], "Close": [102.0]})

    with pytest.raises(KeyError, match="validation is missing required columns"):
        scale_time_series_partitions(
            train, validation, test, ["feature"], target_col="Close"
        )


def test_scaler_rejects_empty_partition():
    train = pd.DataFrame({"feature": [1.0], "Close": [100.0]})
    validation = pd.DataFrame(columns=["feature", "Close"])
    test = pd.DataFrame({"feature": [3.0], "Close": [102.0]})

    with pytest.raises(ValueError, match="validation partition must not be empty"):
        scale_time_series_partitions(
            train, validation, test, ["feature"], target_col="Close"
        )


def test_scaler_rejects_non_finite_partition_values():
    train = pd.DataFrame({"feature": [1.0], "Close": [100.0]})
    validation = pd.DataFrame({"feature": [np.nan], "Close": [101.0]})
    test = pd.DataFrame({"feature": [3.0], "Close": [102.0]})

    with pytest.raises(ValueError, match="validation partition must contain only finite"):
        scale_time_series_partitions(
            train, validation, test, ["feature"], target_col="Close"
        )
