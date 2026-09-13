import numpy as np
import pandas as pd
import pytest

from src.data_loader import build_holdout_sequences, prepare_forecast_data


def test_prepare_forecast_data_aligns_next_day_target():
    index = pd.date_range("2026-01-05", periods=4, freq="B")
    df = pd.DataFrame(
        {
            "feature": [10.0, 20.0, 30.0, 40.0],
            "Close": [100.0, 101.5, 99.0, 104.0],
        },
        index=index,
    )

    X, y, target_index = prepare_forecast_data(
        df, ["feature"], target_col="Close", horizon=1
    )

    expected_X = df[["feature"]].iloc[:-1]
    expected_y = pd.Series(
        [101.5, 99.0, 104.0],
        index=index[:-1],
        name="Close_t_plus_1",
    )

    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y, expected_y)
    pd.testing.assert_index_equal(target_index, index[1:])


def test_prepare_forecast_data_supports_multi_day_horizon():
    index = pd.date_range("2026-01-05", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "feature": np.arange(5, dtype=float),
            "Close": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
        index=index,
    )

    X, y, target_index = prepare_forecast_data(df, ["feature"], horizon=2)

    assert X["feature"].tolist() == [0.0, 1.0, 2.0]
    assert y.tolist() == [12.0, 13.0, 14.0]
    pd.testing.assert_index_equal(target_index, index[2:])


@pytest.mark.parametrize("horizon", [0, -1])
def test_prepare_forecast_data_rejects_non_positive_horizon(horizon):
    df = pd.DataFrame({"feature": [1, 2], "Close": [10, 11]})

    with pytest.raises(ValueError, match="at least 1"):
        prepare_forecast_data(df, ["feature"], horizon=horizon)


@pytest.mark.parametrize("horizon", [1.5, True, "1"])
def test_prepare_forecast_data_rejects_non_integer_horizon(horizon):
    df = pd.DataFrame({"feature": [1, 2], "Close": [10, 11]})

    with pytest.raises(TypeError, match="positive integer"):
        prepare_forecast_data(df, ["feature"], horizon=horizon)


def test_build_holdout_sequences_uses_training_tail_for_first_target():
    train = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    test = np.array(
        [
            [5.0, 50.0],
            [6.0, 60.0],
        ]
    )

    X, y = build_holdout_sequences(train, test, seq_len=3, target_idx=1)

    expected_X = np.array(
        [
            [[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        ]
    )
    expected_y = np.array([50.0, 60.0])

    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)
    assert len(y) == len(test)


def test_build_holdout_sequences_handles_empty_test_split():
    train = np.arange(12, dtype=float).reshape(6, 2)
    test = np.empty((0, 2), dtype=float)

    X, y = build_holdout_sequences(train, test, seq_len=3, target_idx=1)

    assert X.shape == (0, 3, 2)
    assert y.shape == (0,)


def test_build_holdout_sequences_rejects_insufficient_training_history():
    train = np.arange(4, dtype=float).reshape(2, 2)
    test = np.arange(4, 8, dtype=float).reshape(2, 2)

    with pytest.raises(ValueError, match="at least seq_len"):
        build_holdout_sequences(train, test, seq_len=3)


def test_build_holdout_sequences_rejects_mismatched_feature_widths():
    train = np.arange(12, dtype=float).reshape(6, 2)
    test = np.arange(9, dtype=float).reshape(3, 3)

    with pytest.raises(ValueError, match="same number of columns"):
        build_holdout_sequences(train, test, seq_len=3)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_build_holdout_sequences_rejects_non_positive_sequence_length(seq_len):
    data = np.arange(12, dtype=float).reshape(6, 2)

    with pytest.raises(ValueError, match="at least 1"):
        build_holdout_sequences(data, data[:2], seq_len=seq_len)


@pytest.mark.parametrize("seq_len", [1.5, True, "3"])
def test_build_holdout_sequences_rejects_non_integer_sequence_length(seq_len):
    data = np.arange(12, dtype=float).reshape(6, 2)

    with pytest.raises(TypeError, match="positive integer"):
        build_holdout_sequences(data, data[:2], seq_len=seq_len)
