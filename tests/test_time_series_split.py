import numpy as np
import pandas as pd
import pytest

from src.data_loader import time_series_split


def test_time_series_split_preserves_chronological_order():
    index = pd.date_range("2026-01-05", periods=10, freq="B")
    df = pd.DataFrame({"value": np.arange(10)}, index=index)

    train, test = time_series_split(df, train_ratio=0.7)

    assert train.index.tolist() == index[:7].tolist()
    assert test.index.tolist() == index[7:].tolist()
    assert train["value"].tolist() == list(range(7))
    assert test["value"].tolist() == list(range(7, 10))


@pytest.mark.parametrize(
    "train_ratio",
    [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, -np.inf],
)
def test_time_series_split_rejects_invalid_numeric_ratios(train_ratio):
    df = pd.DataFrame({"value": range(10)})

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        time_series_split(df, train_ratio=train_ratio)


@pytest.mark.parametrize("train_ratio", [True, False, "0.8", None])
def test_time_series_split_rejects_non_numeric_ratios(train_ratio):
    df = pd.DataFrame({"value": range(10)})

    with pytest.raises(TypeError, match="real number"):
        time_series_split(df, train_ratio=train_ratio)


@pytest.mark.parametrize(
    "df,train_ratio",
    [
        (pd.DataFrame({"value": []}), 0.8),
        (pd.DataFrame({"value": [1]}), 0.8),
        (pd.DataFrame({"value": [1, 2, 3]}), 0.1),
    ],
)
def test_time_series_split_rejects_empty_partition(df, train_ratio):
    with pytest.raises(ValueError, match="non-empty train and test"):
        time_series_split(df, train_ratio=train_ratio)
