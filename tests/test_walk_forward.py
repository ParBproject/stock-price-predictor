import numpy as np
import pandas as pd
import pytest

from src.data_loader import prepare_forecast_data
from src.validation import forecast_split_leaks, iter_expanding_forecast_splits


def test_expanding_splits_cover_the_tail_and_keep_a_gap():
    splits = iter_expanding_forecast_splits(100, n_splits=4, gap=1, min_train_size=50)

    assert splits[-1][2] == 100
    train_ends = [train_end for train_end, _, _ in splits]
    assert train_ends == sorted(train_ends)
    for train_end, test_start, test_end in splits:
        assert test_start - train_end == 1
        assert test_end > test_start


def test_expanding_splits_reject_a_too_small_sample():
    with pytest.raises(ValueError, match="Not enough rows"):
        iter_expanding_forecast_splits(10, n_splits=4, gap=1, min_train_size=8)


def test_one_step_gap_does_not_leak_the_next_close():
    index = pd.bdate_range("2024-01-01", periods=80)
    frame = pd.DataFrame(
        {"feature": np.arange(len(index), dtype=float), "Close": np.linspace(10, 20, len(index))},
        index=index,
    )
    features, _, target_index = prepare_forecast_data(frame, ["feature"], horizon=1)
    splits = iter_expanding_forecast_splits(len(features), n_splits=3, gap=1, min_train_size=30)

    for train_end, test_start, _ in splits:
        assert forecast_split_leaks(features.index, target_index, train_end, test_start) is False


def test_zero_gap_leaks_the_next_close_into_the_test_features():
    index = pd.bdate_range("2024-01-01", periods=40)
    frame = pd.DataFrame(
        {"feature": np.arange(len(index), dtype=float), "Close": np.arange(len(index), dtype=float)},
        index=index,
    )
    features, _, target_index = prepare_forecast_data(frame, ["feature"], horizon=1)
    train_end, test_start, _ = iter_expanding_forecast_splits(
        len(features), n_splits=2, gap=0, min_train_size=15
    )[0]

    assert forecast_split_leaks(features.index, target_index, train_end, test_start) is True
