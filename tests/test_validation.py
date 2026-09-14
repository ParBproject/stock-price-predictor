import numpy as np
import pytest

from src.validation import make_forecast_time_series_split


def test_forecast_time_series_split_embargoes_one_step_target_boundary():
    splitter = make_forecast_time_series_split(n_splits=3, forecast_horizon=1)

    for train_idx, validation_idx in splitter.split(np.arange(24)):
        assert validation_idx[0] - train_idx[-1] - 1 == 1


def test_forecast_time_series_split_gap_matches_multi_step_horizon():
    splitter = make_forecast_time_series_split(n_splits=3, forecast_horizon=2)

    for train_idx, validation_idx in splitter.split(np.arange(24)):
        assert validation_idx[0] - train_idx[-1] - 1 == 2


@pytest.mark.parametrize("n_splits", [0, 1])
def test_forecast_time_series_split_rejects_too_few_splits(n_splits):
    with pytest.raises(ValueError, match="at least 2"):
        make_forecast_time_series_split(n_splits=n_splits)


@pytest.mark.parametrize("n_splits", [True, 2.5, "3"])
def test_forecast_time_series_split_rejects_non_integer_splits(n_splits):
    with pytest.raises(TypeError, match="n_splits must be an integer"):
        make_forecast_time_series_split(n_splits=n_splits)


@pytest.mark.parametrize("forecast_horizon", [0, -1])
def test_forecast_time_series_split_rejects_non_positive_horizon(forecast_horizon):
    with pytest.raises(ValueError, match="at least 1"):
        make_forecast_time_series_split(forecast_horizon=forecast_horizon)


@pytest.mark.parametrize("forecast_horizon", [True, 1.5, "1"])
def test_forecast_time_series_split_rejects_non_integer_horizon(forecast_horizon):
    with pytest.raises(TypeError, match="forecast_horizon must be an integer"):
        make_forecast_time_series_split(forecast_horizon=forecast_horizon)
