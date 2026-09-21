import numpy as np
import pandas as pd
import pytest

from src.baseline import persistence_forecast, same_day_feature_target, score_forecast
from src.data_loader import prepare_forecast_data


def test_persistence_forecast_mae_on_synthetic_series():
    """Tomorrow's close equals today's close, so the errors are known exactly.

    Closes: 10, 12, 11, 15.
    Persistence predictions for the last three closes: 10, 12, 11.
    Absolute errors: |12-10|, |11-12|, |15-11| = 2, 1, 4. MAE = 7/3.
    Direction versus the previous close never calls up, so the only hit is
    the middle day (11 is not above 12). Hit rate = 1/3.
    """
    close = pd.Series(
        [10.0, 12.0, 11.0, 15.0],
        index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    previous_close = close.iloc[:-1].to_numpy()
    next_close = close.iloc[1:].to_numpy()

    predicted = persistence_forecast(previous_close)
    scores = score_forecast(next_close, predicted, previous_close, label="Persistence")

    np.testing.assert_array_equal(predicted, previous_close)
    assert scores["MAE"] == pytest.approx(7.0 / 3.0)
    assert scores["directional_hit_rate"] == pytest.approx(1.0 / 3.0)


def test_same_day_target_is_the_close_on_that_row():
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    frame = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0], "Close": [10.0, 12.0, 11.0]},
        index=index,
    )

    features, target = same_day_feature_target(frame, ["feature"], index)
    _, next_day, _ = prepare_forecast_data(frame, ["feature"], horizon=1)

    pd.testing.assert_index_equal(features.index, index)
    assert target.tolist() == [10.0, 12.0, 11.0]
    assert target.name == "Close_same_day_leaky"
    assert next_day.tolist() != target.tolist()
