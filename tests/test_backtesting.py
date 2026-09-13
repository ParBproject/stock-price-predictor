import numpy as np
import pytest

from src.backtesting import next_day_direction_signals


def test_next_day_direction_signals_compare_forecast_with_current_close():
    predicted_next_close = np.array([101.0, 99.0, 105.0])
    current_close = np.array([100.0, 100.0, 104.0])

    result = next_day_direction_signals(predicted_next_close, current_close)

    np.testing.assert_array_equal(result, np.array([1, 0, 1], dtype=np.int8))


def test_next_day_direction_signals_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        next_day_direction_signals(np.array([101.0, 102.0]), np.array([100.0]))


@pytest.mark.parametrize(
    "predicted,current",
    [
        (np.array([[101.0]]), np.array([100.0])),
        (np.array([101.0]), np.array([[100.0]])),
    ],
)
def test_next_day_direction_signals_require_one_dimensional_inputs(predicted, current):
    with pytest.raises(ValueError, match="must be 1-D"):
        next_day_direction_signals(predicted, current)


@pytest.mark.parametrize(
    "predicted,current",
    [
        (np.array([np.nan]), np.array([100.0])),
        (np.array([101.0]), np.array([np.inf])),
    ],
)
def test_next_day_direction_signals_reject_non_finite_values(predicted, current):
    with pytest.raises(ValueError, match="must be finite"):
        next_day_direction_signals(predicted, current)
