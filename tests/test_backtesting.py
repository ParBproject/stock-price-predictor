import numpy as np
import pytest

from src.backtesting import (
    commission_aware_position_size,
    next_day_direction_signals,
)


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


def test_commission_aware_position_size_keeps_total_cost_within_cash():
    cash = 10_000.0
    price = 100.0
    commission_rate = 0.001

    size = commission_aware_position_size(cash, price, commission_rate)

    assert size == 99
    assert size * price * (1.0 + commission_rate) <= cash
    assert (size + 1) * price * (1.0 + commission_rate) > cash


@pytest.mark.parametrize(
    "cash,price,commission_rate,expected",
    [
        (0.0, 100.0, 0.001, 0),
        (50.0, 100.0, 0.001, 0),
        (1_000.0, 100.0, 0.0, 10),
    ],
)
def test_commission_aware_position_size_boundary_cases(
    cash, price, commission_rate, expected
):
    assert commission_aware_position_size(cash, price, commission_rate) == expected


@pytest.mark.parametrize(
    "cash,price,commission_rate,match",
    [
        (-1.0, 100.0, 0.001, "cash must be non-negative"),
        (100.0, 0.0, 0.001, "price must be positive"),
        (100.0, -5.0, 0.001, "price must be positive"),
        (100.0, 10.0, -0.001, "commission_rate must be non-negative"),
        (np.nan, 10.0, 0.001, "must be finite"),
        (100.0, np.inf, 0.001, "must be finite"),
    ],
)
def test_commission_aware_position_size_rejects_invalid_inputs(
    cash, price, commission_rate, match
):
    with pytest.raises(ValueError, match=match):
        commission_aware_position_size(cash, price, commission_rate)
