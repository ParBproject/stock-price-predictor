import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    buy_and_hold_equity_values,
    commission_aware_position_size,
    is_terminal_order,
    long_flat_returns_from_signals,
    next_day_direction_signals,
    next_open_long_flat_returns,
    one_step_strategy_returns,
    portfolio_value_series,
)


class DummyOrder:
    Submitted = 1
    Accepted = 2
    Partial = 3
    Completed = 4
    Canceled = 5
    Expired = 6
    Margin = 7
    Rejected = 8

    def __init__(self, status):
        self.status = status


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


def test_long_flat_returns_charge_commission_only_on_position_turnover():
    signals = np.array([1, 1, 0, 1], dtype=float)
    realized = np.array([0.02, 0.03, -0.04, 0.05])

    result = long_flat_returns_from_signals(
        signals, realized, commission_rate=0.001
    )

    expected = np.array([0.019, 0.03, -0.001, 0.049])
    np.testing.assert_allclose(result, expected)


def test_long_flat_returns_reject_invalid_signals():
    with pytest.raises(ValueError, match="only 0 .* or 1"):
        long_flat_returns_from_signals(
            np.array([1.0, -1.0]), np.array([0.01, 0.02])
        )


@pytest.mark.parametrize("commission_rate", [-0.001, np.nan])
def test_long_flat_returns_reject_invalid_commission(commission_rate):
    with pytest.raises(ValueError, match="finite and non-negative"):
        long_flat_returns_from_signals(
            np.array([1.0]), np.array([0.01]), commission_rate=commission_rate
        )


def test_next_open_returns_follow_backtrader_execution_timing():
    signals = np.array([1, 1, 0, 1], dtype=float)
    opens = np.array([105.0, 111.0, 107.0, 98.0])
    closes = np.array([110.0, 108.0, 100.0, 102.0])

    result = next_open_long_flat_returns(
        signals, opens, closes, commission_rate=0.001
    )

    expected = np.array(
        [
            110.0 / 105.0 - 1.0 - 0.001,
            108.0 / 110.0 - 1.0,
            107.0 / 108.0 - 1.0 - 0.001,
            102.0 / 98.0 - 1.0 - 0.001,
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_next_open_returns_exclude_untradeable_gap_on_entry():
    signals = np.array([1.0])
    opens = np.array([110.0])
    closes = np.array([111.0])

    result = next_open_long_flat_returns(signals, opens, closes)

    np.testing.assert_allclose(result, np.array([111.0 / 110.0 - 1.0]))


def test_next_open_returns_stay_zero_while_flat():
    result = next_open_long_flat_returns(
        np.array([0.0, 0.0]),
        np.array([100.0, 105.0]),
        np.array([103.0, 101.0]),
        commission_rate=0.001,
    )

    np.testing.assert_array_equal(result, np.zeros(2))


@pytest.mark.parametrize(
    "signals,opens,closes,commission_rate,match",
    [
        (np.array([1.0, 0.0]), np.array([100.0]), np.array([101.0]), 0.0, "matching shapes"),
        (np.array([[1.0]]), np.array([100.0]), np.array([101.0]), 0.0, "must be 1-D"),
        (np.array([2.0]), np.array([100.0]), np.array([101.0]), 0.0, "only 0 .* or 1"),
        (np.array([1.0]), np.array([0.0]), np.array([101.0]), 0.0, "must be positive"),
        (np.array([1.0]), np.array([100.0]), np.array([np.nan]), 0.0, "must be finite"),
        (np.array([1.0]), np.array([100.0]), np.array([101.0]), -0.001, "finite and non-negative"),
    ],
)
def test_next_open_returns_reject_invalid_inputs(
    signals, opens, closes, commission_rate, match
):
    with pytest.raises(ValueError, match=match):
        next_open_long_flat_returns(
            signals, opens, closes, commission_rate=commission_rate
        )


def test_one_step_strategy_returns_uses_known_previous_closes():
    predicted = np.array([110.0, 109.0, 90.0])
    actual = np.array([108.0, 107.0, 95.0])

    result = one_step_strategy_returns(predicted, actual, initial_previous_close=100.0)

    previous = np.array([100.0, 108.0, 107.0])
    realized = (actual - previous) / previous
    expected_signals = np.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(result, expected_signals * realized)


def test_one_step_strategy_returns_deducts_commission_on_entry_and_exit():
    result = one_step_strategy_returns(
        np.array([105.0, 99.0]),
        np.array([104.0, 100.0]),
        initial_previous_close=100.0,
        commission_rate=0.001,
    )

    expected = np.array([0.039, -0.001])
    np.testing.assert_allclose(result, expected)


def test_one_step_strategy_returns_down_forecast_stays_flat_in_falling_market():
    result = one_step_strategy_returns(
        np.array([90.0]),
        np.array([95.0]),
        initial_previous_close=100.0,
    )

    np.testing.assert_allclose(result, np.array([0.0]))


def test_one_step_strategy_returns_includes_first_holdout_trade():
    result = one_step_strategy_returns(
        np.array([105.0]),
        np.array([104.0]),
        initial_previous_close=100.0,
    )

    np.testing.assert_allclose(result, np.array([0.04]))


def test_one_step_strategy_returns_handles_empty_holdout():
    result = one_step_strategy_returns(
        np.array([], dtype=float),
        np.array([], dtype=float),
        initial_previous_close=100.0,
    )

    assert result.shape == (0,)


@pytest.mark.parametrize(
    "predicted,actual,initial_previous_close,match",
    [
        (np.array([101.0, 102.0]), np.array([101.0]), 100.0, "matching shapes"),
        (np.array([[101.0]]), np.array([101.0]), 100.0, "must be 1-D"),
        (np.array([np.nan]), np.array([101.0]), 100.0, "must be finite"),
        (np.array([101.0]), np.array([0.0]), 100.0, "must be positive"),
        (np.array([101.0]), np.array([101.0]), 0.0, "finite and positive"),
    ],
)
def test_one_step_strategy_returns_rejects_invalid_inputs(
    predicted, actual, initial_previous_close, match
):
    with pytest.raises(ValueError, match=match):
        one_step_strategy_returns(predicted, actual, initial_previous_close)


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


def test_buy_and_hold_enters_on_first_executable_next_open():
    opens = np.array([100.0, 110.0, 120.0])
    closes = np.array([105.0, 115.0, 125.0])

    result = buy_and_hold_equity_values(
        opens,
        closes,
        initial_cash=1_000.0,
        commission_rate=0.001,
        entry_index=1,
    )

    shares = 9
    remaining_cash = 1_000.0 - shares * 110.0 * 1.001
    expected = np.array(
        [
            1_000.0,
            remaining_cash + shares * 115.0,
            remaining_cash + shares * 125.0,
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_buy_and_hold_excludes_first_feature_bar_overnight_gap():
    result = buy_and_hold_equity_values(
        np.array([100.0, 150.0]),
        np.array([100.0, 150.0]),
        initial_cash=1_000.0,
        entry_index=1,
    )

    np.testing.assert_allclose(result, np.array([1_000.0, 1_000.0]))


def test_buy_and_hold_stays_in_cash_when_no_entry_bar_exists():
    result = buy_and_hold_equity_values(
        np.array([100.0]),
        np.array([105.0]),
        initial_cash=1_000.0,
        entry_index=1,
    )

    np.testing.assert_array_equal(result, np.array([1_000.0]))


@pytest.mark.parametrize(
    "opens,closes,initial_cash,commission_rate,entry_index,error,match",
    [
        (np.array([100.0, 101.0]), np.array([100.0]), 1_000.0, 0.0, 1, ValueError, "matching shapes"),
        (np.array([[100.0]]), np.array([100.0]), 1_000.0, 0.0, 0, ValueError, "one-dimensional"),
        (np.array([0.0]), np.array([100.0]), 1_000.0, 0.0, 0, ValueError, "must be positive"),
        (np.array([100.0]), np.array([np.nan]), 1_000.0, 0.0, 0, ValueError, "finite values"),
        (np.array([100.0]), np.array([100.0]), -1.0, 0.0, 0, ValueError, "non-negative"),
        (np.array([100.0]), np.array([100.0]), 1_000.0, -0.001, 0, ValueError, "non-negative"),
        (np.array([100.0]), np.array([100.0]), 1_000.0, 0.0, -1, ValueError, "non-negative"),
        (np.array([100.0]), np.array([100.0]), 1_000.0, 0.0, True, TypeError, "integer"),
    ],
)
def test_buy_and_hold_rejects_invalid_inputs(
    opens,
    closes,
    initial_cash,
    commission_rate,
    entry_index,
    error,
    match,
):
    with pytest.raises(error, match=match):
        buy_and_hold_equity_values(
            opens,
            closes,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            entry_index=entry_index,
        )


@pytest.mark.parametrize(
    "status",
    [
        DummyOrder.Completed,
        DummyOrder.Canceled,
        DummyOrder.Margin,
        DummyOrder.Rejected,
        DummyOrder.Expired,
    ],
)
def test_terminal_order_statuses_release_pending_order(status):
    assert is_terminal_order(DummyOrder(status))


@pytest.mark.parametrize(
    "status",
    [DummyOrder.Submitted, DummyOrder.Accepted, DummyOrder.Partial],
)
def test_active_order_statuses_remain_pending(status):
    assert not is_terminal_order(DummyOrder(status))


def test_portfolio_value_series_aligns_one_value_per_date():
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    values = [10_000.0, 10_050.0, 10_025.0]

    result = portfolio_value_series(values, dates)

    expected = pd.Series(values, index=dates, name="Portfolio Value")
    pd.testing.assert_series_equal(result, expected)


def test_portfolio_value_series_rejects_length_mismatch():
    dates = pd.date_range("2026-01-05", periods=3, freq="B")

    with pytest.raises(ValueError, match="same length"):
        portfolio_value_series([10_000.0, 10_050.0, 10_025.0, 10_100.0], dates)


def test_portfolio_value_series_rejects_non_finite_values():
    dates = pd.date_range("2026-01-05", periods=2, freq="B")

    with pytest.raises(ValueError, match="must be finite"):
        portfolio_value_series([10_000.0, np.nan], dates)
