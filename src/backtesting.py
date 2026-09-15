"""Utilities for turning next-day forecasts into backtest trading signals."""

import numpy as np
import pandas as pd


def next_day_direction_signals(
    predicted_next_close: np.ndarray,
    current_close: np.ndarray,
) -> np.ndarray:
    """Return long/exit signals from next-day close forecasts.

    A prediction generated from features observed at date ``t`` estimates
    ``Close[t+1]``. The actionable direction at the end of date ``t`` is
    therefore determined by comparing that forecast with the currently known
    ``Close[t]``.

    Returns ``1`` when the forecast is above the current close and ``0``
    otherwise.
    """
    predicted = np.asarray(predicted_next_close, dtype=float)
    current = np.asarray(current_close, dtype=float)

    if predicted.ndim != 1 or current.ndim != 1:
        raise ValueError("predicted_next_close and current_close must be 1-D")
    if predicted.shape != current.shape:
        raise ValueError("predicted_next_close and current_close must have matching shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(current).all():
        raise ValueError("forecast and current-close values must be finite")

    return (predicted > current).astype(np.int8)


def long_flat_returns_from_signals(
    signals: np.ndarray,
    realized_returns: np.ndarray,
    commission_rate: float = 0.0,
) -> np.ndarray:
    """Apply long/flat signals to returns and charge commission on turnover.

    Signals are position states: ``1`` is long and ``0`` is flat. Commission is
    deducted when the state changes (entry, exit, or re-entry), matching the
    transaction events in the Backtrader strategy. The initial position is flat.
    """
    signals = np.asarray(signals, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)
    commission_rate = float(commission_rate)

    if signals.ndim != 1 or realized_returns.ndim != 1:
        raise ValueError("signals and realized_returns must be 1-D")
    if signals.shape != realized_returns.shape:
        raise ValueError("signals and realized_returns must have matching shapes")
    if not np.isfinite(signals).all() or not np.isfinite(realized_returns).all():
        raise ValueError("signals and realized_returns must be finite")
    if not np.isin(signals, [0.0, 1.0]).all():
        raise ValueError("signals must contain only 0 (flat) or 1 (long)")
    if not np.isfinite(commission_rate) or commission_rate < 0:
        raise ValueError("commission_rate must be finite and non-negative")
    if len(signals) == 0:
        return np.empty((0,), dtype=float)

    previous_signals = np.concatenate(([0.0], signals[:-1]))
    turnover = np.abs(signals - previous_signals)
    return signals * realized_returns - turnover * commission_rate


def next_open_long_flat_returns(
    signals: np.ndarray,
    target_open: np.ndarray,
    target_close: np.ndarray,
    commission_rate: float = 0.0,
) -> np.ndarray:
    """Return long/flat strategy returns with market orders filled next open.

    ``signals[i]`` is known only after the feature bar preceding target bar
    ``i`` has closed. A change in desired position therefore executes at
    ``target_open[i]`` rather than at the preceding close.

    The period return follows the actual position path:

    - flat -> long: enter at current open, then earn current open-to-close;
    - long -> long: stay invested across the overnight gap and current session;
    - long -> flat: remain invested through the overnight gap, then exit at open;
    - flat -> flat: remain in cash.

    Commission is deducted once for each entry or exit transition. The initial
    state is flat, matching the Backtrader strategy.
    """
    signals = np.asarray(signals, dtype=float)
    target_open = np.asarray(target_open, dtype=float)
    target_close = np.asarray(target_close, dtype=float)
    commission_rate = float(commission_rate)

    if signals.ndim != 1 or target_open.ndim != 1 or target_close.ndim != 1:
        raise ValueError("signals, target_open, and target_close must be 1-D")
    if not (signals.shape == target_open.shape == target_close.shape):
        raise ValueError("signals, target_open, and target_close must have matching shapes")
    if not np.isfinite(signals).all():
        raise ValueError("signals must be finite")
    if not np.isin(signals, [0.0, 1.0]).all():
        raise ValueError("signals must contain only 0 (flat) or 1 (long)")
    if not np.isfinite(target_open).all() or not np.isfinite(target_close).all():
        raise ValueError("target_open and target_close must be finite")
    if (target_open <= 0).any() or (target_close <= 0).any():
        raise ValueError("target_open and target_close must be positive")
    if not np.isfinite(commission_rate) or commission_rate < 0:
        raise ValueError("commission_rate must be finite and non-negative")
    if len(signals) == 0:
        return np.empty((0,), dtype=float)

    strategy_returns = np.zeros(len(signals), dtype=float)
    previous_signal = 0.0

    for i, signal in enumerate(signals):
        if previous_signal == 0.0 and signal == 1.0:
            period_return = target_close[i] / target_open[i] - 1.0
        elif previous_signal == 1.0 and signal == 1.0:
            period_return = target_close[i] / target_close[i - 1] - 1.0
        elif previous_signal == 1.0 and signal == 0.0:
            period_return = target_open[i] / target_close[i - 1] - 1.0
        else:
            period_return = 0.0

        if signal != previous_signal:
            period_return -= commission_rate

        strategy_returns[i] = period_return
        previous_signal = signal

    return strategy_returns


def one_step_strategy_returns(
    predicted_close: np.ndarray,
    actual_close: np.ndarray,
    initial_previous_close: float,
    commission_rate: float = 0.0,
) -> np.ndarray:
    """Return realized long/flat returns for one-step close forecasts.

    The first holdout forecast is compared with ``initial_previous_close``
    (normally the final training close). Each later forecast is compared with
    the preceding realized holdout close. A forecast above the known prior
    close is long (1); otherwise the strategy is flat (0), matching the
    Backtrader strategy's buy/exit behavior. Transaction costs are deducted on
    position changes when ``commission_rate`` is non-zero.
    """
    predicted = np.asarray(predicted_close, dtype=float)
    actual = np.asarray(actual_close, dtype=float)
    initial_previous_close = float(initial_previous_close)

    if predicted.ndim != 1 or actual.ndim != 1:
        raise ValueError("predicted_close and actual_close must be 1-D")
    if predicted.shape != actual.shape:
        raise ValueError("predicted_close and actual_close must have matching shapes")
    if not np.isfinite(initial_previous_close) or initial_previous_close <= 0:
        raise ValueError("initial_previous_close must be finite and positive")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("predicted_close and actual_close values must be finite")
    if (actual <= 0).any():
        raise ValueError("actual_close values must be positive")
    if len(actual) == 0:
        return np.empty((0,), dtype=float)

    previous_close = np.concatenate(
        ([initial_previous_close], actual[:-1])
    )
    realized_returns = (actual - previous_close) / previous_close
    signals = (predicted > previous_close).astype(float)
    return long_flat_returns_from_signals(
        signals, realized_returns, commission_rate=commission_rate
    )


def commission_aware_position_size(
    cash: float,
    price: float,
    commission_rate: float = 0.0,
) -> int:
    """Return the largest whole-share position affordable after commission.

    ``commission_rate`` is expressed as a fraction of notional value, e.g.
    ``0.001`` for 0.1%. The calculation uses the supplied reference price and
    does not assume future execution prices.
    """
    cash = float(cash)
    price = float(price)
    commission_rate = float(commission_rate)

    if not np.isfinite(cash) or not np.isfinite(price) or not np.isfinite(commission_rate):
        raise ValueError("cash, price, and commission_rate must be finite")
    if cash < 0:
        raise ValueError("cash must be non-negative")
    if price <= 0:
        raise ValueError("price must be positive")
    if commission_rate < 0:
        raise ValueError("commission_rate must be non-negative")

    cost_per_share = price * (1.0 + commission_rate)
    return int(np.floor(cash / cost_per_share))


def buy_and_hold_equity_values(
    opens: np.ndarray,
    closes: np.ndarray,
    initial_cash: float,
    commission_rate: float = 0.0,
    entry_index: int = 1,
) -> np.ndarray:
    """Mark a Buy & Hold benchmark from the first executable market open.

    The benchmark remains in cash before ``entry_index``. At that bar's open it
    buys the largest whole-share position affordable after commission, then
    marks the remaining cash plus shares to each close. No synthetic exit fee is
    charged because the benchmark is not liquidated at the end of the series.
    """
    opens = np.asarray(opens, dtype=float)
    closes = np.asarray(closes, dtype=float)
    initial_cash = float(initial_cash)
    commission_rate = float(commission_rate)

    if opens.ndim != 1 or closes.ndim != 1:
        raise ValueError("opens and closes must be one-dimensional")
    if opens.shape != closes.shape:
        raise ValueError("opens and closes must have matching shapes")
    if not np.isfinite(opens).all() or not np.isfinite(closes).all():
        raise ValueError("opens and closes must contain only finite values")
    if (opens <= 0).any() or (closes <= 0).any():
        raise ValueError("opens and closes must be positive")
    if not np.isfinite(initial_cash) or initial_cash < 0:
        raise ValueError("initial_cash must be finite and non-negative")
    if not np.isfinite(commission_rate) or commission_rate < 0:
        raise ValueError("commission_rate must be finite and non-negative")
    if isinstance(entry_index, bool) or not isinstance(entry_index, (int, np.integer)):
        raise TypeError("entry_index must be a non-negative integer")
    if entry_index < 0:
        raise ValueError("entry_index must be non-negative")

    equity = np.full(len(closes), initial_cash, dtype=float)
    if len(closes) == 0 or entry_index >= len(closes):
        return equity

    shares = commission_aware_position_size(
        initial_cash, opens[entry_index], commission_rate
    )
    entry_cost = shares * opens[entry_index] * (1.0 + commission_rate)
    remaining_cash = initial_cash - entry_cost
    equity[entry_index:] = remaining_cash + shares * closes[entry_index:]
    return equity


def is_terminal_order(order) -> bool:
    """Return whether a Backtrader order has reached a terminal status.

    The helper deliberately uses attributes from the supplied order object so
    the utility module does not need to import Backtrader. Terminal states must
    release any pending-order guard in a strategy; submitted, accepted, and
    partial orders remain active.
    """
    terminal_statuses = (
        order.Completed,
        order.Canceled,
        order.Margin,
        order.Rejected,
        order.Expired,
    )
    return order.status in terminal_statuses


def portfolio_value_series(
    portfolio_values,
    dates,
    name: str = "Portfolio Value",
) -> pd.Series:
    """Build a portfolio-value Series with one value per backtest date."""
    values = np.asarray(portfolio_values, dtype=float)
    index = pd.Index(dates)

    if values.ndim != 1:
        raise ValueError("portfolio_values must be one-dimensional")
    if len(values) != len(index):
        raise ValueError(
            "portfolio_values and dates must have the same length "
            f"({len(values)} values for {len(index)} dates)"
        )
    if not np.isfinite(values).all():
        raise ValueError("portfolio_values must be finite")

    return pd.Series(values, index=index, name=name)
