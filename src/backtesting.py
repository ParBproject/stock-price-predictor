"""Utilities for turning next-day forecasts into backtest trading signals."""

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

# A handful of intentional gaps is normal. An overlap this small usually means
# the price index and the signal index are not on the same clock.
_LOW_BAR_MATCH_FRACTION = 0.5


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


class SignalLookup:
    """Timestamp-to-signal map built once and reused for every bar.

    ``1`` enters long, ``0`` exits, and ``-1`` takes no new action.
    Timezone-aware keys are stored as naive UTC, matching the clock a
    Backtrader pandas feed reports for that same instant. A timestamp with no
    entry returns ``-1``, so one skipped bar cannot shift later signals.
    """

    def __init__(self, signals):
        self._signals = _signal_lookup(signals)

    def __len__(self) -> int:
        return len(self._signals)

    def get(self, timestamp) -> int:
        return self._signals.get(_bar_timestamp(timestamp), -1)

    def require_bar_coverage(self, bar_timestamps) -> None:
        """Raise when no bar shares a timestamp with this lookup.

        Individual missing bars stay no-action. Zero overlap, or only a small
        fraction of bars overlapping, means the price clock and the signal
        clock disagree; a backtest would otherwise emit no trades and look
        successful. Zero matches raise. A match share below half warns.
        """
        matched = 0
        total = 0
        for timestamp in bar_timestamps:
            total += 1
            if _bar_timestamp(timestamp) in self._signals:
                matched += 1
        if total == 0:
            return
        if matched == 0:
            raise ValueError(
                "no bar timestamps matched a signal; check that the price "
                "index and the signal index use the same clock "
                f"({total} bars, {len(self)} signals)"
            )
        if matched / total < _LOW_BAR_MATCH_FRACTION:
            warnings.warn(
                (
                    f"only {matched} of {total} bar timestamps matched a signal; "
                    "the price index and the signal index may not use the same clock"
                ),
                UserWarning,
                stacklevel=2,
            )


def signal_for_timestamp(signals, timestamp) -> int:
    """Return the trading signal that belongs to ``timestamp``.

    This builds a :class:`SignalLookup` for one query. A backtest should build
    that lookup once and call :meth:`SignalLookup.get` on each bar.
    """
    return SignalLookup(signals).get(timestamp)


def _signal_lookup(signals) -> dict:
    pairs = _signal_pairs(signals)
    lookup = {}
    for raw_timestamp, raw_signal in pairs:
        key = _bar_timestamp(raw_timestamp)
        if key in lookup:
            raise ValueError(f"duplicate signal for timestamp {key.isoformat()}")
        lookup[key] = _coerce_signal(raw_signal)
    return lookup


def _signal_pairs(signals):
    if isinstance(signals, pd.Series):
        if isinstance(signals.index, pd.MultiIndex):
            raise TypeError("timestamps must be dates or datetimes")
        return zip(signals.index, signals.to_numpy())
    if isinstance(signals, Mapping):
        return signals.items()
    raise TypeError(
        "signals must be a pandas Series or a mapping of timestamps to signals"
    )


def _bar_timestamp(value) -> pd.Timestamp:
    if value is None or isinstance(
        value, (bool, np.bool_, int, np.integer, float, np.floating)
    ):
        raise TypeError("timestamps must be dates or datetimes")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be valid dates or datetimes") from exc
    if pd.isna(timestamp):
        raise ValueError("timestamps must be valid dates or datetimes")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _coerce_signal(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("signal values must be integers, not booleans")
    if isinstance(value, (int, np.integer)):
        signal = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError("signal values must be -1, 0, or 1")
        signal = int(value)
    else:
        raise TypeError("signal values must be integers")
    if signal not in (-1, 0, 1):
        raise ValueError("signal values must be -1, 0, or 1")
    return signal


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


def assemble_signal_frame(
    df: pd.DataFrame,
    feature_index: pd.Index,
    target_index: pd.Index,
    predictions: np.ndarray,
    signals: np.ndarray,
) -> pd.DataFrame:
    """Attach next-day forecasts to the bars a backtest is allowed to see.

    The frame starts on the first feature date and ends on the last target
    date so an order placed on the final feature bar can fill on the next bar.
    That terminal bar has no new signal.
    """
    predictions = np.asarray(predictions, dtype=float)
    signals = np.asarray(signals)
    if len(feature_index) == 0:
        raise ValueError("feature_index must not be empty")
    if len(feature_index) != len(predictions) or len(feature_index) != len(signals):
        raise ValueError("predictions and signals must align with feature_index")
    if len(target_index) != len(feature_index):
        raise ValueError("target_index must align with feature_index")

    start = feature_index[0]
    end = target_index[-1]
    frame = df.loc[start:end].copy()
    if frame.empty:
        raise ValueError("backtest window is empty")
    missing = pd.Index(feature_index).difference(frame.index)
    if len(missing):
        raise ValueError("feature dates are missing from the backtest window")

    frame["Pred_Close"] = np.nan
    frame["Signal"] = -1
    frame.loc[feature_index, "Pred_Close"] = predictions
    frame.loc[feature_index, "Signal"] = signals
    return frame


def equity_period_returns(equity: pd.Series | np.ndarray) -> np.ndarray:
    """Return simple period returns between successive equity marks."""
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("equity must contain at least two marks")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("equity marks must be finite and positive")
    return values[1:] / values[:-1] - 1.0


def equity_from_period_returns(
    returns: np.ndarray,
    initial_cash: float,
    dates,
) -> pd.Series:
    """Build an equity curve that starts at ``initial_cash`` before the first return.

    ``dates`` must contain one more timestamp than ``returns``: the starting
    mark, then one mark after each period return.
    """
    returns = np.asarray(returns, dtype=float)
    initial_cash = float(initial_cash)
    dates = pd.Index(dates)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.isfinite(returns).all():
        raise ValueError("returns must be finite")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    if len(dates) != len(returns) + 1:
        raise ValueError(
            "dates must contain the starting mark plus one mark per return "
            f"({len(dates)} dates for {len(returns)} returns)"
        )

    values = np.empty(len(dates), dtype=float)
    values[0] = initial_cash
    if len(returns):
        values[1:] = initial_cash * np.cumprod(1.0 + returns)
    return portfolio_value_series(values, dates)


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
