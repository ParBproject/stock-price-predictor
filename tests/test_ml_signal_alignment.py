"""Regression tests for pairing each market bar with its own signal."""

import json
from datetime import date
from pathlib import Path

import backtrader as bt
import pandas as pd

from src.backtesting import (
    commission_aware_position_size,
    is_terminal_order,
    signal_for_timestamp,
)


class _DelayFiller:
    """Leave each order unfilled for ``skip`` execution attempts, then fill it."""

    def __init__(self, skip):
        self.skip = skip
        self.calls = {}

    def __call__(self, order, price, ago):
        seen = self.calls.get(order.ref, 0)
        self.calls[order.ref] = seen + 1
        if seen < self.skip:
            return 0
        return abs(order.executed.remsize)


class _CompletedOrders(bt.Analyzer):
    def __init__(self):
        self.orders = []
        self._seen = set()

    def notify_order(self, order):
        if order.status != order.Completed or order.ref in self._seen:
            return
        self._seen.add(order.ref)
        self.orders.append(
            {
                "created": bt.num2date(order.created.dt).date(),
                "executed": bt.num2date(order.executed.dt).date(),
                "size": order.executed.size,
            }
        )


def _strategy_source() -> str:
    notebook = json.loads(Path("notebooks/backtesting.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "class MLSignalStrategy" in source:
            return source
    raise AssertionError("MLSignalStrategy was not found in notebooks/backtesting.ipynb")


def _load_strategy():
    namespace = {
        "bt": bt,
        "commission_aware_position_size": commission_aware_position_size,
        "is_terminal_order": is_terminal_order,
        "signal_for_timestamp": signal_for_timestamp,
    }
    exec(_strategy_source(), namespace)
    return namespace["MLSignalStrategy"]


def _prices(index) -> pd.DataFrame:
    count = len(index)
    return pd.DataFrame(
        {
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.0] * count,
            "volume": [1_000_000] * count,
        },
        index=pd.DatetimeIndex(index),
    )


def _run(prices, signals, filler=None):
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=prices))
    cerebro.addstrategy(
        _load_strategy(),
        signals=signals,
        commission_rate=0.0,
    )
    cerebro.addanalyzer(_CompletedOrders, _name="orders")
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(commission=0.0)
    if filler is not None:
        cerebro.broker.set_filler(filler)
    strategy = cerebro.run()[0]
    return strategy, strategy.analyzers.orders.orders


def test_pending_order_does_not_shift_signals_onto_later_bars():
    """A buy that stays pending must not replay an older exit on a later bar.

    Prices cover eight sessions. The Jan 4 signal is intentionally absent, and
    the Jan 2 buy cannot fill on Jan 3 or Jan 4. Jan 3's exit belongs only to
    Jan 3; the position must stay open until the Jan 8 exit.
    """
    prices = _prices(pd.date_range("2024-01-02", periods=8, freq="B"))
    signals = pd.Series(
        [1, 0, -1, 0, -1, -1, -1],
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
            ]
        ),
    )

    strategy, orders = _run(prices, signals, filler=_DelayFiller(skip=2))

    assert [(order["created"], order["size"]) for order in orders] == [
        (date(2024, 1, 2), 100),
        (date(2024, 1, 8), -100),
    ]
    assert [order["executed"] for order in orders] == [
        date(2024, 1, 5),
        date(2024, 1, 11),
    ]
    assert strategy.position.size == 0
    assert len(strategy.portfolio_values) == len(prices)


def test_aligned_signals_trade_on_their_own_bars_when_orders_fill_normally():
    prices = _prices(pd.date_range("2024-01-02", periods=5, freq="B"))
    signals = pd.Series([1, 1, 0, -1, -1], index=prices.index)

    strategy, orders = _run(prices, signals)

    assert [(order["created"], order["size"]) for order in orders] == [
        (date(2024, 1, 2), 100),
        (date(2024, 1, 4), -100),
    ]
    assert [order["executed"] for order in orders] == [
        date(2024, 1, 3),
        date(2024, 1, 5),
    ]
    assert strategy.position.size == 0
    assert len(strategy.portfolio_values) == len(prices)
