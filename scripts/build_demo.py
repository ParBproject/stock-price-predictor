"""Build the static demo JSON from the repository's own forecast code.

The page at ``site/`` only displays this file. Nothing in it is typed by hand.

What runs
----------
* Adjusted daily bars from ``src.data_loader.fetch_stock_data``
* Next-day targets from ``src.data_loader.prepare_forecast_data``
* Embargoed walk-forward folds from ``src.validation.make_forecast_time_series_split``
* An untuned ``RandomForestRegressor`` from ``src.model_trainer.train_random_forest_regressor``
  (``tune=False``: 200 trees, ``max_depth=20``, ``random_state=42``)
* MAE and RMSE from ``src.evaluator.regression_metrics``
* A persistence baseline: the forecast for ``Close[t+1]`` is ``Close[t]``
* A Backtrader long/flat equity curve on the last fold, against buy-and-hold

The LSTM is not trained. ``src.model_trainer`` imports TensorFlow at module
scope, so when TensorFlow is not installed this script inserts placeholders
for those imports and then calls ``train_random_forest_regressor``. The LSTM
functions are never called. Sentiment is forced to the neutral fallback by
clearing ``NEWS_API_KEY`` before the sentiment module is imported.

The Backtrader strategy lives in this script on purpose. The notebook strategy
on main advances a positional signal index only when no order is pending, so a
fill that stays open can attach an older signal to a later bar (the bug fixed
on the unmerged pull request #69). This demo looks up the signal stored for
each bar's date instead, and it does not edit ``src/backtesting.py``.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Neutral sentiment, and a non-interactive matplotlib backend, before any
# project import that would snapshot the environment.
os.environ["NEWS_API_KEY"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_JSON_BYTES = 500_000
TICKERS = ("AAPL", "MSFT", "SPY")
START = "2015-01-01"
END = "2024-12-31"  # yfinance end is exclusive, matching the notebooks
N_SPLITS = 5
HORIZON = 1
INITIAL_CASH = 10_000.0
COMMISSION_RATE = 0.001
OUTPUT_PATH = ROOT / "site" / "data" / "demo.json"

FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Volume",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "RSI_7",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Width",
    "ATR_14",
    "Vol_Change",
    "OBV",
    "Log_Return",
    "Pct_Change",
    "Sentiment",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_5",
    "Close_Lag_10",
]


def persistence_forecast(current_close: np.ndarray) -> np.ndarray:
    """Forecast the next close as the close already known at feature time t."""
    current = np.asarray(current_close, dtype=float)
    if current.ndim != 1:
        raise ValueError("current close must be one-dimensional")
    if len(current) == 0:
        raise ValueError("current close must not be empty")
    if not np.isfinite(current).all():
        raise ValueError("current close must contain only finite values")
    return current.copy()


def mae_ratio(model_mae: float, baseline_mae: float) -> float:
    """How many times the persistence error the model error is."""
    model_mae = float(model_mae)
    baseline_mae = float(baseline_mae)
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae):
        raise ValueError("MAE values must be finite")
    if baseline_mae <= 0:
        raise ValueError("baseline MAE must be positive")
    return model_mae / baseline_mae


def dumps_payload(payload: dict) -> str:
    """Serialize compact JSON and refuse a file over the page budget."""
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    size = len(text.encode("utf-8"))
    if size > MAX_JSON_BYTES:
        raise RuntimeError(
            f"demo JSON is {size} bytes, over the {MAX_JSON_BYTES} byte limit"
        )
    return text


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"invalid date {value!r}")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.date()


def _iso(value) -> str:
    return _as_date(value).isoformat()


def _round_list(values, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def _metric_block(metrics: dict, baseline_mae: float | None = None) -> dict:
    block = {
        "mae": round(float(metrics["MAE"]), 4),
        "rmse": round(float(metrics["RMSE"]), 4),
    }
    if baseline_mae is not None:
        block["mae_ratio"] = round(mae_ratio(metrics["MAE"], baseline_mae), 4)
    return block


def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _load_features(ticker: str):
    from src.data_loader import fetch_stock_data, prepare_forecast_data
    from src.sentiment_analyzer import add_sentiment_to_df

    last_error = None
    df = None
    for attempt in range(3):
        try:
            df = fetch_stock_data(ticker, START, END)
            break
        except Exception as exc:  # network or empty payload; retry then raise
            last_error = exc
            time.sleep(2 * (attempt + 1))
    if df is None:
        raise RuntimeError(f"failed to download {ticker}") from last_error

    df = _naive_index(df)
    df = add_sentiment_to_df(df, ticker, START, END)
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker} is missing features: {missing}")

    X, y, target_index = prepare_forecast_data(
        df, FEATURE_COLS, target_col="Close", horizon=HORIZON
    )
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{ticker} features contain non-finite values")

    current_close = df.loc[X.index, "Close"].to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    if len(current_close) != len(y_values):
        raise RuntimeError(f"{ticker} target alignment length mismatch")
    return df, X, y_values, target_index, current_close


def _run_backtest(df: pd.DataFrame, feature_index, predictions, current_close, target_index) -> dict:
    """Long/flat Backtrader curve for one out-of-sample fold, plus buy-and-hold."""
    import backtrader as bt

    from src.backtesting import (
        buy_and_hold_equity_values,
        commission_aware_position_size,
        is_terminal_order,
        next_day_direction_signals,
        portfolio_value_series,
    )

    predictions = np.asarray(predictions, dtype=float)
    current_close = np.asarray(current_close, dtype=float)
    signals = next_day_direction_signals(predictions, current_close)
    signal_by_date = {
        _as_date(ts): int(signal) for ts, signal in zip(feature_index, signals)
    }

    start = feature_index[0]
    end = target_index[-1]
    test_df = df.loc[start:end, ["Open", "High", "Low", "Close", "Volume"]].copy()
    if test_df.empty:
        raise RuntimeError("backtest window is empty")
    if not np.isfinite(test_df.to_numpy(dtype=float)).all():
        raise RuntimeError("backtest prices are not finite")

    class DateSignalStrategy(bt.Strategy):
        """Trade the signal stored for this bar's date.

        A pending order still blocks a new order on the current bar. The next
        bar reads its own date, so the skipped signal is not reused later.
        """

        params = dict(signal_by_date=None, commission_rate=0.0)

        def __init__(self):
            self.order = None
            self.portfolio_values = []
            self.bar_dates = []

        def next(self):
            self.portfolio_values.append(float(self.broker.getvalue()))
            bar_date = _as_date(self.data.datetime.date(0))
            self.bar_dates.append(bar_date)
            if self.order:
                return

            signal = self.p.signal_by_date.get(bar_date, -1)
            if signal == 1 and not self.position:
                size = commission_aware_position_size(
                    self.broker.getcash(),
                    float(self.data.close[0]),
                    self.p.commission_rate,
                )
                if size > 0:
                    self.order = self.buy(size=size)
            elif signal == 0 and self.position:
                self.order = self.sell(size=self.position.size)

        def notify_order(self, order):
            if is_terminal_order(order):
                self.order = None

    bt_data = test_df.copy()
    bt_data.index = pd.to_datetime(bt_data.index)
    bt_data.columns = ["open", "high", "low", "close", "volume"]
    feed = bt.feeds.PandasData(dataname=bt_data)

    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(
        DateSignalStrategy,
        signal_by_date=signal_by_date,
        commission_rate=COMMISSION_RATE,
    )
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION_RATE)
    results = cerebro.run()
    strategy = results[0]

    price_dates = [_as_date(ts) for ts in test_df.index]
    if strategy.bar_dates != price_dates:
        raise RuntimeError(
            "Backtrader bar dates did not match the price index "
            f"({len(strategy.bar_dates)} bars vs {len(price_dates)} prices)"
        )

    equity = portfolio_value_series(strategy.portfolio_values, test_df.index)
    buy_hold = buy_and_hold_equity_values(
        test_df["Open"].to_numpy(dtype=float),
        test_df["Close"].to_numpy(dtype=float),
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_RATE,
        entry_index=1,
    )
    if len(buy_hold) != len(equity):
        raise RuntimeError("buy-and-hold length does not match the strategy curve")

    matched = sum(1 for bar_date in price_dates if bar_date in signal_by_date)
    if matched == 0:
        raise RuntimeError("no backtest bar matched a signal date")

    strategy_final = round(float(equity.iloc[-1]), 2)
    buy_hold_final = round(float(buy_hold[-1]), 2)
    return {
        "ok": True,
        "start": _iso(test_df.index[0]),
        "end": _iso(test_df.index[-1]),
        "initial_cash": INITIAL_CASH,
        "commission_rate": COMMISSION_RATE,
        "bars": int(len(equity)),
        "signals_matched": int(matched),
        "dates": [_iso(ts) for ts in equity.index],
        "strategy": _round_list(equity.to_numpy(), 2),
        "buy_hold": _round_list(buy_hold, 2),
        "strategy_final": strategy_final,
        "buy_hold_final": buy_hold_final,
        "strategy_return": round(strategy_final / INITIAL_CASH - 1.0, 4),
        "buy_hold_return": round(buy_hold_final / INITIAL_CASH - 1.0, 4),
    }


def _tensorflow_loaded() -> bool:
    import sys

    module = sys.modules.get("tensorflow")
    return module is not None and getattr(module, "__file__", None) is not None


def _train_random_forest_regressor():
    """Return the repository trainer, without requiring TensorFlow to be installed."""
    try:
        from src.model_trainer import train_random_forest_regressor
    except ModuleNotFoundError as exc:
        if exc.name != "tensorflow" and not str(exc.name).startswith("tensorflow."):
            raise
        import sys

        sys.modules.pop("src.model_trainer", None)
        _allow_trainer_import_without_tensorflow()
        from src.model_trainer import train_random_forest_regressor
    return train_random_forest_regressor


def _allow_trainer_import_without_tensorflow() -> None:
    import sys
    import types

    if "src.model_trainer" in sys.modules:
        return

    def ensure(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    tensorflow = ensure("tensorflow")
    keras = ensure("tensorflow.keras")
    models = ensure("tensorflow.keras.models")
    layers = ensure("tensorflow.keras.layers")
    callbacks = ensure("tensorflow.keras.callbacks")
    optimizers = ensure("tensorflow.keras.optimizers")
    tensorflow.keras = keras
    keras.models = models
    keras.layers = layers
    keras.callbacks = callbacks
    keras.optimizers = optimizers
    models.Sequential = object
    models.load_model = object
    layers.LSTM = object
    layers.Dense = object
    layers.Dropout = object
    layers.BatchNormalization = object
    callbacks.EarlyStopping = object
    callbacks.ReduceLROnPlateau = object
    callbacks.ModelCheckpoint = object
    optimizers.Adam = object


def evaluate_ticker(ticker: str) -> dict:
    from src.evaluator import regression_metrics
    from src.validation import make_forecast_time_series_split

    train_random_forest_regressor = _train_random_forest_regressor()

    df, X, y_values, target_index, current_close = _load_features(ticker)
    splitter = make_forecast_time_series_split(
        n_splits=N_SPLITS, forecast_horizon=HORIZON
    )

    folds = []
    actual_parts = []
    pred_parts = []
    base_parts = []
    series_dates: list[str] = []
    last_fold = None
    model_params = None

    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        embargo = int(val_idx[0] - train_idx[-1] - 1)
        model = train_random_forest_regressor(
            X.iloc[train_idx],
            y_values[train_idx],
            tune=False,
            forecast_horizon=HORIZON,
        )
        if model_params is None:
            model_params = {
                "n_estimators": int(model.n_estimators),
                "max_depth": None if model.max_depth is None else int(model.max_depth),
                "random_state": int(model.random_state),
            }

        predicted = np.asarray(model.predict(X.iloc[val_idx]), dtype=float)
        actual = y_values[val_idx]
        baseline = persistence_forecast(current_close[val_idx])
        model_metrics = regression_metrics(
            actual, predicted, f"{ticker} fold {fold_number} random forest"
        )
        baseline_metrics = regression_metrics(
            actual, baseline, f"{ticker} fold {fold_number} persistence"
        )
        target_dates = pd.DatetimeIndex(target_index[val_idx])
        folds.append(
            {
                "fold": fold_number,
                "start": _iso(target_dates[0]),
                "end": _iso(target_dates[-1]),
                "n": int(len(val_idx)),
                "embargo_rows": embargo,
                "model": _metric_block(model_metrics, baseline_metrics["MAE"]),
                "baseline": _metric_block(baseline_metrics),
            }
        )
        actual_parts.append(actual)
        pred_parts.append(predicted)
        base_parts.append(baseline)
        series_dates.extend(_iso(ts) for ts in target_dates)
        last_fold = {
            "fold": fold_number,
            "feature_index": X.index[val_idx],
            "predictions": predicted,
            "current_close": current_close[val_idx],
            "target_index": target_dates,
        }

    actual_all = np.concatenate(actual_parts)
    pred_all = np.concatenate(pred_parts)
    base_all = np.concatenate(base_parts)
    if not (len(series_dates) == len(actual_all) == len(pred_all) == len(base_all)):
        raise RuntimeError(f"{ticker} forecast series lengths do not match")
    overall_model = regression_metrics(actual_all, pred_all, f"{ticker} overall random forest")
    overall_base = regression_metrics(actual_all, base_all, f"{ticker} overall persistence")
    backtest = _run_backtest(
        df,
        last_fold["feature_index"],
        last_fold["predictions"],
        last_fold["current_close"],
        last_fold["target_index"],
    )
    backtest["fold"] = last_fold["fold"]

    ratios = [fold["model"]["mae_ratio"] for fold in folds]
    worst_offset = int(np.argmax(ratios))
    return {
        "symbol": ticker,
        "rows": int(len(df)),
        "last_bar": _iso(df.index[-1]),
        "model_params": model_params,
        "series": {
            "dates": series_dates,
            "actual": _round_list(actual_all, 4),
            "forecast": _round_list(pred_all, 4),
            "baseline": _round_list(base_all, 4),
        },
        "folds": folds,
        "overall": {
            "n": int(len(actual_all)),
            "model": _metric_block(overall_model, overall_base["MAE"]),
            "baseline": _metric_block(overall_base),
        },
        "worst_fold": folds[worst_offset]["fold"],
        "worst_fold_is_last": worst_offset == len(folds) - 1,
        "backtest": backtest,
    }


def build_payload(tickers: tuple[str, ...] = TICKERS) -> dict:
    import sklearn

    results = [evaluate_ticker(ticker) for ticker in tickers]
    data_as_of = max(item["last_bar"] for item in results)
    params = results[0]["model_params"]
    return {
        "generated_on": date.today().isoformat(),
        "data_as_of": data_as_of,
        "study": {
            "start": START,
            "end_exclusive": END,
            "horizon_days": HORIZON,
            "n_splits": N_SPLITS,
            "model": "RandomForestRegressor",
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "random_state": params["random_state"],
            "tuned": False,
            "lstm_included": False,
            "target": "Close[t+1]",
            "baseline": "persistence",
            "baseline_definition": "Close[t] as the forecast of Close[t+1]",
            "sentiment": "neutral",
            "initial_cash": INITIAL_CASH,
            "commission_rate": COMMISSION_RATE,
            "trainer": "src.model_trainer.train_random_forest_regressor",
            "splitter": "src.validation.make_forecast_time_series_split",
            "metrics": "src.evaluator.regression_metrics",
            "lstm_trained": False,
            "tensorflow_loaded": _tensorflow_loaded(),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "tickers": results,
    }


def main() -> None:
    payload = build_payload()
    text = dumps_payload(payload)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    size = OUTPUT_PATH.stat().st_size
    print(f"\nWrote {OUTPUT_PATH} ({size} bytes)")
    for item in payload["tickers"]:
        model_mae = item["overall"]["model"]["mae"]
        base_mae = item["overall"]["baseline"]["mae"]
        ratio = item["overall"]["model"]["mae_ratio"]
        bt = item["backtest"]
        print(
            f"  {item['symbol']}: model MAE {model_mae:.4f} | "
            f"persistence MAE {base_mae:.4f} | ratio {ratio:.4f} | "
            f"strategy ${bt['strategy_final']:.2f} vs buy-hold ${bt['buy_hold_final']:.2f}"
        )


if __name__ == "__main__":
    main()
