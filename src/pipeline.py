"""Reproducible next-day forecast run: baselines, walk-forward, and charts.

Run from the repository root:

    python -m src.pipeline
    python -m src.pipeline --demo
    python -m src.pipeline --lstm --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting import (
    assemble_signal_frame,
    buy_and_hold_equity_values,
    equity_from_period_returns,
    equity_period_returns,
    next_day_direction_signals,
    next_open_long_flat_returns,
    portfolio_value_series,
)
from src.baselines import (
    directional_accuracy,
    drop_constant_features,
    mae_skill_score,
    majority_class_accuracy,
    persistence_predictions,
    regression_error_scores,
)
from src.data_loader import (
    add_technical_indicators,
    build_holdout_sequences,
    build_sequences,
    fetch_stock_data,
    prepare_forecast_data,
    time_series_split,
)
from src.evaluator import (
    classification_metrics,
    max_drawdown,
    CHART_ACCENT,
    CHART_ACCENT_SOFT,
    CHART_SECONDARY,
    plot_confusion_matrix,
    plot_equity_curve,
    plot_feature_importance,
    plot_loss_curves,
    plot_predictions,
    save_chart,
    style_chart,
    sharpe_ratio,
)
from src.feature_store import (
    load_dataset_meta,
    load_feature_frame,
    monthly_close_summary,
    save_feature_frame,
)
from src.model_trainer import (
    train_random_forest_classifier,
    train_random_forest_regressor,
)
from src.sentiment_analyzer import add_sentiment_to_df
from src.validation import forecast_split_leaks, iter_expanding_forecast_splits

RF_FEATURE_COLUMNS = [
    "Open", "High", "Low", "Volume",
    "SMA_10", "SMA_20", "SMA_50",
    "RSI_14", "RSI_7", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "ATR_14", "Vol_Change", "OBV",
    "Log_Return", "Pct_Change", "Sentiment",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3",
    "Close_Lag_5", "Close_Lag_10",
]
LSTM_SEQ_LEN = 60
REPO_ROOT = Path(__file__).resolve().parents[1]


class DataFetchError(RuntimeError):
    """Raised when live market data cannot be downloaded."""


@dataclass
class PipelineConfig:
    """Inputs for one reproducible forecast run."""

    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    demo: bool = False
    refresh: bool = False
    tune: bool = False
    lstm: bool = False
    epochs: int = 15
    n_estimators: int = 200
    max_depth: int = 20
    n_splits: int = 4
    n_jobs: int = 1
    commission_rate: float = 0.001
    initial_cash: float = 10_000.0
    risk_free_rate: float = 0.04
    random_state: int = 42
    output_dir: Path = REPO_ROOT / "results"
    database_path: Path = REPO_ROOT / "data" / "market.sqlite"


def command_line(config: PipelineConfig) -> str:
    """Return the CLI that reproduces this configuration."""
    parts = [
        "python -m src.pipeline",
        "--ticker", config.ticker,
        "--start", config.start,
        "--end", config.end,
    ]
    if config.demo:
        parts.append("--demo")
    if config.refresh:
        parts.append("--refresh")
    if config.tune:
        parts.append("--tune")
    if config.lstm:
        parts.extend(["--lstm", "--epochs", str(config.epochs)])
    if config.n_estimators != 200:
        parts.extend(["--n-estimators", str(config.n_estimators)])
    if config.n_splits != 4:
        parts.extend(["--n-splits", str(config.n_splits)])
    return " ".join(parts)


def synthetic_ohlcv(periods: int = 420, seed: int = 7) -> pd.DataFrame:
    """Build a seeded random-walk OHLCV frame for the offline smoke test.

    These rows are not market data and must not be quoted as forecast results.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=periods)
    returns = rng.normal(0.0004, 0.012, periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, periods))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, periods))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, periods))
    volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    frame = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    frame.index.name = "Date"
    return frame


def _sentiment_mode() -> str:
    return "newsapi" if os.getenv("NEWS_API_KEY") else "neutral"


def _load_market_frame(config: PipelineConfig) -> tuple[pd.DataFrame, str]:
    sentiment_mode = "neutral" if config.demo else _sentiment_mode()
    ticker = "DEMO" if config.demo else config.ticker

    if not config.demo and not config.refresh:
        meta = load_dataset_meta(config.database_path, ticker)
        if (
            meta
            and meta["request_start"] == config.start
            and meta["request_end"] == config.end
            and meta["sentiment_mode"] == sentiment_mode
        ):
            cached = load_feature_frame(
                config.database_path, ticker, config.start, config.end
            )
            if not cached.empty:
                print(f"[Pipeline] Loaded {len(cached)} cached rows from {config.database_path}")
                return cached, sentiment_mode

    if config.demo:
        frame = add_technical_indicators(synthetic_ohlcv())
        frame["Sentiment"] = 0.0
    else:
        try:
            frame = fetch_stock_data(config.ticker, config.start, config.end)
            frame = add_sentiment_to_df(frame, config.ticker, config.start, config.end)
        except Exception as exc:
            raise DataFetchError(
                "Could not download market data. Check the ticker, date range, and "
                "network connection. Run `python -m src.pipeline --demo` for an "
                "offline smoke test that does not use market prices."
            ) from exc

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise DataFetchError("Feature engineering removed every row.")

    save_feature_frame(
        frame,
        config.database_path,
        ticker,
        config.start,
        config.end,
        sentiment_mode,
    )
    months = monthly_close_summary(config.database_path, ticker)
    print(
        f"[Pipeline] Stored {len(frame)} rows in SQLite "
        f"({len(months)} calendar months)."
    )
    return frame, sentiment_mode


def _score_fold(y_true: np.ndarray, y_pred: np.ndarray, current_close: np.ndarray) -> dict:
    model_scores = regression_error_scores(y_true, y_pred)
    baseline_pred = persistence_predictions(current_close)
    baseline_scores = regression_error_scores(y_true, baseline_pred)
    return {
        "model_mae": model_scores["MAE"],
        "model_rmse": model_scores["RMSE"],
        "model_mape_pct": model_scores["MAPE%"],
        "persistence_mae": baseline_scores["MAE"],
        "persistence_rmse": baseline_scores["RMSE"],
        "persistence_mape_pct": baseline_scores["MAPE%"],
        "mae_skill_score": mae_skill_score(model_scores["MAE"], baseline_scores["MAE"]),
        "direction_accuracy": directional_accuracy(y_true, y_pred, current_close),
    }


def _mean_key(folds: list[dict], key: str) -> float:
    return float(np.mean([fold[key] for fold in folds]))


def _inverse_close(scaled_values: np.ndarray, scaler, close_col_idx: int, n_cols: int) -> np.ndarray:
    dummy = np.zeros((len(scaled_values), n_cols))
    dummy[:, close_col_idx] = np.asarray(scaled_values, dtype=float).reshape(-1)
    return scaler.inverse_transform(dummy)[:, close_col_idx]


def _run_lstm(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_dates: pd.Index,
    config: PipelineConfig,
    output_dir: Path,
) -> dict:
    """Train an LSTM on history before ``target_dates`` and score those dates."""
    try:
        from src.model_trainer import train_lstm
        from src.scaling import scale_time_series_partitions
    except ModuleNotFoundError as exc:
        return {"status": "unavailable", "reason": str(exc)}

    history = df.loc[df.index < target_dates[0]]
    test_rows = df.reindex(target_dates)
    if history.empty or test_rows.isna().any().any():
        return {"status": "skipped", "reason": "Holdout dates are not present in the feature frame"}
    if len(history) <= LSTM_SEQ_LEN + 20:
        return {"status": "skipped", "reason": "Not enough history for an LSTM lookback of 60 days"}

    fit_rows, validation_rows = time_series_split(history, 0.9)
    try:
        fit_scaled, val_scaled, test_scaled, scaler = scale_time_series_partitions(
            fit_rows, validation_rows, test_rows, feature_columns, target_col="Close"
        )
    except (ValueError, KeyError) as exc:
        return {"status": "skipped", "reason": str(exc)}

    target_idx = len(feature_columns)
    X_train, y_train = build_sequences(fit_scaled, LSTM_SEQ_LEN, target_idx)
    X_val, y_val = build_holdout_sequences(fit_scaled, val_scaled, LSTM_SEQ_LEN, target_idx)
    X_test, y_test = build_holdout_sequences(
        np.concatenate([fit_scaled, val_scaled], axis=0),
        test_scaled,
        LSTM_SEQ_LEN,
        target_idx,
    )
    if min(len(y_train), len(y_val), len(y_test)) == 0:
        return {"status": "skipped", "reason": "Sequence builder returned an empty split"}

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)
    model, history_obj = train_lstm(
        X_train, y_train, X_val, y_val,
        epochs=config.epochs,
        save_path=str(output_dir / "lstm_model.keras"),
    )
    n_cols = len(feature_columns) + 1
    predicted = _inverse_close(model.predict(X_test, verbose=0).reshape(-1), scaler, target_idx, n_cols)
    actual = _inverse_close(y_test, scaler, target_idx, n_cols)
    plot_loss_curves(
        history_obj, show=False, save_path=str(output_dir / "lstm_loss_curves.png")
    )
    plot_predictions(
        actual, predicted, label="LSTM", dates=target_dates,
        show=False, save_path=str(output_dir / "lstm_predictions.png"),
    )
    return {
        "status": "trained",
        "epochs_requested": config.epochs,
        "epochs_ran": len(history_obj.history["loss"]),
        "predictions": predicted,
        "actual": actual,
    }


def _plot_eda(df: pd.DataFrame, ticker: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    axes[0].plot(df.index, df["Close"], color=CHART_SECONDARY, label="Close", linewidth=1.2)
    if "SMA_50" in df.columns:
        axes[0].plot(df.index, df["SMA_50"], color=CHART_ACCENT, label="SMA 50", linewidth=1.0)
    axes[0].set_title(f"{ticker} close")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[1].bar(df.index, df["Volume"], color=CHART_ACCENT, width=1.0)
    axes[1].set_ylabel("Volume")
    axes[2].hist(df["Log_Return"].dropna(), bins=80, color=CHART_ACCENT)
    axes[2].set_title("Log-return distribution")
    axes[3].plot(df.index, df["RSI_14"], color=CHART_ACCENT, linewidth=1.0)
    axes[3].axhline(70, color=CHART_ACCENT_SOFT, linestyle="--", linewidth=0.8)
    axes[3].axhline(30, color=CHART_ACCENT_SOFT, linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("RSI 14")
    style_chart(fig, axes)
    fig.tight_layout()
    save_chart(fig, path)
    plt.close(fig)


def _plot_baseline_bars(comparison: dict[str, dict], path: Path) -> None:
    names = list(comparison)
    mae = [comparison[name]["MAE"] for name in names]
    rmse = [comparison[name]["RMSE"] for name in names]
    positions = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(positions - width / 2, mae, width, label="MAE", color=CHART_ACCENT)
    ax.bar(positions + width / 2, rmse, width, label="RMSE", color=CHART_ACCENT_SOFT)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel("USD")
    ax.set_title("Final holdout price error versus a no-change forecast")
    ax.legend()
    style_chart(fig, ax)
    fig.tight_layout()
    save_chart(fig, path)
    plt.close(fig)


def _performance_block(equity: pd.Series, initial_cash: float, risk_free_rate: float) -> dict:
    returns = equity_period_returns(equity)
    final_value = float(equity.iloc[-1])
    return {
        "initial_cash": initial_cash,
        "final_value": final_value,
        "total_return": final_value / initial_cash - 1.0,
        "sharpe": float(sharpe_ratio(returns, risk_free_rate=risk_free_rate, verbose=False)),
        "max_drawdown": float(max_drawdown(equity.to_numpy(), verbose=False)),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def run_pipeline(config: PipelineConfig) -> dict:
    """Fit the walk-forward study and write metrics plus charts."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, sentiment_mode = _load_market_frame(config)
    ticker = "DEMO" if config.demo else config.ticker
    if config.demo:
        print("[Pipeline] DEMO data: results below are a synthetic smoke test, not market performance.")

    candidates = [column for column in RF_FEATURE_COLUMNS if column in frame.columns]
    feature_columns, dropped = drop_constant_features(frame, candidates)
    X, y, target_index = prepare_forecast_data(frame, feature_columns, target_col="Close", horizon=1)
    current_close = frame.loc[X.index, "Close"].to_numpy(dtype=float)
    splits = iter_expanding_forecast_splits(len(X), n_splits=config.n_splits, gap=1)

    fold_reports = []
    final = None
    for fold_number, (train_end, test_start, test_end) in enumerate(splits, start=1):
        if forecast_split_leaks(X.index, target_index, train_end, test_start):
            raise RuntimeError(
                f"Walk-forward fold {fold_number} would train on a close that "
                "is already visible in the test features."
            )
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end].to_numpy(dtype=float)
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end].to_numpy(dtype=float)
        current_test = current_close[test_start:test_end]
        model = train_random_forest_regressor(
            X_train, y_train,
            tune=config.tune,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            n_jobs=config.n_jobs,
            forecast_horizon=1,
        )
        predicted = model.predict(X_test)
        scores = _score_fold(y_test, predicted, current_test)
        scores.update({
            "fold": fold_number,
            "train_rows": int(train_end),
            "test_rows": int(test_end - test_start),
            "test_start": pd.Timestamp(target_index[test_start]).strftime("%Y-%m-%d"),
            "test_end": pd.Timestamp(target_index[test_end - 1]).strftime("%Y-%m-%d"),
        })
        fold_reports.append(scores)
        final = {
            "model": model,
            "predicted": predicted,
            "y_test": y_test,
            "current_test": current_test,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        }
        print(
            f"[Pipeline] Fold {fold_number}: RF MAE {scores['model_mae']:.4f} | "
            f"persistence MAE {scores['persistence_mae']:.4f} | "
            f"skill {scores['mae_skill_score']:.4f}"
        )

    assert final is not None
    test_start = final["test_start"]
    test_end = final["test_end"]
    feature_index = X.index[test_start:test_end]
    holdout_targets = target_index[test_start:test_end]
    y_train_dir = (final["y_train"] > current_close[: final["train_end"]]).astype(int)
    y_test_dir = (final["y_test"] > final["current_test"]).astype(int)
    classifier = train_random_forest_classifier(
        final["X_train"], y_train_dir,
        tune=False,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        n_jobs=config.n_jobs,
    )
    predicted_dir = classifier.predict(final["X_test"])
    class_scores = classification_metrics(y_test_dir, predicted_dir, verbose=False)
    majority_accuracy, majority_label = majority_class_accuracy(y_test_dir, y_train_dir)

    signals = next_day_direction_signals(final["predicted"], final["current_test"])
    backtest_frame = assemble_signal_frame(
        frame, feature_index, holdout_targets, final["predicted"], signals
    )
    strategy_returns = next_open_long_flat_returns(
        signals.astype(float),
        frame.loc[holdout_targets, "Open"].to_numpy(dtype=float),
        final["y_test"],
        commission_rate=config.commission_rate,
    )
    strategy_equity = equity_from_period_returns(
        strategy_returns, config.initial_cash, backtest_frame.index
    )
    buy_hold_values = buy_and_hold_equity_values(
        backtest_frame["Open"].to_numpy(dtype=float),
        backtest_frame["Close"].to_numpy(dtype=float),
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        entry_index=1,
    )
    buy_hold_equity = portfolio_value_series(
        buy_hold_values, backtest_frame.index, name="Buy & Hold"
    )

    _plot_eda(frame, ticker, output_dir / "eda_dashboard.png")
    plot_predictions(
        final["y_test"], final["predicted"], label="Random Forest",
        dates=holdout_targets, show=False,
        save_path=str(output_dir / "rf_predictions.png"),
    )
    plot_feature_importance(
        final["model"], feature_columns, top_n=min(15, len(feature_columns)),
        show=False, save_path=str(output_dir / "rf_feature_importance.png"),
    )
    plot_confusion_matrix(
        y_test_dir, predicted_dir, label="RF Classifier",
        show=False, save_path=str(output_dir / "rf_confusion_matrix.png"),
    )
    plot_equity_curve(
        strategy_equity, label="ML Strategy", benchmark=buy_hold_equity,
        show=False, save_path=str(output_dir / "equity_curve.png"),
    )

    model_error = regression_error_scores(final["y_test"], final["predicted"])
    persistence_error = regression_error_scores(
        final["y_test"], persistence_predictions(final["current_test"])
    )
    comparison = {
        "Persistence": persistence_error,
        "Random Forest": model_error,
    }

    lstm_report = {"status": "skipped"}
    if config.lstm and not config.demo:
        print("[Pipeline] Training LSTM on the final holdout targets...")
        lstm_report = _run_lstm(frame, feature_columns, holdout_targets, config, output_dir)
        if lstm_report.get("status") == "trained":
            if len(lstm_report["predictions"]) != len(final["y_test"]):
                lstm_report = {
                    "status": "failed",
                    "reason": "LSTM prediction length did not match the holdout",
                }
            else:
                lstm_error = regression_error_scores(final["y_test"], lstm_report["predictions"])
                comparison["LSTM"] = lstm_error
                lstm_report = {
                    "status": "trained",
                    "epochs_requested": lstm_report["epochs_requested"],
                    "epochs_ran": lstm_report["epochs_ran"],
                    "MAE": lstm_error["MAE"],
                    "RMSE": lstm_error["RMSE"],
                    "MAPE_pct": lstm_error["MAPE%"],
                    "direction_accuracy": directional_accuracy(
                        final["y_test"], lstm_report["predictions"], final["current_test"]
                    ),
                    "mae_skill_score": mae_skill_score(
                        lstm_error["MAE"], persistence_error["MAE"]
                    ),
                }
        else:
            print(f"[Pipeline] LSTM {lstm_report.get('status')}: {lstm_report.get('reason')}")

    _plot_baseline_bars(comparison, output_dir / "baseline_comparison.png")

    metrics = {
        "command": command_line(config),
        "demo": config.demo,
        "ticker": ticker,
        "start": config.start,
        "end": config.end,
        "sentiment_mode": sentiment_mode,
        "dropped_constant_features": dropped,
        "feature_columns": feature_columns,
        "n_feature_rows": int(len(frame)),
        "n_supervised_rows": int(len(X)),
        "disclaimer": "Research output. Not financial advice.",
        "walk_forward": {
            "n_splits": config.n_splits,
            "gap": 1,
            "folds": fold_reports,
            "mean_model_mae": _mean_key(fold_reports, "model_mae"),
            "mean_persistence_mae": _mean_key(fold_reports, "persistence_mae"),
            "mean_model_rmse": _mean_key(fold_reports, "model_rmse"),
            "mean_persistence_rmse": _mean_key(fold_reports, "persistence_rmse"),
            "mean_mae_skill_score": _mean_key(fold_reports, "mae_skill_score"),
            "mean_direction_accuracy": _mean_key(fold_reports, "direction_accuracy"),
        },
        "final_holdout": {
            "start": fold_reports[-1]["test_start"],
            "end": fold_reports[-1]["test_end"],
            "n": fold_reports[-1]["test_rows"],
            "random_forest": {
                "MAE": model_error["MAE"],
                "RMSE": model_error["RMSE"],
                "MAPE_pct": model_error["MAPE%"],
                "direction_accuracy": directional_accuracy(
                    final["y_test"], final["predicted"], final["current_test"]
                ),
                "mae_skill_score": mae_skill_score(model_error["MAE"], persistence_error["MAE"]),
            },
            "persistence": {
                "MAE": persistence_error["MAE"],
                "RMSE": persistence_error["RMSE"],
                "MAPE_pct": persistence_error["MAPE%"],
            },
            "direction_classifier": {
                **class_scores,
                "majority_baseline_accuracy": majority_accuracy,
                "majority_label": majority_label,
            },
            "strategy": {
                **_performance_block(strategy_equity, config.initial_cash, config.risk_free_rate),
                "engine": "next_open_long_flat_returns",
                "commission_rate": config.commission_rate,
            },
            "buy_and_hold": _performance_block(
                buy_hold_equity, config.initial_cash, config.risk_free_rate
            ),
            "lstm": lstm_report,
        },
        "config": {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "tune": config.tune,
            "n_jobs": config.n_jobs,
            "lstm": config.lstm,
            "epochs": config.epochs if config.lstm else None,
            "random_state": config.random_state,
        },
    }
    payload = _json_ready(metrics)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[Pipeline] Wrote {metrics_path}")
    _print_summary(payload)
    return payload


def _print_summary(metrics: dict) -> None:
    walk = metrics["walk_forward"]
    holdout = metrics["final_holdout"]
    print("\nWalk-forward mean MAE")
    print(f"  Random Forest : {walk['mean_model_mae']:.4f}")
    print(f"  Persistence   : {walk['mean_persistence_mae']:.4f}")
    print(f"  Skill score   : {walk['mean_mae_skill_score']:.4f}")
    strategy = holdout["strategy"]
    benchmark = holdout["buy_and_hold"]
    print("\nFinal holdout backtest")
    print(f"  Strategy      : {strategy['total_return']:.2%}  Sharpe {strategy['sharpe']:.3f}")
    print(f"  Buy & hold    : {benchmark['total_return']:.2%}  Sharpe {benchmark['sharpe']:.3f}")
    classifier = holdout["direction_classifier"]
    print(
        f"  Direction clf : {classifier['Accuracy']:.2%} vs "
        f"majority baseline {classifier['majority_baseline_accuracy']:.2%}"
    )


def config_from_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Run the next-day forecast study and write results/."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--demo", action="store_true",
                        help="Run on seeded synthetic prices instead of downloading data.")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the SQLite cache and download again.")
    parser.add_argument("--tune", action="store_true",
                        help="Grid-search the random forest inside each walk-forward fold.")
    parser.add_argument("--lstm", action="store_true",
                        help="Also train the LSTM on the final holdout. Requires TensorFlow.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--database", type=Path, default=REPO_ROOT / "data" / "market.sqlite")
    args = parser.parse_args(argv)
    return PipelineConfig(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        demo=args.demo,
        refresh=args.refresh,
        tune=args.tune,
        lstm=args.lstm,
        epochs=args.epochs,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
        database_path=args.database,
    )


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(argv)
    try:
        run_pipeline(config)
    except DataFetchError as exc:
        print(f"[Pipeline] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
