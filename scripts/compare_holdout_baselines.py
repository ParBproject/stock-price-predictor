"""Score persistence, LSTM, and Random Forest on one chronological holdout.

The holdout is the shifted 80/20 sample boundary used by
``notebooks/random_forest_model.ipynb`` and ``notebooks/backtesting.ipynb``:
features at date ``t`` predict ``Close[t + 1]``. When an LSTM artifact is
available, its test dates are intersected with that boundary so all three
forecasts are scored on the same sessions.

A same-day Random Forest, labeled leaky, is fit on ``Close[t]`` from features
at ``t``. That is the leakage case in the README, not a forecast.

Run from the repository root::

    python scripts/compare_holdout_baselines.py

Existing ``results/rf_regressor.pkl`` and ``results/lstm_model.keras`` files
are scored when their saved data fingerprint matches the frame just loaded.
Otherwise the script trains the same model families the notebooks train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import (  # noqa: E402
    persistence_forecast,
    same_day_feature_target,
    score_forecast,
)
from src.data_loader import (  # noqa: E402
    build_holdout_sequences,
    build_sequences,
    fetch_stock_data,
    prepare_forecast_data,
    time_series_split,
)
from src.model_artifacts import validate_sklearn_feature_schema  # noqa: E402
from src.scaling import scale_time_series_partitions  # noqa: E402
from src.sentiment_analyzer import add_sentiment_to_df  # noqa: E402

TICKER = "AAPL"
START = "2015-01-01"
END = "2024-12-31"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
SEQ_LEN = 60
EPOCHS = 50
BATCH_SIZE = 32
FORECAST_HORIZON = 1

RF_FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "SMA_10", "SMA_20", "SMA_50",
    "RSI_14", "RSI_7", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "ATR_14", "Vol_Change", "OBV",
    "Log_Return", "Pct_Change", "Sentiment",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3",
    "Close_Lag_5", "Close_Lag_10",
]
LSTM_FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "SMA_10", "SMA_20", "SMA_50",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "ATR_14", "Vol_Change", "Log_Return", "Sentiment",
]

RESULTS_DIR = ROOT / "results"
DATA_CSV = ROOT / "data" / "AAPL_features.csv"
RF_PATH = RESULTS_DIR / "rf_regressor.pkl"
LSTM_PATH = RESULTS_DIR / "lstm_model.keras"
LSTM_SCALER_PATH = RESULTS_DIR / "lstm_scaler.joblib"
ARTIFACT_META_PATH = RESULTS_DIR / "holdout_artifact_meta.json"
TABLE_CSV = RESULTS_DIR / "holdout_baseline.csv"
TABLE_MD = RESULTS_DIR / "holdout_baseline.md"
TABLE_JSON = RESULTS_DIR / "holdout_baseline.json"


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash the modeling frame so a saved model is not scored on new prices."""
    numeric = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
    index = df.index.astype("int64").to_numpy()
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, df.columns)).encode())
    digest.update(np.ascontiguousarray(index).tobytes())
    digest.update(numeric.tobytes())
    return digest.hexdigest()


def load_saved_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_market_frame(use_cache: bool = False) -> tuple[pd.DataFrame, str]:
    """Download the notebook universe, falling back to a previously saved CSV."""
    if use_cache:
        if not DATA_CSV.exists():
            raise FileNotFoundError(f"No cached feature frame at {DATA_CSV}")
        print(f"[Baseline] Loading cached feature frame {DATA_CSV}")
        return load_saved_frame(), "data/AAPL_features.csv saved from yfinance"
    try:
        frame = fetch_stock_data(TICKER, START, END)
        frame = add_sentiment_to_df(frame, TICKER, START, END)
    except Exception as exc:
        if not DATA_CSV.exists():
            raise RuntimeError(
                "Market-data download failed and no committed feature CSV is available"
            ) from exc
        print(f"[Baseline] Download failed ({exc}). Loading {DATA_CSV}.")
        return load_saved_frame(), f"data/AAPL_features.csv after download failure: {exc}"

    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_CSV)
    return frame, "yfinance"


def present_feature_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    columns = [column for column in requested if column in df.columns]
    missing = [column for column in requested if column not in df.columns]
    if missing:
        raise KeyError(f"Modeling frame is missing notebook features: {missing}")
    if "Close" in columns:
        raise ValueError("Close must stay out of the feature list; it is the target")
    return columns


def build_random_forest_samples(df: pd.DataFrame, feature_cols: list[str]):
    """Shifted samples and the 80/20 cut used by the Random Forest notebook."""
    features, target, target_dates = prepare_forecast_data(
        df, feature_cols, target_col="Close", horizon=FORECAST_HORIZON
    )
    split = int(len(features) * TRAIN_RATIO)
    if split <= 0 or split >= len(features):
        raise ValueError("Random Forest split did not leave train and test rows")

    current_close = df.loc[features.index, "Close"].to_numpy(dtype=float)
    return {
        "X_train": features.iloc[:split].copy(),
        "X_test": features.iloc[split:].copy(),
        "y_train": target.iloc[:split].to_numpy(dtype=float),
        "y_test": pd.Series(target.iloc[split:].to_numpy(dtype=float), index=target_dates[split:]),
        "previous_close": pd.Series(current_close[split:], index=target_dates[split:]),
        "target_dates": target_dates[split:],
        "split": split,
        "n_samples": len(features),
    }


def inverse_close(scaled_vals: np.ndarray, scaler, close_col_idx: int, n_cols: int) -> np.ndarray:
    """Invert a single scaled close column, matching the LSTM notebook."""
    dummy = np.zeros((len(scaled_vals), n_cols))
    dummy[:, close_col_idx] = np.asarray(scaled_vals, dtype=float)
    restored = scaler.inverse_transform(dummy)
    return restored[:, close_col_idx]


def lstm_partitions(df: pd.DataFrame, feature_cols: list[str]):
    """Fit / validation / test partitions from the LSTM notebook."""
    train_df, test_df = time_series_split(df, TRAIN_RATIO)
    fit_df, val_df = time_series_split(train_df, 1.0 - VAL_RATIO)
    fit_scaled, val_scaled, test_scaled, scaler = scale_time_series_partitions(
        fit_df, val_df, test_df, feature_cols, target_col="Close"
    )
    target_idx = len(feature_cols)
    observed_train_scaled = np.concatenate([fit_scaled, val_scaled], axis=0)
    X_train, y_train = build_sequences(fit_scaled, SEQ_LEN, target_idx)
    X_val, y_val = build_holdout_sequences(fit_scaled, val_scaled, SEQ_LEN, target_idx)
    X_test, y_test = build_holdout_sequences(
        observed_train_scaled, test_scaled, SEQ_LEN, target_idx
    )
    if len(y_test) != len(test_df) or len(y_val) != len(val_df):
        raise RuntimeError("LSTM holdout sequences did not cover every test row")
    return {
        "fit_df": fit_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_df": train_df,
        "scaler": scaler,
        "target_idx": target_idx,
        "n_cols": len(feature_cols) + 1,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def predictions_in_dollars(partitions: dict) -> pd.Series:
    model = partitions["model"]
    scaled = model.predict(partitions["X_test"], verbose=0).reshape(-1)
    dollars = inverse_close(
        scaled, partitions["scaler"], partitions["target_idx"], partitions["n_cols"]
    )
    return pd.Series(dollars, index=partitions["test_df"].index, name="lstm_pred")


def load_artifact_meta() -> dict | None:
    if not ARTIFACT_META_PATH.exists():
        return None
    return json.loads(ARTIFACT_META_PATH.read_text(encoding="utf-8"))


def save_artifact_meta(payload: dict) -> None:
    ARTIFACT_META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def random_forest_model(samples: dict, fingerprint: str, retrain: bool):
    """Load a fingerprint-matched forest, otherwise train the notebook model."""
    meta = load_artifact_meta() or {}
    fingerprint_ok = meta.get("dataset_fingerprint") == fingerprint
    if RF_PATH.exists() and fingerprint_ok and not retrain:
        model = joblib.load(RF_PATH)
        validate_sklearn_feature_schema(model, samples["X_train"].columns)
        print(f"[Baseline] Loaded Random Forest from {RF_PATH}")
        return model, "loaded"

    print("[Baseline] Training Random Forest with the notebook grid search")
    try:
        from src.model_trainer import train_random_forest_regressor

        model = train_random_forest_regressor(
            samples["X_train"],
            samples["y_train"],
            tune=True,
            n_splits=5,
            save_path=str(RF_PATH),
            forecast_horizon=FORECAST_HORIZON,
        )
    except ImportError as exc:
        print(f"[Baseline] model_trainer import failed ({exc}); fitting sklearn forest directly")
        model = _train_random_forest_like_notebook(
            samples["X_train"], samples["y_train"], str(RF_PATH)
        )
    validate_sklearn_feature_schema(model, samples["X_train"].columns)
    return model, "trained"


def _train_random_forest_like_notebook(X_train: pd.DataFrame, y_train: np.ndarray, save_path: str):
    """Same grid as ``train_random_forest_regressor`` without importing TensorFlow."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    from src.validation import make_forecast_time_series_split

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=make_forecast_time_series_split(n_splits=5, forecast_horizon=FORECAST_HORIZON),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    joblib.dump(model, save_path)
    print(f"[RF] Best params: {search.best_params_}")
    print(f"[RF] Model saved → {save_path}")
    return model


def lstm_model(partitions: dict, fingerprint: str, retrain: bool):
    """Load a fingerprint-matched LSTM and scaler, otherwise train."""
    meta = load_artifact_meta() or {}
    fingerprint_ok = meta.get("dataset_fingerprint") == fingerprint
    scaler_ok = (
        fingerprint_ok
        and LSTM_PATH.exists()
        and LSTM_SCALER_PATH.exists()
        and not retrain
    )
    if scaler_ok:
        from src.model_trainer import load_lstm

        loaded_scaler = joblib.load(LSTM_SCALER_PATH)
        current = partitions["scaler"]
        if not (
            np.allclose(loaded_scaler.data_min_, current.data_min_)
            and np.allclose(loaded_scaler.data_max_, current.data_max_)
        ):
            raise RuntimeError(
                "Saved LSTM scaler does not match the scaler fit on this frame"
            )
        partitions["model"] = load_lstm(str(LSTM_PATH))
        print(f"[Baseline] Loaded LSTM from {LSTM_PATH}")
        return "loaded"

    import random

    import tensorflow as tf

    from src.model_trainer import train_lstm

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    print("[Baseline] Training LSTM with the notebook architecture and split")
    model, _history = train_lstm(
        partitions["X_train"],
        partitions["y_train"],
        partitions["X_val"],
        partitions["y_val"],
        units=64,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_path=str(LSTM_PATH),
    )
    joblib.dump(partitions["scaler"], LSTM_SCALER_PATH)
    partitions["model"] = model
    return "trained"


def slice_to_dates(series: pd.Series, dates: pd.Index) -> np.ndarray:
    values = series.loc[dates]
    if isinstance(values, pd.DataFrame):
        raise ValueError("expected a single series when aligning holdout dates")
    return values.to_numpy(dtype=float)


def score_named(name: str, y_true, y_pred, previous, leaky: bool, sample: str) -> dict:
    scores = score_forecast(y_true, y_pred, previous, label=name)
    return {
        "model": name,
        "sample": sample,
        "leaky": leaky,
        "mae": scores["MAE"],
        "directional_hit_rate": scores["directional_hit_rate"],
    }


def fit_leaky_random_forest(df: pd.DataFrame, feature_cols: list[str], samples: dict, honest_model, dates):
    """Refit the honest forest's hyperparameters on a same-day target.

    Returns the holdout prediction, the holdout closes, and the in-sample
    prediction on the training feature dates. The in-sample score is the
    leakage fit: features at ``t`` already determine ``Close[t]``.
    """
    leaky_model = clone(honest_model)
    train_dates = samples["X_train"].index
    X_train, y_train = same_day_feature_target(df, feature_cols, train_dates)
    X_test, y_test = same_day_feature_target(df, feature_cols, dates)
    overlap = X_train.index.intersection(X_test.index)
    if len(overlap):
        raise RuntimeError("Leaky training dates overlap the holdout dates")
    leaky_model.fit(X_train, y_train)
    previous_train = df.loc[train_dates, "Close_Lag_1"].to_numpy(dtype=float)
    if not np.isfinite(previous_train).all():
        raise RuntimeError("Training rows are missing Close_Lag_1")
    return {
        "holdout_pred": leaky_model.predict(X_test),
        "holdout_actual": y_test.to_numpy(dtype=float),
        "train_pred": leaky_model.predict(X_train),
        "train_actual": y_train.to_numpy(dtype=float),
        "train_previous": previous_train,
        "train_dates": train_dates,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Holdout baseline",
        "",
        report["holdout_definition"],
        "",
        (
            f"Requested window: {report['ticker']} {report['requested_start']} "
            f"to {report['requested_end']} (yfinance `end` is exclusive). "
            f"Feature frame after indicator warm-up: {report['feature_start']} "
            f"to {report['feature_end']} ({report['feature_rows']} rows). "
            f"Data source: {report['data_source']}."
        ),
        "",
        (
            f"Scored sessions: {report['holdout_start']} to {report['holdout_end']} "
            f"({report['n_holdout']} targets). "
            f"Random Forest native holdout: {report['rf_holdout_start']} to "
            f"{report['rf_holdout_end']} ({report['n_rf_holdout']}). "
            f"LSTM test sessions: {report['lstm_test_start']} to "
            f"{report['lstm_test_end']} ({report['n_lstm_test']}). "
            f"Leaky in-sample fit: {report['leaky_train_start']} to "
            f"{report['leaky_train_end']} ({report['n_leaky_train']} training feature dates)."
        ),
        "",
        (
            "MAE is adjusted dollars (`regression_metrics` / "
            "`sklearn.metrics.mean_absolute_error`). "
            "Directional hit rate is the fraction of sessions where the forecast "
            "is on the same side of the previous close as the realized close. "
            "Up means strictly above the previous close. "
            "Persistence sets the next close equal to the previous close, so its "
            "hit rate is the share of holdout sessions that were not up."
        ),
        "",
        "| Model | Sample | MAE | Directional hit rate | Leaky |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['model']} | {row['sample']} | {row['mae']:.6f} | "
            f"{row['directional_hit_rate']:.4f} | {'yes' if row['leaky'] else 'no'} |"
        )
    lines.extend([
        "",
        (
            "Both same-day rows are leaky. They fit the Random Forest family on "
            "`Close[t]` from features at `t`. Those features already determine that "
            "close (`Close_Lag_1 * (1 + Pct_Change)` reconstructs it). "
            "The in-sample row is how good that fit looks on the dates it was trained on. "
            "The holdout row uses the same cheat on the future sessions above and is not a forecast. "
            "A tree still predicts training leaf averages, so a new high in the holdout "
            "is not a small error even when the close is visible in the features."
        ),
        "",
        (
            f"Random Forest artifact: {report['rf_source']} (`results/rf_regressor.pkl`). "
            f"LSTM artifact: {report['lstm_status']} (`results/lstm_model.keras`)."
        ),
    ])
    if report.get("lstm_error"):
        lines.extend(["", "LSTM was not scored:", "", "```", report["lstm_error"], "```"])
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(report["rows"])
    frame.to_csv(TABLE_CSV, index=False)
    TABLE_MD.write_text(render_markdown(report), encoding="utf-8")
    TABLE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Baseline] Wrote {TABLE_CSV}")
    print(f"[Baseline] Wrote {TABLE_MD}")


def shared_dates(rf_dates: pd.Index, lstm_dates: pd.Index | None) -> pd.Index:
    if lstm_dates is None:
        return pd.Index(rf_dates)
    shared = pd.Index(rf_dates).intersection(pd.Index(lstm_dates))
    if len(shared) == 0:
        raise RuntimeError("Random Forest and LSTM holdouts do not overlap")
    return pd.Index(rf_dates).intersection(shared)


def run(retrain: bool, use_cache: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, source = load_market_frame(use_cache=use_cache)
    fingerprint = dataset_fingerprint(df)
    rf_features = present_feature_columns(df, RF_FEATURE_COLS)
    lstm_features = present_feature_columns(df, LSTM_FEATURE_COLS)
    samples = build_random_forest_samples(df, rf_features)

    rf_model, rf_source = random_forest_model(samples, fingerprint, retrain)
    rf_pred = pd.Series(
        np.asarray(rf_model.predict(samples["X_test"]), dtype=float),
        index=samples["target_dates"],
        name="rf_pred",
    )

    lstm_status = "not scored"
    lstm_error = None
    lstm_pred = None
    lstm_test_index = None
    try:
        partitions = lstm_partitions(df, lstm_features)
        lstm_status = lstm_model(partitions, fingerprint, retrain)
        lstm_pred = predictions_in_dollars(partitions)
        lstm_test_index = partitions["test_df"].index
        raw_close = partitions["test_df"]["Close"].to_numpy(dtype=float)
        restored_close = inverse_close(
            partitions["y_test"],
            partitions["scaler"],
            partitions["target_idx"],
            partitions["n_cols"],
        )
        if not np.allclose(raw_close, restored_close, rtol=1e-6, atol=1e-5):
            raise RuntimeError("LSTM scaled targets do not round-trip to Close")
    except Exception:
        lstm_error = traceback.format_exc()
        print("[Baseline] LSTM scoring failed:\n" + lstm_error)

    dates = shared_dates(samples["target_dates"], lstm_test_index)
    y_true = slice_to_dates(samples["y_test"], dates)
    previous = slice_to_dates(samples["previous_close"], dates)
    raw_previous = df["Close"].shift(1).loc[dates].to_numpy(dtype=float)
    if not np.allclose(previous, raw_previous):
        raise RuntimeError("Holdout previous close is not the prior session close")
    if not np.allclose(y_true, df.loc[dates, "Close"].to_numpy(dtype=float)):
        raise RuntimeError("Holdout target is not Close on the target date")

    rows = [
        score_named(
            "Persistence",
            y_true,
            persistence_forecast(previous),
            previous,
            leaky=False,
            sample="holdout",
        ),
        score_named(
            "Random Forest",
            y_true,
            slice_to_dates(rf_pred, dates),
            previous,
            leaky=False,
            sample="holdout",
        ),
    ]
    if lstm_pred is not None:
        rows.insert(
            1,
            score_named(
                "LSTM",
                y_true,
                slice_to_dates(lstm_pred, dates),
                previous,
                leaky=False,
                sample="holdout",
            ),
        )

    leaky = fit_leaky_random_forest(df, rf_features, samples, rf_model, dates)
    if not np.allclose(leaky["holdout_actual"], y_true):
        raise RuntimeError("Leaky evaluation closes differ from the honest holdout closes")
    rows.append(
        score_named(
            "Random Forest same-day (leaky)",
            leaky["holdout_actual"],
            leaky["holdout_pred"],
            previous,
            leaky=True,
            sample="holdout",
        )
    )
    rows.append(
        score_named(
            "Random Forest same-day (leaky)",
            leaky["train_actual"],
            leaky["train_pred"],
            leaky["train_previous"],
            leaky=True,
            sample="in-sample fit",
        )
    )

    rf_index = pd.Index(samples["target_dates"])
    lstm_index = None if lstm_test_index is None else pd.Index(lstm_test_index)
    if lstm_index is not None and rf_index.equals(lstm_index):
        definition = (
            "One chronological holdout: features observed at date t predict Close[t+1], "
            "then the first 80% of those shifted samples are training and the rest are "
            "the holdout (`split = int(len(samples) * 0.80)`). This is the boundary in "
            "`notebooks/random_forest_model.ipynb` and `notebooks/backtesting.ipynb`. "
            "The LSTM test dates from `notebooks/lstm_model.ipynb` are these same sessions."
        )
    elif lstm_index is not None and rf_index.isin(lstm_index).all():
        definition = (
            "Chronological holdout from the Random Forest and backtesting notebooks: "
            "features at date t predict Close[t+1], with `split = int(len(samples) * 0.80)`. "
            "The LSTM notebook's test window contains this holdout plus one earlier session. "
            "That extra session is not scored, so persistence, LSTM, and Random Forest "
            "use the same closes."
        )
    elif lstm_pred is None:
        definition = (
            "Chronological holdout from the Random Forest and backtesting notebooks: "
            "features at date t predict Close[t+1], with `split = int(len(samples) * 0.80)`. "
            "LSTM predictions were not available, so this table does not include an LSTM row."
        )
    else:
        definition = (
            "Shared chronological holdout: the intersection of the Random Forest shifted "
            "80/20 target dates and the LSTM test dates. Persistence, LSTM, and Random Forest "
            "are scored only on sessions both models can forecast out of sample."
        )

    def _span(index: pd.Index | None) -> tuple[str, str, int]:
        if index is None or len(index) == 0:
            return "n/a", "n/a", 0
        return str(pd.Timestamp(index[0]).date()), str(pd.Timestamp(index[-1]).date()), len(index)

    rf_start, rf_end, n_rf = _span(samples["target_dates"])
    lstm_start, lstm_end, n_lstm = _span(lstm_test_index)
    holdout_start, holdout_end, n_holdout = _span(dates)
    train_start, train_end, n_train = _span(leaky["train_dates"])
    for row in rows:
        if row["sample"] == "in-sample fit":
            row["target_start"] = train_start
            row["target_end"] = train_end
            row["n"] = n_train
        else:
            row["target_start"] = holdout_start
            row["target_end"] = holdout_end
            row["n"] = n_holdout

    report = {
        "ticker": TICKER,
        "requested_start": START,
        "requested_end": END,
        "data_source": source,
        "dataset_fingerprint": fingerprint,
        "feature_rows": int(len(df)),
        "feature_start": str(pd.Timestamp(df.index[0]).date()),
        "feature_end": str(pd.Timestamp(df.index[-1]).date()),
        "holdout_definition": definition,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "n_holdout": n_holdout,
        "rf_holdout_start": rf_start,
        "rf_holdout_end": rf_end,
        "n_rf_holdout": n_rf,
        "lstm_test_start": lstm_start,
        "lstm_test_end": lstm_end,
        "n_lstm_test": n_lstm,
        "leaky_train_start": train_start,
        "leaky_train_end": train_end,
        "n_leaky_train": n_train,
        "rf_source": rf_source,
        "lstm_status": lstm_status,
        "lstm_error": lstm_error,
        "rf_params": {
            key: value
            for key, value in rf_model.get_params().items()
            if key in {"n_estimators", "max_depth", "min_samples_split", "random_state"}
        },
        "rows": rows,
    }
    save_artifact_meta({
        "dataset_fingerprint": fingerprint,
        "feature_start": report["feature_start"],
        "feature_end": report["feature_end"],
        "feature_rows": report["feature_rows"],
        "rf_source": rf_source,
        "lstm_status": lstm_status,
    })
    write_report(report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train even when fingerprint-matched artifacts already exist",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Score data/AAPL_features.csv instead of downloading again",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(retrain=args.retrain, use_cache=args.use_cache)
