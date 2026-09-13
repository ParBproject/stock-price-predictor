"""
data_loader.py
--------------
Fetches and preprocesses historical stock data using yfinance,
engineers technical features, and performs stationarity checks.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ── Feature Engineering ────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2):
    """Upper, middle, and lower Bollinger Bands."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive set of technical indicators to the OHLCV dataframe.
    All indicators are computed on the 'Close' price unless stated otherwise.
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # --- Trend indicators ---
    df["SMA_10"]  = close.rolling(10).mean()
    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()

    # --- Momentum indicators ---
    df["RSI_14"]  = compute_rsi(close, 14)
    df["RSI_7"]   = compute_rsi(close, 7)
    macd, sig, hist = compute_macd(close)
    df["MACD"]        = macd
    df["MACD_Signal"] = sig
    df["MACD_Hist"]   = hist

    # --- Volatility indicators ---
    upper, mid, lower = compute_bollinger_bands(close)
    df["BB_Upper"] = upper
    df["BB_Mid"]   = mid
    df["BB_Lower"] = lower
    df["BB_Width"] = (upper - lower) / (mid + 1e-10)

    # Average True Range (ATR)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # --- Volume indicators ---
    df["Vol_Change"]   = vol.pct_change()
    df["Vol_SMA_20"]   = vol.rolling(20).mean()
    df["OBV"]          = (np.sign(close.diff()) * vol).fillna(0).cumsum()

    # --- Price transforms ---
    df["Log_Return"]   = np.log(close / close.shift(1))
    df["Price_Change"] = close.diff()                   # first difference (stationarity)
    df["Pct_Change"]   = close.pct_change()

    # --- Lagged close prices (useful for LSTM features) ---
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_Lag_{lag}"] = close.shift(lag)

    return df


# ── Stationarity ────────────────────────────────────────────────────────────────

def adf_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Augmented Dickey-Fuller test.
    H0: series has a unit root (non-stationary).
    Reject H0 (p < 0.05) → stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    output = {
        "name":       name,
        "adf_stat":   round(result[0], 4),
        "p_value":    round(result[1], 4),
        "n_lags":     result[2],
        "n_obs":      result[3],
        "stationary": result[1] < 0.05,
    }
    print(f"[ADF] {name}: stat={output['adf_stat']}, p={output['p_value']} "
          f"→ {'STATIONARY' if output['stationary'] else 'NON-STATIONARY'}")
    return output


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str = "AAPL",
                     start: str = "2015-01-01",
                     end:   str = "2024-12-31",
                     save_path: str | None = None) -> pd.DataFrame:
    """
    Downloads OHLCV data via yfinance, engineers features,
    and optionally saves to CSV.
    """
    print(f"[DataLoader] Fetching {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         "Check the symbol and date range.")

    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"[DataLoader] Raw shape: {df.shape}")

    # ADF on raw close
    adf_test(df["Close"], "Close (raw)")

    # Add features
    df = add_technical_indicators(df)

    # ADF on differenced close (should be stationary)
    adf_test(df["Price_Change"].dropna(), "Close (1st diff)")

    # Drop rows with NaN introduced by rolling windows
    df.dropna(inplace=True)
    print(f"[DataLoader] Shape after feature engineering & dropna: {df.shape}")

    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(save_path)
        print(f"[DataLoader] Saved to {save_path}")

    return df


# ── Supervised Forecast Alignment ──────────────────────────────────────────────

def prepare_forecast_data(df: pd.DataFrame,
                          feature_cols: list[str],
                          target_col: str = "Close",
                          horizon: int = 1):
    """Align features at time ``t`` with a future target at ``t + horizon``.

    The returned feature rows contain only information from their original
    timestamps. Targets are shifted backward so that row ``t`` predicts the
    future value at ``t + horizon``. The corresponding future timestamps are
    returned separately for plotting and evaluation.

    Returns
    -------
    X : pd.DataFrame
        Feature rows ending before the final ``horizon`` observations.
    y : pd.Series
        Future target values aligned positionally with ``X``.
    target_index : pd.Index
        Timestamps of the target observations.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be a positive integer")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    required = list(dict.fromkeys([*feature_cols, target_col]))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if len(df) <= horizon:
        raise ValueError("Not enough rows for the requested forecast horizon")

    X = df.loc[:, feature_cols].iloc[:-horizon].copy()
    y = df[target_col].shift(-horizon).iloc[:-horizon].copy()
    y.name = f"{target_col}_t_plus_{horizon}"
    target_index = df.index[horizon:].copy()

    return X, y, target_index


# ── Train / Test Split ─────────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame,
                      train_ratio: float = 0.80):
    """Chronological split with non-empty train and test partitions."""
    if isinstance(train_ratio, bool) or not isinstance(
        train_ratio, (int, float, np.integer, np.floating)
    ):
        raise TypeError("train_ratio must be a real number between 0 and 1")

    train_ratio = float(train_ratio)
    if not np.isfinite(train_ratio) or not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be finite and strictly between 0 and 1")

    split = int(len(df) * train_ratio)
    if split <= 0 or split >= len(df):
        raise ValueError(
            "Not enough rows for train_ratio to create non-empty train and test splits"
        )

    return df.iloc[:split], df.iloc[split:]


# ── Scaling ────────────────────────────────────────────────────────────────────

def scale_features(train: pd.DataFrame,
                   test:  pd.DataFrame,
                   feature_cols: list[str],
                   target_col:   str = "Close"):
    """
    Fits a MinMaxScaler on the training set only (avoids data leakage),
    then transforms both splits.  Returns scaled arrays and the fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    cols   = feature_cols + [target_col]

    train_scaled = scaler.fit_transform(train[cols])
    test_scaled  = scaler.transform(test[cols])

    return train_scaled, test_scaled, scaler


# ── LSTM Sequence Builder ──────────────────────────────────────────────────────

def build_sequences(data: np.ndarray,
                    seq_len: int = 60,
                    target_idx: int = -1):
    """
    Converts a 2-D array into overlapping sequences for LSTM input.

    Returns
    -------
    X : np.ndarray  shape (n_samples, seq_len, n_features)
    y : np.ndarray  shape (n_samples,)
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, (int, np.integer)):
        raise TypeError("seq_len must be a positive integer")
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    if isinstance(target_idx, bool) or not isinstance(target_idx, (int, np.integer)):
        raise TypeError("target_idx must be an integer")

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")

    n_features = data.shape[1]
    normalized_target_idx = int(target_idx)
    if normalized_target_idx < 0:
        normalized_target_idx += n_features
    if not 0 <= normalized_target_idx < n_features:
        raise IndexError("target_idx is out of bounds for the feature columns")

    if len(data) <= seq_len:
        return (
            np.empty((0, seq_len, n_features), dtype=data.dtype),
            np.empty((0,), dtype=data.dtype),
        )

    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len: i, :])
        y.append(data[i, normalized_target_idx])
    return np.asarray(X), np.asarray(y)


def build_holdout_sequences(train_data: np.ndarray,
                            test_data: np.ndarray,
                            seq_len: int = 60,
                            target_idx: int = -1):
    """Build holdout sequences with trailing training history as context.

    Every row in ``test_data`` is kept as a target. The first test target uses
    the final ``seq_len`` training observations as its history; later targets
    roll forward through already-observed test rows. No future holdout row is
    included in a sequence.
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, (int, np.integer)):
        raise TypeError("seq_len must be a positive integer")
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")

    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    if train_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError("train_data and test_data must be 2-D arrays")
    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError("train_data and test_data must have the same number of columns")
    if len(train_data) < seq_len:
        raise ValueError("train_data must contain at least seq_len rows")

    n_features = train_data.shape[1]
    if len(test_data) == 0:
        return (
            np.empty((0, seq_len, n_features), dtype=train_data.dtype),
            np.empty((0,), dtype=train_data.dtype),
        )

    context = train_data[-seq_len:]
    combined = np.concatenate([context, test_data], axis=0)
    return build_sequences(combined, seq_len=seq_len, target_idx=target_idx)


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_stock_data(
        ticker    = "AAPL",
        start     = "2015-01-01",
        end       = "2024-12-31",
        save_path = "data/AAPL_features.csv",
    )
    print(df.tail(3))
