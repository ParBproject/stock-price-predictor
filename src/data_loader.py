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


# ── Stationarity ───────────────────────────────────────────────────────────────

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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"[DataLoader] Saved to {save_path}")

    return df


# ── Train / Test Split ─────────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame,
                      train_ratio: float = 0.80):
    """Chronological split – no shuffle to prevent look-ahead bias."""
    split = int(len(df) * train_ratio)
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
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len: i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_stock_data(
        ticker    = "AAPL",
        start     = "2015-01-01",
        end       = "2024-12-31",
        save_path = "data/AAPL_features.csv",
    )
    print(df.tail(3))
