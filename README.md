# 📈 Stock Price Predictor – LSTM & Random Forest

> **Educational disclaimer**: Stock market prediction is inherently uncertain.
> This project is built for learning purposes only and should **not** be used for real
> investment decisions. Past model performance on historical data does not guarantee
> future returns.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Why This Project Is Sophisticated](#why-this-project-is-sophisticated)
3. [Repository Structure](#repository-structure)
4. [Setup & Installation](#setup--installation)
5. [Usage](#usage)
6. [Methodology](#methodology)
7. [Results & Interpretation](#results--interpretation)
8. [Limitations & Ethical Notes](#limitations--ethical-notes)

---

## Project Overview

This project predicts **AAPL (Apple Inc.) next-day closing prices** using two
complementary machine learning approaches and evaluates them through full
**backtesting simulation**.

| Component | Tool / Library |
|-----------|----------------|
| Data fetching | `yfinance` |
| Technical features | `pandas`, `numpy` |
| Sentiment analysis | `vaderSentiment` + NewsAPI |
| Stationarity testing | `statsmodels` ADF |
| LSTM model | `TensorFlow/Keras` |
| Random Forest | `scikit-learn` |
| Backtesting | `backtrader` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |

---

## Why This Project Is Sophisticated

### 1. Time-Series Complexity
Raw stock prices are **non-stationary** – their mean and variance change over time,
which violates assumptions of most ML algorithms. We apply:
- **Augmented Dickey-Fuller (ADF) test** to detect unit roots
- **First differencing** and **log returns** to achieve stationarity

### 2. Dual-Model Comparison
| | LSTM | Random Forest |
|---|---|---|
| Type | Deep learning (sequential) | Ensemble (tree-based) |
| Temporal awareness | Native (gated memory cells) | Engineered via lag features |
| Interpretability | Low (black box) | High (feature importances) |
| Training speed | Slower (GPU recommended) | Fast (parallel trees) |

### 3. Overfitting Prevention
- **LSTM**: Dropout (0.2), Batch Normalisation, Early Stopping, ReduceLROnPlateau
- **RF**: TimeSeriesSplit cross-validation, max_depth tuning, min_samples_split
- **Both**: Strict train/test chronological split (no data leakage)

### 4. Finance-Specific Evaluation
Beyond standard ML metrics (MAE, RMSE), we compute:
- **Sharpe Ratio**: risk-adjusted return `(mean_return − risk_free) / std`
- **Maximum Drawdown**: worst peak-to-trough portfolio loss
- **Backtest simulation** via `backtrader` with realistic commission (0.1%)

### 5. Sentiment Integration
News headline sentiment (VADER compound scores) is incorporated as an additional
feature, capturing market mood alongside price action.

---

## Repository Structure

```
stock-predictor/
│
├── data/
│   ├── fetch_data.py          # Script to download fresh data
│   └── AAPL_features.csv      # (generated) Feature-engineered dataset
│
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis
│   ├── lstm_model.ipynb       # LSTM training & evaluation
│   ├── random_forest_model.ipynb  # RF training & evaluation
│   └── backtesting.ipynb      # Backtrader simulation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data fetching, feature engineering, scaling
│   ├── sentiment_analyzer.py  # VADER sentiment scoring
│   ├── model_trainer.py       # LSTM & RF model building/training
│   └── evaluator.py           # Metrics, plots, Sharpe ratio
│
├── results/                   # Saved models, metrics JSON, plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- (Optional) NVIDIA GPU for faster LSTM training

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/stock-predictor.git
cd stock-predictor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set NewsAPI Key
For live sentiment data, sign up at https://newsapi.org (free tier) and export:
```bash
export NEWS_API_KEY="your_key_here"
```
Without this key the pipeline automatically uses a synthetic sentiment fallback.

---

## Usage

### Quickstart – fetch data then run notebooks in order

**Step 1: Download data**
```bash
python data/fetch_data.py
```

**Step 2: Launch Jupyter**
```bash
jupyter notebook notebooks/
```
Run notebooks in this order:
1. `eda.ipynb`
2. `lstm_model.ipynb`
3. `random_forest_model.ipynb`
4. `backtesting.ipynb`

### Using `src/` modules directly

```python
from src.data_loader import fetch_stock_data, time_series_split, scale_features, build_sequences
from src.model_trainer import train_lstm, train_random_forest_regressor
from src.evaluator import regression_metrics, sharpe_ratio

# Fetch & prepare
df = fetch_stock_data('AAPL', '2015-01-01', '2024-12-31')
train, test = time_series_split(df)

# Train RF (no sequences needed)
feature_cols = ['SMA_20', 'RSI_14', 'MACD', 'Log_Return', 'Sentiment']
X_tr = train[feature_cols].values
y_tr = train['Close'].values
model = train_random_forest_regressor(X_tr, y_tr, tune=False)

# Evaluate
y_hat = model.predict(test[feature_cols].values)
regression_metrics(test['Close'].values, y_hat, 'Quick RF')
```

---

## Methodology

### Feature Engineering
| Feature | Description |
|---------|-------------|
| SMA 10/20/50 | Simple Moving Averages |
| EMA 12/26 | Exponential Moving Averages |
| RSI 7/14 | Relative Strength Index (momentum oscillator) |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands | Volatility bands (upper/lower/width) |
| ATR 14 | Average True Range (volatility) |
| OBV | On-Balance Volume |
| Log Returns | `ln(P_t / P_{t-1})` – stationary price change |
| Sentiment | Daily mean VADER compound score from news headlines |
| Close Lag 1–10 | Past closing prices as direct features |

### LSTM Architecture
```
Input (seq_len=60, n_features)
  └─ LSTM(64, return_sequences=True)
  └─ BatchNorm → Dropout(0.2)
  └─ LSTM(32)
  └─ BatchNorm → Dropout(0.2)
  └─ Dense(32, relu)
  └─ Dense(1)   ← predicted next-day Close
```

### Random Forest
- `n_estimators` ∈ {100, 200, 300} – tuned via GridSearchCV
- `max_depth` ∈ {10, 20, None}
- `TimeSeriesSplit(n_splits=5)` for cross-validation

---

## Results & Interpretation

After training you will find the following in `results/`:

| File | Content |
|------|---------|
| `lstm_metrics.json` | MAE, RMSE, Sharpe, MaxDrawdown |
| `rf_metrics.json` | Same metrics for RF |
| `lstm_predictions.png` | Actual vs predicted price chart |
| `rf_regressor_predictions.png` | RF price chart |
| `lstm_loss_curves.png` | Train / val MSE and MAE over epochs |
| `rf_feature_importance.png` | Top-20 feature importances |
| `rf_confusion_matrix.png` | Direction classification confusion matrix |
| `ml_strategy_equity.png` | Equity curve vs buy-and-hold benchmark |

### Interpreting Sharpe Ratio
- **> 1.0**: Strategy generates good risk-adjusted returns
- **0.5–1.0**: Acceptable, but modest
- **< 0**: Strategy loses money after accounting for risk

### Interpreting Max Drawdown
Lower is better. A drawdown of –25 % means the strategy lost 25 % from its peak
before recovering.

---

## Limitations & Ethical Notes

1. **Prediction uncertainty**: Financial markets are influenced by countless
   unpredictable factors (geopolitics, black-swan events, regulatory changes).
   No model can reliably predict future prices.

2. **Survivorship bias**: Using only AAPL ignores stocks that delisted or failed.

3. **Look-ahead bias mitigation**: We use a strict chronological train/test split
   and fit scalers only on training data. Cross-validation uses `TimeSeriesSplit`.

4. **Transaction costs**: The backtest applies 0.1 % commission but ignores bid-ask
   spread, market impact, and tax implications.

5. **Ethical note on algorithmic trading**: High-frequency algorithmic strategies
   can contribute to market instability. Any real deployment should comply with
   relevant financial regulations (SEC, FINRA, MiFID II, etc.).

---

## License
MIT – see `LICENSE` for details.
