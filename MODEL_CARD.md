# Model & Research Card

## Purpose

This repository is an educational quantitative-research project for evaluating whether
machine-learning forecasts contain useful out-of-sample information about next-period
equity prices and returns.

It is designed to demonstrate forecasting methodology, leakage controls, benchmark
comparison, and economic evaluation. It is not a production trading system and does
not provide investment advice.

## Forecasting tasks

The project supports two related views:

- **next-close regression** — estimate the next observed close from information known at time t;
- **directional classification** — estimate whether the next-period return is positive.

The research-evaluation layer additionally converts price forecasts into predicted
returns so models can be compared in return space rather than only by price-level error.

## Data

Historical OHLCV data is obtained through `yfinance`.

Feature engineering includes:

- lagged prices;
- simple/log returns;
- moving averages;
- RSI;
- MACD;
- Bollinger Bands;
- ATR;
- volume transforms;
- optional sentiment features.

External market data can be delayed, revised, incomplete, or unavailable.

## Leakage controls

The project treats time alignment as correctness-critical.

Controls include:

- future targets are shifted explicitly;
- features at t predict targets at t + horizon;
- train/test splits are chronological;
- LSTM scalers are fit on training data only;
- holdout sequences use trailing training history without future test rows;
- sentiment is aligned only after it becomes observable;
- missing sentiment is neutral rather than fabricated;
- cross-validation uses chronological folds with a forecast-horizon gap;
- walk-forward windows keep training observations strictly before validation observations.

## Baselines

A forecasting model should not be judged without simple alternatives.

The repository now includes:

- **random-walk baseline:** next close equals current close;
- **trailing-mean-return baseline:** extrapolates only from trailing realized returns.

Model results should be interpreted relative to these baselines on identical holdout
observations.

## Statistical evaluation

The research layer supports:

- MAE;
- RMSE;
- MAPE;
- return MAE;
- return RMSE;
- directional accuracy;
- return correlation;
- classification precision/recall/F1.

Price-level accuracy and return-space accuracy answer different questions and are
reported separately.

## Economic evaluation

Forecasts can be translated into long/flat signals using only information available
when the signal is formed.

Evaluation includes:

- commission-aware returns;
- cumulative strategy return;
- annualized Sharpe ratio;
- maximum drawdown;
- Backtrader validation;
- fee-aware position sizing;
- buy-and-hold comparison.

Backtested results are research evidence, not a forecast of future performance.

## Known limitations

- Historical behavior may not persist across regimes.
- Price prediction can appear accurate even when directional information is weak.
- Transaction-cost models are simplified.
- Slippage, market impact, borrow costs, taxes, and execution latency are not fully modeled.
- News sentiment quality depends on source coverage and timestamp quality.
- Hyperparameter searches can overfit if repeatedly tuned to the same evaluation period.
- LSTM and tree models do not guarantee stable feature relationships.

## Production requirements not implemented

A production system would additionally require:

- walk-forward model retraining and model-selection governance;
- realistic execution/slippage simulation;
- market-impact assumptions;
- feature/data drift monitoring;
- model performance monitoring;
- exposure and loss limits;
- secure credentials and broker controls;
- reproducible model registry/versioning;
- independent validation.

## Responsible use

This repository is for education, portfolio demonstration, and research methodology.
Historical or simulated results should not be used as a sole basis for financial decisions.
