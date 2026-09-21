# Holdout baseline

Chronological holdout from the Random Forest and backtesting notebooks: features at date t predict Close[t+1], with `split = int(len(samples) * 0.80)`. The LSTM notebook's test window contains this holdout plus one earlier session. That extra session is not scored, so persistence, LSTM, and Random Forest use the same closes.

Requested window: AAPL 2015-01-01 to 2024-12-31 (yfinance `end` is exclusive). Feature frame after indicator warm-up: 2015-03-16 to 2024-12-30 (2466 rows). Data source: data/AAPL_features.csv saved from yfinance.

Scored sessions: 2023-01-13 to 2024-12-30 (493 targets). Random Forest native holdout: 2023-01-13 to 2024-12-30 (493). LSTM test sessions: 2023-01-12 to 2024-12-30 (494). Leaky in-sample fit: 2015-03-16 to 2023-01-11 (1972 training feature dates).

MAE is adjusted dollars (`regression_metrics` / `sklearn.metrics.mean_absolute_error`). Directional hit rate is the fraction of sessions where the forecast is on the same side of the previous close as the realized close. Up means strictly above the previous close. Persistence sets the next close equal to the previous close, so its hit rate is the share of holdout sessions that were not up.

| Model | Sample | MAE | Directional hit rate | Leaky |
| --- | --- | ---: | ---: | --- |
| Persistence | holdout | 1.879523 | 0.4381 | no |
| LSTM | holdout | 26.904755 | 0.4584 | no |
| Random Forest | holdout | 20.702787 | 0.4523 | no |
| Random Forest same-day (leaky) | holdout | 18.326268 | 0.5700 | yes |
| Random Forest same-day (leaky) | in-sample fit | 0.181557 | 0.9437 | yes |

Both same-day rows are leaky. They fit the Random Forest family on `Close[t]` from features at `t`. Those features already determine that close (`Close_Lag_1 * (1 + Pct_Change)` reconstructs it). The in-sample row is how good that fit looks on the dates it was trained on. The holdout row uses the same cheat on the future sessions above and is not a forecast. A tree still predicts training leaf averages, so a new high in the holdout is not a small error even when the close is visible in the features.

Random Forest artifact: loaded (`results/rf_regressor.pkl`). LSTM artifact: loaded (`results/lstm_model.keras`).
