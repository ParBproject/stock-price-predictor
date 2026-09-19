"""Research-grade forecast evaluation and baseline utilities.

This module separates statistical forecast quality from economic usefulness.
It provides simple benchmarks, return-space metrics, directional accuracy,
strategy diagnostics, and explicit chronological walk-forward windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtesting import long_flat_returns_from_signals
from .evaluator import max_drawdown_from_returns, sharpe_ratio


@dataclass(frozen=True)
class ForecastEvaluation:
    """Statistical and economic diagnostics for one-step forecasts."""

    price_mae: float
    price_rmse: float
    return_mae: float
    return_rmse: float
    directional_accuracy: float
    return_correlation: float
    strategy_total_return: float
    strategy_sharpe: float
    strategy_max_drawdown: float


@dataclass(frozen=True)
class WalkForwardWindow:
    """One chronological train/test window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)


def random_walk_forecast(current_close: np.ndarray | pd.Series) -> np.ndarray:
    """Naïve one-step price forecast: tomorrow's close equals today's close."""
    current = np.asarray(current_close, dtype=float)
    if current.ndim != 1:
        raise ValueError("current_close must be one-dimensional")
    if len(current) == 0:
        raise ValueError("current_close must not be empty")
    if not np.isfinite(current).all() or np.any(current <= 0):
        raise ValueError("current_close must contain finite positive prices")
    return current.copy()


def trailing_mean_return_forecast(
    close: np.ndarray | pd.Series,
    *,
    window: int = 20,
) -> np.ndarray:
    """Forecast next close using only trailing historical simple returns.

    The value at position i uses returns ending at i. Early observations with
    insufficient history use the random-walk forecast, avoiding fabricated
    pre-history.
    """
    prices = np.asarray(close, dtype=float)
    if prices.ndim != 1:
        raise ValueError("close must be one-dimensional")
    if len(prices) == 0:
        raise ValueError("close must not be empty")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("close must contain finite positive prices")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError("window must be a positive integer")
    if window < 1:
        raise ValueError("window must be at least 1")

    forecast = prices.copy()
    if len(prices) < 2:
        return forecast

    returns = prices[1:] / prices[:-1] - 1.0
    for index in range(1, len(prices)):
        available = returns[max(0, index - window) : index]
        if len(available):
            forecast[index] = prices[index] * (1.0 + float(available.mean()))
    return forecast


def evaluate_one_step_forecasts(
    predicted_next_close: np.ndarray | pd.Series,
    current_close: np.ndarray | pd.Series,
    actual_next_close: np.ndarray | pd.Series,
    *,
    commission_rate: float = 0.0,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> ForecastEvaluation:
    """Evaluate next-close forecasts in both price and return space."""
    predicted = np.asarray(predicted_next_close, dtype=float)
    current = np.asarray(current_close, dtype=float)
    actual = np.asarray(actual_next_close, dtype=float)

    if predicted.ndim != 1 or current.ndim != 1 or actual.ndim != 1:
        raise ValueError("forecast arrays must be one-dimensional")
    if not (predicted.shape == current.shape == actual.shape):
        raise ValueError("forecast arrays must have matching shapes")
    if len(predicted) == 0:
        raise ValueError("forecast arrays must not be empty")
    if not (
        np.isfinite(predicted).all()
        and np.isfinite(current).all()
        and np.isfinite(actual).all()
    ):
        raise ValueError("forecast arrays must contain finite values")
    if np.any(current <= 0) or np.any(actual <= 0):
        raise ValueError("current and actual close prices must be positive")

    predicted_returns = predicted / current - 1.0
    actual_returns = actual / current - 1.0

    price_error = predicted - actual
    return_error = predicted_returns - actual_returns

    predicted_direction = predicted_returns > 0
    actual_direction = actual_returns > 0
    directional_accuracy = float(np.mean(predicted_direction == actual_direction))

    if (
        np.std(predicted_returns) > 0
        and np.std(actual_returns) > 0
        and len(predicted_returns) > 1
    ):
        return_correlation = float(np.corrcoef(predicted_returns, actual_returns)[0, 1])
    else:
        return_correlation = 0.0

    signals = predicted_direction.astype(float)
    strategy_returns = long_flat_returns_from_signals(
        signals,
        actual_returns,
        commission_rate=commission_rate,
    )
    total_return = float(np.prod(1.0 + strategy_returns) - 1.0)
    strategy_sharpe = sharpe_ratio(
        strategy_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    strategy_drawdown = max_drawdown_from_returns(strategy_returns)

    return ForecastEvaluation(
        price_mae=float(np.mean(np.abs(price_error))),
        price_rmse=float(np.sqrt(np.mean(price_error**2))),
        return_mae=float(np.mean(np.abs(return_error))),
        return_rmse=float(np.sqrt(np.mean(return_error**2))),
        directional_accuracy=directional_accuracy,
        return_correlation=return_correlation,
        strategy_total_return=total_return,
        strategy_sharpe=float(strategy_sharpe),
        strategy_max_drawdown=float(strategy_drawdown),
    )


def compare_with_random_walk(
    predicted_next_close: np.ndarray | pd.Series,
    current_close: np.ndarray | pd.Series,
    actual_next_close: np.ndarray | pd.Series,
    *,
    commission_rate: float = 0.0,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compare a model against the random-walk baseline on identical observations."""
    model = evaluate_one_step_forecasts(
        predicted_next_close,
        current_close,
        actual_next_close,
        commission_rate=commission_rate,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    baseline = evaluate_one_step_forecasts(
        random_walk_forecast(current_close),
        current_close,
        actual_next_close,
        commission_rate=commission_rate,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    return pd.DataFrame(
        {
            "Model": vars(model),
            "Random Walk": vars(baseline),
        }
    )


def walk_forward_windows(
    n_samples: int,
    *,
    min_train_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
) -> tuple[WalkForwardWindow, ...]:
    """Create chronological walk-forward windows with no overlap into the future."""
    for name, value in {
        "n_samples": n_samples,
        "min_train_size": min_train_size,
        "test_size": test_size,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    if step_size is None:
        step_size = test_size
    if isinstance(step_size, bool) or not isinstance(step_size, (int, np.integer)):
        raise TypeError("step_size must be a positive integer")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")
    if min_train_size + test_size > n_samples:
        raise ValueError("not enough samples for the requested first walk-forward window")

    windows: list[WalkForwardWindow] = []
    test_start = min_train_size
    while test_start < n_samples:
        test_end = min(test_start + test_size, n_samples)
        if test_end <= test_start:
            break

        train_start = 0 if expanding else test_start - min_train_size
        train_end = test_start
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start += step_size

    return tuple(windows)
