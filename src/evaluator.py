"""
evaluator.py
------------
Computes regression and classification metrics, plus finance-specific
metrics (Sharpe Ratio, cumulative returns) for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Regression Metrics ─────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       label:  str = "Model") -> dict:
    """Returns MAE, MSE, RMSE and MAPE."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE%": mape}
    print(f"\n── {label} Regression Metrics ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.4f}")
    return metrics


# ── Classification Metrics ─────────────────────────────────────────────────────

def classification_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            label:  str = "Model") -> dict:
    """Returns accuracy, precision, recall, F1 for binary classification."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"\n── {label} Classification Metrics ──────────────")
    print(classification_report(y_true, y_pred,
                                 target_names=["Down", "Up"],
                                 zero_division=0))
    return metrics


# ── Sharpe Ratio ───────────────────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray | pd.Series,
                 risk_free_rate: float = 0.04,
                 periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe Ratio.

    Parameters
    ----------
    returns          : daily portfolio returns (fractions, not %)
    risk_free_rate   : annual risk-free rate (default 4 % ≈ T-bill 2024)
    periods_per_year : 252 for daily data
    """
    returns = np.asarray(returns)
    daily_rf = risk_free_rate / periods_per_year
    excess   = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    sr = (excess.mean() / excess.std()) * np.sqrt(periods_per_year)
    print(f"  Sharpe Ratio: {sr:.4f}")
    return sr


def max_drawdown(equity_curve: np.ndarray | pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    equity = np.asarray(equity_curve)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-10)
    mdd    = dd.min()
    print(f"  Max Drawdown: {mdd:.2%}")
    return mdd


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     label:  str = "Model",
                     dates:  pd.DatetimeIndex | None = None,
                     save:   bool = True):
    """Overlay of actual vs predicted close prices."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = dates if dates is not None else np.arange(len(y_true))
    ax.plot(x, y_true, label="Actual",    color="steelblue",  linewidth=1.5)
    ax.plot(x, y_pred, label="Predicted", color="darkorange", linewidth=1.5,
            linestyle="--")
    ax.set_title(f"{label} – Actual vs Predicted Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f"{label.lower().replace(' ', '_')}_predictions.png")
        plt.savefig(path, dpi=150)
        print(f"[Evaluator] Saved → {path}")
    plt.show()


def plot_loss_curves(history, save: bool = True):
    """Training vs validation loss for LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"],     label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "lstm_loss_curves.png")
        plt.savefig(path, dpi=150)
        print(f"[Evaluator] Saved → {path}")
    plt.show()


def plot_feature_importance(model,
                             feature_names: list[str],
                             top_n: int = 20,
                             save:  bool = True):
    """Horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    ax.barh([feature_names[i] for i in idx],
            importances[idx], color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "rf_feature_importance.png")
        plt.savefig(path, dpi=150)
        print(f"[Evaluator] Saved → {path}")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           label:  str = "RF Classifier",
                           save:   bool = True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down", "Up"],
                yticklabels=["Down", "Up"], ax=ax)
    ax.set_title(f"{label} – Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "rf_confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"[Evaluator] Saved → {path}")
    plt.show()


def plot_equity_curve(equity_curve: pd.Series,
                      label: str = "Strategy",
                      benchmark: pd.Series | None = None,
                      save: bool = True):
    """Plots portfolio equity curve vs an optional buy-and-hold benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity_curve.index, equity_curve.values, label=label, linewidth=1.5)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values,
                label="Buy & Hold", linewidth=1.5, linestyle="--", color="grey")
    ax.set_title(f"Equity Curve – {label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f"{label.lower().replace(' ', '_')}_equity.png")
        plt.savefig(path, dpi=150)
        print(f"[Evaluator] Saved → {path}")
    plt.show()
