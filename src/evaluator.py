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
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Opaque dark canvas so the figures stay readable in GitHub light and dark mode.
CHART_BG = "#0B1220"
CHART_PANEL = "#111827"
CHART_TEXT = "#E5E7EB"
CHART_MUTED = "#94A3B8"
CHART_GRID = "#1F2937"
CHART_SPINE = "#334155"
CHART_ACCENT = "#10B981"
CHART_ACCENT_SOFT = "#6EE7B7"
CHART_SECONDARY = "#CBD5E1"
EMERALD_CMAP = LinearSegmentedColormap.from_list(
    "emerald", ["#0B1220", "#065F46", "#10B981", "#A7F3D0"]
)


def style_chart(fig, axes=None) -> None:
    """Apply the shared dark theme after axes, labels, and legends exist."""
    fig.patch.set_facecolor(CHART_BG)
    if axes is None:
        axes = fig.axes
    for ax in np.ravel(np.asarray(axes, dtype=object)):
        ax.set_facecolor(CHART_PANEL)
        ax.tick_params(colors=CHART_MUTED, labelcolor=CHART_MUTED)
        ax.xaxis.label.set_color(CHART_TEXT)
        ax.yaxis.label.set_color(CHART_TEXT)
        ax.title.set_color(CHART_TEXT)
        for spine in ax.spines.values():
            spine.set_color(CHART_SPINE)
        ax.grid(True, color=CHART_GRID, linewidth=0.6)
        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(CHART_PANEL)
            frame.set_edgecolor(CHART_SPINE)
            for text in legend.get_texts():
                text.set_color(CHART_TEXT)


def save_chart(fig, path: str) -> None:
    """Write a chart with its dark background baked into the image."""
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")


# ═══════════════════════════════════════════════════════════════════════════════
# Regression Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       label:  str = "Model",
                       verbose: bool = True) -> dict:
    """Returns MAE, MSE, RMSE and MAPE using positional sample alignment."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true and y_pred must contain only finite values")

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE%": mape}
    if verbose:
        print(f"\n── {label} Regression Metrics ──────────────────")
        for k, v in metrics.items():
            print(f"  {k:8s}: {v:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Classification Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def classification_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            label:  str = "Model",
                            verbose: bool = True) -> dict:
    """Returns accuracy, precision, recall, F1 for binary classification."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    if verbose:
        print(f"\n── {label} Classification Metrics ──────────────")
        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Down", "Up"],
            zero_division=0,
        ))
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Finance Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def sharpe_ratio(returns: np.ndarray | pd.Series,
                 risk_free_rate: float = 0.04,
                 periods_per_year: int = 252,
                 verbose: bool = True) -> float:
    """
    Annualised Sharpe Ratio.

    Parameters
    ----------
    returns          : daily portfolio returns (fractions, not %)
    risk_free_rate   : annual risk-free rate (default 4 % ≈ T-bill 2024)
    periods_per_year : 252 for daily data
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if len(returns) == 0:
        raise ValueError("returns must not be empty")
    if not np.isfinite(returns).all():
        raise ValueError("returns must contain only finite values")
    if isinstance(periods_per_year, bool) or not isinstance(
        periods_per_year, (int, np.integer)
    ):
        raise TypeError("periods_per_year must be a positive integer")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")
    if isinstance(risk_free_rate, bool) or not isinstance(
        risk_free_rate, (int, float, np.integer, np.floating)
    ):
        raise TypeError("risk_free_rate must be a real number")

    risk_free_rate = float(risk_free_rate)
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if risk_free_rate <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1")

    # Match Backtrader's default `convertrate=True` annual-to-period conversion.
    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - periodic_rf
    volatility = excess.std()
    if volatility == 0:
        return 0.0
    sr = (excess.mean() / volatility) * np.sqrt(periods_per_year)
    if verbose:
        print(f"  Sharpe Ratio: {sr:.4f}")
    return sr


def max_drawdown(equity_curve: np.ndarray | pd.Series, verbose: bool = True) -> float:
    """Maximum peak-to-trough drawdown."""
    equity = np.asarray(equity_curve)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-10)
    mdd    = dd.min()
    if verbose:
        print(f"  Max Drawdown: {mdd:.2%}")
    return mdd


def max_drawdown_from_returns(
    returns: np.ndarray | pd.Series,
    initial_equity: float = 1.0,
    verbose: bool = True,
) -> float:
    """Compute max drawdown from periodic returns including starting equity.

    The initial equity point is prepended before the first return is applied, so
    a loss in the first strategy period is measured against the actual starting
    capital rather than being treated as the initial peak.
    """
    returns = np.asarray(returns, dtype=float)
    initial_equity = float(initial_equity)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.isfinite(returns).all():
        raise ValueError("returns must contain only finite values")
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")

    equity_curve = np.empty(len(returns) + 1, dtype=float)
    equity_curve[0] = initial_equity
    if len(returns):
        equity_curve[1:] = initial_equity * np.cumprod(1.0 + returns)

    return max_drawdown(equity_curve, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     label:  str = "Model",
                     dates:  pd.DatetimeIndex | None = None,
                     save:   bool = True,
                     show:   bool = True,
                     save_path: str | None = None):
    """Overlay of actual vs predicted close prices."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = dates if dates is not None else np.arange(len(y_true))
    ax.plot(x, y_true, label="Actual", color=CHART_SECONDARY, linewidth=1.5)
    ax.plot(x, y_pred, label="Predicted", color=CHART_ACCENT, linewidth=1.5,
            linestyle="--")
    ax.set_title(f"{label} – Actual vs Predicted Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    style_chart(fig, ax)
    plt.tight_layout()
    if save:
        path = save_path or os.path.join(
            RESULTS_DIR, f"{label.lower().replace(' ', '_')}_predictions.png"
        )
        save_chart(fig, path)
        print(f"[Evaluator] Saved → {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_loss_curves(history, save: bool = True, show: bool = True,
                     save_path: str | None = None):
    """Training vs validation loss for LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss", color=CHART_ACCENT)
    axes[0].plot(history.history["val_loss"], label="Val Loss", color=CHART_SECONDARY)
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train MAE", color=CHART_ACCENT)
    axes[1].plot(history.history["val_mae"], label="Val MAE", color=CHART_SECONDARY)
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    style_chart(fig, axes)
    plt.tight_layout()
    if save:
        path = save_path or os.path.join(RESULTS_DIR, "lstm_loss_curves.png")
        save_chart(fig, path)
        print(f"[Evaluator] Saved → {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_feature_importance(model,
                             feature_names: list[str],
                             top_n: int = 20,
                             save:  bool = True,
                             show:  bool = True,
                             save_path: str | None = None):
    """Horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    ax.barh([feature_names[i] for i in idx],
            importances[idx], color=CHART_ACCENT)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    style_chart(fig, ax)
    plt.tight_layout()
    if save:
        path = save_path or os.path.join(RESULTS_DIR, "rf_feature_importance.png")
        save_chart(fig, path)
        print(f"[Evaluator] Saved → {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           label:  str = "RF Classifier",
                           save:   bool = True,
                           show:   bool = True,
                           save_path: str | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=EMERALD_CMAP,
        xticklabels=["Down", "Up"],
        yticklabels=["Down", "Up"], ax=ax,
        annot_kws={"color": CHART_TEXT},
        linewidths=0.6, linecolor=CHART_BG,
        cbar_kws={"shrink": 0.85},
    )
    ax.set_title(f"{label} – Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    style_chart(fig, ax)
    ax.grid(False)
    colorbar = None
    if ax.collections:
        colorbar = getattr(ax.collections[0], "colorbar", None)
    if colorbar is not None:
        colorbar.ax.yaxis.set_tick_params(color=CHART_MUTED)
        plt.setp(colorbar.ax.get_yticklabels(), color=CHART_MUTED)
        colorbar.outline.set_edgecolor(CHART_SPINE)
    plt.tight_layout()
    if save:
        path = save_path or os.path.join(RESULTS_DIR, "rf_confusion_matrix.png")
        save_chart(fig, path)
        print(f"[Evaluator] Saved → {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_equity_curve(equity_curve: pd.Series,
                      label: str = "Strategy",
                      benchmark: pd.Series | None = None,
                      save: bool = True,
                      show: bool = True,
                      save_path: str | None = None):
    """Plots portfolio equity curve vs an optional buy-and-hold benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity_curve.index, equity_curve.values, label=label,
            linewidth=1.5, color=CHART_ACCENT)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values,
                label="Buy & Hold", linewidth=1.5, linestyle="--",
                color=CHART_SECONDARY)
    ax.set_title(f"Equity Curve – {label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    style_chart(fig, ax)
    plt.tight_layout()
    if save:
        path = save_path or os.path.join(
            RESULTS_DIR, f"{label.lower().replace(' ', '_')}_equity.png"
        )
        save_chart(fig, path)
        print(f"[Evaluator] Saved → {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
