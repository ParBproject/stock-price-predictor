"""
model_trainer.py
----------------
Builds, trains, and serialises both models:
  1. LSTM  – Keras sequential model for time-series regression
  2. Random Forest – scikit-learn ensemble baseline

Both models predict the next-day Close price (regression).
A thin classification wrapper is also provided (directional: up/down).
"""

import os
from typing import Any

import numpy as np
import joblib

# ── scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.validation import make_forecast_time_series_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_keras() -> dict[str, Any]:
    """Import Keras symbols on demand so tree models do not require TensorFlow."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam

    return {
        "Sequential": Sequential,
        "load_model": load_model,
        "LSTM": LSTM,
        "Dense": Dense,
        "Dropout": Dropout,
        "BatchNormalization": BatchNormalization,
        "EarlyStopping": EarlyStopping,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "ModelCheckpoint": ModelCheckpoint,
        "Adam": Adam,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM
# ═══════════════════════════════════════════════════════════════════════════════

def build_lstm(seq_len: int,
               n_features: int,
               units: int = 64,
               dropout: float = 0.2,
               learning_rate: float = 1e-3) -> Any:
    """
    Stacked LSTM with Batch Normalisation and Dropout to reduce overfitting.

    Architecture
    ------------
    LSTM(units, return_sequences=True) → BN → Dropout
    LSTM(units // 2)                   → BN → Dropout
    Dense(32, relu)                    → Dense(1, linear)
    """
    keras = _load_keras()
    model = keras["Sequential"]([
        keras["LSTM"](units, return_sequences=True,
                      input_shape=(seq_len, n_features)),
        keras["BatchNormalization"](),
        keras["Dropout"](dropout),

        keras["LSTM"](units // 2, return_sequences=False),
        keras["BatchNormalization"](),
        keras["Dropout"](dropout),

        keras["Dense"](32, activation="relu"),
        keras["Dense"](1),              # linear output → regression
    ])

    model.compile(
        optimizer=keras["Adam"](learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def train_lstm(X_train: np.ndarray,
               y_train: np.ndarray,
               X_val:   np.ndarray,
               y_val:   np.ndarray,
               units:        int   = 64,
               dropout:      float = 0.2,
               learning_rate: float = 1e-3,
               epochs:       int   = 50,
               batch_size:   int   = 32,
               save_path:    str | None = None):
    """
    Trains the LSTM with early stopping and LR reduction on plateau.

    Returns (model, history)
    """
    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]
    keras = _load_keras()

    model = build_lstm(seq_len, n_features, units, dropout, learning_rate)
    model.summary()

    ckpt_path = os.path.join(RESULTS_DIR, "lstm_best.keras")
    callbacks = [
        keras["EarlyStopping"](monitor="val_loss", patience=8,
                               restore_best_weights=True, verbose=1),
        keras["ReduceLROnPlateau"](monitor="val_loss", factor=0.5,
                                   patience=4, min_lr=1e-6, verbose=1),
        keras["ModelCheckpoint"](ckpt_path, monitor="val_loss",
                                 save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if save_path:
        model.save(save_path)
        print(f"[LSTM] Model saved → {save_path}")

    return model, history


def load_lstm(path: str) -> Any:
    return _load_keras()["load_model"](path)


# ═══════════════════════════════════════════════════════════════════════════════
# Random Forest
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest_regressor(X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  tune: bool = True,
                                  n_splits: int = 5,
                                  save_path: str | None = None,
                                  forecast_horizon: int = 1,
                                  n_estimators: int = 200,
                                  max_depth: int | None = 20,
                                  n_jobs: int = -1,
                                  ) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor.

    When tune=True, performs GridSearchCV with chronological folds and a gap
    equal to ``forecast_horizon``. The embargo prevents shifted future targets
    from overlapping the validation feature period.
    """
    if tune:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth":    [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        tscv = make_forecast_time_series_split(
            n_splits=n_splits,
            forecast_horizon=forecast_horizon,
        )
        rf   = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
        gs   = GridSearchCV(rf, param_grid, cv=tscv,
                            scoring="neg_mean_squared_error",
                            n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"[RF] Best params: {gs.best_params_}")
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=42, n_jobs=n_jobs)
        model.fit(X_train, y_train)

    if save_path:
        joblib.dump(model, save_path)
        print(f"[RF] Model saved → {save_path}")

    return model


def train_random_forest_classifier(X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   tune: bool = False,
                                   save_path: str | None = None,
                                   forecast_horizon: int = 1,
                                   n_estimators: int = 200,
                                   max_depth: int | None = 20,
                                   n_jobs: int = -1,
                                   ) -> RandomForestClassifier:
    """
    Binary classifier predicting price direction: 1 = up, 0 = down.
    y_train should be binary labels derived from next-period price returns.
    """
    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth":    [None, 10],
        }
        tscv = make_forecast_time_series_split(
            n_splits=5,
            forecast_horizon=forecast_horizon,
        )
        rf    = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
        gs    = GridSearchCV(rf, param_grid, cv=tscv,
                             scoring="f1", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"[RF-Clf] Best params: {gs.best_params_}")
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=42, n_jobs=n_jobs)
        model.fit(X_train, y_train)

    if save_path:
        joblib.dump(model, save_path)
        print(f"[RF-Clf] Model saved → {save_path}")

    return model


def load_rf(path: str):
    return joblib.load(path)
