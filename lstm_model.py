"""
lstm_model.py
─────────────
LSTM-based models for:
  - Multi-label failure classification (sequence → probability per label)
  - RUL regression (sequence → scalar)

Design decisions:
  - Input: sliding window of SEQ_LEN timesteps
  - Stacked bidirectional LSTM layers with dropout
  - Separate heads for classification vs regression
  - Early stopping + LR reduction on plateau
  - Class weights passed to loss for imbalance
  - Scaler fit on training sequences only (no leakage)

Requirements: tensorflow >= 2.12 or torch. We use Keras (tf.keras) here.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TARGET_COLS, PRIMARY_TARGET,
    LSTM_SEQ_LEN, LSTM_UNITS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LR, LSTM_PATIENCE,
    RANDOM_STATE, DEFAULT_THRESHOLD,
)
from baseline_model import tune_threshold

logger = logging.getLogger(__name__)

# ── Optional TF import ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks

    tf.random.set_seed(RANDOM_STATE)
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed — LSTM models unavailable.")


# ── Sequence Construction ─────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = LSTM_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a flat feature matrix into overlapping sliding-window sequences.

    Args:
        X:       (N, features) scaled feature array
        y:       (N, targets) or (N,) label array
        seq_len: number of timesteps per sequence

    Returns:
        X_seq: (N - seq_len, seq_len, features)
        y_seq: (N - seq_len, targets) — label at the LAST timestep of each window
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])   # predict state at end of window
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ── Model Architectures ───────────────────────────────────────────────────────

def build_lstm_classifier(
    seq_len: int,
    n_features: int,
    n_labels: int,
    lstm_units: list[int] = LSTM_UNITS,
    dropout: float = LSTM_DROPOUT,
) -> "keras.Model":
    """
    Stacked Bidirectional LSTM for multi-label binary classification.

    Architecture:
      Input → BiLSTM(64) → Dropout → BiLSTM(32) → Dropout
            → Dense(32, relu) → Dense(n_labels, sigmoid)

    Each output neuron is independently calibrated — sigmoid, not softmax.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for LSTM models.")

    inp = keras.Input(shape=(seq_len, n_features), name="sensor_sequence")
    x   = inp

    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=return_seq),
            name=f"bilstm_{i}",
        )(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    x   = layers.Dense(32, activation="relu", name="dense_hidden")(x)
    out = layers.Dense(n_labels, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="lstm_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_regressor(
    seq_len: int,
    n_features: int,
    lstm_units: list[int] = LSTM_UNITS,
    dropout: float = LSTM_DROPOUT,
) -> "keras.Model":
    """
    Stacked Bidirectional LSTM for RUL regression.

    Output: single neuron, linear activation (unbounded positive).
    Loss: Huber — less sensitive to RUL outliers than MSE.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for LSTM models.")

    inp = keras.Input(shape=(seq_len, n_features), name="sensor_sequence")
    x   = inp

    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=return_seq),
            name=f"bilstm_{i}",
        )(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    x   = layers.Dense(32, activation="relu", name="dense_hidden")(x)
    out = layers.Dense(1, activation="linear", name="rul_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="lstm_regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss=keras.losses.Huber(delta=10.0),  # Huber with delta=10 cycles
        metrics=["mae"],
    )
    return model


# ── Training Helpers ──────────────────────────────────────────────────────────

def _get_callbacks(monitor: str = "val_loss") -> list:
    return [
        callbacks.EarlyStopping(
            monitor=monitor, patience=LSTM_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=5,
            min_lr=1e-5, verbose=1,
        ),
    ]


def _compute_class_weights(y_labels: np.ndarray) -> np.ndarray:
    """
    Per-label class weights averaged into a sample-weight vector.
    Addresses the severe imbalance in failure labels.
    """
    sample_weights = np.ones(len(y_labels))
    for i in range(y_labels.shape[1]):
        classes = np.unique(y_labels[:, i])
        if len(classes) < 2:
            continue
        cw = compute_class_weight(
            "balanced", classes=classes, y=y_labels[:, i]
        )
        cw_dict = dict(zip(classes.astype(int), cw))
        sample_weights *= np.where(
            y_labels[:, i] == 1, cw_dict.get(1, 1.0), cw_dict.get(0, 1.0)
        )
    # Normalize to [1, max_weight] range
    sample_weights = 1 + (sample_weights - sample_weights.min()) / (
        sample_weights.max() - sample_weights.min() + 1e-8
    )
    return sample_weights


# ── Train LSTM Classifier ─────────────────────────────────────────────────────

def train_lstm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
) -> tuple["keras.Model", dict, dict]:
    """
    Train the LSTM classifier and tune per-label thresholds on validation set.

    Args:
        X_train, y_train: training sequences (already scaled)
        X_val,   y_val:   validation sequences (already scaled)
        scaler:           the fitted scaler (stored for inference)

    Returns:
        model:      trained Keras model
        thresholds: {label: float}
        history:    Keras training history dict
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for LSTM models.")

    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]
    n_labels   = y_train.shape[1]

    model = build_lstm_classifier(seq_len, n_features, n_labels)
    model.summary(print_fn=logger.info)

    sample_weights = _compute_class_weights(y_train)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        sample_weight=sample_weights,
        callbacks=_get_callbacks("val_loss"),
        verbose=1,
    )

    # Tune thresholds on validation set
    val_probs  = model.predict(X_val, verbose=0)
    thresholds = {}
    for i, col in enumerate(TARGET_COLS):
        thresholds[col] = tune_threshold(y_val[:, i], val_probs[:, i])
        logger.info(f"  [LSTM] {col}: threshold={thresholds[col]:.3f}")

    return model, thresholds, history.history


def evaluate_lstm_classifier(
    model: "keras.Model",
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: dict,
) -> dict:
    """Evaluate the LSTM classifier on a held-out test set."""
    probs = model.predict(X_test, verbose=0)
    preds = np.zeros_like(probs, dtype=int)
    for i, col in enumerate(TARGET_COLS):
        preds[:, i] = (probs[:, i] >= thresholds.get(col, DEFAULT_THRESHOLD)).astype(int)

    report      = classification_report(
        y_test, preds, target_names=TARGET_COLS, zero_division=0, output_dict=True
    )
    exact_match = float(np.mean(np.all(y_test == preds, axis=1)))
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    logger.info(f"\n[LSTM] Test Set Results:")
    logger.info(f"  Exact Match Accuracy : {exact_match*100:.2f}%")
    logger.info(f"  Weighted F1          : {weighted_f1*100:.2f}%")

    return {
        "model":       "LSTM",
        "exact_match": exact_match,
        "weighted_f1": weighted_f1,
        "per_label":   report,
        "y_true":      y_test,
        "y_pred":      preds,
        "y_prob":      probs,
    }


# ── Train LSTM Regressor ──────────────────────────────────────────────────────

def train_lstm_regressor(
    X_train: np.ndarray,
    y_rul_train: np.ndarray,
    X_val: np.ndarray,
    y_rul_val: np.ndarray,
) -> tuple["keras.Model", dict]:
    """
    Train the LSTM RUL regressor.

    Returns:
        model:   trained Keras model
        metrics: {mae, rmse, r2} on validation set
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for LSTM models.")

    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_lstm_regressor(seq_len, n_features)

    model.fit(
        X_train, y_rul_train,
        validation_data=(X_val, y_rul_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=_get_callbacks("val_mae"),
        verbose=1,
    )

    preds = model.predict(X_val, verbose=0).flatten()
    mae   = mean_absolute_error(y_rul_val, preds)
    rmse  = np.sqrt(mean_squared_error(y_rul_val, preds))
    r2    = r2_score(y_rul_val, preds)

    logger.info(f"\n[LSTM] RUL Regression — Val Set:")
    logger.info(f"  MAE  : {mae:.2f} cycles")
    logger.info(f"  RMSE : {rmse:.2f} cycles")
    logger.info(f"  R²   : {r2:.4f}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}
