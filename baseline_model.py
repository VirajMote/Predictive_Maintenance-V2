"""
baseline_model.py
─────────────────
Random Forest and XGBoost models for:
  - Multi-label failure classification (6 targets)
  - RUL regression

Key design decisions:
  - Scaler is fit INSIDE each CV fold to prevent leakage
  - Threshold tuning uses a held-out validation set (not the test fold)
  - scale_pos_weight for XGBoost is computed dynamically per label
  - Final models are retrained on full training data after CV
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("XGBoost not installed — skipping XGB models.")

from config import (
    TARGET_COLS, PRIMARY_TARGET, CV_FOLDS, RANDOM_STATE,
    RF_PARAMS, XGB_PARAMS, TEST_SIZE, DEFAULT_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── Threshold Tuning ──────────────────────────────────────────────────────────

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1 on the given data.
    Used on an independent validation set — never the test set.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


# ── Cross-Validation Evaluation ───────────────────────────────────────────────

def evaluate_classifier_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str = "RF",
) -> dict:
    """
    Stratified K-Fold CV for multi-label failure classification.

    Correct leakage-free pipeline per fold:
      1. Split into train / val / test (80/10/10)
         - train: fit scaler + model
         - val:   tune decision threshold
         - test:  final evaluation (threshold fixed)

    Returns dict with per-label metrics and aggregated scores.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold, (train_val_idx, test_idx) in enumerate(
        skf.split(X, y[PRIMARY_TARGET]), 1
    ):
        logger.info(f"  [{model_name}] Fold {fold}/{CV_FOLDS}")

        X_train_val = X.iloc[train_val_idx]
        y_train_val = y.iloc[train_val_idx]
        X_test      = X.iloc[test_idx]
        y_test      = y.iloc[test_idx]

        # Split train_val → train + val for threshold tuning
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.15,
            stratify=y_train_val[PRIMARY_TARGET],
            random_state=RANDOM_STATE,
        )

        # Fit scaler on training data only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        fold_pred = np.zeros((len(X_test), len(TARGET_COLS)), dtype=int)
        fold_prob = np.zeros((len(X_test), len(TARGET_COLS)), dtype=float)

        for i, col in enumerate(TARGET_COLS):
            clf = _build_classifier(model_name, y_train[col])
            clf.fit(X_train_s, y_train[col])

            # Tune threshold on validation set
            val_prob  = clf.predict_proba(X_val_s)[:, 1]
            threshold = tune_threshold(y_val[col].values, val_prob)

            # Evaluate on test fold with tuned threshold
            test_prob = clf.predict_proba(X_test_s)[:, 1]
            fold_prob[:, i] = test_prob
            fold_pred[:, i] = (test_prob >= threshold).astype(int)

        all_y_true.append(y_test.values)
        all_y_pred.append(fold_pred)
        all_y_prob.append(fold_prob)

    y_true = np.vstack(all_y_true)
    y_pred = np.vstack(all_y_pred)

    report = classification_report(
        y_true, y_pred, target_names=TARGET_COLS, zero_division=0, output_dict=True
    )
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    logger.info(f"\n[{model_name}] CV Results:")
    logger.info(f"  Exact Match Accuracy : {exact_match*100:.2f}%")
    logger.info(f"  Weighted F1          : {weighted_f1*100:.2f}%")

    return {
        "model":        model_name,
        "exact_match":  exact_match,
        "weighted_f1":  weighted_f1,
        "per_label":    report,
        "y_true":       y_true,
        "y_pred":       y_pred,
        "y_prob":       np.vstack(all_y_prob),
    }


# ── Final Model Training ──────────────────────────────────────────────────────

def train_final_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    model_name: str = "RF",
) -> tuple[dict, dict, StandardScaler]:
    """
    Train final classifiers on full training set.
    Scaler is fit on X_train, thresholds tuned on X_val.

    Returns:
        models:     {label: fitted classifier}
        thresholds: {label: optimal probability threshold}
        scaler:     fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    models, thresholds = {}, {}

    for col in TARGET_COLS:
        clf = _build_classifier(model_name, y_train[col])
        clf.fit(X_train_s, y_train[col])

        val_prob       = clf.predict_proba(X_val_s)[:, 1]
        thresholds[col] = tune_threshold(y_val[col].values, val_prob)
        models[col]    = clf

        logger.info(
            f"  [{model_name}] {col}: threshold={thresholds[col]:.3f}"
        )

    return models, thresholds, scaler


# ── RUL Regression ────────────────────────────────────────────────────────────

def train_rul_model(
    X_train: pd.DataFrame,
    y_rul_train: pd.Series,
    X_val: pd.DataFrame,
    y_rul_val: pd.Series,
    scaler: StandardScaler,
    model_name: str = "RF",
) -> dict:
    """
    Train a RUL regression model using the already-fit scaler.

    NOTE: scaler must be fit on X_train before calling this.
          Pass the same scaler used for classifiers to avoid drift.

    Returns:
        {model, metrics}
    """
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)

    if model_name == "XGB" and XGB_AVAILABLE:
        reg = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        reg = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    reg.fit(X_train_s, y_rul_train)

    preds  = reg.predict(X_val_s)
    mae    = mean_absolute_error(y_rul_val, preds)
    rmse   = np.sqrt(mean_squared_error(y_rul_val, preds))
    r2     = r2_score(y_rul_val, preds)

    logger.info(f"\n[{model_name}] RUL Regression — Val Set:")
    logger.info(f"  MAE  : {mae:.2f} cycles")
    logger.info(f"  RMSE : {rmse:.2f} cycles")
    logger.info(f"  R²   : {r2:.4f}")

    return {"model": reg, "mae": mae, "rmse": rmse, "r2": r2}


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_classifier(model_name: str, y_label: pd.Series):
    """Instantiate the correct classifier with dynamic pos_weight for XGB."""
    if model_name == "XGB" and XGB_AVAILABLE:
        neg  = (y_label == 0).sum()
        pos  = (y_label == 1).sum()
        spw  = float(neg / pos) if pos > 0 else 1.0
        params = {**XGB_PARAMS, "scale_pos_weight": spw}
        params.pop("use_label_encoder", None)
        return XGBClassifier(**params)
    else:
        return RandomForestClassifier(**RF_PARAMS)
