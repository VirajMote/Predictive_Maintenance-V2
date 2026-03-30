"""
inference.py
────────────
Production inference layer.

Responsibilities:
  - Load the saved model package
  - Preprocess raw input
  - Run classifier + RUL regressor
  - Compute health score
  - Emit structured alerts

This is the module the FastAPI backend calls.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    TARGET_COLS, PRIMARY_TARGET,
    HEALTH_WEIGHTS, ALERT_THRESHOLDS,
    RUL_CAP, LSTM_SEQ_LEN, MODEL_PKG_PATH,
)
from feature_engineering import build_features, get_feature_columns

logger = logging.getLogger(__name__)


# ── Output Schema ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Structured output from a single inference call."""
    # Classification
    failure_probabilities: dict[str, float]   # {label: prob}
    failure_predictions:   dict[str, int]     # {label: 0/1}

    # RUL
    rul_cycles: float                         # predicted cycles remaining
    rul_hours:  float | None                  # converted if cycle_duration_s provided

    # Health
    health_score: float                       # 0 (critical) → 100 (healthy)
    health_label: str                         # "Healthy" / "Warning" / "Critical"

    # Alerts
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "failure_probabilities": self.failure_probabilities,
            "failure_predictions":   self.failure_predictions,
            "rul_cycles":            round(self.rul_cycles, 1),
            "rul_hours":             round(self.rul_hours, 1) if self.rul_hours else None,
            "health_score":          round(self.health_score, 1),
            "health_label":          self.health_label,
            "alerts":                self.alerts,
        }


# ── Model Package ─────────────────────────────────────────────────────────────

class ModelPackage:
    """
    Wrapper around the persisted model artifact.

    Package structure (saved by train.py):
    {
        "mode":               "RF" | "XGB" | "LSTM",
        "scaler":             StandardScaler,
        "label_encoder":      LabelEncoder (for Type col),
        "classifier_models":  {label: model},
        "classifier_thresholds": {label: float},
        "rul_model":          regressor,
        "feature_columns":    [str],
    }
    """

    def __init__(self, pkg: dict):
        self.mode          = pkg["mode"]
        self.scaler        = pkg["scaler"]
        self.classifiers   = pkg["classifier_models"]
        self.thresholds    = pkg["classifier_thresholds"]
        self.rul_model     = pkg["rul_model"]
        self.feature_cols  = pkg["feature_columns"]
        self._is_lstm      = self.mode == "LSTM"

    @classmethod
    def load(cls, path: Path = MODEL_PKG_PATH) -> "ModelPackage":
        logger.info(f"Loading model package from {path}")
        pkg = joblib.load(path)
        return cls(pkg)


# ── Inference Engine ──────────────────────────────────────────────────────────

class PredictiveMaintenanceInference:
    """
    Stateless inference engine.

    Usage:
        engine = PredictiveMaintenanceInference.from_package(path)
        result = engine.predict(raw_df)
        print(result.to_dict())
    """

    def __init__(self, package: ModelPackage):
        self.pkg = package

    @classmethod
    def from_package(cls, path: Path = MODEL_PKG_PATH) -> "PredictiveMaintenanceInference":
        return cls(ModelPackage.load(path))

    def predict(
        self,
        raw_df: pd.DataFrame,
        cycle_duration_seconds: float | None = None,
    ) -> PredictionResult:
        """
        Run full inference pipeline on raw sensor data.

        Args:
            raw_df:                  DataFrame with raw sensor columns.
                                     Can be a single row or a window of rows.
                                     For LSTM, minimum LSTM_SEQ_LEN rows required.
            cycle_duration_seconds:  If provided, converts RUL cycles → hours.

        Returns:
            PredictionResult
        """
        # 1. Feature engineering
        features_df = build_features(raw_df, compute_rul_labels=False)
        X = features_df[self.pkg.feature_cols].fillna(0)

        if self.pkg._is_lstm:
            return self._predict_lstm(X, cycle_duration_seconds)
        else:
            return self._predict_tabular(X, cycle_duration_seconds)

    # ── Tabular (RF / XGB) ────────────────────────────────────────────────────

    def _predict_tabular(
        self, X: pd.DataFrame, cycle_duration_seconds: float | None
    ) -> PredictionResult:
        # Use the last row for prediction (most recent reading)
        X_last = X.iloc[[-1]]
        X_scaled = self.pkg.scaler.transform(X_last)

        probs, preds = {}, {}
        for col in TARGET_COLS:
            prob        = self.pkg.classifiers[col].predict_proba(X_scaled)[0, 1]
            threshold   = self.pkg.thresholds.get(col, 0.5)
            probs[col]  = float(prob)
            preds[col]  = int(prob >= threshold)

        rul = float(self.pkg.rul_model.predict(X_scaled)[0])
        return self._build_result(probs, preds, rul, cycle_duration_seconds)

    # ── LSTM ──────────────────────────────────────────────────────────────────

    def _predict_lstm(
        self, X: pd.DataFrame, cycle_duration_seconds: float | None
    ) -> PredictionResult:
        from lstm_model import build_sequences

        X_arr = self.pkg.scaler.transform(X)

        if len(X_arr) < LSTM_SEQ_LEN:
            raise ValueError(
                f"LSTM requires at least {LSTM_SEQ_LEN} timesteps; "
                f"got {len(X_arr)}."
            )

        # Take the last valid sequence
        seq   = X_arr[-LSTM_SEQ_LEN:][np.newaxis, :, :]  # (1, seq_len, features)
        seq_f = seq.astype(np.float32)

        clf_probs = self.pkg.classifiers.predict(seq_f, verbose=0)[0]  # (n_labels,)
        rul_val   = float(self.pkg.rul_model.predict(seq_f, verbose=0)[0, 0])

        probs = {col: float(clf_probs[i]) for i, col in enumerate(TARGET_COLS)}
        preds = {
            col: int(probs[col] >= self.pkg.thresholds.get(col, 0.5))
            for col in TARGET_COLS
        }

        return self._build_result(probs, preds, rul_val, cycle_duration_seconds)

    # ── Shared Output Construction ─────────────────────────────────────────────

    def _build_result(
        self,
        probs: dict,
        preds: dict,
        rul: float,
        cycle_duration_seconds: float | None,
    ) -> PredictionResult:
        rul = float(np.clip(rul, 0, RUL_CAP))

        rul_hours = None
        if cycle_duration_seconds is not None:
            rul_hours = rul * cycle_duration_seconds / 3600.0

        health = compute_health_score(probs)
        label  = health_label(health)
        alerts = generate_alerts(probs, rul, health)

        return PredictionResult(
            failure_probabilities=probs,
            failure_predictions=preds,
            rul_cycles=rul,
            rul_hours=rul_hours,
            health_score=health,
            health_label=label,
            alerts=alerts,
        )


# ── Health Score ──────────────────────────────────────────────────────────────

def compute_health_score(probs: dict[str, float]) -> float:
    """
    Weighted aggregation of failure probabilities → health score in [0, 100].

    Score of 100 = no failure risk.
    Score of 0   = certain failure.

    Each label contributes proportionally to its weight in HEALTH_WEIGHTS.
    The score degrades non-linearly: sqrt gives more sensitivity near high probs.
    """
    weighted_risk = sum(
        HEALTH_WEIGHTS.get(label, 0) * prob
        for label, prob in probs.items()
    )
    # Non-linear mapping: penalise high-risk states more sharply
    health = 100 * (1 - np.sqrt(np.clip(weighted_risk, 0, 1)))
    return float(np.clip(health, 0, 100))


def health_label(score: float) -> str:
    if score >= 70:
        return "Healthy"
    elif score >= 40:
        return "Warning"
    else:
        return "Critical"


# ── Alert Generation ──────────────────────────────────────────────────────────

_FAILURE_MODE_NAMES = {
    "Machine failure": "General failure",
    "TWF":  "Tool Wear Failure",
    "HDF":  "Heat Dissipation Failure",
    "PWF":  "Power Failure",
    "OSF":  "Overstrain Failure",
    "RNF":  "Random Failure",
}

def generate_alerts(
    probs: dict[str, float],
    rul: float,
    health_score: float,
) -> list[str]:
    """Generate human-readable alert messages based on predictions."""
    alerts = []

    # Failure mode alerts
    failure_prob = probs.get(PRIMARY_TARGET, 0.0)
    if failure_prob >= ALERT_THRESHOLDS["critical"]:
        alerts.append(
            f"🔴 CRITICAL: Machine failure probability {failure_prob*100:.0f}% "
            f"— immediate inspection required."
        )
    elif failure_prob >= ALERT_THRESHOLDS["warning"]:
        alerts.append(
            f"🟡 WARNING: Machine failure probability {failure_prob*100:.0f}% "
            f"— schedule maintenance soon."
        )

    # Sub-mode alerts
    for label, prob in probs.items():
        if label == PRIMARY_TARGET:
            continue
        if prob >= ALERT_THRESHOLDS["critical"]:
            name = _FAILURE_MODE_NAMES.get(label, label)
            alerts.append(f"🔴 {name} risk: {prob*100:.0f}%")
        elif prob >= ALERT_THRESHOLDS["warning"]:
            name = _FAILURE_MODE_NAMES.get(label, label)
            alerts.append(f"🟡 {name} risk: {prob*100:.0f}%")

    # RUL alerts
    if rul <= 10:
        alerts.append(f"⏱️ URGENT: Only {rul:.0f} cycles remaining before predicted failure.")
    elif rul <= 30:
        alerts.append(f"⏱️ Low RUL: {rul:.0f} cycles remaining — plan maintenance.")

    # Health score alert
    if health_score < 40:
        alerts.append(f"💔 Health Score: {health_score:.0f}/100 — machine in critical state.")

    return alerts
