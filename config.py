"""
Central configuration for the Predictive Maintenance System.
All hyperparameters, paths, and constants live here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "outputs"
LOG_DIR     = BASE_DIR / "logs"

RAW_DATA_PATH   = DATA_DIR / "ai4i2020.csv"
MODEL_PKG_PATH  = MODEL_DIR / "model_package.pkl"

# ── Dataset ──────────────────────────────────────────────────────────────────
DROP_COLS    = ["UDI", "Product ID"]
CATEGORICAL  = ["Type"]

TARGET_COLS  = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
PRIMARY_TARGET = "Machine failure"

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ── RUL ──────────────────────────────────────────────────────────────────────
RUL_CAP = 200          # cycles — clamp RUL to this max (healthy baseline)
DEGRADATION_WINDOW = 30  # look-back window for degradation slope features

# ── Feature Engineering ──────────────────────────────────────────────────────
ROLLING_WINDOWS = [5, 10, 20]   # for rolling mean / std

# ── Training ─────────────────────────────────────────────────────────────────
CV_FOLDS     = 5
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Random Forest ────────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":  200,
    "max_depth":     None,
    "min_samples_split": 5,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

# ── XGBoost ──────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":     "logloss",
    "random_state":    RANDOM_STATE,
    "n_jobs":          -1,
}

# scale_pos_weight is set dynamically per label from class counts

# ── LSTM ─────────────────────────────────────────────────────────────────────
LSTM_SEQ_LEN    = 30       # time steps fed into the LSTM
LSTM_UNITS      = [64, 32] # stacked LSTM hidden sizes
LSTM_DROPOUT    = 0.3
LSTM_EPOCHS     = 50
LSTM_BATCH_SIZE = 64
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 10       # early stopping patience

# ── Inference ─────────────────────────────────────────────────────────────────
# Thresholds are tuned per label via PR curve — these are fallback defaults
DEFAULT_THRESHOLD = 0.5

# Health score weights (must sum to 1)
HEALTH_WEIGHTS = {
    "Machine failure": 0.40,
    "TWF":             0.15,
    "HDF":             0.15,
    "PWF":             0.15,
    "OSF":             0.10,
    "RNF":             0.05,
}

ALERT_THRESHOLDS = {
    "critical": 0.75,   # failure prob above this → RED alert
    "warning":  0.40,   # failure prob above this → YELLOW alert
}
