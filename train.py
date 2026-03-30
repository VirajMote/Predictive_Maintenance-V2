"""
train.py
────────
Training orchestrator. Runs the full pipeline:

  1. Load & validate data
  2. Feature engineering
  3. Train/val/test split
  4. Baseline: RF + XGB (CV evaluation + final model)
  5. Advanced: LSTM (train + evaluate)
  6. Model comparison table
  7. Save best model package

Run:
    python train.py --model all --data data/ai4i2020.csv
    python train.py --model rf  --data data/ai4i2020.csv
    python train.py --model lstm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Local imports ─────────────────────────────────────────────────────────────
from config import (
    RAW_DATA_PATH, MODEL_PKG_PATH, MODEL_DIR, OUTPUT_DIR, LOG_DIR,
    TARGET_COLS, PRIMARY_TARGET, RANDOM_STATE, TEST_SIZE,
    LSTM_SEQ_LEN, RUL_CAP,
)
from feature_engineering import build_features, get_feature_columns
from baseline_model import (
    evaluate_classifier_cv,
    train_final_classifiers,
    train_rul_model,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "train.log"),
    ],
)
logger = logging.getLogger("train")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_and_validate(path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    required = set(TARGET_COLS + [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type",
    ])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Failure rate  : {df[PRIMARY_TARGET].mean()*100:.2f}%")
    for col in TARGET_COLS:
        pos = df[col].sum()
        logger.info(f"  {col:20s}: {pos} positives ({pos/len(df)*100:.2f}%)")

    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def make_splits(
    df_features: pd.DataFrame,
    feature_cols: list[str],
) -> tuple:
    """
    Stratified train / val / test split (70 / 15 / 15).

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test,
        rul_train, rul_val, rul_test
    """
    X   = df_features[feature_cols]
    y   = df_features[TARGET_COLS]
    rul = df_features["RUL"] if "RUL" in df_features.columns else pd.Series(
        np.full(len(df_features), RUL_CAP), index=df_features.index
    )

    strat = y[PRIMARY_TARGET]

    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test, rul_tv, rul_test = train_test_split(
        X, y, rul, test_size=0.15, stratify=strat, random_state=RANDOM_STATE
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val, rul_train, rul_val = train_test_split(
        X_tv, y_tv, rul_tv,
        test_size=0.176,   # 0.15 / 0.85 ≈ 17.6% of train+val → 15% overall
        stratify=y_tv[PRIMARY_TARGET],
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, rul_train, rul_val, rul_test


# ── Baseline Training ─────────────────────────────────────────────────────────

def run_baseline(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    rul_train, rul_val, rul_test,
    model_name: str,
) -> dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training baseline: {model_name}")
    logger.info(f"{'='*60}")

    # CV evaluation
    X_all_tv = pd.concat([X_train, X_val])
    y_all_tv = pd.concat([y_train, y_val])
    cv_results = evaluate_classifier_cv(X_all_tv, y_all_tv, model_name)

    # Final model training
    models, thresholds, scaler = train_final_classifiers(
        X_train, y_train, X_val, y_val, model_name
    )

    # RUL regressor — reuse the same scaler
    rul_results = train_rul_model(
        X_train, rul_train, X_val, rul_val, scaler, model_name
    )

    # Test set evaluation
    X_test_s = scaler.transform(X_test)
    test_preds = np.zeros((len(X_test), len(TARGET_COLS)), dtype=int)
    test_probs = np.zeros((len(X_test), len(TARGET_COLS)), dtype=float)

    for i, col in enumerate(TARGET_COLS):
        prob = models[col].predict_proba(X_test_s)[:, 1]
        test_probs[:, i] = prob
        test_preds[:, i] = (prob >= thresholds[col]).astype(int)

    from sklearn.metrics import f1_score
    test_wf1 = f1_score(y_test.values, test_preds, average="weighted", zero_division=0)
    logger.info(f"[{model_name}] Test Weighted F1: {test_wf1*100:.2f}%")

    return {
        "mode":                   model_name,
        "scaler":                 scaler,
        "classifier_models":      models,
        "classifier_thresholds":  thresholds,
        "rul_model":              rul_results["model"],
        "cv_weighted_f1":         cv_results["weighted_f1"],
        "test_weighted_f1":       test_wf1,
        "rul_mae":                rul_results["mae"],
        "rul_rmse":               rul_results["rmse"],
    }


# ── LSTM Training ─────────────────────────────────────────────────────────────

def run_lstm(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    rul_train, rul_val, rul_test,
    feature_cols: list[str],
) -> dict:
    try:
        from lstm_model import (
            build_sequences,
            train_lstm_classifier,
            evaluate_lstm_classifier,
            train_lstm_regressor,
        )
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        logger.error(f"LSTM dependencies missing: {e}")
        return {}

    logger.info(f"\n{'='*60}")
    logger.info("Training LSTM")
    logger.info(f"{'='*60}")

    # Fit scaler on training data only
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    y_train_arr = y_train[TARGET_COLS].values
    y_val_arr   = y_val[TARGET_COLS].values
    y_test_arr  = y_test[TARGET_COLS].values

    # Build sequences
    X_train_seq, y_train_seq = build_sequences(X_train_s, y_train_arr)
    X_val_seq,   y_val_seq   = build_sequences(X_val_s,   y_val_arr)
    X_test_seq,  y_test_seq  = build_sequences(X_test_s,  y_test_arr)

    rul_train_arr = rul_train.values
    rul_val_arr   = rul_val.values
    rul_test_arr  = rul_test.values
    _, rul_train_seq = build_sequences(X_train_s, rul_train_arr.reshape(-1, 1))
    _, rul_val_seq   = build_sequences(X_val_s,   rul_val_arr.reshape(-1, 1))
    _, rul_test_seq  = build_sequences(X_test_s,  rul_test_arr.reshape(-1, 1))

    rul_train_seq = rul_train_seq.flatten()
    rul_val_seq   = rul_val_seq.flatten()
    rul_test_seq  = rul_test_seq.flatten()

    logger.info(f"Sequence shapes — train: {X_train_seq.shape}, val: {X_val_seq.shape}")

    # Train classifier
    clf_model, thresholds, clf_history = train_lstm_classifier(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, scaler
    )

    # Evaluate on test
    clf_test = evaluate_lstm_classifier(clf_model, X_test_seq, y_test_seq, thresholds)

    # Train RUL regressor
    rul_model, rul_metrics = train_lstm_regressor(
        X_train_seq, rul_train_seq, X_val_seq, rul_val_seq
    )

    return {
        "mode":                  "LSTM",
        "scaler":                scaler,
        "classifier_models":     clf_model,   # single Keras model (not a dict)
        "classifier_thresholds": thresholds,
        "rul_model":             rul_model,
        "test_weighted_f1":      clf_test["weighted_f1"],
        "rul_mae":               rul_metrics["mae"],
        "rul_rmse":              rul_metrics["rmse"],
        "training_history":      clf_history,
    }


# ── Comparison Table ──────────────────────────────────────────────────────────

def print_comparison(results: dict[str, dict]):
    logger.info(f"\n{'='*60}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*60}")
    header = f"{'Model':<8} {'CV F1':>8} {'Test F1':>8} {'RUL MAE':>10} {'RUL RMSE':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, r in results.items():
        cv_f1   = r.get("cv_weighted_f1", float("nan"))
        test_f1 = r.get("test_weighted_f1", float("nan"))
        mae     = r.get("rul_mae", float("nan"))
        rmse    = r.get("rul_rmse", float("nan"))
        logger.info(
            f"{name:<8} {cv_f1*100:>7.2f}% {test_f1*100:>7.2f}% "
            f"{mae:>9.2f}  {rmse:>9.2f}"
        )


# ── Save Package ──────────────────────────────────────────────────────────────

def save_package(result: dict, feature_cols: list[str]):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pkg = {
        "mode":                   result["mode"],
        "scaler":                 result["scaler"],
        "classifier_models":      result["classifier_models"],
        "classifier_thresholds":  result["classifier_thresholds"],
        "rul_model":              result["rul_model"],
        "feature_columns":        feature_cols,
    }
    joblib.dump(pkg, MODEL_PKG_PATH)
    logger.info(f"\nModel package saved → {MODEL_PKG_PATH}")

    # Save metrics as JSON for monitoring / CI
    metrics = {k: v for k, v in result.items()
               if isinstance(v, (int, float, str))}
    metrics_path = OUTPUT_DIR / "metrics.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Predictive Maintenance models")
    p.add_argument("--model", choices=["rf", "xgb", "lstm", "all"], default="all")
    p.add_argument("--data",  type=Path, default=RAW_DATA_PATH)
    p.add_argument("--save",  choices=["rf", "xgb", "lstm", "best"], default="best",
                   help="Which model to save as the production package.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load & engineer features
    raw_df       = load_and_validate(args.data)
    df_features  = build_features(raw_df, compute_rul_labels=True)
    feature_cols = get_feature_columns(df_features)

    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:8]} ...")

    # Split
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     rul_train, rul_val, rul_test) = make_splits(df_features, feature_cols)

    all_results = {}

    # ── Baseline models ────────────────────────────────────────────────────
    if args.model in ("rf", "all"):
        all_results["RF"] = run_baseline(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            rul_train, rul_val, rul_test,
            model_name="RF",
        )

    if args.model in ("xgb", "all"):
        all_results["XGB"] = run_baseline(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            rul_train, rul_val, rul_test,
            model_name="XGB",
        )

    # ── LSTM ────────────────────────────────────────────────────────────────
    if args.model in ("lstm", "all"):
        lstm_result = run_lstm(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            rul_train, rul_val, rul_test,
            feature_cols,
        )
        if lstm_result:
            all_results["LSTM"] = lstm_result

    if not all_results:
        logger.error("No models trained. Exiting.")
        sys.exit(1)

    # ── Comparison ──────────────────────────────────────────────────────────
    print_comparison(all_results)

    # ── Save ────────────────────────────────────────────────────────────────
    if args.save == "best":
        best = max(all_results.items(), key=lambda kv: kv[1].get("test_weighted_f1", 0))
        logger.info(f"\nBest model: {best[0]} (Test F1: {best[1]['test_weighted_f1']*100:.2f}%)")
        save_package(best[1], feature_cols)
    elif args.save in all_results:
        save_package(all_results[args.save], feature_cols)
    else:
        logger.warning(f"Requested save model '{args.save}' was not trained. Saving best.")
        best = max(all_results.items(), key=lambda kv: kv[1].get("test_weighted_f1", 0))
        save_package(best[1], feature_cols)


if __name__ == "__main__":
    main()
