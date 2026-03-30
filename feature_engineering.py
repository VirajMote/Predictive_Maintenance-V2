"""
feature_engineering.py
─────────────────────
All feature construction lives here.

Three categories:
  1. Physics-informed features   — domain-derived combinations
  2. Rolling statistical features — smoothed signals, volatility
  3. RUL label construction       — backward-from-failure labeling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    SENSOR_COLS, CATEGORICAL, DROP_COLS, TARGET_COLS,
    PRIMARY_TARGET, ROLLING_WINDOWS, RUL_CAP,
)


# ── 1. Physics-Informed Features ─────────────────────────────────────────────

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-derived features that carry more signal than raw sensors.

    - Temp Difference   : process-air delta — proxy for thermal stress
    - Power             : torque × angular velocity (actual mechanical load)
    - Strain Measure    : tool wear × torque — captures cumulative fatigue
    - Temp/Speed Ratio  : thermal load per unit speed
    """
    df = df.copy()

    df["feat_temp_diff"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    df["feat_power_W"] = df["Torque [Nm]"] * (
        df["Rotational speed [rpm]"] * 2 * np.pi / 60
    )
    df["feat_strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["feat_temp_speed_ratio"] = (
        df["Process temperature [K]"] / (df["Rotational speed [rpm]"] + 1e-6)
    )

    return df


# ── 2. Rolling Statistical Features ──────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Rolling mean and std for each sensor, over multiple window sizes.

    If group_col is provided (e.g. machine ID), rolling is computed
    per-group to avoid bleeding across machines.

    NOTE: This function assumes the DataFrame is already sorted by time.
    """
    df = df.copy()

    def _roll(data: pd.DataFrame) -> pd.DataFrame:
        for col in SENSOR_COLS:
            for w in windows:
                data[f"feat_{col}_roll_mean_{w}"] = (
                    data[col].rolling(w, min_periods=1).mean()
                )
                data[f"feat_{col}_roll_std_{w}"] = (
                    data[col].rolling(w, min_periods=1).std().fillna(0)
                )
        return data

    if group_col and group_col in df.columns:
        df = df.groupby(group_col, group_keys=False).apply(_roll)
    else:
        df = _roll(df)

    return df


# ── 3. RUL Label Construction ─────────────────────────────────────────────────

def compute_rul(
    df: pd.DataFrame,
    failure_col: str = PRIMARY_TARGET,
    cap: int = RUL_CAP,
    group_col: str | None = None,
) -> pd.Series:
    """
    Construct Remaining Useful Life (RUL) labels from failure timestamps.

    Strategy:
      - Work backwards from each failure event.
      - RUL at failure = 0; at T-minus-N = N.
      - Cap RUL at `cap` so healthy machines don't get unbounded values.
      - Piecewise linear: no degradation assumption beyond the cap.

    Args:
        df:          DataFrame sorted by time, containing failure_col.
        failure_col: Column name of the binary failure indicator.
        cap:         Maximum RUL value assigned (healthy baseline).
        group_col:   If provided, compute RUL per machine group.

    Returns:
        pd.Series of RUL values, same index as df.
    """

    def _rul_for_group(grp: pd.DataFrame) -> pd.Series:
        rul = np.full(len(grp), cap, dtype=float)
        failure_indices = grp.index[grp[failure_col] == 1].tolist()

        if not failure_indices:
            return pd.Series(rul, index=grp.index)

        # Walk backwards from each failure to assign RUL
        for fi in failure_indices:
            pos = grp.index.get_loc(fi)
            for j in range(pos + 1):
                steps_before = pos - j
                rul[j] = min(rul[j], steps_before)

        return pd.Series(np.minimum(rul, cap), index=grp.index)

    if group_col and group_col in df.columns:
        return df.groupby(group_col, group_keys=False).apply(
            _rul_for_group
        ).sort_index()
    else:
        return _rul_for_group(df)


# ── 4. Full Pipeline ──────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    compute_rul_labels: bool = True,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    End-to-end feature construction.

    Steps:
      1. Drop non-model columns
      2. Encode categoricals
      3. Add physics features
      4. Add rolling features
      5. Optionally compute RUL labels

    Args:
        df:                 Raw DataFrame from CSV.
        compute_rul_labels: Whether to append RUL column (True for training).
        group_col:          Machine ID column for grouped rolling (optional).

    Returns:
        DataFrame ready for model input, with feature columns and targets.
    """
    df = df.copy()

    # Drop identifiers
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Encode Type: L=0, M=1, H=2
    if "Type" in df.columns:
        le = LabelEncoder()
        df["Type"] = le.fit_transform(df["Type"])

    # Physics features
    df = add_physics_features(df)

    # Rolling features (time-ordered assumed)
    df = add_rolling_features(df, group_col=group_col)

    # RUL label
    if compute_rul_labels and PRIMARY_TARGET in df.columns:
        df["RUL"] = compute_rul(df, group_col=group_col)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all model input columns: original sensors + engineered features.
    Excludes target columns and RUL.
    """
    exclude = set(TARGET_COLS + ["RUL"])
    return [c for c in df.columns if c not in exclude]
