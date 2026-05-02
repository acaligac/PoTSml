"""
Feature engineering — strictly causal (no future information).

Every feature at time t uses only data from times <= t.
No global means, no full-series statistics, no fillna(0) tricks.

The forecasting label: will a symptom occur in the next HORIZON minutes?
"""

import pandas as pd
import numpy as np
from config import DEFAULT_HORIZON

HORIZON = DEFAULT_HORIZON  # predict symptoms N minutes ahead


def build_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Build features and forecasting label.

    Returns a new DataFrame with features and 'label' column.
    Drops rows where features are undefined (start of each patient's series).
    """
    df = df.copy()

    # --- Forecasting label ---
    # For each patient, label[t] = 1 if any symptom occurs in (t+1, t+horizon]
    df["label"] = (
        df.groupby("patient_id")["symptom"]
        .transform(lambda x: x.shift(-1).rolling(horizon, min_periods=1).max())
    )

    # --- Strictly causal features ---
    # All use .shift() or backward-looking rolling windows

    # Expanding mean of HR up to current time (causal baseline estimate)
    # This replaces the old global mean which leaked future data
    df["hr_expanding_mean"] = (
        df.groupby("patient_id")["heart_rate"]
        .transform(lambda x: x.expanding().mean())
    )
    df["delta_hr"] = df["heart_rate"] - df["hr_expanding_mean"]

    # Rolling statistics (backward-looking, 5-minute window)
    for col in ["heart_rate", "hrv_proxy"]:
        prefix = "hr" if col == "heart_rate" else "hrv"
        grp = df.groupby("patient_id")[col]
        df[f"{prefix}_roll5_mean"] = grp.transform(lambda x: x.rolling(5, min_periods=5).mean())
        df[f"{prefix}_roll5_std"] = grp.transform(lambda x: x.rolling(5, min_periods=5).std())

    # Longer window for trend detection
    df["hr_roll30_mean"] = (
        df.groupby("patient_id")["heart_rate"]
        .transform(lambda x: x.rolling(30, min_periods=30).mean())
    )

    # HR trend: difference between short and long rolling means
    df["hr_trend"] = df["hr_roll5_mean"] - df["hr_roll30_mean"]

    # Lag features
    for lag in [1, 3, 5, 10]:
        df[f"hr_lag{lag}"] = df.groupby("patient_id")["heart_rate"].shift(lag)
        df[f"hrv_lag{lag}"] = df.groupby("patient_id")["hrv_proxy"].shift(lag)

    # Posture: current value + how long patient has been in current posture
    df["posture_duration"] = (
        df.groupby("patient_id")["posture"]
        .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    # Recent posture burden: fraction of last 10 minutes spent standing
    df["posture_burden_10"] = (
        df.groupby("patient_id")["posture"]
        .transform(lambda x: x.rolling(10, min_periods=1).mean())
    )

    # HR acceleration: rate of change of the smoothed signal over last 3 minutes.
    # Using hr_roll5_mean instead of raw heart_rate suppresses sensor noise,
    # which otherwise dominates the raw 3-minute diff.
    df["hr_accel"] = (
        df["hr_roll5_mean"] - df.groupby("patient_id")["hr_roll5_mean"].shift(3)
    )

    # --- Drop rows with undefined features ---
    feature_cols = get_feature_columns()
    df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for modeling."""
    return [
        "delta_hr",
        "heart_rate",
        "hrv_proxy",
        "posture",
        "posture_duration",
        "posture_burden_10",
        "hr_roll5_mean",
        "hr_roll5_std",
        "hrv_roll5_mean",
        "hrv_roll5_std",
        "hr_roll30_mean",
        "hr_trend",
        "hr_accel",
        "hr_lag1", "hr_lag3", "hr_lag5", "hr_lag10",
        "hrv_lag1", "hrv_lag3", "hrv_lag5", "hrv_lag10",
    ]
