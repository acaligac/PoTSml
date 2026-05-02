"""
Automated checks for data leakage.

These tests verify that the pipeline doesn't accidentally introduce
future information or label-correlated features.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_patient
from features import build_features, get_feature_columns


def test_no_future_leakage_in_features():
    """
    Verify that feature values at time t don't change when future data changes.

    Method: generate a patient, compute features, then corrupt data after
    time t=500 and recompute. Features at t<=500 must be identical.
    """
    df1 = generate_patient(0, seed=99)
    df1_feat = build_features(df1)

    # Corrupt future data
    df2 = df1.copy()
    df2.loc[df2.index > 500, "heart_rate"] = 999.0
    df2.loc[df2.index > 500, "hrv_proxy"] = 999.0
    df2_feat = build_features(df2)

    feature_cols = get_feature_columns()

    # Find rows that exist in both and have time index <= 500
    # (row indices may differ after dropna, so match on time)
    merged = df1_feat.merge(df2_feat, on=["time", "patient_id"], suffixes=("_orig", "_corrupt"))

    cutoff_time = df1.loc[500, "time"]
    before = merged[merged["time"] <= cutoff_time]

    for col in feature_cols:
        orig = before[f"{col}_orig"].values
        corrupt = before[f"{col}_corrupt"].values
        if not np.allclose(orig, corrupt, equal_nan=True):
            print(f"FAIL: Feature '{col}' leaks future data")
            return False

    print("PASS: No future leakage detected in features")
    return True


def test_label_not_deterministic_from_features():
    """
    Check that features don't perfectly predict the label.

    A simple correlation check — if any single feature has >0.9 correlation
    with the label, something is likely wrong.
    """
    df = generate_patient(0, seed=42)
    df = build_features(df)

    feature_cols = get_feature_columns()
    max_corr = 0
    max_col = ""

    for col in feature_cols:
        corr = abs(df[col].corr(df["label"]))
        if corr > max_corr:
            max_corr = corr
            max_col = col

    if max_corr > 0.9:
        print(f"FAIL: Feature '{max_col}' has {max_corr:.3f} correlation with label (likely leakage)")
        return False

    print(f"PASS: Max feature-label correlation = {max_corr:.3f} (feature: {max_col})")
    return True


def test_latent_state_not_in_features():
    """Verify the latent state column is never used as a feature."""
    feature_cols = get_feature_columns()
    if "_latent_state" in feature_cols:
        print("FAIL: _latent_state is in feature columns")
        return False
    print("PASS: _latent_state not in feature columns")
    return True


def test_symptom_not_in_features():
    """Verify the raw symptom column is never used as a feature."""
    feature_cols = get_feature_columns()
    if "symptom" in feature_cols:
        print("FAIL: symptom is in feature columns")
        return False
    print("PASS: symptom not in feature columns")
    return True


if __name__ == "__main__":
    results = [
        test_no_future_leakage_in_features(),
        test_label_not_deterministic_from_features(),
        test_latent_state_not_in_features(),
        test_symptom_not_in_features(),
    ]

    print(f"\n{'='*40}")
    print(f"  {sum(results)}/{len(results)} tests passed")
    print(f"{'='*40}")

    sys.exit(0 if all(results) else 1)
