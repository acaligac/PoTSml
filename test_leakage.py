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


def test_label_correctness():
    """
    Verify the label construction is a strictly future window.

    Properties checked:
      1. label[t] == 1 iff at least one symptom occurs in (t+1, t+horizon]
         — checked by manually scanning symptom[t+1..t+horizon] for each row.
      2. label[t] never fires on a symptom at time t itself (no same-tick leakage).
      3. label[t] never fires on symptoms strictly before t (no past leakage).
    """
    from config import DEFAULT_HORIZON

    df = generate_patient(0, n_days=3, seed=7)
    df_feat = build_features(df, horizon=DEFAULT_HORIZON)

    # Align raw symptom series to the feature DataFrame via time index
    sym = df.set_index("time")["symptom"]

    n_checked = 0
    for _, row in df_feat.iterrows():
        t = row["time"]
        patient = row["patient_id"]
        label = int(row["label"])

        # Build the future window (t+1 .. t+horizon) in the raw series
        future_times = pd.date_range(t, periods=DEFAULT_HORIZON + 1, freq="1min")[1:]
        future_syms = sym.reindex(future_times).fillna(0).values
        expected = int(future_syms.max())

        if label != expected:
            print(f"FAIL: label mismatch at {t} (patient {patient}): "
                  f"label={label}, expected={expected}")
            return False

        n_checked += 1

    print(f"PASS: Label correctness verified across {n_checked} rows "
          f"(horizon={DEFAULT_HORIZON} min)")
    return True


if __name__ == "__main__":
    results = [
        test_no_future_leakage_in_features(),
        test_label_not_deterministic_from_features(),
        test_latent_state_not_in_features(),
        test_symptom_not_in_features(),
        test_label_correctness(),
    ]

    print(f"\n{'='*40}")
    print(f"  {sum(results)}/{len(results)} tests passed")
    print(f"{'='*40}")

    sys.exit(0 if all(results) else 1)
