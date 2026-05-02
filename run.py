"""
Single entry point for the PoTSml pipeline.

Usage: python run.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_dataset
from features import build_features, get_feature_columns
from train import train_and_evaluate, print_results
from evaluate import compute_metrics
from plots import generate_all_plots
from config import (
    N_PATIENTS, N_DAYS, SEED,
    DEFAULT_HORIZON, FORECAST_HORIZONS, N_FOLDS,
    DATA_DIR, MODEL_DIR, XGB_PARAMS,
)


def run_horizon_sensitivity(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Sweep forecasting horizons [5, 10, 15, 30 min] and return per-horizon
    fold metrics for plotting. Uses default XGB params (no tuning) for speed.
    Returns: {horizon: {"roc_auc": [...folds], "pr_auc": [...folds]}}
    """
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb

    gkf = GroupKFold(n_splits=N_FOLDS)
    horizon_results = {}

    print(f"\n{'Horizon':>10}  {'ROC-AUC':>12}  {'PR-AUC':>12}  {'Recall':>10}")
    print("-" * 50)

    for horizon in FORECAST_HORIZONS:
        df_h = build_features(df, horizon=horizon)
        X_h  = df_h[feature_cols]
        y_h  = df_h["label"].values
        g_h  = df_h["patient_id"].values

        fold_roc, fold_pr, fold_rec = [], [], []
        for tr, te in gkf.split(X_h, y_h, g_h):
            m = xgb.XGBClassifier(**XGB_PARAMS)
            m.fit(X_h.iloc[tr], y_h[tr])
            met = compute_metrics(y_h[te], m.predict_proba(X_h.iloc[te])[:, 1])
            fold_roc.append(met["roc_auc"])
            fold_pr.append(met["pr_auc"])
            fold_rec.append(met["recall"])

        horizon_results[horizon] = {"roc_auc": fold_roc, "pr_auc": fold_pr}
        mark = "  ← default" if horizon == DEFAULT_HORIZON else ""
        print(f"{horizon:>10}  "
              f"{np.mean(fold_roc):.3f}±{np.std(fold_roc):.3f}  "
              f"{np.mean(fold_pr):.3f}±{np.std(fold_pr):.3f}  "
              f"{np.mean(fold_rec):.3f}±{np.std(fold_rec):.3f}"
              f"{mark}")

    return horizon_results


def main():
    print("=" * 60)
    print("  PoTSml v2 — POTS Episode Forecasting Pipeline")
    print("=" * 60)
    print(f"  Patients: {N_PATIENTS} | Days/patient: {N_DAYS} | Seed: {SEED}")
    print(f"  Horizon: {DEFAULT_HORIZON} min | CV folds: {N_FOLDS}")

    # 1. Generate data
    print(f"\n[1/5] Generating synthetic dataset ({N_PATIENTS}p × {N_DAYS}d)...")
    df = generate_dataset(n_patients=N_PATIENTS, n_days=N_DAYS, seed=SEED)
    print(f"  {len(df):,} rows | symptom rate: {df['symptom'].mean():.3f} "
          f"| standing rate: {df['posture'].mean():.3f}")
    os.makedirs(DATA_DIR, exist_ok=True)
    df.drop(columns=["_latent_state"]).to_csv(f"{DATA_DIR}/pots_dataset.csv", index=False)

    # 2. Feature engineering
    print(f"\n[2/5] Building features (horizon={DEFAULT_HORIZON} min)...")
    df_feat      = build_features(df, horizon=DEFAULT_HORIZON)
    feature_cols = get_feature_columns()
    print(f"  {len(df_feat):,} rows | {len(feature_cols)} features "
          f"| label prevalence: {df_feat['label'].mean():.3f}")

    # 3. Train + evaluate
    print(f"\n[3/5] Training — {N_FOLDS}-fold GroupKFold + tuning...")
    print("  (takes a few minutes due to nested CV)")
    results = train_and_evaluate(df_feat, feature_cols, n_folds=N_FOLDS, tune=True)

    # 4. Results table
    print("\n[4/5] Results")
    print_results(results)
    if results["best_xgb_params"]:
        print(f"\n  Calibrated model → {MODEL_DIR}/xgboost_calibrated.pkl")

    # 5. Horizon sensitivity
    print("\n[5/5] Horizon sensitivity analysis")
    print("=" * 60)
    horizon_results = run_horizon_sensitivity(df, feature_cols)

    # 6. Plots
    print("\n[6/6] Generating plots")
    generate_all_plots(df, results, horizon_results)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
