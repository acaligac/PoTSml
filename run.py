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
from config import (
    N_PATIENTS,
    N_DAYS,
    SEED,
    DEFAULT_HORIZON,
    FORECAST_HORIZONS,
    N_FOLDS,
    DATA_DIR,
    MODEL_DIR,
)


def run_horizon_sensitivity(df, feature_cols_base: list[str]) -> None:
    """
    Sensitivity analysis: how does XGBoost PR-AUC degrade as the forecasting
    horizon grows from 5 to 30 minutes?

    Each horizon produces a fresh label and re-runs 5-fold GroupKFold.
    This is a key result for any paper on this approach.
    """
    print("\n" + "=" * 60)
    print("  Horizon Sensitivity Analysis")
    print(f"  Horizons tested: {FORECAST_HORIZONS} minutes")
    print("=" * 60)

    from sklearn.model_selection import GroupKFold
    import xgboost as xgb
    from config import XGB_PARAMS

    gkf = GroupKFold(n_splits=N_FOLDS)

    print(f"\n{'Horizon (min)':<16s} {'ROC-AUC':>12s} {'PR-AUC':>12s} {'Recall':>10s}")
    print("-" * 52)

    for horizon in FORECAST_HORIZONS:
        df_h = build_features(df, horizon=horizon)
        X_h = df_h[feature_cols_base]
        y_h = df_h["label"].values
        groups_h = df_h["patient_id"].values

        fold_roc, fold_pr, fold_rec = [], [], []
        for train_idx, test_idx in gkf.split(X_h, y_h, groups_h):
            X_tr, X_te = X_h.iloc[train_idx], X_h.iloc[test_idx]
            y_tr, y_te = y_h[train_idx], y_h[test_idx]

            m = xgb.XGBClassifier(**XGB_PARAMS)
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_te)[:, 1]

            metrics = compute_metrics(y_te, probs)
            fold_roc.append(metrics["roc_auc"])
            fold_pr.append(metrics["pr_auc"])
            fold_rec.append(metrics["recall"])

        marker = " <-- default" if horizon == DEFAULT_HORIZON else ""
        print(
            f"{horizon:<16d} "
            f"{np.mean(fold_roc):.3f}±{np.std(fold_roc):.3f}  "
            f"{np.mean(fold_pr):.3f}±{np.std(fold_pr):.3f}  "
            f"{np.mean(fold_rec):.3f}±{np.std(fold_rec):.3f}"
            f"{marker}"
        )


def main():
    print("=" * 60)
    print("  PoTSml v2 — POTS Episode Forecasting Pipeline")
    print("=" * 60)
    print(f"  Patients: {N_PATIENTS} | Days per patient: {N_DAYS} | Seed: {SEED}")
    print(f"  Forecast horizon: {DEFAULT_HORIZON} min | CV folds: {N_FOLDS}")

    # Step 1: Generate data
    print(f"\n[1/5] Generating synthetic dataset ({N_PATIENTS} patients × {N_DAYS} days)...")
    df = generate_dataset(n_patients=N_PATIENTS, n_days=N_DAYS, seed=SEED)
    print(f"  {len(df):,} rows, {df['patient_id'].nunique()} patients")
    print(f"  Symptom rate:  {df['symptom'].mean():.3f}")
    print(f"  Standing rate: {df['posture'].mean():.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.drop(columns=["_latent_state"]).to_csv(f"{DATA_DIR}/pots_dataset.csv", index=False)

    # Step 2: Feature engineering
    print(f"\n[2/5] Building features (horizon={DEFAULT_HORIZON} min)...")
    df_feat = build_features(df, horizon=DEFAULT_HORIZON)
    feature_cols = get_feature_columns()
    print(f"  {len(df_feat):,} rows after dropping incomplete windows")
    print(f"  {len(feature_cols)} features")
    print(f"  Label prevalence: {df_feat['label'].mean():.3f}")

    # Step 3: Train and evaluate (with tuning + calibration)
    print(f"\n[3/5] Training models with {N_FOLDS}-fold GroupKFold + hyperparameter tuning...")
    print("  (This may take a few minutes due to nested CV)")
    results = train_and_evaluate(df_feat, feature_cols, n_folds=N_FOLDS, tune=True)

    # Step 4: Print results
    print("\n[4/5] Results")
    print_results(results)

    if results["best_xgb_params"]:
        print(f"\n  XGBoost model saved to {MODEL_DIR}/xgboost.pkl")

    # Step 5: Horizon sensitivity analysis
    print(f"\n[5/5] Running horizon sensitivity analysis...")
    run_horizon_sensitivity(df, feature_cols)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
