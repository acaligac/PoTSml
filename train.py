"""
Training pipeline with proper patient-level cross-validation.

Models:
  1. Rule-based baseline (threshold on delta_hr + hrv)
     — Establishes whether ML adds anything beyond a simple heuristic.
  2. Logistic regression
     — True statistical baseline. If this matches XGBoost, the problem
       is likely linear and XGBoost is unnecessary.
  3. XGBoost
     — Captures nonlinear interactions and temporal patterns from lag features.

All use GroupKFold to ensure no patient appears in both train and test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.evaluate import compute_metrics, print_fold_summary


N_FOLDS = 5


def rule_baseline_predict(X: pd.DataFrame) -> np.ndarray:
    """
    Simple threshold rule: predict symptom if HR is elevated and HRV is low.
    This is the "would a doctor eyeballing the chart catch this?" baseline.
    """
    prob = np.zeros(len(X))
    elevated_hr = X["delta_hr"] > 15
    low_hrv = X["hrv_proxy"] < 30
    standing = X["posture"] == 1

    prob[elevated_hr & low_hrv] = 0.7
    prob[elevated_hr & ~low_hrv] = 0.3
    prob[~elevated_hr & low_hrv] = 0.2
    prob[standing & ~elevated_hr & ~low_hrv] = 0.15

    return prob


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = N_FOLDS,
) -> dict[str, list[dict]]:
    """
    Train all models with GroupKFold CV and return per-fold metrics.
    """
    X = df[feature_cols]
    y = df["label"].values
    groups = df["patient_id"].values

    gkf = GroupKFold(n_splits=n_folds)

    results = {"rule_baseline": [], "logistic_regression": [], "xgboost": []}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        print(f"  Train patients: {np.unique(groups[train_idx]).tolist()[:5]}... ({len(np.unique(groups[train_idx]))} total)")
        print(f"  Test patients:  {np.unique(groups[test_idx]).tolist()[:5]}... ({len(np.unique(groups[test_idx]))} total)")
        print(f"  Train prevalence: {y_train.mean():.3f}, Test prevalence: {y_test.mean():.3f}")

        # 1. Rule baseline
        rule_probs = rule_baseline_predict(X_test)
        results["rule_baseline"].append(compute_metrics(y_test, rule_probs))

        # 2. Logistic regression (needs scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
        results["logistic_regression"].append(compute_metrics(y_test, lr_probs))

        # 3. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train, verbose=False)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        results["xgboost"].append(compute_metrics(y_test, xgb_probs))

    return results


def print_results(results: dict[str, list[dict]]) -> None:
    """Print summary for all models."""
    for name, fold_metrics in results.items():
        print_fold_summary(fold_metrics, name)

    # Comparison table
    print("\n\nModel Comparison (mean ± std across folds):")
    print(f"{'Model':<25s} {'ROC-AUC':>12s} {'PR-AUC':>12s} {'F1':>12s}")
    print("-" * 65)
    for name, fold_metrics in results.items():
        roc = [m["roc_auc"] for m in fold_metrics]
        pr = [m["pr_auc"] for m in fold_metrics]
        f1 = [m["f1"] for m in fold_metrics]
        print(f"{name:<25s} {np.mean(roc):.4f}±{np.std(roc):.3f} {np.mean(pr):.4f}±{np.std(pr):.3f} {np.mean(f1):.4f}±{np.std(f1):.3f}")
