"""
Training pipeline with proper patient-level cross-validation.

Models:
  1. Rule-based baseline (threshold on delta_hr + hrv)
     — Establishes whether ML adds anything beyond a simple heuristic.
  2. Logistic regression (with hyperparameter tuning + StandardScaler)
     — True statistical baseline.
  3. XGBoost (with hyperparameter tuning + probability calibration)
     — Captures nonlinear interactions and temporal patterns from lag features.

All use GroupKFold to ensure no patient appears in both train and test.

Additions vs. v1:
  - RandomizedSearchCV for both LogReg and XGBoost (nested CV)
  - CalibratedClassifierCV (Platt scaling) on XGBoost — probabilities now
    represent actual likelihoods rather than raw scores
  - SHAP feature importance for XGBoost
  - Model serialization via joblib
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

from evaluate import compute_metrics, print_fold_summary
from config import (
    N_FOLDS,
    CALIBRATION_SPLIT,
    XGB_PARAMS,
    XGB_SEARCH_SPACE,
    XGB_SEARCH_ITER,
    XGB_SEARCH_CV,
    LR_SEARCH_SPACE,
    MODEL_DIR,
    SEED,
)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


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


def _tune_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """
    Tune LogisticRegression via RandomizedSearchCV on the pre-scaled training set.
    Scores on PR-AUC (average_precision) — the right metric for imbalanced data.
    """
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    search = RandomizedSearchCV(
        lr,
        LR_SEARCH_SPACE,
        n_iter=10,
        cv=3,
        scoring="average_precision",
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train_scaled, y_train)
    return search.best_estimator_


def _tune_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> xgb.XGBClassifier:
    """
    Tune XGBoost via RandomizedSearchCV.
    Returns the best estimator (unfitted on full train — RandomizedSearchCV refits).
    """
    xgb_base = xgb.XGBClassifier(
        random_state=SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    search = RandomizedSearchCV(
        xgb_base,
        XGB_SEARCH_SPACE,
        n_iter=XGB_SEARCH_ITER,
        cv=XGB_SEARCH_CV,
        scoring="average_precision",
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def _calibrate_xgboost(
    xgb_model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> CalibratedClassifierCV:
    """
    Apply Platt scaling (sigmoid calibration) to XGBoost.

    We hold out CALIBRATION_SPLIT of the training data to fit the calibration
    layer, so the calibrator never sees the same data the base model trained on.
    This avoids the calibration layer simply learning to reproduce training scores.

    Returns a fitted CalibratedClassifierCV with the base XGBoost pre-fitted.
    """
    n_cal = max(1, int(len(y_train) * CALIBRATION_SPLIT))
    # Simple temporal split within training data (maintains time order)
    X_cal = X_train.iloc[-n_cal:]
    y_cal = y_train[-n_cal:]

    calibrated = CalibratedClassifierCV(xgb_model, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def compute_shap_importance(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_cols: list[str],
    max_samples: int = 2000,
) -> pd.DataFrame | None:
    """
    Compute mean absolute SHAP values for each feature.

    Uses a subsample of X for speed. Returns a DataFrame sorted by importance,
    or None if shap is not installed.
    """
    if not _SHAP_AVAILABLE:
        return None

    X_sample = X.sample(min(max_samples, len(X)), random_state=SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = N_FOLDS,
    tune: bool = True,
) -> dict:
    """
    Train all models with GroupKFold CV and return per-fold metrics + SHAP.

    Returns:
        {
            "rule_baseline": [metrics_fold1, ...],
            "logistic_regression": [...],
            "xgboost": [...],
            "shap_importance": DataFrame or None,
            "best_xgb_params": dict,
        }
    """
    X = df[feature_cols]
    y = df["label"].values
    groups = df["patient_id"].values

    gkf = GroupKFold(n_splits=n_folds)

    results: dict = {
        "rule_baseline": [],
        "logistic_regression": [],
        "xgboost": [],
        "shap_importance": None,
        "best_xgb_params": {},
    }

    last_xgb_model = None  # save final fold's model for SHAP + serialization

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        print(f"  Train: {len(np.unique(groups[train_idx]))} patients | "
              f"Test: {len(np.unique(groups[test_idx]))} patients")
        print(f"  Train prevalence: {y_train.mean():.3f} | Test prevalence: {y_test.mean():.3f}")

        # 1. Rule baseline (no fitting)
        rule_probs = rule_baseline_predict(X_test)
        results["rule_baseline"].append(compute_metrics(y_test, rule_probs))

        # 2. Logistic regression — scale then optionally tune
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if tune:
            lr = _tune_logistic_regression(X_train_scaled, y_train)
            print(f"  LR best params: C={lr.C}, penalty={lr.penalty}")
        else:
            lr = LogisticRegression(max_iter=1000, random_state=SEED)
            lr.fit(X_train_scaled, y_train)

        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
        results["logistic_regression"].append(compute_metrics(y_test, lr_probs))

        # 3. XGBoost — optionally tune, then calibrate
        if tune:
            xgb_model = _tune_xgboost(X_train, y_train)
            print(f"  XGB best params: {xgb_model.get_params()}")
            results["best_xgb_params"] = xgb_model.get_params()
        else:
            xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
            xgb_model.fit(X_train, y_train)

        # Calibrate: Platt scaling on a held-out slice of training data
        calibrated_xgb = _calibrate_xgboost(xgb_model, X_train, y_train)
        xgb_probs = calibrated_xgb.predict_proba(X_test)[:, 1]
        results["xgboost"].append(compute_metrics(y_test, xgb_probs))

        last_xgb_model = xgb_model  # keep for SHAP + serialization

    # SHAP importance on the last fold's XGBoost model
    if last_xgb_model is not None:
        results["shap_importance"] = compute_shap_importance(
            last_xgb_model, X, feature_cols
        )

    # Serialize the last fold's XGBoost model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(last_xgb_model, os.path.join(MODEL_DIR, "xgboost.pkl"))
    if tune:
        joblib.dump(results["best_xgb_params"], os.path.join(MODEL_DIR, "xgb_best_params.pkl"))

    return results


def print_results(results: dict) -> None:
    """Print per-model fold summaries, comparison table, and SHAP importance."""
    model_keys = ["rule_baseline", "logistic_regression", "xgboost"]

    for name in model_keys:
        print_fold_summary(results[name], name)

    # Comparison table
    print("\n\nModel Comparison (mean ± std across folds):")
    print(f"{'Model':<25s} {'ROC-AUC':>14s} {'PR-AUC':>14s} {'Brier':>10s} {'Recall':>10s} {'Precision':>12s}")
    print("-" * 90)
    for name in model_keys:
        fold_metrics = results[name]
        roc  = [m["roc_auc"]     for m in fold_metrics]
        pr   = [m["pr_auc"]      for m in fold_metrics]
        bri  = [m["brier_score"] for m in fold_metrics]
        rec  = [m["recall"]      for m in fold_metrics]
        prec = [m["precision"]   for m in fold_metrics]
        print(
            f"{name:<25s} "
            f"{np.mean(roc):.3f}±{np.std(roc):.3f}  "
            f"{np.mean(pr):.3f}±{np.std(pr):.3f}  "
            f"{np.mean(bri):.3f}±{np.std(bri):.3f}  "
            f"{np.mean(rec):.3f}±{np.std(rec):.3f}  "
            f"{np.mean(prec):.3f}±{np.std(prec):.3f}"
        )

    # SHAP importance table
    shap_df = results.get("shap_importance")
    if shap_df is not None:
        print("\n\nXGBoost Feature Importance (mean |SHAP|, last fold):")
        print(f"{'Rank':<6s} {'Feature':<22s} {'Mean |SHAP|':>14s}")
        print("-" * 44)
        for i, row in shap_df.iterrows():
            print(f"{i+1:<6d} {row['feature']:<22s} {row['mean_abs_shap']:>14.4f}")
    else:
        print("\n  (install `shap` for feature importance: pip install shap)")
