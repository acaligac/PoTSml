"""
Evaluation utilities.

Reports the metrics that actually matter for a clinical alerting system:
  - ROC-AUC: overall discrimination ability
  - PR-AUC: critical for imbalanced data (rare symptom events)
  - Precision/Recall at a chosen threshold: what the system would
    actually deliver if deployed

All evaluation uses patient-level splits to prevent leakage.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all relevant metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "prevalence": y_true.mean(),
        "threshold": threshold,
    }

    return metrics


def print_metrics(metrics: dict, model_name: str) -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}  (random baseline: {metrics['prevalence']:.4f})")
    print(f"  Precision:  {metrics['precision']:.4f}  @ threshold={metrics['threshold']}")
    print(f"  Recall:     {metrics['recall']:.4f}  @ threshold={metrics['threshold']}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"{'='*50}")


def print_fold_summary(all_metrics: list[dict], model_name: str) -> None:
    """Summarize metrics across CV folds."""
    print(f"\n{'#'*55}")
    print(f"  {model_name} — {len(all_metrics)}-fold summary")
    print(f"{'#'*55}")

    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key:<12s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"  prevalence    {np.mean([m['prevalence'] for m in all_metrics]):.4f}")
    print(f"{'#'*55}")
