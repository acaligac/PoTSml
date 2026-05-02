"""
Evaluation utilities.

Reports the metrics that actually matter for a clinical alerting system:
  - ROC-AUC: overall discrimination ability
  - PR-AUC: critical for imbalanced data (rare symptom events)
  - Brier score: proper scoring rule for calibration quality
  - Precision/Recall at a clinically chosen threshold: what the system would
    actually deliver if deployed

The operating threshold is selected by find_clinical_threshold(), which finds
the highest-precision threshold that still meets a minimum recall target.
At ~5-8% prevalence, the default 0.5 threshold is almost always wrong.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from config import CLINICAL_RECALL_TARGET


def find_clinical_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_target: float = CLINICAL_RECALL_TARGET,
) -> float:
    """
    Find the highest-precision decision threshold that meets a minimum recall target.

    In a clinical alerting system, missing a real episode (false negative) is more
    costly than a false alarm (false positive). We therefore fix a minimum recall
    (sensitivity) and find the threshold with the best precision subject to that
    constraint. This is the correct way to set an operating threshold at low
    prevalence — not blindly using 0.5.

    Returns the threshold value (float between 0 and 1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns one extra element (at recall=1.0 boundary)
    # Align to threshold array length
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Among all thresholds that achieve recall >= target, pick the one with
    # the highest precision (fewest false alarms)
    mask = recalls >= recall_target
    if not mask.any():
        # Can't achieve target recall — fall back to threshold maximising F1
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        return float(thresholds[np.argmax(f1)])

    best_idx = np.argmax(precisions[mask])
    return float(thresholds[mask][best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    recall_target: float = CLINICAL_RECALL_TARGET,
) -> dict:
    """
    Compute all relevant metrics.

    If threshold is None, it is chosen automatically via find_clinical_threshold
    to achieve recall_target with maximum precision.
    """
    if threshold is None:
        threshold = find_clinical_threshold(y_true, y_prob, recall_target)

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "prevalence": float(y_true.mean()),
        "threshold": threshold,
    }


def print_metrics(metrics: dict, model_name: str) -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.4f}  (random baseline: {metrics['prevalence']:.4f})")
    print(f"  Brier score:  {metrics['brier_score']:.4f}  (lower = better calibrated)")
    print(f"  Precision:    {metrics['precision']:.4f}  @ threshold={metrics['threshold']:.3f}")
    print(f"  Recall:       {metrics['recall']:.4f}  @ threshold={metrics['threshold']:.3f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(f"{'='*55}")


def print_fold_summary(all_metrics: list[dict], model_name: str) -> None:
    """Summarize metrics across CV folds."""
    print(f"\n{'#'*60}")
    print(f"  {model_name} — {len(all_metrics)}-fold summary")
    print(f"{'#'*60}")

    for key in ["roc_auc", "pr_auc", "brier_score", "precision", "recall", "f1", "threshold"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key:<14s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"  {'prevalence':<14s}  {np.mean([m['prevalence'] for m in all_metrics]):.4f}")
    print(f"{'#'*60}")
