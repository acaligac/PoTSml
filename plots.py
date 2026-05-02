"""
Visualizations for the PoTSml pipeline.

All plots use a consistent pink palette. generate_all_plots() is called
from run.py after training and receives the results dict directly —
no re-training needed.

Output: plots/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

from config import PLOTS_DIR, DEFAULT_HORIZON

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
PINK      = "#FF69B4"
DEEP_PINK = "#C71585"
LIGHT_PINK = "#FFB6C1"
BLUSH     = "#FFF0F5"
MAUVE     = "#DDA0DD"
TEAL      = "#89CCC4"
CHARCOAL  = "#4A3040"

MODEL_COLORS = {
    "rule_baseline":       MAUVE,
    "logistic_regression": TEAL,
    "xgboost":             DEEP_PINK,
}
MODEL_LABELS = {
    "rule_baseline":       "Rule baseline",
    "logistic_regression": "Logistic Regression",
    "xgboost":             "XGBoost (calibrated)",
}


def _theme(fig, axes):
    fig.patch.set_facecolor(BLUSH)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor("#FFF8FC")
        for spine in ax.spines.values():
            spine.set_edgecolor(LIGHT_PINK)
        ax.tick_params(colors=CHARCOAL, labelsize=9)
        ax.xaxis.label.set_color(CHARCOAL)
        ax.yaxis.label.set_color(CHARCOAL)
        ax.title.set_color(CHARCOAL)
        ax.grid(color=LIGHT_PINK, alpha=0.5, linewidth=0.8)


def _save(fig, name: str) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Patient example — one day of wearable signals
# ---------------------------------------------------------------------------
def plot_patient_example(df, patient_id: int = 0) -> str:
    pat = df[df["patient_id"] == patient_id].copy().reset_index(drop=True).iloc[:1440]
    hours = np.arange(len(pat)) / 60
    sym = pat["symptom"].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("One patient · one day of wearable data",
                 fontsize=13, color=CHARCOAL, fontweight="bold", y=1.01)

    for ax, col, color, title in [
        (axes[0], "heart_rate", DEEP_PINK, "Heart Rate (bpm)  ·  pink shading = symptomatic minutes"),
        (axes[1], "hrv_proxy",  TEAL,      "HRV proxy  ·  lower = more autonomic strain"),
    ]:
        ax.plot(hours, pat[col], color=color, lw=1.1, alpha=0.9)
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_title(title, fontsize=9)
        for i, s in enumerate(sym):
            if s:
                ax.axvspan(hours[i], hours[min(i+1, len(hours)-1)],
                           color=PINK, alpha=0.22, linewidth=0)

    axes[2].fill_between(hours, pat["posture"], step="pre",
                         color=MAUVE, alpha=0.75)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["seated / supine", "standing"], fontsize=8)
    axes[2].set_xlabel("Hour of day")
    axes[2].set_title("Posture", fontsize=9)

    _theme(fig, axes)
    fig.tight_layout()
    return _save(fig, "patient_example.png")


# ---------------------------------------------------------------------------
# 2. PR curves — uses pre-computed predictions from results dict
# ---------------------------------------------------------------------------
def plot_pr_curves(results: dict) -> str:
    preds = results["predictions"]
    y_true = preds["y_true"]
    prevalence = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(prevalence, color=LIGHT_PINK, lw=1.5, ls="--",
               label=f"Random baseline  (prevalence = {prevalence:.2f})")

    for model_key, prob_key in [
        ("rule_baseline",       "rule_baseline"),
        ("logistic_regression", "logistic_regression"),
        ("xgboost",             "xgboost_cal"),
    ]:
        probs = preds[prob_key]
        p, r, _ = precision_recall_curve(y_true, probs)
        auc = float(-np.trapz(p, r))
        ax.plot(r, p, color=MODEL_COLORS[model_key], lw=2,
                label=f"{MODEL_LABELS[model_key]}  (PR-AUC = {auc:.3f})")

    ax.set_xlabel("Recall  (sensitivity)")
    ax.set_ylabel("Precision  (PPV)")
    ax.set_title(f"Precision-Recall Curves  ·  {DEFAULT_HORIZON}-min forecast horizon",
                 fontsize=12)
    ax.legend(framealpha=0.9, facecolor=BLUSH, edgecolor=LIGHT_PINK, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _theme(fig, ax)
    fig.tight_layout()
    return _save(fig, "pr_curves.png")


# ---------------------------------------------------------------------------
# 3. SHAP importance bar chart
# ---------------------------------------------------------------------------
def plot_shap_importance(shap_df) -> str:
    top = shap_df.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [DEEP_PINK if i >= len(top) - 3 else PINK for i in range(len(top))]
    ax.barh(top["feature"], top["mean_abs_shap"],
            color=colors, edgecolor=DEEP_PINK, linewidth=0.5, alpha=0.85)

    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost feature importance\n(mean |SHAP|  ·  held-out test set)",
                 fontsize=12)
    _theme(fig, ax)
    fig.tight_layout()
    return _save(fig, "shap_importance.png")


# ---------------------------------------------------------------------------
# 4. Calibration curve — raw vs Platt-scaled
# ---------------------------------------------------------------------------
def plot_calibration(results: dict) -> str:
    preds  = results["predictions"]
    y_true = preds["y_true"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=LIGHT_PINK, lw=1.5, ls="--",
            label="Perfect calibration")

    for probs, color, label in [
        (preds["xgboost_raw"], MAUVE,     "XGBoost (raw)"),
        (preds["xgboost_cal"], DEEP_PINK, "XGBoost (Platt scaled)"),
    ]:
        frac, mean = calibration_curve(y_true, probs, n_bins=10)
        ax.plot(mean, frac, color=color, lw=2, marker="o", ms=5, label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve  ·  reliability diagram", fontsize=12)
    ax.legend(framealpha=0.9, facecolor=BLUSH, edgecolor=LIGHT_PINK, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _theme(fig, ax)
    fig.tight_layout()
    return _save(fig, "calibration.png")


# ---------------------------------------------------------------------------
# 5. Horizon sensitivity
# ---------------------------------------------------------------------------
def plot_horizon_sensitivity(horizon_results: dict) -> str:
    """
    horizon_results: {horizon_int: {"roc_auc": [...], "pr_auc": [...]}}
    """
    horizons  = sorted(horizon_results)
    roc_means = [np.mean(horizon_results[h]["roc_auc"]) for h in horizons]
    roc_stds  = [np.std(horizon_results[h]["roc_auc"])  for h in horizons]
    pr_means  = [np.mean(horizon_results[h]["pr_auc"])  for h in horizons]
    pr_stds   = [np.std(horizon_results[h]["pr_auc"])   for h in horizons]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(horizons, roc_means, yerr=roc_stds,
                color=TEAL, lw=2, marker="o", ms=6, capsize=4, label="ROC-AUC")
    ax.errorbar(horizons, pr_means, yerr=pr_stds,
                color=DEEP_PINK, lw=2, marker="o", ms=6, capsize=4, label="PR-AUC")
    ax.axvline(DEFAULT_HORIZON, color=LIGHT_PINK, lw=1.5, ls="--",
               label=f"Default ({DEFAULT_HORIZON} min)")

    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Score  (5-fold mean ± std)")
    ax.set_title("Performance vs. forecast horizon", fontsize=12)
    ax.set_xticks(horizons)
    ax.legend(framealpha=0.9, facecolor=BLUSH, edgecolor=LIGHT_PINK, fontsize=9)
    _theme(fig, ax)
    fig.tight_layout()
    return _save(fig, "horizon_sensitivity.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def generate_all_plots(df, results: dict, horizon_results: dict) -> None:
    print(f"\n  Saving plots to {PLOTS_DIR}/")

    for fn, args in [
        (plot_patient_example, (df,)),
        (plot_pr_curves,       (results,)),
        (plot_calibration,     (results,)),
    ]:
        path = fn(*args)
        print(f"  ✓ {path}")

    if results.get("shap_importance") is not None:
        path = plot_shap_importance(results["shap_importance"])
        print(f"  ✓ {path}")

    if horizon_results:
        path = plot_horizon_sensitivity(horizon_results)
        print(f"  ✓ {path}")
