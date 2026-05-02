"""
Central configuration for the PoTSml pipeline.

All tunable parameters live here. Import this module instead of
scattering magic numbers across files.
"""

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
N_PATIENTS = 50
N_DAYS = 7          # days of data per patient (was 1 — more temporal diversity)
SEED = 42

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
DEFAULT_HORIZON = 15                        # default forecasting window (minutes)
FORECAST_HORIZONS = [5, 10, 15, 30]        # sensitivity analysis over horizons

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_FOLDS = 5

# Clinical operating threshold: minimum recall (sensitivity) we want.
# False negatives (missed episodes) are more costly than false positives,
# so we require at least 80% recall and pick the threshold with best precision
# subject to that constraint.
CLINICAL_RECALL_TARGET = 0.80

# Fraction of training data used as calibration holdout (for Platt scaling)
CALIBRATION_SPLIT = 0.20

# ---------------------------------------------------------------------------
# XGBoost defaults
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "random_state": SEED,
    "eval_metric": "logloss",
    "verbosity": 0,
}

# Randomized search space for XGBoost hyperparameter tuning
XGB_SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "subsample": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

XGB_SEARCH_ITER = 20    # RandomizedSearchCV iterations
XGB_SEARCH_CV = 3       # inner CV folds for hyperparameter search

# Logistic regression search space
# Note: penalty= was deprecated in sklearn 1.8 — search over C only
LR_SEARCH_SPACE = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0],
}

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
