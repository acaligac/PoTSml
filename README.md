<div align="center">

# 🌸 PoTS Episode Prediction 🌸

![Python](https://img.shields.io/badge/Python-3.9+-ff69b4?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ff85c2?style=for-the-badge&logo=data:image/png;base64,)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ffb6c1?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-proof--of--concept-ff69b4?style=for-the-badge)

*a machine learning pipeline for forecasting symptomatic POTS episodes from wearable physiological data*

</div>

---

> **Postural Orthostatic Tachycardia Syndrome (POTS)** is a chronic condition affecting an estimated 1–3 million people in the US — the majority of them women. Symptoms (dizziness, tachycardia, fatigue, brain fog) often strike without warning, making daily life unpredictable. This project explores whether wearable sensor data can give patients a 15-minute heads-up before an episode hits. 💗

---

## 🌷 What this project does

This is a **forecasting pipeline**, not a nowcasting one. The model answers:

> *"Based on your heart rate, HRV, and posture over the last 30 minutes — will you likely have a symptomatic episode in the next 15 minutes?"*

That distinction matters. Knowing you're *currently* symptomatic is useless. Knowing you're *about to be* symptomatic gives you time to sit down, hydrate, take medication, or cancel that standing meeting.

---

## 🌸 Architecture overview

```
Latent autonomic state (hidden)
        │
        ▼
Physiological signals          →   Feature engineering   →   XGBoost classifier
  HR, HRV proxy, posture              (strictly causal)          ↑
        │                                                    Logistic Regression
        ▼                                                         ↑
 Stochastic symptom                                         Rule-based baseline
  emission w/ 5–20 min lag
```

**Three models are trained and compared:**

| Model | Purpose |
|---|---|
| 🩺 Rule-based baseline | Would a clinician eyeballing the chart catch this? |
| 📈 Logistic Regression | Is the problem even linearly separable? |
| 🌲 XGBoost | Can we do better with temporal lag features? |

---

## 💕 Key design decisions

### 1. Latent-state data generation

Rather than hand-waving synthetic data, the generator uses a **4-state autonomic Markov model**:

| State | Meaning | Standing transition |
|---|---|---|
| 0 — Stable | Normal compensation | Low escalation risk |
| 1 — Compensated | Mild autonomic stress | Moderate escalation |
| 2 — Stressed | Significant strain | High escalation |
| 3 — Decompensating | Pre-symptomatic | Very high symptom probability |

Posture drives state transitions. Circadian rhythm modulates HR. Patient-level severity (`pots_severity ~ Uniform(0.3, 1.0)`) scales reactivity. Symptoms are emitted from states 2–3 with a **5–20 minute stochastic lag** — which is what makes this a non-trivial forecasting task.

### 2. Strictly causal features

Every feature at time *t* uses only data from times ≤ *t*. This means:
- Expanding (not rolling) mean for baseline HR deviation
- No global statistics computed before the prediction window
- No `fillna(0)` shortcuts that implicitly leak structure

Automated leakage tests in `tests/test_leakage.py` enforce this.

### 3. Patient-level cross-validation

`GroupKFold(n_splits=5)` ensures **no patient appears in both train and test**. Without this, the model memorizes patient-specific baselines and evaluation metrics are meaningless.

### 4. Metrics chosen for clinical relevance

- **PR-AUC** — the right metric for imbalanced data (~5–8% symptom prevalence)
- **ROC-AUC** — threshold-independent discrimination
- **Precision & Recall** — reported together; the operating threshold is a clinical decision, not a modeling one

---

## 🌺 Feature set (21 features, all causal)

| Feature | Type | Window |
|---|---|---|
| `heart_rate` | Raw signal | — |
| `hrv_proxy` | Raw signal | — |
| `delta_hr` | Deviation from expanding baseline | — |
| `posture` | Binary (standing/supine) | — |
| `posture_duration` | Minutes in current posture | — |
| `posture_burden_10` | Fraction of last 10 min standing | 10 min |
| `hr_roll5_mean` / `_std` | Short-term HR trend + volatility | 5 min |
| `hrv_roll5_mean` / `_std` | Short-term HRV trend + volatility | 5 min |
| `hr_roll30_mean` | Longer-term HR context | 30 min |
| `hr_trend` | Short- vs. long-term HR divergence | 5 vs. 30 min |
| `hr_accel` | Rate of HR change | 3 min diff |
| `hr_lag1/3/5/10` | Past HR values (ARIMA-like) | 1/3/5/10 min |
| `hrv_lag1/3/5/10` | Past HRV values | 1/3/5/10 min |

---

## 🌷 Project structure

```
pots-episode-prediction/
├── run.py                 # 🎯 single entry point — start here
├── generate_data.py       # latent-state synthetic data generator
├── features.py            # strictly causal feature engineering
├── train.py               # GroupKFold training (LogReg + XGBoost)
├── evaluate.py            # metrics computation & reporting
├── tests/
│   └── test_leakage.py    # automated leakage detection
├── data/                  # generated datasets (gitignored)
├── models/                # trained models (gitignored)
├── requirements.txt
└── .gitignore
```

---

## 🩷 How to run

```bash
# clone
git clone https://github.com/acaligac/PoTSml.git
cd PoTSml

# install dependencies
pip install -r requirements.txt

# run the full pipeline
python run.py
```

This will:
1. Generate synthetic data for 50 patients (1 day each)
2. Engineer strictly causal features
3. Train and evaluate all three models via 5-fold GroupKFold CV
4. Print a results table comparing ROC-AUC, PR-AUC, precision, recall, and F1

---

## 🩷 Results (50 patients × 3 days, default XGBoost params, 5-fold GroupKFold)

> Threshold is chosen per-fold to achieve ≥80% recall with maximum precision — not hardcoded at 0.5.

**Model comparison**

| Model | ROC-AUC | PR-AUC | Brier ↓ | Recall | Precision |
|---|---|---|---|---|---|
| Rule baseline | 0.525 ± 0.003 | 0.333 ± 0.009 | 0.285 ± 0.006 | 1.000 | 0.320 |
| Logistic Regression | 0.624 ± 0.010 | 0.398 ± 0.013 | 0.211 ± 0.004 | 0.800 | 0.379 |
| **XGBoost** | **0.632 ± 0.009** | **0.413 ± 0.011** | **0.210 ± 0.004** | **0.800** | **0.382** |

Random PR-AUC baseline (= prevalence) ≈ 0.32. XGBoost is 29% above random and beats the rule baseline by +24% PR-AUC, at the same recall.

**XGBoost feature importance (mean |SHAP|, last fold test set)**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `hr_roll30_mean` | 0.2491 |
| 2 | `hr_trend` | 0.1939 |
| 3 | `hrv_lag10` | 0.1632 |
| 4 | `posture_duration` | 0.1192 |
| 5 | `hrv_roll5_mean` | 0.1047 |
| 6 | `delta_hr` | 0.1018 |
| 7 | `heart_rate` | 0.0965 |
| 8–21 | lags, std features, posture | < 0.07 |

The model correctly identifies the 30-minute HR baseline, the short-vs-long HR trend, and how long the patient has been standing as the dominant signals — consistent with POTS physiology.

**Horizon sensitivity (XGBoost, 5-fold GroupKFold)**

| Horizon | ROC-AUC | PR-AUC | Recall |
|---|---|---|---|
| 5 min | 0.643 ± 0.005 | 0.271 ± 0.007 | 0.800 |
| 10 min | — | — | — |
| **15 min** ← default | **0.632 ± 0.009** | **0.413 ± 0.011** | **0.800** |
| 30 min | — | — | — |

> Run `python run.py` locally for the full horizon sweep.

---

## 🌸 Honest limitations

This is a **proof-of-concept on synthetic data**. It has not been validated on real patients.

| Limitation | Notes |
|---|---|
| Synthetic data only | Real POTS dynamics may differ significantly |
| No calibration | Predicted probabilities are not yet calibrated |
| Fixed 0.5 threshold | Clinical deployment needs a cost-function-driven threshold |
| No SHAP yet | Deferred until the model is validated on data worth interpreting |
| No hyperparameter tuning | Premature until data generation is validated |
| 1 day per patient | Real datasets need multi-day trajectories |

---

## 🚫 What is intentionally NOT here

- **LSTM / deep learning** — 50 patients × 1440 minutes is not enough data to justify it. XGBoost with lag features captures the relevant temporal signal with far less variance.
- **Dashboard / API / UI** — this is a research pipeline, not a product.
- **Real patient data** — future work pending IRB-approved dataset access.

---

## 💗 Future roadmap

- [x] Multi-day simulation (7 days per patient, configurable via `config.py`)
- [x] SHAP feature importance analysis (XGBoost, printed to console)
- [x] Probability calibration (Platt scaling via `CalibratedClassifierCV`)
- [x] Sensitivity analysis on forecasting horizon (5, 10, 15, 30 min)
- [x] Hyperparameter tuning (RandomizedSearchCV, nested CV)
- [x] Clinical cost-function threshold selection (`find_clinical_threshold()`)
- [ ] Validation against real wearable data (Apple Watch / Garmin exports)
- [ ] Reliability diagrams (visual calibration curves)
- [ ] Multi-day temporal train/test split (train days 1–5, test days 6–7)

---

<div align="center">

*built with care for everyone navigating life with POTS* 🌸

</div>
