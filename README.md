<div align="center">

# 🌸 PoTS episode prediction 🌸

![Python](https://img.shields.io/badge/Python-3.9+-ff69b4?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ff85c2?style=for-the-badge&logo=data:image/png;base64,)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ffb6c1?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-proof--of--concept-ff69b4?style=for-the-badge)

*a machine learning pipeline for forecasting symptomatic POTS episodes from wearable physiological data*

</div>

---

> **postural orthostatic tachycardia syndrome (PoTS)** is a chronic condition affecting an estimated 1–3 million people in the US, the majority of them women. symptoms (dizziness, tachycardia, fatigue, brain fog) often strike without warning, making daily life unpredictable.
>
> this project explores whether wearable sensor data can give patients a 15-minute heads-up before an episode hits. 💗

---

## 🌷 What this project does

this is a **forecasting pipeline**, not a nowcasting one. the model answers:

> *"based on your heart rate, HRV, and posture over the last 30 minutes will you likely have a symptomatic episode in the next 15 minutes?"*

knowing you're *currently* symptomatic is useless. knowing you're *about to be* symptomatic gives you time to sit down, hydrate, take medication, or cancel that standing meeting.

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

**three models are trained and compared:**

| model | purpose |
|---|---|
| 🩺 rule-based baseline | would a clinician eyeballing the chart catch this? |
| 📈 logistic Regression | is the problem even linearly separable? |
| 🌲 XGBoost | can we do better with temporal lag features? |

---

## 💕 key design decisions

### 1. latent-state data generation

rather than hand-waving synthetic data, the generator uses a **4-state autonomic Markov model**:

| State | Meaning | Standing transition |
|---|---|---|
| 0 — stable | normal compensation | low escalation risk |
| 1 — compensated | mild autonomic stress | moderate escalation |
| 2 — stressed | significant strain | high escalation |
| 3 — decompensating | pre-symptomatic | very high symptom probability |

posture drives state transitions. circadian rhythm modulates HR. patient-level severity (`pots_severity ~ Uniform(0.3, 1.0)`) scales reactivity. symptoms are emitted from states 2–3 with a **5–20 minute stochastic lag** which is what makes this a non-trivial forecasting task.

### 2. strictly causal features

every feature at time *t* uses only data from times ≤ *t*. this means:
- expanding (not rolling) mean for baseline HR deviation
- no global statistics computed before the prediction window
- no `fillna(0)` shortcuts that implicitly leak structure

automated leakage tests in `tests/test_leakage.py` enforce this.

### 3. patient-level cross-validation

`GroupKFold(n_splits=5)` ensures **no patient appears in both train and test**. without this, the model memorizes patient-specific baselines and evaluation metrics are meaningless.

### 4. metrics chosen for clinical relevance

- **PR-AUC** — the right metric for imbalanced data (~5–8% symptom prevalence)
- **ROC-AUC** — threshold-independent discrimination
- **Precision & Recall** — reported together; the operating threshold is a clinical decision, not a modeling one

---

## 🌺 feature set (21 features, all causal)

| feature | type | window |
|---|---|---|
| `heart_rate` | raw signal | — |
| `hrv_proxy` | raw signal | — |
| `delta_hr` | deviation from expanding baseline | — |
| `posture` | binary (standing/supine) | — |
| `posture_duration` | minutes in current posture | — |
| `posture_burden_10` | fraction of last 10 min standing | 10 min |
| `hr_roll5_mean` / `_std` | short-term HR trend + volatility | 5 min |
| `hrv_roll5_mean` / `_std` | short-term HRV trend + volatility | 5 min |
| `hr_roll30_mean` | longer-term HR context | 30 min |
| `hr_trend` | short- vs. long-term HR divergence | 5 vs. 30 min |
| `hr_accel` | rate of HR change | 3 min diff |
| `hr_lag1/3/5/10` | past HR values (ARIMA-like) | 1/3/5/10 min |
| `hrv_lag1/3/5/10` | past HRV values | 1/3/5/10 min |

---

## 🌷 project structure

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

## 🩷 how to run

```bash
# clone
git clone https://github.com/acaligac/PoTSml.git
cd PoTSml

# install dependencies
pip install -r requirements.txt

# run the full pipeline
python run.py
```

this will:
1. generate synthetic data for 50 patients (1 day each)
2. engineer strictly causal features
3. train and evaluate all three models via 5-fold GroupKFold CV
4. print a results table comparing ROC-AUC, PR-AUC, precision, recall, and F1

---

## 🌸 honest limitations

this is a **proof-of-concept on synthetic data**. it has not been validated on real patients.

| limitation | notes |
|---|---|
| synthetic data only | real PoTS dynamics may differ significantly |
| no calibration | predicted probabilities are not yet calibrated |
| fixed 0.5 threshold | clinical deployment needs a cost-function-driven threshold |
| no SHAP yet | Deferred until the model is validated on data worth interpreting |
| bo hyperparameter tuning | Premature until data generation is validated |
| 1 day per patient | real datasets need multi-day trajectories |

---


## 💗 Future roadmap

- [ ] validation against real wearable data (Apple Watch / Garmin exports)
- [ ] reliability diagrams (visual calibration curves)
- [ ] multi-day temporal train/test split (train days 1–5, test days 6–7)

---

<div align="center">

*built with care* 🌸

</div>
