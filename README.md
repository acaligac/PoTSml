# PoTSml v2 — Redesigned

Predict symptomatic POTS episodes from synthetic physiological time-series data.

## What this project does

Uses a **latent-state generative model** to produce realistic synthetic wearable data,
then trains classifiers to **forecast** whether a symptomatic episode will occur in the
next N minutes — a clinically useful framing that gives patients advance warning.

## Key design decisions

1. **Latent-state data generation**: A hidden Markov-like autonomic state drives both
   observed signals (HR, HRV proxy, posture) and symptoms. Features and labels share
   a common *cause* but have no direct mapping.

2. **Forecasting, not nowcasting**: The label is "will a symptom occur in the next 15
   minutes?" — not "is this minute symptomatic?" This is the only formulation that
   would be useful to a real patient.

3. **Strictly causal features**: All features use only past data. No global means,
   no future-looking aggregations.

4. **Patient-level evaluation**: GroupKFold with multiple splits. Reports ROC-AUC,
   PR-AUC, precision, recall, and compares against a trivial rule baseline.

## Project structure

```
potsml_v2/
├── src/
│   ├── generate_data.py      # Latent-state synthetic data generator
│   ├── features.py            # Strictly causal feature engineering
│   ├── train.py               # Training pipeline (LogReg + XGBoost)
│   └── evaluate.py            # Evaluation with proper metrics
├── data/                      # Generated datasets (gitignored)
├── models/                    # Trained models (gitignored)
├── tests/
│   └── test_leakage.py        # Automated leakage checks
├── run.py                     # Single entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
python run.py
```

## What is intentionally NOT included

- **LSTM / deep learning**: The data is 50 patients × 1440 minutes. An LSTM is not
  justified. XGBoost with lag features captures the relevant temporal signal.
- **SHAP**: Would be easy to add but is not included until the model is validated
  on data worth interpreting.
- **Dashboard / API / UI**: This is a research pipeline, not a product.
- **Hyperparameter tuning**: Premature until the data generation is validated.
