<div align="center">

# 🌸 PoTS episode prediction 🌸

*can a smartwatch warn you before a PoTS episode hits?*

</div>

---

PoTS episodes are brutal partly because they come out of nowhere. by the time you feel one starting, you're already in it.

this project is my attempt to see if that's actually true.

---

## what it does

takes HR, HRV, and posture data from the last 30 minutes and asks: **will there be a symptomatic episode in the next 15 minutes?**


---

## results

![patient example](plots/patient_example.png)

*HR climbs and HRV drops before symptoms appear. that lag is what the model learns.*

![shap importance](plots/shap_importance.png)

*top features: 30-min HR baseline, short-vs-long HR trend, time spent standing. basically what a doctor would look for.*

| model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| rule baseline | 0.525 | 0.333 | 1.000 | 0.320 |
| logistic regression | 0.624 | 0.398 | 0.800 | 0.379 |
| **XGBoost** | **0.632** | **0.413** | **0.800** | **0.382** |

random PR-AUC ≈ 0.32. XGBoost sits 29% above that at 80% recall.

---

## a few things i cared about getting right

 a hidden Markov model generates autonomic state → signals → symptoms, with a 5–20 min stochastic lag. features and labels never touch the same variable directly.

every feature at time *t* only uses data from *t* and earlier. automated tests in `test_leakage.py` check this, including a row-by-row label correctness test.

GroupKFold in both the outer evaluation loop and the hyperparameter search. same patient can't bleed across the boundary either way.

raw XGBoost scores aren't probabilities. Platt scaling fixes that using a held-out slice of each training fold.


---

## how to run

```bash
git clone https://github.com/acaligac/PoTSml.git
cd PoTSml
pip install -r requirements.txt
python run.py
```

generates data, trains everything, prints results, saves plots to `plots/`.

---

## honest caveats

synthetic data only — real PoTS dynamics could look different. no real patient validation yet. HR signal is i.i.d. per timestep (real wearables have autocorrelation). 50 patients sounds like a lot until you account for temporal dependence.

it's just a proof of concept.

