import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from src.features.build_features import build_features

df = pd.read_csv("data/pots_dataset.csv")
df = build_features(df)

features = ["delta_hr", "rmssd", "hr_mean_5", "hr_std_5"]

gss = GroupShuffleSplit(test_size=0.2, n_splits=1)
train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

X_train = train_df[features]
y_train = train_df["symptom"]

X_test = test_df[features]
y_test = test_df["symptom"]

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, probs)

print(f"ROC-AUC: {roc:.3f}")

joblib.dump(model, "models/xgb_model.pkl")

