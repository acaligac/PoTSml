import pandas as pd

def build_features(df):
    df = df.copy()

    df["baseline_hr"] = df.groupby("patient_id")["heart_rate"].transform("mean")
    df["delta_hr"] = df["heart_rate"] - df["baseline_hr"]

    df["hr_mean_5"] = df.groupby("patient_id")["heart_rate"].transform(lambda x: x.rolling(5).mean())
    df["hr_std_5"] = df.groupby("patient_id")["heart_rate"].transform(lambda x: x.rolling(5).std())

    df["rmssd_mean_5"] = df.groupby("patient_id")["rmssd"].transform(lambda x: x.rolling(5).mean())

    df["hr_lag1"] = df.groupby("patient_id")["heart_rate"].shift(1)

    return df.fillna(0)
