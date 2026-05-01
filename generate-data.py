import numpy as np
import pandas as pd

def generate_patient_data(patient_id, n_minutes=1440):
    np.random.seed(patient_id)

    time = pd.date_range("2024-01-01", periods=n_minutes, freq="1min")

    baseline_hr = np.random.normal(70, 5)

    hr, rmssd, posture, symptom = [], [], [], []

    for t in range(n_minutes):
        post = 1 if np.random.rand() < 0.02 else 0
        posture.append(post)

        if post == 1:
            delta = np.random.normal(25, 10)
            current_hr = baseline_hr + delta
            current_rmssd = np.random.normal(20, 5)
        else:
            current_hr = baseline_hr + np.random.normal(0, 3)
            current_rmssd = np.random.normal(40, 10)

        hr.append(current_hr)
        rmssd.append(current_rmssd)

        symptom.append(1 if (current_hr - baseline_hr > 30 and current_rmssd < 25) else 0)

    return pd.DataFrame({
        "time": time,
        "patient_id": patient_id,
        "heart_rate": hr,
        "rmssd": rmssd,
        "posture": posture,
        "symptom": symptom
    })


def generate_dataset(n_patients=50):
    return pd.concat(
        [generate_patient_data(i) for i in range(n_patients)]
    ).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/pots_dataset.csv", index=False)

