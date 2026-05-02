"""
Synthetic POTS data generator using a latent autonomic state model.

Design principles:
  - A hidden state drives everything. Features and labels are both noisy
    consequences of this state, never directly coupled.
  - Posture has realistic duration and circadian structure.
  - Symptoms emerge from the latent state with a LAG, so the forecasting
    task is non-trivial.
  - Patient-level variability in baseline physiology.

Latent states:
  0 = stable        (resting, autonomic nervous system compensating well)
  1 = compensated   (mild stress, HR slightly elevated, HRV slightly reduced)
  2 = stressed      (autonomic strain, significant HR rise, HRV drop)
  3 = decompensating (pre-symptomatic, high HR, very low HRV)

Transitions depend on current posture and time-of-day.
Symptoms are emitted from states 2-3 with a stochastic lag of 5-20 minutes.
"""

import numpy as np
import pandas as pd


# --- Transition matrices ---
# Rows = from state, Cols = to state
# Standing makes upward transitions more likely
TRANSITION_SEATED = np.array([
    [0.90, 0.08, 0.02, 0.00],  # stable -> mostly stays stable
    [0.15, 0.75, 0.08, 0.02],  # compensated -> can recover or worsen
    [0.05, 0.20, 0.65, 0.10],  # stressed -> can recover slowly
    [0.02, 0.08, 0.30, 0.60],  # decompensating -> tends to persist
])

TRANSITION_STANDING = np.array([
    [0.60, 0.25, 0.12, 0.03],  # stable -> much more likely to escalate
    [0.05, 0.50, 0.30, 0.15],  # compensated -> escalates faster
    [0.02, 0.08, 0.55, 0.35],  # stressed -> hard to recover while standing
    [0.01, 0.04, 0.20, 0.75],  # decompensating -> very sticky while standing
])


# --- Emission parameters per state: (hr_offset_mean, hr_offset_std, hrv_mean, hrv_std) ---
# HR offset is relative to patient baseline + circadian component
# HRV is absolute (simplified proxy, not true RMSSD)
EMISSION_PARAMS = {
    0: {"hr_offset": (0, 3),    "hrv": (55, 10)},   # stable
    1: {"hr_offset": (8, 4),    "hrv": (42, 8)},     # compensated
    2: {"hr_offset": (20, 6),   "hrv": (28, 6)},     # stressed
    3: {"hr_offset": (35, 8),   "hrv": (15, 5)},     # decompensating
}


def _circadian_hr(minute_of_day: int) -> float:
    """Sinusoidal circadian HR modulation. Nadir ~4am, peak ~2pm."""
    # minute 0 = midnight
    phase = 2 * np.pi * (minute_of_day - 240) / 1440  # 240 min = 4am nadir
    return 5 * np.sin(phase)  # ±5 bpm swing


def _generate_posture_sequence(n_minutes: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate realistic posture sequence (0=seated/supine, 1=standing).

    Properties:
      - Nighttime (11pm-6am): almost always supine
      - Daytime: alternating sit/stand bouts
      - Standing bouts last 5-30 minutes
      - Sitting bouts last 15-90 minutes
    """
    posture = np.zeros(n_minutes, dtype=int)
    t = 0
    is_standing = False

    while t < n_minutes:
        minute_of_day = t % 1440
        is_night = minute_of_day < 360 or minute_of_day >= 1380  # before 6am or after 11pm

        if is_night:
            # almost always supine at night
            bout_len = rng.integers(30, 120)
            is_standing = rng.random() < 0.03  # rare nighttime standing (bathroom etc)
            if is_standing:
                bout_len = rng.integers(2, 8)
        else:
            if is_standing:
                bout_len = rng.integers(5, 30)
            else:
                bout_len = rng.integers(15, 90)

        end = min(t + bout_len, n_minutes)
        posture[t:end] = int(is_standing)
        t = end
        is_standing = not is_standing

    return posture


def _generate_symptom_from_state_history(
    states: np.ndarray, rng: np.random.Generator, lag_range: tuple = (5, 20)
) -> np.ndarray:
    """
    Generate symptoms as a LAGGED, STOCHASTIC consequence of latent state.

    When the latent state enters state 2 or 3, a symptom may appear
    lag_range minutes later with probability proportional to state severity.
    Symptoms persist for a random duration (3-15 minutes).

    This ensures features (which observe HR/HRV at time t) cannot directly
    determine the symptom label at time t.
    """
    n = len(states)
    symptoms = np.zeros(n, dtype=int)

    # Find episodes where state reaches 2 or 3
    t = 0
    while t < n:
        if states[t] >= 2:
            # Probability of triggering a symptom episode
            trigger_prob = 0.3 if states[t] == 2 else 0.6
            if rng.random() < trigger_prob:
                lag = rng.integers(lag_range[0], lag_range[1] + 1)
                duration = rng.integers(3, 15)
                start = t + lag
                end = min(start + duration, n)
                if start < n:
                    symptoms[start:end] = 1

            # Skip ahead past this elevated state bout to avoid double-triggering
            while t < n and states[t] >= 2:
                t += 1
        else:
            t += 1

    # Add rare spontaneous symptoms (noise — real patients sometimes feel
    # symptoms without obvious triggers)
    n_spontaneous = rng.poisson(2 * n_days)  # ~2 per day
    for _ in range(n_spontaneous):
        start = rng.integers(0, n)
        duration = rng.integers(2, 8)
        end = min(start + duration, n)
        symptoms[start:end] = 1

    return symptoms


def generate_patient(
    patient_id: int,
    n_days: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic POTS data for one patient over n_days days.

    Multi-day simulation provides more realistic episode count distributions
    and enables testing temporal generalization across days.
    """
    n_minutes = n_days * 1440
    rng = np.random.default_rng(seed if seed is not None else patient_id)

    # Patient-level physiology
    baseline_hr = rng.normal(72, 7)  # resting HR varies across patients
    pots_severity = rng.uniform(0.3, 1.0)  # scales how reactive this patient is

    # Generate structured posture
    posture = _generate_posture_sequence(n_minutes, rng)

    # Evolve latent state
    states = np.zeros(n_minutes, dtype=int)
    state = 0

    for t in range(1, n_minutes):
        trans = TRANSITION_STANDING if posture[t] == 1 else TRANSITION_SEATED
        # Scale transition probabilities by patient severity
        # Higher severity = harder to recover (reduce leftward transitions)
        adjusted = trans[state].copy()
        if state > 0:
            recovery_damping = 1.0 - 0.4 * pots_severity
            adjusted[:state] *= recovery_damping
            adjusted /= adjusted.sum()  # renormalize

        states[t] = rng.choice(4, p=adjusted)
        state = states[t]

    # Emit observed signals
    hr = np.zeros(n_minutes)
    hrv = np.zeros(n_minutes)

    for t in range(n_minutes):
        s = states[t]
        params = EMISSION_PARAMS[s]
        circadian = _circadian_hr(t % 1440)

        hr_offset_mean, hr_offset_std = params["hr_offset"]
        hrv_mean, hrv_std = params["hrv"]

        # Scale HR response by severity
        hr[t] = baseline_hr + circadian + hr_offset_mean * pots_severity + rng.normal(0, hr_offset_std)
        hrv[t] = max(5, hrv_mean - 10 * pots_severity + rng.normal(0, hrv_std))

        # Add sensor noise (real wearables are noisy)
        hr[t] += rng.normal(0, 1.5)
        hrv[t] += rng.normal(0, 2.0)

    # Generate symptoms from latent state (LAGGED)
    symptoms = _generate_symptom_from_state_history(states, rng)

    time = pd.date_range("2024-01-01", periods=n_minutes, freq="1min", tz=None)

    return pd.DataFrame({
        "time": time,
        "patient_id": patient_id,
        "heart_rate": np.round(hr, 1),
        "hrv_proxy": np.round(hrv, 1),
        "posture": posture,
        "symptom": symptoms,
        # Store latent state for validation only — NEVER used as a feature
        "_latent_state": states,
    })


def generate_dataset(n_patients: int = 50, n_days: int = 1, seed: int = 42) -> pd.DataFrame:
    """Generate full dataset with reproducible per-patient seeds."""
    rng = np.random.default_rng(seed)
    patient_seeds = rng.integers(0, 2**31, size=n_patients)

    dfs = [generate_patient(pid, n_days=n_days, seed=int(s)) for pid, s in enumerate(patient_seeds)]
    df = pd.concat(dfs, ignore_index=True)

    return df


if __name__ == "__main__":
    import os
    from config import N_PATIENTS, N_DAYS, SEED, DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_dataset(n_patients=N_PATIENTS, n_days=N_DAYS, seed=SEED)
    # Drop latent state before saving — it's for validation only
    df.drop(columns=["_latent_state"]).to_csv(f"{DATA_DIR}/pots_dataset.csv", index=False)
    print(f"Generated {len(df)} rows, {df['patient_id'].nunique()} patients, {N_DAYS} days each")
    print(f"Symptom rate: {df['symptom'].mean():.3f}")
    print(f"Posture=1 rate: {df['posture'].mean():.3f}")
