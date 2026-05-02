"""
Single entry point for the PoTSml pipeline.

Usage: python run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generate_data import generate_dataset
from src.features import build_features, get_feature_columns
from src.train import train_and_evaluate, print_results


def main():
    print("=" * 60)
    print("  PoTSml v2 — POTS Episode Forecasting Pipeline")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/4] Generating synthetic dataset...")
    df = generate_dataset(n_patients=50, seed=42)
    print(f"  {len(df)} rows, {df['patient_id'].nunique()} patients")
    print(f"  Symptom rate: {df['symptom'].mean():.3f}")
    print(f"  Standing rate: {df['posture'].mean():.3f}")

    # Save (without latent state)
    os.makedirs("data", exist_ok=True)
    df.drop(columns=["_latent_state"]).to_csv("data/pots_dataset.csv", index=False)

    # Step 2: Feature engineering
    print("\n[2/4] Building features...")
    df_feat = build_features(df)
    feature_cols = get_feature_columns()
    print(f"  {len(df_feat)} rows after dropping incomplete windows")
    print(f"  {len(feature_cols)} features")
    print(f"  Label prevalence: {df_feat['label'].mean():.3f}")

    # Step 3: Train and evaluate
    print("\n[3/4] Training models with 5-fold GroupKFold...")
    results = train_and_evaluate(df_feat, feature_cols)

    # Step 4: Print results
    print("\n[4/4] Results")
    print_results(results)


if __name__ == "__main__":
    main()
