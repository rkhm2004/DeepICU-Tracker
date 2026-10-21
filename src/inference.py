import json
import os
import sys
from pathlib import Path

# Ensure repository-root imports work when invoked as python src/inference.py.
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
from src.markov_mgf.mgf_calculator import load_transient_generator, phase_type_moments

sys.path.insert(0, os.path.abspath("src/vae"))
from model import ICU_VAE


STATE_NAMES = ["Low Risk", "Medium Risk", "High Risk", "Critical"]


def load_vae_metadata(path="data/processed/vae_metadata.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latent_to_state(risk_score, cutoffs):
    raw_bin = int(np.digitize(risk_score, np.asarray(cutoffs)))
    return 3 - raw_bin


def load_phase_type_summary(q_path="data/processed/q_matrix.npy"):
    Q, T = load_transient_generator(q_path)
    means, _, stds = phase_type_moments(T)
    return Q, means, stds


def main():
    print("=========================================")
    print("   DEEP-ICU RESEARCH PROTOTYPE           ")
    print("=========================================")
    print("Research estimates only — not validated for clinical decision-making.\n")

    tensor_path = Path("data/processed/mimic_tensor.pt")
    index_path = Path("data/processed/mimic_index.csv")
    checkpoint_path = Path("src/vae/vae_checkpoint.pt")
    metadata_path = Path("data/processed/vae_metadata.json")
    q_path = Path("data/processed/q_matrix.npy")

    required = [tensor_path, index_path, checkpoint_path, metadata_path, q_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts. Run the data parser, VAE training, and CTMC "
            f"estimator first: {missing}"
        )

    metadata = load_vae_metadata(metadata_path)
    cutoffs = metadata["cutoffs"]

    Q, expected_times, std_times = load_phase_type_summary(q_path)

    print("Loading VAE model...")
    model = ICU_VAE(input_dim=3, hidden_dim=16, latent_dim=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    print("Loading patient-aware MIMIC data...")
    real_data_tensor = torch.load(tensor_path, map_location="cpu")
    index_df = pd.read_csv(index_path)

    if len(real_data_tensor) != len(index_df):
        raise ValueError("Tensor rows and trajectory index rows are not aligned.")
    missingness_columns = [
        "heart_rate_was_missing",
        "sbp_was_missing",
        "wbc_was_missing",
    ]
    if not set(missingness_columns).issubset(index_df.columns):
        raise ValueError(
            "Trajectory index lacks missingness indicators. Re-run "
            "src/data_pipeline/mimic_parser.py before inference."
        )

    print(f"Total records found: {len(real_data_tensor):,}")

    rows = []
    with torch.no_grad():
        for i, patient_tensor in enumerate(real_data_tensor):
            # Inference uses the deterministic posterior mean. Sampling z is
            # useful during VAE training but unnecessary for state assignment.
            mu, _ = model.encode(patient_tensor.unsqueeze(0))
            risk_score = float(mu.item())

            state = latent_to_state(risk_score, cutoffs)

            if state == 3:
                countdown_mean = 0.0
                countdown_std = 0.0
            else:
                countdown_mean = float(expected_times[state])
                countdown_std = float(std_times[state])

            meta = index_df.iloc[i]
            missing_feature_count = int(meta[missingness_columns].sum())
            rows.append(
                {
                    "record_id": i + 1,
                    "subject_id": int(meta["subject_id"]),
                    "stay_id": int(meta["stay_id"]),
                    "hour": int(meta["hour"]),
                    "heart_rate": float(meta["heart_rate"]),
                    "sbp": float(meta["sbp"]),
                    "wbc": float(meta["wbc"]),
                    "risk_score_mu": round(risk_score, 6),
                    "markov_state": state,
                    "state_name": STATE_NAMES[state],
                    "state_interpretation": "Model-defined VAE latent bin; not a clinically validated severity category.",
                    "hours_to_critical_mean": round(countdown_mean, 4),
                    "hours_to_critical_sd": round(countdown_std, 4),
                    "prognosis_interpretation": (
                        "Critical state: model absorption time is set to zero."
                        if state == 3
                        else "Expected absorption time under fitted CTMC assumptions; not an individual countdown or clinical prediction."
                    ),
                    "source_features_missing_count": missing_feature_count,
                    "input_quality_warning": (
                        "One or more source features were missing in this hourly bin and imputed."
                        if missing_feature_count > 0
                        else "All three source features had measurements in this hourly bin."
                    ),
                }
            )

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    csv_filename = "results/inference_results.csv"
    results.to_csv(csv_filename, index=False)

    print("\n--- CTMC Q Matrix ---")
    print(np.round(Q, 6))
    print("\n--- Example model-defined states and expected times (not individual countdowns) ---")
    print(
        results[
            [
                "record_id",
                "stay_id",
                "hour",
                "risk_score_mu",
                "state_name",
                "hours_to_critical_mean",
                "hours_to_critical_sd",
            ]
        ].head(5).to_string(index=False)
    )
    print(f"\nSUCCESS! Saved {len(results):,} predictions to {csv_filename}")


if __name__ == "__main__":
    main()
