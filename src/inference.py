import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import inv

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
    Q = np.asarray(np.load(q_path), dtype=float)
    if Q.shape != (4, 4) or not np.allclose(Q.sum(axis=1), 0.0, atol=1e-10):
        raise ValueError("q_matrix.npy is not a valid 4-state CTMC generator.")
    if not np.allclose(Q[3], 0.0, atol=1e-10):
        raise ValueError("Critical state must be absorbing.")

    T = Q[:3, :3]
    ones = np.ones(3)
    T_inv = inv(T)
    means = -T_inv @ ones
    second_moments = 2.0 * (T_inv @ T_inv) @ ones
    variances = np.maximum(second_moments - means**2, 0.0)
    stds = np.sqrt(variances)
    return Q, means, stds


def main():
    print("=========================================")
    print("   DEEP-ICU EARLY WARNING SYSTEM LIVE    ")
    print("=========================================\n")

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
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    print("Loading patient-aware MIMIC data...")
    real_data_tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
    index_df = pd.read_csv(index_path)

    if len(real_data_tensor) != len(index_df):
        raise ValueError("Tensor rows and trajectory index rows are not aligned.")

    print(f"Total records found: {len(real_data_tensor):,}")

    rows = []
    with torch.no_grad():
        for i, patient_tensor in enumerate(real_data_tensor):
            _, mu, _ = model(patient_tensor.unsqueeze(0))
            risk_score = float(mu.item())

            state = latent_to_state(risk_score, cutoffs)

            if state == 3:
                countdown_mean = 0.0
                countdown_std = 0.0
            else:
                countdown_mean = float(expected_times[state])
                countdown_std = float(std_times[state])

            meta = index_df.iloc[i]
            rows.append(
                {
                    "record_id": i + 1,
                    "stay_id": int(meta["stay_id"]),
                    "hour": int(meta["hour"]),
                    "heart_rate": float(meta["heart_rate"]),
                    "sbp": float(meta["sbp"]),
                    "wbc": float(meta["wbc"]),
                    "risk_score_mu": round(risk_score, 6),
                    "markov_state": state,
                    "state_name": STATE_NAMES[state],
                    "hours_to_critical_mean": round(countdown_mean, 4),
                    "hours_to_critical_sd": round(countdown_std, 4),
                }
            )

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    csv_filename = "results/inference_results.csv"
    results.to_csv(csv_filename, index=False)

    print("\n--- CTMC Q Matrix ---")
    print(np.round(Q, 6))
    print("\n--- Example predictions ---")
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
