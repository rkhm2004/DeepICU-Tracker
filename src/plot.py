import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_ORDER = ["Low Risk", "Medium Risk", "High Risk", "Critical"]


def generate_three_plots():
    csv_path = "results/inference_results.csv"
    metadata_path = "data/processed/vae_metadata.json"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run inference.py first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run train.py first."
        )

    df = pd.read_csv(csv_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    os.makedirs("results", exist_ok=True)

    # Graph 1: VAE latent score and the exact thresholds used by training.
    plt.figure(figsize=(10, 6))
    for state in STATE_ORDER:
        values = df.loc[df["state_name"] == state, "risk_score_mu"]
        if len(values):
            plt.hist(values, bins=50, alpha=0.45, label=state)

    for cutoff in metadata["cutoffs"]:
        plt.axvline(cutoff, linestyle="--", linewidth=2)

    plt.title("VAE Latent Risk Score (mu) and CTMC State Cutoffs")
    plt.xlabel("VAE latent risk score (mu)")
    plt.ylabel("Patient-hour observations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/graph_1_vae_distribution.png", dpi=300)
    plt.close()

    # Graph 2: clinical variables mapped to the learned Markov states.
    plt.figure(figsize=(10, 6))
    for state in STATE_ORDER:
        subset = df[df["state_name"] == state]
        if len(subset):
            plt.scatter(
                subset["heart_rate"],
                subset["sbp"],
                alpha=0.45,
                s=12,
                label=state,
            )

    plt.title("Clinical Vitals Mapped to CTMC Risk States")
    plt.xlabel("Heart Rate (BPM)")
    plt.ylabel("Systolic Blood Pressure (mmHg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/graph_2_clinical_scatter.png", dpi=300)
    plt.close()

    # Graph 3: computed Phase-Type mean and one-standard-deviation interval.
    transient = df[df["state_name"] != "Critical"]
    summary = (
        transient.groupby("state_name")
        .agg(
            mean_hours=("hours_to_critical_mean", "first"),
            sd_hours=("hours_to_critical_sd", "first"),
        )
        .reindex(["Low Risk", "Medium Risk", "High Risk"])
    )

    plt.figure(figsize=(8, 6))
    x = np.arange(len(summary))
    plt.bar(
        x,
        summary["mean_hours"].values,
        yerr=summary["sd_hours"].values,
        capsize=5,
    )
    plt.xticks(x, summary.index)
    plt.ylabel("Expected time to Critical (hours)")
    plt.xlabel("Current transient state")
    plt.title("Phase-Type Expected Time to Critical Collapse")
    plt.tight_layout()
    plt.savefig("results/graph_3_mgf_countdown.png", dpi=300)
    plt.close()

    print("Successfully generated all 3 graphs in results/.")


if __name__ == "__main__":
    generate_three_plots()
