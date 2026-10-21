import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATE_ORDER = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
TRANSIENT_STATES = ["Low Risk", "Medium Risk", "High Risk"]
STATE_COLORS = {
    "Low Risk": "#4C9BD6",
    "Medium Risk": "#F39C3D",
    "High Risk": "#4BAF63",
    "Critical": "#D9535D",
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_latent_distributions(df, cutoffs, output_path):
    """Four small multiples keep the dominant Critical histogram from hiding others."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    axes = axes.ravel()
    score = pd.to_numeric(df["risk_score_mu"], errors="coerce")

    for ax, state in zip(axes, STATE_ORDER):
        values = score[df["state_name"] == state].dropna()
        if values.empty:
            ax.text(0.5, 0.5, "No observations", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.hist(values, bins=45, color=STATE_COLORS[state],
                    alpha=0.82, edgecolor="white", linewidth=0.25)
        for cutoff in cutoffs:
            ax.axvline(float(cutoff), color="#263746", linestyle="--",
                       linewidth=1.4, alpha=0.9)
        ax.set_title(f"{state} (n={len(values):,})", fontsize=11)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylabel("Patient-hour count")

    axes[2].set_xlabel("VAE latent score (mu)")
    axes[3].set_xlabel("VAE latent score (mu)")
    fig.suptitle("VAE Latent Scores by Model-Defined State", fontsize=15, y=1.01)
    fig.text(
        0.5, -0.015,
        "Dashed lines are training-derived cutoffs; state labels are model-defined, not clinically validated.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_feature_scatter(df, output_path, max_points=12000):
    """Plot a reproducible sample so dense classes remain visually interpretable."""
    required = {"heart_rate", "sbp", "state_name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"inference_results.csv is missing columns: {sorted(missing)}")

    plot_df = df.dropna(subset=["heart_rate", "sbp", "state_name"]).copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    for state in STATE_ORDER:
        subset = plot_df[plot_df["state_name"] == state]
        if subset.empty:
            continue
        ax.scatter(
            subset["heart_rate"], subset["sbp"],
            s=14, alpha=0.48, label=f"{state} (n={len(subset):,})",
            color=STATE_COLORS[state], edgecolors="none", rasterized=True,
        )
    ax.set_title("Heart Rate vs Systolic Blood Pressure by Model-Defined State")
    ax.set_xlabel("Heart rate (beats/min)")
    ax.set_ylabel("Systolic blood pressure (mmHg)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True, fontsize=9, markerscale=1.2)
    fig.text(
        0.5, 0.01,
        f"Showing {len(plot_df):,} observations"
        + (" (reproducible sample)" if len(df) > max_points else "")
        + "; colors represent model-defined states.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_phase_type(df, output_path):
    required = {"state_name", "hours_to_critical_mean", "hours_to_critical_sd"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"inference_results.csv is missing columns: {sorted(missing)}")

    transient = df[df["state_name"].isin(TRANSIENT_STATES)].copy()
    summary = (
        transient.groupby("state_name")
        .agg(
            mean_hours=("hours_to_critical_mean", "median"),
            sd_hours=("hours_to_critical_sd", "median"),
        )
        .reindex(TRANSIENT_STATES)
    )
    summary = summary.dropna(subset=["mean_hours", "sd_hours"])
    if summary.empty:
        raise ValueError("No valid Phase-Type summary values found in inference_results.csv.")

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(summary))
    means = summary["mean_hours"].to_numpy(dtype=float)
    stds = summary["sd_hours"].to_numpy(dtype=float)
    ax.bar(
        x, means, yerr=stds, capsize=6,
        color=[STATE_COLORS[s] for s in summary.index],
        edgecolor="white", linewidth=0.8, alpha=0.9,
    )
    ax.set_xticks(x, summary.index)
    ax.set_ylabel("Expected absorption time (hours)")
    ax.set_xlabel("Current transient state")
    ax.set_title("Phase-Type Expected Time to Critical")
    ax.grid(axis="y", alpha=0.22)
    ax.text(
        0.01, -0.22,
        "Bars show model-based means; whiskers show the reported standard deviations. "
        "These are not individual countdowns or calibrated clinical predictions.",
        transform=ax.transAxes, fontsize=8.5, va="top", wrap=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_ctmc_generator(results_dir, output_path):
    """Plot the fitted CTMC generator Q (rates/hour), not transition counts."""
    report_path = results_dir / "evaluation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"{report_path} is required to plot the fitted CTMC generator. "
            "Run the evaluation pipeline first so the saved Q matrix is available."
        )

    report = _load_json(report_path)
    try:
        q = np.asarray(report["ctmc_training"]["Q"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "evaluation_report.json does not contain ctmc_training.Q."
        ) from exc

    if q.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 CTMC generator matrix; found shape {q.shape}.")
    if not np.all(np.isfinite(q)):
        raise ValueError("CTMC generator matrix contains non-finite values.")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    limit = max(float(np.max(np.abs(q))), 1e-9)
    image = ax.imshow(q, cmap="RdYlBu_r", vmin=-limit, vmax=limit, aspect="equal")
    ax.set_xticks(range(4), STATE_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(4), STATE_ORDER)
    ax.set_xlabel("Destination state")
    ax.set_ylabel("Current state")
    ax.set_title("Estimated CTMC Generator Matrix Q", pad=14)

    for row in range(4):
        for col in range(4):
            value = q[row, col]
            color = "white" if abs(value) > 0.52 * limit else "black"
            ax.text(
                col, row, f"{value:.4f}",
                ha="center", va="center", color=color,
                fontsize=10, fontweight="bold" if row == col else "normal",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition rate (per hour)")
    fig.text(
        0.5, 0.015,
        "Off-diagonal entries are transition rates; diagonal entries are negative row sums. "
        "Critical is absorbing by model assumption.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_plots():
    """Generate the three original plots plus a readable CTMC generator heatmap."""
    root = Path(".")
    results_dir = root / "results"
    csv_path = results_dir / "inference_results.csv"
    metadata_path = root / "data" / "processed" / "vae_metadata.json"

    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found. Run inference.py first.")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{metadata_path} not found. Run train.py first.")

    df = pd.read_csv(csv_path)
    metadata = _load_json(metadata_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    if "risk_score_mu" not in df.columns or "state_name" not in df.columns:
        raise ValueError(
            "inference_results.csv must contain 'risk_score_mu' and 'state_name'."
        )

    cutoffs = metadata.get("cutoffs")
    if not isinstance(cutoffs, list) or len(cutoffs) != 3:
        raise ValueError("vae_metadata.json must contain exactly three training cutoffs.")

    _plot_latent_distributions(
        df, cutoffs, results_dir / "graph_1_vae_distribution.png"
    )
    _plot_feature_scatter(
        df, results_dir / "graph_2_clinical_scatter.png"
    )
    _plot_phase_type(
        df, results_dir / "graph_3_phase_type_expected_absorption.png"
    )
    _plot_ctmc_generator(
        results_dir, results_dir / "graph_4_ctmc_generator.png"
    )

    print("Successfully generated four plots in results/:")
    print("  graph_1_vae_distribution.png")
    print("  graph_2_clinical_scatter.png")
    print("  graph_3_phase_type_expected_absorption.png")
    print("  graph_4_ctmc_generator.png")


# Keep compatibility with any existing code importing the old function name.
def generate_three_plots():
    generate_plots()


if __name__ == "__main__":
    generate_plots()
