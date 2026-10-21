import json
import sys
from pathlib import Path

# Allow this file to run both as `python src/evaluation.py` and when imported
# from the repository root during unit tests.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm

from src.markov_mgf.ctmc_estimator import estimate_generator
from src.markov_mgf.mgf_calculator import phase_type_moments
from src.vae.model import ICU_VAE


STATE_NAMES = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
SPLITS = ["train", "validation", "test"]


def _require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the pipeline first.")
    return path


def _evaluate_vae(model, tensor, indices):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    with torch.no_grad():
        for start in range(0, len(tensor), 1024):
            batch = tensor[start:start + 1024]
            # Deterministic evaluation: use posterior mean z=mu rather than
            # sampling z. Training still uses stochastic reparameterization.
            mu, logvar = model.encode(batch)
            recon = model.decode(mu)
            mse = torch.nn.functional.mse_loss(recon, batch, reduction="sum")
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_mse += float(mse.item())
            total_kl += float(kl.item())
            total_loss += float((mse + kl).item())

    n = len(tensor)
    return {
        "records": int(n),
        "loss_per_record": total_loss / n,
        "reconstruction_mse_per_record": total_mse / n,
        "kl_per_record": total_kl / n,
        "patients": int(indices["subject_id"].nunique()),
        "icu_stays": int(indices["stay_id"].nunique()),
    }


def _first_critical_remaining_hours(stay):
    """Return hours until first Critical, excluding observations after it.

    A Critical state is treated as the first absorbing event for prognosis
    evaluation. Records after that event are not valid "time remaining"
    observations and are therefore marked NaN.
    """
    critical = stay.loc[stay["state"] == 3, "hour"]
    if critical.empty:
        return pd.Series(np.nan, index=stay.index)

    first_critical = float(critical.min())
    remaining = first_critical - stay["hour"].astype(float)

    # Keep the event time at 0, retain only pre-event observations, and
    # exclude post-Critical observations from the prognosis comparison.
    return remaining.where(remaining >= 0, np.nan)

def _empirical_one_step_transition_matrix(df):
    """Estimate observed one-hour transition probabilities without fitting Q.

    Counts every consecutive hourly state observation within each ICU stay,
    including self-transitions, and stops each trajectory at its first
    Critical observation so the empirical test comparison matches the
    absorbing-Critical CTMC semantics.
    """
    counts = np.zeros((4, 4), dtype=float)

    required = {"stay_id", "hour", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.sort_values(["stay_id", "hour"]).copy()
    for _, stay in data.groupby("stay_id", sort=False):
        rows = stay[["hour", "state"]].to_numpy()
        for current, nxt in zip(rows[:-1], rows[1:]):
            current_hour, current_state = int(current[0]), int(current[1])
            next_hour, next_state = int(nxt[0]), int(nxt[1])

            if next_hour - current_hour != 1:
                continue
            if current_state == 3:
                break

            counts[current_state, next_state] += 1.0

            if next_state == 3:
                break

    probabilities = np.zeros_like(counts)
    row_totals = counts.sum(axis=1)
    for state in range(4):
        if row_totals[state] > 0:
            probabilities[state] = counts[state] / row_totals[state]

    return probabilities, counts


def evaluate_pipeline():
    tensor_path = _require("data/processed/mimic_tensor.pt")
    index_path = _require("data/processed/mimic_index.csv")
    checkpoint_path = _require("src/vae/vae_checkpoint.pt")
    preprocessing_path = _require("data/processed/preprocessing.json")
    quality_path = Path("data/processed/data_quality_report.json")
    metadata_path = _require("data/processed/vae_metadata.json")
    q_path = _require("data/processed/q_matrix.npy")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor = torch.load(tensor_path, map_location="cpu")
    index_df = pd.read_csv(index_path)
    with open(preprocessing_path, "r", encoding="utf-8") as f:
        preprocessing = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    quality_report = None
    if quality_path.exists():
        with open(quality_path, "r", encoding="utf-8") as f:
            quality_report = json.load(f)
    training_metrics_path = Path("data/processed/vae_training_metrics.json")
    training_metrics = None
    if training_metrics_path.exists():
        with open(training_metrics_path, "r", encoding="utf-8") as f:
            training_metrics = json.load(f)

    if len(tensor) != len(index_df):
        raise ValueError("Tensor rows and index rows are not aligned.")

    split_subjects = {
        split: set(index_df.loc[index_df["split"] == split, "subject_id"])
        for split in SPLITS
    }
    overlap = {
        f"{a}_vs_{b}": sorted(split_subjects[a] & split_subjects[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1:]
    }
    if any(overlap.values()):
        raise RuntimeError(f"Patient leakage detected: {overlap}")

    if preprocessing.get("fit_on_split") != "train":
        raise RuntimeError("Preprocessing metadata does not report train-only fitting.")
    if metadata.get("cutoffs_fit_on_split") != "train":
        raise RuntimeError("VAE cutoffs are not marked as training-only.")

    model = ICU_VAE(input_dim=tensor.shape[1], hidden_dim=16, latent_dim=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    vae_metrics = {}
    latent_records = []
    with torch.no_grad():
        # Deterministic latent score: state assignment uses mu, not sampled z.
        mu_all, _ = model.encode(tensor)
        mu_all = mu_all.cpu().numpy().reshape(-1)

    for split in SPLITS:
        mask = index_df["split"].to_numpy() == split
        vae_metrics[split] = _evaluate_vae(model, tensor[mask], index_df.loc[mask])

    evaluation_df = index_df.copy()
    evaluation_df["risk_score_mu"] = mu_all
    cutoffs = np.asarray(metadata["cutoffs"], dtype=float)
    states = 3 - np.digitize(mu_all, cutoffs)
    evaluation_df["state"] = states.astype(int)
    evaluation_df["state_name"] = [STATE_NAMES[s] for s in states]

    state_counts = {}
    for split in SPLITS:
        subset = evaluation_df[evaluation_df["split"] == split]
        counts = subset["state_name"].value_counts().reindex(STATE_NAMES, fill_value=0)
        state_counts[split] = {
            state: {"records": int(counts[state]), "fraction": float(counts[state] / len(subset))}
            for state in STATE_NAMES
        }

    # Split-distribution diagnostics help distinguish model overfitting from
    # a small validation cohort whose feature distribution differs from train.
    feature_distribution_by_split = {}
    feature_names = preprocessing.get("features", ["feature_0", "feature_1", "feature_2"])
    for split in SPLITS:
        mask = index_df["split"].to_numpy() == split
        subset = tensor[mask].detach().cpu().numpy()
        feature_distribution_by_split[split] = {
            "records": int(len(subset)),
            "mean": np.mean(subset, axis=0).astype(float).tolist(),
            "std": np.std(subset, axis=0, ddof=0).astype(float).tolist(),
            "p01": np.quantile(subset, 0.01, axis=0).astype(float).tolist(),
            "median": np.quantile(subset, 0.50, axis=0).astype(float).tolist(),
            "p99": np.quantile(subset, 0.99, axis=0).astype(float).tolist(),
        }

    transition_counts, exposure = None, None
    train_df = evaluation_df[evaluation_df["split"] == "train"]
    Q, transition_counts, exposure = estimate_generator(train_df)

    # Compare the model's state-conditioned prognosis with observed remaining
    # hours to the first Critical state on uncensored test trajectories.
    test_df = evaluation_df[evaluation_df["split"] == "test"].copy()
    test_transition_probabilities, test_transition_counts = (
        _empirical_one_step_transition_matrix(test_df)
    )
    model_transition_probabilities = expm(Q)
    transition_errors = np.abs(
        test_transition_probabilities[:3] - model_transition_probabilities[:3]
    )
    transition_validation = {
        "time_step_hours": 1.0,
        "test_transition_counts_including_self": test_transition_counts.astype(int).tolist(),
        "observed_probabilities": test_transition_probabilities.tolist(),
        "model_probabilities_from_Q": model_transition_probabilities.tolist(),
        "mean_absolute_error_by_state": {
            STATE_NAMES[state]: float(np.mean(transition_errors[state]))
            for state in range(3)
            if test_transition_counts[state].sum() > 0
        },
        "overall_mae_transient_states": float(
            np.mean(
                transition_errors[
                    np.array([test_transition_counts[s].sum() > 0 for s in range(3)])
                ]
            )
        ) if any(test_transition_counts[s].sum() > 0 for s in range(3)) else None,
        "note": "Held-out test transitions are compared with P(1h)=exp(Q) using only pre-Critical transitions. This is a transition-model diagnostic, not a clinical accuracy metric."
    }
    remaining_parts = []
    for stay_id, stay in test_df.groupby("stay_id", sort=False):
        remaining = _first_critical_remaining_hours(stay)
        part = stay[["subject_id", "stay_id", "hour", "state", "state_name"]].copy()
        part["observed_remaining_hours"] = remaining.values
        remaining_parts.append(part)
    test_observed = pd.concat(remaining_parts, ignore_index=True)
    uncensored = test_observed["observed_remaining_hours"].notna()

    observed_comparison = {}
    means = np.full(3, np.nan)
    transient_T = Q[:3, :3]
    try:
        means, _, _ = phase_type_moments(transient_T)
    except (ValueError, np.linalg.LinAlgError):
        # Keep evaluation explicit if a future fitted Q fails absorption checks.
        means = np.full(3, np.nan)
    for state in range(3):
        subset = test_observed[(test_observed["state"] == state) & uncensored]
        if len(subset):
            observed = subset["observed_remaining_hours"].to_numpy()
            predicted = np.full(len(observed), means[state])
            observed_comparison[STATE_NAMES[state]] = {
                "records": int(len(subset)),
                "predicted_mean_hours": float(means[state]),
                "observed_mean_remaining_hours": float(np.mean(observed)),
                "mae_hours": float(np.mean(np.abs(predicted - observed))),
            }
        else:
            observed_comparison[STATE_NAMES[state]] = {
                "records": 0,
                "predicted_mean_hours": float(means[state]) if np.isfinite(means[state]) else None,
                "observed_mean_remaining_hours": None,
                "mae_hours": None,
            }

    # Stay-level prognosis comparison: one row per ICU stay, using the
    # state at hour 0 to select the Phase-Type mean and time to first
    # model-defined Critical state as the observed event time. Stays with no
    # observed Critical state are right-censored, not treated as events.
    stay_level_rows = []
    for stay_id, stay in test_df.sort_values("hour").groupby("stay_id", sort=False):
        stay = stay.sort_values("hour")
        first_row = stay.iloc[0]
        initial_state = int(first_row["state"])
        last_hour = float(stay["hour"].max())
        critical_hours = stay.loc[stay["state"] == 3, "hour"]
        event_observed = not critical_hours.empty
        event_time = float(critical_hours.min()) if event_observed else last_hour
        predicted_mean = (
            float(means[initial_state])
            if initial_state < 3 and np.isfinite(means[initial_state])
            else 0.0
        )
        stay_level_rows.append({
            "subject_id": int(first_row["subject_id"]),
            "stay_id": int(stay_id),
            "initial_state": STATE_NAMES[initial_state],
            "predicted_mean_hours": predicted_mean,
            "observed_time_hours": event_time,
            "event_observed": bool(event_observed),
            "followup_hours": last_hour,
        })

    stay_level_df = pd.DataFrame(stay_level_rows)
    stay_level_df.to_csv(out_dir / "prognosis_stay_level_test.csv", index=False)
    stay_level_summary = {}
    if not stay_level_df.empty:
        for state_name in STATE_NAMES[:3]:
            subset = stay_level_df[stay_level_df["initial_state"] == state_name]
            events = subset[subset["event_observed"]]
            stay_level_summary[state_name] = {
                "stays": int(len(subset)),
                "events_observed": int(subset["event_observed"].sum()),
                "right_censored": int((~subset["event_observed"]).sum()),
                "predicted_mean_hours_mean": (
                    float(subset["predicted_mean_hours"].mean()) if len(subset) else None
                ),
                "observed_event_time_mean_hours": (
                    float(events["observed_time_hours"].mean()) if len(events) else None
                ),
                "event_only_mae_hours": (
                    float(np.mean(np.abs(
                        events["predicted_mean_hours"].to_numpy()
                        - events["observed_time_hours"].to_numpy()
                    ))) if len(events) else None
                ),
                "warning": (
                    "Descriptive stay-level comparison only; not a censoring-aware survival score "
                    "and not clinical prognosis validation."
                ),
            }

    report = {
        "split_integrity": {
            "subject_overlap": overlap,
            "all_subjects_exclusive": not any(overlap.values()),
            "records": {split: int((index_df["split"] == split).sum()) for split in SPLITS},
            "patients": {split: int((index_df["split"] == split).sum() and index_df.loc[index_df["split"] == split, "subject_id"].nunique()) for split in SPLITS},
            "icu_stays": {split: int(index_df.loc[index_df["split"] == split, "stay_id"].nunique()) for split in SPLITS},
        },
        "preprocessing": {
            "fit_on_split": preprocessing.get("fit_on_split"),
            "split_strategy": preprocessing.get("split_strategy", "not recorded"),
            "features": preprocessing.get("features"),
            "imputation_method": preprocessing.get("imputation_method"),
            "max_forward_fill_hours": preprocessing.get("max_forward_fill_hours"),
            "missingness_indicator_columns": preprocessing.get("missingness_indicator_columns"),
            "scaler_mean": preprocessing.get("scaler_mean"),
            "scaler_scale": preprocessing.get("scaler_scale"),
            "data_quality_report_available": quality_report is not None,
        },
        "state_semantics": {
            "meaning": "model-defined VAE latent bins, not clinically validated severity categories",
            "latent_score_direction": "lower latent scores map to higher-numbered risk states by implementation convention; clinical direction is unvalidated",
            "critical_absorbing": "Critical is absorbing in CTMC prognosis; per-hour VAE labels are independently assigned and can later change",
        },
        "feature_scope_note": (
            "Only heart rate, systolic blood pressure and WBC are used. This is not a comprehensive ICU severity assessment."
        ),
        "data_quality": quality_report,
        "feature_distribution_by_split_standardized_tensor": feature_distribution_by_split,
        "vae_training_diagnostics": training_metrics,
        "vae": {
            "evaluation_reconstruction": "deterministic decoder(mu), no latent sampling for validation/test metrics",
            "cutoffs": cutoffs.tolist(),
            "cutoffs_fit_on_split": metadata.get("cutoffs_fit_on_split"),
            "metrics_by_split": vae_metrics,
        },
        "state_distribution": state_counts,
        "ctmc_training": {
            "records": int(len(train_df)),
            "icu_stays": int(train_df["stay_id"].nunique()),
            "transition_counts": transition_counts.astype(int).tolist(),
            "exposure_hours": exposure.tolist(),
            "Q": Q.tolist(),
        },
        "test_transition_validation": transition_validation,
        "test_prognosis_comparison": {
            "censoring_note": (
                "Only test records before the first observed Critical state are compared. "
                "Stays ending without observed Critical are right-censored at their last observed hour; "
                "records after the first Critical state are excluded."
            ),
            "uncensored_records": int(uncensored.sum()),
            "uncensored_stays": int(test_observed.loc[uncensored, "stay_id"].nunique()),
            "total_test_records": int(len(test_observed)),
            "total_test_stays": int(test_observed["stay_id"].nunique()),
            "uncensored_patients": int(test_observed.loc[uncensored, "subject_id"].nunique()),
            "total_test_patients": int(test_observed["subject_id"].nunique()),
            "censored_stays": int(
                test_observed["stay_id"].nunique()
                - test_observed.loc[uncensored, "stay_id"].nunique()
            ),
            "comparison_by_state_record_level_legacy": observed_comparison,
            "comparison_by_initial_state_stay_level": stay_level_summary,
            "stay_level_csv": "results/prognosis_stay_level_test.csv",
            "interpretation_note": (
                "Exploratory descriptive diagnostic only. The uncensored subset may be small, "
                "records within a stay are correlated, and this is not a censoring-aware survival analysis "
                "or a clinical performance metric."
            ),
        },
    }

    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(vae_metrics).T.to_csv(out_dir / "vae_split_metrics.csv")

    # Plot 1: latent score by split.
    plt.figure(figsize=(9, 6))
    for split in SPLITS:
        values = evaluation_df.loc[evaluation_df["split"] == split, "risk_score_mu"]
        plt.hist(values, bins=40, alpha=0.35, label=split)
    for cutoff in cutoffs:
        plt.axvline(cutoff, linestyle="--", linewidth=1.5)
    plt.xlabel("VAE latent score (mu)")
    plt.ylabel("Patient-hour observations")
    plt.title("Latent Risk Distribution by Data Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_latent_by_split.png", dpi=300)
    plt.close()

    # Plot 2: state distribution by split.
    distribution = pd.DataFrame({
        split: [state_counts[split][state]["fraction"] for state in STATE_NAMES]
        for split in SPLITS
    }, index=STATE_NAMES)
    ax = distribution.plot(kind="bar", figsize=(9, 6))
    ax.set_ylabel("Fraction of records")
    ax.set_xlabel("Risk state")
    ax.set_title("Risk-State Distribution by Data Split")
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_state_distribution.png", dpi=300)
    plt.close()

    # Plot 3: training transition counts.
    plt.figure(figsize=(7, 6))
    plt.imshow(transition_counts, aspect="auto")
    plt.colorbar(label="Observed transitions")
    plt.xticks(range(4), STATE_NAMES, rotation=25, ha="right")
    plt.yticks(range(4), STATE_NAMES)
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.title("CTMC Training Transition Counts")
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_transition_heatmap.png", dpi=300)
    plt.close()

    # Plot 4: reconstruction/KL/total loss by split.
    metric_frame = pd.DataFrame(vae_metrics).T
    ax = metric_frame[
        ["loss_per_record", "reconstruction_mse_per_record", "kl_per_record"]
    ].plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Loss per record")
    ax.set_xlabel("Data split")
    ax.set_title("VAE Evaluation Metrics by Split")
    ax.legend(["Total loss", "Reconstruction MSE", "KL"])
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_vae_metrics.png", dpi=300)
    plt.close()

    # Plot 5: model expected time versus observed mean remaining time.
    # Plot the stay-level comparison, not the correlated hourly-record comparison.
    labels = [
        state for state in STATE_NAMES[:3]
        if stay_level_summary.get(state, {}).get("stays", 0) > 0
        and stay_level_summary[state].get("events_observed", 0) > 0
    ]
    if labels:
        x = np.arange(len(labels))
        predicted = [
            stay_level_summary[s]["predicted_mean_hours_mean"] for s in labels
        ]
        observed = [
            stay_level_summary[s]["observed_event_time_mean_hours"] for s in labels
        ]
        width = 0.35
        plt.figure(figsize=(9, 6))
        plt.bar(x - width / 2, predicted, width, label="Predicted mean by initial state")
        plt.bar(x + width / 2, observed, width, label="Observed first-Critical time (events only)")
        plt.xticks(x, labels)
        plt.ylabel("Hours from start of ICU stay")
        plt.xlabel("Initial model-defined state at hour 0")
        plt.title("Stay-Level Phase-Type Check (Exploratory)")
        plt.legend()
        plt.figtext(
            0.5, 0.01,
            "One observation per stay. Censored stays are reported separately; Critical is model-defined, not a clinical endpoint.",
            ha="center", fontsize=8,
        )
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.savefig(out_dir / "evaluation_prognosis_comparison.png", dpi=300)
        plt.close()

    print("Evaluation completed.")
    print(f"Saved report: {out_dir / 'evaluation_report.json'}")
    print(f"Saved split metrics: {out_dir / 'vae_split_metrics.csv'}")
    print("Saved evaluation graphs in results/.")
    return report


if __name__ == "__main__":
    evaluate_pipeline()
