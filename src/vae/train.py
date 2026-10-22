import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import ICU_VAE


STATE_NAMES = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
RANDOM_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE = 10


def vae_loss_function(recon_x, x, mu, logvar, observed_mask=None, beta=1.0):
    """VAE objective with reconstruction targets limited to originally observed features.

    Inputs remain the causally imputed tensor. Missing values do not count as
    ground-truth reconstruction targets; KL regularization is retained.
    """
    if observed_mask is None:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    else:
        mask = observed_mask.to(dtype=x.dtype)
        if mask.shape != x.shape:
            raise ValueError("observed_mask must have the same shape as x.")
        # Normalize each feature by its observed count within the batch, so
        # frequently measured heart rate does not dominate sparse WBC targets.
        squared = (recon_x - x).pow(2) * mask
        counts = mask.sum(dim=0)
        present = counts > 0
        if present.any():
            per_feature_mse = squared.sum(dim=0)[present] / counts[present]
            # Scale by batch size to keep the reconstruction/KL terms comparable
            # to the summed-per-record VAE objective.
            recon_loss = per_feature_mse.sum() * x.shape[0]
        else:
            recon_loss = recon_x.sum() * 0.0
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss


def latent_to_state(mu_values, cutoffs):
    """Map ascending latent-score bins to Low, Medium, High, Critical.

    State indices must match STATE_NAMES and the CTMC convention that state 3
    (Critical) is absorbing. This is a model-defined ordering, not a clinical
    severity rule; the latent score's clinical direction remains unvalidated.
    """
    return np.digitize(mu_values, cutoffs).astype(int)


def _observed_mask(index_frame, features):
    """Build true-observation mask from pre-imputation missingness indicators."""
    columns = [f"{feature}_was_missing" for feature in features]
    missing = [column for column in columns if column not in index_frame.columns]
    if missing:
        raise ValueError(
            "Masked VAE training requires original missingness indicators in mimic_index.csv: "
            + ", ".join(missing)
        )
    return torch.tensor(
        1.0 - index_frame[columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )


def _mean_loss(model, data, observed_mask):
    """Deterministic masked VAE loss; only observed values are reconstruction targets."""
    if len(data) == 0:
        raise ValueError("Cannot evaluate VAE loss on an empty split.")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(data), 1024):
            batch = data[start:start + 1024]
            mask = observed_mask[start:start + 1024]
            mu, logvar = model.encode(batch)
            recon_x = model.decode(mu)
            total_loss += vae_loss_function(
                recon_x, batch, mu, logvar, observed_mask=mask
            ).item()
    return total_loss / len(data)


def _feature_mse(model, data):
    """Per-feature deterministic reconstruction MSE, useful for diagnosing shift."""
    model.eval()
    squared_error = torch.zeros(data.shape[1], dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(data), 1024):
            batch = data[start:start + 1024]
            mu, _ = model.encode(batch)
            recon = model.decode(mu)
            squared_error += ((recon - batch) ** 2).sum(dim=0).double()
    return (squared_error / len(data)).tolist()


def train_vae(
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    patience=DEFAULT_PATIENCE,
):
    """Train VAE on train stays, select by validation loss, and report test loss.

    The test split is never used for optimization or checkpoint selection.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    tensor_path = Path("data/processed/mimic_tensor.pt")
    index_path = Path("data/processed/mimic_index.csv")
    if not tensor_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            "Run src/data_pipeline/mimic_parser.py first so mimic_tensor.pt "
            "and mimic_index.csv exist."
        )

    print("1. Loading processed MIMIC tensor...")
    data = torch.load(tensor_path)
    index_df = pd.read_csv(index_path)

    if len(data) != len(index_df):
        raise ValueError("Tensor rows and trajectory index rows are not aligned.")
    if "split" not in index_df.columns:
        raise ValueError("mimic_index.csv must contain a stay-level split column.")

    split = index_df["split"].to_numpy()
    train_mask = split == "train"
    val_mask = split == "validation"
    test_mask = split == "test"

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]
    features = ["heart_rate", "sbp", "wbc"] if data.shape[1] == 3 else [
        f"feature_{i}" for i in range(data.shape[1])
    ]
    all_observed_mask = _observed_mask(index_df, features)
    train_observed_mask = all_observed_mask[train_mask]
    val_observed_mask = all_observed_mask[val_mask]
    test_observed_mask = all_observed_mask[test_mask]
    observed_counts = {
        "train": train_observed_mask.sum(dim=0).int().tolist(),
        "validation": val_observed_mask.sum(dim=0).int().tolist(),
        "test": test_observed_mask.sum(dim=0).int().tolist(),
    }
    print("   Masked reconstruction targets (observed values only):")
    print(f"   Train feature counts: {dict(zip(features, observed_counts['train']))}")
    print(f"   Validation feature counts: {dict(zip(features, observed_counts['validation']))}")
    print(f"   Test feature counts (reporting only): {dict(zip(features, observed_counts['test']))}")

    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError("Train, validation, and test splits must all contain records.")

    print(
        f"   Train: {len(train_data):,} records / "
        f"{index_df.loc[train_mask, 'stay_id'].nunique():,} ICU stays"
    )
    print(
        f"   Validation: {len(val_data):,} records / "
        f"{index_df.loc[val_mask, 'stay_id'].nunique():,} ICU stays"
    )
    print(
        f"   Test: {len(test_data):,} records / "
        f"{index_df.loc[test_mask, 'stay_id'].nunique():,} ICU stays"
    )

    train_loader = DataLoader(
        TensorDataset(train_data, train_observed_mask),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    model = ICU_VAE(input_dim=data.shape[1], hidden_dim=16, latent_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("2. Starting VAE training...")
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0

        for x, observed_mask in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar, observed_mask=observed_mask)
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        # Compare train and validation using the same deterministic metric.
        train_loss = _mean_loss(model, train_data, train_observed_mask)
        val_loss = _mean_loss(model, val_data, val_observed_mask)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
            }
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"   Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(f"   Early stopping at epoch {epoch} (patience={patience}).")
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")

    model.load_state_dict(best_state)
    test_loss = _mean_loss(model, test_data, test_observed_mask)
    train_loss_at_best = _mean_loss(model, train_data, train_observed_mask)

    checkpoint_path = Path("src/vae/vae_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)

    split_feature_mse = {
        "train": _feature_mse(model, train_data),
        "validation": _feature_mse(model, val_data),
        "test": _feature_mse(model, test_data),
    }
    feature_names = features
    feature_gap_diagnostics = {}
    for feature_index, feature_name in enumerate(feature_names):
        train_mse = float(split_feature_mse["train"][feature_index])
        val_mse = float(split_feature_mse["validation"][feature_index])
        test_mse = float(split_feature_mse["test"][feature_index])
        feature_gap_diagnostics[feature_name] = {
            "train_mse": train_mse,
            "validation_mse": val_mse,
            "test_mse": test_mse,
            "test_to_train_mse_ratio": test_mse / train_mse if train_mse > 0 else None,
            "test_to_validation_mse_ratio": test_mse / val_mse if val_mse > 0 else None,
        }

    metrics = {
        "evaluation_reconstruction": "deterministic decoder(mu), no latent sampling for validation/test metrics",
        "reconstruction_objective": "per-feature observed-only masked MSE normalized by observed count in each batch, summed across present features, plus KL divergence; imputed values are encoder inputs but not reconstruction targets",
        "feature_balancing": "each feature contributes its own observed-only mean squared error rather than raw observation-count-weighted squared error; sparse WBC still has high sampling uncertainty",
        "observed_target_counts_by_split": {
            split_name: dict(zip(features, counts))
            for split_name, counts in observed_counts.items()
        },
        "feature_order": ["heart_rate", "sbp", "wbc"] if data.shape[1] == 3 else list(range(data.shape[1])),
        "per_feature_reconstruction_mse_by_split": split_feature_mse,
        "per_feature_generalization_gap": feature_gap_diagnostics,
        "seed": RANDOM_SEED,
        "epochs_requested": epochs,
        "epochs_trained": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "best_validation_loss": float(best_val_loss),
        "train_loss_at_best_checkpoint": float(train_loss_at_best),
        "test_loss_at_best_checkpoint": float(test_loss),
        "split_stays": {
            "train": int(index_df.loc[train_mask, "stay_id"].nunique()),
            "validation": int(index_df.loc[val_mask, "stay_id"].nunique()),
            "test": int(index_df.loc[test_mask, "stay_id"].nunique()),
        },
        "history": history,
    }
    with open("data/processed/vae_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(history).to_csv(
        "data/processed/vae_training_history.csv", index=False
    )

    print(f"3. Saved best model weights to {checkpoint_path}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Test loss: {test_loss:.4f}")

    print("4. Extracting latent risk scores...")
    model.eval()
    with torch.no_grad():
        _, mu_all, _ = model(data)
    mu_np = mu_all.cpu().numpy().reshape(-1)

    # State thresholds are learned from training stays only. Validation/test
    # records are classified using these frozen training cutoffs.
    train_mu = mu_np[train_mask]
    cutoffs = np.quantile(train_mu, [0.25, 0.50, 0.75])
    states = latent_to_state(mu_np, cutoffs)

    metadata = {
        "state_names": STATE_NAMES,
        "cutoffs": cutoffs.tolist(),
        "mapping": "state = digitize(mu, cutoffs); 0=Low, 1=Medium, 2=High, 3=Critical",
        "state_semantics": "model-defined latent bins; not clinically validated severity categories",
        "clinical_outcome_validation": "not performed",
        "interpretation_warning": (
            "VAE reconstruction and latent bins do not establish clinical risk. "
            "Validate state direction and cutoffs against independent clinical outcomes before clinical claims."
        ),
        "cutoffs_fit_on_split": "train",
        "seed": RANDOM_SEED,
        "latent_dimension": 1,
        "best_validation_loss": float(best_val_loss),
        "test_loss": float(test_loss),
    }
    with open("data/processed/vae_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    latent_df = index_df.copy()
    latent_df["risk_score_mu"] = mu_np
    latent_df["state"] = states.astype(int)
    latent_df["state_name"] = [STATE_NAMES[s] for s in states]
    latent_df.to_csv("data/processed/latent_states.csv", index=False)

    torch.save(torch.tensor(states, dtype=torch.long), "data/processed/latent_states.pt")

    print(f"   Cutoffs (training mu, ascending): {cutoffs}")
    print(f"   All-record state distribution: {np.bincount(states, minlength=4)}")
    print("Saved latent trajectory to data/processed/latent_states.csv")


if __name__ == "__main__":
    train_vae()
