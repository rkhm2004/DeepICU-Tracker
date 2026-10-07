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


def vae_loss_function(recon_x, x, mu, logvar):
    """VAE objective = reconstruction error + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


def latent_to_state(mu_values, cutoffs):
    """Map higher latent scores to safer states and lower scores to higher risk."""
    raw_bins = np.digitize(mu_values, cutoffs)
    return 3 - raw_bins


def _mean_loss(model, data):
    """Average VAE loss per record for a fixed dataset."""
    if len(data) == 0:
        raise ValueError("Cannot evaluate VAE loss on an empty split.")

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(data), 1024):
            batch = data[start:start + 1024]
            recon_x, mu, logvar = model(batch)
            total_loss += vae_loss_function(recon_x, batch, mu, logvar).item()
    return total_loss / len(data)


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
        TensorDataset(train_data),
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

        for (x,) in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        train_loss = train_total / len(train_loader.dataset)
        val_loss = _mean_loss(model, val_data)
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
    test_loss = _mean_loss(model, test_data)
    train_loss_at_best = _mean_loss(model, train_data)

    checkpoint_path = Path("src/vae/vae_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)

    metrics = {
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
        "mapping": "state = 3 - digitize(mu, cutoffs)",
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
