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


def vae_loss_function(recon_x, x, mu, logvar):
    """VAE objective = reconstruction error + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


def latent_to_state(mu_values, cutoffs):
    """Map higher latent scores to safer states and lower scores to higher risk.

    The PDF's interpretation is positive/stable and negative/critical. With
    three ascending cutoffs, the safest state is therefore the highest
    quantile and Critical is the lowest quantile.
    """
    raw_bins = np.digitize(mu_values, cutoffs)
    return 3 - raw_bins


def train_vae(epochs=50, batch_size=64, learning_rate=1e-3):
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

    dataset = TensorDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    model = ICU_VAE(input_dim=data.shape[1], hidden_dim=16, latent_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("2. Starting VAE training...")
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        for (x,) in dataloader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            avg_loss = train_loss / len(dataloader.dataset)
            print(f"   Epoch {epoch:02d}/{epochs} | Avg Loss: {avg_loss:.4f}")

    checkpoint_path = Path("src/vae/vae_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"3. Saved model weights to {checkpoint_path}")

    print("4. Extracting latent risk scores...")
    model.eval()
    with torch.no_grad():
        _, mu_all, _ = model(data)
    mu_np = mu_all.cpu().numpy().reshape(-1)

    cutoffs = np.quantile(mu_np, [0.25, 0.50, 0.75])
    states = latent_to_state(mu_np, cutoffs)

    metadata = {
        "state_names": STATE_NAMES,
        "cutoffs": cutoffs.tolist(),
        "mapping": "state = 3 - digitize(mu, cutoffs)",
        "seed": RANDOM_SEED,
        "latent_dimension": 1,
    }
    with open("data/processed/vae_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    latent_df = index_df.copy()
    latent_df["risk_score_mu"] = mu_np
    latent_df["state"] = states.astype(int)
    latent_df["state_name"] = [STATE_NAMES[s] for s in states]
    latent_df.to_csv("data/processed/latent_states.csv", index=False)

    torch.save(torch.tensor(states, dtype=torch.long), "data/processed/latent_states.pt")

    print(f"   Cutoffs (ascending mu): {cutoffs}")
    print(f"   State distribution: {np.bincount(states, minlength=4)}")
    print("Saved latent trajectory to data/processed/latent_states.csv")


if __name__ == "__main__":
    train_vae()
