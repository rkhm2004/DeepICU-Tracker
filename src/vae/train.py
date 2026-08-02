import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from model import ICU_VAE

def vae_loss_function(recon_x, x, mu, logvar):
    # Reconstruction Loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence Loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def train_vae(epochs=50, batch_size=64, learning_rate=1e-3):
    print("1. Loading processed MIMIC tensor...")
    tensor_path = Path("data/processed/mimic_tensor.pt")
    if not tensor_path.exists():
        raise FileNotFoundError("mimic_tensor.pt not found! Run mimic_parser.py first.")
    
    data = torch.load(tensor_path)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model, Optimizer
    model = ICU_VAE(input_dim=3, hidden_dim=16, latent_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("2. Starting VAE Training Loop...")
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = train_loss / len(dataloader.dataset)
            print(f"   Epoch {epoch:02d}/{epochs} | Avg Loss: {avg_loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), "src/vae/vae_checkpoint.pt")
    print("\n3. Saved trained model weights to src/vae/vae_checkpoint.pt")

    # Extract latent space (mu) for all records and discretize into 4 states
    print("4. Extracting latent risk scores and discretizing into 4 states...")
    model.eval()
    with torch.no_grad():
        _, mu_all, _ = model(data)
        mu_np = mu_all.numpy().flatten()

    # Discretize using quantiles into 4 discrete health risk states:
    # 0 = Low Risk, 1 = Medium Risk, 2 = High Risk, 3 = Critical
    quantiles = np.quantile(mu_np, [0.25, 0.50, 0.75])
    states = np.digitize(mu_np, quantiles)  # Outputs values 0, 1, 2, 3

    # Save discretized states for Member 2 (CTMC)
    torch.save(torch.tensor(states), "data/processed/latent_states.pt")
    print(f"   Cutoffs used for states: {quantiles}")
    print(f"   State distribution: {np.bincount(states)}")
    print("Saved discretized risk sequence to data/processed/latent_states.pt")

if __name__ == "__main__":
    train_vae()