

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


CONFIG = {
    'device': torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'),
    'T': 30,  # small dataset
    'num_epochs': 500,  # More epochs to compensate
    'learning_rate': 5e-4,  # Higher LR for better convergence
    'batch_size': 32,  # Use full data more
    'hidden_dim': 128,  # VERY SMALL - prevent overfitting
    'time_dim': 32,
    'ema_decay': 0.999,  # EMA usef to stabilize the sampking model
    'data_path': '../data/processed/',
    'model_path': './',
}

print(f"Device: {CONFIG['device']}")
print(f"Config: T={CONFIG['T']}, Epochs={CONFIG['num_epochs']}, LR={CONFIG['learning_rate']}")


def linear_beta_schedule(timesteps: int, device: str = 'cpu'):
    """Linear schedule."""
    betas = torch.linspace(0.0001, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

T = CONFIG['T']
device = CONFIG['device']
betas, alphas, alphas_cumprod = linear_beta_schedule(T, device=device)

print(f"✓ Schedule: T={T}")


def sample_timesteps(batch_size: int, T: int, device: str):
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)

def add_noise_to_batch(x0, t, alphas_cumprod, device):
    noise = torch.randn_like(x0)
    alpha_bar_t = alphas_cumprod[t].view(-1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise

def compute_loss(eps_hat, noise, t, alphas_cumprod):
    """Simple MSE loss (no weighting - better for small data)."""
    mse = F.mse_loss(eps_hat, noise)
    return mse


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        self._backup_params()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self._backup[name]

    def _backup_params(self):
        self._backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, T: int):
        super().__init__()
        self.time_dim = time_dim
        self.T = T
        self.proj = nn.Linear(time_dim, time_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(1) / (self.T - 1)
        half_dim = self.time_dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half_dim, device=t.device))
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)

class SimpleDiffusion(nn.Module):
    """VERY SIMPLE model - won't overfit."""

    def __init__(self, dim: int, T: int = 30, time_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim

        self.time_mlp = TimeEmbedding(time_dim, T)

        # 2 simple layers
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)
        x_in = torch.cat([x_t, temb], dim=-1)
        return self.net(x_in)




class TabularDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path)
        self.X = torch.tensor(df.values, dtype=torch.float32)
        self.columns = list(df.columns)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]
    
def main():
    print("\nLoading data...")
    try:
        train_ds = TabularDataset(f'{CONFIG["data_path"]}train_combo_scaled.csv')
        val_ds = TabularDataset(f'{CONFIG["data_path"]}val_combo_scaled.csv')
        test_ds = TabularDataset(f'{CONFIG["data_path"]}test_combo_scaled.csv')
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    except Exception as e:
        print(f"et CONFIG['data_path']! Error: {e}")
        raise

    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    dim = train_ds.X.shape[1]
    model = SimpleDiffusion(
        dim=dim,
        T=T,
        time_dim=CONFIG['time_dim'],
        hidden_dim=CONFIG['hidden_dim'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    ema = ExponentialMovingAverage(model, decay=0.999)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters (SMALL - won't overfit)")


    def train():
        train_losses = []
        val_losses = []

        print("\n" + "="*70)
        print("TRAINING (SIMPLE MODEL)")
        print("="*70)

        for epoch in range(CONFIG['num_epochs']):
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            for x0 in train_dl:
                x0 = x0.to(device)
                B = x0.size(0)

                t = sample_timesteps(B, T, device)
                x_t, noise = add_noise_to_batch(x0, t, alphas_cumprod, device)

                eps_hat = model(x_t, t)
                loss = compute_loss(eps_hat, noise, t, alphas_cumprod)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update()

                epoch_loss += loss.item() * B
                batch_count += 1

            epoch_loss /= batch_count if batch_count > 0 else 1
            train_losses.append(epoch_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for x0_val in val_dl:
                    x0_val = x0_val.to(device)
                    t_val = sample_timesteps(x0_val.size(0), T, device)
                    x_t_val, noise_val = add_noise_to_batch(x0_val, t_val, alphas_cumprod, device)

                    eps_hat_val = model(x_t_val, t_val)
                    loss_val = compute_loss(eps_hat_val, noise_val, t_val, alphas_cumprod)

                    val_loss += loss_val.item() * x0_val.size(0)
                    val_batches += 1

            val_loss /= val_batches if val_batches > 0 else 1
            val_losses.append(val_loss)
            scheduler.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:3d}/{CONFIG['num_epochs']} - Train: {epoch_loss:.6f} - Val: {val_loss:.6f}")

        print(f"Complete")
        return train_losses, val_losses

    train_losses, val_losses = train()


    os.makedirs(CONFIG['model_path'], exist_ok=True)
    torch.save(model.state_dict(), f"{CONFIG['model_path']}ddpm_simple.pt")
    print(f"Model Saved")

if __name__ == "__main__":
    main()
