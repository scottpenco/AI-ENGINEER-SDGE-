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
from typing import Optional





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
    'model_path': '../models/',
}



class SimpleDiffusion(nn.Module):
    """Simple Diffusion model was chosen to avoid overfitting"""

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

print(f"Device: {CONFIG['device']}")

ckpt_path = (f'{CONFIG["model_path"]}ddpm_simple.pt')
state_dict = torch.load(f'{CONFIG["model_path"]}ddpm_simple.pt', map_location = CONFIG["device"])


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


def linear_beta_schedule(timesteps: int, device: str = 'cpu'):
    """Linear schedule."""
    betas = torch.linspace(0.0001, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

T = CONFIG['T']
device = CONFIG['device']
betas, alphas, alphas_cumprod = linear_beta_schedule(T, device=device)

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


def sample_ddpm(model, n_samples, dim, T, betas, alphas, alphas_cumprod, device):
        model.eval()
        x = torch.randn(n_samples, dim, device=device) * 0.5  # smaller noise

        print(f"Sampling {n_samples} samples...")

        with torch.no_grad():
            for i in reversed(range(T)):
                t = torch.full((n_samples,), i, device=device, dtype=torch.long)

                noise_pred = model(x, t)
                noise_pred = torch.clamp(noise_pred, -3, 3)

                alpha_t = alphas[i]
                alpha_bar_t = alphas_cumprod[i]
                beta_t = betas[i]

                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * noise_pred
                )

                if i > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta_t) * noise
                else:
                    x = mean

                x = torch.clamp(x, -3, 3)

        return x.cpu().numpy()

class TabularDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path)
        self.X = torch.tensor(df.values, dtype=torch.float32)
        self.columns = list(df.columns)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]



def generate_synthetic(n_samples: int, config_path: Optional[str] = None) -> pd.DataFrame:
    print("\nLoading data...")
    try:
        train_ds = TabularDataset(f'{CONFIG["data_path"]}train_combo_scaled.csv')
        val_ds = TabularDataset(f'{CONFIG["data_path"]}val_combo_scaled.csv')
        test_ds = TabularDataset(f'{CONFIG["data_path"]}test_combo_scaled.csv')
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    except Exception as e:
        print(f"et CONFIG['data_path']! Error: {e}")
        raise


    dim = train_ds.X.shape[1]
    model = SimpleDiffusion(
            dim=dim,
            T=T,
            time_dim=CONFIG['time_dim'],
            hidden_dim=CONFIG['hidden_dim'],
        ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print('Loaded DDPM model:')
    ema = ExponentialMovingAverage(model, decay=0.999)

    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    dim = train_ds.X.shape[1]
    ema.apply_shadow()
    samples = sample_ddpm(model, n_samples, dim, T, betas, alphas, alphas_cumprod, device)
    ema.restore()

    print(f"{n_samples} Samples: range [{samples.min():.4f}, {samples.max():.4f}]")

    print(f"Using raw samples")
    samples_unscaled = samples

    test_df_real = pd.read_csv(f'{CONFIG["data_path"]}test_combo_scaled.csv')
    syn_df = pd.DataFrame(samples_unscaled, columns=test_df_real.columns)
            
    if config_path:
        syn_df.to_csv('synthetic_data.csv', index=False)
        print(f"Saved: synthetic_data.csv")
    return syn_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=str, default=os.path.join(CONFIG['data_path'], "synthetic.csv"))
    args = parser.parse_args()

    df = generate_synthetic(args.n, args.out)
    print("[generate] Done. Shape:", df.shape)
