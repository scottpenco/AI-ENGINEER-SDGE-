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
    'model_path': '../models',
}


test_df_real = pd.read_csv(f'{CONFIG["data_path"]}test_combo_scaled.csv')
syn_df = pd.read_csv('../synthetic_data.csv')

def evaluate(real_df, syn_df, continuous_cols):
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)

        metrics = {}

        # KS-test
        print("\n1. KS-TEST")
        ks_stats = []
        for col in continuous_cols[:5]:
            try:
                stat, pval = ks_2samp(real_df[col], syn_df[col])
                ks_stats.append(pval)
                status = "✓" if pval > 0.05 else "✗"
                print(f"  {col}: p={pval:.4f} {status}")
            except:
                ks_stats.append(0)
        metrics['ks_test_pass_rate'] = sum(1 for p in ks_stats if p > 0.05) / len(ks_stats) if ks_stats else 0
        print(f"  Pass rate: {metrics['ks_test_pass_rate']:.1%}")

        # Correlation
        print("\n2. CORRELATION")
        try:
            real_corr = real_df[continuous_cols].corr()
            syn_corr = syn_df[continuous_cols].corr()
            frobenius = np.linalg.norm(real_corr - syn_corr, 'fro')
            metrics['correlation_frobenius'] = frobenius
            print(f"  Frobenius: {frobenius:.4f}")
        except:
            metrics['correlation_frobenius'] = np.nan

        # Diversity
        print("\n3. DIVERSITY")
        try:
            syn_data = syn_df[continuous_cols].values
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(syn_data)
            distances, _ = nn.kneighbors(syn_data)
            distances = distances[:, 1]
            metrics['duplicate_ratio'] = (distances < 1e-4).sum() / len(syn_data)
            metrics['mean_nn_distance'] = distances.mean()
            print(f"  Duplicate: {metrics['duplicate_ratio']:.4f}, NN: {metrics['mean_nn_distance']:.4f}")
        except:
            metrics['duplicate_ratio'] = 0
            metrics['mean_nn_distance'] = 0

        metrics['utility_ratio'] = 0
        metrics['chi2_test_pass_rate'] = 0

        return metrics
def main():
    continuous_cols = [col for col in test_df_real.columns if col != 'death_event']
    metrics = evaluate(real_df=test_df_real, syn_df=syn_df, continuous_cols=continuous_cols)

    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    print("="*70)
    print(f"\nResults:")
    print(f"  KS-test: {metrics['ks_test_pass_rate']:.1%}")
    print(f"  Correlation: {metrics['correlation_frobenius']:.4f}")

if __name__ == "__main__":
    main()