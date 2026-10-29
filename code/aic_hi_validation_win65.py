#!/usr/bin/env python3
"""
AIC Health Indicator Validation - Win65 Model
==============================================

使用 window_size_ablation/win_65/model.pth 模型进行推理。
对比两种计分方式:
1. 不用 Speed: s_t = 0.50*E_t + 0.50*D_t  
2. 用 Speed: s_t = 0.10*E_t + 0.30*D_t + 0.60*S_t

计算 CDI/SHI 并验证与物理 proxy (VDV, PeakJerk) 的相关性。

Model Parameters (from seasonal_analysis_win65.py):
- WIN_SIZE = 65
- IN_DIM = 3 (X, Y, Z only, no magnitude G)
- HIDDEN_DIM = 128
- LATENT_DIM = 64
- CODEBOOK_SIZE = 1024
- D_MODEL = 128
- N_HEADS = 8
- N_LAYERS = 4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ============================================================================
# Configuration for Win65 Model
# ============================================================================
class ConfigWin65:
    """Configuration for win_size=65 model"""
    # Data parameters
    FS = 10  # Sampling rate (Hz)
    WIN_SIZE = 65  # Window size in samples (6.5s)
    STEP = 5  # Stride in samples (0.5s)
    
    # Model parameters (from seasonal_analysis_win65.py)
    IN_DIM = 3  # Only X, Y, Z (no magnitude G)
    HIDDEN_DIM = 128
    LATENT_DIM = 64
    CODEBOOK_SIZE = 1024
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 4
    
    # Health indicator parameters
    K_SMOOTH = 5  # Rolling median half-window
    
    # Proxy computation
    HP_CUTOFF = 0.3
    HP_ORDER = 4
    
    # Validation
    N_BOOTSTRAP = 2000
    RANDOM_SEED = 42
    
    # Feature columns (only 3 axes, no G)
    FEATURE_COLUMNS = ['X', 'Y', 'Z']

config = ConfigWin65()


# ============================================================================
# Model Architecture (copied from tran_update2.py)
# ============================================================================
class RelPosBias(nn.Module):
    def __init__(self, n_heads: int, max_dist: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist
        self.rel_bias = nn.Parameter(torch.zeros(2*max_dist + 1, n_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, q_len: int, k_len: int, device=None):
        device = device or self.rel_bias.device
        qs = torch.arange(q_len, device=device)[:, None]
        ks = torch.arange(k_len, device=device)[None, :]
        rel = (ks - qs).clamp(-self.max_dist, self.max_dist) + self.max_dist
        bias = self.rel_bias[rel]
        bias = bias.permute(2, 0, 1)
        return bias


class TSBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_mult=4, drop=0.1, max_dist=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_dist = max_dist

        self.qkv_proj = nn.Linear(d_model, d_model*3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.bias = RelPosBias(n_heads, max_dist)
        self.dropout = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*ff_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model*ff_mult, d_model),
            nn.Dropout(drop)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (C // self.n_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn + self.bias(T, T, x.device)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x, attn


class SeqEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, n_layers=2, bidirectional=True, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True, 
                           bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, latent_dim)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        z_e = self.proj(out)
        return z_e


class VectorQuantizerEMA(nn.Module):
    def __init__(self, K=512, D=32, decay=0.99, eps=1e-5, commitment_cost=0.25):
        super().__init__()
        self.K, self.D, self.decay, self.eps = K, D, decay, eps
        self.commitment_cost = commitment_cost
        self.embed = nn.Embedding(K, D)
        self.register_buffer('ema_count', torch.zeros(K))
        self.register_buffer('ema_weight', self.embed.weight.data.clone())
        nn.init.kaiming_uniform_(self.embed.weight)
        
    def forward(self, z_e):
        flat = z_e.reshape(-1, self.D)
        flat_norm = torch.sum(flat ** 2, dim=1, keepdim=True)
        embed_norm = torch.sum(self.embed.weight ** 2, dim=1)
        dist = flat_norm + embed_norm - 2 * torch.matmul(flat, self.embed.weight.t())
        encoding_indices = torch.argmin(dist, dim=1)
        quantized = self.embed(encoding_indices).view_as(z_e)
        
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        q_latent_loss = F.mse_loss(quantized, z_e.detach())
        loss = e_latent_loss + self.commitment_cost * q_latent_loss
        quantized = z_e + (quantized - z_e).detach()
        return quantized, loss, encoding_indices


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        
    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = in_channels if i == 0 else channels[i-1]
            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        return x.transpose(1, 2)


class VQTransAE(nn.Module):
    def __init__(self, win_size, in_dim=3, hidden=128, latent=64, codebook=1024,
                 d_model=128, heads=8, layers=4, dropout=0.1):
        super().__init__()
        self.encoder = SeqEncoder(in_dim, hidden, latent, n_layers=2, bidirectional=True, dropout=dropout)
        self.quant = VectorQuantizerEMA(codebook, latent, decay=0.99, eps=1e-5, commitment_cost=0.25)
        self.embed = nn.Linear(latent, d_model)
        self.tcn = TCN(d_model, [d_model, d_model, d_model], kernel_size=3, dropout=dropout)
        self.tf_layers = nn.ModuleList([TSBlock(d_model, heads, ff_mult=4, drop=dropout, max_dist=win_size) for _ in range(layers)])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.ReLU(),
            nn.Linear(d_model, in_dim)
        )
        self.use_skip = True
        if self.use_skip:
            self.skip_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, loss_vq, indices = self.quant(z_e)
        h = self.embed(z_q)
        h = self.tcn(h)
        attn_list = []
        for blk in self.tf_layers:
            h, attn = blk(h)
            attn_list.append(attn)
        recon = self.decoder(h)
        if self.use_skip:
            recon = recon + self.skip_proj(x)
        return recon, attn_list, loss_vq, indices


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load VQTransAE model with win65 config"""
    model = VQTransAE(
        win_size=config.WIN_SIZE,
        in_dim=config.IN_DIM,
        hidden=config.HIDDEN_DIM,
        latent=config.LATENT_DIM,
        codebook=config.CODEBOOK_SIZE,
        d_model=config.D_MODEL,
        heads=config.N_HEADS,
        layers=config.N_LAYERS
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Win size: {config.WIN_SIZE}, In dim: {config.IN_DIM}")
    return model


# ============================================================================
# Signal Processing
# ============================================================================
def butter_highpass_filtfilt(x: np.ndarray, fs: float, cutoff_hz: float,
                              order: int = 4) -> np.ndarray:
    """Apply Butterworth high-pass filter (zero-phase)"""
    nyquist = fs / 2
    normalized_cutoff = np.clip(cutoff_hz / nyquist, 0.001, 0.999)
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x
    
    return signal.filtfilt(b, a, x)


def compute_window_proxies(ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
                           fs: float = 10, hp_cutoff: float = 0.3) -> Dict[str, float]:
    """Compute VDV and PeakJerk for a single window."""
    ax_hp = butter_highpass_filtfilt(ax, fs, hp_cutoff)
    ay_hp = butter_highpass_filtfilt(ay, fs, hp_cutoff)
    az_hp = butter_highpass_filtfilt(az, fs, hp_cutoff)
    
    a_m = np.sqrt(ax_hp**2 + ay_hp**2 + az_hp**2)
    
    VDV_w = np.power(np.mean(np.abs(a_m)**4), 0.25)
    
    dt = 1.0 / fs
    j_m = np.zeros_like(a_m)
    j_m[1:] = np.abs(np.diff(a_m)) / dt
    PeakJerk_w = np.max(np.abs(j_m))
    
    return {'VDV_w': VDV_w, 'PeakJerk_w': PeakJerk_w}


# ============================================================================
# Anomaly Score Computation
# ============================================================================
def compute_scores_from_model(model: nn.Module, data: np.ndarray, 
                               device: torch.device) -> Dict[str, np.ndarray]:
    """
    Compute raw E_t (reconstruction error) and D_t (codebook distance) from model.
    
    Returns normalized scores.
    """
    model.eval()
    
    n_windows = (len(data) - config.WIN_SIZE) // config.STEP + 1
    
    E_scores = []
    D_scores = []
    
    with torch.no_grad():
        for i in range(n_windows):
            start = i * config.STEP
            end = start + config.WIN_SIZE
            
            window = data[start:end]
            
            if np.any(np.isnan(window)):
                E_scores.append(np.nan)
                D_scores.append(np.nan)
                continue
            
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)
            
            recon, attn_list, loss_vq, indices = model(x)
            
            # Reconstruction error
            E_t = F.mse_loss(recon, x, reduction='none').mean(dim=[1, 2]).item()
            
            # Codebook distance (from VQ layer)
            z_e = model.encoder(x)
            codebook = model.quant.embed.weight
            flat = z_e.view(-1, z_e.shape[-1])
            dists = torch.cdist(flat, codebook)
            min_dists = dists.min(dim=1).values
            D_t = min_dists.view(z_e.shape[0], -1).mean(dim=1).item()
            
            E_scores.append(E_t)
            D_scores.append(D_t)
    
    E_scores = np.array(E_scores)
    D_scores = np.array(D_scores)
    
    # Normalize
    E_min, E_max = np.nanmin(E_scores), np.nanmax(E_scores)
    D_min, D_max = np.nanmin(D_scores), np.nanmax(D_scores)
    
    E_norm = (E_scores - E_min) / (E_max - E_min + 1e-8)
    D_norm = (D_scores - D_min) / (D_max - D_min + 1e-8)
    
    return {
        'E_t': E_scores,
        'D_t': D_scores,
        'E_norm': E_norm,
        'D_norm': D_norm,
    }


def compute_speed_score(speeds: np.ndarray, n_windows: int, 
                         win_size: int, step: int) -> np.ndarray:
    """
    Compute Speed-based smoothness score S_t.
    
    Lower speed = rougher surface = higher anomaly score.
    Formula: S_t = 1 - (speed_w - speed_min) / (speed_max - speed_min)
    """
    S_raw = []
    
    for i in range(n_windows):
        start = i * step
        end = start + win_size
        center = start + win_size // 2
        
        if center < len(speeds):
            speed = speeds[center]
            S_raw.append(speed if not np.isnan(speed) else np.nan)
        else:
            S_raw.append(np.nan)
    
    S_raw = np.array(S_raw)
    S_min, S_max = np.nanmin(S_raw), np.nanmax(S_raw)
    
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_raw)
    else:
        # Invert: low speed = high anomaly score
        S_norm = 1.0 - (S_raw - S_min) / (S_max - S_min + 1e-8)
    
    return S_norm, S_raw


def compute_composite_scores(E_norm: np.ndarray, D_norm: np.ndarray, 
                              S_norm: np.ndarray = None,
                              use_speed: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute composite anomaly scores with two methods.
    
    Method 1 (no Speed): s_t = 0.50*E_t + 0.50*D_t
    Method 2 (with Speed): s_t = 0.10*E_t + 0.30*D_t + 0.60*S_t
    """
    if use_speed and S_norm is not None:
        s_t = 0.10 * E_norm + 0.30 * D_norm + 0.60 * S_norm
        formula = "s_t = 0.10*E + 0.30*D + 0.60*S"
    else:
        s_t = 0.50 * E_norm + 0.50 * D_norm
        formula = "s_t = 0.50*E + 0.50*D"
    
    return {'s_t': s_t, 'formula': formula}


# ============================================================================
# Health Indicators (CDI/SHI)
# ============================================================================
def compute_health_indicators(s_t: np.ndarray, k_smooth: int = 5) -> Dict[str, np.ndarray]:
    """
    Compute CDI and SHI from anomaly score time series.
    
    CDI: Rolling median of s_t (persistent roughness)
    SHI: max(0, smooth→delta) (abrupt transitions)
    """
    window = 2 * k_smooth + 1
    s_smooth = pd.Series(s_t).rolling(window=window, center=True, min_periods=1).median().values
    
    CDI_t = s_smooth
    
    delta_s = np.zeros_like(s_smooth)
    delta_s[1:] = np.diff(s_smooth)
    SHI_t = np.maximum(0, delta_s)
    
    return {'s_smooth': s_smooth, 'CDI_t': CDI_t, 'SHI_t': SHI_t}


# ============================================================================
# Proxy Computation
# ============================================================================
def compute_proxies_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VDV_w and PeakJerk_w from raw acceleration data."""
    ax_full = df['X'].values
    ay_full = df['Y'].values
    az_full = df['Z'].values
    
    n_windows = (len(df) - config.WIN_SIZE) // config.STEP + 1
    
    VDV_list = []
    PeakJerk_list = []
    
    for i in range(n_windows):
        start = i * config.STEP
        end = start + config.WIN_SIZE
        
        ax_w = ax_full[start:end]
        ay_w = ay_full[start:end]
        az_w = az_full[start:end]
        
        if np.any(np.isnan(ax_w)) or np.any(np.isnan(ay_w)) or np.any(np.isnan(az_w)):
            VDV_list.append(np.nan)
            PeakJerk_list.append(np.nan)
            continue
        
        proxies = compute_window_proxies(ax_w, ay_w, az_w, config.FS, config.HP_CUTOFF)
        VDV_list.append(proxies['VDV_w'])
        PeakJerk_list.append(proxies['PeakJerk_w'])
    
    return pd.DataFrame({'VDV_w': VDV_list, 'PeakJerk_w': PeakJerk_list})


# ============================================================================
# Validation Metrics
# ============================================================================
def spearman_with_ci(x: np.ndarray, y: np.ndarray, 
                     n_boot: int = 2000, seed: int = 42) -> Dict:
    """Compute Spearman correlation with bootstrap CI"""
    np.random.seed(seed)
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return {'rho': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': 0}
    
    rho, _ = stats.spearmanr(x_clean, y_clean)
    
    n = len(x_clean)
    boot_rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(x_clean[idx], y_clean[idx])
        if np.isfinite(r):
            boot_rhos.append(r)
    
    if len(boot_rhos) == 0:
        return {'rho': rho, 'ci_low': np.nan, 'ci_high': np.nan, 'n': n}
    
    ci_low = np.percentile(boot_rhos, 2.5)
    ci_high = np.percentile(boot_rhos, 97.5)
    
    return {'rho': rho, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n}


# ============================================================================
# Main Pipeline
# ============================================================================
def run_comparison_pipeline(
    model_path: str,
    test_csv: str,
    outdir: str,
    k_smooth: int = 5,
):
    """
    Run comparison between Speed-based and non-Speed scoring.
    
    Parameters
    ----------
    model_path : str
        Path to win_65/model.pth
    test_csv : str
        Path to test data CSV (with X, Y, Z, Speed columns)
    outdir : str
        Output directory
    """
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ================================================================
    # Step 1: Load model
    # ================================================================
    print("\n" + "="*70)
    print("[1] Loading VQTransAE model (win65)")
    print("="*70)
    
    model = load_model(model_path, device)
    
    # ================================================================
    # Step 2: Load data
    # ================================================================
    print("\n[2] Loading test data...")
    df_raw = pd.read_csv(test_csv)
    print(f"  Loaded {len(df_raw)} samples")
    print(f"  Columns: {list(df_raw.columns)}")
    
    # Prepare features (X, Y, Z only)
    for col in ['X', 'Y', 'Z']:
        if col not in df_raw.columns:
            raise ValueError(f"Missing column: {col}")
    
    df_raw[['X', 'Y', 'Z']] = df_raw[['X', 'Y', 'Z']].fillna(0)
    data = df_raw[['X', 'Y', 'Z']].values.astype(np.float32)
    
    # Speed
    speeds = None
    if 'Speed' in df_raw.columns:
        speeds = df_raw['Speed'].values
        print(f"  Found Speed column: min={np.nanmin(speeds):.2f}, max={np.nanmax(speeds):.2f}")
    
    n_windows = (len(data) - config.WIN_SIZE) // config.STEP + 1
    print(f"  N windows: {n_windows}")
    
    # ================================================================
    # Step 3: Compute model-based scores (E, D)
    # ================================================================
    print("\n[3] Computing model-based scores (E, D)...")
    scores = compute_scores_from_model(model, data, device)
    
    E_norm = scores['E_norm']
    D_norm = scores['D_norm']
    
    print(f"  E_t: min={np.nanmin(scores['E_t']):.6f}, max={np.nanmax(scores['E_t']):.6f}")
    print(f"  D_t: min={np.nanmin(scores['D_t']):.6f}, max={np.nanmax(scores['D_t']):.6f}")
    
    # ================================================================
    # Step 4: Compute Speed score (if available)
    # ================================================================
    S_norm = None
    if speeds is not None:
        print("\n[4] Computing Speed-based score (S)...")
        S_norm, S_raw = compute_speed_score(speeds, n_windows, config.WIN_SIZE, config.STEP)
        print(f"  Speed range: {np.nanmin(S_raw):.2f} - {np.nanmax(S_raw):.2f} m/s")
        print(f"  S_t range: {np.nanmin(S_norm):.4f} - {np.nanmax(S_norm):.4f}")
    
    # ================================================================
    # Step 5: Compute composite scores (both methods)
    # ================================================================
    print("\n[5] Computing composite scores...")
    
    # Method 1: Without Speed
    result_no_speed = compute_composite_scores(E_norm, D_norm, None, use_speed=False)
    s_t_no_speed = result_no_speed['s_t']
    print(f"  Without Speed: {result_no_speed['formula']}")
    
    # Method 2: With Speed
    if S_norm is not None:
        result_with_speed = compute_composite_scores(E_norm, D_norm, S_norm, use_speed=True)
        s_t_with_speed = result_with_speed['s_t']
        print(f"  With Speed: {result_with_speed['formula']}")
    else:
        s_t_with_speed = None
    
    # ================================================================
    # Step 6: Compute CDI/SHI
    # ================================================================
    print("\n[6] Computing health indicators (CDI/SHI)...")
    
    hi_no_speed = compute_health_indicators(s_t_no_speed, k_smooth)
    CDI_no_speed = hi_no_speed['CDI_t']
    SHI_no_speed = hi_no_speed['SHI_t']
    
    print(f"  [No Speed] CDI: min={np.nanmin(CDI_no_speed):.4f}, median={np.nanmedian(CDI_no_speed):.4f}, max={np.nanmax(CDI_no_speed):.4f}")
    print(f"  [No Speed] SHI: min={np.nanmin(SHI_no_speed):.4f}, 95th={np.nanpercentile(SHI_no_speed, 95):.4f}, max={np.nanmax(SHI_no_speed):.4f}")
    
    if s_t_with_speed is not None:
        hi_with_speed = compute_health_indicators(s_t_with_speed, k_smooth)
        CDI_with_speed = hi_with_speed['CDI_t']
        SHI_with_speed = hi_with_speed['SHI_t']
        
        print(f"  [With Speed] CDI: min={np.nanmin(CDI_with_speed):.4f}, median={np.nanmedian(CDI_with_speed):.4f}, max={np.nanmax(CDI_with_speed):.4f}")
        print(f"  [With Speed] SHI: min={np.nanmin(SHI_with_speed):.4f}, 95th={np.nanpercentile(SHI_with_speed, 95):.4f}, max={np.nanmax(SHI_with_speed):.4f}")
    
    # ================================================================
    # Step 7: Compute proxies
    # ================================================================
    print("\n[7] Computing proxy metrics (VDV, PeakJerk)...")
    df_proxies = compute_proxies_from_raw(df_raw)
    VDV_w = df_proxies['VDV_w'].values
    PeakJerk_w = df_proxies['PeakJerk_w'].values
    
    print(f"  VDV: min={np.nanmin(VDV_w):.4f}, median={np.nanmedian(VDV_w):.4f}, max={np.nanmax(VDV_w):.4f}")
    print(f"  PeakJerk: min={np.nanmin(PeakJerk_w):.4f}, median={np.nanmedian(PeakJerk_w):.4f}, max={np.nanmax(PeakJerk_w):.4f}")
    
    # ================================================================
    # Step 8: Validation - Spearman correlations
    # ================================================================
    print("\n" + "="*70)
    print("[8] Validation: Spearman Correlations")
    print("="*70)
    
    results = {
        'config': {
            'model_path': str(model_path),
            'test_csv': str(test_csv),
            'win_size': config.WIN_SIZE,
            'step': config.STEP,
            'in_dim': config.IN_DIM,
            'k_smooth': k_smooth,
            'n_windows': n_windows,
        },
        'methods': {}
    }
    
    # Method 1: No Speed
    print("\n--- Method 1: Without Speed (s_t = 0.50*E + 0.50*D) ---")
    correlations_no_speed = {}
    
    for name, (hi_vals, proxy_vals) in [
        ('CDI_vs_VDV', (CDI_no_speed, VDV_w)),
        ('SHI_vs_PeakJerk', (SHI_no_speed, PeakJerk_w)),
        ('s_t_vs_VDV', (s_t_no_speed, VDV_w)),
        ('s_t_vs_PeakJerk', (s_t_no_speed, PeakJerk_w)),
    ]:
        corr = spearman_with_ci(hi_vals, proxy_vals)
        correlations_no_speed[name] = corr
        print(f"  {name}: rho={corr['rho']:.4f} [{corr['ci_low']:.3f}, {corr['ci_high']:.3f}] (n={corr['n']})")
    
    results['methods']['no_speed'] = {
        'formula': 's_t = 0.50*E + 0.50*D',
        'correlations': correlations_no_speed
    }
    
    # Method 2: With Speed
    if s_t_with_speed is not None:
        print("\n--- Method 2: With Speed (s_t = 0.10*E + 0.30*D + 0.60*S) ---")
        correlations_with_speed = {}
        
        for name, (hi_vals, proxy_vals) in [
            ('CDI_vs_VDV', (CDI_with_speed, VDV_w)),
            ('SHI_vs_PeakJerk', (SHI_with_speed, PeakJerk_w)),
            ('s_t_vs_VDV', (s_t_with_speed, VDV_w)),
            ('s_t_vs_PeakJerk', (s_t_with_speed, PeakJerk_w)),
        ]:
            corr = spearman_with_ci(hi_vals, proxy_vals)
            correlations_with_speed[name] = corr
            print(f"  {name}: rho={corr['rho']:.4f} [{corr['ci_low']:.3f}, {corr['ci_high']:.3f}] (n={corr['n']})")
        
        results['methods']['with_speed'] = {
            'formula': 's_t = 0.10*E + 0.30*D + 0.60*S',
            'correlations': correlations_with_speed
        }
    
    # ================================================================
    # Step 9: Save results
    # ================================================================
    print("\n[9] Saving results...")
    
    # Save JSON
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed CSV
    df_results = pd.DataFrame({
        'window_idx': np.arange(n_windows),
        'E_norm': E_norm,
        'D_norm': D_norm,
        's_t_no_speed': s_t_no_speed,
        'CDI_no_speed': CDI_no_speed,
        'SHI_no_speed': SHI_no_speed,
        'VDV_w': VDV_w,
        'PeakJerk_w': PeakJerk_w,
    })
    
    if S_norm is not None:
        df_results['S_norm'] = S_norm
        df_results['s_t_with_speed'] = s_t_with_speed
        df_results['CDI_with_speed'] = CDI_with_speed
        df_results['SHI_with_speed'] = SHI_with_speed
    
    df_results.to_csv(output_dir / 'detailed_scores.csv', index=False)
    
    # ================================================================
    # Step 10: Generate comparison plot
    # ================================================================
    print("\n[10] Generating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: CDI vs VDV comparison
    ax = axes[0, 0]
    ax.scatter(VDV_w, CDI_no_speed, alpha=0.3, s=10, label='No Speed', color='blue')
    if s_t_with_speed is not None:
        ax.scatter(VDV_w, CDI_with_speed, alpha=0.3, s=10, label='With Speed', color='red')
    ax.set_xlabel('VDV (Vibration Dose Value)')
    ax.set_ylabel('CDI (Comfort Degradation Index)')
    ax.set_title('CDI vs VDV Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: SHI vs PeakJerk comparison
    ax = axes[0, 1]
    ax.scatter(PeakJerk_w, SHI_no_speed, alpha=0.3, s=10, label='No Speed', color='blue')
    if s_t_with_speed is not None:
        ax.scatter(PeakJerk_w, SHI_with_speed, alpha=0.3, s=10, label='With Speed', color='red')
    ax.set_xlabel('PeakJerk')
    ax.set_ylabel('SHI (Safety Hazard Index)')
    ax.set_title('SHI vs PeakJerk Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Score time series
    ax = axes[1, 0]
    x_axis = np.arange(min(500, n_windows))
    ax.plot(x_axis, s_t_no_speed[:len(x_axis)], alpha=0.7, label='No Speed', color='blue')
    if s_t_with_speed is not None:
        ax.plot(x_axis, s_t_with_speed[:len(x_axis)], alpha=0.7, label='With Speed', color='red')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Anomaly Score s_t')
    ax.set_title('Score Time Series (first 500 windows)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Correlation comparison bar chart
    ax = axes[1, 1]
    metrics = ['CDI_vs_VDV', 'SHI_vs_PeakJerk']
    x = np.arange(len(metrics))
    width = 0.35
    
    rhos_no_speed = [correlations_no_speed[m]['rho'] for m in metrics]
    ax.bar(x - width/2, rhos_no_speed, width, label='No Speed', color='blue', alpha=0.7)
    
    if s_t_with_speed is not None:
        rhos_with_speed = [correlations_with_speed[m]['rho'] for m in metrics]
        ax.bar(x + width/2, rhos_with_speed, width, label='With Speed', color='red', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylabel('Spearman Correlation (rho)')
    ax.set_title('Correlation Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: win65 (IN_DIM={config.IN_DIM}, WIN_SIZE={config.WIN_SIZE})")
    print(f"Test data: {test_csv}")
    print(f"Windows: {n_windows}")
    print()
    print("Spearman Correlations:")
    print("-"*50)
    print(f"{'Metric':<20} {'No Speed':>12} {'With Speed':>12}")
    print("-"*50)
    
    for metric in ['CDI_vs_VDV', 'SHI_vs_PeakJerk']:
        rho_no = correlations_no_speed[metric]['rho']
        if s_t_with_speed is not None:
            rho_with = correlations_with_speed[metric]['rho']
            print(f"{metric:<20} {rho_no:>12.4f} {rho_with:>12.4f}")
        else:
            print(f"{metric:<20} {rho_no:>12.4f} {'N/A':>12}")
    
    print("="*70)
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='AIC Health Indicator Validation with Win65 Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aic_hi_validation_win65.py \\
      --model_path code_version_sep18/window_size_ablation/win_65/model.pth \\
      --test_csv vqtransae_results/processed_data/test.csv \\
      --outdir aic_validation_win65_results
        """
    )
    
    parser.add_argument('--model_path', '-m', type=str, required=True,
                        help='Path to win_65 model checkpoint')
    parser.add_argument('--test_csv', '-t', type=str, required=True,
                        help='Path to test data CSV (with X, Y, Z, Speed columns)')
    parser.add_argument('--outdir', '-o', type=str, default='aic_validation_win65',
                        help='Output directory')
    parser.add_argument('--k_smooth', type=int, default=5,
                        help='Smoothing half-window for CDI/SHI (default: 5)')
    
    args = parser.parse_args()
    
    run_comparison_pipeline(
        model_path=args.model_path,
        test_csv=args.test_csv,
        outdir=args.outdir,
        k_smooth=args.k_smooth,
    )


if __name__ == '__main__':
    main()
