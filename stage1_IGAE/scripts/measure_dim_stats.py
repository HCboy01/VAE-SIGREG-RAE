#!/usr/bin/env python3
"""
차원별 μ 평균/편차 분포 측정.

Usage:
    python scripts/measure_dim_stats.py --ckpt checkpoints/s1_grid_b1e-3_l100_lindec/best.pt
"""
import sys, argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR  = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/dim_stats")

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    m = OvercompleteVariationalAE(
        input_dim  = int(a.get("input_dim", 768)),
        latent_dim = int(a.get("latent_dim", 6144)),
        hidden_dim = a.get("hidden_dim"),
        num_layers = int(a.get("num_layers", 4)),
    ).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval().requires_grad_(False)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = Path(args.ckpt).parents[0].name

    model = load_model(args.ckpt)
    raw = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
    emb = (raw if isinstance(raw, torch.Tensor) else next(iter(raw.values()))).float()

    mus = []
    with torch.no_grad():
        for i in range(0, len(emb), 1024):
            mu, _ = model.encode(emb[i:i+1024].to(DEVICE))
            mus.append(mu.float().cpu())
    mu_all = torch.cat(mus)   # [N, D]

    dim_mean = mu_all.mean(0)  # [D] 각 dim의 전체 이미지 평균
    dim_std  = mu_all.std(0)   # [D] 각 dim의 이미지 간 편차

    print(f"\n=== {tag} ===")
    print(f"\n[차원별 μ 평균]  (0이어야 prior와 일치)")
    print(f"  전체 평균     : {dim_mean.mean():+.6f}")
    print(f"  표준편차      : {dim_mean.std():.6f}  ← 사용자가 언급한 편차")
    print(f"  min / max     : {dim_mean.min():+.4f} / {dim_mean.max():+.4f}")
    print(f"  |mean| > 0.1  : {(dim_mean.abs()>0.1).sum().item()} dims")
    print(f"  |mean| > 0.3  : {(dim_mean.abs()>0.3).sum().item()} dims")
    print(f"  |mean| > 0.5  : {(dim_mean.abs()>0.5).sum().item()} dims")

    print(f"\n[차원별 μ std]  (이미지마다 얼마나 다르게 인코딩되는가)")
    print(f"  전체 평균     : {dim_std.mean():.6f}")
    print(f"  표준편차      : {dim_std.std():.6f}")
    print(f"  min / max     : {dim_std.min():.4f} / {dim_std.max():.4f}")
    print(f"  std > 0.2     : {(dim_std>0.2).sum().item()} dims")
    print(f"  std > 0.5     : {(dim_std>0.5).sum().item()} dims")
    print(f"  std > 1.0     : {(dim_std>1.0).sum().item()} dims")

    # 플롯
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{tag}\nPer-dimension μ statistics across all images", fontsize=11)

    # 왼쪽: dim_mean 분포
    ax = axes[0]
    ax.hist(dim_mean.numpy(), bins=100, color="#3498db", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="k", ls="--", lw=1.5, label="0 (ideal)")
    ax.axvline( dim_mean.std().item(), color="r", ls=":", lw=1, label=f"±std={dim_mean.std():.4f}")
    ax.axvline(-dim_mean.std().item(), color="r", ls=":", lw=1)
    ax.set_xlabel("dim_mean  (per-dim mean of μ across all images)")
    ax.set_ylabel("# dimensions")
    ax.set_title("Distribution of per-dim mean\n(ideal: spike at 0)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 오른쪽: dim_std 분포
    ax = axes[1]
    ax.hist(dim_std.numpy(), bins=100, color="#e74c3c", alpha=0.8, edgecolor="none")
    ax.axvline(dim_std.mean().item(), color="k", ls="--", lw=1.5,
               label=f"mean={dim_std.mean():.4f}")
    ax.set_xlabel("dim_std  (per-dim std of μ across images)")
    ax.set_ylabel("# dimensions")
    ax.set_title("Distribution of per-dim std\n(higher = more discriminative)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = OUT_DIR / f"{tag}_dim_stats.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {save_path}")

if __name__ == "__main__":
    main()
