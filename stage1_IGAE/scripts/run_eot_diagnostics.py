#!/usr/bin/env python3
"""
EOT diagnostics re-run for all Stage-1 checkpoints.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/run_eot_diagnostics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vae_sigreg import (
    OvercompleteVariationalAE,
    compute_kl_active_dims,
    compute_marginal_moments,
    compute_z_frechet_distance,
)

CKPT_BASE = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
VAL_FRAC   = 0.05
BATCH_SIZE = 512
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE  = torch.bfloat16

def load_val_loader():
    import numpy as np
    obj = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    emb = obj.float() if isinstance(obj, torch.Tensor) else next(iter(obj.values())).float()
    n_val = max(1, int(len(emb) * VAL_FRAC))
    val_ds = TensorDataset(emb[-n_val:])
    return DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

@torch.no_grad()
def run_eot(ckpt_path: Path, val_loader: DataLoader):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    model = OvercompleteVariationalAE(
        input_dim  = int(saved_args.get("input_dim",  768)),
        latent_dim = int(saved_args.get("latent_dim", 6144)),
        hidden_dim = saved_args.get("hidden_dim", None),
        num_layers = int(saved_args.get("num_layers", 4)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_z, all_mu, all_lgv = [], [], []
    for (x,) in val_loader:
        x = x.to(DEVICE)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
            out = model(x)
        all_z.append(out["z"].float().cpu())
        all_mu.append(out["mu"].float().cpu())
        all_lgv.append(out["logvar"].float().cpu())

    z_all  = torch.cat(all_z).to(DEVICE)
    mu_all = torch.cat(all_mu).to(DEVICE)
    lgv_all= torch.cat(all_lgv).to(DEVICE)

    fd  = compute_z_frechet_distance(z_all)
    kl  = compute_kl_active_dims(mu_all, lgv_all, thresholds=(0.1, 0.5, 1.0))
    mom = compute_marginal_moments(z_all)

    # save report
    lines = [
        "# Stage-1 End-of-Training Diagnostics",
        f"checkpoint: {ckpt_path}",
        f"beta_kl={saved_args.get('beta_kl')}  lambda_sigreg={saved_args.get('lambda_sigreg')}  kl_warmup={saved_args.get('beta_kl_warmup_epochs')}",
        "",
        "## z-space Frechet Distance",
        f"  z_fd_diag = {fd['z_fd_diag']:.4f}",
        f"  z_fd_proj = {fd['z_fd_proj']:.6f}",
        "",
        "## KL Active Dimensions",
        f"  kl_total   = {kl['kl_total']:.1f}",
        f"  active@0.1 = {int(kl['active_dims_at_0_1'])}",
        f"  active@0.5 = {int(kl['active_dims_at_0_5'])}",
        f"  active@1.0 = {int(kl['active_dims_at_1'])}",
        "",
        "## Marginal Moments",
        f"  mean_abs_mean = {mom['marginal_mean_abs_mean']:.4f}",
        f"  std_mean      = {mom['marginal_std_mean']:.4f}",
        f"  std_std       = {mom['marginal_std_std']:.4f}",
        f"  skew_abs_mean = {mom['marginal_skew_abs_mean']:.4f}",
        f"  kurt_abs_mean = {mom['marginal_excess_kurt_abs_mean']:.4f}",
    ]
    (ckpt_path.parent / "diagnostics_report.md").write_text("\n".join(lines) + "\n")

    # KL per-dim 분포 통계 (집중도 측정)
    kl_dim      = 0.5 * (mu_all.pow(2) + lgv_all.exp() - lgv_all - 1).mean(0)
    kl_dim_std  = kl_dim.std().item()
    kl_dim_max  = kl_dim.max().item()
    kl_dim_mean = kl_dim.mean().item()
    kl_conc     = kl_dim_max / max(kl_dim_mean, 1e-8)

    return {
        "z_fd_diag":      fd["z_fd_diag"],
        "z_fd_proj":      fd["z_fd_proj"],
        "kl_total":       kl["kl_total"],
        "kl_dim_std":     kl_dim_std,
        "kl_dim_max":     kl_dim_max,
        "kl_concentration": kl_conc,
        "active_0_1":     int(kl["active_dims_at_0_1"]),
        "active_0_5":     int(kl["active_dims_at_0_5"]),
        "active_1_0":     int(kl["active_dims_at_1"]),
        "std_mean":       mom["marginal_std_mean"],
        "std_std":        mom["marginal_std_std"],
    }


def main():
    ckpts = sorted(CKPT_BASE.glob("*/best.pt"))
    if not ckpts:
        print("No checkpoints found.")
        return

    print(f"Found {len(ckpts)} checkpoints. Loading val set...", flush=True)
    val_loader = load_val_loader()
    print(f"Val samples: {len(val_loader.dataset)}\n", flush=True)

    results = []
    for ckpt in ckpts:
        tag = ckpt.parent.name
        print(f"[{tag}] running...", flush=True)
        stats = run_eot(ckpt, val_loader)
        results.append((tag, stats))
        print(
            f"  z_fd_diag={stats['z_fd_diag']:.1f}  kl={stats['kl_total']:.1f}"
            f"  kl_std={stats['kl_dim_std']:.4f}  kl_max={stats['kl_dim_max']:.3f}"
            f"  conc={stats['kl_concentration']:.1f}x"
            f"  active@0.1={stats['active_0_1']}",
            flush=True,
        )

    # summary table
    print("\n" + "="*100)
    print(f"{'run':<35} {'z_fd_diag':>10} {'kl_total':>9} {'kl_std':>8} {'kl_max':>8} {'conc':>7} {'act@0.1':>8}")
    print("-"*100)
    results.sort(key=lambda x: x[1]["z_fd_diag"])
    for tag, s in results:
        print(
            f"{tag:<35} {s['z_fd_diag']:>10.1f} {s['kl_total']:>9.1f}"
            f" {s['kl_dim_std']:>8.4f} {s['kl_dim_max']:>8.3f} {s['kl_concentration']:>6.1f}x"
            f" {s['active_0_1']:>8} {s['active_0_5']:>8} {s['active_1_0']:>8}"
        )
    print("="*100)
    print("\n(sorted by z_fd_diag ascending — lower = better prior match)")
    print("Stage-2 candidates: top 3 by z_fd_diag with kl_total in [30, 80]")


if __name__ == "__main__":
    main()
