"""
Are the latent dims independent?  Gaussian marginals on both axes (per-sample
over dims, and per-dim over samples) do NOT imply independence. Independence
needs: (a) jointly Gaussian and (b) zero cross-dim correlation.

This script encodes all samples, then measures, on a random subset of dims:
  - per-dim across-sample marginal moments (skew, excess kurtosis)
  - cross-dim correlation matrix → |off-diagonal| stats vs the finite-sample
    noise floor 1/sqrt(N) (what you'd see under true independence)

Usage:
    python scripts/z_independence_check.py --ckpt checkpoints/<run>/best.pt [--n_dims 3000]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.visualize_feature_extremes import (
    DEFAULT_IMG_DIR, DEFAULT_EMB_PATH, DEFAULT_VAL_IDX, AMP_DTYPE,
    build_train_image_paths, load_embeddings, load_model,
)

OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sigma_combined_lindec")


@torch.no_grad()
def encode_subset(model, emb, dsub, target, bs, device):
    out = []
    for i in range(0, len(emb), bs):
        x = emb[i:i+bs].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            z = model.encode(x)[0] if target == "mu" else model(x)["z"]
        out.append(z[:, dsub].float().cpu())
    return torch.cat(out, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--target", choices=["z", "mu"], default="z")
    p.add_argument("--n_dims", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.ckpt.parent.name

    paths = build_train_image_paths(DEFAULT_IMG_DIR, DEFAULT_VAL_IDX)
    emb = load_embeddings(DEFAULT_EMB_PATH)
    n_total = min(len(paths), len(emb)); emb = emb[:n_total]
    N = n_total

    model = load_model(args.ckpt, device)
    D = int(model.latent_dim)
    g = torch.Generator().manual_seed(args.seed)
    dsub = torch.randperm(D, generator=g)[:min(args.n_dims, D)]
    Z = encode_subset(model, emb, dsub, args.target, args.batch_size, device)  # [N, Dsub]
    Ds = Z.shape[1]
    print(f"{run_name}: encoded Z[{N}, {Ds}] (subset of D={D}), target={args.target}")

    # ── per-dim across-sample marginal moments ──────────────────────────────
    zc = Z - Z.mean(0, keepdim=True)
    std = Z.std(0, unbiased=False)
    skew = (zc.pow(3).mean(0) / (std.pow(3) + 1e-9)).numpy()
    exkurt = (zc.pow(4).mean(0) / (std.pow(4) + 1e-9) - 3.0).numpy()
    print(f"  per-dim across-sample: mean std={std.mean():.3f}  "
          f"mean|skew|={np.abs(skew).mean():.3f}  mean|exkurt|={np.abs(exkurt).mean():.3f}")

    # ── cross-dim correlation ────────────────────────────────────────────────
    C = torch.corrcoef(Z.T.to(device)).cpu().numpy()
    iu = np.triu_indices(Ds, k=1)
    off = C[iu]
    noise = 1.0 / np.sqrt(N)            # std of sample corr under independence
    print(f"  cross-dim |corr| (off-diag, {len(off):,} pairs):")
    print(f"    mean={np.abs(off).mean():.4f}  median={np.median(np.abs(off)):.4f}  "
          f"p99={np.quantile(np.abs(off),0.99):.4f}  max={np.abs(off).max():.4f}")
    print(f"    noise floor 1/sqrt(N) = {noise:.4f}")
    for thr in (0.05, 0.1, 0.2):
        print(f"    frac |corr|>{thr}: {np.mean(np.abs(off)>thr):.4f}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle(f"{run_name}  |  independence check (Dsub={Ds}, N={N}, target={args.target})",
                 fontsize=12)
    ax[0].hist(off, bins=200, color="#4da6e8", alpha=0.8, density=True)
    for s, c in [(noise, "r"), (-noise, "r"), (2*noise, "orange"), (-2*noise, "orange")]:
        ax[0].axvline(s, color=c, ls="--", lw=1, alpha=0.7)
    ax[0].set_title(f"cross-dim correlations\nmean|corr|={np.abs(off).mean():.4f}  "
                    f"vs noise 1/√N={noise:.4f}  (red=±1σ, orange=±2σ)", fontsize=9.5)
    ax[0].set_xlabel("Pearson correlation (off-diagonal)"); ax[0].set_ylabel("density")
    ax[0].grid(True, alpha=0.2)

    ax[1].hist(exkurt, bins=80, color="#6bbf72", alpha=0.7, label="excess kurtosis")
    ax[1].hist(skew, bins=80, color="#e87c4d", alpha=0.7, label="skew")
    ax[1].axvline(0, color="k", lw=1, ls="--", alpha=0.6)
    ax[1].set_title(f"per-dim across-sample shape\nmean|skew|={np.abs(skew).mean():.3f}  "
                    f"mean|exkurt|={np.abs(exkurt).mean():.3f}", fontsize=9.5)
    ax[1].set_xlabel("value"); ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{run_name}_independence_check.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
