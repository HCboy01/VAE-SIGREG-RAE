"""
Histogram of one sample's z values across latent dims, vs N(0,1).

For a single image, encode -> z [D]. Plot the distribution of those D latent
values (density) against the standard normal prior, on linear and log-y axes,
and report mean/std/skew/excess-kurtosis and sparsity (frac |z|<0.1).

Usage:
    python scripts/z_hist_one_sample.py --ckpt checkpoints/<run>/best.pt \\
        [--img_idx 0] [--target z]
Output:
    visualizations/mu_sigma_combined_lindec/<run>_zhist_sample<idx>.png
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--img_idx", type=int, default=0)
    p.add_argument("--target", choices=["z", "mu"], default="z")
    p.add_argument("--img_dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--emb_path", type=Path, default=DEFAULT_EMB_PATH)
    p.add_argument("--val_idx", type=Path, default=DEFAULT_VAL_IDX)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.ckpt.parent.name

    paths = build_train_image_paths(args.img_dir, args.val_idx)
    emb = load_embeddings(args.emb_path)
    n_total = min(len(paths), len(emb))
    x = emb[args.img_idx : args.img_idx + 1].to(device)
    img_name = paths[args.img_idx].stem if args.img_idx < len(paths) else str(args.img_idx)

    model = load_model(args.ckpt, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            z = (model.encode(x)[0] if args.target == "mu" else model(x)["z"])[0].float().cpu()
    z = z.numpy()
    D = z.size

    mean, std = z.mean(), z.std()
    zc = (z - mean)
    skew = (zc**3).mean() / (std**3 + 1e-9)
    exkurt = (zc**4).mean() / (std**4 + 1e-9) - 3.0
    frac0 = np.mean(np.abs(z) < 0.1)
    print(f"{run_name}  sample idx={args.img_idx} ({img_name})  D={D}  target={args.target}")
    print(f"  mean={mean:+.4f} std={std:.4f} skew={skew:+.3f} excess_kurtosis={exkurt:+.3f}")
    print(f"  range=[{z.min():+.3f}, {z.max():+.3f}]  frac|z|<0.1={frac0:.3f}")

    lim = max(4.0, np.abs(z).max() * 1.05)
    bins = np.linspace(-lim, lim, 121)
    xs = np.linspace(-lim, lim, 400)
    npdf = np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    fig.suptitle(
        f"{run_name}  |  one sample (idx {args.img_idx}, {img_name})  |  "
        f"{args.target} value distribution over D={D} dims  vs N(0,1)", fontsize=12)
    for ax, logy in zip(axes, [False, True]):
        ax.hist(z, bins=bins, density=True, color="#4da6e8", alpha=0.6,
                label=f"{args.target}  (mean={mean:+.2f}, std={std:.2f})")
        ax.plot(xs, npdf, "k--", lw=1.6, label="N(0,1)")
        ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel(f"{args.target} value"); ax.set_xlim(-lim, lim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        if logy:
            ax.set_yscale("log")
            ax.set_title(f"log-y (tails)  |  skew={skew:+.2f}  exkurt={exkurt:+.2f}", fontsize=10)
        else:
            ax.set_ylabel("density")
            ax.set_title(f"linear-y  |  frac|z|<0.1={frac0:.2f}", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{run_name}_zhist_sample{args.img_idx}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
