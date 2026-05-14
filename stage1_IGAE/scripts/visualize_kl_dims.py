"""
Three KL/mu visualizations:
  1. Per-dim mean KL distribution (sorted S-curve)
  2. mu heatmap (images x active dims)
  3. mu histogram per active dim
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE


def load_embeddings(path):
    p = Path(path)
    shape_path = p.with_name(p.stem + "_shape.npy")
    shape = np.load(str(shape_path))
    return torch.from_numpy(
        np.fromfile(str(p), dtype=np.float32).reshape(shape)
    ).float()


@torch.no_grad()
def get_mu_logvar(model, data, batch_size=512, device="cuda"):
    model.eval()
    mus, logvars = [], []
    for i in range(0, len(data), batch_size):
        x = data[i : i + batch_size].to(device)
        mu, logvar = model.encode(x)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    return torch.cat(mus), torch.cat(logvars)  # [N, D]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--val_data", default="/root/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin")
    p.add_argument("--out_dir", default="visualizations")
    p.add_argument("--kl_threshold", type=float, default=0.01)
    p.add_argument("--n_images", type=int, default=500, help="number of images for heatmap")
    p.add_argument("--top_dims", type=int, default=5, help="number of top dims for histogram")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"],
        latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"),
        num_layers=a["num_layers"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # load data
    val_data = load_embeddings(args.val_data)
    print(f"val data: {val_data.shape}")

    mu, logvar = get_mu_logvar(model, val_data, device=device)
    print(f"mu: {mu.shape}, logvar: {logvar.shape}")

    # per-dim mean KL
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=0)  # [D]
    kl_sorted, sort_idx = kl_dim.sort(descending=True)
    active_mask = kl_dim > args.kl_threshold
    n_active = active_mask.sum().item()
    print(f"active dims (KL>{args.kl_threshold}): {n_active}")

    # ── 1. Per-dim mean KL S-curve ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    x_axis = np.arange(len(kl_sorted))
    ax.plot(x_axis, kl_sorted.numpy(), linewidth=1.2, color="steelblue")
    ax.axhline(args.kl_threshold, color="red", linestyle="--", linewidth=1,
               label=f"threshold={args.kl_threshold}")
    ax.axvline(n_active, color="orange", linestyle="--", linewidth=1,
               label=f"active={n_active}")
    ax.set_xlabel("Dimension (sorted by KL descending)")
    ax.set_ylabel("Mean KL")
    ax.set_title("Per-dim Mean KL Distribution")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = out_dir / "kl_dim_sorted.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"saved: {path1}")

    # ── 2. mu heatmap (images x active dims) ───────────────────────────────
    active_dim_idx = sort_idx[:n_active]
    n_img = min(args.n_images, len(mu))
    # sort images by mu of the highest-KL dim
    primary_dim = active_dim_idx[0]
    img_order = mu[:, primary_dim].argsort()
    img_idx = img_order[:n_img]

    mu_sub = mu[img_idx][:, active_dim_idx].numpy()  # [n_img, n_active]

    fig, ax = plt.subplots(figsize=(min(n_active * 0.15 + 2, 16), 7))
    im = ax.imshow(mu_sub, aspect="auto", cmap="RdBu_r",
                   vmin=-3, vmax=3, interpolation="nearest")
    ax.set_xlabel(f"Active dims ({n_active}, KL descending)")
    ax.set_ylabel(f"Images ({n_img}, sorted by dim0 mu)")
    ax.set_title("mu Heatmap (Images x Active Dims)")
    plt.colorbar(im, ax=ax, label="mu value")
    fig.tight_layout()
    path2 = out_dir / "mu_heatmap.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"saved: {path2}")

    # ── 3. mu histogram per top active dim ─────────────────────────────────
    top_k = min(args.top_dims, n_active)
    fig, axes = plt.subplots(1, top_k, figsize=(4 * top_k, 4), sharey=False)
    if top_k == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        dim_i = active_dim_idx[i].item()
        vals = mu[:, dim_i].numpy()
        kl_val = kl_dim[dim_i].item()
        ax.hist(vals, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"dim {dim_i}\nKL={kl_val:.3f}")
        ax.set_xlabel("mu value")
        if i == 0:
            ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"mu distribution of top-{top_k} active dims ({len(mu)} images)", y=1.02)
    fig.tight_layout()
    path3 = out_dir / "mu_histograms.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path3}")


if __name__ == "__main__":
    main()