"""
Per-image latent scatter: each panel is one image, each point is one latent dim.

For selected dataset image indices, encode each image and plot:
  x = mu_d
  y = sigma_d
  color = per-dim KL_d for that image

Output:
  visualizations/per_image_mu_sigma/<run>_images_<idx...>.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg import OvercompleteVariationalAE

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/per_image_mu_sigma")
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--img_indices", type=int, nargs="+",
                   default=[0, 2685, 5000, 10000, 20000, 40000])
    p.add_argument("--cols", type=int, default=3)
    return p.parse_args()


def load_embeddings(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return (obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))).float()


def load_model(run, device):
    ckpt = torch.load(CKPT_ROOT / run / "best.pt", map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    state = ckpt["model"]
    linear_decoder = "decoder.weight" in state
    model = OvercompleteVariationalAE(
        input_dim=int(a.get("input_dim", 768)),
        latent_dim=int(a.get("latent_dim", 6144)),
        hidden_dim=a.get("hidden_dim"),
        num_layers=int(a.get("num_layers", 4)),
        linear_decoder=linear_decoder,
    ).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model, a


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_embeddings(DATA_PATH)
    indices = [i for i in args.img_indices if 0 <= i < len(data)]
    if not indices:
        raise ValueError("No valid image indices")

    model, ckpt_args = load_model(args.run, device)
    x = data[indices].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
        mu, logvar = model.encode(x)
    mu = mu.float().cpu()
    logvar = logvar.float().cpu()
    sigma = (0.5 * logvar).exp()
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

    n = len(indices)
    cols = max(1, min(args.cols, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 4.6), squeeze=False)
    fig.suptitle(
        f"{args.run}\nper-image latent scatter: x=mu, y=sigma, color=per-dim KL",
        fontsize=11,
    )

    positive_kl = kl[kl > 0]
    vmin = max(float(positive_kl.min().item()) if positive_kl.numel() else 1e-8, 1e-8)
    vmax = max(float(kl.max().item()), vmin * 10)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for ax in axes.ravel():
        ax.axis("off")

    for panel, img_idx in enumerate(indices):
        ax = axes.ravel()[panel]
        ax.axis("on")
        mu_i = mu[panel].numpy()
        sig_i = sigma[panel].numpy()
        kl_i = kl[panel].numpy()
        sc = ax.scatter(mu_i, sig_i, c=kl_i, cmap="plasma", norm=norm, s=6, alpha=0.50, linewidths=0)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.55)
        ax.axhline(1, color="black", lw=0.8, ls=":", alpha=0.55)
        for y, color in [(0.9, "#4CAF50"), (0.7, "#2196F3"), (0.5, "#F44336")]:
            ax.axhline(y, color=color, lw=0.75, ls="--", alpha=0.65)
        low09 = int((sigma[panel] < 0.9).sum().item())
        low07 = int((sigma[panel] < 0.7).sum().item())
        low05 = int((sigma[panel] < 0.5).sum().item())
        top_kl = torch.topk(kl[panel], k=min(5, kl.shape[1]))
        top_dims = ", ".join(str(int(d)) for d in top_kl.indices.tolist())
        ax.set_title(
            f"image idx {img_idx}\n"
            f"sigma mean={sig_i.mean():.3f} min={sig_i.min():.3f} "
            f"<.9/.7/.5={low09}/{low07}/{low05}\n"
            f"top KL dims: {top_dims}",
            fontsize=8,
        )
        ax.set_xlabel("mu")
        ax.set_ylabel("sigma")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    cbar.set_label("per-dim KL", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx_tag = "_".join(str(i) for i in indices)
    out = OUT_DIR / f"{args.run}_images_{idx_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
