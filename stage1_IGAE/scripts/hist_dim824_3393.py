"""
Histogram of latent values for dim 824 (min KL) vs dim 3393 (max KL),
over 3000 encoded images, compared against the standard normal prior N(0,1).

For each dim we plot, with density normalization and a shared x-axis:
  - z  = mu + sigma * eps   (one reparam draw per image; what the decoder sees)
  - mu = posterior mean
  - N(0,1) prior curve (dashed)

Usage:
    python scripts/hist_dim824_3393.py [RUN_NAME]
Output:
    visualizations/mu_sigma_combined_lindec/<run>_hist_dim824_3393.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sigma_combined_lindec")

DEFAULT_DIMS = [824, 3393]
COLORS = ["#4da6e8", "#c94545"]
N_IMAGES = 3000
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="s1_grid_b1e-4_l100_lindec")
    p.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS,
                   help="latent dims to histogram (default 824 3393)")
    return p.parse_args()


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    is_linear = "decoder.weight" in ckpt["model"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"], latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"), num_layers=a["num_layers"],
        linear_decoder=is_linear,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, a


@torch.no_grad()
def encode_all(model, x, device, bs=512):
    mus, lvs = [], []
    for i in range(0, len(x), bs):
        mu, lv = model.encode(x[i:i+bs].to(device))
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, a = load_model(CKPT_ROOT / args.run / "best.pt")
    model.to(device)
    beta, lam = a.get("beta_kl", "?"), a.get("lambda_sigreg", "?")

    raw  = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
    torch.manual_seed(SEED)
    x = data[torch.randperm(len(data))[:N_IMAGES]]

    mu, logvar = encode_all(model, x, device)
    model.cpu()
    sigma = (0.5 * logvar).exp()
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)
    D = mu.shape[1]
    # KL rank of each dim: rank 1 = highest KL
    kl_rank = (kl.argsort(descending=True).argsort() + 1).tolist()

    torch.manual_seed(SEED)              # reproducible reparam draw
    eps = torch.randn_like(mu)
    z = mu + sigma * eps

    # dims to plot, each panel labeled with its KL rank
    DIMS = [(d, f"rank {kl_rank[d]}/{D}", COLORS[i % len(COLORS)])
            for i, d in enumerate(args.dims)]

    # ── figure: 1 row, N cols (one dim each), shared x ──────────────────────
    xs = np.linspace(-4, 4, 400)
    npdf = np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi)
    bins = np.linspace(-4, 4, 81)

    fig, axes = plt.subplots(1, len(DIMS), figsize=(7.5 * len(DIMS), 5.6),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle(
        f"{args.run}   β={beta}  λ={lam}   N={N_IMAGES}   |   "
        f"latent value histogram vs N(0,1) prior", fontsize=12)

    for ax, (dim, lbl, color) in zip(axes, DIMS):
        z_d  = z[:, dim].numpy()
        mu_d = mu[:, dim].numpy()
        sg_d = sigma[:, dim].numpy()

        ax.hist(z_d, bins=bins, density=True, color=color, alpha=0.55,
                label=f"z = μ+σε  (std={z_d.std():.3f})")
        ax.hist(mu_d, bins=bins, density=True, histtype="step", lw=1.8,
                color="black", alpha=0.8, label=f"μ  (std={mu_d.std():.3f})")
        ax.plot(xs, npdf, "k--", lw=1.6, alpha=0.7, label="N(0,1) prior")
        ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.6)

        ax.set_title(f"{lbl}   dim {dim}   KL={kl[dim].item():.4f}\n"
                     f"E[z]={z_d.mean():+.3f}   mean σ={sg_d.mean():.3f}", fontsize=10)
        ax.set_xlabel("value"); ax.set_xlim(-4, 4)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("density")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dim_tag = "_".join(str(d) for d in args.dims)
    out_png = OUT_DIR / f"{args.run}_hist_dim{dim_tag}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    for dim, lbl, _ in DIMS:
        print(f"{lbl} dim {dim}: KL={kl[dim].item():.4f}  "
              f"E[z]={z[:,dim].mean():+.4f}  std(z)={z[:,dim].std():.4f}  "
              f"std(μ)={mu[:,dim].std():.4f}  mean σ={sigma[:,dim].mean():.4f}")
    print(f"\nSaved → {out_png}")


if __name__ == "__main__":
    main()
