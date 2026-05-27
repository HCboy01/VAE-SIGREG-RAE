"""
mu vs sigma scatter plots for ALL linear-decoder checkpoints.

5 dimensions selected by KL quantile:
  0% (min), 25%, 50% (median), 75%, 100% (max)

Output: visualizations/mu_sigma_scatter_lindec/<run_name>.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE

# ── Config ─────────────────────────────────────────────────────────────────
CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR    = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sigma_scatter_lindec")
N_IMAGES   = 3000
QUANTILES  = [0.0, 0.25, 0.50, 0.75, 1.0]
Q_LABELS   = ["Min KL (0%)", "25%", "Median (50%)", "75%", "Max KL (100%)"]
COLORS     = ["#4da6e8", "#6bbf72", "#e8c44d", "#e87c4d", "#c94545"]
# ───────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Load data once
raw  = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
print(f"Data: {data.shape}")

torch.manual_seed(42)
idx = torch.randperm(len(data))[:N_IMAGES]
x_sample = data[idx]
print(f"Sampled {N_IMAGES} images (fixed seed)\n")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect all lindec checkpoint dirs
lindec_dirs = sorted(d for d in CKPT_ROOT.iterdir()
                     if d.is_dir() and "lindec" in d.name and (d / "best.pt").exists())
print(f"Found {len(lindec_dirs)} lindec checkpoints\n")


def load_model(ckpt_path: Path) -> OvercompleteVariationalAE:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    is_linear = "decoder.weight" in ckpt["model"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"],
        latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"),
        num_layers=a["num_layers"],
        linear_decoder=is_linear,
    )
    model.load_state_dict(ckpt["model"])
    return model, ckpt.get("args", {})


@torch.no_grad()
def encode(model, x, batch_size=512):
    model.eval()
    mus, lvs = [], []
    for i in range(0, len(x), batch_size):
        xb = x[i:i+batch_size].to(device)
        mu, lv = model.encode(xb)
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def make_scatter(run_name, mu, logvar, beta, lam):
    sigma  = (0.5 * logvar).exp()                           # [N, D]
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)  # [D]

    kl_sorted = kl_dim.sort().values
    D = len(kl_dim)

    # Pick 5 dims by quantile
    q_indices = [int(q * (D - 1)) for q in QUANTILES]
    dims = [kl_dim.argsort()[i].item() for i in q_indices]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    fig.suptitle(
        f"{run_name}   β={beta}  λ={lam}\n"
        f"kl range: [{kl_dim.min():.5f}, {kl_dim.max():.5f}]   "
        f"N={N_IMAGES} images  |  Prior: μ=0 (dashed), σ=1 (dotted)",
        fontsize=10,
    )

    for ax, dim, qlbl, col in zip(axes, dims, Q_LABELS, COLORS):
        mu_d    = mu[:, dim].numpy()
        sigma_d = sigma[:, dim].numpy()
        kl_val  = kl_dim[dim].item()

        ax.scatter(mu_d, sigma_d, s=5, alpha=0.3, color=col, linewidths=0)
        ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.55)
        ax.axhline(1, color="black", lw=0.9, ls=":",  alpha=0.55)

        ax.set_title(f"{qlbl}\ndim {dim}  KL={kl_val:.5f}", fontsize=8.5)
        ax.set_xlabel("μ", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("σ", fontsize=8)

        textstr = (
            f"μ: {mu_d.mean():+.3f} ± {mu_d.std():.3f}\n"
            f"σ: {sigma_d.mean():.4f} ± {sigma_d.std():.4f}"
        )
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=7,
                va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path = OUT_DIR / f"{run_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


for i, ckpt_dir in enumerate(lindec_dirs):
    run_name = ckpt_dir.name
    print(f"[{i+1:2d}/{len(lindec_dirs)}] {run_name} ...", end=" ", flush=True)

    try:
        model, args = load_model(ckpt_dir / "best.pt")
        model.to(device)
        mu, logvar = encode(model, x_sample)
        model.cpu()

        beta = args.get("beta_kl", "?")
        lam  = args.get("lambda_sigreg", "?")
        out  = make_scatter(run_name, mu, logvar, beta, lam)
        print(f"saved → {out.name}")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nDone. Output folder: {OUT_DIR}")
