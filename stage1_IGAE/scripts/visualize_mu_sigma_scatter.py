"""
Per-image mu vs sigma scatter plot for 3 selected latent dimensions:
  - highest mean-KL dim
  - median mean-KL dim
  - lowest mean-KL dim

Each scatter point = one image (N=3000).
x-axis: mu value,  y-axis: sigma (= exp(0.5 * logvar)).
Reference lines at mu=0, sigma=1 mark the prior N(0,1).
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
CKPT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/s1_tierA_b7e-4_l10_w50/best.pt")
DATA = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
N_IMAGES = 3000
OUT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sigma_scatter.png")
# ───────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Load model
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ckpt["args"]
# Detect decoder type from saved state_dict keys
_is_linear_dec = "decoder.weight" in ckpt["model"]
model = OvercompleteVariationalAE(
    input_dim=a["input_dim"],
    latent_dim=a["latent_dim"],
    hidden_dim=a.get("hidden_dim"),
    num_layers=a["num_layers"],
    linear_decoder=_is_linear_dec,
)
print(f"Decoder type: {'linear' if _is_linear_dec else 'MLP'}")
model.load_state_dict(ckpt["model"])
model.to(device).eval()
print(f"Model loaded  (latent_dim={a['latent_dim']})")

# Load data
raw = torch.load(DATA, map_location="cpu", weights_only=False)
data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
print(f"Data shape: {data.shape}  →  sampling {N_IMAGES} images")

idx = torch.randperm(len(data))[:N_IMAGES]
x = data[idx].to(device)

# Encode
with torch.no_grad():
    mu, logvar = model.encode(x)
    mu = mu.cpu()
    logvar = logvar.cpu()

sigma = (0.5 * logvar).exp()   # [N, D]

# Per-dim mean KL
kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)   # [D]
print(f"kl_dim  min={kl_dim.min():.4f}  median={kl_dim.median():.4f}  max={kl_dim.max():.4f}")

# Pick 3 representative dimensions
kl_sorted_idx = kl_dim.argsort()
dim_low    = kl_sorted_idx[0].item()
dim_median = kl_sorted_idx[len(kl_sorted_idx) // 2].item()
dim_high   = kl_sorted_idx[-1].item()

dims   = [dim_high,   dim_median,    dim_low]
labels = ["Highest KL", "Median KL", "Lowest KL"]
colors = ["#e05c5c",  "#5c9fe0",     "#5cb87a"]

print(f"\nSelected dims:")
for d, lbl in zip(dims, labels):
    print(f"  {lbl}: dim={d:5d}  mean_KL={kl_dim[d]:.5f}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle(
    f"mu vs sigma per image  (N={N_IMAGES}, model=s1_tierA_b7e-4_l10_w50)\n"
    "Prior N(0,1): mu=0 (dashed), sigma=1 (dotted)",
    fontsize=12,
)

for ax, dim, lbl, col in zip(axes, dims, labels, colors):
    mu_d    = mu[:, dim].numpy()       # [N]
    sigma_d = sigma[:, dim].numpy()    # [N]
    kl_val  = kl_dim[dim].item()

    ax.scatter(mu_d, sigma_d, s=6, alpha=0.35, color=col, linewidths=0)

    # Prior reference
    ax.axvline(0,  color="black", lw=1.0, ls="--", alpha=0.6, label="prior μ=0")
    ax.axhline(1,  color="black", lw=1.0, ls=":",  alpha=0.6, label="prior σ=1")

    # Stats annotation
    ax.set_title(f"{lbl}\ndim {dim}  |  mean KL = {kl_val:.5f}", fontsize=10)
    ax.set_xlabel("μ (posterior mean)", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("σ (posterior std)", fontsize=9)

    # Inset text: μ and σ distribution summary
    mu_mean, mu_std = mu_d.mean(), mu_d.std()
    s_mean, s_std   = sigma_d.mean(), sigma_d.std()
    textstr = (
        f"μ: mean={mu_mean:+.3f}, std={mu_std:.3f}\n"
        f"σ: mean={s_mean:.3f}, std={s_std:.4f}"
    )
    ax.text(
        0.03, 0.97, textstr,
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT}")
