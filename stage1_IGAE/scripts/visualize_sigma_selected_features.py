"""
Sigma-selected feature mu-sigma scatter for a single lindec checkpoint.

Selection (based on image IMG0_IDX's per-dim sigma):
  - one feature with sigma closest to 0.8
  - one feature with sigma closest to 0.5
  - one feature with sigma closest to 0.2
  - the feature with the lowest sigma

For each selected feature, scatter mu vs sigma across N_IMAGES images
(each point = one image).

Output: visualizations/mu_sigma_combined_lindec/<run_name>_sigma_selected.png
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

# ── Config ──────────────────────────────────────────────────────────────────
RUN_NAME   = "s1_grid_b1e-4_l100_lindec"
CKPT_ROOT  = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR    = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sigma_combined_lindec")
N_IMAGES   = 3000
IMG0_IDX   = 0
SIGMA_TARGETS = [0.8, 0.5, 0.2]   # pick dim with sigma[img0] closest to each
# ────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

raw  = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
print(f"Data: {data.shape}")

# Same fixed-seed sampling as batch_mu_sigma_combined_lindec.py for consistency
torch.manual_seed(42)
idx      = torch.randperm(len(data))[:N_IMAGES]
x_sample = data[idx]
print(f"Sampled {N_IMAGES} images (fixed seed)")

OUT_DIR.mkdir(parents=True, exist_ok=True)


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


model, args = load_model(CKPT_ROOT / RUN_NAME / "best.pt")
model.to(device)
mu, logvar = encode(model, x_sample)
model.cpu()
beta = args.get("beta_kl", "?")
lam  = args.get("lambda_sigreg", "?")

sigma  = (0.5 * logvar).exp()                                    # [N, D]
kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)  # [D] avg over images
D = sigma.shape[1]

# ── Feature selection from image IMG0_IDX ────────────────────────────────────
sig0 = sigma[IMG0_IDX].numpy()  # [D]

selected = []  # (dim, label, color)
palette = ["#2196F3", "#4CAF50", "#F44336", "#7B1FA2"]
for target, color in zip(SIGMA_TARGETS, palette[:3]):
    dim = int(np.abs(sig0 - target).argmin())
    selected.append((dim, f"sigma~{target}", color))
dim_min = int(sig0.argmin())
selected.append((dim_min, "sigma=min", palette[3]))

print("\nSelected features (by image 0 sigma):")
for dim, lbl, _ in selected:
    print(f"  {lbl:>12s}  dim={dim:5d}  sigma[img0]={sig0[dim]:.4f}  KL_avg={kl_dim[dim]:.5f}")

# ── Figure: 4 panels, mu vs sigma across N images ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
fig.suptitle(
    f"{RUN_NAME}   beta={beta}  lambda={lam}   N={N_IMAGES}\n"
    f"Features selected by image {IMG0_IDX} sigma (~0.8 / ~0.5 / ~0.2 / min)  |  "
    f"prior: mu=0 (dashed), sigma=1 (dotted)",
    fontsize=11,
)

for ax, (dim, lbl, color) in zip(axes, selected):
    mu_d    = mu[:, dim].numpy()
    sigma_d = sigma[:, dim].numpy()
    kl_val  = kl_dim[dim].item()

    ax.scatter(mu_d, sigma_d, s=6, alpha=0.30, color=color, linewidths=0)
    # mark image 0's own point
    ax.scatter(mu_d[IMG0_IDX], sigma_d[IMG0_IDX], s=70, facecolors="none",
               edgecolors="black", linewidths=1.4, zorder=5,
               label=f"image {IMG0_IDX}")
    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.55)
    ax.axhline(1, color="black", lw=0.9, ls=":",  alpha=0.55)

    ax.set_title(f"{lbl}   dim {dim}\n"
                 f"sigma[img0]={sig0[dim]:.4f}   KL_avg={kl_val:.5f}", fontsize=9.5)
    ax.set_xlabel("mu (posterior mean)", fontsize=9)
    ax.set_ylabel("sigma (posterior std)", fontsize=9)
    textstr = (f"mu: {mu_d.mean():+.3f} +/- {mu_d.std():.3f}\n"
               f"sigma: {sigma_d.mean():.4f} +/- {sigma_d.std():.4f}\n"
               f"sigma range: [{sigma_d.min():.4f}, {sigma_d.max():.4f}]")
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=7.5,
            va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.90])
out_path = OUT_DIR / f"{RUN_NAME}_sigma_selected.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_path}")
