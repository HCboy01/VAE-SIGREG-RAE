"""
Does a low-sigma feature actually carry information?

Encode N images, and for each selected latent dim, sort images by that dim's
posterior mean (mu). Lay out a grid where:
  - each ROW    = one selected feature (dim), labeled with sigma[img0] & KL
  - each COLUMN = an evenly-spaced mu percentile bin (low mu -> high mu)
  - each cell   = the representative image at that mu percentile

If a feature carries information, walking left->right (low->high mu) should
show a coherent visual gradient. A near-prior (high-sigma, low-KL) feature
should instead look like unordered random faces.

Default features are the four selected by image 0's sigma in
visualize_sigma_selected_features.py: ~0.8, ~0.5, ~0.2, and min.

Output: visualizations/mu_sorted_feature_grid/<run_name>_mu_sorted_grid.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

# ── Config ──────────────────────────────────────────────────────────────────
RUN_NAME  = "s1_grid_b1e-4_l100_lindec"
CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
EMB_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
IMG_DIR   = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
VAL_IDX   = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sorted_feature_grid")

N_IMAGES   = 3000
IMG0_IDX   = 0
SEED       = 42
N_COLS     = 10           # number of mu-percentile bins (columns)
THUMB      = 128
BATCH_SIZE = 512
SIGMA_TARGETS = [0.8, 0.5, 0.2]   # + the min-sigma dim
AMP_DTYPE = torch.bfloat16
# ────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def build_train_image_paths(img_dir, val_idx):
    val_ids = set(val_idx.read_text().splitlines()) if val_idx.exists() else set()
    all_imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return [p for p in all_imgs if p.stem not in val_ids]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    state = ckpt["model"]
    linear_decoder = bool(a.get("linear_decoder", "decoder.weight" in state))
    model = OvercompleteVariationalAE(
        input_dim=int(a.get("input_dim", 768)),
        latent_dim=int(a.get("latent_dim", 6144)),
        hidden_dim=a.get("hidden_dim", None),
        num_layers=int(a.get("num_layers", 4)),
        linear_decoder=linear_decoder,
    ).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model, a


@torch.no_grad()
def encode(model, emb):
    mus, lvs = [], []
    for i in range(0, len(emb), BATCH_SIZE):
        x = emb[i:i+BATCH_SIZE].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu.float().cpu()); lvs.append(lv.float().cpu())
    return torch.cat(mus), torch.cat(lvs)


# ── Load data (aligned: emb[i] <-> sorted train path[i]) ─────────────────────
paths = build_train_image_paths(IMG_DIR, VAL_IDX)
obj = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
emb = (obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))).float()
n_total = min(len(paths), len(emb))
paths, emb = paths[:n_total], emb[:n_total]
print(f"aligned dataset: {n_total} (paths {len(paths)}, emb {len(emb)})")

torch.manual_seed(SEED)
subset_idx   = torch.randperm(n_total)[:min(N_IMAGES, n_total)]
subset_paths = [paths[i] for i in subset_idx.tolist()]
subset_emb   = emb[subset_idx]
print(f"subset: {len(subset_idx)} images (seed={SEED})")

model, args = load_model(CKPT_ROOT / RUN_NAME / "best.pt")
mu, logvar = encode(model, subset_emb)
sigma  = (0.5 * logvar).exp()
kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)
beta, lam = args.get("beta_kl", "?"), args.get("lambda_sigreg", "?")

# ── Select features from image 0 ─────────────────────────────────────────────
sig0 = sigma[IMG0_IDX].numpy()
selected = []
for target in SIGMA_TARGETS:
    selected.append((int(np.abs(sig0 - target).argmin()), f"sigma~{target}"))
selected.append((int(sig0.argmin()), "sigma=min"))

print("\nSelected features:")
for dim, lbl in selected:
    print(f"  {lbl:>12s}  dim={dim:5d}  sigma[img0]={sig0[dim]:.4f}  KL_avg={kl_dim[dim]:.5f}")

# ── Build grid ───────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
n_rows = len(selected)
fig, axes = plt.subplots(
    n_rows, N_COLS,
    figsize=(N_COLS * (THUMB + 6) / 100, n_rows * (THUMB + 52) / 100),
    squeeze=False,
)
fig.suptitle(
    f"{RUN_NAME}  beta={beta} lambda={lam}  |  N={len(subset_idx)} images, target=mu\n"
    f"Each row = one feature sorted by its mu (low -> high across columns).  "
    f"Coherent visual gradient = feature carries information.",
    fontsize=12,
)

col_pcts = np.linspace(0, 100, N_COLS)
for r, (dim, lbl) in enumerate(selected):
    mu_d = mu[:, dim]
    order = mu_d.argsort()                       # ascending mu
    positions = np.linspace(0, len(order) - 1, N_COLS).round().astype(int)
    chosen = order[positions]
    for c in range(N_COLS):
        ax = axes[r, c]
        ax.axis("off")
        si = chosen[c].item()
        img = Image.open(subset_paths[si]).convert("RGB").resize((THUMB, THUMB), Image.BICUBIC)
        ax.imshow(img)
        ax.set_title(f"mu={mu_d[si].item():+.2f}", fontsize=7, pad=2)
        if c == 0:
            ax.text(-0.18, 0.5,
                    f"dim {dim}\n{lbl}\nsig[img0]={sig0[dim]:.3f}\nKL={kl_dim[dim]:.3f}",
                    transform=ax.transAxes, fontsize=8, va="center", ha="right",
                    rotation=0)
        if r == 0:
            ax.text(0.5, 1.28, f"~{col_pcts[c]:.0f}%", transform=ax.transAxes,
                    fontsize=7, ha="center", color="#777")

fig.tight_layout(rect=(0.04, 0, 1, 0.93))
out_png = OUT_DIR / f"{RUN_NAME}_mu_sorted_grid.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_png}")
