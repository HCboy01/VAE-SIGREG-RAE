"""
Evenly-spaced (by mu percentile) image grid for a single latent dim.

Encode N images, sort by the chosen dim's posterior mean mu, then pick
rows*cols images at evenly spaced percentile positions (0..100%). Lay them
out in reading order (low mu -> high mu), so the grid shows the visual
gradient that the feature encodes.

Usage:
    python scripts/feature_grid_uniform.py [RUN] --dim 3393 [--rows 8 --cols 8] [--n 3000]
Output:
    visualizations/mu_sorted_feature_grid/<run>_uniform_dim<d>_<R>x<C>.png
"""

import argparse
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

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
EMB_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
IMG_DIR   = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
VAL_IDX   = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sorted_feature_grid")

SEED = 42
THUMB = 144
BATCH_SIZE = 512
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="s1_grid_b1e-4_l100_lindec")
    p.add_argument("--dim", type=int, default=3393)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--n", type=int, default=3000)
    return p.parse_args()


def build_train_image_paths(img_dir, val_idx):
    val_ids = set(val_idx.read_text().splitlines()) if val_idx.exists() else set()
    all_imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return [p for p in all_imgs if p.stem not in val_ids]


def load_model(ckpt_path, device):
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
def encode(model, emb, device):
    mus, lvs = [], []
    for i in range(0, len(emb), BATCH_SIZE):
        x = emb[i:i+BATCH_SIZE].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu.float().cpu()); lvs.append(lv.float().cpu())
    return torch.cat(mus), torch.cat(lvs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    paths = build_train_image_paths(IMG_DIR, VAL_IDX)
    obj = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    emb = (obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))).float()
    n_total = min(len(paths), len(emb))
    paths, emb = paths[:n_total], emb[:n_total]

    torch.manual_seed(SEED)
    sub = torch.randperm(n_total)[:min(args.n, n_total)]
    sub_paths = [paths[i] for i in sub.tolist()]
    sub_emb = emb[sub]

    model, a = load_model(CKPT_ROOT / args.run / "best.pt", device)
    mu, logvar = encode(model, sub_emb, device)
    var = logvar.exp()
    kl_dim = 0.5 * (mu.pow(2) + var - logvar - 1).mean(0)
    beta, lam = a.get("beta_kl", "?"), a.get("lambda_sigreg", "?")
    d = args.dim
    N = mu.shape[0]
    ncells = args.rows * args.cols

    mud = mu[:, d]
    order = mud.argsort()                                   # ascending mu
    pos = np.linspace(0, N - 1, ncells).round().astype(int) # evenly spaced percentiles
    pcts = pos / (N - 1) * 100
    chosen = order[pos]

    print(f"dim {d}: KL_avg={kl_dim[d].item():.4f}  "
          f"mu range [{mud.min():+.3f}, {mud.max():+.3f}]  "
          f"picking {ncells} evenly-spaced by mu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        args.rows, args.cols,
        figsize=(args.cols * (THUMB + 6) / 100, args.rows * (THUMB + 20) / 100),
        squeeze=False,
    )
    fig.suptitle(
        f"{args.run}  β={beta} λ={lam}  N={N}  |  dim {d}  KL_avg={kl_dim[d].item():.3f}\n"
        f"{ncells} images evenly spaced by μ percentile (low μ → high μ, reading order)",
        fontsize=12,
    )
    for k in range(ncells):
        r, c = divmod(k, args.cols)
        ax = axes[r, c]; ax.axis("off")
        si = chosen[k].item()
        img = Image.open(sub_paths[si]).convert("RGB").resize((THUMB, THUMB), Image.BICUBIC)
        ax.imshow(img)
        ax.set_title(f"{pcts[k]:.0f}%  μ={mud[si].item():+.2f}", fontsize=6.5, pad=1.5)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = OUT_DIR / f"{args.run}_uniform_dim{d}_{args.rows}x{args.cols}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
