"""
For one latent feature, show the most negative-mu and most positive-mu images.

Usage:
    python scripts/feature_mu_extreme_grid.py RUN --dim 4560 --topk 10

Output:
    visualizations/feature_mu_extremes/<run>_dim<d>_mu_extreme_top<k>.png
    visualizations/feature_mu_extremes/<run>_dim<d>_mu_extreme_top<k>.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
EMB_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
IMG_DIR = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
VAL_IDX = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/feature_mu_extremes")

BATCH_SIZE = 512
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n", type=int, default=63001)
    p.add_argument("--thumb", type=int, default=160)
    return p.parse_args()


def build_train_image_paths(img_dir, val_idx):
    val_ids = set(val_idx.read_text().splitlines()) if val_idx.exists() else set()
    all_imgs = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return [p for p in all_imgs if p.stem not in val_ids]


def load_embeddings(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return (obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))).float()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    state = ckpt["model"]
    linear_decoder = "decoder.weight" in state
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
def encode_dim(model, emb, dim, device):
    mus, lvs = [], []
    for i in range(0, len(emb), BATCH_SIZE):
        x = emb[i:i + BATCH_SIZE].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu[:, dim].float().cpu())
        lvs.append(lv[:, dim].float().cpu())
    return torch.cat(mus), torch.cat(lvs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = build_train_image_paths(IMG_DIR, VAL_IDX)
    emb = load_embeddings(EMB_PATH)
    n_total = min(len(paths), len(emb), args.n)
    paths, emb = paths[:n_total], emb[:n_total]

    model, _ = load_model(CKPT_ROOT / args.run / "best.pt", device)
    mu_d, logvar_d = encode_dim(model, emb, args.dim, device)
    sigma_d = (0.5 * logvar_d).exp()
    kl_d = 0.5 * (mu_d.pow(2) + logvar_d.exp() - logvar_d - 1)

    k = min(args.topk, len(mu_d))
    neg_idx = torch.topk(mu_d, k, largest=False).indices  # most negative first
    pos_idx = torch.topk(mu_d, k, largest=True).indices   # most positive first

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}_dim{args.dim}_mu_extreme_top{k}"
    out_png = OUT_DIR / f"{stem}.png"
    out_csv = OUT_DIR / f"{stem}.csv"

    fig, axes = plt.subplots(2, k, figsize=(k * 1.95, 4.8), squeeze=False)
    fig.suptitle(
        f"{args.run} | dim {args.dim}\n"
        f"top-{k} negative mu and top-{k} positive mu images",
        fontsize=11,
    )

    rows = []
    for row_i, (side, idxs) in enumerate([("negative_mu", neg_idx), ("positive_mu", pos_idx)]):
        for col, idx_t in enumerate(idxs):
            idx = int(idx_t.item())
            img = Image.open(paths[idx]).convert("RGB").resize((args.thumb, args.thumb), Image.BICUBIC)
            ax = axes[row_i, col]
            ax.axis("off")
            ax.imshow(img)
            ax.set_title(
                f"mu={mu_d[idx].item():+.2f}\n"
                f"KL={kl_d[idx].item():.2f}\n"
                f"{paths[idx].stem}",
                fontsize=7,
                pad=2,
            )
            if col == 0:
                ax.text(
                    -0.18, 0.5,
                    "most negative\nmu" if side == "negative_mu" else "most positive\nmu",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
            rows.append({
                "side": side,
                "rank": col + 1,
                "dataset_idx": idx,
                "image_path": str(paths[idx]),
                "mu": mu_d[idx].item(),
                "sigma": sigma_d[idx].item(),
                "kl": kl_d[idx].item(),
            })

    fig.tight_layout(rect=(0.035, 0, 1, 0.88))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"dim={args.dim} searched={n_total} topk={k}")
    print(f"negative mu range: {mu_d[neg_idx[-1]].item():+.4f} .. {mu_d[neg_idx[0]].item():+.4f}")
    print(f"positive mu range: {mu_d[pos_idx[-1]].item():+.4f} .. {mu_d[pos_idx[0]].item():+.4f}")
    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
