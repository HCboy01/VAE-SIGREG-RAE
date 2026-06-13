"""
For one query image, select top latent dims by per-image KL and sort those dims
by the query's posterior mean mu.

Usage:
    python scripts/query_kl_top_dims_grid.py RUN --img_idx 2685 --top_dims 10

Output:
    visualizations/query_kl_top_dims/<run>_img<idx>_top<k>kl_dims_by_mu.png
    visualizations/query_kl_top_dims/<run>_img<idx>_top<k>kl_dims_by_mu.csv
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
OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/query_kl_top_dims")
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--img_idx", type=int, default=2685)
    p.add_argument("--top_dims", type=int, default=10)
    p.add_argument("--thumb", type=int, default=192)
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = build_train_image_paths(IMG_DIR, VAL_IDX)
    emb = load_embeddings(EMB_PATH)
    n_total = min(len(paths), len(emb))
    paths, emb = paths[:n_total], emb[:n_total]
    if not 0 <= args.img_idx < n_total:
        raise IndexError(f"--img_idx {args.img_idx} outside dataset size {n_total}")

    model, ckpt_args = load_model(CKPT_ROOT / args.run / "best.pt", device)
    x = emb[args.img_idx:args.img_idx + 1].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
        mu, logvar = model.encode(x)
    mu = mu.float().cpu()[0]
    logvar = logvar.float().cpu()[0]
    sigma = (0.5 * logvar).exp()
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

    top_dims = torch.topk(kl, args.top_dims).indices.tolist()
    # This is the requested ordering: among KL-top dims, sort by query mu.
    sorted_dims = sorted(top_dims, key=lambda d: mu[d].item())
    rows = []
    for order, d in enumerate(sorted_dims, start=1):
        rows.append({
            "mu_sorted_order": order,
            "dim": d,
            "kl_rank_within_query": top_dims.index(d) + 1,
            "query_kl": kl[d].item(),
            "query_mu": mu[d].item(),
            "query_sigma": sigma[d].item(),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}_img{args.img_idx}_top{args.top_dims}kl_dims_by_mu"
    out_png = OUT_DIR / f"{stem}.png"
    out_csv = OUT_DIR / f"{stem}.csv"

    img = Image.open(paths[args.img_idx]).convert("RGB").resize((args.thumb, args.thumb), Image.BICUBIC)
    n_cols = 5
    n_rows = (args.top_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.1, n_rows * 3.4), squeeze=False)
    fig.suptitle(
        f"{args.run} | image {args.img_idx} ({paths[args.img_idx].stem})\n"
        f"Top-{args.top_dims} dims by query KL, sorted by query mu "
        f"(left-to-right, top-to-bottom)",
        fontsize=11,
    )

    max_kl = max(kl[d].item() for d in sorted_dims)
    min_mu = min(mu[d].item() for d in sorted_dims)
    max_mu = max(mu[d].item() for d in sorted_dims)
    for ax in axes.ravel():
        ax.axis("off")

    for i, d in enumerate(sorted_dims):
        ax = axes.ravel()[i]
        ax.imshow(img)
        qkl = kl[d].item()
        qmu = mu[d].item()
        qsig = sigma[d].item()
        color = plt.cm.plasma(qkl / max(max_kl, 1e-9))
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fill=False, edgecolor=color, linewidth=5))
        ax.set_title(
            f"mu order {i + 1} | KL rank {top_dims.index(d) + 1}\n"
            f"dim {d}  mu={qmu:+.3f}\n"
            f"KL={qkl:.3f}  sigma={qsig:.3f}",
            fontsize=9,
            pad=4,
        )
        # Small horizontal marker showing where this dim's mu sits among the selected top-K dims.
        x_pos = (qmu - min_mu) / max(max_mu - min_mu, 1e-9)
        ax.plot([0.08, 0.92], [-0.08, -0.08], transform=ax.transAxes, color="#999", lw=2, clip_on=False)
        ax.plot([0.08 + 0.84 * x_pos], [-0.08], transform=ax.transAxes,
                marker="o", color="#111", ms=6, clip_on=False)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"query image: {args.img_idx} {paths[args.img_idx].name}")
    print("KL-top dims sorted by query mu:")
    for row in rows:
        print(
            f"  order={row['mu_sorted_order']:2d} dim={row['dim']:5d} "
            f"kl_rank={row['kl_rank_within_query']:2d} "
            f"mu={row['query_mu']:+.4f} KL={row['query_kl']:.4f} "
            f"sigma={row['query_sigma']:.4f}"
        )
    print(f"\nSaved -> {out_png}")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
