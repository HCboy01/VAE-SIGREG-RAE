"""
For one latent feature, select top-K images by per-image KL for that feature,
then sort those selected images by posterior mean mu for the same feature.

Usage:
    python scripts/feature_topkl_sorted_by_mu.py RUN --dim 4560 --topk 10

Output:
    visualizations/feature_topkl_mu_sorted/<run>_dim<d>_top<k>kl_by_mu.png
    visualizations/feature_topkl_mu_sorted/<run>_dim<d>_top<k>kl_by_mu.csv
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
OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/feature_topkl_mu_sorted")

BATCH_SIZE = 512
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n", type=int, default=63001,
                   help="number of aligned train images to search")
    p.add_argument("--thumb", type=int, default=180)
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

    model, ckpt_args = load_model(CKPT_ROOT / args.run / "best.pt", device)
    mu_d, logvar_d = encode_dim(model, emb, args.dim, device)
    sigma_d = (0.5 * logvar_d).exp()
    kl_d = 0.5 * (mu_d.pow(2) + logvar_d.exp() - logvar_d - 1)

    top_idx = torch.topk(kl_d, min(args.topk, len(kl_d))).indices
    sorted_idx = top_idx[mu_d[top_idx].argsort()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}_dim{args.dim}_top{args.topk}kl_by_mu"
    out_png = OUT_DIR / f"{stem}.png"
    out_csv = OUT_DIR / f"{stem}.csv"

    fig, axes = plt.subplots(1, len(sorted_idx), figsize=(len(sorted_idx) * 2.15, 2.85), squeeze=False)
    fig.suptitle(
        f"{args.run} | dim {args.dim}\n"
        f"top-{len(sorted_idx)} images by per-image KL for this feature, sorted by mu low -> high",
        fontsize=10,
    )

    rows = []
    for col, idx_t in enumerate(sorted_idx):
        idx = int(idx_t.item())
        img = Image.open(paths[idx]).convert("RGB").resize((args.thumb, args.thumb), Image.BICUBIC)
        ax = axes[0, col]
        ax.axis("off")
        ax.imshow(img)
        ax.set_title(
            f"mu={mu_d[idx].item():+.2f}\n"
            f"KL={kl_d[idx].item():.2f}\n"
            f"{paths[idx].stem}",
            fontsize=7,
            pad=2,
        )
        rows.append({
            "mu_sorted_order": col + 1,
            "dataset_idx": idx,
            "image_path": str(paths[idx]),
            "mu": mu_d[idx].item(),
            "sigma": sigma_d[idx].item(),
            "kl": kl_d[idx].item(),
        })

    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"dim={args.dim}  searched={n_total}  topk={len(sorted_idx)}")
    for row in rows:
        print(
            f"  order={row['mu_sorted_order']:2d} idx={row['dataset_idx']:5d} "
            f"mu={row['mu']:+.4f} sigma={row['sigma']:.4f} KL={row['kl']:.4f} "
            f"{Path(row['image_path']).name}"
        )
    print(f"\nSaved -> {out_png}")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
