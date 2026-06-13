"""
For one query image, select top latent dims by per-image KL, then visualize
each selected dim as a row of images sorted by posterior mean mu.

Usage:
    python scripts/query_kl_top_mu_sorted_grid.py RUN --img_idx 2685 --top_dims 10

Output:
    visualizations/mu_sorted_feature_grid/<run>_img<idx>_top<k>kl_mu_sorted.png
    visualizations/mu_sorted_feature_grid/<run>_img<idx>_top<k>kl_mu_sorted.csv
"""

import argparse
import csv
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
EMB_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
IMG_DIR = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
VAL_IDX = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sorted_feature_grid")

BATCH_SIZE = 512
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--img_idx", type=int, default=2685)
    p.add_argument("--top_dims", type=int, default=10)
    p.add_argument("--n", type=int, default=10000,
                   help="number of train images to sort by mu; query image is always included")
    p.add_argument("--cols", type=int, default=10,
                   help="number of mu-percentile images per selected dim")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thumb", type=int, default=128)
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
def encode(model, emb, device):
    mus, lvs = [], []
    for i in range(0, len(emb), BATCH_SIZE):
        x = emb[i:i + BATCH_SIZE].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu.float().cpu())
        lvs.append(lv.float().cpu())
    return torch.cat(mus), torch.cat(lvs)


def thumb(path, size):
    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    paths = build_train_image_paths(IMG_DIR, VAL_IDX)
    emb = load_embeddings(EMB_PATH)
    n_total = min(len(paths), len(emb))
    paths, emb = paths[:n_total], emb[:n_total]
    if not 0 <= args.img_idx < n_total:
        raise IndexError(f"--img_idx {args.img_idx} outside dataset size {n_total}")
    print(f"aligned dataset: {n_total} (paths {len(paths)}, emb {len(emb)})")

    model, ckpt_args = load_model(CKPT_ROOT / args.run / "best.pt", device)
    beta = ckpt_args.get("beta_kl", "?")
    lam = ckpt_args.get("lambda_sigreg", "?")
    sigdev = ckpt_args.get("lambda_sigma_dev", 0.0)
    target = ckpt_args.get("sigma_dev_target", 1.0)

    # Select dims from the actual query image.
    q_emb = emb[args.img_idx:args.img_idx + 1]
    q_mu, q_lv = encode(model, q_emb, device)
    q_sigma = (0.5 * q_lv).exp()
    q_kl = 0.5 * (q_mu.pow(2) + q_lv.exp() - q_lv - 1)
    top = torch.topk(q_kl[0], args.top_dims)
    dims = top.indices.tolist()
    print(f"query image {args.img_idx}: {paths[args.img_idx].name}")
    print("top dims by query per-image KL:")
    for rank, d in enumerate(dims, start=1):
        print(
            f"  {rank:2d}. dim={d:5d}  "
            f"KL={q_kl[0, d].item():.4f}  "
            f"mu={q_mu[0, d].item():+.4f}  "
            f"sigma={q_sigma[0, d].item():.4f}"
        )

    # Sort a sampled subset by mu for each selected dim. Always include query.
    torch.manual_seed(args.seed)
    n_sample = min(args.n, n_total)
    subset = torch.randperm(n_total)[:n_sample]
    if not (subset == args.img_idx).any():
        subset[-1] = args.img_idx
    subset = torch.unique(subset, sorted=False)
    sub_paths = [paths[i] for i in subset.tolist()]
    sub_emb = emb[subset]
    print(f"sorting subset: {len(subset)} images (seed={args.seed}, query included)")

    mu, logvar = encode(model, sub_emb, device)
    sigma = (0.5 * logvar).exp()
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_rows = len(dims)
    n_cols = args.cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * (args.thumb + 8) / 100, n_rows * (args.thumb + 48) / 100),
        squeeze=False,
    )
    fig.suptitle(
        f"{args.run} | query image {args.img_idx} ({paths[args.img_idx].stem})\n"
        f"Top-{args.top_dims} query dims by per-image KL; each row sorted by mu "
        f"(low -> high). beta={beta} lambda={lam} sigma_dev={sigdev}@{target}",
        fontsize=11,
    )

    rows_for_csv = []
    col_pcts = np.linspace(0, 100, n_cols)
    positions = np.linspace(0, len(subset) - 1, n_cols).round().astype(int)

    for r, d in enumerate(dims):
        mu_d = mu[:, d]
        order = mu_d.argsort()
        chosen = order[positions]
        for c, si_t in enumerate(chosen):
            si = int(si_t.item())
            dataset_idx = int(subset[si].item())
            ax = axes[r, c]
            ax.axis("off")
            ax.imshow(thumb(sub_paths[si], args.thumb))
            ax.set_title(
                f"mu={mu_d[si].item():+.2f}\n"
                f"KL={kl[si, d].item():.2f}",
                fontsize=7,
                pad=2,
            )
            if r == 0:
                ax.text(
                    0.5, 1.30, f"~{col_pcts[c]:.0f}%",
                    transform=ax.transAxes, fontsize=7, ha="center", color="#666"
                )
            if c == 0:
                ax.text(
                    -0.22, 0.5,
                    f"rank {r + 1}\ndim {d}\n"
                    f"qKL={q_kl[0, d].item():.2f}\n"
                    f"qmu={q_mu[0, d].item():+.2f}\n"
                    f"qsig={q_sigma[0, d].item():.2f}",
                    transform=ax.transAxes, fontsize=8, va="center", ha="right",
                )
            rows_for_csv.append({
                "feature_rank": r + 1,
                "dim": d,
                "col": c,
                "percentile": col_pcts[c],
                "dataset_idx": dataset_idx,
                "image_path": str(sub_paths[si]),
                "mu": mu_d[si].item(),
                "sigma": sigma[si, d].item(),
                "kl": kl[si, d].item(),
                "query_kl": q_kl[0, d].item(),
                "query_mu": q_mu[0, d].item(),
                "query_sigma": q_sigma[0, d].item(),
            })

    fig.tight_layout(rect=(0.055, 0, 1, 0.94))
    stem = f"{args.run}_img{args.img_idx}_top{args.top_dims}kl_mu_sorted"
    out_png = OUT_DIR / f"{stem}.png"
    out_csv = OUT_DIR / f"{stem}.csv"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"\nSaved -> {out_png}")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
