"""
Apply visualize_feature_extremes to the top-K most active dims of a model.

"Active" = highest variance of the chosen target (z or mu) across the data.

Encodes the full latent once, ranks dims by Var, then for each of the top-K
dims renders the same per-feature extremes grid + CSV as
visualize_feature_extremes.py, with identical file naming.

Usage:
    python scripts/feature_extremes_topk.py \\
        --ckpt checkpoints/<run>/best.pt --target z --topk 10 \\
        [--n_images 100000] [--n_each 10]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE
# Reuse helpers from the single-feature script.
from scripts.visualize_feature_extremes import (
    DEFAULT_IMG_DIR, DEFAULT_EMB_PATH, DEFAULT_VAL_IDX, DEFAULT_OUT_DIR, AMP_DTYPE,
    build_train_image_paths, load_embeddings, load_model, draw_grid, write_selection_csv,
)


@torch.no_grad()
def encode_all(model, emb, target, batch_size, device):
    """Return full [N, D] tensor of mu (target=mu) or z (target=z)."""
    out = []
    for i in range(0, len(emb), batch_size):
        x = emb[i : i + batch_size].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            if target == "mu":
                mu, _ = model.encode(x)
                out.append(mu.float().cpu())
            else:
                out.append(model(x)["z"].float().cpu())
    return torch.cat(out, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--target", choices=["z", "mu"], default="z")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n_images", type=int, default=100000)
    p.add_argument("--n_each", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--thumb_size", type=int, default=128)
    p.add_argument("--img_dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--emb_path", type=Path, default=DEFAULT_EMB_PATH)
    p.add_argument("--val_idx", type=Path, default=DEFAULT_VAL_IDX)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.ckpt.parent.name
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = build_train_image_paths(args.img_dir, args.val_idx)
    emb = load_embeddings(args.emb_path)
    n_total = min(len(paths), len(emb))
    paths, emb = paths[:n_total], emb[:n_total]

    torch.manual_seed(args.seed)
    sub_idx = torch.randperm(n_total)[: min(args.n_images, n_total)]
    sub_paths = [paths[i] for i in sub_idx.tolist()]
    sub_emb = emb[sub_idx]
    N = len(sub_idx)
    print(f"device: {device}  subset: {N} / {n_total}")

    model = load_model(args.ckpt, device)
    Z = encode_all(model, sub_emb, args.target, args.batch_size, device)   # [N, D]
    D = Z.shape[1]

    var = Z.var(dim=0, unbiased=False)
    top_dims = var.argsort(descending=True)[: args.topk].tolist()
    print(f"\nTop-{args.topk} dims by Var({args.target}) over {N} images:")
    print(f"{'rank':>4} | {'dim':>5} | {'var':>9} | {'std':>7} | "
          f"{'mean':>8} | {'min':>8} | {'max':>8}")
    for r, d in enumerate(top_dims, start=1):
        col = Z[:, d]
        print(f"{r:>4} | {d:>5} | {col.var(unbiased=False).item():>9.4f} | "
              f"{col.std(unbiased=False).item():>7.4f} | {col.mean().item():>+8.3f} | "
              f"{col.min().item():>+8.3f} | {col.max().item():>+8.3f}")

    # Render one extremes grid per top dim, naming compatible with visualize_feature_extremes
    for d in top_dims:
        values = Z[:, d]
        low_order = values.argsort(descending=False)[: args.n_each]
        high_order = values.argsort(descending=True)[: args.n_each]
        low_paths = [sub_paths[i] for i in low_order.tolist()]
        high_paths = [sub_paths[i] for i in high_order.tolist()]
        low_vals, high_vals = values[low_order], values[high_order]

        stem = f"{run_name}_{args.target}_feature{d}_n{N}_extremes"
        out_png = out_dir / f"{stem}.png"
        out_csv = out_dir / f"{stem}.csv"
        title = (f"{run_name} | target={args.target} | feature {d} | "
                 f"N={N} images | seed={args.seed}  "
                 f"(rank {top_dims.index(d)+1}/{args.topk} by Var)")
        draw_grid(
            [("smallest", low_paths, low_vals), ("largest", high_paths, high_vals)],
            out_png, title, args.thumb_size,
        )
        write_selection_csv(
            out_csv,
            [("smallest", low_order.tolist(), low_paths, low_vals),
             ("largest",  high_order.tolist(), high_paths, high_vals)],
        )
        print(f"  saved feature {d:5d} -> {out_png.name}")

    print(f"\nAll {args.topk} grids in: {out_dir}")


if __name__ == "__main__":
    main()
