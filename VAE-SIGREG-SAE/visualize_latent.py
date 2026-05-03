"""
For a given latent dimension, find images with most extreme z values
(farthest from N(0,1)) and visualize as a grid.

Top row:    8 images with highest z_d  (most positive)
Bottom row: 8 images with lowest  z_d  (most negative)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from src.vae_sigreg import OvercompleteVariationalAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"],
        latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"),
        num_layers=a["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["epoch"]


@torch.no_grad()
def encode_all(model, embeddings: torch.Tensor, device: torch.device,
               batch_size: int = 1024) -> torch.Tensor:
    """Return mu for all embeddings (deterministic encoding)."""
    parts = []
    for i in range(0, len(embeddings), batch_size):
        mu, _ = model.encode(embeddings[i:i + batch_size].to(device))
        parts.append(mu.cpu())
    return torch.cat(parts)   # [N, latent_dim]


def load_image(path: str, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def add_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 255))
    return out


def make_grid(images: list, nrow: int, padding: int = 4) -> Image.Image:
    w, h = images[0].size
    ncols = nrow
    nrows = (len(images) + ncols - 1) // ncols
    gw = ncols * w + (ncols + 1) * padding
    gh = nrows * h + (nrows + 1) * padding
    grid = Image.new("RGB", (gw, gh), (180, 180, 180))
    for idx, img in enumerate(images):
        r, c = divmod(idx, ncols)
        grid.paste(img, (padding + c * (w + padding), padding + r * (h + padding)))
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--embeddings",
                   default="/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin")
    p.add_argument("--paths",
                   default="/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_paths.txt")
    p.add_argument("--dim_idx", type=int, default=None,
                   help="Which latent dim to visualize. Auto-selects highest-std dim if omitted.")
    p.add_argument("--n_dims", type=int, default=1,
                   help="How many dims to visualize (stacked rows)")
    p.add_argument("--n_per_side", type=int, default=8,
                   help="Images per side (top-N + bottom-N = 2N images)")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--output_dir", default="visualizations")
    p.add_argument("--rank_by", choices=["std", "max_abs", "skew"], default="std",
                   help="How to auto-rank dims: std=most varying, max_abs=most extreme, skew=most asymmetric")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, epoch = load_model(args.ckpt, device)
    print(f"Model loaded  (epoch={epoch})")

    # Load embeddings
    shape = np.load(args.embeddings.replace(".bin", "_shape.npy"))
    embeddings = torch.from_numpy(
        np.fromfile(args.embeddings, dtype=np.float32).reshape(shape)
    )
    paths = [ln.strip() for ln in open(args.paths)]
    print(f"Embeddings: {embeddings.shape}  |  paths: {len(paths)}")

    # Encode → mu
    print("Encoding all embeddings...")
    mu = encode_all(model, embeddings, device)   # [N, D]
    print(f"mu: {mu.shape}")

    # Select dims to visualize
    if args.dim_idx is not None:
        dim_indices = [args.dim_idx]
    else:
        if args.rank_by == "std":
            scores = mu.std(dim=0)
        elif args.rank_by == "max_abs":
            scores = mu.abs().max(dim=0).values
        else:  # skew
            mu_c = mu - mu.mean(dim=0)
            std  = mu.std(dim=0).clamp(min=1e-6)
            scores = (mu_c.pow(3).mean(dim=0) / std.pow(3)).abs()
        dim_indices = scores.argsort(descending=True)[:args.n_dims].tolist()
        print(f"Auto-selected dims ({args.rank_by}): {dim_indices}")

    # Build image rows per dim
    Path(args.output_dir).mkdir(exist_ok=True)
    all_imgs = []

    for dim in dim_indices:
        z = mu[:, dim]
        top_idx = z.argsort(descending=True)[:args.n_per_side].tolist()
        bot_idx = z.argsort(descending=False)[:args.n_per_side].tolist()

        print(f"\ndim {dim:4d}  std={z.std():.3f}  "
              f"max={z.max():.2f}  min={z.min():.2f}")
        print(f"  top z: {[round(z[i].item(),2) for i in top_idx]}")
        print(f"  bot z: {[round(z[i].item(),2) for i in bot_idx]}")

        top_imgs = [add_label(load_image(paths[i], args.img_size),
                              f"+{z[i]:.2f}") for i in top_idx]
        bot_imgs = [add_label(load_image(paths[i], args.img_size),
                              f"{z[i]:.2f}") for i in bot_idx]
        all_imgs.extend(top_imgs + bot_imgs)

    grid = make_grid(all_imgs, nrow=args.n_per_side)

    dims_str = "_".join(str(d) for d in dim_indices)
    out_path = Path(args.output_dir) / f"dim{dims_str}_{args.rank_by}.png"
    grid.save(str(out_path))
    print(f"\nSaved → {out_path}  ({grid.width}×{grid.height})")


if __name__ == "__main__":
    main()
