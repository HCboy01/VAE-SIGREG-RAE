#!/usr/bin/env python3
"""Visualize images with smallest/largest activation for one IGAE feature."""

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


DEFAULT_IMG_DIR = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
DEFAULT_EMB_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
DEFAULT_VAL_IDX = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
DEFAULT_OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/feature_extremes")
AMP_DTYPE = torch.bfloat16


def build_train_image_paths(img_dir: Path, val_idx: Path) -> list[Path]:
    val_ids = set(val_idx.read_text().splitlines()) if val_idx.exists() else set()
    all_imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return [p for p in all_imgs if p.stem not in val_ids]


def load_embeddings(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    emb = obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))
    return emb.float()


def load_model(ckpt_path: Path, device: torch.device) -> OvercompleteVariationalAE:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    state = ckpt.get("model", ckpt)
    linear_decoder = bool(args.get("linear_decoder", "decoder.weight" in state))

    model = OvercompleteVariationalAE(
        input_dim=int(args.get("input_dim", 768)),
        latent_dim=int(args.get("latent_dim", 6144)),
        hidden_dim=args.get("hidden_dim", None),
        num_layers=int(args.get("num_layers", 4)),
        linear_decoder=linear_decoder,
    ).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def encode_feature(
    model: OvercompleteVariationalAE,
    emb: torch.Tensor,
    feature_dim: int,
    target: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for i in range(0, len(emb), batch_size):
        x = emb[i : i + batch_size].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            if target == "mu":
                mu, _ = model.encode(x)
                v = mu[:, feature_dim]
            else:
                out = model(x)
                v = out["z"][:, feature_dim]
        values.append(v.float().cpu())
    return torch.cat(values, dim=0)


def draw_grid(
    rows: list[tuple[str, list[Path], torch.Tensor]],
    out_png: Path,
    title: str,
    thumb_size: int,
) -> None:
    n_cols = max(len(paths) for _, paths, _ in rows)
    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(n_cols * thumb_size / 100, len(rows) * (thumb_size + 28) / 100),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=12)

    for r, (row_name, paths, vals) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(paths):
                continue
            img = Image.open(paths[c]).convert("RGB").resize((thumb_size, thumb_size), Image.BICUBIC)
            ax.imshow(img)
            ax.set_title(f"{row_name}\nz={vals[c].item():+.3f}", fontsize=8, pad=3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_selection_csv(
    out_csv: Path,
    labels: list[tuple[str, list[int], list[Path], torch.Tensor]],
) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "rank", "sample_index_within_subset", "image_path", "feature_value"])
        for side, indices, paths, vals in labels:
            for rank, (idx, path, val) in enumerate(zip(indices, paths, vals), start=1):
                writer.writerow([side, rank, idx, str(path), f"{val.item():.8g}"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--feature_dim", type=int, default=0)
    p.add_argument("--target", choices=["z", "mu"], default="z")
    p.add_argument("--n_images", type=int, default=3000)
    p.add_argument("--n_each", type=int, default=10)
    p.add_argument("--mode", choices=["extremes", "quintiles"], default="extremes")
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
    paths = paths[:n_total]
    emb = emb[:n_total]

    torch.manual_seed(args.seed)
    subset_idx = torch.randperm(n_total)[: min(args.n_images, n_total)]
    subset_paths = [paths[i] for i in subset_idx.tolist()]
    subset_emb = emb[subset_idx]

    model = load_model(args.ckpt, device)
    values = encode_feature(model, subset_emb, args.feature_dim, args.target, args.batch_size, device)

    stem = f"{run_name}_{args.target}_feature{args.feature_dim}_n{len(subset_idx)}_{args.mode}"
    out_png = out_dir / f"{stem}.png"
    out_csv = out_dir / f"{stem}.csv"

    title = (
        f"{run_name} | target={args.target} | feature {args.feature_dim} | "
        f"N={len(subset_idx)} images | seed={args.seed}"
    )

    if args.mode == "extremes":
        low_order = values.argsort(descending=False)[: args.n_each]
        high_order = values.argsort(descending=True)[: args.n_each]
        low_paths = [subset_paths[i] for i in low_order.tolist()]
        high_paths = [subset_paths[i] for i in high_order.tolist()]
        low_vals = values[low_order]
        high_vals = values[high_order]
        rows = [
            ("smallest", low_paths, low_vals),
            ("largest", high_paths, high_vals),
        ]
        labels = [
            ("smallest", low_order.tolist(), low_paths, low_vals),
            ("largest", high_order.tolist(), high_paths, high_vals),
        ]
    else:
        sorted_order = values.argsort(descending=False)
        chunks = torch.chunk(sorted_order, 5)
        rows = []
        labels = []
        for qi, chunk in enumerate(chunks, start=1):
            if len(chunk) <= args.n_each:
                chosen = chunk
            else:
                positions = torch.linspace(0, len(chunk) - 1, args.n_each).round().long()
                chosen = chunk[positions]
            chosen_paths = [subset_paths[i] for i in chosen.tolist()]
            chosen_vals = values[chosen]
            row_name = f"Q{qi}"
            rows.append((row_name, chosen_paths, chosen_vals))
            labels.append((row_name, chosen.tolist(), chosen_paths, chosen_vals))

    draw_grid(rows, out_png, title, args.thumb_size)
    write_selection_csv(out_csv, labels)

    print(f"device: {device}", flush=True)
    print(f"subset: {len(subset_idx)} / {n_total} images", flush=True)
    print(
        f"feature stats: min={values.min().item():+.4f} "
        f"mean={values.mean().item():+.4f} std={values.std(unbiased=False).item():.4f} "
        f"max={values.max().item():+.4f}",
        flush=True,
    )
    print(f"saved plot: {out_png}", flush=True)
    print(f"saved csv: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
