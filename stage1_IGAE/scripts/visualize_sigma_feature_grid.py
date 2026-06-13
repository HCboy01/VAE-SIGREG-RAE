#!/usr/bin/env python3
"""
σ < threshold 기준으로 활성화된 이미지들을 μ로 정렬해서 그리드 시각화.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/visualize_sigma_feature_grid.py \
        --ckpt checkpoints/s1_grid_b1e-4_l100/best.pt \
        --feature_dims 0 1 2 3 4 5 6 7 8 9 10 \
        --sigma_thr 0.2 \
        --n_per_side 8
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

IMG_DIR  = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
EMB_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
VAL_IDX  = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP      = torch.bfloat16
THUMB    = 96


def build_train_paths():
    val_ids = set(Path(VAL_IDX).read_text().splitlines())
    return [p for p in sorted(IMG_DIR.iterdir())
            if p.suffix == ".png" and p.stem not in val_ids]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    m = OvercompleteVariationalAE(
        int(a.get("input_dim", 768)), int(a.get("latent_dim", 6144)),
        hidden_dim=a.get("hidden_dim"), num_layers=int(a.get("num_layers", 4)),
    ).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval().requires_grad_(False)
    return m


@torch.no_grad()
def encode_all(model, emb, batch=1024):
    mus, sigmas = [], []
    for i in range(0, len(emb), batch):
        x = emb[i:i+batch].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=AMP, enabled=(DEVICE.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu.float().cpu())
        sigmas.append((0.5 * lv).exp().float().cpu())
    return torch.cat(mus), torch.cat(sigmas)  # [N, D]


def thumb(path, size=THUMB):
    try:
        return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    except Exception:
        return Image.new("RGB", (size, size), (180, 180, 180))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--feature_dims", type=int, nargs="+", default=list(range(11)))
    p.add_argument("--sigma_thr",    type=float, default=0.2)
    p.add_argument("--n_per_side",   type=int, default=8)
    p.add_argument("--out_dir",      type=str,
                   default="/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/sigma_feature_grid")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = args.n_per_side
    T = THUMB

    print("Loading paths & embeddings...", flush=True)
    train_paths = build_train_paths()
    raw = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
    emb = (raw if isinstance(raw, torch.Tensor) else next(iter(raw.values()))).float()
    n = min(len(train_paths), len(emb))
    train_paths, emb = train_paths[:n], emb[:n]

    print(f"Encoding {n} images...", flush=True)
    model = load_model(args.ckpt)
    mu_all, sigma_all = encode_all(model, emb)   # [N, 6144]
    print(f"  done. mu: {mu_all.shape}", flush=True)

    for dim in args.feature_dims:
        mu_d    = mu_all[:, dim].numpy()      # [N]
        sigma_d = sigma_all[:, dim].numpy()   # [N]

        # σ < threshold인 이미지만 선택
        active_idx = np.where(sigma_d < args.sigma_thr)[0]
        n_active = len(active_idx)

        if n_active == 0:
            print(f"  dim {dim}: no images with σ<{args.sigma_thr}, skipping", flush=True)
            continue

        # μ 기준 정렬
        sorted_by_mu = active_idx[np.argsort(mu_d[active_idx])[::-1]]  # 높은 μ → 낮은 μ

        pos_idx = sorted_by_mu[:N]          # μ 큰 쪽
        neg_idx = sorted_by_mu[-N:][::-1]   # μ 작은 쪽 (반전해서 가장 낮은게 마지막)

        pos_mu = mu_d[pos_idx];  pos_sigma = sigma_d[pos_idx]
        neg_mu = mu_d[neg_idx];  neg_sigma = sigma_d[neg_idx]

        print(
            f"  dim {dim:3d}: σ<{args.sigma_thr} → {n_active} imgs | "
            f"μ range [{mu_d[active_idx].min():.2f}, {mu_d[active_idx].max():.2f}]",
            flush=True,
        )

        # 그리드 그리기
        fig, axes = plt.subplots(
            3, N,
            figsize=(N * T / 72, 3 * T / 72 + 1.5),
            gridspec_kw={"height_ratios": [T, 0.25, T], "hspace": 0.05, "wspace": 0.04},
        )
        ckpt_tag = Path(args.ckpt).parents[0].name
        fig.suptitle(
            f"dim {dim}  |  σ<{args.sigma_thr}: {n_active} images  |  {ckpt_tag}",
            fontsize=10,
        )

        for col in range(N):
            # + 방향 (높은 μ)
            ax_pos = axes[0, col]
            if col < len(pos_idx):
                ax_pos.imshow(thumb(train_paths[pos_idx[col]]))
                ax_pos.set_title(
                    f"μ={pos_mu[col]:+.2f}\nσ={pos_sigma[col]:.3f}",
                    fontsize=6, pad=2, color="#c0392b",
                )
            ax_pos.axis("off")

            axes[1, col].axis("off")

            # - 방향 (낮은 μ)
            ax_neg = axes[2, col]
            if col < len(neg_idx):
                ax_neg.imshow(thumb(train_paths[neg_idx[col]]))
                ax_neg.set_title(
                    f"μ={neg_mu[col]:+.2f}\nσ={neg_sigma[col]:.3f}",
                    fontsize=6, pad=2, color="#2980b9",
                )
            ax_neg.axis("off")

        # 방향 레이블
        fig.text(0.01, 0.80, "(+)", fontsize=10, fontweight="bold",
                 color="#c0392b", va="center")
        fig.text(0.01, 0.22, "(−)", fontsize=10, fontweight="bold",
                 color="#2980b9", va="center")

        save_path = out_dir / f"dim{dim:04d}_sigma{args.sigma_thr}.png"
        plt.savefig(save_path, dpi=96, bbox_inches="tight")
        plt.close()

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()
