#!/usr/bin/env python3
"""
High-KL feature dim 시각화: + / - 방향 이미지 그리드.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/visualize_feature_grids.py \
        --ckpt checkpoints/s1_grid_b1e-3_l100_lindec/best.pt \
        --top_dims 10 \
        --n_images_per_side 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

IMG_DIR   = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
EMB_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
VAL_IDX   = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/feature_grids")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16


def build_train_image_paths():
    """val_indices.txt 제외한 순서대로 train 이미지 경로 목록."""
    val_ids = set(Path(VAL_IDX).read_text().splitlines())
    all_imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix == ".png")
    train_imgs = [p for p in all_imgs if p.stem not in val_ids]
    return train_imgs


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    model = OvercompleteVariationalAE(
        input_dim  = int(a.get("input_dim",  768)),
        latent_dim = int(a.get("latent_dim", 6144)),
        hidden_dim = a.get("hidden_dim", None),
        num_layers = int(a.get("num_layers", 4)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def encode_all(model, emb: torch.Tensor, batch_size=1024):
    mus, logvars = [], []
    for i in range(0, len(emb), batch_size):
        x = emb[i:i+batch_size].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
            mu, logvar = model.encode(x)
        mus.append(mu.float().cpu())
        logvars.append(logvar.float().cpu())
    return torch.cat(mus), torch.cat(logvars)


def make_grid_image(img_paths, title, n_cols=8, thumb_size=128):
    n = len(img_paths)
    n_rows = (n + n_cols - 1) // n_cols
    grid = Image.new("RGB", (n_cols * thumb_size, n_rows * thumb_size), (200, 200, 200))
    for i, p in enumerate(img_paths):
        row, col = divmod(i, n_cols)
        try:
            img = Image.open(p).convert("RGB").resize((thumb_size, thumb_size), Image.BICUBIC)
            grid.paste(img, (col * thumb_size, row * thumb_size))
        except Exception:
            pass
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",              required=True)
    p.add_argument("--feature_dim",       type=int, default=None, help="특정 dim 지정 (없으면 high-KL 자동 탐색)")
    p.add_argument("--top_dims",          type=int, default=10)
    p.add_argument("--n_images_per_side", type=int, default=8,  help="+ 쪽 / - 쪽 각각 몇 장")
    p.add_argument("--kl_threshold",      type=float, default=0.05)
    p.add_argument("--thumb_size",        type=int, default=128)
    p.add_argument("--out_dir",           type=str, default=None)
    args = p.parse_args()

    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("이미지 경로 목록 구성...", flush=True)
    train_paths = build_train_image_paths()
    print(f"  train images: {len(train_paths)}", flush=True)

    print("임베딩 로드...", flush=True)
    _emb_raw = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
    emb = (_emb_raw if isinstance(_emb_raw, torch.Tensor) else next(iter(_emb_raw.values()))).float()
    print(f"  embeddings: {emb.shape}", flush=True)

    n = min(len(train_paths), len(emb))
    train_paths = train_paths[:n]
    emb = emb[:n]

    print("VAE 인코딩...", flush=True)
    model = load_model(args.ckpt)
    mu, logvar = encode_all(model, emb)
    print(f"  mu: {mu.shape}", flush=True)

    # KL per dim
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)  # [D]

    if args.feature_dim is not None:
        # 특정 dim만
        dim_list = [(0, args.feature_dim, kl_dim[args.feature_dim].item())]
        print(f"\ndim {args.feature_dim}: KL={kl_dim[args.feature_dim]:.4f}  "
              f"mu_mean={mu[:, args.feature_dim].mean():+.4f}  "
              f"mu_std={mu[:, args.feature_dim].std():.4f}")
    else:
        top_kl_vals, top_kl_idx = kl_dim.topk(args.top_dims)
        dim_list = [(rank, idx, kl_v)
                    for rank, (idx, kl_v) in enumerate(zip(top_kl_idx.tolist(), top_kl_vals.tolist()))]
        print(f"\nTop {args.top_dims} high-KL dims:")
        print(f"{'rank':>4} {'dim':>6} {'KL':>8} {'mu_mean':>9} {'mu_std':>8}")
        print("-" * 42)
        for rank, dim_idx, kl_v in dim_list:
            mu_d = mu[:, dim_idx]
            print(f"{rank+1:>4} {dim_idx:>6} {kl_v:>8.4f} {mu_d.mean():>+9.4f} {mu_d.std():>8.4f}")

    # 각 dim에 대해 그리드 생성
    N = args.n_images_per_side

    for rank, dim_idx, kl_v in dim_list:
        mu_d = mu[:, dim_idx]  # [N_images]

        # + 방향: mu 큰 순
        pos_order = mu_d.argsort(descending=True)[:N].tolist()
        # - 방향: mu 작은 순
        neg_order = mu_d.argsort(descending=False)[:N].tolist()

        sigma_d = (0.5 * logvar[:, dim_idx]).exp()

        pos_paths  = [train_paths[i] for i in pos_order]
        neg_paths  = [train_paths[i] for i in neg_order]
        pos_vals   = [mu_d[i].item()    for i in pos_order]
        neg_vals   = [mu_d[i].item()    for i in neg_order]
        pos_sigmas = [sigma_d[i].item() for i in pos_order]
        neg_sigmas = [sigma_d[i].item() for i in neg_order]

        # 그리드 이미지 합치기: 위 = + 방향, 아래 = - 방향
        T = args.thumb_size
        total_h = T * 2 + 40  # 40px separator
        total_w = T * N
        canvas = Image.new("RGB", (total_w, total_h), (240, 240, 240))

        for col, (pp, pv, np_, nv) in enumerate(zip(pos_paths, pos_vals, neg_paths, neg_vals)):
            # + 이미지
            try:
                img = Image.open(pp).convert("RGB").resize((T, T), Image.BICUBIC)
                canvas.paste(img, (col * T, 0))
            except Exception:
                pass
            # - 이미지
            try:
                img = Image.open(np_).convert("RGB").resize((T, T), Image.BICUBIC)
                canvas.paste(img, (col * T, T + 40))
            except Exception:
                pass

        # matplotlib으로 그리드 + 라벨
        fig_w = N * T / 72
        fig_h = (total_h + 60) / 72 + 1.5  # 각 이미지 아래 텍스트 공간
        fig, axes = plt.subplots(
            3, N,
            figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [T, 0.3, T], "hspace": 0.05},
        )
        fig.suptitle(
            f"dim {dim_idx}  |  KL={kl_v:.4f}\n"
            f"TOP: + direction  |  BOTTOM: - direction",
            fontsize=10,
        )

        for col in range(N):
            # + 방향 이미지
            ax_pos = axes[0, col]
            try:
                img = Image.open(pos_paths[col]).convert("RGB").resize((T, T), Image.BICUBIC)
                ax_pos.imshow(img)
            except Exception:
                ax_pos.set_facecolor("#cccccc")
            ax_pos.set_title(f"μ={pos_vals[col]:+.3f}\nσ={pos_sigmas[col]:.3f}", fontsize=7, pad=2)
            ax_pos.axis("off")

            # 구분선
            axes[1, col].axis("off")

            # - 방향 이미지
            ax_neg = axes[2, col]
            try:
                img = Image.open(neg_paths[col]).convert("RGB").resize((T, T), Image.BICUBIC)
                ax_neg.imshow(img)
            except Exception:
                ax_neg.set_facecolor("#cccccc")
            ax_neg.set_title(f"μ={neg_vals[col]:+.3f}\nσ={neg_sigmas[col]:.3f}", fontsize=7, pad=2)
            ax_neg.axis("off")

        # + / - 레이블
        fig.text(0.01, 0.80, "(+)", fontsize=11, fontweight="bold", color="#e74c3c", va="center")
        fig.text(0.01, 0.22, "(−)", fontsize=11, fontweight="bold", color="#3498db", va="center")

        plt.tight_layout()
        save_path = OUT_DIR / f"rank{rank+1:02d}_dim{dim_idx}_kl{kl_v:.3f}.png"
        plt.savefig(save_path, dpi=96, bbox_inches="tight")
        plt.close()
        print(f"  saved: {save_path.name}", flush=True)

    print(f"\n완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
