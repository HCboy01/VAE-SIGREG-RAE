#!/usr/bin/env python3
"""
쿼리 이미지의 top-K 활성화 dim별 +/- 그리드.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/visualize_query_image.py \
        --ckpt checkpoints/s1_grid_b1e-3_l100_lindec/best.pt \
        --query /path/to/image.png \
        --top_k 5 \
        --n_per_side 6
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

IMG_DIR  = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
EMB_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
VAL_IDX  = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR  = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/query_viz")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP      = torch.bfloat16
T        = 112   # thumbnail size px


def build_train_paths():
    val_ids = set(Path(VAL_IDX).read_text().splitlines())
    return [p for p in sorted(IMG_DIR.iterdir())
            if p.suffix == ".png" and p.stem not in val_ids]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    m = OvercompleteVariationalAE(
        input_dim  = int(a.get("input_dim",  768)),
        latent_dim = int(a.get("latent_dim", 6144)),
        hidden_dim = a.get("hidden_dim", None),
        num_layers = int(a.get("num_layers", 4)),
    ).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval().requires_grad_(False)
    return m


def load_dino(path="facebook/dinov2-with-registers-base"):
    from transformers import AutoImageProcessor, Dinov2WithRegistersModel
    proc = AutoImageProcessor.from_pretrained(path)
    dino = Dinov2WithRegistersModel.from_pretrained(path).to(DEVICE)
    dino.eval().requires_grad_(False)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(DEVICE)
    std  = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(DEVICE)
    return dino, mean, std


@torch.no_grad()
def encode_image(path, dino, mean, std, model):
    to_t = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img = to_t(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    x = (img - mean) / std
    with torch.amp.autocast("cuda", dtype=AMP, enabled=(DEVICE.type == "cuda")):
        cls = dino(x).last_hidden_state[:, 0, :].float()
        mu, lv = model.encode(cls)
    return mu[0].float().cpu(), (0.5 * lv[0]).exp().float().cpu()


@torch.no_grad()
def encode_all(model, emb, batch=1024):
    mus, lvs = [], []
    for i in range(0, len(emb), batch):
        x = emb[i:i+batch].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=AMP, enabled=(DEVICE.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu.float().cpu()); lvs.append(lv.float().cpu())
    return torch.cat(mus), torch.cat(lvs)


def thumb(path, size=T):
    try:
        return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    except Exception:
        return Image.new("RGB", (size, size), (180, 180, 180))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--query",      required=True)
    p.add_argument("--top_k",      type=int, default=5)
    p.add_argument("--n_per_side", type=int, default=6)
    p.add_argument("--dino_path",  default="facebook/dinov2-with-registers-base")
    p.add_argument("--out_dir",    default=str(OUT_DIR))
    args = p.parse_args()

    K = args.top_k
    N = args.n_per_side
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("모델/DINO 로드...", flush=True)
    model = load_model(args.ckpt)
    dino, dm, ds = load_dino(args.dino_path)

    print("쿼리 인코딩...", flush=True)
    q_mu, q_sig = encode_image(args.query, dino, dm, ds, model)
    top_vals, top_dims = q_mu.abs().topk(K)

    print("전체 임베딩 인코딩...", flush=True)
    train_paths = build_train_paths()
    raw = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
    emb = (raw if isinstance(raw, torch.Tensor) else next(iter(raw.values()))).float()
    n = min(len(train_paths), len(emb))
    train_paths, emb = train_paths[:n], emb[:n]
    mu_all, lv_all = encode_all(model, emb)
    sig_all = (0.5 * lv_all).exp()

    # ── 레이아웃 ─────────────────────────────────────────────────────────────
    # 열: 1(쿼리) + N(그리드) = N+1
    # 행: K개 dim × (헤더 + + 행 + - 행) = K*3, 최상단 1행은 쿼리
    PX  = T / 96          # 1 thumb = 1 unit in inches
    HEADER_H = 0.35       # section header height (inches)
    THUMB_H  = PX + 0.35  # thumb + label

    n_cols  = N + 1
    col_w   = PX

    fig_w = col_w * n_cols + 0.3
    fig_h = THUMB_H * 2 + HEADER_H * K + THUMB_H * 2 * K + 0.5

    fig, axes = plt.subplots(
        K * 3, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [HEADER_H, THUMB_H, THUMB_H] * K,
                     "hspace": 0.05, "wspace": 0.04},
    )
    if K == 1:
        axes = axes.reshape(3, n_cols)

    # 전체 제목
    fig.suptitle(
        f"Query: {Path(args.query).name}   |   model: {Path(args.ckpt).parents[0].name}",
        fontsize=11, fontweight="bold", y=1.01,
    )

    # 쿼리 이미지 — 첫 번째 열 전체에 걸쳐 표시 (첫 섹션 헤더 행에)
    q_img = thumb(args.query, T * 2)

    for row_block in range(K):
        dim_idx = top_dims[row_block].item()
        q_mu_d  = q_mu[dim_idx].item()
        q_sig_d = q_sig[dim_idx].item()
        mu_d    = mu_all[:, dim_idx]
        sig_d   = sig_all[:, dim_idx]

        pos_idx = mu_d.argsort(descending=True)[:N].tolist()
        neg_idx = mu_d.argsort(descending=False)[:N].tolist()

        row_h   = row_block * 3   # header row
        row_pos = row_h + 1
        row_neg = row_h + 2

        # ── 섹션 헤더 ──────────────────────────────────────────────────────
        ax_h = axes[row_h, 0]
        for c in range(n_cols):
            axes[row_h, c].set_visible(False)
        ax_h.set_visible(True)
        ax_h.axis("off")
        sign = "▲" if q_mu_d > 0 else "▼"
        ax_h.text(
            0.0, 0.5,
            f" dim {dim_idx}   query μ = {q_mu_d:+.3f} {sign}   σ = {q_sig_d:.3f}   "
            f"[dataset μ range: {mu_d.min():+.2f} ~ {mu_d.max():+.2f}]",
            fontsize=9, fontweight="bold",
            va="center", ha="left",
            transform=ax_h.transAxes,
            color="#c0392b" if q_mu_d > 0 else "#2980b9",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="#fdecea" if q_mu_d > 0 else "#eaf4fb",
                      edgecolor="none"),
        )

        # ── 쿼리 이미지 (첫 번째 열, + 및 - 행 span) ────────────────────
        for row in [row_pos, row_neg]:
            ax_q = axes[row, 0]
            ax_q.axis("off")

        # 첫 번째 열, + 행에 쿼리 이미지 표시
        ax_q = axes[row_pos, 0]
        ax_q.imshow(q_img)
        ax_q.set_title("query", fontsize=7, pad=2, color="gray")
        ax_q.axis("off")
        axes[row_neg, 0].axis("off")
        axes[row_neg, 0].text(
            0.5, 0.5, f"μ={q_mu_d:+.3f}\nσ={q_sig_d:.3f}",
            ha="center", va="center", fontsize=7,
            transform=axes[row_neg, 0].transAxes, color="gray",
        )

        # ── + 방향 이미지 ─────────────────────────────────────────────────
        for col, idx in enumerate(pos_idx, start=1):
            ax = axes[row_pos, col]
            ax.imshow(thumb(train_paths[idx]))
            ax.set_title(
                f"μ={mu_d[idx]:+.3f}\nσ={sig_d[idx]:.3f}",
                fontsize=6.5, pad=2, color="#c0392b",
            )
            ax.axis("off")
            if col == 1:
                ax.set_ylabel("(+)", fontsize=8, color="#c0392b",
                              rotation=0, labelpad=25, va="center")

        # ── - 방향 이미지 ─────────────────────────────────────────────────
        for col, idx in enumerate(neg_idx, start=1):
            ax = axes[row_neg, col]
            ax.imshow(thumb(train_paths[idx]))
            ax.set_title(
                f"μ={mu_d[idx]:+.3f}\nσ={sig_d[idx]:.3f}",
                fontsize=6.5, pad=2, color="#2980b9",
            )
            ax.axis("off")
            if col == 1:
                ax.set_ylabel("(−)", fontsize=8, color="#2980b9",
                              rotation=0, labelpad=25, va="center")

    stem = Path(args.query).stem
    save_path = out_dir / f"query_{stem}_top{K}.png"
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {save_path}")


if __name__ == "__main__":
    main()
