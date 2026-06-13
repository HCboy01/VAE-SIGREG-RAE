"""
For selected latent dims, show the actual FFHQ images that are extreme along
that dim. Encode N images; for each dim pick:
  - top-K images by per-image KL of that dim   (most "active" for this feature)
  - bottom-K images by mu (smallest mu)
  - top-K images by mu (largest mu)
and lay them out in a grid (each dim = 3 rows; columns = the K picks).

Usage:
    python scripts/feature_image_grid_kl_mu.py [RUN] --dims 3303 824 [--n 3000] [--topk 5]
Output:
    visualizations/mu_sorted_feature_grid/<run>_klmu_grid_dim<...>.png
"""

import argparse
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
EMB_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
IMG_DIR   = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
VAL_IDX   = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sorted_feature_grid")

SEED = 42
THUMB = 160
BATCH_SIZE = 512
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="s1_grid_b1e-4_l100_lindec")
    p.add_argument("--dims", type=int, nargs="+", default=[3303, 824])
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def build_train_image_paths(img_dir, val_idx):
    val_ids = set(val_idx.read_text().splitlines()) if val_idx.exists() else set()
    all_imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return [p for p in all_imgs if p.stem not in val_ids]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    state = ckpt["model"]
    linear_decoder = bool(a.get("linear_decoder", "decoder.weight" in state))
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
        x = emb[i:i+BATCH_SIZE].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu.float().cpu()); lvs.append(lv.float().cpu())
    return torch.cat(mus), torch.cat(lvs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    paths = build_train_image_paths(IMG_DIR, VAL_IDX)
    obj = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    emb = (obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))).float()
    n_total = min(len(paths), len(emb))
    paths, emb = paths[:n_total], emb[:n_total]
    print(f"aligned dataset: {n_total} (paths {len(paths)}, emb {len(emb)})")

    torch.manual_seed(SEED)
    sub = torch.randperm(n_total)[:min(args.n, n_total)]
    sub_paths = [paths[i] for i in sub.tolist()]
    sub_emb = emb[sub]

    model, a = load_model(CKPT_ROOT / args.run / "best.pt", device)
    mu, logvar = encode(model, sub_emb, device)
    var = logvar.exp()
    kl_img = 0.5 * (mu.pow(2) + var - logvar - 1)          # [N, D] per-image per-dim KL
    kl_dim = kl_img.mean(0)                                 # [D]
    D = mu.shape[1]
    kl_rank = (kl_dim.argsort(descending=True).argsort() + 1)   # rank 1 = highest KL
    beta, lam = a.get("beta_kl", "?"), a.get("lambda_sigreg", "?")

    print("\nRequested dims:")
    for d in args.dims:
        flag = "  <-- low KL (near-dead); did you mean a high-KL dim?" \
               if kl_rank[d].item() > D // 2 else ""
        print(f"  dim {d:5d}: KL_avg={kl_dim[d].item():.5f}  rank={kl_rank[d].item()}/{D}{flag}")

    K = args.topk
    # (criterion label, selecting fn) per dim -> list of (image idx-in-subset, metric value)
    rows = []   # each: (dim, crit_label, [(si, val), ...])
    for d in args.dims:
        kld = kl_img[:, d]
        mud = mu[:, d]
        top_kl  = torch.topk(kld, K).indices.tolist()
        low_mu  = torch.topk(mud, K, largest=False).indices.tolist()
        high_mu = torch.topk(mud, K, largest=True).indices.tolist()
        rows.append((d, f"top KL", [(si, kld[si].item()) for si in top_kl]))
        rows.append((d, f"low mu", [(si, mud[si].item()) for si in low_mu]))
        rows.append((d, f"high mu", [(si, mud[si].item()) for si in high_mu]))

    # ── grid ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, K,
        figsize=(K * (THUMB + 8) / 100, n_rows * (THUMB + 34) / 100),
        squeeze=False,
    )
    dimstr = ", ".join(str(d) for d in args.dims)
    fig.suptitle(
        f"{args.run}  β={beta} λ={lam}  N={len(sub)}  |  dims [{dimstr}]\n"
        f"per dim: top-{K} by per-image KL  /  lowest-{K} μ  /  highest-{K} μ",
        fontsize=12,
    )

    for r, (d, crit, picks) in enumerate(rows):
        is_kl = crit.startswith("top KL")
        for c in range(K):
            ax = axes[r, c]; ax.axis("off")
            si, val = picks[c]
            img = Image.open(sub_paths[si]).convert("RGB").resize((THUMB, THUMB), Image.BICUBIC)
            ax.imshow(img)
            tag = f"KL={val:.2f}" if is_kl else f"μ={val:+.2f}"
            ax.set_title(f"{tag}\n{sub_paths[si].stem}", fontsize=7, pad=2)
            if c == 0:
                ax.text(-0.30, 0.5,
                        f"dim {d}\nrank {kl_rank[d].item()}/{D}\nKL_avg={kl_dim[d].item():.3f}\n[{crit}]",
                        transform=ax.transAxes, fontsize=8.5, va="center", ha="right",
                        fontweight=("bold" if is_kl else "normal"))

    fig.tight_layout(rect=(0.06, 0, 1, 0.94))
    tag = "_".join(str(d) for d in args.dims)
    out_png = OUT_DIR / f"{args.run}_klmu_grid_dim{tag}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_png}")


if __name__ == "__main__":
    main()
