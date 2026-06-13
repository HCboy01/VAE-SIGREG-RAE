"""
What does the latent origin (z = 0) look like?

The Stage-1 decoder reconstructs DINO features, not pixels, so "the z=0 image"
can only be shown by retrieving the real FFHQ images closest to that point.
Two complementary views (both rendered):

  feature-NN : decode(z=0) gives a target DINO feature (for a linear decoder
               this is exactly decoder.bias). We retrieve the train images
               whose DINO feature is most similar (cosine) to it. Since
               E[z]~0, decode(0) ~ dataset-mean feature ~ the "average face".

  latent-NN  : encode every image and retrieve those with the smallest ||mu||,
               i.e. the images that actually sit at the latent origin.

Reports decode(0) vs data-mean-feature similarity and the ||mu|| distribution,
and prints how much the two retrieval sets overlap.

Usage:
    python scripts/zero_center_image.py [RUN] [--topk 16] [--n 20000]
Output:
    visualizations/zero_center/<run>_zero_center.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
EMB_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
IMG_DIR   = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/train/images")
VAL_IDX   = Path("/scratch/x3411a10/datasets/ffhq256/val_indices.txt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/zero_center")

SEED = 42
THUMB = 144
BATCH_SIZE = 512
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="s1_grid_b1e-3_l100_lindec")
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--n", type=int, default=20000, help="num images to search over")
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
def encode_mu(model, emb, device):
    out = []
    for i in range(0, len(emb), BATCH_SIZE):
        x = emb[i:i+BATCH_SIZE].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            mu, _ = model.encode(x)
        out.append(mu.float().cpu())
    return torch.cat(out)


def render_row(axes_row, idxs, paths, captions):
    for ax, i, cap in zip(axes_row, idxs, captions):
        ax.axis("off")
        img = Image.open(paths[i]).convert("RGB").resize((THUMB, THUMB), Image.BICUBIC)
        ax.imshow(img)
        ax.set_title(cap, fontsize=6.5, pad=1.5)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  run: {args.run}")

    paths = build_train_image_paths(IMG_DIR, VAL_IDX)
    obj = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    emb = (obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))).float()
    n_total = min(len(paths), len(emb))
    paths, emb = paths[:n_total], emb[:n_total]

    torch.manual_seed(SEED)
    sub = torch.randperm(n_total)[:min(args.n, n_total)]
    sub_paths = [paths[i] for i in sub.tolist()]
    sub_emb = emb[sub].contiguous()

    model, a = load_model(CKPT_ROOT / args.run / "best.pt", device)
    D = int(a.get("latent_dim", 6144))
    beta, lam = a.get("beta_kl", "?"), a.get("lambda_sigreg", "?")

    # decode(z=0) -> target feature
    with torch.no_grad():
        z0 = torch.zeros(1, D, device=device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            f0 = model.decode(z0).float().cpu().squeeze(0)  # [768]

    data_mean = sub_emb.mean(0)  # [768]
    cos_dm = F.cosine_similarity(f0, data_mean, dim=0).item()
    l2_dm = (f0 - data_mean).norm().item()
    print(f"decode(0) vs data-mean feature:  cos={cos_dm:.4f}  L2={l2_dm:.4f}  "
          f"(||decode(0)||={f0.norm():.3f}, ||data_mean||={data_mean.norm():.3f})")

    # feature-NN: cosine similarity of every image feature to decode(0)
    sim = F.cosine_similarity(sub_emb, f0.unsqueeze(0), dim=1)  # [N]
    feat_nn = sim.topk(args.topk).indices

    # latent-NN: smallest ||mu||
    mu = encode_mu(model, sub_emb, device)
    mu_norm = mu.norm(dim=1)  # [N]
    lat_nn = (-mu_norm).topk(args.topk).indices  # smallest norm
    print(f"||mu|| over {len(mu)} imgs: min={mu_norm.min():.3f} "
          f"median={mu_norm.median():.3f} max={mu_norm.max():.3f}")

    overlap = len(set(feat_nn.tolist()) & set(lat_nn.tolist()))
    print(f"overlap between feature-NN and latent-NN top-{args.topk}: {overlap}")

    # render: row 0 = feature-NN, row 1 = latent-NN
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = args.topk
    fig, axes = plt.subplots(2, cols, figsize=(cols * (THUMB + 6) / 100, 2 * (THUMB + 30) / 100),
                             squeeze=False)
    fig.suptitle(
        f"{args.run}  beta={beta} lambda={lam}  |  latent origin z=0\n"
        f"row0: nearest to decode(0) by cosine (decode0 vs data-mean cos={cos_dm:.3f})   "
        f"row1: smallest ||mu||",
        fontsize=11,
    )
    render_row(axes[0], feat_nn.tolist(), sub_paths,
               [f"cos={sim[i]:.3f}" for i in feat_nn.tolist()])
    render_row(axes[1], lat_nn.tolist(), sub_paths,
               [f"||mu||={mu_norm[i]:.2f}" for i in lat_nn.tolist()])

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = OUT_DIR / f"{args.run}_zero_center.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
