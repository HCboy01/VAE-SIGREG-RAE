"""
Per-neuron extreme image visualization.

For each selected neuron j:
  - Score each image by KL_j = 0.5*(mu_j^2 + exp(logvar_j) - logvar_j - 1)
    (combines mean AND variance deviation from N(0,1))
  - Take top-16 most extreme images
  - Save as a 4x4 grid → visualizations/neuron_{j}_kl{score:.2f}.png

Outputs one PNG per neuron (10 total).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from src.vae_sigreg import OvercompleteVariationalAE


# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
def encode_all(model, embeddings, device, batch_size=1024):
    """Returns mu [N,D] and logvar [N,D] for all embeddings."""
    mus, lvs = [], []
    for i in range(0, len(embeddings), batch_size):
        mu, lv = model.encode(embeddings[i:i+batch_size].to(device))
        mus.append(mu.cpu())
        lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def load_image(path, size):
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def annotate(img, line1, line2=""):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0, 180))
    draw.text((2, 1),  line1, fill=(255, 220, 80))
    if line2:
        draw.text((2, 11), line2, fill=(200, 200, 200))
    return out


def make_grid(images, nrow=4, padding=4, bg=160):
    w, h = images[0].size
    ncols = nrow
    nrows = (len(images) + ncols - 1) // ncols
    gw = ncols * w + (ncols + 1) * padding
    gh = nrows * h + (nrows + 1) * padding
    grid = Image.new("RGB", (gw, gh), (bg, bg, bg))
    for idx, img in enumerate(images):
        r, c = divmod(idx, ncols)
        grid.paste(img, (padding + c*(w+padding), padding + r*(h+padding)))
    return grid


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/best.pt")
    p.add_argument("--embeddings", default="/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin")
    p.add_argument("--paths",      default="/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_paths.txt")
    p.add_argument("--n_neurons",  type=int, default=10,   help="How many neurons to visualize")
    p.add_argument("--n_images",   type=int, default=16,   help="Images per neuron (shown as grid)")
    p.add_argument("--img_size",   type=int, default=128)
    p.add_argument("--output_dir", default="visualizations/neurons")
    p.add_argument("--neuron_ids", type=int, nargs="*", default=None,
                   help="Manually specify neuron indices. If omitted, auto-selects top-N.")
    p.add_argument("--rank_neurons_by", choices=["max_kl", "std_mu"], default="max_kl")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, epoch = load_model(args.ckpt, device)
    print(f"Model loaded (epoch={epoch})")

    # Load embeddings + paths
    shape = np.load(args.embeddings.replace(".bin", "_shape.npy"))
    embeddings = torch.from_numpy(
        np.fromfile(args.embeddings, dtype=np.float32).reshape(shape)
    )
    paths = [ln.strip() for ln in open(args.paths)]
    print(f"Embeddings: {embeddings.shape}  paths: {len(paths)}")

    # Encode → mu, logvar
    print("Encoding...")
    mu, logvar = encode_all(model, embeddings, device)   # [N, D]
    print(f"mu: {mu.shape}  logvar: {logvar.shape}")

    # KL( N(mu_j, sigma_j²) || N(0,1) ) = 0.5*(mu_j² + sigma_j² - log(sigma_j²) - 1)
    # 정보이론적으로 올바른 prior 이탈 척도
    print("Computing per-image per-neuron KL divergence from N(0,1)...")
    var = logvar.exp()                             # [N, D]
    deviation = 0.5 * (mu.pow(2) + var - logvar - 1)  # [N, D]

    # Select neurons
    if args.neuron_ids:
        neuron_ids = args.neuron_ids
    else:
        if args.rank_neurons_by == "max_kl":
            scores = deviation.max(dim=0).values   # 이미지 하나라도 가장 극단적인 뉴런
        else:
            scores = mu.std(dim=0)
        neuron_ids = scores.argsort(descending=True)[:args.n_neurons].tolist()

    print(f"Selected neurons: {neuron_ids}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_half = args.n_images // 2   # e.g. 8 positive + 8 negative

    for rank, j in enumerate(neuron_ids):
        dev_j = deviation[:, j]
        mu_j  = mu[:, j]
        var_j = var[:, j]

        # Split by mu sign, rank within each side by combined deviation (mu²+(var-1)²)
        pos_mask = mu_j > 0
        neg_mask = mu_j < 0
        pos_scores = torch.where(pos_mask, dev_j, torch.tensor(-1.0))
        neg_scores = torch.where(neg_mask, dev_j, torch.tensor(-1.0))
        pos_idx = pos_scores.argsort(descending=True)[:n_half].tolist()
        neg_idx = neg_scores.argsort(descending=True)[:n_half].tolist()

        pos_imgs, neg_imgs = [], []
        for i in pos_idx:
            img = load_image(paths[i], args.img_size)
            l1 = f"mu=+{mu_j[i]:.3f}"
            l2 = f"var={var_j[i]:.3f} kl={dev_j[i]:.3f}"
            pos_imgs.append(annotate(img, l1, l2))
        for i in neg_idx:
            img = load_image(paths[i], args.img_size)
            l1 = f"mu={mu_j[i]:.3f}"
            l2 = f"var={var_j[i]:.3f} kl={dev_j[i]:.3f}"
            neg_imgs.append(annotate(img, l1, l2))

        grid = make_grid(pos_imgs + neg_imgs, nrow=n_half)
        fname = out_dir / f"rank{rank:02d}_neuron{j}_kl{dev_j.max():.3f}.png"
        grid.save(str(fname))

        print(f"  [{rank+1:2d}/10] neuron {j:4d} | "
              f"mu=[{mu_j.min():.3f}, {mu_j.max():.3f}]  "
              f"var=[{var_j.min():.3f}, {var_j.max():.3f}]  "
              f"max_dev={dev_j.max():.3f}  → {fname.name}")

    print(f"\nDone. {len(neuron_ids)} grids saved to {out_dir}/")


if __name__ == "__main__":
    main()
