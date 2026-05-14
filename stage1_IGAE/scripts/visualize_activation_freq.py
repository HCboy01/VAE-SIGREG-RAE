"""
Which neurons are active per image?
  1. Per-dim activation frequency (fraction of images with |mu| > threshold)
  2. Per-image active neuron count distribution
  3. Binary activation heatmap (images x all dims, sorted)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE


def load_embeddings(path):
    p = Path(path)
    shape_path = p.with_name(p.stem + "_shape.npy")
    shape = np.load(str(shape_path))
    return torch.from_numpy(
        np.fromfile(str(p), dtype=np.float32).reshape(shape)
    ).float()


@torch.no_grad()
def get_mu_logvar(model, data, batch_size=512, device="cuda"):
    model.eval()
    mus, logvars = [], []
    for i in range(0, len(data), batch_size):
        x = data[i : i + batch_size].to(device)
        mu, logvar = model.encode(x)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    return torch.cat(mus), torch.cat(logvars)  # [N, D]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--val_data", default="/root/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin")
    p.add_argument("--out_dir", default="visualizations")
    p.add_argument("--snr_threshold", type=float, default=1.0, help="|mu|/sigma > this → neuron active")
    p.add_argument("--n_images", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"], latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"), num_layers=a["num_layers"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    val_data = load_embeddings(args.val_data)
    mu, logvar = get_mu_logvar(model, val_data, device=device)  # [N, D]
    N, D = mu.shape
    print(f"mu: {mu.shape}")

    # SNR = |mu| / sigma  where sigma = exp(0.5 * logvar)
    sigma = (0.5 * logvar).exp()
    snr = mu.abs() / sigma.clamp(min=1e-6)
    active = (snr > args.snr_threshold)  # [N, D] bool

    # ── 1. Per-dim activation frequency ────────────────────────────────────
    freq = active.float().mean(dim=0)  # [D]: fraction of images where dim is active
    freq_sorted, sort_idx = freq.sort(descending=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_sorted.numpy(), linewidth=1.0, color="steelblue")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="50%")
    ax.axhline(0.1, color="orange", linestyle="--", linewidth=1, label="10%")
    n_always = (freq > 0.9).sum().item()
    n_sometimes = (freq > 0.1).sum().item()
    ax.set_xlabel("Dimension (sorted by activation freq descending)")
    ax.set_ylabel(f"Activation frequency (SNR>{args.snr_threshold})")
    ax.set_title(f"Per-dim Activation Frequency  |  always(>90%): {n_always}  sometimes(>10%): {n_sometimes}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = out_dir / "activation_freq.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"saved: {path1}  (always>90%: {n_always}, sometimes>10%: {n_sometimes})")

    # ── 2. Per-image active neuron count distribution ───────────────────────
    count_per_image = active.float().sum(dim=1)  # [N]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(count_per_image.numpy(), bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(count_per_image.mean().item(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={count_per_image.mean():.1f}")
    ax.set_xlabel(f"# active dims per image (SNR>{args.snr_threshold})")
    ax.set_ylabel("Count")
    ax.set_title("Active Neuron Count per Image")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = out_dir / "active_count_per_image.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"saved: {path2}  (mean active per image: {count_per_image.mean():.1f}, std: {count_per_image.std():.1f})")

    # ── 3. Binary activation heatmap ───────────────────────────────────────
    # show only dims that are active in at least 5% of images
    show_mask = freq > 0.05
    n_show = show_mask.sum().item()
    show_idx = freq.argsort(descending=True)[:n_show]

    n_img = min(args.n_images, N)
    # sort images by total active count
    img_order = count_per_image.argsort(descending=True)[:n_img]

    binary_sub = active[img_order][:, show_idx].float().numpy()  # [n_img, n_show]

    fig, ax = plt.subplots(figsize=(min(n_show * 0.12 + 2, 18), 7))
    ax.imshow(binary_sub, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel(f"Dims active in >5% images ({n_show} dims, freq descending)")
    ax.set_ylabel(f"Images ({n_img}, sorted by total active count desc)")
    ax.set_title(f"Binary Activation Heatmap  (SNR>{args.snr_threshold})")
    fig.tight_layout()
    path3 = out_dir / "binary_activation_heatmap.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"saved: {path3}  (dims shown: {n_show})")


if __name__ == "__main__":
    main()
