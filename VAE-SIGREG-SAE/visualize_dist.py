"""
Per-neuron mu distribution plot.
Shows histogram of mu_j across all images + N(0,1) reference.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from src.vae_sigreg import OvercompleteVariationalAE


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"], latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"), num_layers=a["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["epoch"]


@torch.no_grad()
def encode_all(model, embeddings, device, batch_size=1024):
    mus, lvs = [], []
    for i in range(0, len(embeddings), batch_size):
        mu, lv = model.encode(embeddings[i:i+batch_size].to(device))
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, epoch = load_model("checkpoints/best.pt", device)
    print(f"Model epoch={epoch}")

    shape = np.load("/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features_shape.npy")
    embeddings = torch.from_numpy(
        np.fromfile("/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin",
                    dtype=np.float32).reshape(shape)
    )
    print("Encoding...")
    mu, logvar = encode_all(model, embeddings, device)   # [N, D]

    # Select same top-10 neurons by max KL
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    neuron_ids = kl.max(dim=0).values.argsort(descending=True)[:10].tolist()
    print(f"Neurons: {neuron_ids}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"Per-neuron mu distribution (N={len(mu):,} images, epoch={epoch})",
                 fontsize=14, fontweight="bold")
    x_ref = np.linspace(-6, 6, 300)

    for ax, j in zip(axes.flat, neuron_ids):
        vals = mu[:, j].numpy()
        mu_val  = vals.mean()
        std_val = vals.std()

        # Normality test (Shapiro on subsample)
        sw_stat, sw_p = stats.shapiro(vals[:5000])

        # KL from N(0,1): KL(N(mu,s^2) || N(0,1)) = 0.5*(mu^2 + s^2 - ln(s^2) - 1)
        kl_empirical = 0.5 * (mu_val**2 + std_val**2 - np.log(std_val**2) - 1)

        ax.hist(vals, bins=80, density=True, color="#4C72B0", alpha=0.7,
                edgecolor="none", label="mu distribution")

        # Fitted Gaussian
        ax.plot(x_ref, stats.norm.pdf(x_ref, mu_val, std_val),
                "C1-", lw=2, label=f"fit N({mu_val:.2f},{std_val:.2f}²)")

        # N(0,1) reference
        ax.plot(x_ref, stats.norm.pdf(x_ref, 0, 1),
                "k--", lw=1.5, alpha=0.6, label="N(0,1)")

        ax.set_title(f"neuron {j}", fontsize=11, fontweight="bold")
        ax.set_xlabel("μ value")
        ax.set_xlim(-6, 6)

        # Stats box
        info = (f"mean={mu_val:.3f}  std={std_val:.3f}\n"
                f"KL↔N(0,1)={kl_empirical:.3f}\n"
                f"Shapiro p={sw_p:.3f}")
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                fontsize=7.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    out = Path("visualizations/neuron_distributions.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
