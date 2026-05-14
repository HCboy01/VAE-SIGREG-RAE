"""
Latent covariance analysis — showing near-identity structure.

Three panels:
  1. Correlation heatmap (vmin/vmax=±0.05, small values appear white naturally)
  2. Off-diagonal correlation distribution
  3. Per-dim variance of z (should cluster tightly around 1)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

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
    return model, ckpt["epoch"], a.get("beta_kl", "?")


@torch.no_grad()
def encode_all(model, embeddings, device, batch_size=1024):
    zs = []
    for i in range(0, len(embeddings), batch_size):
        mu, lv = model.encode(embeddings[i:i+batch_size].to(device))
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        zs.append(z.cpu())
    return torch.cat(zs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch, beta_kl = load_model("checkpoints/epoch_0200.pt", device)
    print(f"epoch={epoch}, beta_kl={beta_kl}")

    shape = np.load("/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features_shape.npy")
    embeddings = torch.from_numpy(
        np.fromfile("/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin",
                    dtype=np.float32).reshape(shape)
    )
    print("Encoding...")
    z = encode_all(model, embeddings, device)   # [N, D]
    N, D = z.shape
    print(f"z: {z.shape}")

    # ── Correlation matrix (first 64 dims) ──────────────────────────────────
    n_sub = min(64, D)
    z_sub = z[:, :n_sub].numpy()
    corr = np.corrcoef(z_sub, rowvar=False)   # [n_sub, n_sub]

    # ── Off-diagonal correlations ────────────────────────────────────────────
    off_mask = ~np.eye(n_sub, dtype=bool)
    off_vals = corr[off_mask]
    abs_off = np.abs(off_vals)

    # ── Per-dim variance (all dims) ─────────────────────────────────────────
    var_per_dim = z.numpy().var(axis=0)   # [D]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Latent Covariance Analysis — z (sampled), epoch={epoch}, beta_kl={beta_kl}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: correlation heatmap (wide colormap range → small values ≈ white)
    ax = axes[0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.05, vmax=0.05,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Correlation heatmap (first {n_sub} dims)", fontsize=10)
    ax.set_xlabel("dim j"); ax.set_ylabel("dim i")

    # Panel 2: off-diagonal correlation distribution
    ax = axes[1]
    ax.hist(off_vals, bins=80, color="#4C72B0", edgecolor="none", density=True)
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("correlation coefficient")
    ax.set_ylabel("density")
    ax.set_title(
        f"Off-diagonal correlation distribution\n"
        f"mean|r|={abs_off.mean():.4f},  max|r|={abs_off.max():.4f}",
        fontsize=10,
    )
    ax.text(0.97, 0.97,
            f"mean|r|={abs_off.mean():.5f}\nmax|r|={abs_off.max():.5f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Panel 3: per-dim variance
    ax = axes[2]
    ax.hist(var_per_dim, bins=60, color="#DD8452", edgecolor="none", density=True)
    ax.axvline(1.0, color="red", ls="--", lw=1.5, label="Target = 1")
    ax.axvline(var_per_dim.mean(), color="navy", ls="-", lw=1.5,
               label=f"mean={var_per_dim.mean():.4f}")
    ax.set_xlabel("Var(z_j)")
    ax.set_ylabel("density")
    ax.set_title(f"Per-dim variance (D={D})\nstd={var_per_dim.std():.5f}", fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = Path("visualizations/covariance_z.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
