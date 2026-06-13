"""
Combined mu-sigma scatter batch script for all MLP-decoder grid checkpoints.

Per checkpoint, generates a single PNG with two rows:
  Row 1 (5 panels): KL quantile dims - 0%, 25%, 50%, 75%, 100%
                    x=mu, y=sigma, each point = one image
  Row 2 (1 wide panel): per-dim scatter for image index 0
                    x=mu[0,d], y=sigma[0,d], each point = one latent dim
                    colored by per-image KL of that dim

Output: visualizations/mu_sigma_combined_mlp/<run_name>.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sigma_combined_mlp")
N_IMAGES = 3000
IMG0_IDX = 0


parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="*", help="optional checkpoint directory names to visualize")
args_cli = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

raw = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
print(f"Data: {data.shape}")

torch.manual_seed(42)
idx = torch.randperm(len(data))[:N_IMAGES]
x_sample = data[idx]
print(f"Sampled {N_IMAGES} images (fixed seed)\n")

OUT_DIR.mkdir(parents=True, exist_ok=True)

mlp_dirs = sorted(
    d
    for d in CKPT_ROOT.iterdir()
    if d.is_dir()
    and d.name.startswith("s1_grid_")
    and "lindec" not in d.name
    and (d / "best.pt").exists()
)
if args_cli.runs:
    requested = set(args_cli.runs)
    mlp_dirs = [d for d in mlp_dirs if d.name in requested]
    missing = sorted(requested - {d.name for d in mlp_dirs})
    if missing:
        raise FileNotFoundError(f"Missing MLP checkpoints: {missing}")
print(f"Found {len(mlp_dirs)} MLP checkpoints\n")


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    is_linear = "decoder.weight" in ckpt["model"]
    if is_linear:
        raise ValueError(f"{ckpt_path} is a linear-decoder checkpoint")

    model = OvercompleteVariationalAE(
        input_dim=args["input_dim"],
        latent_dim=args["latent_dim"],
        hidden_dim=args.get("hidden_dim"),
        num_layers=args["num_layers"],
        linear_decoder=False,
    )
    model.load_state_dict(ckpt["model"])
    return model, args


@torch.no_grad()
def encode(model, x, batch_size=512):
    model.eval()
    mus, lvs = [], []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(device)
        mu, lv = model.encode(xb)
        mus.append(mu.cpu())
        lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def make_combined(run_name, mu, logvar, beta, lam):
    sigma = (0.5 * logvar).exp()
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)
    d_latent = len(kl_dim)

    q_labels = ["Min KL (0%)", "25%", "Median (50%)", "75%", "Max KL (100%)"]
    q_colors = ["#4da6e8", "#6bbf72", "#e8c44d", "#e87c4d", "#c94545"]
    q_indices = [int(q * (d_latent - 1)) for q in [0.0, 0.25, 0.50, 0.75, 1.0]]
    dims_q = [kl_dim.argsort()[i].item() for i in q_indices]

    mu0 = mu[IMG0_IDX].numpy()
    sig0 = sigma[IMG0_IDX].numpy()
    kl0 = (
        0.5
        * (
            mu[IMG0_IDX].pow(2)
            + logvar[IMG0_IDX].exp()
            - logvar[IMG0_IDX]
            - 1
        )
    ).numpy()

    fig = plt.figure(figsize=(26, 10))
    fig.suptitle(
        f"{run_name}   MLP decoder   beta={beta}  lambda={lam}\n"
        f"kl range (avg): [{kl_dim.min():.5f}, {kl_dim.max():.5f}]  "
        f"N={N_IMAGES}  |  prior: mu=0 (dashed), sigma=1 (dotted)",
        fontsize=10,
    )

    gs = gridspec.GridSpec(
        2,
        5,
        figure=fig,
        hspace=0.45,
        wspace=0.32,
        top=0.88,
        bottom=0.07,
        left=0.05,
        right=0.98,
    )

    for col, (dim, label, color) in enumerate(zip(dims_q, q_labels, q_colors)):
        ax = fig.add_subplot(gs[0, col])
        mu_d = mu[:, dim].numpy()
        sigma_d = sigma[:, dim].numpy()
        kl_val = kl_dim[dim].item()
        ax.scatter(mu_d, sigma_d, s=5, alpha=0.30, color=color, linewidths=0)
        ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.55)
        ax.axhline(1, color="black", lw=0.9, ls=":", alpha=0.55)
        ax.set_title(f"{label}\ndim {dim}  KL={kl_val:.5f}", fontsize=8.5)
        ax.set_xlabel("mu", fontsize=8)
        if col == 0:
            ax.set_ylabel("sigma", fontsize=8)
        text = (
            f"mu: {mu_d.mean():+.3f} +/- {mu_d.std():.3f}\n"
            f"sigma: {sigma_d.mean():.4f} +/- {sigma_d.std():.4f}"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
        )
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[1, :])
    norm_kl = mcolors.LogNorm(vmin=max(kl0.min(), 1e-6), vmax=kl0.max() + 1e-9)
    sc = ax2.scatter(mu0, sig0, c=kl0, cmap="plasma", norm=norm_kl, s=6, alpha=0.5, linewidths=0)
    plt.colorbar(sc, ax=ax2, label="KL per dim (image 0)", pad=0.01)

    ax2.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax2.axhline(1, color="black", lw=1.0, ls=":", alpha=0.6)
    for color, thr in [("#2196F3", 0.8), ("#4CAF50", 0.5), ("#F44336", 0.2)]:
        n = (sig0 < thr).sum()
        ax2.axhline(thr, color=color, lw=0.8, ls="--", alpha=0.7, label=f"sigma<{thr}: {n} dims")

    ax2.set_title(
        f"Per-dim mu vs sigma - image idx {IMG0_IDX}  (each point = 1 latent dim, D={d_latent})\n"
        f"mu: mean={mu0.mean():+.3f}  std={mu0.std():.3f}  |  "
        f"sigma: mean={sig0.mean():.4f}  std={sig0.std():.4f}  "
        f"min={sig0.min():.4f}  max={sig0.max():.4f}",
        fontsize=9,
    )
    ax2.set_xlabel("mu (posterior mean)", fontsize=9)
    ax2.set_ylabel("sigma (posterior std)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=8)

    fig.text(0.005, 0.72, "KL quantile\n(0/25/50/75/100%)", fontsize=8, va="center", rotation=90, color="#555")
    fig.text(0.005, 0.28, f"Per-dim scatter\n(image {IMG0_IDX})", fontsize=8, va="center", rotation=90, color="#555")

    out_path = OUT_DIR / f"{run_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


for i, ckpt_dir in enumerate(mlp_dirs):
    run_name = ckpt_dir.name
    print(f"[{i + 1:2d}/{len(mlp_dirs)}] {run_name} ...", end=" ", flush=True)
    try:
        model, args = load_model(ckpt_dir / "best.pt")
        model.to(device)
        mu, logvar = encode(model, x_sample)
        model.cpu()
        beta = args.get("beta_kl", "?")
        lam = args.get("lambda_sigreg", "?")
        out = make_combined(run_name, mu, logvar, beta, lam)
        print(f"saved -> {out.name}")
    except Exception as exc:
        print(f"ERROR: {exc}")

print(f"\nDone. Output: {OUT_DIR}")
