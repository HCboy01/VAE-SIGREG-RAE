"""
Combined Stage-1 VAE visualization.

Usage:
    python scripts/visualize_combined.py <checkpoint_dir_or_best.pt> [--n 3000] [--out OUT_DIR]

Output:
    visualizations/combined_all/<run_name>.png

Figure layout:
  Row 1 (5 panels): mu vs sigma scatter for 5 KL-quantile dims (0%, 25%, 50%, 75%, 100%)
  Row 2 (3 panels): KL-per-dim histogram | activation freq histogram | summary stats text
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE

DATA_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
DEFAULT_OUT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/combined_all")

QUANTILES = [0.0, 0.25, 0.50, 0.75, 1.0]
Q_LABELS  = ["Min KL (0%)", "25%", "Median (50%)", "75%", "Max KL (100%)"]
COLORS    = ["#4da6e8", "#6bbf72", "#e8c44d", "#e87c4d", "#c94545"]
THRESHOLDS = [0.1, 0.5, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path, help="checkpoint dir (contains best.pt) or best.pt path")
    p.add_argument("--n", type=int, default=3000, help="number of images to sample")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    is_linear = "decoder.weight" in ckpt["model"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"],
        latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"),
        num_layers=a["num_layers"],
        linear_decoder=is_linear,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, a


@torch.no_grad()
def encode_all(model, x, batch_size=512):
    mus, lvs = [], []
    for i in range(0, len(x), batch_size):
        mu, lv = model.encode(x[i:i+batch_size])
        mus.append(mu); lvs.append(lv)
    return torch.cat(mus), torch.cat(lvs)


def compute_metrics(mu, logvar):
    sigma = (0.5 * logvar).exp()                                          # [N, D]
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)       # [D]
    kl_per = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)               # [N, D]

    metrics = {}
    for thr in THRESHOLDS:
        active = (kl_per > thr).float()
        freq   = active.mean(0)   # per-dim fraction of images
        metrics[thr] = {
            "freq":           freq,
            "dead":           (freq == 0).float().mean().item(),
            "always":         (freq == 1).float().mean().item(),
            "freq_p10":       freq.quantile(0.10).item(),
            "freq_p50":       freq.quantile(0.50).item(),
            "freq_p90":       freq.quantile(0.90).item(),
            "img_active_frac": active.mean(1).mean().item(),
        }

    batch_mu_mean  = mu.mean(0)
    batch_sig2_mean = logvar.exp().mean(0)
    return sigma, kl_dim, metrics, batch_mu_mean, batch_sig2_mean


def make_figure(run_name, mu, logvar, sigma, kl_dim, metrics, batch_mu_mean, batch_sig2_mean, args_dict, n):
    beta = args_dict.get("beta_kl", "?")
    lam  = args_dict.get("lambda_sigreg", "?")
    D = len(kl_dim)

    fig = plt.figure(figsize=(24, 9))
    fig.suptitle(
        f"{run_name}   β={beta}  λ={lam}   N={n}\n"
        f"KL range: [{kl_dim.min():.4f}, {kl_dim.max():.4f}]  "
        f"median={kl_dim.median():.4f}  |  Prior: μ=0 (dashed), σ=1 (dotted)",
        fontsize=10,
    )

    # ── Row 1: mu-sigma scatter (5 quantile dims) ──────────────────────────
    q_indices = [int(q * (D - 1)) for q in QUANTILES]
    dims = [kl_dim.argsort()[i].item() for i in q_indices]

    for col_i, (dim, qlbl, color) in enumerate(zip(dims, Q_LABELS, COLORS)):
        ax = fig.add_subplot(2, 5, col_i + 1)
        mu_d    = mu[:, dim].numpy()
        sigma_d = sigma[:, dim].numpy()
        kl_val  = kl_dim[dim].item()
        freq_d  = metrics[0.1]["freq"][dim].item()

        ax.scatter(mu_d, sigma_d, s=5, alpha=0.3, color=color, linewidths=0)
        ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.55)
        ax.axhline(1, color="black", lw=0.9, ls=":",  alpha=0.55)
        ax.set_title(f"{qlbl}\ndim {dim}  KL={kl_val:.4f}", fontsize=8.5)
        ax.set_xlabel("μ", fontsize=8)
        if col_i == 0:
            ax.set_ylabel("σ", fontsize=8)
        textstr = (
            f"μ: {mu_d.mean():+.3f} ± {mu_d.std():.3f}\n"
            f"σ: {sigma_d.mean():.4f} ± {sigma_d.std():.4f}\n"
            f"freq@0.1: {freq_d:.3f}"
        )
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    # ── Row 2 left: KL-per-dim histogram ───────────────────────────────────
    ax_kl = fig.add_subplot(2, 5, 6)
    ax_kl.hist(kl_dim.numpy(), bins=100, color="#4da6e8", alpha=0.85)
    for thr, tc in zip(THRESHOLDS, ["red", "orange", "purple"]):
        n_active = (kl_dim > thr).sum().item()
        ax_kl.axvline(thr, color=tc, lw=1.2, ls="--",
                      label=f">{thr}: {n_active} ({100*n_active/D:.1f}%)")
    ax_kl.set_xlabel("KL per dim (avg over batch)", fontsize=8)
    ax_kl.set_ylabel("# dims", fontsize=8)
    ax_kl.set_title("KL per-dim distribution", fontsize=9)
    ax_kl.legend(fontsize=7)
    ax_kl.tick_params(labelsize=7)
    ax_kl.grid(True, alpha=0.2)

    # ── Row 2 mid: activation freq histogram (@0.1) ─────────────────────────
    ax_freq = fig.add_subplot(2, 5, 7)
    freq_01 = metrics[0.1]["freq"].numpy()
    ax_freq.hist(freq_01, bins=100, color="#e87c4d", alpha=0.85)
    ax_freq.axvline(0.5, color="black", lw=1, ls="--", alpha=0.5, label="50%")
    ax_freq.set_xlabel("Fraction of images active (KL>0.1)", fontsize=8)
    ax_freq.set_ylabel("# dims", fontsize=8)
    ax_freq.set_title("Per-dim activation frequency", fontsize=9)
    ax_freq.legend(fontsize=7)
    ax_freq.tick_params(labelsize=7)
    ax_freq.grid(True, alpha=0.2)

    # ── Row 2 mid-right: batch μ/σ² distributions ──────────────────────────
    ax_bm = fig.add_subplot(2, 5, 8)
    ax_bm.hist(batch_mu_mean.numpy(), bins=100, color="#6bbf72", alpha=0.85, label="batch mean μ")
    ax_bm.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax_bm.set_xlabel("value", fontsize=8)
    ax_bm.set_ylabel("# dims", fontsize=8)
    ax_bm.set_title(f"Batch mean μ per dim\nmean_abs={batch_mu_mean.abs().mean():.4f}", fontsize=9)
    ax_bm.tick_params(labelsize=7)
    ax_bm.grid(True, alpha=0.2)

    ax_bv = fig.add_subplot(2, 5, 9)
    ax_bv.hist(batch_sig2_mean.numpy(), bins=100, color="#b07cd4", alpha=0.85, label="batch mean σ²")
    ax_bv.axvline(1, color="black", lw=1, ls="--", alpha=0.6)
    ax_bv.set_xlabel("value", fontsize=8)
    ax_bv.set_title(f"Batch mean σ² per dim\nmean={batch_sig2_mean.mean():.4f}", fontsize=9)
    ax_bv.tick_params(labelsize=7)
    ax_bv.grid(True, alpha=0.2)

    # ── Row 2 right: summary text ────────────────────────────────────────────
    ax_txt = fig.add_subplot(2, 5, 10)
    ax_txt.axis("off")
    lines = [f"{'─'*28}", f"  {run_name}", f"{'─'*28}", ""]
    lines += [f"  β={beta}  λ={lam}", ""]
    lines += [f"  KL dims (D={D}):"]
    lines += [f"    total kl = {kl_dim.sum():.1f}"]
    for thr in THRESHOLDS:
        n_act = (kl_dim > thr).sum().item()
        lines.append(f"    active@{thr} = {n_act} ({100*n_act/D:.1f}%)")
    lines += [""]
    for thr in THRESHOLDS:
        m = metrics[thr]
        lines += [
            f"  threshold={thr}:",
            f"    dead   = {m['dead']:.4f}",
            f"    always = {m['always']:.4f}",
            f"    freq p50 = {m['freq_p50']:.3f}",
            f"    img_frac = {m['img_active_frac']:.4f}",
            "",
        ]
    lines += [f"  batch_mean_abs_μ = {batch_mu_mean.abs().mean():.4f}"]
    lines += [f"  batch_mean_σ²    = {batch_sig2_mean.mean():.4f}"]

    ax_txt.text(0.02, 0.98, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=7, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", alpha=0.9))

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    ckpt_path = args.ckpt
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    run_name = ckpt_path.parent.name

    print(f"Loading model: {ckpt_path}")
    model, args_dict = load_model(ckpt_path)

    print(f"Loading data: {DATA_PATH}")
    raw  = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
    torch.manual_seed(args.seed)
    x = data[torch.randperm(len(data))[:args.n]]
    print(f"Sampled {args.n} images from {data.shape}")

    print("Encoding...")
    mu, logvar = encode_all(model, x)
    sigma, kl_dim, metrics, batch_mu_mean, batch_sig2_mean = compute_metrics(mu, logvar)

    print(f"kl_dim: min={kl_dim.min():.5f}  median={kl_dim.median():.5f}  max={kl_dim.max():.5f}")
    for thr in THRESHOLDS:
        m = metrics[thr]
        print(f"  thr={thr}: dead={m['dead']:.4f}  always={m['always']:.4f}  "
              f"freq_p50={m['freq_p50']:.3f}  img_frac={m['img_active_frac']:.4f}")

    fig = make_figure(run_name, mu, logvar, sigma, kl_dim, metrics,
                      batch_mu_mean, batch_sig2_mean, args_dict, args.n)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{run_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
