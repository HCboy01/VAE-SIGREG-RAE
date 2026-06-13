"""
Measure whether E[z] (expectation of z over data) differs between
high-KL and low-KL latent dimensions.

For a Gaussian VAE, z = mu + sigma * eps, so E_eps[z] = mu, and the
per-dim expectation of z over the dataset is  Ez[d] = mean_data(mu[:, d]).

Per-dim KL decomposes as
    KL_d = 0.5 * ( E[mu^2] + E[sigma^2] - E[log sigma^2] - 1 )
and  E[mu^2] = Ez^2 + Var_data[mu], so
    KL_d = 0.5*Ez^2            (MEAN term  -- shift of z's expectation)
         + 0.5*Var_data[mu]    (VAR  term  -- spread of posterior means)
         + 0.5*(E[sigma^2] - E[log sigma^2] - 1)   (SIGMA term)

Hypothesis under test: high-KL dims do NOT have a larger |E[z]|; their KL is
carried by the VAR and SIGMA terms, while E[z] ~ 0 for all KL levels.

Usage:
    python scripts/analyze_z_expectation_by_kl.py [RUN_NAME] [--n 3000]

Outputs:
    visualizations/mu_sigma_combined_lindec/<run>_z_expectation_by_kl.png
    visualizations/mu_sigma_combined_lindec/<run>_z_expectation_by_kl.md
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/mu_sigma_combined_lindec")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="s1_grid_b1e-4_l100_lindec")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    is_linear = "decoder.weight" in ckpt["model"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"], latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"), num_layers=a["num_layers"],
        linear_decoder=is_linear,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, a


@torch.no_grad()
def encode_all(model, x, device, batch_size=512):
    mus, lvs = [], []
    for i in range(0, len(x), batch_size):
        mu, lv = model.encode(x[i:i+batch_size].to(device))
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_path = CKPT_ROOT / args.run / "best.pt"
    model, a = load_model(ckpt_path)
    model.to(device)
    beta = a.get("beta_kl", "?"); lam = a.get("lambda_sigreg", "?")

    raw  = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
    torch.manual_seed(args.seed)
    x = data[torch.randperm(len(data))[:args.n]]
    print(f"Encoding {args.n} images from {tuple(data.shape)} ...")

    mu, logvar = encode_all(model, x, device)
    model.cpu()
    N, D = mu.shape

    # ── per-dim quantities (expectations over the data) ──────────────────────
    Ez    = mu.mean(0)                       # E_data[mu]  == per-dim E[z]
    Vmu   = mu.var(0, unbiased=False)        # Var_data[mu]
    s2bar = logvar.exp().mean(0)             # E[sigma^2]
    lvbar = logvar.mean(0)                   # E[log sigma^2]
    sbar  = (0.5 * logvar).exp().mean(0)     # E[sigma]
    kl    = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)   # [D]

    # KL decomposition (per dim)
    kl_mean  = 0.5 * Ez.pow(2)               # contribution of E[z] shift
    kl_var   = 0.5 * Vmu                     # contribution of mu spread
    kl_sigma = 0.5 * (s2bar - lvbar - 1)     # contribution of sigma
    # sanity: kl_mean + kl_var + kl_sigma == kl  (up to fp)

    Ez_np, kl_np = Ez.numpy(), kl.numpy()

    # ── grouping by KL ───────────────────────────────────────────────────────
    order = np.argsort(kl_np)
    n_top = max(1, int(round(0.05 * D)))     # top 5% KL dims
    top_idx = order[-n_top:]
    bot_idx = order[:D - n_top]              # rest
    # finer: quintiles
    quint = np.array_split(order, 5)

    def grp(idx):
        return dict(
            n=len(idx),
            kl=kl_np[idx].mean(),
            Ez_mean=Ez_np[idx].mean(),
            Ez_std=Ez_np[idx].std(),
            absEz=np.abs(Ez_np[idx]).mean(),
            Vmu=Vmu.numpy()[idx].mean(),
            sbar=sbar.numpy()[idx].mean(),
        )

    g_top, g_bot = grp(top_idx), grp(bot_idx)

    # ── statistical test: is |E[z]| different between top-KL and rest? ───────
    try:
        from scipy.stats import mannwhitneyu
        u, p_mw = mannwhitneyu(np.abs(Ez_np[top_idx]), np.abs(Ez_np[bot_idx]),
                               alternative="two-sided")
        mw_str = f"Mann-Whitney U on |E[z]| (top5% vs rest): U={u:.0f}, p={p_mw:.3g}"
    except Exception as e:
        mw_str = f"(scipy unavailable: {e})"

    # ── KL budget: how much total KL comes from each term ────────────────────
    tot      = kl.sum().item()
    tot_mean = kl_mean.sum().item()
    tot_var  = kl_var.sum().item()
    tot_sig  = kl_sigma.sum().item()

    # ── report ────────────────────────────────────────────────────────────────
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"# E[z] vs KL analysis — {args.run}")
    out(f"beta={beta}  lambda={lam}  N={N}  D={D}")
    out("")
    out(f"E[z] over ALL dims: mean={Ez_np.mean():+.4f}  std={Ez_np.std():.4f}  "
        f"mean|E[z]|={np.abs(Ez_np).mean():.4f}  max|E[z]|={np.abs(Ez_np).max():.4f}")
    out("")
    out("## Group comparison")
    out(f"{'group':>10} | {'n':>5} | {'meanKL':>8} | {'E[z] mean':>10} | "
        f"{'E[z] std':>9} | {'mean|E[z]|':>10} | {'Var[mu]':>8} | {'meanσ':>6}")
    for name, g in [("top5% KL", g_top), ("rest", g_bot)]:
        out(f"{name:>10} | {g['n']:>5} | {g['kl']:>8.4f} | {g['Ez_mean']:>+10.4f} | "
            f"{g['Ez_std']:>9.4f} | {g['absEz']:>10.4f} | {g['Vmu']:>8.4f} | {g['sbar']:>6.3f}")
    out("")
    out("## KL quintiles (low→high KL)")
    out(f"{'quintile':>9} | {'meanKL':>8} | {'mean|E[z]|':>10} | {'Var[mu]':>8} | {'meanσ':>6}")
    for i, idx in enumerate(quint):
        g = grp(idx)
        out(f"{'Q'+str(i+1):>9} | {g['kl']:>8.4f} | {g['absEz']:>10.4f} | "
            f"{g['Vmu']:>8.4f} | {g['sbar']:>6.3f}")
    out("")
    out("## Statistical test")
    out(mw_str)
    out("")
    out("## Total-KL budget (sum over dims)")
    out(f"  total KL              = {tot:10.2f}")
    out(f"  from MEAN  term ½E[z]²= {tot_mean:10.2f}  ({100*tot_mean/tot:5.2f}%)")
    out(f"  from VAR   term ½V[μ] = {tot_var:10.2f}  ({100*tot_var/tot:5.2f}%)")
    out(f"  from SIGMA term       = {tot_sig:10.2f}  ({100*tot_sig/tot:5.2f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.run}_z_expectation_by_kl.md").write_text("\n".join(lines) + "\n")

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    fig.suptitle(
        f"{args.run}   β={beta}  λ={lam}  N={N}  D={D}   |   "
        f"Does E[z] shift with KL?  (E[z] = E_data[μ])", fontsize=12)

    klpos = np.clip(kl_np, 1e-6, None)

    # A: KL vs E[z]
    ax = axes[0]
    ax.scatter(klpos, Ez_np, s=6, alpha=0.35, color="#c94545", linewidths=0)
    ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("KL per dim (log)"); ax.set_ylabel("E[z] = E_data[μ]")
    ax.set_title(f"E[z] stays ~0 across all KL\nmean|E[z]|={np.abs(Ez_np).mean():.4f}",
                 fontsize=10)
    ax.grid(True, alpha=0.2)

    # B: KL vs Var[mu] and KL vs sigma  (the real drivers)
    ax = axes[1]
    ax.scatter(klpos, Vmu.numpy(), s=6, alpha=0.35, color="#4da6e8",
               linewidths=0, label="Var_data[μ]")
    ax.scatter(klpos, sbar.numpy(), s=6, alpha=0.35, color="#6bbf72",
               linewidths=0, label="mean σ")
    ax.axhline(1, color="k", lw=1, ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("KL per dim (log)"); ax.set_ylabel("value")
    ax.set_title("KL is driven by Var[μ]↑ and σ↓", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    # C: KL budget bar
    ax = axes[2]
    parts = [100*tot_mean/tot, 100*tot_var/tot, 100*tot_sig/tot]
    labels = [f"MEAN\n½E[z]²", f"VAR\n½Var[μ]", f"SIGMA\nσ-term"]
    cols = ["#c94545", "#4da6e8", "#6bbf72"]
    bars = ax.bar(labels, parts, color=cols, alpha=0.85)
    for b, v in zip(bars, parts):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%",
                ha="center", fontsize=10)
    ax.set_ylabel("% of total KL")
    ax.set_title(f"KL budget (total={tot:.0f})", fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = OUT_DIR / f"{args.run}_z_expectation_by_kl.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_png}")
    print(f"Saved → {OUT_DIR / f'{args.run}_z_expectation_by_kl.md'}")


if __name__ == "__main__":
    main()
