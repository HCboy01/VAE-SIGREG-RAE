#!/usr/bin/env python3
"""Build paper figures from Stage 5 200-epoch SIGReg sweeps."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader, TensorDataset

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_run import load_embeddings  # noqa: E402
from src.vae_sigreg import OvercompleteVariationalAE  # noqa: E402


OUT = ROOT / "visualizations" / "paper_figures"
VAL_PATH = "/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin"
BETAS = ["1e-4", "3e-4", "1e-3"]
SHARED_GRID = [0, 1, 3, 10, 30, 100, 300, 1000]
BEST = {
    "1e-4": {"lambda": 3,  "ckpt": ROOT / "best_ckpt/beta_1e-4_lambda_3_best.pt"},
    "3e-4": {"lambda": 3,  "ckpt": ROOT / "best_ckpt/beta_3e-4_lambda_3_best.pt"},
    "1e-3": {"lambda": 10, "ckpt": ROOT / "best_ckpt/beta_1e-3_lambda_10_best.pt"},
}
NULL = {
    "1e-4": {"lambda": 0, "ckpt": ROOT / "best_ckpt/beta_1e-4_lambda_0_best.pt"},
    "3e-4": {"lambda": 0, "ckpt": ROOT / "best_ckpt/beta_3e-4_lambda_0_best.pt"},
    "1e-3": {"lambda": 0, "ckpt": ROOT / "best_ckpt/beta_1e-3_lambda_0_best.pt"},
}
PALETTE = {"1e-4": "#2166ac", "3e-4": "#1a9850", "1e-3": "#d6604d"}


def arg_value(config_text: str, name: str) -> str | None:
    m = re.search(r"- --" + re.escape(name) + r'\n\s+- "?([^"\n]+)"?', config_text)
    return m.group(1) if m else None


def load_stage5_summaries() -> pd.DataFrame:
    rows = []
    for cfg in (ROOT / "wandb").glob("run-20260504_*/files/config.yaml"):
        text = cfg.read_text(errors="ignore")
        if "stage5" not in text or "200epoch" not in text:
            continue
        beta = arg_value(text, "beta_kl")
        lam = arg_value(text, "lambda_sigreg")
        ckpt_dir = arg_value(text, "ckpt_dir")
        run_name = arg_value(text, "wandb_run_name")
        if beta not in BETAS or lam is None or ckpt_dir is None:
            continue
        lam_f = float(lam)
        if lam_f not in SHARED_GRID:
            continue
        summary_path = cfg.with_name("wandb-summary.json")
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text())
        if int(data.get("epoch", -1)) != 200:
            continue
        rows.append(
            {
                "beta_kl": beta,
                "lambda_sigreg": lam_f,
                "run_name": run_name,
                "ckpt_dir": ckpt_dir,
                "rec": data.get("val/rec_loss"),
                "kl": data.get("val/kl_loss"),
                "offdiag": data.get("val/covariance_offdiag_error"),
                "active_0_1": data.get("val/ever_active_units_thr_0_1"),
                "active_0_5": data.get("val/ever_active_units_thr_0_5"),
                "total": data.get("val/total_loss"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Stage 5 summaries found.")
    # If duplicate beta/lambda runs exist, keep the newest ckpt timestamp lexicographically.
    df["ckpt_stamp"] = df["ckpt_dir"].str.extract(r"_(\d{8}_\d{6})$")
    df = df.sort_values("ckpt_stamp").drop_duplicates(["beta_kl", "lambda_sigreg"], keep="last")
    return df.sort_values(["beta_kl", "lambda_sigreg"])


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    model = OvercompleteVariationalAE(
        input_dim=args.get("input_dim", 768),
        latent_dim=args.get("latent_dim", 3072),
        hidden_dim=args.get("hidden_dim"),
        num_layers=args.get("num_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def encode_mu(model, embeddings: torch.Tensor, device: torch.device, batch_size: int = 2048) -> torch.Tensor:
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for (x,) in loader:
        mu, _ = model.encode(x.to(device))
        chunks.append(mu.float().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_mu_logvar(model, embeddings: torch.Tensor, device: torch.device, batch_size: int = 2048):
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False, num_workers=0)
    mus, logvars = [], []
    for (x,) in loader:
        mu, logvar = model.encode(x.to(device))
        mus.append(mu.float().cpu())
        logvars.append(logvar.float().cpu())
    return torch.cat(mus, dim=0), torch.cat(logvars, dim=0)


def posthoc_active_for_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df[df["active_0_1"].isna() | df["active_0_5"].isna()].copy()
    if missing.empty:
        return df
    cache_path = OUT / "stage5_posthoc_active.csv"
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    cache_keys = set()
    if not cache.empty:
        cache_keys = set(zip(cache["ckpt_dir"], cache["lambda_sigreg"]))

    val = load_embeddings(VAL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_rows = []
    for _, row in missing.iterrows():
        key = (row["ckpt_dir"], row["lambda_sigreg"])
        if key in cache_keys:
            continue
        ckpt = ROOT / row["ckpt_dir"] / "best.pt"
        if not ckpt.exists():
            continue
        model = load_model(ckpt, device)
        mu = encode_mu(model, val, device)
        active01 = int(mu.abs().gt(0.1).any(dim=0).sum().item())
        active05 = int(mu.abs().gt(0.5).any(dim=0).sum().item())
        new_rows.append({"ckpt_dir": row["ckpt_dir"], "lambda_sigreg": row["lambda_sigreg"], "active_0_1": active01, "active_0_5": active05})
        del model, mu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    if not cache.empty:
        df = df.merge(cache, on=["ckpt_dir", "lambda_sigreg"], how="left", suffixes=("", "_post"))
        for col in ["active_0_1", "active_0_5"]:
            df[col] = df[col].fillna(df[f"{col}_post"])
            df = df.drop(columns=[f"{col}_post"])
    return df


def style_ax(ax):
    ax.grid(axis="y", linestyle="--", color="#9aa0a6", alpha=0.65)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def lambda_tick(x, _):
    if x >= 1:
        return f"{x:g}"
    return "0"


def plot_active(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=220)
    plot_df = df.copy()
    plot_df["lambda_plot"] = plot_df["lambda_sigreg"].replace(0, 0.3)
    for beta in BETAS:
        sub = plot_df[plot_df["beta_kl"] == beta].sort_values("lambda_plot")
        ax.plot(sub["lambda_plot"], sub["active_0_1"], marker="o", linewidth=3.5, markersize=8, color=PALETTE[beta], label=rf"$\beta={beta}$, thr=0.1")
        ax.plot(sub["lambda_plot"], sub["active_0_5"], marker="s", linewidth=2.4, markersize=6, linestyle="--", color=PALETTE[beta], alpha=0.65, label=rf"$\beta={beta}$, thr=0.5")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 1200)
    ax.set_ylim(-80, 3200)
    ax.set_xticks([0.3, 1, 3, 10, 30, 100, 300, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda_tick))
    ax.set_xlabel(r"$\lambda_{\mathrm{SIGReg}}$ (log scale)", fontsize=17, fontweight="bold")
    ax.set_ylabel("Active neurons", fontsize=17, fontweight="bold")
    ax.tick_params(labelsize=13)
    style_ax(ax)
    ax.legend(fontsize=9, ncol=2, frameon=True,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(OUT / "fig2a_active_vs_lambda.png", bbox_inches="tight")
    fig.savefig(OUT / "fig2a_active_vs_lambda.pdf", bbox_inches="tight")
    plt.close(fig)


def _select_ids(ref: torch.Tensor, n: int = 64) -> torch.Tensor:
    """Top-n dims by std of ref tensor (any dtype/target)."""
    return ref.std(dim=0).argsort(descending=True)[:n]


def _corr_matrix(x: torch.Tensor, ids: torch.Tensor) -> np.ndarray:
    """Pearson correlation matrix for the given dim indices."""
    X = x[:, ids].float()
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
    return (X.T @ X / max(X.shape[0] - 1, 1)).detach().numpy()


def _encode_z(model, embeddings: torch.Tensor, device: torch.device,
              gen: torch.Generator, batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, z_sampled) for all embeddings."""
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False, num_workers=0)
    mus, zs = [], []
    for (x,) in loader:
        mu, logvar = model.encode(x.to(device))
        mu, logvar = mu.float().cpu(), logvar.float().cpu()
        z = mu + (0.5 * logvar).exp() * torch.randn(mu.shape, generator=gen)
        mus.append(mu); zs.append(z)
    return torch.cat(mus), torch.cat(zs)


def _plot_corr_heatmaps_impl(target: str, fname_stem: str):
    """
    target: 'mu' or 'z'
    Dim selection is always based on mu of the SIGReg model.
    Correlation is computed on `target`.
    """
    assert target in ("mu", "z")
    val = load_embeddings(VAL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(42)

    null_available = all(NULL[b]["ckpt"].exists() for b in BETAS)
    n_rows = 2 if null_available else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, 4.2 * n_rows), dpi=220)
    if n_rows == 1:
        axes = axes[None, :]

    for col, beta in enumerate(BETAS):
        null_loaded = null_available and NULL[beta]["ckpt"].exists()

        # Encode both models
        model = load_model(BEST[beta]["ckpt"], device)
        mu_best, z_best = _encode_z(model, val, device, gen)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if null_loaded:
            model = load_model(NULL[beta]["ckpt"], device)
            mu_null, z_null = _encode_z(model, val, device, gen)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Select dims by null model's target (z or mu) std
            ref_null_for_sel = mu_null if target == "mu" else z_null
            ids = _select_ids(ref_null_for_sel)
        else:
            ref_for_sel = mu_best if target == "mu" else z_best
            ids = _select_ids(ref_for_sel)
            mu_null = z_null = None

        vmax = 0.15 if target == "z" else 1.0

        def _annotate(ax, corr):
            off = corr[~np.eye(corr.shape[0], dtype=bool)]
            ax.text(0.97, 0.03, f"mean|r|={np.abs(off).mean():.3f}",
                    transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ref_best = mu_best if target == "mu" else z_best
        corr_best = _corr_matrix(ref_best, ids)
        del mu_best, z_best, ref_best

        sns.heatmap(corr_best, ax=axes[0, col], cmap="vlag", vmin=-vmax, vmax=vmax, center=0,
                    cbar=(col == 2), xticklabels=False, yticklabels=False, square=True)
        _annotate(axes[0, col], corr_best)
        axes[0, col].set_title(
            rf"$\beta={beta}$, $\lambda={BEST[beta]['lambda']}$",
            fontsize=11, fontweight="bold",
        )

        if null_loaded:
            ref_null = mu_null if target == "mu" else z_null
            corr_null = _corr_matrix(ref_null, ids)
            del mu_null, z_null, ref_null

            sns.heatmap(corr_null, ax=axes[1, col], cmap="vlag", vmin=-vmax, vmax=vmax, center=0,
                        cbar=(col == 2), xticklabels=False, yticklabels=False, square=True)
            _annotate(axes[1, col], corr_null)
            axes[1, col].set_title(
                rf"$\beta={beta}$, $\lambda=0$ (no SIGReg)",
                fontsize=11, fontweight="bold",
            )

    if not null_available:
        print("WARNING: λ=0 checkpoints not found; skipping second row.")

    axes[0, 0].set_ylabel("SIGReg", fontsize=12, fontweight="bold")
    if null_available:
        axes[1, 0].set_ylabel("no SIGReg\n(same dims)", fontsize=12, fontweight="bold")

    fig.suptitle(f"Correlation heatmap — target: {target}", fontsize=13, y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}/{fname_stem}.png")


def plot_corr_heatmaps():
    _plot_corr_heatmaps_impl(target="mu", fname_stem="fig2b_active_neuron_corr_heatmap_mu")
    _plot_corr_heatmaps_impl(target="z",  fname_stem="fig2b_active_neuron_corr_heatmap_z")


def parse_rec_curve(log_path: Path) -> pd.DataFrame:
    rows = []
    pat = re.compile(r"^\[(\d+)/200\].*?rec=([0-9.]+).*?kl=([0-9.]+).*?sig=([0-9.]+).*?total=([0-9.]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            rows.append({"epoch": int(m.group(1)), "rec": float(m.group(2)), "kl": float(m.group(3)), "sig": float(m.group(4)), "total": float(m.group(5))})
    return pd.DataFrame(rows)


def plot_recon_curves():
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=220)
    ax2 = ax.twinx()
    log_paths = {
        "1e-4": ROOT / "logs/s5_b1e_4_sig_3_200ep_20260504_171413.log",
        "3e-4": ROOT / "logs/s5_b3e_4_sig_3_200ep_20260504_173157.log",
        "1e-3": ROOT / "logs/s5_b1e_3_sig_10_200ep_20260504_175617.log",
    }
    curves = {}
    for beta, path in log_paths.items():
        curve = parse_rec_curve(path)
        curves[beta] = curve
        ax.plot(curve["epoch"], curve["rec"], linewidth=3.0, color=PALETTE[beta], label=rf"$\beta={beta}$, $\lambda={BEST[beta]['lambda']}$")
    for beta, curve in curves.items():
        ax2.plot(curve["epoch"], curve["kl"], linewidth=1.8, color=PALETTE[beta], linestyle="--", alpha=0.6)
    ax.set_xlabel("Epoch", fontsize=17, fontweight="bold")
    ax.set_ylabel("Validation recon loss", fontsize=17, fontweight="bold")
    ax2.set_ylabel("Validation KL loss", fontsize=15, fontweight="bold", color="#555555")
    ax2.tick_params(labelsize=11, colors="#555555")
    ax2.spines["right"].set_linewidth(1.4)
    ax.tick_params(labelsize=13)
    style_ax(ax)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, fontsize=12, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "fig2c_recon_vs_epoch_best.png", bbox_inches="tight")
    fig.savefig(OUT / "fig2c_recon_vs_epoch_best.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    df = posthoc_active_for_missing(load_stage5_summaries())
    df.to_csv(OUT / "stage5_summary_for_figures.csv", index=False)
    plot_active(df)
    plot_corr_heatmaps()
    plot_recon_curves()
    print(f"Wrote Figure 2 panels to {OUT}")


if __name__ == "__main__":
    main()
