#!/usr/bin/env python3
"""Bar chart: Total Correlation of mu, SIGReg vs no-SIGReg per beta."""

from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_run import load_embeddings
from src.vae_sigreg import OvercompleteVariationalAE

OUT = ROOT / "visualizations" / "paper_figures"
VAL_PATH = "/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin"

BETAS = ["1e-4", "3e-4", "1e-3"]
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
N_DIMS = 256   # active dim subset for numerical stability


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = OvercompleteVariationalAE(
        input_dim=a.get("input_dim", 768),
        latent_dim=a.get("latent_dim", 3072),
        hidden_dim=a.get("hidden_dim"),
        num_layers=a.get("num_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def encode_mu(model, embeddings: torch.Tensor, device: torch.device, batch_size: int = 2048):
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for (x,) in loader:
        mu, _ = model.encode(x.to(device))
        chunks.append(mu.float().cpu())
    return torch.cat(chunks, dim=0)


def total_correlation(mu: torch.Tensor, n_dims: int = N_DIMS) -> float:
    """
    Gaussian TC on top-n_dims active dimensions of mu.
    TC = 0.5 * (Σ_j log σ_j² - log det Σ)
       = 0.5 * log(Π_j σ_j² / det Σ)
    """
    active = mu.abs().gt(0.1).any(dim=0)
    std = mu.std(dim=0)
    candidate = torch.where(active, std, torch.tensor(-1.0))
    ids = candidate.argsort(descending=True)[:n_dims]
    X = mu[:, ids].double()

    cov = torch.cov(X.T)                           # [n_dims, n_dims]
    log_diag_sum = cov.diag().log().sum()           # Σ log σ_j²
    sign, log_det = torch.linalg.slogdet(cov)      # log det Σ
    if sign <= 0:
        return float("nan")

    tc = 0.5 * (log_diag_sum - log_det).item()
    return max(tc, 0.0)    # numerical floor at 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = load_embeddings(VAL_PATH)

    tc_with: dict[str, float] = {}
    tc_null: dict[str, float] = {}

    for beta in BETAS:
        print(f"beta={beta} ...", end=" ", flush=True)

        model = load_model(BEST[beta]["ckpt"], device)
        mu = encode_mu(model, val, device)
        tc_with[beta] = total_correlation(mu)
        del model, mu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = load_model(NULL[beta]["ckpt"], device)
        mu = encode_mu(model, val, device)
        tc_null[beta] = total_correlation(mu)
        del model, mu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"TC(SIGReg)={tc_with[beta]:.2f}  TC(null)={tc_null[beta]:.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=220)

    x = np.arange(len(BETAS))
    w = 0.35

    for i, beta in enumerate(BETAS):
        color = PALETTE[beta]
        lam_with = BEST[beta]["lambda"]
        ax.bar(x[i] - w/2, tc_with[beta], width=w, color=color, alpha=0.9,
               label=rf"$\lambda={lam_with}$ (SIGReg)" if i == 0 else "_")
        ax.bar(x[i] + w/2, tc_null[beta], width=w, color=color, alpha=0.35,
               hatch="//", edgecolor=color,
               label=r"$\lambda=0$ (no SIGReg)" if i == 0 else "_")
        # value labels
        for val_tc, offset in [(tc_with[beta], -w/2), (tc_null[beta], w/2)]:
            ax.text(x[i] + offset, val_tc + 0.3, f"{val_tc:.1f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\beta={b}$" for b in BETAS], fontsize=13)
    ax.set_ylabel(f"Total Correlation (top-{N_DIMS} active dims, μ)", fontsize=13)
    ax.set_title("Total Correlation: SIGReg vs no SIGReg", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # unified legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="gray", alpha=0.9, label="SIGReg (best λ)"),
        Patch(facecolor="gray", alpha=0.35, hatch="//", edgecolor="gray", label="no SIGReg (λ=0)"),
    ]
    ax.legend(handles=handles, fontsize=11, frameon=True)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        out = OUT / f"fig_total_correlation.{ext}"
        fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUT}/fig_total_correlation.png")


if __name__ == "__main__":
    main()
