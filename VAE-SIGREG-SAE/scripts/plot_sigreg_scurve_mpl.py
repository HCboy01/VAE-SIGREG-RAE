#!/usr/bin/env python3
"""Publication-style SIGReg S-curve plots with matplotlib/seaborn."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualizations" / "sigreg_sweeps"


DATA = [
    ("1e-4", 10, 3072, 0.0215),
    ("1e-4", 15, 3071.5, 0.0206),
    ("1e-4", 20, 3070.25, 0.0194),
    ("1e-4", 25, 3062.75, 0.0191),
    ("1e-4", 30, 3005.5, 0.0189),
    ("1e-4", 35, 2924, 0.0196),
    ("1e-4", 40, 2761, 0.0198),
    ("1e-4", 45, 2447.75, 0.0191),
    ("1e-4", 50, 2122, 0.0191),
    ("1e-4", 100, 575.5, 0.0190),
    ("1e-4", 200, 253.5, 0.0182),
    ("1e-4", 500, 172.5, 0.0178),
    ("1e-4", 1000, 183.25, 0.0180),
    ("1e-4", 2000, 147, 0.0180),
    ("3e-4", 0.01, 3071.5, 0.0202),
    ("3e-4", 0.03, 3072, 0.0197),
    ("3e-4", 0.05, 3072, 0.0196),
    ("3e-4", 0.1, 3072, 0.0195),
    ("3e-4", 0.2, 3072, 0.0191),
    ("3e-4", 0.5, 3072, 0.0202),
    ("3e-4", 1, 3070.5, 0.0192),
    ("3e-4", 2, 3071, 0.0197),
    ("3e-4", 5, 3072, 0.0195),
    ("3e-4", 10, 3070.5, 0.0191),
    ("3e-4", 20, 3063, 0.0189),
    ("3e-4", 50, 2894, 0.0188),
    ("3e-4", 100, 1940.75, 0.0187),
    ("3e-4", 200, 120.75, 0.0178),
    ("3e-4", 500, 64, 0.0175),
    ("1e-3", 1, 2925, 0.0183),
    ("1e-3", 2, 2940.25, 0.0182),
    ("1e-3", 5, 2899, 0.0182),
    ("1e-3", 10, 2942.75, 0.0183),
    ("1e-3", 20, 2904.75, 0.0185),
    ("1e-3", 50, 2651, 0.0187),
    ("1e-3", 100, 2413.75, 0.0182),
    ("1e-3", 200, 2028, 0.0188),
    ("3e-3", 0, 786, 0.0181),
    ("3e-3", 0.001, 798.75, 0.0179),
    ("3e-3", 0.003, 721.75, 0.0178),
    ("3e-3", 0.005, 783.75, 0.0180),
    ("3e-3", 0.01, 758, 0.0178),
    ("3e-3", 0.02, 767.75, 0.0180),
    ("3e-3", 0.03, 776.25, 0.0175),
    ("3e-3", 0.05, 728.25, 0.0180),
]

PALETTE = {
    "1e-4": "#ef3b2c",
    "3e-4": "#fb6a4a",
    "1e-3": "#fc9272",
    "3e-3": "#fcbba1",
}


def lambda_tick(x: float, _: int) -> str:
    if x >= 1:
        return f"{x:g}"
    if x >= 0.01:
        return f"{x:.2g}"
    return f"{x:.1g}"


def load_frame() -> pd.DataFrame:
    df = pd.DataFrame(DATA, columns=["beta_kl", "lambda_sigreg", "active_units", "cov_offdiag"])
    # Log-scale plotting cannot show zero. Put lambda=0 just left of the first positive point.
    min_positive = df.loc[df["lambda_sigreg"] > 0, "lambda_sigreg"].min()
    df["lambda_plot"] = df["lambda_sigreg"].mask(df["lambda_sigreg"] == 0, min_positive / 3)
    df["active_pct"] = df["active_units"] / 3072 * 100
    return df


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=24, fontweight="bold", labelpad=12)
    ax.set_ylabel(ylabel, fontsize=24, fontweight="bold", labelpad=14)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.8, length=6)
    ax.tick_params(axis="both", which="minor", width=1.2, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("#2f2f2f")
    ax.grid(axis="y", linestyle="--", linewidth=1.0, color="#9aa0a6", alpha=0.7)
    ax.grid(axis="x", visible=False)


def plot_active_scurve(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=220)
    sns.lineplot(
        data=df,
        x="lambda_plot",
        y="active_pct",
        hue="beta_kl",
        palette=PALETTE,
        marker="o",
        markersize=12,
        linewidth=4.5,
        alpha=0.92,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    style_axes(ax, r"$\lambda_{\mathrm{SIGReg}}$", "Active Units (%)")
    legend = ax.legend(
        title=r"$\beta_{\mathrm{KL}}$",
        loc="upper right",
        frameon=True,
        fontsize=18,
        title_fontsize=18,
        borderpad=0.6,
        handlelength=1.6,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_edgecolor("#c7c7c7")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"active_scurve_mpl.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_active_scurve_lambda_1_100(df: pd.DataFrame) -> None:
    view = df[(df["beta_kl"] != "3e-3") & (df["lambda_sigreg"] >= 1) & (df["lambda_sigreg"] <= 100)].copy()
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=220)
    sns.lineplot(
        data=view,
        x="lambda_sigreg",
        y="active_pct",
        hue="beta_kl",
        palette={k: PALETTE[k] for k in ["1e-4", "3e-4", "1e-3"]},
        marker="o",
        markersize=13,
        linewidth=4.8,
        alpha=0.92,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 104)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 25, 50, 75, 100])
    style_axes(ax, r"$\lambda_{\mathrm{SIGReg}}$", "Active Units (%)")
    legend = ax.legend(
        title=r"$\beta_{\mathrm{KL}}$",
        loc="lower left",
        frameon=True,
        fontsize=18,
        title_fontsize=18,
        borderpad=0.6,
        handlelength=1.6,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_edgecolor("#c7c7c7")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"active_scurve_lambda_1_100_mpl.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_active_scurve_lambda_1_1000(df: pd.DataFrame) -> None:
    view = df[(df["beta_kl"] != "3e-3") & (df["lambda_sigreg"] >= 1) & (df["lambda_sigreg"] <= 1000)].copy()
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=220)
    sns.lineplot(
        data=view,
        x="lambda_sigreg",
        y="active_pct",
        hue="beta_kl",
        palette={k: PALETTE[k] for k in ["1e-4", "3e-4", "1e-3"]},
        marker="o",
        markersize=13,
        linewidth=4.8,
        alpha=0.92,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 104)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 25, 50, 75, 100])
    style_axes(ax, r"$\lambda_{\mathrm{SIGReg}}$", "Active Units (%)")
    legend = ax.legend(
        title=r"$\beta_{\mathrm{KL}}$",
        loc="upper right",
        frameon=True,
        fontsize=18,
        title_fontsize=18,
        borderpad=0.6,
        handlelength=1.6,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_edgecolor("#c7c7c7")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"active_scurve_lambda_1_1000_mpl.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_cov_active(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=220)
    sns.lineplot(
        data=df,
        x="active_pct",
        y="cov_offdiag",
        hue="beta_kl",
        palette=PALETTE,
        marker="o",
        markersize=12,
        linewidth=4.0,
        alpha=0.9,
        ax=ax,
    )
    ax.set_xlim(-2, 104)
    ax.set_ylim(0.0172, 0.0218)
    ax.set_xticks([0, 25, 50, 75, 100])
    style_axes(ax, "Active Units (%)", "Off-diagonal Cov.")
    legend = ax.legend(
        title=r"$\beta_{\mathrm{KL}}$",
        loc="upper right",
        frameon=True,
        fontsize=18,
        title_fontsize=18,
        borderpad=0.6,
        handlelength=1.6,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_edgecolor("#c7c7c7")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"covariance_vs_active_mpl.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    df = load_frame()
    df.to_csv(OUT / "sigreg_scurve_mpl_data.csv", index=False)
    plot_active_scurve(df)
    plot_active_scurve_lambda_1_100(df)
    plot_active_scurve_lambda_1_1000(df)
    plot_cov_active(df)
    print(f"Wrote figures under {OUT}")


if __name__ == "__main__":
    main()
