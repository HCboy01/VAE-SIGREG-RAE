#!/usr/bin/env python3
"""Generate SVG figures for SIGReg lambda sweeps without plotting dependencies."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualizations" / "sigreg_sweeps"


SERIES = {
    "beta=1e-4": {
        "color": "#D55E00",
        "points": [
            (10, 3072, 0.0215),
            (15, 3071.5, 0.0206),
            (20, 3070.25, 0.0194),
            (25, 3062.75, 0.0191),
            (30, 3005.5, 0.0189),
            (35, 2924, 0.0196),
            (40, 2761, 0.0198),
            (45, 2447.75, 0.0191),
            (50, 2122, 0.0191),
            (100, 575.5, 0.0190),
            (200, 253.5, 0.0182),
            (500, 172.5, 0.0178),
            (1000, 183.25, 0.0180),
            (2000, 147, 0.0180),
        ],
        "selected": 35,
    },
    "beta=3e-4": {
        "color": "#0072B2",
        "points": [
            (0.01, 3071.5, 0.0202),
            (0.03, 3072, 0.0197),
            (0.05, 3072, 0.0196),
            (0.1, 3072, 0.0195),
            (0.2, 3072, 0.0191),
            (0.5, 3072, 0.0202),
            (1, 3070.5, 0.0192),
            (2, 3071, 0.0197),
            (5, 3072, 0.0195),
            (10, 3070.5, 0.0191),
            (20, 3063, 0.0189),
            (50, 2894, 0.0188),
            (100, 1940.75, 0.0187),
            (200, 120.75, 0.0178),
            (500, 64, 0.0175),
        ],
        "selected": 20,
    },
    "beta=1e-3": {
        "color": "#009E73",
        "points": [
            (1, 2925, 0.0183),
            (2, 2940.25, 0.0182),
            (5, 2899, 0.0182),
            (10, 2942.75, 0.0183),
            (20, 2904.75, 0.0185),
            (50, 2651, 0.0187),
            (100, 2413.75, 0.0182),
            (200, 2028, 0.0188),
        ],
        "selected": 20,
    },
    "beta=3e-3": {
        "color": "#CC79A7",
        "points": [
            (0, 786, 0.0181),
            (0.001, 798.75, 0.0179),
            (0.003, 721.75, 0.0178),
            (0.005, 783.75, 0.0180),
            (0.01, 758, 0.0178),
            (0.02, 767.75, 0.0180),
            (0.03, 776.25, 0.0175),
            (0.05, 728.25, 0.0180),
        ],
        "selected": 0.001,
    },
}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_lambda(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:g}"


def lambda_scale():
    positive = [x for s in SERIES.values() for x, _, _ in s["points"] if x > 0]
    min_pos = min(positive)
    lo = math.log10(min_pos / 3)
    hi = math.log10(max(positive))

    def scale(x: float, left: float, right: float) -> float:
        lx = lo if x <= 0 else math.log10(x)
        return left + (lx - lo) / (hi - lo) * (right - left)

    return scale


def linear_scale(src_min: float, src_max: float, dst_min: float, dst_max: float):
    def scale(x: float) -> float:
        return dst_min + (x - src_min) / (src_max - src_min) * (dst_max - dst_min)

    return scale


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#202124}",
        ".title{font-size:24px;font-weight:700}.axis{font-size:14px}.tick{font-size:12px;fill:#5f6368}",
        ".note{font-size:13px;fill:#5f6368}.legend{font-size:13px}",
        ".grid{stroke:#e8eaed;stroke-width:1}.axisline{stroke:#3c4043;stroke-width:1.4}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def save_active_vs_lambda(path: Path) -> None:
    width, height = 1120, 700
    left, right, top, bottom = 92, 1038, 88, 590
    xscale = lambda_scale()
    yscale = linear_scale(0, 3200, bottom, top)
    ticks_x = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ticks_y = [0, 512, 1024, 1536, 2048, 2560, 3072]

    lines = svg_header(width, height)
    lines.append('<text class="title" x="92" y="42">Active units vs SIGReg strength</text>')
    lines.append('<text class="note" x="92" y="66">latent_dim=3072, FFHQ256 DINO CLS, EP-SIGReg sweeps</text>')

    for y in ticks_y:
        py = yscale(y)
        lines.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}"/>')
        lines.append(f'<text class="tick" x="{left-12}" y="{py+4:.1f}" text-anchor="end">{y:g}</text>')
    for x in ticks_x:
        px = xscale(x, left, right)
        lines.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}"/>')
        lines.append(f'<text class="tick" x="{px:.1f}" y="{bottom+24}" text-anchor="middle">{fmt_lambda(x)}</text>')

    lines.append(f'<line class="axisline" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    lines.append(f'<line class="axisline" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    lines.append(f'<text class="axis" x="{(left+right)/2:.1f}" y="{height-34}" text-anchor="middle">lambda_SIGReg, log scale</text>')
    lines.append(f'<text class="axis" x="24" y="{(top+bottom)/2:.1f}" transform="rotate(-90 24 {(top+bottom)/2:.1f})" text-anchor="middle">active units</text>')

    for name, spec in SERIES.items():
        color = spec["color"]
        pts = [(xscale(x, left, right), yscale(active)) for x, active, _ in spec["points"]]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        selected = spec["selected"]
        for lam, active, _ in spec["points"]:
            px = xscale(lam, left, right)
            py = yscale(active)
            radius = 6 if lam == selected else 4
            fill = "#ffffff" if lam == selected else color
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="2"/>')

    legend_x, legend_y = 800, 102
    for i, (name, spec) in enumerate(SERIES.items()):
        y = legend_y + i * 24
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+30}" y2="{y}" stroke="{spec["color"]}" stroke-width="3"/>')
        lines.append(f'<text class="legend" x="{legend_x+40}" y="{y+4}">{esc(name)} selected λ={fmt_lambda(spec["selected"])}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_cov_vs_active(path: Path) -> None:
    width, height = 1120, 700
    left, right, top, bottom = 92, 1038, 88, 590
    all_points = [(active, cov) for s in SERIES.values() for _, active, cov in s["points"]]
    xscale = linear_scale(0, 3200, left, right)
    yscale = linear_scale(0.0172, 0.0218, bottom, top)
    ticks_x = [0, 512, 1024, 1536, 2048, 2560, 3072]
    ticks_y = [0.0175, 0.0180, 0.0185, 0.0190, 0.0195, 0.0200, 0.0205, 0.0210, 0.0215]

    lines = svg_header(width, height)
    lines.append('<text class="title" x="92" y="42">Covariance error vs active units</text>')
    lines.append('<text class="note" x="92" y="66">Lower off-diagonal covariance error improves Gaussianity; left side indicates latent collapse.</text>')

    for y in ticks_y:
        py = yscale(y)
        lines.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}"/>')
        lines.append(f'<text class="tick" x="{left-12}" y="{py+4:.1f}" text-anchor="end">{y:.4f}</text>')
    for x in ticks_x:
        px = xscale(x)
        lines.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}"/>')
        lines.append(f'<text class="tick" x="{px:.1f}" y="{bottom+24}" text-anchor="middle">{x:g}</text>')

    lines.append(f'<line class="axisline" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    lines.append(f'<line class="axisline" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    lines.append(f'<text class="axis" x="{(left+right)/2:.1f}" y="{height-34}" text-anchor="middle">active units</text>')
    lines.append(f'<text class="axis" x="24" y="{(top+bottom)/2:.1f}" transform="rotate(-90 24 {(top+bottom)/2:.1f})" text-anchor="middle">covariance off-diagonal error</text>')

    for name, spec in SERIES.items():
        color = spec["color"]
        pts = [(xscale(active), yscale(cov)) for _, active, cov in spec["points"]]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.55"/>')
        selected = spec["selected"]
        for lam, active, cov in spec["points"]:
            px = xscale(active)
            py = yscale(cov)
            radius = 6 if lam == selected else 4
            fill = "#ffffff" if lam == selected else color
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="2"/>')
            if lam in (selected, 50, 100, 200):
                lines.append(f'<text class="tick" x="{px+7:.1f}" y="{py-7:.1f}">λ={fmt_lambda(lam)}</text>')

    legend_x, legend_y = 790, 102
    for i, (name, spec) in enumerate(SERIES.items()):
        y = legend_y + i * 24
        lines.append(f'<circle cx="{legend_x+12}" cy="{y}" r="5" fill="{spec["color"]}"/>')
        lines.append(f'<text class="legend" x="{legend_x+30}" y="{y+4}">{esc(name)}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["beta", "lambda_sigreg", "active_units", "cov_offdiag_error", "selected"])
        for name, spec in SERIES.items():
            for lam, active, cov in spec["points"]:
                writer.writerow([name.replace("beta=", ""), lam, active, cov, lam == spec["selected"]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUT / "sigreg_sweeps.csv")
    save_active_vs_lambda(OUT / "active_vs_lambda.svg")
    save_cov_vs_active(OUT / "covariance_vs_active.svg")
    print(f"Wrote {OUT / 'sigreg_sweeps.csv'}")
    print(f"Wrote {OUT / 'active_vs_lambda.svg'}")
    print(f"Wrote {OUT / 'covariance_vs_active.svg'}")


if __name__ == "__main__":
    main()
