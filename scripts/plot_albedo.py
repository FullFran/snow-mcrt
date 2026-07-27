#!/usr/bin/env python3
"""Render figures from the committed reference CSVs.

This script runs no physics. It reads what ``run_albedo.py`` wrote and draws
it, which is the whole point of the split: the numbers are an artefact that
outlives any particular plot, and a figure can be restyled a dozen times
without anyone waiting on a Monte Carlo.

Usage::

    python scripts/plot_albedo.py --input data/reference --output docs/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display required; this must run in a batch job.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GRAIN_SERIES = [
    ("pure-r50um", "50 μm"),
    ("pure-r100um", "100 μm"),
    ("pure-r250um", "250 μm"),
    ("pure-r500um", "500 μm"),
    ("pure-r1000um", "1000 μm"),
]

CARBON_SERIES = [
    ("bc0ngg-r100um", "clean"),
    ("bc1ngg-r100um", "1 ng/g"),
    ("bc10ngg-r100um", "10 ng/g"),
    ("bc100ngg-r100um", "100 ng/g"),
    ("bc1000ngg-r100um", "1000 ng/g"),
]

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#d8d8de"


def load(path: Path) -> dict[str, np.ndarray]:
    """Read a reference CSV into named columns."""
    with path.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader])
    return {name: rows[:, i] for i, name in enumerate(header)}


def style_axes(ax, xlabel: str, ylabel: str, title: str, subtitle: str = "") -> None:
    """Consistent look: light grid, no top or right spine, wavelength in nm."""
    ax.set_xscale("log")
    ax.set_xlim(300, 2500)
    ax.set_xticks([300, 400, 600, 800, 1000, 1500, 2000, 2500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=26)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=9,
            va="bottom",
        )


def series_colours(n: int, cmap: str) -> list:
    return [plt.get_cmap(cmap)(v) for v in np.linspace(0.15, 0.88, n)]


def plot_series(ax, data_dir: Path, series, column: str, cmap: str) -> None:
    colours = series_colours(len(series), cmap)
    for (stem, label), colour in zip(series, colours):
        path = data_dir / f"{stem}.csv"
        if not path.exists():
            continue
        columns = load(path)
        ax.plot(
            columns["wavelength_nm"],
            columns[column],
            label=label,
            color=colour,
            linewidth=2.0,
        )


def figure_grain_size(data_dir: Path, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    plot_series(ax, data_dir, GRAIN_SERIES, "albedo", "viridis")
    style_axes(
        ax,
        "wavelength (nm)",
        "spectral albedo",
        "Spectral albedo of pure snow",
        "semi-infinite pack · 300 kg/m³ · Warren & Brandt (2008) ice constants",
    )
    ax.set_ylim(0, 1.02)
    ax.legend(title="grain radius", frameon=False, fontsize=9, title_fontsize=9)
    ax.annotate(
        "visible: nearly independent of grain size",
        xy=(430, 0.985),
        xytext=(330, 0.72),
        color=MUTED,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    ax.annotate(
        "near infrared: the grain-size signal",
        xy=(1300, 0.42),
        xytext=(1350, 0.72),
        color=MUTED,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_black_carbon(data_dir: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)

    plot_series(axes[0], data_dir, CARBON_SERIES, "albedo", "inferno")
    style_axes(
        axes[0],
        "wavelength (nm)",
        "spectral albedo",
        "Black carbon in snow",
        "100 μm grains · 300 kg/m³",
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(title="mixing ratio", frameon=False, fontsize=9, title_fontsize=9)

    # The visible detail is where the whole effect lives, and it is invisible
    # on a 0-1 axis.
    colours = series_colours(len(CARBON_SERIES), "inferno")
    for (stem, label), colour in zip(CARBON_SERIES, colours):
        path = data_dir / f"{stem}.csv"
        if not path.exists():
            continue
        columns = load(path)
        axes[1].plot(
            columns["wavelength_nm"],
            columns["albedo"],
            label=label,
            color=colour,
            linewidth=2.0,
        )
    style_axes(
        axes[1],
        "wavelength (nm)",
        "spectral albedo",
        "The visible, magnified",
        "one part per billion is already visible",
    )
    axes[1].set_xlim(300, 900)
    axes[1].set_xticks([300, 400, 500, 600, 700, 800, 900])
    axes[1].set_ylim(0.85, 1.005)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_penetration(data_dir: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)

    plot_series(axes[0], data_dir, GRAIN_SERIES, "co_albedo", "viridis")
    axes[0].set_yscale("log")
    style_axes(
        axes[0],
        "wavelength (nm)",
        "1 − ω  (co-albedo)",
        "What actually varies",
        "ω itself sits within a millionth of unity across the visible",
    )
    axes[0].legend(title="grain radius", frameon=False, fontsize=9, title_fontsize=9)

    plot_series(axes[0 + 1], data_dir, GRAIN_SERIES, "e_folding_depth_m", "viridis")
    axes[1].set_yscale("log")
    style_axes(
        axes[1],
        "wavelength (nm)",
        "e-folding depth (m)",
        "How deep light reaches",
        "the depth over which diffuse flux falls by 1/e",
    )
    axes[1].axhline(0.01, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1].text(310, 0.0115, "1 cm", color=MUTED, fontsize=8)
    axes[1].axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1].text(310, 1.15, "1 m", color=MUTED, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_ice_constants(constants_path: Path, out: Path) -> Path:
    table = np.loadtxt(constants_path)
    lam = table[:, 0] * 1000.0
    window = (lam >= 300) & (lam <= 2500)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    ax.plot(lam[window], table[window, 2], color="#2a628f", linewidth=2.0)
    ax.set_yscale("log")
    style_axes(
        ax,
        "wavelength (nm)",
        "k  (imaginary refractive index)",
        "Absorption of pure ice",
        "Warren & Brandt (2008) · six orders of magnitude across the solar spectrum",
    )
    # Below 390 nm the tabulated value is a reported upper limit, not a
    # measurement. Saying so on the figure keeps the caveat attached to it.
    ax.axvspan(300, 390, color="#f0a202", alpha=0.13)
    ax.text(
        305,
        table[window, 2].max() * 0.3,
        "reported upper limit,\nnot a measurement",
        color="#a06a00",
        fontsize=8,
        va="top",
    )
    ax.annotate(
        "measured minimum\n400 nm, k = 2.4e-11",
        xy=(400, 2.365e-11),
        xytext=(480, 3e-10),
        color=MUTED,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reference"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--constants",
        type=Path,
        default=Path("data/ice/warren_brandt_2008.dat"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    written = [
        figure_grain_size(args.input, args.output / "spectral-albedo-grain-size.png"),
        figure_black_carbon(args.input, args.output / "black-carbon.png"),
        figure_penetration(args.input, args.output / "co-albedo-and-penetration.png"),
        figure_ice_constants(args.constants, args.output / "ice-absorption.png"),
    ]
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
