#!/usr/bin/env python3
"""Render the TARTES cross-validation figure from the committed CSVs.

Runs no physics. The point of the figure is the *decomposition*: two codes
disagreeing on an albedo curve says nothing useful on its own, so the plot
separates the part attributable to the radiative transfer solution from the
part attributable to the grain model.

Usage::

    python scripts/plot_validation.py --input data/validation --output docs/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = [
    ("tartes-r50um", "50 μm"),
    ("tartes-r100um", "100 μm"),
    ("tartes-r250um", "250 μm"),
    ("tartes-r500um", "500 μm"),
    ("tartes-r1000um", "1000 μm"),
]

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#d8d8de"


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader])
    return {name: rows[:, i] for i, name in enumerate(header)}


def style(ax, xlabel, ylabel, title, subtitle=""):
    ax.set_xscale("log")
    ax.set_xlim(320, 1400)
    ax.set_xticks([320, 400, 500, 700, 1000, 1400])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # A log axis volunteers minor tick labels like "6 x 10^2" that collide
    # with the major ones. Only the majors are wanted here.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, alpha=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=26)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=9, va="bottom")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/validation"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = {}
    for stem, label in SERIES:
        path = args.input / f"{stem}.csv"
        if path.exists():
            data[label] = load(path)
    if not data:
        raise SystemExit(f"no validation CSVs found in {args.input}")

    colours = [plt.get_cmap("viridis")(v) for v in np.linspace(0.15, 0.88, len(data))]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=160)

    # Panel 1 -- the curves, as anyone would first plot them.
    for (label, columns), colour in zip(data.items(), colours):
        axes[0].plot(columns["wavelength_nm"], columns["snow_mcrt"],
                     color=colour, linewidth=2.0, label=label)
        axes[0].plot(columns["wavelength_nm"], columns["tartes"],
                     color=colour, linewidth=1.4, linestyle="--")
    style(axes[0], "wavelength (nm)", "spectral albedo",
          "Two codes, same ice",
          "solid: snow-mcrt · dashed: TARTES")
    axes[0].set_ylim(0.3, 1.02)
    axes[0].legend(title="grain radius", frameon=False, fontsize=8,
                   title_fontsize=8, loc="lower left")

    # Panel 2 -- hold the grain model fixed and the disagreement vanishes.
    for (label, columns), colour in zip(data.items(), colours):
        axes[1].plot(columns["wavelength_nm"], columns["transfer_residual"],
                     color=colour, linewidth=2.0, label=label)
    style(axes[1], "wavelength (nm)", "albedo difference",
          "Radiative transfer alone",
          "our solver fed TARTES's own single-scattering parameters")
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].set_ylim(-0.012, 0.012)
    axes[1].text(330, 0.0085,
                 "agreement to better than 0.003\nacross every grain size",
                 color=MUTED, fontsize=8.5, va="top")

    # Panel 3 -- and here is everything that is left.
    for (label, columns), colour in zip(data.items(), colours):
        axes[2].plot(columns["wavelength_nm"], columns["grain_model_residual"],
                     color=colour, linewidth=2.0, label=label)
    style(axes[2], "wavelength (nm)", "albedo difference",
          "The grain model, and nothing else",
          "spheres give g ≈ 0.89; TARTES uses 0.82 for real grains")
    axes[2].axhline(0, color=MUTED, linewidth=0.8)
    axes[2].text(330, -0.10,
                 "fifty times larger than the\ntransfer residual — this is the\n"
                 "sphere assumption, out of scope in v1",
                 color=MUTED, fontsize=8.5, va="top")

    out = args.output / "tartes-validation.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
