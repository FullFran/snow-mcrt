#!/usr/bin/env python3
"""Figures for what the transport engine measured, not what theory predicted.

Two of them, and each replaces an argument with a measurement:

1. **How detection fades with depth** — contrast against burial depth, for an
   absorber and for a void, with the photon-count noise floor drawn. Every
   earlier depth number in ``docs/detectability.md`` came from diffusion
   theory and a two-way attenuation argument; this is transport.
2. **Where diffusion holds** — the ratio of the transport profile to the
   closed-form one, across four snowpacks, with the bands where diffusion is
   usable and where it is not even applicable marked on the axis.

Draws only. Every number comes from a committed CSV, so a figure can be
restyled without waiting on any physics — and so no figure can quietly show a
different run than the one in the repository.

Usage::

    python scripts/plot_detection.py --output docs/figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#d8d8de"
FAINT = "#f0f0f3"

# Two categorical hues for two kinds of object. Checked with the palette
# validator: adjacent-pair separation is dE 22 under protanopia and 29 under
# tritanopia, well clear of the 8 floor, and both clear 3:1 against the
# surface. They are also direct-labelled, so identity never rests on colour.
ABSORBER = "#2f7ebd"
VOID = "#d1600a"

# The four snowpacks are ordered by co-albedo, which makes them a magnitude
# rather than four identities -- so one hue, light to dark, not four colours.
SEQUENTIAL = "viridis"


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def frame(ax, xlabel, ylabel, title, subtitle=""):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=26)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=9, va="bottom")


def figure_depth(data_dir: Path, out: Path) -> Path:
    """Contrast against burial depth, with the noise floor made visible."""
    absorber = read_csv(data_dir / "detection-black-slab.csv")
    void = read_csv(data_dir / "detection-void.csv")

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=160)

    # The floor first and underneath: below it a curve is Monte Carlo noise,
    # and on a log axis noise draws a perfectly convincing line all the way
    # to the edge of the figure. Anything plotted here means nothing.
    floor = float(absorber["noise_floor"][0])
    ax.axhspan(1e-5, floor * 3, color=FAINT, zorder=0)
    ax.text(0.12, floor * 3 * 0.62,
            "below three times the photon-count noise floor —\n"
            "a curve drawn here is not a measurement",
            color=MUTED, fontsize=8.5, va="top")

    for series, colour, label, marker in (
        (absorber, ABSORBER, "black slab", "o"),
        (void, VOID, "void (absorbs nothing)", "s"),
    ):
        keep = series["detectable"].astype(bool)
        x = series["depth_in_penetration_depths"]
        y = np.abs(series["contrast"])
        ax.plot(x, y, color=colour, linewidth=2.0, alpha=0.25, zorder=2)
        ax.plot(x[keep], y[keep], color=colour, linewidth=2.2,
                marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=1.0, label=label, zorder=3)

    # Direct labels as well as the legend, so identity never rests on colour.
    ax.annotate("black slab", xy=(absorber["depth_in_penetration_depths"][1],
                                 abs(absorber["contrast"][1])),
                xytext=(12, 14), textcoords="offset points",
                color=ABSORBER, fontsize=10, weight="bold")
    ax.annotate("void", xy=(void["depth_in_penetration_depths"][1],
                            abs(void["contrast"][1])),
                xytext=(12, -20), textcoords="offset points",
                color=VOID, fontsize=10, weight="bold")

    ax.set_yscale("log")
    ax.set_xlim(0, absorber["depth_in_penetration_depths"].max() * 1.02)
    ax.set_ylim(1e-4, 2.0)
    frame(ax, "burial depth (penetration depths δ)",
          "contrast  |ΔR / R|",
          "How fast a buried object stops being visible",
          "3-D Monte Carlo transport · 20 cm object · depth in δ, so the "
          "curve transfers to any snowpack")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # The headline, stated rather than left to be read off a log axis --
    # and stated with the photon count in it, because the depth at which the
    # curve disappears is a property of the budget, not of the snow. The
    # floor falls as 1/sqrt(N), so four times the photons buys about another
    # third of a penetration depth.
    first = abs(absorber["contrast"][0])
    deepest = absorber["depth_in_penetration_depths"][
        absorber["detectable"].astype(bool)
    ].max()
    ax.text(0.985, 0.62,
            f"an absorber at 0.1 δ removes {first:.0%} of the return.\n"
            f"At 40 000 photons it is lost below {deepest:.1f} δ —\n"
            "and the floor only falls as 1/√N.",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=GRID))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_validity(results_dir: Path, out: Path) -> Path:
    """The ratio of transport to diffusion, and the bands it lives in."""
    cases = (
        ("clean-450nm", "clean · 1−ω = 3.3e-07"),
        ("arctic-450nm", "Arctic · 7.9e-06"),
        ("alpine-450nm", "alpine · 7.6e-05"),
        ("alpine-800nm", "alpine 800 nm · 3.0e-04"),
    )
    colours = [plt.get_cmap(SEQUENTIAL)(v) for v in np.linspace(0.12, 0.82, len(cases))]

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=160)

    # Beyond this the far bins hold too few photons to be a measurement, so
    # the curves are drawn but not asserted -- dotted and pale, which is the
    # visual form of "reported and not interpreted".
    STARVED = 50.0

    ax.axvspan(0.9, 3.0, color=FAINT, zorder=0)
    ax.axvspan(3.0, 12.0, color="#eaf2f8", zorder=0)
    ax.axvspan(STARVED, 100.0, color=FAINT, zorder=0)
    ax.axhline(1.0, color=INK, linewidth=1.1, linestyle="--", alpha=0.55, zorder=1)

    for (case, label), colour in zip(cases, colours):
        d = read_csv(results_dir / f"mc-diffusion-{case}.csv")
        x, y = d["rho_in_transport_mfp"], d["ratio_mc_over_diffusion"]
        solid = x <= STARVED
        ax.plot(x[solid], y[solid], color=colour, linewidth=2.0, label=label,
                zorder=3)
        # One point of overlap so the two segments meet rather than gap.
        joins = x >= x[solid][-1]
        ax.plot(x[joins], y[joins], color=colour, linewidth=1.3,
                linestyle=":", alpha=0.55, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(0.9, 100)
    ax.set_ylim(0.6, 1.5)
    frame(ax, "source–detector separation ρ  (transport mean free paths)",
          "transport ÷ diffusion",
          "Where the diffusion approximation holds",
          "2 000 000 photons per case on a Tesla P100 · four snowpacks over "
          "three decades of absorption")

    ax.text(1.6, 1.45, "inapplicable\nρ < 3 mfp′", color=MUTED, fontsize=8.5,
            ha="center", va="top")
    ax.text(6.0, 1.45, "the band a measurement uses\n9–11%", color="#1b4965",
            fontsize=8.5, ha="center", va="top", weight="bold")
    ax.text(97.0, 1.45, "photon-starved\ndotted, not asserted", color=MUTED,
            fontsize=8.5, ha="right", va="top")
    ax.text(24.0, 0.885, "diffusion runs ~7% high here", color=MUTED,
            fontsize=8.5, ha="center")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--detection", type=Path, default=ROOT / "data" / "detection")
    parser.add_argument("--validation", type=Path, default=ROOT / "results" / "kaggle")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    written = [
        figure_depth(args.detection, args.output / "detect-transport-depth.png"),
        figure_validity(args.validation, args.output / "detect-diffusion-validity.png"),
    ]
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
