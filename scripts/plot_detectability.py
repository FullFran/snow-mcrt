#!/usr/bin/env python3
"""Figures for the buried-object detectability note.

Four of them, and each answers a question the prose alone leaves abstract:

1. **What limits depth** — penetration against wavelength and impurity loading.
2. **What a measurement actually sees** — the sensitivity banana, and how it
   grows with source-detector separation.
3. **How deep, for what snow** — a detectability map over cleanliness and
   wavelength, with real snowpacks marked on it.
4. **What the instrument looks like** — the geometry, drawn.

Usage::

    python scripts/plot_detectability.py --output docs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snow_mcrt.adapters.miepython_solver import MiepythonSolver  # noqa: E402
from snow_mcrt.adapters.tabulated_constants import TabulatedConstants  # noqa: E402
from snow_mcrt.domain.diffusion import (  # noqa: E402
    DiffusionParameters,
    two_way_detection_depth,
)
from snow_mcrt.domain.medium import (  # noqa: E402
    BLACK_CARBON,
    ImpurityLoading,
    SnowLayer,
    compute_layer_properties,
)

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#d8d8de"
SNOW_FILL = "#eef2f7"
ACCENT = "#c1440e"

DEFAULT_CONSTANTS = (
    Path(__file__).resolve().parent.parent / "data" / "ice" / "warren_brandt_2008.dat"
)

# Representative loadings, with the places they correspond to.
LOADINGS = [
    (0.0, "pure ice"),
    (0.1, "0.1 ng/g — interior Antarctica"),
    (1.0, "1 ng/g"),
    (10.0, "10 ng/g — Arctic"),
    (100.0, "100 ng/g — alpine"),
]


def frame(ax, xlabel, ylabel, title, subtitle=""):
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


def snow_properties(solver, constants, wavelengths, ng_per_g, radius_m=100e-6):
    """Diffusion parameters for a snowpack at each wavelength."""
    impurities = (
        (ImpurityLoading.from_ng_per_g(BLACK_CARBON, ng_per_g),) if ng_per_g else ()
    )
    layer = SnowLayer(radius_m, 300.0, impurities=impurities)
    props = compute_layer_properties(
        solver, layer, constants.m_at(wavelengths), wavelengths
    )
    return [
        DiffusionParameters.from_optical_properties(
            props.single_scattering_albedo[i],
            props.asymmetry[i],
            props.extinction_coefficient[i],
        )
        for i in range(len(wavelengths))
    ]


def figure_penetration(solver, constants, out: Path) -> Path:
    wavelengths = np.linspace(360.0, 1000.0, 40)
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    colours = [plt.get_cmap("magma")(v) for v in np.linspace(0.15, 0.80, len(LOADINGS))]

    for (ng, label), colour in zip(LOADINGS, colours):
        params = snow_properties(solver, constants, wavelengths, ng)
        depths = np.array([p.penetration_depth for p in params])
        ax.plot(wavelengths, depths * 100, color=colour, linewidth=2.2, label=label)

    ax.set_yscale("log")
    frame(ax, "wavelength (nm)", "penetration depth (cm)",
          "What actually limits how deep you can see",
          "100 μm grains · 300 kg/m³ · black carbon loading")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.axhspan(50, 200, color=ACCENT, alpha=0.07)
    ax.text(995, 140, "typical avalanche burial depth", color=ACCENT,
            fontsize=8.5, ha="right", style="italic")
    # Each decade of black carbon costs roughly a factor of three in depth --
    # and beyond about 700 nm the curves converge, because there the ice
    # itself absorbs far more strongly than any trace impurity.
    ax.annotate(
        "each decade of black carbon\ncosts a factor of ~3",
        xy=(392, 175), xytext=(430, 12),
        color=MUTED, fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8,
                        connectionstyle="arc3,rad=0.2"),
    )
    ax.annotate(
        "above 700 nm the ice itself absorbs\nmore than any impurity, and the\n"
        "curves collapse onto one",
        xy=(800, 3.2), xytext=(640, 0.75),
        color=MUTED, fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    ax.set_ylim(0.5, 320)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_banana(solver, constants, out: Path) -> Path:
    params = snow_properties(solver, constants, np.array([450.0]), 1.0)[0]
    separations = [0.2, 0.6, 1.2]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=160)
    for ax, separation in zip(axes, separations):
        span = separation * 1.5
        x = np.linspace(-0.3 * separation, 1.3 * separation, 260)
        z = np.linspace(0.002, span * 0.75, 220)
        kernel = params.sensitivity_kernel(separation, x, z)

        # Log scale, because the maximum sits at the source and detector where
        # the fluence diverges. On a linear scale this is two bright dots and
        # no banana at all.
        with np.errstate(divide="ignore"):
            shown = np.log10(np.maximum(kernel, 1e-8))
        # The kernel is normalised to its maximum, which sits at the source
        # where fluence diverges. Levels are clipped to the range the banana
        # actually occupies -- spanning all the way to 0 would spend the whole
        # colour map on two singular points.
        mesh = ax.contourf(x * 100, z * 100, shown, levels=np.linspace(-6, -1, 26),
                           cmap="magma", extend="both")
        ax.contour(x * 100, z * 100, shown, levels=[-5, -4, -3, -2],
                   colors="white", linewidths=0.7, alpha=0.5)

        depth = params.probing_depth(separation)
        ax.plot([0, separation * 100], [0, 0], "o", color="#4cc9f0", markersize=9,
                markeredgecolor="white", markeredgewidth=1.2, clip_on=False, zorder=5)
        ax.axhline(depth * 100, color="#4cc9f0", linewidth=1.2, linestyle="--")
        ax.text(x[2] * 100, depth * 100 * 1.30,
                f"peak sensitivity {depth * 100:.0f} cm  =  ρ/{separation / depth:.1f}",
                color="#0b7285", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.75))

        ax.invert_yaxis()
        ax.set_title(f"ρ = {separation * 100:.0f} cm", color=INK, fontsize=11,
                     loc="left", pad=10)
        ax.set_xlabel("horizontal distance (cm)", color=INK)
        if ax is axes[0]:
            ax.set_ylabel("depth (cm)", color=INK)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Where the detected light has actually been",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.10)
    fig.text(0.005, 1.028,
             "sensitivity kernel, log scale over six decades · clean alpine snow at "
             "450 nm · source and detector marked in blue",
             color=MUTED, fontsize=9, ha="left")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.02, pad=0.015)
    cbar.set_label("log₁₀ relative sensitivity", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_detectability_map(solver, constants, out: Path) -> Path:
    wavelengths = np.linspace(360.0, 900.0, 34)
    loadings = np.logspace(-2, 2.3, 28)

    depths = np.zeros((len(loadings), len(wavelengths)))
    for i, ng in enumerate(loadings):
        params = snow_properties(solver, constants, wavelengths, float(ng))
        depths[i] = [
            two_way_detection_depth(p.penetration_depth) * 100 for p in params
        ]

    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=160)
    levels = np.logspace(np.log10(1), np.log10(300), 40)
    mesh = ax.contourf(wavelengths, loadings, depths, levels=levels,
                       cmap="viridis", norm=matplotlib.colors.LogNorm(),
                       extend="both")
    contours = ax.contour(wavelengths, loadings, depths, levels=[5, 10, 25, 50, 100],
                          colors="white", linewidths=0.9, alpha=0.8)
    ax.clabel(contours, fmt="%d cm", fontsize=8, colors="white")

    ax.set_yscale("log")
    frame(ax, "wavelength (nm)", "black carbon (ng/g)",
          "How deep, for what snow",
          "depth at which a two-way signal falls to 1% · 100 μm grains")

    for ng, label, colour in [
        (0.2, "interior Antarctica", "#ffd166"),
        (20.0, "Arctic", "#ff8fa3"),
        (80.0, "alpine", "#ff5a5f"),
    ]:
        ax.axhline(ng, color=colour, linewidth=1.3, linestyle="--", alpha=0.9)
        ax.text(366, ng * 1.22, label, color=colour, fontsize=8.5, ha="left",
                weight="bold")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015,
                        ticks=[1, 2, 5, 10, 25, 50, 100, 200])
    cbar.ax.set_yticklabels(["1", "2", "5", "10", "25", "50", "100", "200"])
    cbar.set_label("detection depth (cm)", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_geometry(solver, constants, out: Path) -> Path:
    params = snow_properties(solver, constants, np.array([450.0]), 1.0)[0]
    separation = 0.9
    x = np.linspace(-0.15, 1.15, 320)
    z = np.linspace(0.002, 0.70, 250)
    kernel = params.sensitivity_kernel(separation, x, z)
    with np.errstate(divide="ignore"):
        shown = np.log10(np.maximum(kernel, 1e-8))

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=160)

    # Air above, snow below.
    ax.add_patch(Rectangle((-15, -18), 130, 18, facecolor="white",
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((-15, 0), 130, 70, facecolor=SNOW_FILL,
                           edgecolor="none", zorder=0))
    ax.contourf(x * 100, z * 100, shown, levels=np.linspace(-6, -1, 22),
                cmap="magma", alpha=0.85, extend="both", zorder=1)
    ax.contour(x * 100, z * 100, shown, levels=[-4, -3, -2],
               colors="white", linewidths=0.8, alpha=0.55, zorder=2)
    ax.axhline(0, color=INK, linewidth=1.6, zorder=4)

    depth = params.probing_depth(separation) * 100

    # Source, detector, and the buried object at the depth this geometry probes.
    ax.plot(0, 0, "o", color="#4cc9f0", markersize=13, markeredgecolor="white",
            markeredgewidth=1.5, zorder=6)
    ax.text(0, -5, "source", color="#0b7285", fontsize=10, ha="center", weight="bold")
    ax.plot(separation * 100, 0, "s", color="#4cc9f0", markersize=12,
            markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.text(separation * 100, -5, "detector", color="#0b7285", fontsize=10,
            ha="center", weight="bold")

    ax.add_patch(Circle((separation * 50, depth), 6.0, facecolor="#1b1b1f",
                        edgecolor="white", linewidth=1.4, zorder=6))
    ax.text(separation * 50 + 9, depth,
            f"buried object\nat {depth:.0f} cm", color="white", fontsize=9,
            va="center", zorder=7)

    ax.add_patch(FancyArrowPatch((0, -12), (separation * 100, -12),
                                 arrowstyle="<->", color=MUTED, linewidth=1.2,
                                 mutation_scale=12, zorder=5))
    ax.text(separation * 50, -14, f"ρ = {separation * 100:.0f} cm",
            color=MUTED, fontsize=9.5, ha="center")

    ax.text(107, -6, "air", color=MUTED, fontsize=9.5, style="italic")
    ax.text(107, 8, "snow", color="#4a4a55", fontsize=9.5, style="italic")
    ax.text(-13, 60,
            "light diffuses through the banana-shaped\n"
            "volume between source and detector.\n"
            "An object outside it is invisible, however\n"
            "strongly it absorbs.",
            color=INK, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=GRID, alpha=0.92))

    ax.set_xlim(-15, 115)
    ax.set_ylim(68, -18)
    ax.set_xlabel("horizontal distance (cm)", color=INK)
    ax.set_ylabel("depth (cm)", color=INK)
    ax.set_title("The measurement, drawn", color=INK, fontsize=13, loc="left", pad=26)
    ax.text(0.0, 1.015,
            "clean alpine snow at 450 nm · peak sensitivity sits at about ρ/3",
            transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    solver = MiepythonSolver()
    constants = TabulatedConstants(
        args.constants, name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()

    written = [
        figure_penetration(solver, constants, args.output / "detect-penetration.png"),
        figure_banana(solver, constants, args.output / "detect-banana.png"),
        figure_detectability_map(
            solver, constants, args.output / "detect-map.png"
        ),
        figure_geometry(solver, constants, args.output / "detect-geometry.png"),
    ]
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
