#!/usr/bin/env python3
"""Sweep an object's burial depth and write the contrast it produces.

The detectability note predicts how detection fades with depth from diffusion
theory and a two-way attenuation argument. This measures it: bury an object,
trace photons, compare against the same snowpack with nothing in it.

Depth is written in penetration depths as well as metres, because that is the
coordinate the answer transfers in. The same object at the same centimetre
depth is easy in clean polar snow and hopeless in dirty alpine snow -- those
differ by a factor of thirty in how far light reaches -- and quoting `delta`
collapses both onto one curve.

Two objects, chosen to bracket the physics:

- a **black slab**, which removes light by absorbing it;
- a **void**, which absorbs nothing at all and removes light anyway, by
  carrying photons in a straight line to its far wall well below the depth
  anything returns from.

Headless, like every run script here: it computes and writes, and draws
nothing.

Usage::

    python scripts/run_detection.py --output data/detection
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snow_mcrt.adapters.cached_mie_solver import (  # noqa: E402
    CachedMieSolver,
    frozen_table_paths,
)
from snow_mcrt.adapters.miepython_solver import MiepythonSolver  # noqa: E402
from snow_mcrt.adapters.numpy_backend import NumpyBackend  # noqa: E402
from snow_mcrt.adapters.tabulated_constants import TabulatedConstants  # noqa: E402
from snow_mcrt.application.detection import sweep_burial_depth  # noqa: E402
from snow_mcrt.domain.medium import (  # noqa: E402
    BLACK_CARBON,
    ImpurityLoading,
    SnowLayer,
    compute_layer_properties,
)
from snow_mcrt.domain.transport import TransportConfig  # noqa: E402

DEFAULT_CONSTANTS = (
    Path(__file__).resolve().parent.parent / "data" / "ice" / "warren_brandt_2008.dat"
)
MIE_TABLE_DIR = Path(__file__).resolve().parent.parent / "data" / "mie"

# Alpine snow at 450 nm. Chosen because its penetration depth is a few
# centimetres, so the whole detection curve fits in a run a CPU can finish.
# The curve is written in units of that depth, so it is not a statement about
# alpine snow in particular.
WAVELENGTH_NM = 450.0
BLACK_CARBON_NG_PER_G = 100.0
GRAIN_RADIUS_M = 100e-6
SNOW_DENSITY = 300.0

OBJECTS = (
    ("black-slab", dict(object_extinction=1e5, object_albedo=0.0, object_index=1.31)),
    ("void", dict(object_extinction=0.0, object_albedo=0.0, object_index=1.0)),
)


def write_csv(sweep, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sweep.columns()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        for row in zip(*columns.values()):
            writer.writerow(f"{value:.10g}" for value in row)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/detection"))
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--photons", type=int, default=60_000)
    parser.add_argument("--max-scatters", type=int, default=4_000)
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    backend = NumpyBackend()
    constants = TabulatedConstants(
        args.constants, name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()

    with ExitStack() as stack:
        solver = MiepythonSolver()
        if not args.no_cache:
            solver = stack.enter_context(
                CachedMieSolver(
                    solver, frozen_paths=frozen_table_paths(solver, MIE_TABLE_DIR)
                )
            )
        grid = np.array([WAVELENGTH_NM])
        layer = SnowLayer(
            GRAIN_RADIUS_M,
            SNOW_DENSITY,
            impurities=(
                ImpurityLoading.from_ng_per_g(BLACK_CARBON, BLACK_CARBON_NG_PER_G),
            ),
        )
        props = compute_layer_properties(solver, layer, constants.m_at(grid), grid)

    omega = float(props.single_scattering_albedo[0])
    g = float(props.asymmetry[0])
    beta = float(props.extinction_coefficient[0])

    # Out to four penetration depths. Past that the contrast is below the
    # noise floor of any photon count a CPU can reach, which the sweep
    # reports rather than hides.
    from snow_mcrt.domain.diffusion import DiffusionParameters

    delta = DiffusionParameters.from_optical_properties(
        omega, g, beta, refractive_index=1.31
    ).penetration_depth
    depths = np.linspace(0.1, 4.0, args.points) * delta

    print(f"snow      : {WAVELENGTH_NM:.0f} nm, {BLACK_CARBON_NG_PER_G:.0f} ng/g BC,"
          f" {GRAIN_RADIUS_M * 1e6:.0f} um grains")
    print(f"1 - omega : {1 - omega:.3e}")
    print(f"delta     : {delta * 100:.2f} cm")
    print(f"photons   : {args.photons} per point, seed {args.seed}")
    print()

    for label, kind in OBJECTS:
        sweep = sweep_burial_depth(
            backend,
            omega,
            g,
            beta,
            depths_m=depths,
            label=label,
            config=TransportConfig(
                n_photons=args.photons,
                seed=args.seed,
                max_scatters=args.max_scatters,
            ),
            **kind,
        )
        write_csv(sweep, args.output / f"detection-{label}.csv")
        print(f"  {label}  (plain return {sweep.plain_reflected:.4f},"
              f" noise floor {sweep.noise_floor:.2%})")
        print(f"    {'depth/delta':>12} {'contrast':>10} {'resolved':>9}")
        for i in range(depths.size):
            print(f"    {sweep.depth_in_penetration_depths[i]:12.2f}"
                  f" {sweep.contrast[i]:10.3%}"
                  f" {'yes' if sweep.detectable[i] else 'no':>9}")
        print()

    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
