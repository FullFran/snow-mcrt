#!/usr/bin/env python3
"""Measure where the diffusion approximation holds, using transport as truth.

Every figure in ``docs/detectability.md`` is a diffusion calculation, and none
of them had ever been checked against a transport solution. This script closes
that: it runs the 3-D Monte Carlo over the snowpacks those figures describe and
writes the ratio ``R_MC(rho) / R_diffusion(rho)`` bin by bin.

The output is a validity range rather than a verdict. Diffusion is expected to
work a few transport mean free paths out from the source and to underestimate
the far tail, where the surviving photons are those that travelled relatively
straight and the near-isotropic assumption is worst. Publishing the size of
that departure is more useful than publishing a pass.

Headless, like every run script here: it computes and writes, and draws
nothing.

Usage::

    python scripts/run_diffusion_validation.py --output data/validation
    python scripts/run_diffusion_validation.py --photons 50000   # quick look
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
from snow_mcrt.application.validate_diffusion import (  # noqa: E402
    compare_with_diffusion,
)
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

# The snowpacks the detectability figures actually describe. Impurity loading
# is what sets the co-albedo in the visible, and therefore the penetration
# depth -- so it is the axis along which diffusion is most likely to break.
CASES = (
    ("clean-450nm", 450.0, 0.0),
    ("arctic-450nm", 450.0, 10.0),
    ("alpine-450nm", 450.0, 100.0),
    ("alpine-800nm", 800.0, 100.0),
)


def write_csv(comparison, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = comparison.columns()
    sampled = comparison.sampled()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        for row in zip(*(np.asarray(v)[sampled] for v in columns.values())):
            writer.writerow(f"{value:.10g}" for value in row)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/validation"))
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--photons", type=int, default=400_000)
    parser.add_argument("--max-scatters", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="evaluate every Mie point from scratch instead of reusing the table",
    )
    args = parser.parse_args()

    backend = NumpyBackend()
    constants = TabulatedConstants(
        args.constants, name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()

    print("Diffusion theory against the 3-D transport engine")
    print("Both solvers see one surface (n = 1.31, Fresnel vs its effective")
    print("internal reflection) and one source (a pencil beam).")
    print(f"{args.photons} photons per case, seed {args.seed}")
    print()
    print(f"  {'case':>14} {'mfp(cm)':>9} {'delta(cm)':>10} {'R':>7}"
          f" {'worst<12mfp':>12} {'trunc':>9}")

    with ExitStack() as stack:
        solver = MiepythonSolver()
        if not args.no_cache:
            solver = stack.enter_context(
                CachedMieSolver(
                    solver, frozen_paths=frozen_table_paths(solver, MIE_TABLE_DIR)
                )
            )

        for label, wavelength_nm, ng_per_g in CASES:
            grid = np.array([wavelength_nm])
            impurities = (
                (ImpurityLoading.from_ng_per_g(BLACK_CARBON, ng_per_g),)
                if ng_per_g
                else ()
            )
            layer = SnowLayer(100e-6, 300.0, impurities=impurities)
            props = compute_layer_properties(
                solver, layer, constants.m_at(grid), grid
            )
            comparison = compare_with_diffusion(
                backend,
                float(props.single_scattering_albedo[0]),
                float(props.asymmetry[0]),
                float(props.extinction_coefficient[0]),
                config=TransportConfig(
                    n_photons=args.photons,
                    seed=args.seed,
                    max_scatters=args.max_scatters,
                ),
            )
            write_csv(comparison, args.output / f"mc-diffusion-{label}.csv")
            print(
                f"  {label:>14} {comparison.transport_mfp_m * 100:9.3f}"
                f" {comparison.penetration_depth_m * 100:10.2f}"
                f" {comparison.reflected:7.4f}"
                f" {comparison.worst_ratio_within(12.0):12.1%}"
                f" {comparison.truncated:9.1e}"
            )

    print()
    print("'worst<12mfp' is the largest departure from unity within twelve")
    print("transport mean free paths of the source -- the range the")
    print("detectability figures actually use. Beyond it the Monte Carlo runs")
    print("above diffusion and the gap widens, which is the expected")
    print("direction: diffusion underestimates the straight-path tail.")
    print()
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
