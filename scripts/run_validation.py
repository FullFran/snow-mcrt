#!/usr/bin/env python3
"""Cross-validate against TARTES and write the decomposed comparison.

Headless, like every other run script here: it computes and writes, and draws
nothing.

Usage::

    python scripts/run_validation.py --output data/validation
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
from snow_mcrt.adapters.tabulated_constants import TabulatedConstants  # noqa: E402
from snow_mcrt.application.validate_tartes import (  # noqa: E402
    compare_with_tartes,
    specific_surface_area,
)

DEFAULT_CONSTANTS = (
    Path(__file__).resolve().parent.parent / "data" / "ice" / "warren_brandt_2008.dat"
)
# The committed Mie table. Read-only here: it is regenerated deliberately by
# scripts/build_mie_cache.py, never as a side effect of a run.
MIE_TABLE_DIR = Path(__file__).resolve().parent.parent / "data" / "mie"
GRAIN_RADII_UM = (50.0, 100.0, 250.0, 500.0, 1000.0)


def write_csv(comparison, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = comparison.columns()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        for row in zip(*columns.values()):
            writer.writerow(f"{value:.10g}" for value in row)
    return path


def compare(args: argparse.Namespace, solver) -> int:
    """Run every grain size, using the solver it is handed."""
    constants = TabulatedConstants(
        args.constants, name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()
    wavelengths = np.logspace(np.log10(320.0), np.log10(1400.0), args.points)

    print("Cross-validation against TARTES (Libois, Picard et al. 2013)")
    print("Both codes read Warren & Brandt (2008); the constants are not a variable.")
    print()
    print(f"  {'grain':>8} {'SSA':>8} {'transfer resid':>16} {'grain-model resid':>19}")

    worst_transfer = 0.0
    for radius_um in GRAIN_RADII_UM:
        comparison = compare_with_tartes(
            solver, constants, wavelengths, grain_radius_m=radius_um * 1e-6
        )
        write_csv(comparison, args.output / f"tartes-r{radius_um:g}um.csv")
        transfer = float(np.max(np.abs(comparison.transfer_residual)))
        grain = float(np.max(np.abs(comparison.grain_model_residual)))
        worst_transfer = max(worst_transfer, transfer)
        print(
            f"  {radius_um:6.0f}um {specific_surface_area(radius_um * 1e-6):8.1f}"
            f" {transfer:16.5f} {grain:19.5f}"
        )

    print()
    print(f"Worst transfer residual across every grain size: {worst_transfer:.5f}")
    print("That is the radiative transfer solution agreeing with an independent")
    print("published implementation. What remains is the grain model: full Mie on")
    print("spheres gives g ~ 0.89, TARTES uses 0.82 for non-spherical grains.")
    print()
    print(f"written to {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/validation"))
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--points", type=int, default=120)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="evaluate every Mie point from scratch instead of reusing the table",
    )
    args = parser.parse_args()

    with ExitStack() as stack:
        solver = MiepythonSolver()
        if not args.no_cache:
            solver = stack.enter_context(
                CachedMieSolver(
                    solver, frozen_paths=frozen_table_paths(solver, MIE_TABLE_DIR)
                )
            )
        status = compare(args, solver)
        if isinstance(solver, CachedMieSolver):
            print(f"Mie cache: {solver.hits} reused, {solver.misses} evaluated"
                  f" ({solver.cache_path})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
