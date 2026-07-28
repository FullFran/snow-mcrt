#!/usr/bin/env python3
"""Build the committed Mie table behind the published results.

The table under ``data/mie/`` is a reproducibility artifact, not a cache that
happens to be in the repository. The difference is that it is regenerated
*deliberately*, by this script, covering exactly the configurations that
produced the committed CSVs and figures -- so someone who clones the
repository can rerun or vary those runs without spending four minutes of
saturated CPU re-deriving numbers that are already settled.

That is also why the working cache stays out of the tree. An npz is
compressed binary, so git cannot delta it: committing a table that grows
every time anyone widens a grid would rewrite the whole file on every run and
turn the history into a pile of blobs.

The table carries its provenance -- solver, solver version, NumPy, platform --
and readers verify it. A machine whose environment differs will refuse it and
recompute rather than reuse numbers it cannot account for. That is the point:
the alternative is silently publishing one solver's results under another's
name.

Usage::

    python scripts/build_mie_cache.py                  # into data/mie
    python scripts/build_mie_cache.py --check          # verify, write nothing
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from snow_mcrt.adapters.cached_mie_solver import (  # noqa: E402
    CachedMieSolver,
    frozen_table_paths,
    table_filename,
)
from snow_mcrt.adapters.miepython_solver import MiepythonSolver  # noqa: E402

DEFAULT_OUTPUT = ROOT / "data" / "mie"


def _load_script(name: str):
    """Import a sibling run script by path.

    ``scripts/`` is not a package, and making it one would imply the scripts
    are an API. They are not. Loading by path keeps the single source of truth
    for *which* configurations matter inside the scripts that publish them,
    which is the only way this table cannot drift away from the results it is
    supposed to cover.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_curves(solver, scratch: Path) -> None:
    """Every curve in ``data/reference`` -- research questions 1 and 2."""
    run_albedo = _load_script("run_albedo")
    args = argparse.Namespace(
        output=scratch / "reference",
        constants=run_albedo.DEFAULT_CONSTANTS,
        points=160,
        monte_carlo=False,
    )
    run_albedo.generate(args, solver)


def _detectability_figures(solver, scratch: Path) -> None:
    """Every figure in ``docs/figures`` for the buried-object direction."""
    plot = _load_script("plot_detectability")
    constants = plot.TabulatedConstants(
        plot.DEFAULT_CONSTANTS,
        name="Warren & Brandt 2008",
        wavelength_scale_to_nm=1000.0,
    ).load()
    out = scratch / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot.figure_penetration(solver, constants, out / "a.png")
    plot.figure_banana(solver, constants, out / "b.png")
    plot.figure_detectability_map(solver, constants, out / "c.png")
    plot.figure_geometry(solver, constants, out / "d.png")


def _tartes_validation(solver, scratch: Path) -> bool:
    """The cross-validation in ``data/validation``, if TARTES is installed.

    Returns whether it ran. TARTES is an optional extra, so a table built
    without it is smaller but not wrong -- and saying so beats pretending the
    coverage is complete.
    """
    try:
        import tartes  # noqa: F401
    except ImportError:
        return False
    run_validation = _load_script("run_validation")
    args = argparse.Namespace(
        output=scratch / "validation",
        constants=run_validation.DEFAULT_CONSTANTS,
        points=120,
    )
    run_validation.compare(args, solver)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the committed table covers every published run,"
        " and write nothing",
    )
    args = parser.parse_args()

    inner = MiepythonSolver()
    print(f"solver   : {inner.name} {inner.version}")
    print(f"table    : {args.output / table_filename(inner.name)}")
    print()

    # In --check mode the existing table is read as a frozen input, so a miss
    # means it failed to cover a published run. In build mode it is the
    # writable table, so a miss means work to do.
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        if args.check:
            solver = CachedMieSolver(
                inner,
                cache_dir=scratch / "unused",
                frozen_paths=frozen_table_paths(inner, args.output),
            )
        else:
            solver = CachedMieSolver(inner, cache_dir=args.output)

        _reference_curves(solver, scratch)
        _detectability_figures(solver, scratch)
        ran_tartes = _tartes_validation(solver, scratch)

        if not args.check:
            solver.save()

    print()
    print(f"points   : {len(solver)} distinct (m, x)")
    print(f"reused   : {solver.hits}")
    print(f"computed : {solver.misses}")
    if not ran_tartes:
        print()
        print("TARTES is not installed, so the cross-validation grid is NOT")
        print("covered. Install the [validation] extra and rerun for a")
        print("complete table.")

    if args.check:
        if solver.misses:
            print()
            print(f"INCOMPLETE: {solver.misses} points are missing from the")
            print("committed table. Rerun without --check to rebuild it.")
            return 1
        print()
        print("Complete: every published run is covered.")
        return 0

    for key, value in solver.provenance().items():
        print(f"  {key:14} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
