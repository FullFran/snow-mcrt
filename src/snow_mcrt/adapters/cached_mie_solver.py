"""A memoising decorator over the Mie solver port.

Mie theory at snow size parameters is the single most expensive thing this
codebase does. A 1 mm grain at 400 nm sits at ``x ~ 1.6e4``, and the series
needs on the order of ``x`` terms; multiply that by sixteen quadrature radii
and a few hundred wavelengths and regenerating one figure costs minutes of
saturated CPU. It is also, almost always, the *same* computation: the grain
population and wavelength grid are fixed while the thing under study --
impurity loading, snow density, layer thickness -- varies downstream of it.
``figure_detectability_map`` is the clearest case, sweeping black carbon over
an ice-grain Mie table that never changes.

So this is a decorator over :class:`~snow_mcrt.ports.mie_solver.MieSolver`
rather than a cache inside ``MiepythonSolver``. The port is the seam that
already exists; wrapping it means any solver can be memoised, the domain is
untouched, and the caching can be dropped from a run by not wrapping.

**The key is the exact bits of** ``(Re m, Im m, x)``. Not a rounded key, not a
hash of the whole array. Two consequences, both intended:

- A hit is only ever possible on bit-identical inputs, so the cache cannot
  introduce numerical error. It either returns precisely what the wrapped
  solver would have returned, or it calls it. There is no third outcome.
- Keys are per element, so widening a wavelength grid costs only the new
  points. A whole-array hash would throw away the entire table whenever one
  endpoint moved, which is exactly what happens while a figure is being tuned.

Grids here are built by ``np.linspace`` and ``LogNormalGrainSizes`` from
parameters in the manifest, so they reproduce bit-for-bit and exact keys hit.
A grid arrived at some other way simply misses, and misses are correct.

The cache is an optimisation and behaves like one: a corrupt or unreadable
file costs time, never a run. Nothing is written until :meth:`save` is called
(or the context manager exits), and saving merges with whatever is already on
disk so two scripts running at once do not erase each other.

**Two tables, two jobs.** A *frozen* table under ``data/mie/`` is a committed
reproducibility artifact: it covers the configurations behind the published
results, it is regenerated deliberately by ``scripts/build_mie_cache.py``, and
it is only ever read. The *working* cache outside the repository is scratch
space that grows on its own. Conflating them would put a churning binary blob
in git history -- npz is compressed, so every regeneration rewrites the whole
file and git cannot delta it.

**Every table carries provenance and it is verified on load.** Solver, solver
version, NumPy version and platform. The bit-exact key makes this sharper than
it looks: without a stamp, a reader on a newer ``miepython`` asks for a point,
gets a *hit*, and walks away with someone else's numbers believing they are
their own. A cache that can silently mask a change of solver is not a cache,
it is a way to invalidate a validation. So a stamp that does not match is
refused and the points are recomputed -- the same choice ``OpticalConstants``
makes when asked to extrapolate, which raises rather than clamping.
"""

from __future__ import annotations

import json
import os
import platform
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Sequence

import numpy as np

from snow_mcrt.ports.mie_solver import MieSolver

# Bumped whenever the stored layout changes. Old files then miss rather than
# being misread, which is the only safe failure mode for a numerical cache.
CACHE_FORMAT_VERSION = 2

_FIELDS = ("m_real", "m_imag", "x", "q_ext", "q_sca", "g")
_PROVENANCE = "provenance"

Key = tuple[float, float, float]
Table = dict[Key, tuple[float, float, float]]


def table_filename(solver_name: str) -> str:
    """The filename a table for ``solver_name`` is stored under.

    The solver is in the name because two implementations of the port must not
    share a table -- telling them apart is the entire reason the port exists.
    """
    stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in solver_name)
    return f"mie-efficiencies-{stem}-v{CACHE_FORMAT_VERSION}.npz"


def frozen_table_paths(solver: MieSolver, directory: Path | str) -> list[Path]:
    """Committed tables in ``directory`` that belong to ``solver``.

    Empty when the directory or the table is absent, which is the ordinary
    case for an installed package with no repository around it.
    """
    path = Path(directory) / table_filename(solver.name)
    return [path] if path.exists() else []


def default_cache_dir() -> Path:
    """Where tables live when the caller does not say.

    ``SNOW_MCRT_CACHE_DIR`` wins, then the XDG cache home, then
    ``~/.cache/snow-mcrt``. Never a path inside the working tree: a cache under
    the repository is a cache that eventually lands in a commit, and these
    files are large, machine-specific, and regenerable.
    """
    override = os.environ.get("SNOW_MCRT_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "snow-mcrt"


class CachedMieSolver:
    """Memoises :meth:`efficiencies` in memory and on disk.

    Args:
        solver: The solver to wrap. Any implementation of the port.
        cache_dir: Directory holding the writable table. Defaults to
            :func:`default_cache_dir`.
        frozen_paths: Committed tables to read and never write. See
            :func:`frozen_table_paths`.

    Attributes:
        hits: Elements answered from a table since construction.
        misses: Elements handed to the wrapped solver since construction.
    """

    def __init__(
        self,
        solver: MieSolver,
        cache_dir: Path | str | None = None,
        frozen_paths: Iterable[Path | str] = (),
    ) -> None:
        self._solver = solver
        self.name = f"cached({solver.name})"
        # The numbers are the wrapped solver's, so the version is too.
        self.version = getattr(solver, "version", "unknown")
        self._dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._frozen_paths: Sequence[Path] = [Path(p) for p in frozen_paths]
        self._frozen: Table = {}
        self._table: Table = {}
        self._loaded = False
        self._dirty = False
        self.hits = 0
        self.misses = 0

    @property
    def inner(self) -> MieSolver:
        """The wrapped solver."""
        return self._solver

    @property
    def cache_path(self) -> Path:
        """The writable table this instance reads and writes."""
        return self._dir / table_filename(self._solver.name)

    def __len__(self) -> int:
        """Number of memoised ``(m, x)`` points, frozen and local together."""
        self._load()
        return len(self._frozen | self._table)

    def __enter__(self) -> CachedMieSolver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Saved even on the way out of an exception. The points already
        # computed are still valid, and a run that died three quarters of the
        # way through a figure is exactly when not having to redo them matters.
        self.save()

    def efficiencies(self, m: Any, x: Any) -> tuple[Any, Any, Any]:
        """Return ``(q_ext, q_sca, g)``, computing only what is not cached."""
        self._load()
        m_arr, x_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(m, dtype=complex)),
            np.atleast_1d(np.asarray(x, dtype=float)),
        )
        shape = x_arr.shape
        m_flat = np.ascontiguousarray(m_arr).ravel()
        x_flat = np.ascontiguousarray(x_arr).ravel()

        keys = [
            (float(mi.real), float(mi.imag), float(xi))
            for mi, xi in zip(m_flat, x_flat)
        ]
        # Unique misses only. A quadrature grid repeats the same radius across
        # wavelengths, and a call that asked the solver for the same point
        # twice would defeat the wrapper on its very first use.
        missing: dict[Key, int] = {}
        for i, key in enumerate(keys):
            if key not in self._table and key not in self._frozen and key not in missing:
                missing[key] = i

        if missing:
            index = np.fromiter(missing.values(), dtype=int, count=len(missing))
            q_ext, q_sca, g = self._solver.efficiencies(m_flat[index], x_flat[index])
            # Only past the solver call, so a rejected input leaves no trace.
            for key, qe, qs, gi in zip(
                missing,
                np.asarray(q_ext, dtype=float).ravel(),
                np.asarray(q_sca, dtype=float).ravel(),
                np.asarray(g, dtype=float).ravel(),
            ):
                self._table[key] = (float(qe), float(qs), float(gi))
            self._dirty = True

        self.misses += len(missing)
        self.hits += len(keys) - len(missing)

        # Local before frozen: if a point sits in both, the one this machine
        # computed is the one it can account for.
        values = np.array(
            [
                self._table[key] if key in self._table else self._frozen[key]
                for key in keys
            ],
            dtype=float,
        )
        return (
            values[:, 0].reshape(shape),
            values[:, 1].reshape(shape),
            values[:, 2].reshape(shape),
        )

    def phase_function(self, m: complex, x: float, mu: Any) -> Any:
        """Delegate unchanged.

        Not cached, and not an oversight. A phase function is an array over the
        angle grid rather than three numbers, so it is orders of magnitude
        larger per entry, and nothing in the spectral pipeline calls it in a
        loop. Caching it would trade a lot of disk for no measured time.
        """
        return self._solver.phase_function(m, x, mu)

    def save(self) -> None:
        """Write the table, merging with whatever is already on disk.

        A no-op when nothing new was computed. The merge is what makes two
        concurrent figure scripts safe: the second to finish adds its points to
        the first's rather than replacing the file wholesale.
        """
        if not self._dirty:
            return
        merged = dict(self._read(self.cache_path))
        merged.update(self._table)
        self._table = merged

        self._dir.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(merged.keys()), dtype=float).reshape(-1, 3)
        values = np.array(list(merged.values()), dtype=float).reshape(-1, 3)
        columns = dict(zip(_FIELDS, (*keys.T, *values.T)))
        columns[_PROVENANCE] = np.array(json.dumps(self.provenance()))

        # Written beside the target and moved into place: os.replace is atomic
        # within a filesystem, so a reader never sees a half-written table and
        # an interrupted save leaves the previous one intact.
        tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        try:
            np.savez(tmp, **columns)
            os.replace(tmp, self.cache_path)
        finally:
            tmp.unlink(missing_ok=True)
        self._dirty = False

    def clear(self) -> None:
        """Forget the writable table, in memory and on disk.

        The frozen tables are untouched: they are committed inputs, not this
        instance's to delete.
        """
        self._table = {}
        self._dirty = False
        self._loaded = True
        self.cache_path.unlink(missing_ok=True)

    def provenance(self) -> dict[str, str]:
        """What produced the numbers in a table written now.

        Everything here can move the last bits of a Mie efficiency, so
        everything here is compared before a stored table is trusted. The
        Python patch version is deliberately absent: the series is evaluated
        in NumPy, and pinning tables to an interpreter release would refuse
        them after an upgrade that cannot have changed a result.
        """
        return {
            "format": str(CACHE_FORMAT_VERSION),
            "solver": self._solver.name,
            "solver_version": self.version,
            "numpy": np.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        }

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for path in self._frozen_paths:
            self._frozen.update(self._read(path, announce=True))
        self._table.update(self._read(self.cache_path, announce=True))

    def _read(self, path: Path, announce: bool = False) -> Table:
        """Read a table, or return an empty one.

        Deliberately broad on failure. Absent, truncated, half-written,
        unreadable: every one has the same correct response, which is to
        recompute. A cache that can abort a six-hour sweep is worse than no
        cache.

        A provenance mismatch is different from a missing file and says so.
        Silence there would look exactly like "the committed table did not
        cover your grid", which is a different and much less interesting
        problem than "the table was built by a solver you are not running".
        """
        if not path.exists():
            return {}
        try:
            with np.load(path) as data:
                stored = json.loads(str(data[_PROVENANCE]))
                columns = [np.asarray(data[f], dtype=float) for f in _FIELDS]
        except Exception:
            if announce:
                warnings.warn(
                    f"ignoring unreadable Mie table {path}; recomputing",
                    stacklevel=3,
                )
            return {}
        current = self.provenance()
        if stored != current:
            if announce:
                differing = sorted(
                    k for k in current if stored.get(k) != current[k]
                )
                warnings.warn(
                    f"ignoring Mie table {path}: provenance differs in "
                    f"{', '.join(differing)}. It was built by "
                    f"{stored.get('solver_version', '?')} on "
                    f"{stored.get('machine', '?')}; recomputing rather than "
                    f"reusing numbers this run cannot account for.",
                    stacklevel=3,
                )
            return {}
        if len({c.shape for c in columns}) != 1:
            return {}
        m_real, m_imag, x, q_ext, q_sca, g = columns
        return {
            (float(a), float(b), float(c)): (float(d), float(e), float(f))
            for a, b, c, d, e, f in zip(m_real, m_imag, x, q_ext, q_sca, g)
        }
