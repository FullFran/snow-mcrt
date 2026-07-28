"""The cache must be invisible: same numbers, fewer Mie series evaluations.

Every test here is written against a counting stub rather than the real solver.
That is deliberate. The question a cache has to answer is not "is Mie theory
right" -- ``test_mie.py`` owns that -- but "did the wrapper return exactly what
the wrapped solver would have returned, and did it avoid calling it twice for
the same input". A stub makes both observable; the real solver makes neither.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from snow_mcrt.adapters.cached_mie_solver import (
    CachedMieSolver,
    default_cache_dir,
    frozen_table_paths,
)
from snow_mcrt.adapters.miepython_solver import MiepythonSolver
from snow_mcrt.ports.mie_solver import MieSolver

ICE_M_VISIBLE = 1.3105 + 2.0e-9j


class CountingSolver:
    """A solver whose answers are cheap, deterministic, and countable.

    The values are nonsense physically and that is the point: if the cache ever
    returns something it computed itself rather than something the inner solver
    produced, these numbers will not match.
    """

    name = "counting"
    version = "0.0.0"

    def __init__(self) -> None:
        self.calls = 0
        self.elements = 0

    def efficiencies(self, m, x):
        m_arr, x_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(m, dtype=complex)),
            np.atleast_1d(np.asarray(x, dtype=float)),
        )
        self.calls += 1
        self.elements += x_arr.size
        q_sca = 2.0 + x_arr + m_arr.real
        return q_sca + 1.0, q_sca, np.tanh(x_arr) * 0.5

    def phase_function(self, m, x, mu):
        self.calls += 1
        return np.full_like(np.asarray(mu, dtype=float), float(x))


@pytest.fixture
def inner():
    return CountingSolver()


@pytest.fixture
def solver(inner, tmp_path):
    return CachedMieSolver(inner, cache_dir=tmp_path)


class TestItIsAMieSolver:
    def test_satisfies_the_port(self, solver):
        assert isinstance(solver, MieSolver)

    def test_reports_the_wrapped_solver_in_its_name(self, inner, tmp_path):
        assert "counting" in CachedMieSolver(inner, cache_dir=tmp_path).name

    def test_wrapping_the_real_solver_changes_no_number(self, tmp_path):
        real = MiepythonSolver()
        cached = CachedMieSolver(real, cache_dir=tmp_path)
        x = np.array([0.01, 1.0, 250.0, 1256.6])

        expected = real.efficiencies(ICE_M_VISIBLE, x)
        for _ in range(2):
            got = cached.efficiencies(ICE_M_VISIBLE, x)
            for a, b in zip(got, expected):
                # Bit-for-bit, not approximately. A cache that perturbs the
                # numbers it hands back is a second, silent source of error on
                # top of the transport noise.
                np.testing.assert_array_equal(a, b)


class TestTheAnswersAreUnchanged:
    def test_a_hit_returns_what_the_inner_solver_returned(self, solver, inner):
        x = np.array([1.0, 10.0, 100.0])
        first = solver.efficiencies(ICE_M_VISIBLE, x)
        second = solver.efficiencies(ICE_M_VISIBLE, x)
        assert inner.calls == 1
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_preserves_the_shape_of_a_two_dimensional_grid(self, solver):
        # The (radius, wavelength) quadrature grid compute_layer_properties
        # builds. Flattening it and forgetting to restore the shape is the
        # obvious way to break this wrapper.
        m = np.full((1, 4), ICE_M_VISIBLE)
        x = np.linspace(10.0, 40.0, 3)[:, np.newaxis]
        q_ext, q_sca, g = solver.efficiencies(m, x)
        assert q_ext.shape == q_sca.shape == g.shape == (3, 4)

    def test_a_cached_grid_comes_back_in_the_same_shape(self, solver):
        m = np.full((1, 4), ICE_M_VISIBLE)
        x = np.linspace(10.0, 40.0, 3)[:, np.newaxis]
        first = solver.efficiencies(m, x)
        second = solver.efficiencies(m, x)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_scalar_arguments_survive_the_round_trip(self, solver):
        first = solver.efficiencies(ICE_M_VISIBLE, 3.0)
        second = solver.efficiencies(ICE_M_VISIBLE, 3.0)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_repeated_entries_in_one_call_are_answered_consistently(
        self, solver, inner
    ):
        x = np.array([5.0, 5.0, 7.0, 5.0])
        q_ext, _, _ = solver.efficiencies(ICE_M_VISIBLE, x)
        assert inner.elements == 2  # 5.0 and 7.0, each evaluated once
        assert q_ext[0] == q_ext[1] == q_ext[3]


class TestItActuallyAvoidsWork:
    def test_the_second_identical_call_reaches_no_solver(self, solver, inner):
        x = np.linspace(1.0, 100.0, 32)
        solver.efficiencies(ICE_M_VISIBLE, x)
        solver.efficiencies(ICE_M_VISIBLE, x)
        assert inner.calls == 1

    def test_only_the_new_points_of_an_extended_grid_are_evaluated(
        self, solver, inner
    ):
        # Element-wise keys, not one hash over the whole array. Widening a
        # wavelength range by ten points should cost ten Mie evaluations, not
        # a full recomputation -- this is the difference between a cache that
        # survives ordinary use and one that only helps on an exact rerun.
        solver.efficiencies(ICE_M_VISIBLE, np.arange(1.0, 11.0))
        inner.elements = 0
        solver.efficiencies(ICE_M_VISIBLE, np.arange(1.0, 16.0))
        assert inner.elements == 5

    def test_a_different_refractive_index_is_a_miss(self, solver, inner):
        solver.efficiencies(1.31 + 0j, 10.0)
        inner.elements = 0
        solver.efficiencies(1.32 + 0j, 10.0)
        assert inner.elements == 1

    def test_the_absorbing_part_of_the_index_is_part_of_the_key(
        self, solver, inner
    ):
        # k moves by orders of magnitude across the spectrum while n barely
        # stirs. A key that keeps only the real part would serve visible
        # answers for near-infrared questions and look entirely plausible.
        solver.efficiencies(1.31 + 1e-9j, 10.0)
        inner.elements = 0
        solver.efficiencies(1.31 + 1e-4j, 10.0)
        assert inner.elements == 1

    def test_a_perturbed_size_parameter_is_a_miss_not_a_stale_hit(
        self, solver, inner
    ):
        solver.efficiencies(ICE_M_VISIBLE, 100.0)
        inner.elements = 0
        solver.efficiencies(ICE_M_VISIBLE, 100.0 * (1 + 1e-12))
        assert inner.elements == 1

    def test_counts_hits_and_misses(self, solver):
        solver.efficiencies(ICE_M_VISIBLE, np.arange(1.0, 5.0))
        solver.efficiencies(ICE_M_VISIBLE, np.arange(3.0, 7.0))
        assert solver.misses == 6
        assert solver.hits == 2


class TestItSurvivesTheProcess:
    def test_a_saved_cache_is_reused_by_a_new_instance(self, inner, tmp_path):
        x = np.linspace(1.0, 50.0, 16)
        first = CachedMieSolver(inner, cache_dir=tmp_path)
        expected = first.efficiencies(ICE_M_VISIBLE, x)
        first.save()

        second_inner = CountingSolver()
        second = CachedMieSolver(second_inner, cache_dir=tmp_path)
        got = second.efficiencies(ICE_M_VISIBLE, x)
        assert second_inner.calls == 0
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(a, b)

    def test_nothing_is_written_until_asked(self, solver, tmp_path):
        solver.efficiencies(ICE_M_VISIBLE, 1.0)
        assert list(tmp_path.glob("*.npz")) == []

    def test_leaving_the_context_saves(self, inner, tmp_path):
        with CachedMieSolver(inner, cache_dir=tmp_path) as s:
            s.efficiencies(ICE_M_VISIBLE, 1.0)
        assert list(tmp_path.glob("*.npz")) != []

    def test_saving_an_untouched_cache_writes_nothing(self, solver, tmp_path):
        solver.save()
        assert list(tmp_path.glob("*.npz")) == []

    def test_creates_its_directory(self, inner, tmp_path):
        nested = tmp_path / "does" / "not" / "exist"
        with CachedMieSolver(inner, cache_dir=nested) as s:
            s.efficiencies(ICE_M_VISIBLE, 1.0)
        assert list(nested.glob("*.npz")) != []

    def test_two_solvers_do_not_share_a_file(self, tmp_path):
        a = CachedMieSolver(CountingSolver(), cache_dir=tmp_path)
        b = CachedMieSolver(MiepythonSolver(), cache_dir=tmp_path)
        assert a.cache_path != b.cache_path

    def test_a_concurrent_writer_is_merged_not_clobbered(self, tmp_path):
        # Two figure scripts running at once is the ordinary case, not an
        # exotic one. Whoever saves second must not erase what the first
        # computed.
        first = CachedMieSolver(CountingSolver(), cache_dir=tmp_path)
        first.efficiencies(ICE_M_VISIBLE, 1.0)
        second = CachedMieSolver(CountingSolver(), cache_dir=tmp_path)
        second.efficiencies(ICE_M_VISIBLE, 2.0)
        first.save()
        second.save()

        reader_inner = CountingSolver()
        reader = CachedMieSolver(reader_inner, cache_dir=tmp_path)
        reader.efficiencies(ICE_M_VISIBLE, np.array([1.0, 2.0]))
        assert reader_inner.calls == 0

    def test_a_corrupt_cache_file_is_ignored_not_fatal(self, inner, tmp_path):
        # A cache is an optimisation. A truncated file from an interrupted run
        # must cost time, never a crash.
        solver = CachedMieSolver(inner, cache_dir=tmp_path)
        solver.cache_path.parent.mkdir(parents=True, exist_ok=True)
        solver.cache_path.write_bytes(b"not an npz")
        with pytest.warns(UserWarning, match="unreadable"):
            q_ext, _, _ = solver.efficiencies(ICE_M_VISIBLE, 1.0)
        assert np.isfinite(q_ext).all()

    def test_clear_forgets_both_memory_and_disk(self, inner, tmp_path):
        solver = CachedMieSolver(inner, cache_dir=tmp_path)
        solver.efficiencies(ICE_M_VISIBLE, 1.0)
        solver.save()
        solver.clear()
        assert not solver.cache_path.exists()
        inner.calls = 0
        solver.efficiencies(ICE_M_VISIBLE, 1.0)
        assert inner.calls == 1


class TestItDelegatesWhatItDoesNotCache:
    def test_the_phase_function_passes_straight_through(self, solver, inner):
        mu = np.linspace(-1.0, 1.0, 5)
        np.testing.assert_array_equal(
            solver.phase_function(ICE_M_VISIBLE, 7.0, mu),
            np.full(5, 7.0),
        )
        assert inner.calls == 1

    def test_validation_errors_still_reach_the_caller(self, tmp_path):
        cached = CachedMieSolver(MiepythonSolver(), cache_dir=tmp_path)
        with pytest.raises(ValueError, match="gain"):
            cached.efficiencies(1.31 - 1e-8j, 100.0)

    def test_a_rejected_call_leaves_nothing_behind(self, tmp_path):
        cached = CachedMieSolver(MiepythonSolver(), cache_dir=tmp_path)
        with pytest.raises(ValueError):
            cached.efficiencies(ICE_M_VISIBLE, -1.0)
        assert len(cached) == 0


class TestProvenanceIsPartOfTheTable:
    """A table with no record of what produced it is a table that cannot be
    committed. The bit-exact key makes this sharper than usual: without a
    stamp, a reader running a different solver version gets a *hit* and walks
    away with someone else's numbers believing they are their own."""

    def test_the_stamp_is_written_alongside_the_numbers(self, solver):
        solver.efficiencies(ICE_M_VISIBLE, 1.0)
        solver.save()
        with np.load(solver.cache_path) as data:
            stamp = json.loads(str(data["provenance"]))
        assert stamp["solver"] == "counting"
        assert stamp["solver_version"] == "0.0.0"
        assert stamp["numpy"] == np.__version__

    def test_a_table_from_another_solver_version_is_not_used(
        self, inner, tmp_path
    ):
        first = CachedMieSolver(inner, cache_dir=tmp_path)
        first.efficiencies(ICE_M_VISIBLE, 1.0)
        first.save()

        upgraded = CountingSolver()
        upgraded.version = "9.9.9"
        second = CachedMieSolver(upgraded, cache_dir=tmp_path)
        with pytest.warns(UserWarning, match="provenance"):
            second.efficiencies(ICE_M_VISIBLE, 1.0)
        assert upgraded.calls == 1

    def test_a_table_from_another_numpy_is_not_used(
        self, inner, tmp_path, monkeypatch
    ):
        first = CachedMieSolver(inner, cache_dir=tmp_path)
        first.efficiencies(ICE_M_VISIBLE, 1.0)
        first.save()

        monkeypatch.setattr(np, "__version__", "0.0.0-not-a-real-numpy")
        second_inner = CountingSolver()
        second = CachedMieSolver(second_inner, cache_dir=tmp_path)
        with pytest.warns(UserWarning):
            second.efficiencies(ICE_M_VISIBLE, 1.0)
        assert second_inner.calls == 1

    def test_a_table_with_no_stamp_at_all_is_not_used(self, inner, tmp_path):
        path = CachedMieSolver(inner, cache_dir=tmp_path).cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            m_real=np.array([1.31]),
            m_imag=np.array([0.0]),
            x=np.array([1.0]),
            q_ext=np.array([1.0]),
            q_sca=np.array([1.0]),
            g=np.array([0.0]),
        )
        solver = CachedMieSolver(inner, cache_dir=tmp_path)
        with pytest.warns(UserWarning):
            solver.efficiencies(1.31 + 0j, 1.0)
        assert inner.calls == 1

    def test_a_matching_stamp_is_silent(self, inner, tmp_path, recwarn):
        first = CachedMieSolver(inner, cache_dir=tmp_path)
        first.efficiencies(ICE_M_VISIBLE, 1.0)
        first.save()
        CachedMieSolver(CountingSolver(), cache_dir=tmp_path).efficiencies(
            ICE_M_VISIBLE, 1.0
        )
        assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


class TestTheFrozenTable:
    """The committed table and the working cache do different jobs. One is a
    reproducibility artifact, regenerated deliberately; the other is scratch
    space that grows on its own. Conflating them puts a churning binary blob
    in git history."""

    @pytest.fixture
    def frozen(self, tmp_path):
        source = CachedMieSolver(CountingSolver(), cache_dir=tmp_path / "build")
        source.efficiencies(ICE_M_VISIBLE, np.array([1.0, 2.0, 3.0]))
        source.save()
        return source.cache_path

    def test_is_read(self, frozen, inner, tmp_path):
        solver = CachedMieSolver(
            inner, cache_dir=tmp_path / "local", frozen_paths=[frozen]
        )
        solver.efficiencies(ICE_M_VISIBLE, np.array([1.0, 2.0]))
        assert inner.calls == 0

    def test_is_never_written_to(self, frozen, inner, tmp_path):
        before = frozen.read_bytes()
        with CachedMieSolver(
            inner, cache_dir=tmp_path / "local", frozen_paths=[frozen]
        ) as solver:
            solver.efficiencies(ICE_M_VISIBLE, 99.0)
        assert frozen.read_bytes() == before

    def test_only_the_new_points_land_in_the_local_cache(
        self, frozen, inner, tmp_path
    ):
        # Copying the frozen table into the local one would duplicate a
        # committed artifact on every machine for no gain.
        local = tmp_path / "local"
        with CachedMieSolver(
            inner, cache_dir=local, frozen_paths=[frozen]
        ) as solver:
            solver.efficiencies(ICE_M_VISIBLE, np.array([1.0, 2.0, 99.0]))
            local_path = solver.cache_path
        with np.load(local_path) as data:
            assert data["x"].tolist() == [99.0]

    def test_a_frozen_hit_still_counts_as_a_hit(self, frozen, inner, tmp_path):
        solver = CachedMieSolver(
            inner, cache_dir=tmp_path / "local", frozen_paths=[frozen]
        )
        solver.efficiencies(ICE_M_VISIBLE, np.array([1.0, 99.0]))
        assert (solver.hits, solver.misses) == (1, 1)

    def test_a_missing_frozen_table_is_harmless(self, inner, tmp_path):
        solver = CachedMieSolver(
            inner,
            cache_dir=tmp_path / "local",
            frozen_paths=[tmp_path / "nope.npz"],
        )
        q_ext, _, _ = solver.efficiencies(ICE_M_VISIBLE, 1.0)
        assert np.isfinite(q_ext).all()

    def test_a_frozen_table_from_another_version_is_refused_loudly(
        self, frozen, tmp_path
    ):
        # Silence here would look exactly like "the committed table did not
        # cover your grid", which is a different and much less interesting
        # problem.
        upgraded = CountingSolver()
        upgraded.version = "9.9.9"
        solver = CachedMieSolver(
            upgraded, cache_dir=tmp_path / "local", frozen_paths=[frozen]
        )
        with pytest.warns(UserWarning, match="provenance"):
            solver.efficiencies(ICE_M_VISIBLE, 1.0)
        assert upgraded.calls == 1

    def test_the_local_cache_wins_over_the_frozen_one(
        self, frozen, inner, tmp_path
    ):
        local = tmp_path / "local"
        seeded = CachedMieSolver(CountingSolver(), cache_dir=local)
        seeded.efficiencies(ICE_M_VISIBLE, 1.0)
        seeded.save()
        solver = CachedMieSolver(inner, cache_dir=local, frozen_paths=[frozen])
        assert solver.efficiencies(ICE_M_VISIBLE, 1.0)[0].size == 1
        assert inner.calls == 0


class TestWhereTheFrozenTableLives:
    def test_finds_a_committed_table_for_the_solver(self, tmp_path):
        path = tmp_path / "mie-efficiencies-miepython-v2.npz"
        path.write_bytes(b"")
        assert frozen_table_paths(MiepythonSolver(), tmp_path) == [path]

    def test_an_absent_directory_yields_nothing(self, tmp_path):
        assert frozen_table_paths(MiepythonSolver(), tmp_path / "gone") == []

    def test_ignores_a_table_built_for_a_different_solver(self, tmp_path):
        (tmp_path / "mie-efficiencies-counting-v2.npz").write_bytes(b"")
        assert frozen_table_paths(MiepythonSolver(), tmp_path) == []


class TestTheDefaultLocation:
    def test_honours_an_explicit_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SNOW_MCRT_CACHE_DIR", str(tmp_path / "chosen"))
        assert default_cache_dir() == tmp_path / "chosen"

    def test_falls_back_to_the_xdg_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SNOW_MCRT_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert default_cache_dir() == tmp_path / "snow-mcrt"

    def test_never_lands_inside_the_repository(self, monkeypatch):
        # A cache under the working tree is a cache that ends up in a commit.
        monkeypatch.delenv("SNOW_MCRT_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert default_cache_dir().is_absolute()
