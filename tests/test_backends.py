"""The backend port contract.

Both adapters are checked against the same expectations. The CuPy tests skip
rather than fail on a machine without CUDA -- but the *contract* tests, the
ones asserting that the class satisfies the protocol and reports its own
availability honestly, run everywhere. Those are the ones that catch a port
drifting away from its adapters.
"""

import numpy as np
import pytest

from snow_mcrt.adapters.cupy_backend import CupyBackend, cupy_is_importable
from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.ports.backend import Backend

requires_cupy = pytest.mark.skipif(
    not cupy_is_importable(), reason="CuPy is not installed on this machine"
)


class TestProtocolConformance:
    def test_numpy_backend_satisfies_the_port(self):
        assert isinstance(NumpyBackend(), Backend)

    @requires_cupy
    def test_cupy_backend_satisfies_the_port(self):
        assert isinstance(CupyBackend(), Backend)

    def test_cupy_backend_refuses_to_construct_without_cupy(self):
        if cupy_is_importable():
            pytest.skip("CuPy is installed, so construction should succeed")
        with pytest.raises(RuntimeError, match="CuPy is not installed"):
            CupyBackend()


class TestNumpyBackend:
    def test_is_always_available(self):
        assert NumpyBackend().is_available()

    def test_the_same_seed_reproduces_the_same_draw(self):
        backend = NumpyBackend()
        first = backend.random_uniform(backend.rng(1234), (100,))
        second = backend.random_uniform(backend.rng(1234), (100,))
        assert np.array_equal(first, second)

    def test_different_seeds_give_different_draws(self):
        backend = NumpyBackend()
        first = backend.random_uniform(backend.rng(1), (100,))
        second = backend.random_uniform(backend.rng(2), (100,))
        assert not np.array_equal(first, second)

    def test_draws_lie_in_the_unit_interval(self):
        backend = NumpyBackend()
        values = backend.random_uniform(backend.rng(0), (10_000,))
        assert values.min() >= 0.0
        assert values.max() < 1.0

    def test_round_trips_through_the_device(self):
        backend = NumpyBackend()
        original = np.linspace(0.0, 1.0, 32)
        assert np.array_equal(backend.to_numpy(backend.asarray(original)), original)


@requires_cupy
class TestCupyBackend:
    def test_random_draws_are_allocated_on_the_device(self):
        backend = CupyBackend()
        values = backend.random_uniform(backend.rng(0), (128,))
        # A host array here would mean a transfer per step in the transport
        # loop, which is the cost this port exists to make impossible.
        assert values.__class__.__module__.startswith("cupy")

    def test_round_trips_through_the_device(self):
        backend = CupyBackend()
        original = np.linspace(0.0, 1.0, 32)
        restored = backend.to_numpy(backend.asarray(original))
        assert isinstance(restored, np.ndarray)
        assert np.allclose(restored, original)

    def test_the_same_seed_reproduces_the_same_draw(self):
        backend = CupyBackend()
        first = backend.to_numpy(backend.random_uniform(backend.rng(7), (100,)))
        second = backend.to_numpy(backend.random_uniform(backend.rng(7), (100,)))
        assert np.array_equal(first, second)
