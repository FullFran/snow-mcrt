"""Ray-box intersection, checked against distances worked out by hand.

Geometry is where a Monte Carlo fails silently. A photon that misses a face it
should have hit simply carries on through the object, and the result is a
perfectly smooth profile of the wrong snowpack -- no exception, no warning, no
energy lost. So every case here has an answer known in advance, and the awkward
ones are the point: rays parallel to a face, rays starting exactly on one, rays
already inside, and rays pointing away.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.domain.geometry import Box

# A 20 cm cube buried with its top face 30 cm down.
BURIED = Box(lower=np.array([-0.1, -0.1, 0.3]), upper=np.array([0.1, 0.1, 0.5]))


def rays(origins, directions):
    o = np.asarray(origins, dtype=float).reshape(-1, 3)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


class TestTheBoxItself:
    def test_reports_its_burial_depth(self):
        assert BURIED.top_depth_m == pytest.approx(0.3)

    def test_reports_its_size_and_centre(self):
        np.testing.assert_allclose(BURIED.size, [0.2, 0.2, 0.2])
        np.testing.assert_allclose(BURIED.centre, [0.0, 0.0, 0.4])

    def test_rejects_a_box_with_no_volume(self):
        with pytest.raises(ValueError, match="no inside"):
            Box(lower=np.zeros(3), upper=np.array([1.0, 1.0, 0.0]))

    def test_rejects_a_box_that_is_not_three_dimensional(self):
        with pytest.raises(ValueError, match="three dimensions"):
            Box(lower=np.zeros(2), upper=np.ones(2))


class TestContainment:
    def test_knows_inside_from_outside(self):
        points = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 0.2], [0.5, 0.0, 0.4]])
        np.testing.assert_array_equal(BURIED.contains(points), [True, False, False])

    def test_the_boundary_counts_as_inside(self):
        # A photon that has just refracted through a face sits exactly on it.
        # Calling that outside sends it straight back out, and the object
        # becomes a mirror rather than a medium.
        np.testing.assert_array_equal(
            BURIED.contains(np.array([[0.0, 0.0, 0.3], [0.1, 0.1, 0.5]])),
            [True, True],
        )


class TestDistancesKnownInAdvance:
    def test_a_ray_straight_down_hits_the_top_face(self):
        o, d = rays([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
        distance, normal = BURIED.distance_to_surface(o, d)
        assert distance[0] == pytest.approx(0.3)
        # Oriented against the ray, so it points back up out of the top face.
        np.testing.assert_allclose(normal[0], [0.0, 0.0, -1.0])

    def test_a_ray_from_the_side_hits_the_side_face(self):
        o, d = rays([[-0.5, 0.0, 0.4]], [[1.0, 0.0, 0.0]])
        distance, normal = BURIED.distance_to_surface(o, d)
        assert distance[0] == pytest.approx(0.4)
        np.testing.assert_allclose(normal[0], [-1.0, 0.0, 0.0])

    def test_a_diagonal_ray_hits_at_the_right_distance(self):
        # Aimed at the centre of the top face from 30 cm above and 30 cm to
        # the side: the path is 0.3*sqrt(2) and it arrives at x = 0.
        o, d = rays([[-0.3, 0.0, 0.0]], [[0.3, 0.0, 0.3]])
        distance, normal = BURIED.distance_to_surface(o, d)
        assert distance[0] == pytest.approx(0.3 * np.sqrt(2))
        np.testing.assert_allclose(normal[0], [0.0, 0.0, -1.0])

    def test_a_ray_from_inside_leaves_by_the_far_face(self):
        o, d = rays([[0.0, 0.0, 0.4]], [[0.0, 0.0, 1.0]])
        distance, normal = BURIED.distance_to_surface(o, d)
        assert distance[0] == pytest.approx(0.1)
        # Still against the ray: the photon is heading down, so the normal at
        # the bottom face points back up.
        np.testing.assert_allclose(normal[0], [0.0, 0.0, -1.0])

    def test_a_ray_from_inside_leaves_sideways_when_that_is_nearer(self):
        o, d = rays([[0.09, 0.0, 0.4]], [[1.0, 0.0, 0.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert distance[0] == pytest.approx(0.01)


class TestMisses:
    def test_a_ray_pointing_away_never_arrives(self):
        o, d = rays([[0.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert np.isinf(distance[0])

    def test_a_ray_beside_the_box_misses(self):
        o, d = rays([[0.5, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert np.isinf(distance[0])

    def test_a_ray_parallel_to_a_face_and_outside_it_misses(self):
        # Parallel rays divide by zero in the slab method. This is where a
        # naive implementation produces a nan and every downstream comparison
        # silently answers False.
        o, d = rays([[0.0, 0.0, 0.2]], [[1.0, 0.0, 0.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert np.isinf(distance[0])

    def test_a_ray_parallel_to_a_face_and_inside_the_slab_still_hits(self):
        o, d = rays([[-0.5, 0.0, 0.4]], [[1.0, 0.0, 0.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert np.isfinite(distance[0])

    def test_a_ray_just_past_the_box_misses(self):
        o, d = rays([[0.1 + 1e-6, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert np.isinf(distance[0])


class TestSittingOnASurface:
    def test_a_photon_leaving_a_face_does_not_re_hit_it(self):
        # The classic self-intersection bug. A photon that has just crossed
        # the top face going down is exactly on it; if the same face is
        # returned at distance zero it never gets anywhere and the loop
        # spends its whole budget on one photon.
        o, d = rays([[0.0, 0.0, 0.3]], [[0.0, 0.0, 1.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert distance[0] == pytest.approx(0.2)

    def test_a_photon_on_a_face_heading_out_leaves_immediately(self):
        o, d = rays([[0.0, 0.0, 0.3]], [[0.0, 0.0, -1.0]])
        distance, _ = BURIED.distance_to_surface(o, d)
        assert np.isinf(distance[0])


class TestVectorisationAndInvariants:
    def test_handles_a_population_at_once(self):
        rng = np.random.default_rng(0)
        o = rng.uniform(-0.6, 0.6, size=(512, 3))
        d = rng.normal(size=(512, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        distance, normal = BURIED.distance_to_surface(o, d)
        assert distance.shape == (512,)
        assert normal.shape == (512, 3)

    def test_normals_are_unit_vectors(self):
        rng = np.random.default_rng(1)
        o = rng.uniform(-0.6, 0.6, size=(512, 3))
        d = rng.normal(size=(512, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        _, normal = BURIED.distance_to_surface(o, d)
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0)

    def test_normals_always_oppose_the_ray(self):
        # Fresnel is given cos_i = -dot(d, n) and clips it at zero. A normal
        # facing the wrong way would silently become normal incidence.
        rng = np.random.default_rng(2)
        o = rng.uniform(-0.6, 0.6, size=(1024, 3))
        d = rng.normal(size=(1024, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        distance, normal = BURIED.distance_to_surface(o, d)
        hit = np.isfinite(distance)
        assert hit.any()
        assert np.all(np.sum(normal[hit] * d[hit], axis=1) <= 0)

    def test_the_reported_landing_point_is_on_the_box(self):
        # The strongest statement available without a second implementation:
        # step each ray by the distance it was given and check it arrives on
        # the boundary rather than near it.
        rng = np.random.default_rng(3)
        o = rng.uniform(-0.6, 0.6, size=(2048, 3))
        d = rng.normal(size=(2048, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        distance, _ = BURIED.distance_to_surface(o, d)
        hit = np.isfinite(distance)
        landing = o[hit] + distance[hit, None] * d[hit]
        on_face = np.isclose(landing, BURIED.lower, atol=1e-9) | np.isclose(
            landing, BURIED.upper, atol=1e-9
        )
        assert np.all(on_face.any(axis=1))
        inside = (landing >= BURIED.lower - 1e-9) & (landing <= BURIED.upper + 1e-9)
        assert np.all(inside)
