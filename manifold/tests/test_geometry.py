import numpy as np
import pytest
from manifold.atlas_eval.geometry import Sim3, camera_frame, fit_cameras, opencv_c2w, quaternion_to_matrix
from .synthetic import matrix_to_quaternion, rub_look_at


def _random_rotation(rng):
    q = rng.normal(size=4)
    return quaternion_to_matrix(q)


def _cameras(rng, n, same_center=False):
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        c2w[i, :3, :3] = _random_rotation(rng)
        c2w[i, :3, 3] = 0 if same_center else rng.normal(size=3) * 2
    return c2w


def test_rub_identity_becomes_an_opencv_camera_looking_down_minus_z():
    c2w = opencv_c2w([0, 0, 0], [0, 0, 0, 1], 'rub')
    assert np.allclose(c2w[:3, :3], np.diag([1, -1, -1]))
    point_in_camera = np.linalg.inv(c2w) @ np.array([0, 0, -2, 1.0])
    assert np.allclose(point_in_camera[:3], [0, 0, 2])  # in front, +Z forward
    assert np.allclose(opencv_c2w([1, 2, 3], [0, 0, 0, 1], 'rdf')[:3, :3], np.eye(3))


def test_look_at_round_trips_through_the_quaternion():
    r = rub_look_at([1.0, 0.5, 2.0], (0, 0, -3))
    assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(r)), r, atol=1e-9)


def test_zero_quaternion_and_unknown_convention_are_refused():
    with pytest.raises(ValueError):
        quaternion_to_matrix([0, 0, 0, 0])
    with pytest.raises(ValueError):
        opencv_c2w([0, 0, 0], [0, 0, 0, 1], 'luf')


@pytest.mark.parametrize('n', [2, 5])
def test_fit_recovers_a_known_similarity(n):
    rng = np.random.default_rng(n)
    truth = Sim3(_random_rotation(rng), rng.normal(size=3), 0.37)
    world = _cameras(rng, n)
    splat = np.stack([truth.camera(c) for c in world])
    sim, report = fit_cameras(world, splat)
    assert sim.scale == pytest.approx(0.37)
    assert np.allclose(sim.rotation, truth.rotation, atol=1e-9)
    assert np.allclose(sim.translation, truth.translation, atol=1e-9)
    assert report['center_rmse_world'] < 1e-9 and max(report['rotation_error_deg']) < 1e-6


def test_pure_rotation_leaves_the_scale_open_and_keeps_the_anchor():
    rng = np.random.default_rng(1)
    truth = Sim3(_random_rotation(rng), rng.normal(size=3), 2.0)
    world = _cameras(rng, 4, same_center=True)
    sim, report = fit_cameras(world, np.stack([truth.camera(c) for c in world]))
    assert sim.scale is None and 'center_rmse_world' not in report
    fixed = sim.with_scale(2.0, world[0, :3, 3])
    assert np.allclose(fixed.camera(world[1]), truth.camera(world[1]), atol=1e-9)


def test_single_image_frame_puts_the_input_camera_at_identity():
    rng = np.random.default_rng(2)
    c2w = _cameras(rng, 1)[0]
    assert np.allclose(camera_frame(c2w, 3.0).camera(c2w), np.eye(4), atol=1e-12)
    open_scale = camera_frame(c2w, None)
    assert np.allclose(open_scale.with_scale(3.0, c2w[:3, 3]).camera(c2w), np.eye(4), atol=1e-12)
    other = _cameras(rng, 1)[0]
    expected = camera_frame(c2w, 3.0).camera(other)
    assert np.allclose(open_scale.with_scale(3.0, c2w[:3, 3]).camera(other), expected, atol=1e-12)
