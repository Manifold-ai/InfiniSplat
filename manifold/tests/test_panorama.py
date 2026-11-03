import json
import subprocess
import sys
import numpy as np
import pytest
from PIL import Image
from manifold.atlas_eval import manifest
from manifold.atlas_eval.align import compute
from manifold.atlas_eval.make_manifest import from_atlas, from_images, from_panorama
from manifold.atlas_eval.panorama import depth_scale, equirect_pixel, view_size
from manifold.atlas_eval.run import REPO, auto_arms, parse_arm, plan
from .synthetic import equirect_directions, room_range, write_atlas_scene, write_panorama


def _scene(root, data):
    (root / 'manifest.json').write_text(json.dumps(data))
    return manifest.load(root)


@pytest.fixture
def pano(tmp_path):
    root = tmp_path / 'pano'
    write_panorama(root, depth='exr', confidence=True)
    return root


def test_panorama_manifest_carries_projection_and_range_map(pano):
    data = from_panorama(pano)
    assert data['schema'] == manifest.SCHEMA and data['views'][0]['depth']['kind'] == 'range'
    scene = _scene(pano, data)
    view = scene.panorama
    assert view is scene.inputs[0] and view.projection == 'equirect' and view.size == (64, 32)
    assert view.c2w is None and view.depth.kind == 'range' and view.confidence is not None
    assert manifest.read_depth(view).shape == (32, 64)


def test_panorama_without_depth_and_with_log_png(tmp_path):
    bare = tmp_path / 'bare'
    write_panorama(bare)
    assert _scene(bare, from_panorama(bare)).panorama.depth is None
    log = tmp_path / 'log'
    truth = write_panorama(log, depth='png')
    with pytest.raises(ValueError, match='--log-bounds'):
        from_panorama(log)
    view = _scene(log, from_panorama(log, (0.5, 8.0))).panorama
    assert np.allclose(manifest.read_depth(view), truth, rtol=1e-3)


@pytest.mark.parametrize('mutate, message', [
    (lambda d: d['views'][0].update(camera=dict(intrinsics=dict(fx=1, fy=1, cx=0, cy=0))), 'without camera'),
    (lambda d: d['views'][0].update(role='test'), 'without camera'),
    (lambda d: d['views'].append(dict(d['views'][0], view_id='other', projection='pinhole', depth=None)), 'exactly one view'),
    (lambda d: d['views'][0]['depth'].update(kind='z'), 'range map'),
    (lambda d: d['views'][0].update(projection='fisheye'), 'projection must be'),
    (lambda d: d.update(schema='infinisplat-atlas-eval/1'), 'needs schema'),
])
def test_bad_panorama_manifests_are_refused(pano, mutate, message):
    data = from_panorama(pano)
    mutate(data)
    with pytest.raises(ValueError, match=message):
        _scene(pano, data)


def test_non_panoramas_are_refused(tmp_path):
    root = tmp_path / 'wide'
    root.mkdir()
    Image.new('RGB', (64, 40)).save(root / 'wide.jpg')
    data = dict(schema=manifest.SCHEMA, views=[dict(view_id='wide', role='input', projection='equirect', image='wide.jpg')])
    with pytest.raises(ValueError, match='not a full 2:1'):
        _scene(root, data)
    Image.new('RGB', (64, 32)).save(root / 'two.jpg')
    with pytest.raises(ValueError, match='exactly one image'):
        from_panorama(root)


def test_range_depth_on_a_pinhole_view_is_refused(tmp_path):
    root = tmp_path / 'atlas'
    write_atlas_scene(root, create=2, generate=0, removed=0)
    data = from_atlas(root)
    data['views'][0]['depth']['kind'] = 'range'
    with pytest.raises(ValueError, match='range map'):
        _scene(root, data)


def test_a_lone_panorama_is_not_taken_for_a_photo(tmp_path):
    root = tmp_path / 'photos'
    write_panorama(root)
    with pytest.raises(ValueError, match='make_manifest pano'):
        from_images(root)


def test_panorama_routing(pano, tmp_path):
    scene = _scene(pano, from_panorama(pano))
    assert auto_arms(scene) == ['P']
    spec, views = plan(scene, 'P', parse_arm('P')[1])
    assert spec == dict(mode='panorama', long_side=768, backend='v2-panorama') and views == (scene.panorama,)
    assert parse_arm('X=panorama:518') == ('X', dict(mode='panorama', long_side=518))
    with pytest.raises(ValueError, match='LONG_SIDE only'):
        parse_arm('X=panorama:518:2')
    with pytest.raises(ValueError, match='panorama scene'):
        plan(scene, 'I1', parse_arm('I1')[1])
    atlas = tmp_path / 'atlas'
    write_atlas_scene(atlas, create=3, generate=0, removed=0)
    with pytest.raises(ValueError, match='panorama scene'):
        plan(_scene(atlas, from_atlas(atlas)), 'P', parse_arm('P')[1])
    out = subprocess.run([sys.executable, '-m', 'manifold.atlas_eval.run', '--scene', str(pano), '--out',
                          str(tmp_path / 'runs'), '--dry-run'], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout.split('\t')
    assert out[1:5] == ['P', 'v2-panorama', 'panorama', '768']
    assert view_size(768) == 770 and view_size(518) == 518


def test_equirect_pixel_inverts_pixel_centres():
    rows, cols = np.mgrid[:32, :64]
    row, col = equirect_pixel(equirect_directions(32, 64), 32, 64)
    assert (row == rows).all() and (col == cols).all()
    assert equirect_pixel([0, 0, 1], 32, 64)[1] in (31, 32)  # +z is the centre column
    assert equirect_pixel([0, -1, 0], 32, 64)[0] == 0  # -y (up) is the top row


def test_depth_scale_recovers_the_room_size(tmp_path):
    root = tmp_path / 'room'
    write_panorama(root, size=(1024, 512), depth='npy', confidence=True)
    view = _scene(root, from_panorama(root)).panorama
    directions = np.random.default_rng(1).normal(size=(50000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = directions * (room_range(directions) / 2.5)[:, None]  # the splat is 2.5x smaller than the room
    result = depth_scale(view, points)
    assert result['factor'] == pytest.approx(2.5, rel=0.01) and result['samples'] == len(points)
    lower = depth_scale(view, points, min_confidence=0.5)  # only the trusted lower half counts
    assert lower['factor'] == pytest.approx(2.5, rel=0.01) and lower['samples'] < 0.55 * len(points)
    with pytest.raises(ValueError, match='cannot set the scale'):
        depth_scale(view, points[:50])


def test_panorama_arms_are_not_aligned(pano, tmp_path):
    scene = _scene(pano, from_panorama(pano))
    arm = tmp_path / 'P'
    arm.mkdir()
    (arm / 'run.json').write_text(json.dumps(dict(inputs=['pano'], frame=dict(kind='panorama'))))
    with pytest.raises(ValueError, match='produce-only'):
        compute(scene, arm)
