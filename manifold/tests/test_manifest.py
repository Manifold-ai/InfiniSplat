import json
import numpy as np
import pytest
from manifold.atlas_eval import manifest
from manifold.atlas_eval.make_manifest import from_atlas, from_images
from .synthetic import H, W, write_atlas_scene


def _write(root, data, name='manifest.json'):
    (root / name).write_text(json.dumps(data))
    return manifest.load(root, name)


@pytest.fixture
def atlas(tmp_path):
    root = tmp_path / 'scene'
    write_atlas_scene(root, create=3, generate=8, removed=1, log_depth_views=(4,), no_confidence=(5,))
    return root


def test_atlas_export_orders_create_first_and_splits_by_rule(atlas):
    data = from_atlas(atlas)
    scene = _write(atlas, data)
    ids = [v.view_id for v in scene.views]
    assert len(ids) == 11 and 'v_00000000000b' not in ids  # the removed view is gone
    assert [v.origin for v in scene.views] == ['create'] * 3 + ['generate'] * 8
    assert [v.view_id for v in scene.tests] == ['v_000000000002', 'v_000000000006', 'v_00000000000a']
    assert scene.inputs[0].view_id == 'v_000000000000'
    assert data['source']['split'] == dict(test_create=1, test_every=4)


def test_atlas_export_keeps_atlas_camera_and_depth_records(atlas):
    scene = _write(atlas, from_atlas(atlas))
    v = scene.view('v_000000000004')
    assert v.depth.encoding == 'log' and v.depth.format == 'png' and v.depth.z_max == 8.0
    assert scene.view('v_000000000005').confidence is None
    assert v.size == (W, H) and np.allclose(v.K, [[60, 0, 32], [0, 60, 18], [0, 0, 1]])
    assert np.allclose(v.c2w[:3, :3].T @ v.c2w[:3, :3], np.eye(3), atol=1e-9)


def test_single_create_view_defaults_to_no_create_test(tmp_path):
    root = tmp_path / 'single'
    write_atlas_scene(root, create=1, generate=4, removed=0)
    scene = _write(root, from_atlas(root))
    assert [v.origin for v in scene.inputs][:1] == ['create'] and len(scene.tests) == 1
    with pytest.raises(ValueError, match='at least one create view'):
        from_atlas(root, test_create=1)


def test_tampered_view_file_fails_the_checksum(atlas):
    (atlas / 'scene-v_000000000001.png').write_bytes(b'not the bytes Atlas returned')
    with pytest.raises(ValueError, match='sha256'):
        from_atlas(atlas)
    assert from_atlas(atlas, verify=False)['views']


def test_depth_and_confidence_decode(atlas):
    scene = _write(atlas, from_atlas(atlas))
    linear = manifest.read_depth(scene.view('v_000000000000'))
    assert linear.shape == (H, W) and np.isnan(linear[0, 0]) and linear[-1, -1] == pytest.approx(4.0)
    log = manifest.read_depth(scene.view('v_000000000004'))
    assert np.nanmax(np.abs(log[1:] - linear[1:])) < 2e-4 * 4  # 16-bit log quantisation
    z, mask = manifest.valid_depth(scene.view('v_000000000000'), min_confidence=0.9)
    assert not mask.any()  # confidence is 200/255 everywhere
    assert manifest.valid_depth(scene.view('v_000000000000'))[1].sum() == W * H - 1


def test_log_depth_endpoints():
    assert manifest.log_depth(np.array([1.0, 0.0]), 0.5, 8.0) == pytest.approx([0.5, 8.0])
    assert np.isnan(manifest.log_depth(np.array([1.5]), 0.5, 8.0)).all()


@pytest.mark.parametrize('mutate, message', [
    (lambda d: d['views'][0].update(image='../escape.png'), 'inside the scene folder'),
    (lambda d: d['views'][1].update(view_id=d['views'][0]['view_id']), 'unique'),
    (lambda d: d['views'][0]['depth'].pop('z_min', None) or d['views'][0]['depth'].update(encoding='log'), 'z_min'),
    (lambda d: d['views'][1].pop('camera'), 'every input view or for none'),
    (lambda d: d['views'][2]['camera'].pop('extrinsics'), 'test views need camera.extrinsics'),
    (lambda d: d['views'][0]['camera']['intrinsics'].update(width=1280), 'intrinsics are for'),
    (lambda d: d.update(schema='other/1'), 'schema'),
])
def test_bad_manifests_are_refused(atlas, mutate, message):
    data = from_atlas(atlas)
    mutate(data)
    with pytest.raises(ValueError, match=message):
        _write(atlas, data)


def test_plain_photos_make_an_unposed_manifest(tmp_path):
    root = tmp_path / 'photos'
    root.mkdir()
    from PIL import Image
    for name in ('b.jpg', 'a.png', 'notes.txt'):
        (Image.new('RGB', (8, 8)).save(root / name) if name != 'notes.txt' else (root / name).write_text('x'))
    scene = _write(root, from_images(root))
    assert [v.view_id for v in scene.inputs] == ['a', 'b'] and scene.inputs[0].c2w is None
    assert manifest.select_inputs(scene, 1)[0].view_id == 'a'
    with pytest.raises(ValueError):
        manifest.select_inputs(scene, 3)
