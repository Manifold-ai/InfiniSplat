import json
import subprocess
import sys
import numpy as np
import pytest
from manifold.atlas_eval import manifest
from manifold.atlas_eval.align import compute
from manifold.atlas_eval.geometry import FLIP_YZ, Sim3
from manifold.atlas_eval.make_manifest import from_atlas
from manifold.atlas_eval.run import REPO, auto_arms, parse_arm, plan, prompt_depth
from .synthetic import write_atlas_scene


@pytest.fixture
def scene(tmp_path):
    root = tmp_path / 'scene'
    write_atlas_scene(root, create=3, generate=8, removed=0)
    (root / 'manifest.json').write_text(json.dumps(from_atlas(root)))
    return manifest.load(root)


def test_arm_specs():
    assert parse_arm('I1-v5') == ('I1-v5', dict(mode='provided', long_side=518, inputs=5))
    assert parse_arm('X=da3-pose:1022:7') == ('X', dict(mode='da3-pose', long_side=1022, inputs=7))
    assert parse_arm('Y=v1-rgb') == ('Y', dict(mode='v1-rgb'))
    for bad in ('nope', 'X=fancy', '../X=provided'):
        with pytest.raises(ValueError):
            parse_arm(bad)


def test_single_and_multi_image_routing(scene, tmp_path):
    assert auto_arms(scene) == ['I1']
    spec, views = plan(scene, 'I1', parse_arm('I1')[1])
    assert spec['backend'] == 'v2' and len(views) == len(scene.inputs) == 8
    spec, views = plan(scene, 'S-depth', parse_arm('S-depth')[1])
    assert spec['backend'] == 'v1' and [v.view_id for v in views] == [scene.inputs[0].view_id]
    with pytest.raises(ValueError, match='v2 needs 2-50'):
        plan(scene, 'X', parse_arm('X=provided:518:1')[1])
    with pytest.raises(ValueError, match='single image'):
        plan(scene, 'X', parse_arm('X=v1-rgb::3')[1])
    with pytest.raises(ValueError, match='asked for 11'):
        plan(scene, 'I1-v11', parse_arm('I1-v11')[1])
    single = tmp_path / 'single'
    write_atlas_scene(single, create=1, generate=0, removed=0)
    (single / 'manifest.json').write_text(json.dumps(from_atlas(single)))
    assert auto_arms(manifest.load(single)) == ['S-rgb', 'S-depth']


def test_depth_prompt_lands_inside_the_v1_range(scene):
    depth, scale, kept = prompt_depth(scene.inputs[0], 0.0)
    valid = np.isfinite(depth)
    assert kept == 1.0 and (depth[valid] > 1).all() and (depth[valid] < 100).all()
    z = manifest.read_depth(scene.inputs[0])
    assert np.allclose(depth[valid], z[valid] * scale)


def test_dry_run_prints_the_matrix(scene, tmp_path):
    out = subprocess.run([sys.executable, '-m', 'manifold.atlas_eval.run', '--scene', str(scene.root), '--out',
                          str(tmp_path / 'runs'), '--arm', 'I1', '--arm', 'I3', '--arm', 'S-depth', '--dry-run'],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    assert [line.split('\t')[1:4] for line in out] == [
        ['I1', 'v2', 'provided'], ['I3', 'v2', 'da3-pose'], ['S-depth', 'v1', 'v1-lidar']]
    assert not (tmp_path / 'runs').exists()


def test_alignment_of_each_frame_kind(scene, tmp_path):
    truth = Sim3(FLIP_YZ, [0.1, 0.2, 0.3], 0.5)
    views = scene.inputs
    v2 = tmp_path / 'v2'
    v2.mkdir()
    np.savez(v2 / 'cameras.npz', c2w=np.stack([truth.camera(v.c2w) for v in views]))
    (v2 / 'run.json').write_text(json.dumps(dict(inputs=[v.view_id for v in views],
                                                 frame=dict(kind='cameras', file='cameras.npz'))))
    sim, report = compute(scene, v2)
    assert sim.scale == pytest.approx(0.5) and report['center_rmse_relative'] < 1e-9
    v1 = tmp_path / 'v1'
    v1.mkdir()
    (v1 / 'run.json').write_text(json.dumps(dict(inputs=[views[0].view_id],
                                                 frame=dict(kind='camera', view_id=views[0].view_id, scale=None))))
    sim, report = compute(scene, v1)
    assert sim.scale is None and report['scale_source'] == 'render'
    ext = tmp_path / 'ext'
    ext.mkdir()
    (ext / 'run.json').write_text(json.dumps(dict(frame=dict(kind='world', rotation=FLIP_YZ.tolist()))))
    sim, _ = compute(scene, ext)
    assert sim.scale == 1.0 and np.allclose(sim.camera(views[0].c2w)[:3, 3], FLIP_YZ @ views[0].c2w[:3, 3])
