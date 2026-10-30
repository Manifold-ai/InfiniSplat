import numpy as np
from plyfile import PlyData, PlyElement
from manifold.atlas_eval.splat_io import load_gaussians, vertex_count

V2_ORDER = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
            'scale_0', 'scale_1', 'scale_2', 'opacity', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
V1_ORDER = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']


def _write(path, order, values):
    data = np.empty(len(values['x']), dtype=[(n, '<f4') for n in order])
    for n in order:
        data[n] = values.get(n, 0)
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(str(path))


def test_field_order_does_not_matter(tmp_path):
    rng = np.random.default_rng(0)
    values = {n: rng.normal(size=5).astype(np.float32) for n in V1_ORDER}
    _write(tmp_path / 'v2.ply', V2_ORDER, values)
    _write(tmp_path / 'v1.ply', V1_ORDER, values)
    a, b = load_gaussians(tmp_path / 'v2.ply'), load_gaussians(tmp_path / 'v1.ply')
    for key in a:
        assert np.allclose(a[key], b[key])
    assert np.allclose(np.linalg.norm(a['quats'], axis=1), 1)
    assert np.allclose(a['opacities'], 1 / (1 + np.exp(-values['opacity'])), atol=1e-6)
    assert vertex_count(tmp_path / 'v1.ply') == 5
