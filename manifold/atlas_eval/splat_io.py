"""Read SH0 Gaussian PLYs by property name: v1, v2 and the 17-float scene splat layout all differ in order."""
import numpy as np

REQUIRED = ('x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')


def vertex_count(path):
    """Gaussian count from the PLY header, without reading the body."""
    with open(path, 'rb') as f:
        for _ in range(256):
            line = f.readline()
            if line.startswith(b'element vertex '):
                return int(line.split()[2])
            if line.strip() == b'end_header' or not line:
                break
    raise ValueError(f'{path}: no vertex element in the PLY header')


def load_gaussians(path):
    """means [N,3], quats wxyz [N,4] (normalized), scales [N,3] (linear), opacities [N] (0-1), sh0 [N,3]."""
    from plyfile import PlyData
    vertex = PlyData.read(str(path))['vertex'].data
    missing = [k for k in REQUIRED if k not in vertex.dtype.names]
    if missing:
        raise ValueError(f'{path}: missing PLY properties {missing}')
    column = lambda *names: np.stack([np.asarray(vertex[n], dtype=np.float32) for n in names], axis=1)
    quats = column('rot_0', 'rot_1', 'rot_2', 'rot_3')
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    if (norm < 1e-12).any():
        raise ValueError(f'{path}: zero quaternion')
    return dict(means=column('x', 'y', 'z'), quats=quats / norm,
                scales=np.exp(column('scale_0', 'scale_1', 'scale_2')),
                opacities=1 / (1 + np.exp(-np.asarray(vertex['opacity'], dtype=np.float32))),
                sh0=column('f_dc_0', 'f_dc_1', 'f_dc_2'))
