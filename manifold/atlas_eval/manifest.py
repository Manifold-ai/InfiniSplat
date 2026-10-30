"""Scene folder + manifest.json, compatible with Atlas posed RGBD views. See manifold/README.md."""
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .geometry import intrinsics_matrix, opencv_c2w

SCHEMA = 'infinisplat-atlas-eval/1'
ROLES = ('input', 'test')
ORIGINS = ('create', 'generate', 'photo')
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp')
DEPTH_FORMATS = ('exr', 'png', 'npy')
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Depth:
    path: Path
    encoding: str
    format: str
    z_min: float | None
    z_max: float | None


@dataclass(frozen=True)
class View:
    view_id: str
    role: str
    origin: str
    image: Path
    size: tuple[int, int] | None
    K: np.ndarray | None
    c2w: np.ndarray | None
    depth: Depth | None
    confidence: Path | None


@dataclass(frozen=True)
class Scene:
    root: Path
    manifest: Path
    name: str
    views: tuple[View, ...]

    @property
    def inputs(self):
        return tuple(v for v in self.views if v.role == 'input')

    @property
    def tests(self):
        return tuple(v for v in self.views if v.role == 'test')

    def view(self, view_id):
        return next(v for v in self.views if v.view_id == view_id)


def _file(root, value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{what} must be a relative path')
    if Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError(f'{what} must stay inside the scene folder: {value}')
    path = root / value
    if not path.is_file():
        raise ValueError(f'{what} not found: {value}')
    return path


def _number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{what} must be a finite number')
    return float(value)


def _camera(camera, view_id):
    intr = camera.get('intrinsics')
    if not isinstance(intr, dict):
        raise ValueError(f'{view_id}: camera.intrinsics is required')
    fx, fy, cx, cy = (_number(intr.get(k), f'{view_id}: intrinsics.{k}') for k in ('fx', 'fy', 'cx', 'cy'))
    if fx <= 0 or fy <= 0:
        raise ValueError(f'{view_id}: focal lengths must be positive')
    size = (int(intr['width']), int(intr['height'])) if 'width' in intr and 'height' in intr else None
    c2w = None
    extr = camera.get('extrinsics')
    if extr is not None:
        position = [_number(v, f'{view_id}: position') for v in extr.get('position', ())]
        quaternion = [_number(v, f'{view_id}: quaternion_xyzw') for v in extr.get('quaternion_xyzw', ())]
        if len(position) != 3 or len(quaternion) != 4:
            raise ValueError(f'{view_id}: extrinsics needs position[3] and quaternion_xyzw[4]')
        c2w = opencv_c2w(position, quaternion, extr.get('coordinate_system'))
    return intrinsics_matrix(fx, fy, cx, cy), size, c2w


def _depth(root, depth, view_id):
    encoding, fmt = depth.get('encoding'), depth.get('format')
    path = _file(root, depth.get('file'), f'{view_id}: depth.file')
    fmt = fmt or path.suffix.lower().lstrip('.')
    if encoding not in ('linear', 'log') or fmt not in DEPTH_FORMATS:
        raise ValueError(f'{view_id}: depth needs encoding linear|log and format exr|png|npy')
    bounded = depth.get('z_min') is not None and depth.get('z_max') is not None
    if (encoding == 'log') != bounded or (encoding == 'linear' and fmt == 'png'):
        raise ValueError(f'{view_id}: log depth carries z_min/z_max, linear depth is exr or npy without them')
    z_min = _number(depth['z_min'], f'{view_id}: z_min') if bounded else None
    z_max = _number(depth['z_max'], f'{view_id}: z_max') if bounded else None
    if bounded and not 0 < z_min < z_max:
        raise ValueError(f'{view_id}: need 0 < z_min < z_max')
    return Depth(path, encoding, fmt, z_min, z_max)


def load(scene_dir, manifest='manifest.json'):
    root = Path(scene_dir)
    path = root / manifest
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError(f'{path}: schema must be {SCHEMA}')
    views, seen = [], set()
    for item in data.get('views', ()):
        view_id = item.get('view_id')
        if not isinstance(view_id, str) or not _ID.match(view_id) or view_id in seen:
            raise ValueError(f'view_id must be unique and filename-safe: {view_id!r}')
        seen.add(view_id)
        role, origin = item.get('role'), item.get('origin', 'photo')
        if role not in ROLES or origin not in ORIGINS:
            raise ValueError(f'{view_id}: role must be one of {ROLES}, origin one of {ORIGINS}')
        image = _file(root, item.get('image'), f'{view_id}: image')
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'{view_id}: image must be PNG, JPG or WebP')
        K = size = c2w = None
        if item.get('camera') is not None:
            K, size, c2w = _camera(item['camera'], view_id)
            if size is not None:
                from PIL import Image
                with Image.open(image) as im:
                    if im.size != size:
                        raise ValueError(f'{view_id}: image is {im.size}, intrinsics are for {size}')
        if role == 'test' and c2w is None:
            raise ValueError(f'{view_id}: test views need camera.extrinsics')
        depth = _depth(root, item['depth'], view_id) if item.get('depth') else None
        confidence = _file(root, item['confidence'], f'{view_id}: confidence') if item.get('confidence') else None
        views.append(View(view_id, role, origin, image, size, K, c2w, depth, confidence))
    scene = Scene(root, path, data.get('name') or root.name, tuple(views))
    if not scene.inputs:
        raise ValueError(f'{path}: at least one input view is required')
    posed = [v.c2w is not None for v in scene.inputs]
    if any(posed) and not all(posed):
        raise ValueError(f'{path}: give extrinsics for every input view or for none')
    return scene


def select_inputs(scene, count=None):
    """The first `count` input views in manifest order (all of them by default)."""
    inputs = scene.inputs
    if count is None:
        return inputs
    if not 1 <= count <= len(inputs):
        raise ValueError(f'{scene.name}: asked for {count} inputs, the manifest has {len(inputs)}')
    return inputs[:count]


def _cv2():
    # OpenCV keeps EXR decoding off unless this is set before the first EXR read (opencv#21326).
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import cv2
    return cv2


def _read(path):
    cv2 = _cv2()
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f'cannot decode {path}')
    if array.ndim == 3:  # OpenCV is BGR(A): take R, as the Atlas consumers do.
        array = array[:, :, 2] if array.shape[2] >= 3 else array[:, :, 0]
    return array


def _unit(array):
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535
    if array.dtype.kind == 'f':
        return array.astype(np.float32)
    raise ValueError(f'unsupported bit depth {array.dtype}')


def log_depth(t, z_min, z_max):
    """Atlas logDepthAsset: z = z_min * (z_max / z_min) ** (1 - t); outside [0, 1] -> NaN.
    Provisional reading, identical to the formula the mono Atlas scene fusion uses."""
    t = np.asarray(t, dtype=np.float64)
    inside = np.isfinite(t) & (t >= 0) & (t <= 1)
    z = z_min * (z_max / z_min) ** (1 - np.where(inside, t, 0.0))
    return np.where(inside, z, np.nan).astype(np.float32)


def read_depth(view):
    """Camera-space z in manifest world units, float32 [H,W]; NaN or <= 0 means no geometry."""
    depth = view.depth
    if depth is None:
        raise ValueError(f'{view.view_id} has no depth')
    raw = np.load(depth.path) if depth.format == 'npy' else _read(depth.path)
    if raw.ndim != 2:
        raise ValueError(f'{depth.path}: depth must be single channel')
    if depth.encoding == 'linear':
        if raw.dtype.kind != 'f':
            raise ValueError(f'{depth.path}: linear depth must be floating point')
        return raw.astype(np.float32)
    return log_depth(_unit(raw), depth.z_min, depth.z_max)


def read_confidence(view):
    """Confidence in [0, 1] (8-bit / 255, 16-bit / 65535), or None when the view has none."""
    return None if view.confidence is None else np.clip(_unit(_read(view.confidence)), 0, 1)


def valid_depth(view, min_confidence=0.0):
    z = read_depth(view)
    mask = np.isfinite(z) & (z > 0)
    confidence = read_confidence(view)
    if confidence is not None:
        if confidence.shape != z.shape:
            raise ValueError(f'{view.view_id}: confidence and depth sizes differ')
        mask &= confidence >= min_confidence
    return z, mask
