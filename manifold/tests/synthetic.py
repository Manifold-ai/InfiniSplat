"""A tiny synthetic Atlas scene export: a ring of rub cameras around a plane at z = -3."""
import hashlib
import json
import os
import numpy as np

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402

W, H = 64, 36
INTRINSICS = dict(fx=60.0, fy=60.0, cx=32.0, cy=18.0, width=W, height=H)


def matrix_to_quaternion(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return [float(x), float(y), float(z), float(w)]


def rub_look_at(position, target, up=(0.0, 1.0, 0.0)):
    """rub camera-to-world rotation: the camera's -Z points at target."""
    back = np.asarray(position, float) - np.asarray(target, float)
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back], axis=1)


def _sha(path):
    data = path.read_bytes()
    return dict(sha256=hashlib.sha256(data).hexdigest(), size=len(data))


def write_atlas_scene(root, create=3, generate=8, removed=1, log_depth_views=(), no_confidence=()):
    """Writes atlas-scene.json and the view files; returns the JSON as written."""
    root.mkdir(parents=True, exist_ok=True)
    views = []
    origins = ['create'] * create + ['generate'] * generate + ['create'] * removed
    for i, origin in enumerate(origins):
        view_id = f'v_{i:012x}'
        angle = 2 * np.pi * i / len(origins)
        position = [2 * np.sin(angle), 0.1 * i, 2 * np.cos(angle)]
        rotation = rub_look_at(position, (0, 0, -3))
        rgb = np.full((H, W, 3), (i * 20) % 255, np.uint8)
        cv2.imwrite(str(root / f'scene-{view_id}.png'), rgb)
        z = np.linspace(1.0, 4.0, W * H, dtype=np.float32).reshape(H, W)
        z[0, 0] = np.nan
        if i in log_depth_views:
            z_min, z_max = 0.5, 8.0
            t = 1 - np.log(np.nan_to_num(z, nan=z_max) / z_min) / np.log(z_max / z_min)
            cv2.imwrite(str(root / f'scene-{view_id}-depth.png'), np.round(t * 65535).astype(np.uint16))
            depth = dict(encoding='log', format='png', z_min=z_min, z_max=z_max)
        else:
            cv2.imwrite(str(root / f'scene-{view_id}-depth.exr'), z)
            depth = dict(encoding='linear', format='exr')
        files = dict(image=_sha(root / f'scene-{view_id}.png'),
                     depth=_sha(root / f'scene-{view_id}-depth.{depth["format"]}'))
        confidence = 'absent' if i in no_confidence else 'present'
        if confidence == 'present':
            cv2.imwrite(str(root / f'scene-{view_id}-confidence.png'), np.full((H, W), 200, np.uint8))
            files['confidence'] = _sha(root / f'scene-{view_id}-confidence.png')
        views.append(dict(
            view_id=view_id, status='removed' if i >= create + generate else 'active', origin=origin,
            added_by='wf-test', depth=depth, confidence=confidence, files=files, source={},
            camera=dict(intrinsics=INTRINSICS,
                        extrinsics=dict(position=position, quaternion_xyzw=matrix_to_quaternion(rotation),
                                        coordinate_system='rub'))))
    scene = dict(schema_version=1, coordinate_system='rub', intrinsics=INTRINSICS, created_by='wf-test',
                 pose_refinement_accepted=True, views=views)
    (root / 'atlas-scene.json').write_text(json.dumps(scene, indent=2))
    return scene
