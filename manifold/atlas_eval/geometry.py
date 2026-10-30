"""Camera conventions and the similarity transform from manifest world to a splat frame."""
import numpy as np

# rub (right, up, back; looks down -Z) <-> rdf / OpenCV (right, down, forward).
RUB_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
# A whole-world flip (x, -y, -z), e.g. a +Y-up Atlas world into an OpenCV-style world.
FLIP_YZ = np.diag([1.0, -1.0, -1.0])


def quaternion_to_matrix(xyzw):
    q = np.asarray(xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('camera quaternion must be finite and nonzero')
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def opencv_c2w(position, quaternion_xyzw, coordinate_system):
    """Atlas extrinsics (camera-to-world, xyzw) -> 4x4 OpenCV camera-to-world in the same world."""
    if coordinate_system not in ('rub', 'rdf'):
        raise ValueError('coordinate_system must be rub or rdf')
    rotation = quaternion_to_matrix(quaternion_xyzw)
    if coordinate_system == 'rub':
        rotation = rotation @ RUB_TO_OPENCV
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = np.asarray(position, dtype=np.float64)
    if not np.isfinite(c2w).all():
        raise ValueError('camera position must be finite')
    return c2w


def intrinsics_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def project_rotation(matrix):
    """Nearest proper rotation to a 3x3 matrix."""
    u, _, vt = np.linalg.svd(matrix)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def rotation_angle_deg(a, b):
    # ||A - B||_F = 2 sqrt(2) sin(theta / 2); unlike arccos of the trace, exact near zero.
    chord = np.linalg.norm(np.asarray(a) - np.asarray(b)) / (2 * np.sqrt(2))
    return float(np.degrees(2 * np.arcsin(min(1.0, chord))))


class Sim3:
    """p_splat = scale * R @ p_world + t. scale is None while it is not identifiable."""

    def __init__(self, rotation, translation, scale):
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.translation = np.asarray(translation, dtype=np.float64)
        self.scale = None if scale is None else float(scale)

    def with_scale(self, scale, anchor_world):
        """Fix an unknown scale while keeping anchor_world (an input camera center) where it maps now."""
        anchor_splat = self.rotation @ anchor_world * (self.scale or 0.0) + self.translation
        return Sim3(self.rotation, anchor_splat - scale * self.rotation @ anchor_world, scale)

    def camera(self, c2w_world):
        """OpenCV camera-to-world in manifest world -> camera-to-world in the splat frame."""
        if self.scale is None:
            raise ValueError('scale is not identifiable yet')
        c2w = np.eye(4)
        c2w[:3, :3] = self.rotation @ c2w_world[:3, :3]
        c2w[:3, 3] = self.scale * self.rotation @ c2w_world[:3, 3] + self.translation
        return c2w

    def to_json(self):
        return dict(scale=self.scale, rotation=self.rotation.tolist(), translation=self.translation.tolist())

    @classmethod
    def from_json(cls, data):
        return cls(data['rotation'], data['translation'], data['scale'])


def fit_cameras(world_c2w, splat_c2w):
    """Sim(3) from paired cameras: rotation from the camera rotations, then scale and translation by
    least squares on the centers. Works for two views and collinear rigs; scale is None when every
    center coincides (pure rotation)."""
    world_c2w, splat_c2w = np.asarray(world_c2w, float), np.asarray(splat_c2w, float)
    if world_c2w.shape != splat_c2w.shape or world_c2w.ndim != 3 or len(world_c2w) < 1:
        raise ValueError('cameras must be paired [N,4,4] arrays')
    rotation = project_rotation(sum(s[:3, :3] @ w[:3, :3].T for w, s in zip(world_c2w, splat_c2w)))
    cw, cs = world_c2w[:, :3, 3], splat_c2w[:, :3, 3]
    mw, ms = cw.mean(0), cs.mean(0)
    spread = float(((cw - mw) ** 2).sum())
    extent = float(np.sqrt(((cw - mw) ** 2).sum(1).mean()))
    scale = None
    if spread > 1e-12:
        scale = float(((cs - ms) * ((cw - mw) @ rotation.T)).sum() / spread)
        if scale <= 0:
            raise ValueError('camera fit gave a nonpositive scale; check view order and conventions')
    translation = ms - (scale or 0.0) * rotation @ mw
    sim = Sim3(rotation, translation, scale)
    report = dict(views=len(cw), world_extent=extent,
                  rotation_error_deg=[rotation_angle_deg(rotation @ w[:3, :3], s[:3, :3])
                                      for w, s in zip(world_c2w, splat_c2w)])
    if scale is not None:
        residual = np.linalg.norm(scale * cw @ rotation.T + translation - cs, axis=1) / scale
        report.update(center_rmse_world=float(np.sqrt((residual ** 2).mean())),
                      center_rmse_relative=float(np.sqrt((residual ** 2).mean()) / extent) if extent > 0 else None)
    return sim, report


def camera_frame(c2w_world, scale):
    """Frame of a single-image model whose input camera is the identity: p = scale * w2c @ p_world."""
    rotation = c2w_world[:3, :3].T
    center = c2w_world[:3, 3]
    if scale is None:
        return Sim3(rotation, np.zeros(3), None)
    return Sim3(rotation, -scale * rotation @ center, scale)
