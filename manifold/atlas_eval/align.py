"""Similarity transform from manifest world into each arm's splat frame -> <arm>/alignment.json."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import Sim3, camera_frame, fit_cameras
from .manifest import load


def compute(scene, arm_dir):
    """Returns (Sim3, report). The scale stays None where only rendering can recover it."""
    run = json.loads((arm_dir / 'run.json').read_text())
    frame = run['frame']
    if frame['kind'] == 'cameras':  # v2: effective cameras in the native first-camera-relative frame
        cameras = np.load(arm_dir / frame['file'])['c2w']
        views = [scene.view(i) for i in run['inputs']]
        if len(cameras) != len(views):
            raise ValueError(f'{arm_dir}: {len(cameras)} output cameras for {len(views)} inputs')
        if any(v.c2w is None for v in views):
            raise ValueError(f'{arm_dir}: inputs have no manifest cameras to align against')
        sim, report = fit_cameras(np.stack([v.c2w for v in views]), cameras)
        report['method'] = 'camera_fit'
        return sim, report
    if frame['kind'] == 'camera':  # v1: the input camera is the identity of the splat frame
        view = scene.view(frame['view_id'])
        if view.c2w is None:
            raise ValueError(f'{arm_dir}: input view has no manifest camera')
        return camera_frame(view.c2w, frame['scale']), dict(method='input_camera', scale_source=
                                                             'depth_prompt' if frame['scale'] else 'render')
    if frame['kind'] == 'world':
        rotation = np.eye(3) if frame['rotation'] is None else np.asarray(frame['rotation'])
        return Sim3(rotation, np.zeros(3), 1.0), dict(method='fixed')
    raise ValueError(f'{arm_dir}: unknown frame kind {frame["kind"]!r}')


def write(scene, arm_dir):
    sim, report = compute(scene, arm_dir)
    (arm_dir / 'alignment.json').write_text(json.dumps(dict(sim3=sim.to_json(), report=report), indent=2) + '\n')
    return sim, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--manifest', default='manifest.json')
    parser.add_argument('arms', type=Path, nargs='+', help='Arm directories (runs/<scene>/<arm>)')
    args = parser.parse_args()
    scene = load(args.scene, args.manifest)
    for arm in args.arms:
        sim, report = write(scene, arm)
        print(arm, f'scale={sim.scale}', {k: v for k, v in report.items() if k != 'rotation_error_deg'})


if __name__ == '__main__':
    main()
