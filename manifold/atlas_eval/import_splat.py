"""Register a splat made elsewhere (for example the surfel baseline) as an arm of a scene."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from .geometry import FLIP_YZ
from .manifest import load, sha256
from .run import RUN_SCHEMA
from .splat_io import vertex_count

FRAMES = {'world': None, 'flip-yz': FLIP_YZ}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--manifest', default='manifest.json')
    parser.add_argument('--ply', type=Path, required=True, help='SH0 Gaussian PLY')
    parser.add_argument('--out', type=Path, required=True, help='Runs root; the arm goes to OUT/<scene>/<arm>/')
    parser.add_argument('--arm', default='B0')
    parser.add_argument('--frame', choices=tuple(FRAMES), default='world',
                        help='world: PLY is in manifest world; flip-yz: PLY is (x, -y, -z) of it')
    parser.add_argument('--inputs', nargs='*', default=None, help='View ids the splat was built from')
    parser.add_argument('--note', default=None, help='Free text, e.g. producer and parameters')
    args = parser.parse_args()
    try:
        scene = load(args.scene, args.manifest)
        unknown = set(args.inputs or ()) - {v.view_id for v in scene.views}
        if unknown:
            raise ValueError(f'unknown view ids {sorted(unknown)}')
        count = vertex_count(args.ply)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    out = args.out / scene.name / args.arm
    if out.exists():
        parser.error(f'{out} exists')
    out.mkdir(parents=True)
    shutil.copy2(args.ply, out / 'scene.ply')
    matrix = FRAMES[args.frame]
    record = dict(schema=RUN_SCHEMA, arm=args.arm, spec=dict(backend='external', frame=args.frame), status='ok',
                  scene=dict(name=scene.name, dir=str(scene.root.resolve()), manifest_sha256=sha256(scene.manifest)),
                  inputs=args.inputs, tests=[v.view_id for v in scene.tests], note=args.note,
                  created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  ply='scene.ply', source=dict(path=str(args.ply.resolve()), sha256=sha256(args.ply)),
                  frame=dict(kind='world', rotation=None if matrix is None else matrix.tolist()),
                  gaussians=count, ply_bytes=(out / 'scene.ply').stat().st_size)
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
