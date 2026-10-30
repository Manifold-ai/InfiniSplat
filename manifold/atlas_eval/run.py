"""Reconstruct scene folders: one input view runs v1 (single image, optional depth prompt),
two or more run v2 (multi-view). Writes runs/<scene>/<arm>/ with the PLY and run.json."""
import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .manifest import load, select_inputs, sha256, valid_depth
from .splat_io import vertex_count

REPO = Path(__file__).resolve().parents[2]
RUN_SCHEMA = 'infinisplat-atlas-eval-run/1'
V2_MODES = ('provided', 'da3-pose', 'pose-free')
V1_MODES = {'v1-rgb': 'rgb', 'v1-lidar': 'lidar'}
ARMS = {
    'I1': dict(mode='provided', long_side=518),
    'I1-v5': dict(mode='provided', long_side=518, inputs=5),
    'I1-v11': dict(mode='provided', long_side=518, inputs=11),
    'I2': dict(mode='provided', long_side=770),
    'I3': dict(mode='da3-pose', long_side=518),
    'S-rgb': dict(mode='v1-rgb', inputs=1),
    'S-depth': dict(mode='v1-lidar', inputs=1),
}
# Seeds the unseeded prompt-depth sampling of v1 and reports peak CUDA memory on exit.
V1_WRAPPER = '''import atexit, random, runpy, sys
import numpy, torch
random.seed(0); numpy.random.seed(0); torch.manual_seed(0)
def _peak():
    if torch.cuda.is_available():
        print('ATLAS_EVAL_PEAK_BYTES=%d' % torch.cuda.max_memory_allocated(), flush=True)
atexit.register(_peak)
sys.argv = ['infer_batch_images'] + sys.argv[1:]
runpy.run_module('src.demo.infer_batch_images', run_name='__main__', alter_sys=True)
'''


def parse_arm(text):
    """A preset name, or NAME=MODE[:LONG_SIDE[:INPUTS]] such as X=provided:1022:17."""
    if '=' not in text:
        if text not in ARMS:
            raise ValueError(f'unknown arm {text!r}; presets: {", ".join(ARMS)}')
        return text, dict(ARMS[text])
    name, _, value = text.partition('=')
    mode, *rest = value.split(':')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or mode not in V2_MODES + tuple(V1_MODES):
        raise ValueError(f'bad arm {text!r}')
    spec = dict(mode=mode)
    if rest and rest[0]:
        spec['long_side'] = int(rest[0])
    if len(rest) > 1 and rest[1]:
        spec['inputs'] = int(rest[1])
    return name, spec


def auto_arms(scene):
    if len(scene.inputs) >= 2:
        return ['I1']
    return ['S-rgb'] + (['S-depth'] if scene.inputs[0].depth is not None else [])


def plan(scene, name, spec):
    """Check one arm against one scene; returns (spec with defaults, input views)."""
    spec = dict(spec)
    v1 = spec['mode'] in V1_MODES
    views = select_inputs(scene, spec.get('inputs', 1 if v1 else None))
    if v1:
        if len(views) != 1:
            raise ValueError(f'{name}: v1 reconstructs a single image; set inputs to 1')
        if spec['mode'] == 'v1-lidar' and views[0].depth is None:
            raise ValueError(f'{name}: v1-lidar needs depth on {views[0].view_id}')
    else:
        spec.setdefault('long_side', 518)
        if not 2 <= len(views) <= 50:
            raise ValueError(f'{name}: v2 needs 2-50 input views, got {len(views)}; use a v1 arm for one image')
        if spec['mode'] == 'provided' and views[0].c2w is None:
            raise ValueError(f'{name}: provided mode needs camera extrinsics on the input views')
        if not 112 <= spec['long_side'] <= 2044:
            raise ValueError(f'{name}: long_side must be 112-2044')
    spec['backend'] = 'v1' if v1 else 'v2'
    return spec, views


def _git_commit():
    try:
        return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class V2:
    """Loads the v2 pipeline once and reuses it for every v2 arm in this process."""

    def __init__(self, model, device):
        self.model, self.device, self.pipeline, self.load_s = model, device, None, None

    def __call__(self, views, spec, out):
        import torch
        from infinisplat import Pipeline, export_ply
        if self.pipeline is None:
            start = time.perf_counter()
            self.pipeline = Pipeline.from_pretrained(self.model, self.device)
            self.load_s = time.perf_counter() - start
        cameras = None
        if spec['mode'] == 'provided':
            cameras = dict(c2w=np.stack([v.c2w for v in views]).astype(np.float32),
                           intrinsics=np.stack([v.K for v in views]).astype(np.float32))
        free, total = torch.cuda.mem_get_info(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        scene = self.pipeline.reconstruct([str(v.image) for v in views], cameras=cameras,
                                          long_side=spec['long_side'], max_views=max(2, len(views)),
                                          pose_mode=spec['mode'])
        torch.cuda.synchronize(self.device)
        reconstructed = time.perf_counter()
        export_ply(scene, out)
        exported = time.perf_counter()
        peak = torch.cuda.max_memory_allocated(self.device)
        del scene
        torch.cuda.empty_cache()
        result = json.loads((out / 'result.json').read_text())
        return dict(ply='scene.ply', frame=dict(kind='cameras', file='cameras.npz'),
                    timing_s=dict(model_load=self.load_s, reconstruct=reconstructed - start, export=exported - reconstructed),
                    cuda=dict(device=self.device, name=torch.cuda.get_device_name(self.device),
                              peak_allocated_bytes=peak, free_bytes_before=free, total_bytes=total),
                    versions=dict(torch=torch.__version__, weights_sha256=result.get('weights_sha256')))


def prompt_depth(view, min_confidence):
    """Atlas depth rescaled into v1's accepted prompt range (1, 100), centered in log space."""
    z, mask = valid_depth(view, min_confidence)
    if not mask.any():
        raise ValueError(f'{view.view_id}: no valid depth for the prompt')
    p1, p99 = np.percentile(z[mask], [1, 99])
    scale = float(10 / np.sqrt(p1 * p99))
    scaled = np.where(mask, z * scale, np.nan).astype(np.float32)
    kept = float(((scaled > 1) & (scaled < 100)).sum() / mask.sum())
    return scaled, scale, kept


def run_v1(view, spec, out, args):
    out.mkdir(parents=True)
    command = [args.v1_python, '-c', V1_WRAPPER, '--mode', V1_MODES[spec['mode']], '--input', str(view.image.resolve()),
               '--output-dir', str((out / 'v1').resolve()), '--no-video', '--no-export-html', '--device', args.device]
    if view.K is not None:
        (out / 'intrinsics.json').write_text(json.dumps(dict(intrinsics_px=view.K.tolist())) + '\n')
        command += ['--intrinsics-file', str((out / 'intrinsics.json').resolve())]
    scale = depth_info = None
    if spec['mode'] == 'v1-lidar':
        depth, scale, kept = prompt_depth(view, args.min_confidence)
        np.save(out / 'prompt_depth.npy', depth)
        command += ['--prompt-depth', str((out / 'prompt_depth.npy').resolve())]
        depth_info = dict(scale=scale, kept_fraction=kept, min_confidence=args.min_confidence)
    start = time.perf_counter()
    with open(out / 'v1.log', 'w') as log:
        code = subprocess.run(command, cwd=args.v1_root, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.perf_counter() - start
    text = (out / 'v1.log').read_text(errors='replace')
    if code:
        raise RuntimeError(f'v1 exited with {code}; see {out / "v1.log"}')
    stem = view.image.stem
    ply = out / 'v1' / stem / f'{stem}.ply'
    if not ply.is_file():
        raise RuntimeError(f'v1 wrote no PLY at {ply}')
    peak = re.findall(r'ATLAS_EVAL_PEAK_BYTES=(\d+)', text)
    # The input camera is the identity of the v1 frame; a depth prompt carries its scale.
    return dict(ply=str(ply.relative_to(out)), prompt_depth=depth_info,
                frame=dict(kind='camera', view_id=view.view_id, scale=scale),
                timing_s=dict(total_with_model_load=elapsed),
                cuda=dict(device=args.device, peak_allocated_bytes=int(peak[-1]) if peak else None))


def execute(scene, name, spec, views, out, args, v2):
    record = dict(schema=RUN_SCHEMA, arm=name, spec=spec, status='ok',
                  scene=dict(name=scene.name, dir=str(scene.root.resolve()), manifest_sha256=sha256(scene.manifest)),
                  inputs=[v.view_id for v in views], tests=[v.view_id for v in scene.tests],
                  created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  host=dict(python=platform.python_version(), git_commit=_git_commit()))
    try:
        record.update(v2(views, spec, out) if spec['backend'] == 'v2' else run_v1(views[0], spec, out, args))
        ply = out / record['ply']
        record.update(gaussians=vertex_count(ply), ply_bytes=ply.stat().st_size)
    except Exception as error:  # Keep going through the matrix; the failure is recorded in place.
        out.mkdir(parents=True, exist_ok=True)
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record['status'] == 'ok'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, action='append', required=True, help='Scene folder; repeatable')
    parser.add_argument('--manifest', default='manifest.json')
    parser.add_argument('--out', type=Path, required=True, help='Runs root; arms go to OUT/<scene>/<arm>/')
    parser.add_argument('--arm', action='append', default=None,
                        help=f'auto (default), a preset ({", ".join(ARMS)}) or NAME=MODE[:LONG_SIDE[:INPUTS]]')
    parser.add_argument('--model', default='weights/da3', help='v2 weights directory or Hub ID')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--v1-root', type=Path, default=REPO / 'InfiniSplat_v1')
    parser.add_argument('--v1-python', default=sys.executable, help='Python of the v1 environment')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Depth prompt pixels below it are dropped')
    parser.add_argument('--skip-existing', action='store_true', help='Skip arms whose run.json says ok')
    parser.add_argument('--dry-run', action='store_true', help='Print the matrix and exit')
    args = parser.parse_args()
    jobs = []
    try:
        for directory in args.scene:
            scene = load(directory, args.manifest)
            arms = [a for text in (args.arm or ['auto']) for a in (auto_arms(scene) if text == 'auto' else [text])]
            for name, spec in map(parse_arm, dict.fromkeys(arms)):
                spec, views = plan(scene, name, spec)
                jobs.append((scene, name, spec, views, args.out / scene.name / name))
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    pending = []
    for job in jobs:
        out = job[-1]
        if out.exists():
            done = out / 'run.json'
            if args.skip_existing and done.is_file() and json.loads(done.read_text()).get('status') == 'ok':
                continue
            parser.error(f'{out} exists; remove it to rerun, or pass --skip-existing')
        pending.append(job)
    for scene, name, spec, views, out in pending:
        print(f'{scene.name}\t{name}\t{spec["backend"]}\t{spec["mode"]}\t{spec.get("long_side", "-")}\t{len(views)} inputs\t{out}')
    if args.dry_run:
        return
    v2 = V2(args.model, args.device)
    failed = [out for scene, name, spec, views, out in pending if not execute(scene, name, spec, views, out, args, v2)]
    for out in failed:
        print(f'failed: {out / "run.json"}', file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
