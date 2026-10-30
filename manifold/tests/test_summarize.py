import json
import subprocess
import sys
from PIL import Image
from manifold.atlas_eval.make_manifest import from_atlas
from manifold.atlas_eval.render_eval import METRICS
from manifold.atlas_eval.run import REPO
from .synthetic import H, W, write_atlas_scene
from .test_splat_io import V1_ORDER, _write


def _cli(*args):
    return subprocess.run([sys.executable, '-m', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def test_imported_baseline_and_summary(tmp_path):
    scenes, runs = tmp_path / 'scenes', tmp_path / 'runs'
    write_atlas_scene(scenes / 'S1', create=3, generate=4, removed=0)
    (scenes / 'S1' / 'manifest.json').write_text(json.dumps(from_atlas(scenes / 'S1')))
    _write(tmp_path / 'surfel.ply', V1_ORDER, {n: [0.0, 1.0] for n in V1_ORDER})
    _cli('manifold.atlas_eval.import_splat', '--scene', str(scenes / 'S1'), '--ply', str(tmp_path / 'surfel.ply'),
         '--out', str(runs), '--frame', 'flip-yz', '--note', 'surfel defaults')
    run = json.loads((runs / 'S1' / 'B0' / 'run.json').read_text())
    assert run['gaussians'] == 2 and run['frame']['rotation'][1][1] == -1
    tests = ['v_000000000002', 'v_000000000006']
    for arm, lpips in (('B0', 0.40), ('I1', 0.30)):
        arm_dir = runs / 'S1' / arm
        (arm_dir / 'eval').mkdir(parents=True)
        if arm == 'I1':
            (arm_dir / 'run.json').write_text(json.dumps(dict(run, arm='I1', timing_s=dict(reconstruct=12.0),
                                                              cuda=dict(peak_allocated_bytes=3 * 2 ** 30))))
        metrics = {m: 0.5 for m in METRICS} | dict(lpips=lpips)
        (arm_dir / 'eval.json').write_text(json.dumps(dict(
            scene='S1', arm=arm, alignment=dict(report=dict(method='fixed')),
            groups={'test:all': dict(views=2, **metrics)})))
        for view_id in tests:
            Image.new('RGB', (W, H)).save(arm_dir / 'eval' / f'{view_id}.png')
    _cli('manifold.atlas_eval.summarize', str(runs), '--scenes-root', str(scenes))
    table = (runs / 'summary.md').read_text()
    assert '| S1 | I1 | test:all | 2 |' in table and '0.3 (-0.1✓)' in table and '| 12 | 3 |' in table
    assert sorted(p.name for p in (runs / 'S1' / 'sheets').iterdir()) == [f'{t}.jpg' for t in tests]
    assert Image.open(runs / 'S1' / 'sheets' / f'{tests[0]}.jpg').size == (3 * 640, 360)
