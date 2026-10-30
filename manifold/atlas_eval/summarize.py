"""Collect runs/<scene>/<arm>/{run,eval}.json into summary.csv / summary.md, plus side-by-side sheets."""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw
from .manifest import load
from .render_eval import METRICS

COLUMNS = ('scene', 'arm', 'group', 'views', *METRICS, 'gaussians', 'ply_mb', 'reconstruct_s', 'peak_gb',
           'center_rmse_relative', 'scale_source')
# Lower is better for these; the rest of METRICS are higher-is-better.
LOWER = {'lpips', 'depth_absrel'}


def rows(runs):
    for arm_dir in sorted(p.parent for p in runs.glob('*/*/eval.json')):
        run = json.loads((arm_dir / 'run.json').read_text())
        result = json.loads((arm_dir / 'eval.json').read_text())
        timing, cuda = run.get('timing_s') or {}, run.get('cuda') or {}
        report = result['alignment']['report']
        peak = cuda.get('peak_allocated_bytes')
        for group, metrics in sorted(result['groups'].items()):
            yield dict(scene=result['scene'], arm=result['arm'], group=group, **metrics,
                       gaussians=run.get('gaussians'), ply_mb=round(run['ply_bytes'] / 2 ** 20, 1),
                       reconstruct_s=timing.get('reconstruct', timing.get('total_with_model_load')),
                       peak_gb=None if peak is None else round(peak / 2 ** 30, 2),
                       center_rmse_relative=report.get('center_rmse_relative'),
                       scale_source=report.get('scale_source', report.get('method')))


def _fmt(value):
    return '' if value is None else f'{value:.4g}' if isinstance(value, float) else str(value)


def markdown(table, baseline):
    lines = ['| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|' * len(COLUMNS)]
    base = {(r['scene'], r['group']): r for r in table if r['arm'] == baseline}
    for r in table:
        cells = []
        for c in COLUMNS:
            cell = _fmt(r[c])
            ref = base.get((r['scene'], r['group']))
            if c in METRICS and ref and r['arm'] != baseline and r[c] is not None and ref[c] is not None:
                delta = r[c] - ref[c]
                better = delta < 0 if c in LOWER else delta > 0
                cell += f' ({delta:+.3g}{"✓" if better else ""})'
            cells.append(cell)
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def sheets(runs, scenes_root, manifest, height):
    """Per scene and test view: ground truth, then every arm's render, each labelled."""
    for scene_runs in sorted(p for p in runs.iterdir() if p.is_dir()):
        arms = sorted(p for p in scene_runs.iterdir() if (p / 'eval.json').is_file())
        if not arms:
            continue
        scene = load(scenes_root / scene_runs.name, manifest)
        out = scene_runs / 'sheets'
        out.mkdir(exist_ok=True)
        for view in scene.tests:
            tiles = [('GT', view.image)] + [(a.name, a / 'eval' / f'{view.view_id}.png') for a in arms
                                            if (a / 'eval' / f'{view.view_id}.png').is_file()]
            images = []
            for label, path in tiles:
                with Image.open(path) as im:
                    im = im.convert('RGB')
                    im = im.resize((round(im.width * height / im.height), height))
                ImageDraw.Draw(im).text((8, 8), label, fill=(255, 255, 0))
                images.append(im)
            sheet = Image.new('RGB', (sum(i.width for i in images), height))
            x = 0
            for im in images:
                sheet.paste(im, (x, 0))
                x += im.width
            sheet.save(out / f'{view.view_id}.jpg', quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, help='Runs root (runs/<scene>/<arm>/)')
    parser.add_argument('--baseline', default='B0', help='Arm the deltas are taken against')
    parser.add_argument('--scenes-root', type=Path, default=None, help='Parent of the scene folders; enables sheets')
    parser.add_argument('--manifest', default='manifest.json')
    parser.add_argument('--sheet-height', type=int, default=360)
    args = parser.parse_args()
    table = list(rows(args.runs))
    if not table:
        parser.error(f'no eval.json under {args.runs}/*/*/')
    with open(args.runs / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(table)
    (args.runs / 'summary.md').write_text(markdown(table, args.baseline))
    if args.scenes_root is not None:
        sheets(args.runs, args.scenes_root, args.manifest, args.sheet_height)
    print(args.runs / 'summary.md')


if __name__ == '__main__':
    main()
