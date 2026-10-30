"""Render each arm at the manifest's test cameras and score it -> <arm>/eval.json and <arm>/eval/*.png.
Needs CUDA, torch, gsplat and lpips (see manifold/requirements-eval.txt)."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from .align import write as align
from .manifest import load, valid_depth
from .splat_io import load_gaussians

EVAL_SCHEMA = 'infinisplat-atlas-eval-score/1'
METRICS = ('psnr', 'ssim', 'lpips', 'psnr_masked', 'ssim_masked', 'coverage',
           'depth_absrel', 'depth_delta1', 'depth_coverage')


class Renderer:
    def __init__(self, ply, device, background):
        import torch
        g = load_gaussians(ply)
        t = lambda a: torch.from_numpy(a).to(device)
        self.torch, self.device = torch, device
        self.means, self.quats, self.scales = t(g['means']), t(g['quats']), t(g['scales'])
        self.opacities, self.sh = t(g['opacities']), t(g['sh0'])[:, None, :]
        self.background = torch.tensor(background, dtype=torch.float32, device=device)

    def __call__(self, c2w, K, size):
        from gsplat import rasterization
        torch = self.torch
        viewmat = torch.from_numpy(np.linalg.inv(c2w)).float().to(self.device)
        Ks = torch.from_numpy(K).float().to(self.device)
        width, height = size
        with torch.no_grad():
            render, alpha, _ = rasterization(self.means, self.quats, self.scales, self.opacities, self.sh,
                                             viewmat[None], Ks[None], width, height, sh_degree=0,
                                             render_mode='RGB+ED', backgrounds=self.background[None])
        return render[0, ..., :3].clamp(0, 1), render[0, ..., 3], alpha[0, ..., 0]


def _ssim_map(a, b, torch):
    """Gaussian SSIM map (11x11, sigma 1.5) of [H,W,3] images in [0,1], averaged over channels."""
    x = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    w = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
    w = (w[:, None] * w[None, :]) / w.sum() ** 2
    w = w.expand(3, 1, 11, 11)
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    blur = lambda z: torch.nn.functional.conv2d(z, w, padding=5, groups=3)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return s[0].mean(0)


def _size(view):
    if view.size is not None:
        return view.size
    with Image.open(view.image) as im:
        return im.size


def _psnr(mse):
    return float(-10 * np.log10(max(mse, 1e-12)))


def score(view, rgb, depth, alpha, scale, lpips_fn, min_confidence, torch):
    with Image.open(view.image) as im:
        gt = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255).to(rgb.device)
    if gt.shape != rgb.shape:
        raise ValueError(f'{view.view_id}: render {tuple(rgb.shape)} vs image {tuple(gt.shape)}')
    ssim = _ssim_map(rgb, gt, torch)
    with torch.no_grad():
        lp = lpips_fn(rgb.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1)
    result = dict(psnr=_psnr(float(((rgb - gt) ** 2).mean())), ssim=float(ssim.mean()), lpips=float(lp),
                  coverage=float((alpha > 0.5).float().mean()))
    for key in ('psnr_masked', 'ssim_masked', 'depth_absrel', 'depth_delta1', 'depth_coverage'):
        result[key] = None
    if view.depth is not None:
        z, valid = valid_depth(view, min_confidence)
        valid = torch.from_numpy(valid).to(rgb.device)
        if valid.any():
            result['psnr_masked'] = _psnr(float(((rgb - gt) ** 2)[valid].mean()))
            result['ssim_masked'] = float(ssim[valid].mean())
            hit = valid & (alpha > 0.5)
            result['depth_coverage'] = float(hit.sum() / valid.sum())
            if hit.any():
                gt_z = torch.from_numpy(np.nan_to_num(z)).to(rgb.device)[hit]
                pred = depth[hit] / scale
                result['depth_absrel'] = float(((pred - gt_z).abs() / gt_z).mean())
                result['depth_delta1'] = float((torch.maximum(pred / gt_z, gt_z / pred) < 1.25).float().mean())
    return result


def depth_ratio(view, render, sim, min_confidence, torch):
    """Median rendered/manifest depth at an input view: the splat units per world unit."""
    probe = sim if sim.scale is not None else sim.with_scale(1.0, view.c2w[:3, 3])
    _, depth, alpha = render(probe.camera(view.c2w), view.K, _size(view))
    z, valid = valid_depth(view, min_confidence)
    hit = torch.from_numpy(valid).to(depth.device) & (alpha > 0.5)
    if not hit.any():
        raise ValueError(f'{view.view_id}: no overlap between rendered and manifest depth')
    gt = torch.from_numpy(np.nan_to_num(z)).to(depth.device)[hit]
    return float((depth[hit] / gt).median())


def evaluate(scene, arm_dir, args, lpips_fn):
    import torch
    run = json.loads((arm_dir / 'run.json').read_text())
    if run['status'] != 'ok':
        raise ValueError(f'{arm_dir}: run status is {run["status"]}')
    sim, report = align(scene, arm_dir)
    render = Renderer(arm_dir / run['ply'], args.device, args.background)
    anchor = scene.view(run['inputs'][0]) if run.get('inputs') else None
    if anchor is not None and anchor.depth is not None:
        ratio = depth_ratio(anchor, render, sim, args.min_confidence, torch)
        report['depth_scale_check'] = dict(view_id=anchor.view_id, rendered_over_manifest=ratio)
        if sim.scale is None:
            sim = sim.with_scale(ratio, anchor.c2w[:3, 3])
            report['scale_source'] = 'rendered_depth_ratio'
    if sim.scale is None:
        raise ValueError(f'{arm_dir}: scale is not identifiable without depth on the first input view')
    views = [v for v in scene.views if v.role == 'test' or args.with_inputs]
    (arm_dir / 'eval').mkdir(exist_ok=True)
    rows = []
    for view in views:
        rgb, depth, alpha = render(sim.camera(view.c2w), view.K, _size(view))
        Image.fromarray((rgb.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(arm_dir / 'eval' / f'{view.view_id}.png')
        row = dict(view_id=view.view_id, role=view.role, origin=view.origin)
        row.update(score(view, rgb, depth, alpha, sim.scale, lpips_fn, args.min_confidence, torch))
        rows.append(row)
    groups = {}
    for row in rows:
        for key in (f'{row["role"]}:{row["origin"]}', f'{row["role"]}:all'):
            groups.setdefault(key, []).append(row)
    summary = {key: dict(views=len(items), **{m: (float(np.mean(vals)) if (vals := [r[m] for r in items if r[m] is not None]) else None)
                                                for m in METRICS}) for key, items in groups.items()}
    result = dict(schema=EVAL_SCHEMA, arm=run['arm'], scene=scene.name, alignment=dict(sim3=sim.to_json(), report=report),
                  settings=dict(lpips_net=args.lpips_net, background=args.background, min_confidence=args.min_confidence),
                  created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'), groups=summary, views=rows)
    (arm_dir / 'eval.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--manifest', default='manifest.json')
    parser.add_argument('arms', type=Path, nargs='+', help='Arm directories (runs/<scene>/<arm>)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lpips-net', choices=('alex', 'vgg'), default='alex')
    parser.add_argument('--background', type=lambda s: [float(v) for v in s.split(',')], default=[0.0, 0.0, 0.0])
    parser.add_argument('--min-confidence', type=float, default=0.0)
    parser.add_argument('--with-inputs', action='store_true', help='Also score the input views (reconstruction sanity)')
    args = parser.parse_args()
    import lpips
    scene = load(args.scene, args.manifest)
    lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).to(args.device)
    failed = []
    for arm in (a for a in args.arms if (a / 'run.json').is_file()):  # a glob also matches sheets/
        try:
            result = evaluate(scene, arm, args, lpips_fn)
        except (OSError, KeyError, ValueError) as error:
            failed.append(arm)
            print(f'{arm}: {error}', file=sys.stderr)
            continue
        test = result['groups'].get('test:all', {})
        print(arm, {k: test.get(k) for k in ('views', 'psnr', 'ssim', 'lpips', 'depth_absrel')})
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
