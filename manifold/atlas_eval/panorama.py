"""Panorama arm (produce only): one full equirectangular RGB panorama -> an InfiniSplat v2 splat.

Runs the same steps as `infinisplat-panorama` (24 fixed 60-degree views around one centre, provided
mode, shared centre), reusing the pipeline already loaded by `run`. There is no world pose: the splat
frame is the panorama's own. A range map on the panorama, when given, sets the scale."""
import json
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
from .manifest import valid_depth

FRAME = dict(kind='panorama', origin='capture centre',
             axes='OpenCV: x right, y down, z toward the panorama centre column (longitude 0)',
             level='assumed: horizon on the middle row; no tilt or flip correction')
VIEWS = 24
# Every 16th raw (pre-merge) point: about 0.9 M rays at 770 x 770 per view.
SCALE_STRIDE = 16
MIN_SCALE_SAMPLES = 100
# The initial viewer camera infinisplat-panorama gives scene.html (capture centre, y up).
HTML_VIEW = dict(quaternion=[1, 0, 0, 0], translation=[0, 0, 0],
                 camera=dict(position=[0, 0, 0], target=[0, 0, -1], fov=90))


def view_size(long_side):
    """The square per-view resolution infinisplat-panorama makes of --long-side (768 -> 770)."""
    return round(long_side / 14) * 14


def equirect_pixel(directions, height, width):
    """Nearest (row, col) of a height x width equirect image for directions [..., 3] in the panorama
    frame, with the convention of infinisplat.panorama.slice_panorama: longitude 0 (+z) at the image
    centre, positive longitude toward +x, latitude positive up (-y)."""
    d = np.asarray(directions, dtype=np.float64)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arctan2(-d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    col = np.rint((lon / (2 * np.pi) + .5) * width - .5).astype(np.int64) % width
    row = np.clip(np.rint((.5 - lat / np.pi) * height - .5), 0, height - 1).astype(np.int64)
    return row, col


def depth_scale(view, points, min_confidence=0.0):
    """Range-map units per splat unit: the median of given / predicted range over predicted points
    [M, 3] in the panorama frame (centre at the origin) that land on valid range-map pixels."""
    target, mask = valid_depth(view, min_confidence)
    height, width = target.shape
    if abs(width / height - 2) > .005:
        raise ValueError(f'{view.view_id}: range map is {width}x{height}, expected 2:1')
    points = np.asarray(points, dtype=np.float64)
    predicted = np.linalg.norm(points, axis=1)
    hit = np.isfinite(predicted) & (predicted > 1e-8)
    row, col = equirect_pixel(points[hit] / predicted[hit, None], height, width)
    ok = mask[row, col]
    if ok.sum() < MIN_SCALE_SAMPLES:
        raise ValueError(f'{view.view_id}: only {int(ok.sum())} predicted points land on valid depth; '
                         'cannot set the scale')
    ratio = target[row, col][ok] / predicted[hit][ok]
    return dict(factor=float(np.median(ratio)), samples=int(ok.sum()), points=int(len(points)),
                ratio_p10_p90=[float(v) for v in np.percentile(ratio, [10, 90])],
                min_confidence=min_confidence)


def reconstruct(pipeline, view, spec, out, device, html=False, min_confidence=0.0):
    """Slice, reconstruct, optionally rescale, export. Returns the run.json fields of the arm."""
    import torch
    from infinisplat import export_ply
    from infinisplat.panorama import create_template, load_template, sha256, slice_panorama
    if html:
        from infinisplat.viewer import export_html, require_html_converter
        require_html_converter()
    size = view_size(spec['long_side'])
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory(prefix='atlas-eval-panorama-') as tmp:
        root = Path(tmp)
        start = time.perf_counter()
        create_template(root / 'template', size)
        source = slice_panorama(view.image, root / 'template', root / 'prepared')
        template, cameras = load_template(root / 'template')
        paths = [root / 'prepared' / 'inputs' / v['file'] for v in template['views']]
        sliced = time.perf_counter()
        scene = pipeline.reconstruct(paths, cameras=cameras, pose_mode='provided', long_side=spec['long_side'],
                                     max_views=VIEWS, panorama_shared_center=True)
        torch.cuda.synchronize(device)
        reconstructed = time.perf_counter()
        scale = dict(status='unknown', factor=None,
                     units='native: mean predicted point radius normalisation of the released model')
        if view.depth is not None:
            points = scene['raw_points'][::SCALE_STRIDE].float().cpu().numpy()
            scale = dict(status='depth', units='range map units', **depth_scale(view, points, min_confidence),
                         depth_sha256=sha256(view.depth.path),
                         note='PLY means and scales are multiplied by factor; result.json stats stay native')
            factor = scale['factor']
            scene['splats']['means'] = scene['splats']['means'] * factor
            scene['splats']['scales'] = scene['splats']['scales'] * factor
        scene['metadata']['panorama'].update(template=template, source=source, requested_long_side=spec['long_side'])
        # The upstream record says the scale is not identifiable; say what this export did.
        scene['metadata']['manifold_scale'] = scale
        export_ply(scene, out)
        shutil.copyfile(root / 'prepared' / 'contact_sheet.jpg', out / 'contact_sheet.jpg')
        (out / 'template.json').write_text(json.dumps(template, indent=2) + '\n')
        shutil.copyfile(root / 'template' / 'cameras.npz', out / 'input_cameras.npz')
    exported = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(device)
    weights = scene['metadata'].get('weights_sha256')
    del scene
    torch.cuda.empty_cache()
    result_path = out / 'result.json'
    metadata = json.loads(result_path.read_text())
    if html:
        metadata['viewer'] = export_html(out / 'scene.ply', out / 'scene.html', display_transform=HTML_VIEW)
    for name in ('scene.html', 'contact_sheet.jpg', 'template.json', 'input_cameras.npz'):
        path = out / name
        if path.exists():
            metadata['files'][name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    result_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return dict(ply='scene.ply', frame=FRAME, scale=scale,
                panorama=dict(source_wh=source['source_wh'], view_wh=[size, size], views=VIEWS,
                              template=template['name'], html='scene.html' if html else None),
                timing_s=dict(slice=sliced - start, reconstruct=reconstructed - sliced, export=exported - reconstructed),
                cuda=dict(device=device, name=torch.cuda.get_device_name(device), peak_allocated_bytes=peak,
                          free_bytes_before=free, total_bytes=total),
                versions=dict(torch=torch.__version__, weights_sha256=weights))
