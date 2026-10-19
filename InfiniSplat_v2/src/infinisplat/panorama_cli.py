"""Reconstruct native learned 3D Gaussians from a full equirectangular panorama."""
import argparse
import json
from pathlib import Path
import tempfile

from .panorama import create_template, slice_panorama, load_template, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Full 360 x 180 degree, approximately 2:1 RGB panorama')
    parser.add_argument('--model', default='Edisoneh/InfiniSplat-v2')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--long-side', type=int, default=768, help='Square view resolution, rounded to a multiple of 14 (default: 768 -> 770)')
    parser.add_argument('--no-html', action='store_true', help='Export native Gaussian PLY without HTML')
    args = parser.parse_args()
    if not 112 <= args.long_side <= 2044:
        parser.error('--long-side must be between 112 and 2044')
    if args.output.exists():
        parser.error('--output must be a new directory')
    if not args.input.is_file():
        parser.error('--input must be an existing panorama image')
    try:
        if not args.no_html:
            from .viewer import require_html_converter
            require_html_converter()
        from .pipeline import Pipeline
        from .export import export_ply
        from .viewer import export_html
        # Temporary preprocessing leaves no partial output on invalid input or
        # failed inference. Only the export creates the requested destination.
        with tempfile.TemporaryDirectory(prefix='infinisplat-panorama-') as tmp:
            root = Path(tmp)
            size = round(args.long_side / 14) * 14
            create_template(root / 'template', size)
            manifest = slice_panorama(args.input, root / 'template', root / 'prepared')
            spec, cameras = load_template(root / 'template')
            paths = [root / 'prepared' / 'inputs' / view['file'] for view in spec['views']]
            pipeline = Pipeline.from_pretrained(args.model, args.device)
            scene = pipeline.reconstruct(paths, cameras=cameras, pose_mode='provided',
                                         long_side=args.long_side, max_views=24,
                                         panorama_shared_center=True)
            scene['metadata']['panorama'].update(
                template=spec, source=manifest, requested_long_side=args.long_side)
            ply = export_ply(scene, args.output)
            import shutil
            shutil.copyfile(root / 'prepared' / 'contact_sheet.jpg', args.output / 'contact_sheet.jpg')
            (args.output / 'template.json').write_text(json.dumps(spec, indent=2) + '\n')
            # Template checksum describes the exact input cameras, separate from
            # the effective reconstruction cameras exported by export_ply.
            shutil.copyfile(root / 'template' / 'cameras.npz', args.output / 'input_cameras.npz')
        del scene, pipeline
        if not args.no_html:
            viewer = export_html(ply, args.output / 'scene.html', display_transform=dict(
                quaternion=[1, 0, 0, 0], translation=[0, 0, 0],
                camera=dict(position=[0, 0, 0], target=[0, 0, -1], fov=90)))
        else:
            viewer = None
        result_path = args.output / 'result.json'
        metadata = json.loads(result_path.read_text())
        if viewer is not None:
            metadata['viewer'] = viewer
        for name in ('scene.html', 'contact_sheet.jpg', 'template.json', 'input_cameras.npz'):
            path = args.output / name
            if path.exists():
                metadata['files'][name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        result_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(ply)
    if not args.no_html:
        print(args.output / 'scene.html')


if __name__ == '__main__':
    main()
