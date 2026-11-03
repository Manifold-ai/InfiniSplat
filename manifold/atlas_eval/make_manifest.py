"""Write manifest.json for a scene folder: from an Atlas scene export, plain photos, or one panorama."""
import argparse
import json
from pathlib import Path
from PIL import Image
from .manifest import IMAGE_SUFFIXES, SCHEMA, is_equirect_size, load, sha256

PANORAMA_DEPTH_SUFFIXES = ('.exr', '.npy', '.png')
# File names of an Atlas scene export (one folder per scene).
ATLAS_JSON = 'atlas-scene.json'
IMAGE = 'scene-{view_id}.png'
DEPTH = 'scene-{view_id}-depth.{format}'
CONFIDENCE = 'scene-{view_id}-confidence.png'


def _check(root, name, record, verify):
    path = root / name
    if not path.is_file():
        raise ValueError(f'missing {name}')
    if verify and record is not None:
        if path.stat().st_size != record['size'] or sha256(path) != record['sha256']:
            raise ValueError(f'{name} does not match the sha256/size recorded in {ATLAS_JSON}')
    return name


def from_atlas(root, test_create=None, test_every=4, verify=True):
    """Active views of atlas-scene.json; create views first, then generate views, each in JSON order.
    Tests: the last `test_create` create views (default 1, or 0 for a single-photo scene; never the
    first) and every `test_every`-th generate view."""
    root = Path(root)
    scene = json.loads((root / ATLAS_JSON).read_text())
    if scene.get('schema_version') != 1:
        raise ValueError(f'{ATLAS_JSON}: expected schema_version 1')
    active = [v for v in scene['views'] if v.get('status', 'active') == 'active']
    create = [v for v in active if v['origin'] == 'create']
    generate = [v for v in active if v['origin'] == 'generate']
    if test_create is None:
        test_create = 1 if len(create) >= 2 else 0
    if test_create < 0 or (test_create and test_create >= len(create)):
        raise ValueError(f'test_create={test_create} must leave at least one create view as input')
    tests = {v['view_id'] for v in create[len(create) - test_create:]} if test_create else set()
    if test_every:
        tests |= {v['view_id'] for i, v in enumerate(generate) if (i + 1) % test_every == 0}
    views = []
    for v in create + generate:
        view_id, files = v['view_id'], v.get('files', {})
        depth = v['depth']
        entry = dict(view_id=view_id, origin=v['origin'], role='test' if view_id in tests else 'input',
                     image=_check(root, IMAGE.format(view_id=view_id), files.get('image'), verify),
                     camera=v['camera'],
                     depth=dict(file=_check(root, DEPTH.format(view_id=view_id, format=depth['format']),
                                            files.get('depth'), verify),
                                **{k: depth[k] for k in ('encoding', 'format', 'z_min', 'z_max') if depth.get(k) is not None}))
        if v.get('confidence') == 'present':
            entry['confidence'] = _check(root, CONFIDENCE.format(view_id=view_id), files.get('confidence'), verify)
        views.append(entry)
    return dict(schema=SCHEMA, name=root.name,
                source=dict(kind='atlas_scene', file=ATLAS_JSON, sha256=sha256(root / ATLAS_JSON),
                            coordinate_system=scene['coordinate_system'],
                            split=dict(test_create=test_create, test_every=test_every)),
                views=views)


def _images(root):
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def from_images(root):
    """Every photo directly inside the folder, in filename order, as an unposed input view."""
    root = Path(root)
    images = _images(root)
    if not images:
        raise ValueError(f'{root}: no PNG, JPG or WebP images')
    if len(images) == 1:
        with Image.open(root / images[0]) as im:
            if is_equirect_size(*im.size):
                raise ValueError(f'{images[0]} is 2:1, a panorama: use `make_manifest pano` (one photo would run v1)')
    views = [dict(view_id=Path(name).stem, origin='photo', role='input', image=name) for name in images]
    return dict(schema=SCHEMA, name=root.name, source=dict(kind='images'), views=views)


def from_panorama(root, log_bounds=None):
    """The one 2:1 panorama directly inside the folder, with an optional range map <stem>-depth.(exr|npy|png)
    (log PNG needs log_bounds=(min, max)) and confidence <stem>-confidence.png."""
    root = Path(root)
    images = [n for n in _images(root) if not Path(n).stem.endswith(('-depth', '-confidence'))]
    if len(images) != 1:
        raise ValueError(f'{root}: a panorama scene folder holds exactly one image, found {len(images)}')
    name = images[0]
    stem = Path(name).stem
    view = dict(view_id=stem, origin='photo', role='input', projection='equirect', image=name)
    depths = [f'{stem}-depth{s}' for s in PANORAMA_DEPTH_SUFFIXES if (root / f'{stem}-depth{s}').is_file()]
    if len(depths) > 1:
        raise ValueError(f'{root}: more than one range map: {depths}')
    if depths:
        fmt = Path(depths[0]).suffix.lstrip('.')
        depth = dict(file=depths[0], kind='range', encoding='log' if fmt == 'png' else 'linear', format=fmt)
        if fmt == 'png':
            if log_bounds is None:
                raise ValueError(f'{depths[0]}: a PNG range map is log encoded; pass --log-bounds MIN MAX')
            depth.update(z_min=log_bounds[0], z_max=log_bounds[1])
        view['depth'] = depth
        if (root / f'{stem}-confidence.png').is_file():
            view['confidence'] = f'{stem}-confidence.png'
    elif log_bounds is not None:
        raise ValueError('--log-bounds given but there is no range map')
    return dict(schema=SCHEMA, name=root.name, source=dict(kind='panorama'), views=[view])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='source', required=True)
    atlas = sub.add_parser('atlas', help=f'Folder holding {ATLAS_JSON} and its view files')
    atlas.add_argument('scene_dir', type=Path)
    atlas.add_argument('--test-create', type=int, default=None, help='Default 1, or 0 with a single create view')
    atlas.add_argument('--test-every', type=int, default=4, help='0 keeps every generate view as input')
    atlas.add_argument('--no-verify', action='store_true', help='Skip the sha256/size check against the JSON')
    images = sub.add_parser('images', help='Folder of photos without cameras')
    images.add_argument('scene_dir', type=Path)
    pano = sub.add_parser('pano', help='Folder with one 2:1 panorama, optionally <stem>-depth.* (range map)')
    pano.add_argument('scene_dir', type=Path)
    pano.add_argument('--log-bounds', type=float, nargs=2, metavar=('MIN', 'MAX'),
                      help='Range bounds of a log-encoded PNG range map')
    for p in (atlas, images, pano):
        p.add_argument('--manifest', default='manifest.json')
        p.add_argument('--force', action='store_true', help='Replace an existing manifest')
    args = parser.parse_args()
    out = args.scene_dir / args.manifest
    if out.exists() and not args.force:
        parser.error(f'{out} exists; pass --force to replace it')
    try:
        if args.source == 'atlas':
            data = from_atlas(args.scene_dir, args.test_create, args.test_every, not args.no_verify)
        elif args.source == 'pano':
            data = from_panorama(args.scene_dir, args.log_bounds)
        else:
            data = from_images(args.scene_dir)
        staged = out.with_name(out.name + '.tmp')
        staged.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        try:
            scene = load(args.scene_dir, staged.name)
        except Exception:
            staged.unlink()
            raise
        staged.replace(out)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    if scene.panorama is not None:
        print(f'{out}: panorama {scene.panorama.view_id}, range map {"yes" if scene.panorama.depth else "no"}')
    else:
        print(f'{out}: {len(scene.inputs)} input, {len(scene.tests)} test views')


if __name__ == '__main__':
    main()
