"""Write manifest.json for a scene folder: from an Atlas scene export, or from plain photos."""
import argparse
import json
from pathlib import Path
from .manifest import IMAGE_SUFFIXES, SCHEMA, load, sha256

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


def from_images(root):
    """Every photo directly inside the folder, in filename order, as an unposed input view."""
    root = Path(root)
    images = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f'{root}: no PNG, JPG or WebP images')
    views = [dict(view_id=Path(name).stem, origin='photo', role='input', image=name) for name in images]
    return dict(schema=SCHEMA, name=root.name, source=dict(kind='images'), views=views)


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
    for p in (atlas, images):
        p.add_argument('--manifest', default='manifest.json')
        p.add_argument('--force', action='store_true', help='Replace an existing manifest')
    args = parser.parse_args()
    out = args.scene_dir / args.manifest
    if out.exists() and not args.force:
        parser.error(f'{out} exists; pass --force to replace it')
    try:
        data = (from_atlas(args.scene_dir, args.test_create, args.test_every, not args.no_verify)
                if args.source == 'atlas' else from_images(args.scene_dir))
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
    print(f'{out}: {len(scene.inputs)} input, {len(scene.tests)} test views')


if __name__ == '__main__':
    main()
