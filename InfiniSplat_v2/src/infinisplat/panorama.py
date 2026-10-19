"""Fixed 24-view calibration and equirectangular-to-pinhole projection."""
import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def rotation(yaw, pitch):
    """OpenCV camera-to-world; positive yaw right, positive pitch up."""
    a, b = np.deg2rad([yaw, pitch])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                   [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)],
                   [0, np.sin(b), np.cos(b)]])
    return ry @ rx


def create_template(output, size=770):
    if size < 2:
        raise ValueError('Image size must be at least 2')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    focal = size / (2 * np.tan(np.deg2rad(60) / 2))
    K = np.array([[focal, 0, (size-1)/2], [0, focal, (size-1)/2],
                  [0, 0, 1]], dtype=np.float32)
    poses, views = [], []
    for pitch in (0, -45, 45):
        for yaw in range(0, 360, 45):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotation(yaw, pitch)
            index = len(poses)
            poses.append(pose)
            views.append(dict(index=index, file=f'{index:02d}.png',
                              yaw_deg=yaw, pitch_deg=pitch, roll_deg=0))
    np.savez(output/'cameras.npz', c2w=np.stack(poses),
             intrinsics=np.repeat(K[None], 24, axis=0))
    spec = dict(version=1, name=f'panorama24_fov60_{size}', image_wh=[size, size],
                horizontal_fov_deg=60, vertical_fov_deg=60, views=views,
                camera_convention='OpenCV c2w; x right, y down, z forward',
                world_convention='x right, y down; yaw=0 looks toward panorama center (+z)',
                pixel_convention='integer pixel centers; cx=cy=(size-1)/2',
                source_projection='full 360x180 degree equirectangular RGB panorama',
                source_center='longitude=0 at image center; latitude positive upward',
                scale_status='non_identifiable; all centers zero; no baseline scale fit',
                coverage='polar caps beyond +/-75 degrees are not covered',
                cameras_sha256=sha256(output/'cameras.npz'))
    (output/'template.json').write_text(json.dumps(spec, indent=2)+'\n')
    return spec


def load_template(directory):
    directory = Path(directory)
    spec = json.loads((directory/'template.json').read_text())
    if sha256(directory/'cameras.npz') != spec['cameras_sha256']:
        raise ValueError('Template camera checksum mismatch')
    with np.load(directory/'cameras.npz', allow_pickle=False) as data:
        cameras = {key: data[key].copy() for key in ('c2w', 'intrinsics')}
    poses, K = cameras['c2w'], cameras['intrinsics']
    if poses.shape != (24, 4, 4) or K.shape != (24, 3, 3):
        raise ValueError('Expected 24 c2w and K matrices')
    if not all(np.isfinite(a).all() for a in (poses, K)):
        raise ValueError('Nonfinite camera values')
    if np.count_nonzero(poses[:, :3, 3]) or not np.allclose(poses[:, 3], [0,0,0,1]):
        raise ValueError('Fixed template must have a common zero center')
    if not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-6):
        raise ValueError('Invalid rotation handedness')
    expected = [(p,y) for p in (0,-45,45) for y in range(0,360,45)]
    size = spec['image_wh'][0]
    if spec['image_wh'] != [size, size] or len(spec['views']) != 24:
        raise ValueError('Invalid template dimensions')
    focal = size/(2*np.tan(np.pi/6))
    expected_K = [[focal,0,(size-1)/2],[0,focal,(size-1)/2],[0,0,1]]
    if not np.allclose(K, expected_K, atol=1e-4):
        raise ValueError('Template must use 60-degree FOV and integer pixel centers')
    for i, (pitch,yaw) in enumerate(expected):
        view = spec['views'][i]
        if (view['index'],view['file'],view['pitch_deg'],view['yaw_deg'],view['roll_deg']) != (i,f'{i:02d}.png',pitch,yaw,0):
            raise ValueError('Template view order changed')
        if not np.allclose(poses[i,:3,:3],rotation(yaw,pitch),atol=1e-6):
            raise ValueError('Template rotation does not match its declared direction')
    return spec, cameras


def slice_panorama(source, template, output):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('Install panorama support with pip install ".[panorama]"') from error
    source, template, output = Path(source), Path(template), Path(output)
    spec, cameras = load_template(template)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        with Image.open(source) as im:
            if im.width * im.height > 128_000_000 or max(im.size) >= 32767:
                raise ValueError('Panorama must be at most 128 megapixels and below 32767 pixels per edge')
            if im.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Use a tone-mapped RGB JPG, PNG or WebP panorama')
            if im.getexif().get(274, 1) != 1:
                raise ValueError('Remove EXIF orientation and verify panorama orientation first')
            if abs(im.width / im.height - 2) > .005:
                raise ValueError('Expected a full approximately 2:1 equirectangular panorama')
            pano = np.asarray(im.convert('RGB'))
    height,width = pano.shape[:2]
    if abs(width/height-2) > .005:
        raise ValueError('Expected a full approximately 2:1 equirectangular panorama')
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    size = spec['image_wh'][0]
    v,u = np.mgrid[:size,:size]
    contact = Image.new('RGB', (8*240,3*264), 'white')
    draw = ImageDraw.Draw(contact)
    records = []
    for view, C, K in zip(spec['views'],cameras['c2w'],cameras['intrinsics']):
        rays = np.stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)],-1)
        world = rays @ C[:3,:3].T
        lon = np.arctan2(world[...,0],world[...,2])
        lat = np.arctan2(-world[...,1],np.hypot(world[...,0],world[...,2]))
        mx = ((lon/(2*np.pi)+.5)*width-.5).astype('float32')
        my = np.clip((.5-lat/np.pi)*height-.5,0,height-1).astype('float32')
        rgb = cv2.remap(pano,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        image = Image.fromarray(rgb)
        path = output/'inputs'/view['file']
        image.save(path)
        i = view['index']; col,row = i%8,i//8
        contact.paste(image.resize((240,240),Image.Resampling.LANCZOS),(col*240,row*264+24))
        draw.text((col*240+4,row*264+5),f"{i:02d} yaw {view['yaw_deg']} pitch {view['pitch_deg']:+d}",fill='black')
        records.append(dict(**view,sha256=sha256(path),bytes=path.stat().st_size))
    # Copy the same artifact verbatim: every panorama receives identical cameras.
    shutil.copyfile(template/'cameras.npz',output/'cameras.npz')
    shutil.copyfile(template/'template.json',output/'template.json')
    contact.save(output/'contact_sheet.jpg',quality=92)
    manifest = dict(source=source.name,source_sha256=sha256(source),
                    source_wh=[width,height],output_wh=[size,size],
                    template_sha256=sha256(template/'template.json'),
                    cameras_sha256=sha256(output/'cameras.npz'),
                    sampling='OpenCV bilinear, horizontal wrap, vertical clamp',views=records)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

