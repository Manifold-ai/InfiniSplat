"""CPU regression checks: python -m unittest discover -s tests -v."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from infinisplat.panorama import create_template, load_template, slice_panorama
from infinisplat.pipeline import prepare_images, Pipeline
from infinisplat.model import InfiniSplat
import torch


class PanoramaTests(unittest.TestCase):
    def test_calibration_and_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_template(root/'template', 126)
            spec, cameras = load_template(root/'template')
            self.assertEqual(len(spec['views']), 24)
            np.testing.assert_array_equal(cameras['c2w'][:, :3, 3], 0)
            rotations = cameras['c2w'][:, :3, :3]
            np.testing.assert_allclose(rotations @ rotations.transpose(0,2,1),
                                       np.broadcast_to(np.eye(3),(24,3,3)), atol=1e-6)
            # Encode world ray directions as colors. This continuous periodic
            # panorama exposes flipped axes, wrong pitch, and wrap-seam errors.
            h,w = 512,1024
            v,u = np.mgrid[:h,:w]
            lon = ((u+.5)/w-.5)*2*np.pi
            lat = (.5-(v+.5)/h)*np.pi
            direction = np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),
                                  np.cos(lat)*np.cos(lon)],-1)
            Image.fromarray(np.rint((direction+1)*127.5).astype('uint8')).save(root/'input.png')
            slice_panorama(root/'input.png',root/'template',root/'cut')
            paths = [root/'cut/inputs'/v['file'] for v in spec['views']]
            for path,C,K in zip(paths,cameras['c2w'],cameras['intrinsics']):
                rgb=np.asarray(Image.open(path)).astype(float)
                v,u=np.mgrid[:126,:126]
                rays=np.stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)],-1)@C[:3,:3].T
                rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
                self.assertLess(np.max(np.abs(rgb-(rays+1)*127.5)),1.6)
            _, K, _ = prepare_images(paths, long_side=126,
                                     intrinsics=cameras['intrinsics'],max_views=24)
            np.testing.assert_array_equal(K,cameras['intrinsics'])
            self.assertEqual((root/'cut/cameras.npz').read_bytes(),(root/'template/cameras.npz').read_bytes())
            with self.assertRaises(FileExistsError):
                slice_panorama(root/'input.png',root/'template',root/'cut')

    def test_high_resolution_and_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            create_template(root/'template',round(768/14)*14)
            spec,_=load_template(root/'template')
            self.assertEqual(spec['image_wh'],[770,770])
            Image.new('RGB',(100,100)).save(root/'bad.png')
            with self.assertRaisesRegex(ValueError,'2:1'):
                slice_panorama(root/'bad.png',root/'template',root/'cut')
            self.assertFalse((root/'cut').exists())
            with (root/'template/cameras.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_template(root/'template')

    def test_shared_center_validation_and_no_state_leak(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls=[]
            def predict_cameras(self, images):
                raise AssertionError('Panorama must not predict a camera prior')
            def forward(self, images, poses, K, **kwargs):
                self.calls.append(kwargs)
                return dict(stats={},c2w=poses[0],intrinsics=K[0])
        model=FakeModel()
        pipeline=Pipeline(model,dict(backend='da3',weights_sha256='fixture',
                                    normalization='mean_predicted_point_radius'),'cpu')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new('RGB',(112,112)).save(root/'image.png')
            images=[root/'image.png']*2
            poses=np.repeat(np.eye(4,dtype=np.float32)[None],2,axis=0)
            K=np.repeat(np.array([[100,0,55.5],[0,100,55.5],[0,0,1]],dtype=np.float32)[None],2,axis=0)
            cameras=dict(c2w=poses,intrinsics=K)
            result=pipeline.reconstruct(images,cameras=cameras,long_side=112,
                                        panorama_shared_center=True)
            self.assertFalse(result['metadata']['panorama']['baseline_scale_fit_attempted'])
            pipeline.reconstruct(images,cameras=cameras,long_side=112)
            self.assertEqual(model.calls,[{'panorama_shared_center':True},{}])
            poses[1,0,3]=1
            with self.assertRaisesRegex(ValueError,'shared camera center'):
                pipeline.reconstruct(images,cameras=cameras,panorama_shared_center=True)
            with self.assertRaisesRegex(ValueError,'provided cameras'):
                pipeline.reconstruct(images,panorama_shared_center=True)
            pixels=torch.zeros(1,2,3,112,112)
            with self.assertRaisesRegex(ValueError,'shared camera center'):
                InfiniSplat.forward(None,pixels,torch.from_numpy(poses)[None],
                                   torch.from_numpy(K)[None],panorama_shared_center=True)
            with self.assertRaisesRegex(ValueError,'calibrated cameras'):
                InfiniSplat.forward(None,pixels,panorama_shared_center=True)


if __name__ == '__main__':
    unittest.main()
