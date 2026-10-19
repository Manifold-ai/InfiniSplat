# Example data sources

The project code license does not replace these datasets' original terms.
The bundled samples contain RGB images and, for the calibrated examples, camera
matrices. They contain no depth maps, meshes, labels, or training metadata.

## Hypersim

The `hypersim_provided` and `hypersim_da3_pose` samples come from the public
[Hypersim dataset](https://github.com/apple-aiml-research/ml-hypersim), scenes
`ai_004_009` and `ai_006_001`, respectively, camera trajectory `cam_00`.
Hypersim data is licensed under
[Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The images are tone-mapped RGB derivatives, re-encoded without embedded metadata.
Their resolution and pixel values are preserved from the project-page input files.
The supplied camera arrays use OpenCV camera-to-world coordinates and pixel
intrinsics matching the image files. Frame identities are in
`examples/data/manifest.json` in the sample archive.

Please cite Mike Roberts et al., **Hypersim: A Photorealistic Synthetic Dataset
for Holistic Indoor Scene Understanding**, ICCV 2021.

## DTU

The `dtu_provided` sample comes from scan 24 of the
[DTU MVS dataset](https://roboimagedata.compute.dtu.dk/?page_id=36).
These are rectified, resized/cropped RGB images in the common MVSNet data format
(640 × 512), as used in the project-page case. Supplied pixel intrinsics match
these processed files. PNG metadata has been removed without changing pixels.
DTU data remains subject to the original dataset terms, independently of the
InfiniSplat project license. The dataset page describes the data as freely
available and requests citation of its accompanying paper.

Please cite Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and
Henrik Aanæs, **Large Scale Multi-view Stereopsis Evaluation**, CVPR 2014.

The bundled `panorama/small_empty_house.jpg` is a CC0 Poly Haven panorama; see
[its attribution](panorama/NOTICE.md).
