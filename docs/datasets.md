# Datasets that replace a capture session

Two items in `docs/gaps.md` were blocked on photographs. One of them was already unblocked and
nobody noticed; the other has a public dataset that fits. This records which dataset serves which
gap, and - equally - what no dataset can fix.

## Already done: carving on real photographs

`docs/gaps.md` said carving "has only ever run on synthetic views". That is **wrong**, and
`docs/results.md` has said so since before this session: T-LESS ships exactly what the ChArUco board
was for - 30 objects, 1,296 views each, `cam_K` and `cam_R`/`cam_t` per view, ground-truth masks and
CAD - so `bop.carve_object` needs no capture session.

Measured on 30 objects, 22 views each, 0.8 mm voxels, scored at 1.5 mm against CAD:

| | mean IoU | median | above 0.8 |
|---|---|---|---|
| silhouette carving | 0.719 | 0.729 | 10 of 30 |
| plus depth free space | **0.782** | 0.773 | 12 of 30 |

Against the photo pipeline's ~0.43 solid IoU, and the synthetic estimate of 0.736 proved fair rather
than flattering. **C3 is closed**, and it was closed by a dataset rather than a rig.

## For the sketch ground truth: Fusion 360 Gallery, Reconstruction subset

The sharpest limitation in this repository is that only about 55% of PrintCAD's ground truth is
usable - the reference face has to be *inferred* from a STEP file, and `sketch_score.trustworthy()`
throws out the rest. Every number here is quoted on 1,047 parts for that reason.

The [Fusion 360 Gallery Reconstruction dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset)
is **8,625 designs authored as sketch-and-extrude sequences**, 2.0 GB, with JSON that carries what
this project outputs:

- sketch curves as `SketchLine`, `SketchCircle`, `SketchArc`, and profile loops of `Line3D` /
  `Arc3D` / `Circle3D` with explicit coordinates
- `extent_one.distance.value` - the true extrusion depth, which is the depth model's training target
- geometry in `.smt`, `.step` and `.obj`

Nothing is inferred: the sketch is the one a person drew. That removes the `trustworthy` filter, the
`prism` check and the area-times-depth test in one move, and multiplies the usable ground truth by
about eight.

What it does **not** have is photographs. So it improves what a drawing is scored against, not what
the drawing is made from.

## For tilt labels on real photographs: BOP-Industrial

`tilt_model` reaches 2.63 deg on T-LESS photographs and its gate then refuses PrintCAD ones, so the
missing ingredient is real photographs of *machined parts* with known pose, from more than one
capture setup. The BOP-Industrial group is exactly that, all in BOP format which `bop.py` already
reads:

| dataset | objects | images | note |
|---|---|---|---|
| [IPD](https://github.com/intrinsic-ai/ipd) (Intrinsic) | 10 | RGB-D HDR, 13 cameras | multi-view, released into BOP |
| [XYZ-IBD](https://xyz-ibd.github.io/) | 15 | RGB-D, 3 sensors | 273k real annotated samples, plus 45k synthetic |
| ITODD (MVTec) | 28 | **grayscale**-D | industrial, but grayscale removes the shading the tilt head reads |

Two caveats before anyone downloads 273k samples. These are **bin-picking** scenes - cluttered and
occluded, where our pipeline expects one part on a plain ground - so instances must be filtered by
BOP's own visible-pixel count. And ITODD's grayscale is a poor fit specifically because the tilt
result rests on shading being worth 3.6 deg beyond the silhouette.

For a single part on a plain background, T-LESS `train_primesense` remains the closest match and is
already downloaded.

## What no dataset fixes

**E1 - nobody has been asked which drawing they would rather edit.** `structure_score` weights
loops, curves and elements equally because there is no answer to weight them by. That is a
judgement, not data, and no corpus supplies it.

**Whether the tilt head admits your own photographs.** More real datasets make the head better and
its gate broader, but the gate's verdict on a photograph taken with your camera, of your part, on
your bench is only knowable from such a photograph. The difference is that the training set no
longer has to come from you - only the check.
