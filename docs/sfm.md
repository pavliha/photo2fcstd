# Poses without a printed target

Carving reaches **0.782 solid IoU on real photographs** against the photo pipeline's ~0.43. The only
thing it needs that an ordinary capture lacks is where each camera was, and until now that came from
a ChArUco board in every frame - which is the requirement that stalled this path.

Structure-from-motion takes the poses from the scene instead. `tools/sfm_check.py` is the first
gate: renders with a textured ground plane and exactly known poses, so the only question is whether
COLMAP plus carving reproduces carving with true poses. Synthetic on purpose - if it fails here it
fails everywhere.

## It works, and the pose is as good as the board's

24 views, one part, full-contrast ground texture:

| | |
|---|---|
| images registered | **24 of 24** |
| camera centre error | 0.01 mm of a 38 mm radius (**0.02%**) |
| camera rotation error | median **0.031 deg**, worst 0.060 |
| carving agreement, SfM poses against true poses | **0.987** |

For comparison, a detected ChArUco board gives pose good to 0.016 deg where the carving budget
allows 2. SfM lands in the same order of magnitude, with an order of magnitude of headroom.

Across four parts at 16 views:

| part | registered | rotation error | carve agreement |
|---|---|---|---|
| 00001 | 16/16 | 0.045 deg | **0.997** |
| 00005 | 16/16 | 0.041 | **0.990** |
| 00006 | 16/16 | 0.031 | 0.949 |
| 00002 | 16/16 | 0.058 | - |

00002 carves to nothing **from true poses too**, so it says nothing about SfM: it is 0.25 mm thin,
0.4 of a voxel at 0.6 mm, and CLAUDE.md already records that the voxel cannot resolve anything under
roughly 1.6 mm.

## Two hard requirements, both found by trying to break it

**The background must be strongly textured.** At a quarter of the texture contrast, **0 of 16 images
registered, on every part**. Not degraded - failed outright. The part itself is textureless and
cannot register on its own, so the scene is doing all the work. A bare white desk will not do; a
newspaper, a cutting mat or a patterned cloth will.

**Sixteen views, not eight.** At 8 views only 5 registered. The orbit has to overlap.

## What is still untested

Everything about a real photograph: motion blur, exposure change between frames, rolling shutter,
and whether a real desk carries as much texture as generated noise. This gate measures the
*geometry*, and the geometry is not the risk it looked like.

Scale is unchanged by any of this. SfM is scale-free, so a metric solid still needs one caliper
reading - which the pipeline already asks for, and which the board would also have provided.

## The ask this replaces

Not "print a ChArUco target and shoot every frame with it in view", but: **put the part on something
patterned and take about sixteen photographs walking around it.** No printing, no target, no
alignment. That is the capture `carve.py` needs to reach 0.782 instead of 0.43.
