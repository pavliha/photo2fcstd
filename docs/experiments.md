# What was tried

A ledger so nothing is re-run. Every entry was A/B'd on the same parts, regenerating
both arms, and judged on build reliability as well as accuracy.

## Kept

| change | evidence |
|---|---|
| **Ground truth: pick the true extrusion base**, not the largest planar face | usable prisms 982 to 1,149 of 1,892 |
| **Keep small holes** (`MIN_HOLE_FRAC` 0.002 to 0.001) | 00061 draws 27 loops instead of 9, against a reference of 29. Segmentation already found all 28 holes; the threshold sat in the gap between the large holes (0.028-0.043 of outline area) and the bolt holes (0.0015-0.0019) |
| **Drop quantisation-collapsed edges** | rounding to 0.5 mm can zero a short edge, which FreeCAD rejects with "Both points are equal". Latent only because half the parts never built an outline |
| **Zero-volume solids no longer report `valid`** | ~3% of builds returned a null shape the report called valid |
| **Derive arc direction from the run** against the centre actually stored | correctness fix, exactly neutral at the shipped gates |
| **Fall back when regularising collapses an outline** | 00238 recovers a real outline from another photo, 00332 fails loudly. IoU when drawn 0.5788 to 0.5874 |
| **Learned depth regressor** | see `results.md`. The one learned component that beat its A/B |

## Rejected

| change | result | why it failed |
|---|---|---|
| **Looser arc gates** (span 40 to 15 deg, sag 0.08 to 0.03) | curve fraction 0.45 to 0.60 against an ideal 0.59, but IoU 0.584 to 0.554 and **12 of 160 models broken** | an arc and the chords approximating it cover the same area, so accuracy does not move; more arcs means more tangency constraints and more solver failures |
| **Re-fitting the radius about a snapped centre** | broke 2 parts, fixed 0 | |
| **Learned curve segmentation** (1D CNN, 0.82 per-point accuracy) | sketch IoU 0.579 to 0.461, and 0.452 once a minimum run length removed the fragmentation | cleaning up the segmentation made it *worse*, so the boundaries are misplaced rather than noisy. `approxPolyDP` localises a corner to a geometric extremum; a per-point classifier gives a boundary fuzzy by several points, and a corner is a line endpoint |
| **Reprojection verification** (build several candidates, pick the one that best reprojects onto the photos) | 0.391 against 0.433 | the ground-truth mesh itself reprojects at 0.717 while candidates average 0.801 - truth loses to the best candidate on 171 of 200 parts. Wrong models over-fit the silhouette they were traced from |
| **Three-view visual hull** from photo masks | 0.332 against 0.433, best of 48 poses | with *truth* silhouettes the same code reaches 0.823, so the loss is registration |
| **Outline INTERSECT side elevation** (to get a drawing and keep the two-view solid) | 0.218 against 0.226 for two staircases and 0.280 for the outline alone | same cause as the hull |
| **Learned mode selection** | +0.015 solid IoU, and **0.001** of sketch IoU | routed 0.542 against always-outline 0.543, oracle 0.544 |
| **cadrille** (published multi-modal CAD reconstruction) | see `docs/cadrille.md` | |

## The rule that emerges

**Never intersect two independently-registered photo silhouettes.** Any misalignment
deletes correct material and nothing restores it. The modes survive precisely because
they never combine two separately-framed views. Multi-view only pays once the views
share a pose, which is what `carve.py` gets from the board.

**And: learning wins where information is absent, and loses where geometry already
measures it.** Depth is a scalar the camera genuinely cannot see, so a prior is the only
option and it paid. Curve boundaries and mode choice are both things the geometry
already resolves, and both learned attempts lost.

## Mistakes worth remembering

- The capture error was called "perspective" for most of this work. It is **viewpoint
  tilt**; perspective costs almost nothing. Measured only once someone rendered the
  control.
- A `+140%` thin-axis error in carving looked alarming through three wrong hypotheses
  (floor plane, camera elevation, camera handedness). The cause was that the thinnest
  part is 0.11 mm against a 0.5 mm voxel - a resolution artefact, and the *metric* was
  wrong. Relative error is meaningless below voxel size.
- An eval loop that stepped 64 while slicing 128 produced a predictions tensor twice too
  long. It crashed, which was lucky.
- The camera basis for OpenCV must be `[right, down, forward]` and right-handed
  (`det = +1`). A left-handed basis silently mirrors the projection.
