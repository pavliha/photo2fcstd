# Everything known to be missing, wrong, or unverified

Every claim here carries the measurement behind it and `n=`. Where something is unmeasured, it says
so rather than estimating. Ordered by how much it costs the stated objective - a parametric FreeCAD
sketch a person can edit.

## 1. The tracer draws arcs as straight pieces

**The largest remaining defect, and it is entirely in code.**

Given a *perfect* drawing of the true face - the ideal sketch rasterised from its own `xy` samples,
square on, no walls, no tilt, no matting - the tracer recovers **1.76 curved primitives against a
real 5.19**, barely more than the 1.62 it gets from a photograph (n=180 matched parts,
`tools/tracer_ceiling.py`). It produces roughly the right element *count* (12.82 against 11.98)
with the wrong *types*.

Straight elements are already near-correct (6.22 against 6.78), so `curves` (0.554) and `elements`
(0.626) are one defect, not two: about three and a half missing arcs per sketch.

Closed off already: loosening the arc gates fires on 54% of parts whose real sketch has **no** curve
at all and 57% of parts that do - not selective, and structure falls -0.036 [-0.070, -0.002].
Per-point curve classification (`curvenet`) lost 0.579 to 0.452. Direct primitive prediction
(`sketchnet`) lost twice.

**What would close it**: unknown. Characterise *which* arcs get split before proposing anything.
The prior that beat twelve learned attempts still applies - "there is headroom" is not a reason a
fourth attempt differs.

## 2. Depth is a point estimate inside a 10x band

`depth_model` predicts log(depth/length) at 0.435 median absolute error against 1.045 for the best
constant, 63% within 2x (n=held-out slice). But **the median 80% band spans about 10x**, and only a
couple of percent of parts get a band tighter than 2x.

So even where the drawing is right, the *solid* is not dimensionally trustworthy. "A parametric
sketch you can edit" is largely delivered; "a parametric model you can build from" is not.

It is also PrintCAD-specific: on T-LESS it scores 0.348 against 0.314 for the best constant there,
having beaten a constant two to one here. The ratios differ outright (0.055-0.503 here,
0.377-1.561 there).

**What would close it**: a measurement. Depth is not in an uncalibrated photograph seen face-on.

## 3. Capture: tilt costs 0.212 and cannot be corrected

The whole capture term is viewpoint tilt - eight degrees off-axis costs 0.11 of sketch IoU. Shooting
square is worth about +0.13 with no code at all.

**It cannot be repaired after the fact.** At 15 degrees the projective distortion a homography
removes costs **-0.009 [-0.026, +0.008]**; the side walls coming into view cost **-0.084
[-0.114, -0.058]**; rectifying by the *true* normal scores -0.013 (n=115, `tools/tilt_walls.py`).

**What would close it**: shooting square. Nothing else.

## 4. The reshoot check has no artefact

`tilt_model.square_check` reports how far off square a photograph was taken with no ChArUco board,
and `preflight` calls it. It currently abstains on every PrintCAD photograph.

Tilt magnitude error over the 0-30 degree regime the check lives in:

| trained on | error | constant | within 5 deg |
|---|---|---|---|
| T-LESS photographs | **2.63 deg** | 6.98 | **87%** |
| PrintCAD flat Lambertian renders | 7.62 | 8.81 | 45% |
| PrintCAD Blinn-Phong, specular, antialiased | 7.96 | 8.81 | 43% |

The T-LESS head is good but its gate refuses 100% of PrintCAD images. Renders cannot substitute -
two renderers tried, the better one slightly worse.

**What would close it**: thirty to fifty real photographs with the board in frame, whose solved
poses are exact tilt labels. See `docs/tilt-labels.md`. One afternoon, once.

## 5. Carving is the strongest path and cannot be run on real photographs

Carving reaches 0.736 volumetric IoU against truth (0.791 on parts at least four voxels thick,
n=30) and an expected ~0.715 sketch IoU against the photo path's 0.602 - roughly double the solid
accuracy, with depth *measured* rather than guessed.

It needs the ChArUco board for pose, so it has only ever been validated synthetically
(`carve_check.py`), never on a real capture.

**What would close it**: photographs with the board.

## 6. Nobody has ever been asked which drawing they would rather edit

`sketch_score.structure_score` weights `loops`, `curves` and `elements` **equally**, and that is
arbitrary. Region IoU correlates only 0.22 with structural agreement and 0.33 with exact
primitives, which is why the structural criterion exists - but the criterion itself has never been
checked against a person.

Every quality number in this repository is a proxy chosen by its authors.

**What would close it**: show someone twenty pairs and ask.

## 7. Four builds in sixty are wrong, in two different ways

Building 59 trusted parts end to end (`tools/goal_check.py`): 59 of 59 documents built, 55 (93%)
with a real solid, 59 of 61 sketches solve, **0 sketches with free degrees of freedom**, 38.8
constraints per 12.7 geometry (so genuinely constrained, not fixed points), 1 redundant constraint.

The four failures are not one failure mode:

- **00011, 00039** produced nothing; the sketch solver returned -2 and -5
- **00061, 00086** produced a solid *with* volume that `isValid()` rejects

The second pair is the more concerning kind and has not been investigated.

## 8. Ground truth covers about 55% of the dataset

`sketch_score.trustworthy()` admits 1,047 of ~1,900 parts: the reference face must be the true
extrusion base (`prism`), not a sliver, and its area times depth must equal the solid's volume.
The rest have no reliable reference and their scores mean nothing.

Every number in this repository is on the trusted subset. The pipeline's behaviour on the other 45%
is unmeasured, not good or bad.

## 9. Nothing learned here survives a change of dataset

Three for three, in the same direction:

- `depth_model`: 0.435 on PrintCAD, 1.117 on T-LESS through the pixel path - worse than a T-LESS
  constant's 0.314, with its conformal band covering 24% where it claims 80%
- `axis_model`: 89% on PrintCAD, 29% on T-LESS against 33% for chance, and its confidence runs
  *backwards* out of distribution
- `tilt_model`: 4.8 deg on T-LESS, 34 deg on PrintCAD

Each is now gated (`embedding_is_familiar`, `section_constancy`, `is_familiar`) so it abstains
rather than answering confidently. The gates work. The underlying fragility is unfixed, and no
learned component here should be assumed to generalise to a part it was not fitted on.

## 10. Two independently-registered silhouettes cannot be intersected

Measured twice: a three-view visual hull from photo masks scores 0.332 against 0.433 for the plain
pipeline even choosing the best of 48 poses, while the same code on *truth* silhouettes reaches
0.823 - so the loss is registration. Replacing the front staircase with a traced outline and
intersecting scores 0.218 against 0.280 for the outline alone.

Multi-view only pays once views share a pose, which is what carving gets from the board.

## 11. Smaller and unresolved

- **`bsplinecurve`**: 328 of 2,163 ground-truth elements are b-splines, which the pipeline has no
  primitive for and the scorer counts as curves. Forcing them into "arc" cost 8 points of accuracy
  when tried.
- **Scale**: mm/px needs the board or a known length. Without either the sketch is
  dimensionless.
- **`stations` mode** is never selected on 197 trusted parts and is reachable only when forced.
  `tests/test_regression.py` enforces both. It is not dead code, but it is unexercised in practice.
- **CLAUDE.md contradicts itself** on coverage: one section says `stations` leaves 46% of parts
  drawing nothing, another says coverage is 100% and it is never chosen. The second is current; the
  first predates the mode allowed-list fix.
- **One redundant constraint** across 61 sketches. Harmless, uninvestigated.
- **Learned mode selection** is worth ~0.001 sketch IoU, and a *perfect* mode oracle is worth
  +0.0003 region IoU [-0.022, +0.023]. There is nothing left there.
