# Measured state

Every number here is from this repository's own benchmarks, on the PrintCAD photo sets.
Quote the sample size with the number; several of these changed when the sample grew.

## The ground truth is only partly usable

`data/printcad_ideal_sketches_all.json` holds each part's real sketch, extracted from
its STEP file by `ideal_sketches.py`.

| | count |
|---|---|
| parts with photos | 1,907 |
| with 2D sketch geometry | 1,892 |
| genuinely a prism (face is the extrusion base) | 1,149 (61%) |
| passing `sketch_score.trustworthy` | 1,047 (55%) |

The prism count was **982 (52%)** until the extractor was fixed: it picked each part's
*largest planar face*, which for 167 parts is not the face the part is extruded from.
Those sketches were being scored against the wrong reference entirely.

## Where the pipeline stands (master)

On trusted parts, measured with `sketch_score.py`:

| | value |
|---|---|
| parts that emit a sketch at all | **100%** |
| region IoU over all trusted parts | **0.569** |
| region IoU over the parts that do draw | 0.569 |
| loop count exactly right | 81% |
| curve fraction, ours against ideal | 0.43 / 0.51 |
| solid IoU | ~0.44 |

Every part now draws; see `docs/decisions.md` for the trade that bought it.

### Two things the region metric cannot see

**Scrambled reference loops, now fixed.** A STEP wire lists its edges in topological
order and each may run either way, so concatenating them as stored can scramble the
ring. 698 of 1892 parts (37%) were affected, 315 of them trustworthy, and shapely filled
the scrambled ring into a blob - part 00294 is a thin wire staple whose reference read as
31% filled. `sketch_score.chain_edges` now walks the edges end to end. Correcting it
moved the headline from 0.553 to **0.569**, so the bias was mild, but it was real.

**Wire-thin parts.** Two thin outlines offset by one wire-width overlap almost nowhere,
so region IoU is dominated by alignment and cannot reward a correct-looking sketch.
00294 scores 0.055 with a faithful six-line chevron. A curve-distance measure would suit
those parts; region IoU does not.

## The bottom of the distribution is a shooting fault

Re-ranking against the corrected reference changed which parts are worst. Thirteen
parts gained more than 0.1 and only two lost - 00256 went 0.177 to 0.754, 00171 0.544 to
0.966 - so several apparent failures were correct drawings judged against a blob.

What is left at the bottom is one failure mode. All ten worst parts have the **right
loop count**; what they share is aspect. We trace a sliver (0.04 to 0.22) where the
reference is nearly square (0.43 to 1.00), because the photographs look at the part
**edge-on** and the silhouette is its 1 mm thickness rather than its face.

| | count | mean IoU |
|---|---|---|
| traced outline is a sliver (aspect < 0.15) | 9 of 196 (5%) | **0.104** |
| everything else | 187 | **0.592** |

**No fallback is possible: 0 of those 9 parts has a better view** - all three photos of
each are edge-on, so the broad face is not in the data at all. `spec.assemble` now warns,
in the log and as an `edge_on_warning` row in the sheet, that the photos are edge-on and
the part should be laid flat and reshot. Compare 00198, which is thinner still in truth
(0.008) but photographed flat and scores 0.996.

## The error budget

Against the orthographic silhouette, on trusted parts whose photo shows the extrusion
face (n=94):

| stage | IoU | cost | fixable by |
|---|---|---|---|
| perfect | 1.000 | | |
| photo mask, before any tracing | 0.788 | **0.212** | capture |
| our tracing and regularisation | 0.675 | **0.112** | code |
| against the base face rather than the silhouette | 0.586 | **0.089** | multi-view |

### The capture term is viewpoint, not perspective

`tools/split_capture.py` renders each truth mesh orthographically and in perspective
from the same viewpoint and compares both against the real photo mask:

| camera tilt off the face normal | orthographic | with perspective | real photo |
|---|---|---|---|
| 0 degrees | **0.946** | 0.930 | |
| 8 degrees | 0.834 | 0.835 | |
| 15 degrees | 0.799 | 0.795 | |
| 30 degrees | 0.801 | 0.798 | |
| measured photos | | | **0.814** |

- **Perspective costs 0.003 to 0.016.** A phone at arm's length from a 20 mm part is
  already nearly orthographic.
- **Matting costs about 0.02.** Photos score 0.814 where matting-free synthetic views at
  the same apparent tilt score 0.834. RMBG is not the problem.
- **Viewpoint tilt is the whole term.** Eight degrees off-axis costs 0.11.

**Shooting square to the face is worth roughly +0.13 and requires no code.** It is the
cheapest improvement available to this project.

## Carving is roughly twice as accurate

`carve.py` needs the ChArUco board, so no PrintCAD photo can reach it.
`carve_check.py` drives it with synthetic views of a truth mesh instead - exact poses,
rendered silhouettes - which tests the geometry without any capture.

16 views at 0.25 mm voxels, 40 trusted parts:

| | volumetric IoU | extent error |
|---|---|---|
| all parts | 0.736 mean, **0.810 median** | +0.71 / +0.35 / +0.85 mm |
| at least 4 voxels thick (n=30) | **0.791** | +4.3% / +4.8% / +30.5% |
| thinner than 4 voxels (n=10) | median true thickness 0.43 mm | not resolvable |

For comparison the photo pipeline reaches about 0.43 solid IoU and its best possible
mode choice 0.527. Carving roughly doubles that **and measures the depth** instead of
predicting it.

Size the voxel to about a quarter of the smallest feature. The default `VOXEL_MM = 0.4`
cannot resolve anything under roughly 1.6 mm.

## Depth

Depth cannot be a constant: the true depth/length ratio spans 0.055 to 0.503. Measured
on a test slice that neither training nor calibration saw:

| | median \|log\| error | within 2x |
|---|---|---|
| best constant | 1.045 | 35% |
| geometric estimate (0.46 x edge-on aspect) | 1.118 | 28% |
| **learned regressor** | **0.435** | **63%** |

End to end on 41 held-out outline parts, solid IoU rose **0.404 to 0.483**.

The interval is conformally calibrated, so its coverage is guaranteed: **83% of true
depths fall inside the 80% band** (plain quantile regression covered only 55%). But the
median band spans about **10x**, and only a couple of percent of parts get a band
tighter than 2x. Depth is not in an uncalibrated photo of a part seen face-on. The
`params` note now states the range and whether the number is worth building from.

## Figures

- `figures/results.png` - photo, sketch and solid for four parts, including a `stations`
  part that produces no drawing.
- `figures/current_results.png` - six parts spanning the quality range, with what we draw
  beside what the STEP file contains.
- `figures/cadrille_fixed.png` - the cadrille evaluation, see `docs/cadrille.md`.

## Carving, measured on real photographs

`carve_check.py` proves the geometry on synthetic views; T-LESS proves it on real ones.
That dataset ships what the ChArUco target was for - 30 objects, 1296 views each,
`cam_K` and `cam_R`/`cam_t` per view, ground-truth masks and CAD models - so
`tless.py` needs no capture session at all.

30 objects, 22 views each, 0.8 mm voxels, scored at 1.5 mm against the CAD:

| | mean IoU | median | above 0.8 |
|---|---|---|---|
| silhouette carving | 0.719 | 0.729 | 10 of 30 |
| **plus depth free space** | **0.782** | 0.773 | 12 of 30 |

For comparison the photo-to-sketch pipeline reaches about 0.43 solid IoU on PrintCAD and
0.527 with the best possible mode choice. Carving with known poses is most of the way to
double that, and the synthetic estimate (0.736 mean) was fair rather than flattering.

### Depth carves what a silhouette cannot

A visual hull fills every concavity, because a cavity has no silhouette. Accuracy tracks
convexity almost exactly: the worst objects are volume/convex-hull 0.19 to 0.42, the best
0.86 to 0.89. `fuse.carve_with_depth` removes any voxel that sits in front of a measured
surface, which is precisely the geometry the hull is blind to:

| | silhouette | plus depth |
|---|---|---|
| concave objects (convexity < 0.5, n=7) | 0.506 | **0.640** |
| convex objects (n=23) | 0.784 | **0.825** |

Two details make the difference between this working and not:

- **Only trust depth well inside the mask.** At a silhouette edge the depth pixel is the
  table, far behind the voxel, so real surface reads as free space and gets deleted. With
  that bug present no tolerance value helped at all - a convex object still fell from
  0.917 to 0.770 at a 15 mm tolerance.
- **Require two views to agree.** One noisy depth pixel must not delete a voxel. Going
  from one vote to two recovered the convex objects (0.854 to 0.914) while keeping nearly
  all the concave gain.

### Turning a carved volume into a sketch

`spec_from_carve` traces the plan view of the carved volume through the same
`outline`/`primitives` code the photo path uses. It projected along Z and took the
extrusion depth along Z as well, both hardcoded - and the part's own axis is spread
17/24/17 across X, Y and Z in these files, so two thirds of the time it traced an edge
view and measured the wrong thickness.

Choosing the axis matters more than anything else in that path, measured over 60 parts
with known poses and reference sketches:

| | region IoU |
|---|---|
| always along Z | 0.408 |
| fraction of the bounding box filled | 0.425 |
| largest projected area | 0.551 |
| **thinnest extent** (shipped) | **0.597** |
| best of the three (oracle) | **0.752** |
| the photo path, for comparison | 0.597 |

Four rules have been tried and they all plateau:

| rule | agrees with the oracle | IoU |
|---|---|---|
| fraction of the bounding box filled | 27% (worse than guessing) | 0.425 |
| largest projected area | 53% | 0.551 |
| **thinnest extent** (shipped) | 71% | **0.615** |
| most constant cross-section | 71% | 0.561 |

Note the first is worse than the 33% you get by guessing, so a plausible-sounding
heuristic can be worse than nothing.

The reason none of them wins outright: **for a plate the extrusion is short, for a rod it
is long.** Part 00001 is a rod whose thinnest axis scores 0.23 while its extrusion axis
scores 0.94. No single extent-based rule covers both, and measuring how constant the
cross-section is - which is the actual definition of a prism - agrees no more often.

This is a three-way choice over a carved volume with exact ground truth available from
the STEP files, so it is one of the few places in this project where a small learned
classifier is better posed than a rule.

**The classifier wins, and it is the largest single gain in the carve path.** Rather than
one three-way model, it scores each axis on its own and takes the best. That gives three
training rows per part instead of one, and makes the answer independent of the order the
axes happen to come in. Twelve features per axis: extents and their ratios, projected
area, how much of its box the projection fills, how constant the slice count is along the
axis, and the old thinnest-extent rule as a feature the model can override.

Trained on 400 parts, then measured on 400 different parts never seen in training:

| rule | agrees with the oracle | IoU |
|---|---|---|
| thinnest extent | 69% | 0.664 |
| **learned** (shipped) | **89%** | **0.758** |
| best of the three (oracle) | 100% | 0.785 |
| worst of the three | | 0.219 |

It fixes 93 parts and breaks 13, closing 78% of the gap to the oracle. `docs/figures/axis_model.png`
shows five of the fixed ones: the thinnest-extent rule traces an edge view at IoU 0.00 and
the classifier finds the face at 0.99.

Every one of those is a rod - the shape the extent rules cannot cover - and the scores are
not close: 0.95-0.99 for the right axis against 0.00-0.01 for the two edge views. On a cube,
where all three answers are equally right, the three scores land within 0.01 of each other,
so the model is not merely confident everywhere.

**Carving now beats a single photo for sketching.** It used to tie it. Both paths measured
on the same 397 held-out parts, so this is a like-for-like comparison and not two numbers
from two different sets:

| | region IoU |
|---|---|
| photo path | 0.600 |
| carve, thinnest-extent axis | 0.668 |
| **carve, learned axis** | **0.757** |
| carve, oracle axis | 0.785 |

Carving wins on 70% of parts. The axis choice, not the carving, was what held this path
back: it was worth more than the carving itself was over a single photo.
