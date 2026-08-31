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

It fixes 93 parts and breaks 13, closing 78% of the gap to the oracle. On a cube, where all
three answers are equally right, its three scores land within 0.01 of each other, so it is
not merely confident everywhere.

**Most of the headline gain is on parts that trace as a single circle, and region IoU is
scale-invariant, so any circle matches any circle.** Splitting the same 400 parts:

| stratum | n | thinnest extent | learned |
|---|---|---|---|
| everything | 400 | 0.664 | 0.758 |
| traced as a single circle | 158 | 0.582 | 0.773 |
| a real outline, face at least 12 px | 235 | 0.730 | 0.750 |

On real outlines the gain is +0.02, not +0.09. The rods the extent rule fails hardest on are
exactly the parts whose face is a small circle: 22 parts have a face under 12 px across and
go from 0.133 to 0.830, which is a real fix to a real error but is scored generously. The
axis is genuinely chosen better - 89% against 69% agreement is independent of the metric -
but do not read the 0.09 as 0.09 of drawing quality.

`docs/figures/axis_sketches.png` shows five non-circular parts, and is worth reading for what
the metric misses as much as for the fix: 01409 recovers a visibly correct I-beam and scores
0.49, while 00950 scores 0.81 for a rounded blob that is plainly not the chevron it should be.

**Carving now beats a single photo for sketching.** It used to tie it. Both paths measured
on the same 397 held-out parts, so this is a like-for-like comparison and not two numbers
from two different sets:

| | region IoU |
|---|---|
| photo path | 0.600 |
| carve, thinnest-extent axis | 0.668 |
| **carve, learned axis** | **0.757** |
| carve, oracle axis | 0.785 |

Carving wins on 70% of parts, and unlike the axis gain above this survives stratification -
on the 241 parts that trace as a real outline rather than a single circle it is 0.590 against
0.745, with carving winning 74% of them.

The axis choice, not the carving, was what held this path back.

**But this is not a fair fight, and the gap is smaller than it looks.** The photo path gets
three real photos through RMBG. The carve path gets sixteen analytically exact poses and
silhouettes rasterised straight from the truth mesh - no capture, no matting, no pose error.
`carve_check` was built to test the geometry without capture, which is the right tool for
that job, but it is the wrong number to put beside a real-photo pipeline.

Carving with the silhouettes right and only the poses wrong, over 150 real-outline parts:

| rotation error | IoU | cost |
|---|---|---|
| exact | 0.738 | - |
| 0.5 degrees | 0.723 | -0.015 |
| 1 degree | 0.704 | -0.033 |
| 2 degrees | 0.686 | -0.051 |
| 5 degrees | 0.574 | -0.164 |

The photo path scores 0.590 on this stratum, so **carving stays ahead as long as pose is good
to about 2 degrees and loses below 5.** A ChArUco board resolves pose far better than that, so
the advantage is real - but the honest expectation on real photos is around 0.70, not 0.74,
and the remaining matting cost of roughly 0.02 takes it lower still.

Degradation is graceful, which is the difference between this and the two-view intersections
that failed: those registered each photo independently, which is many degrees of error, and
intersection deletes correct material that nothing restores.

## Predicting corners loses too, and now we know what the fitter actually wants

The note left after the per-point classifier said the next attempt should regress corner
positions rather than label points, because `approxPolyDP` localises a corner to a geometric
extremum while a classifier gives a boundary fuzzy by several points. So: `cornernet.py`
predicts a heatmap over the contour, peaked at the STEP sketch's own edge endpoints, decoded
by peak-picking with a weighted centroid - a position between two points, which is what the
fitter consumes. Same 1D dilated CNN, 195k parameters, seven minutes on a laptop.

On the proxy it wins by a mile, measured against what `approxPolyDP` gives for free:

| | precision | recall | F1 | median offset |
|---|---|---|---|---|
| approxPolyDP | 0.480 | 0.742 | 0.583 | **0.0034** of the diagonal |
| learned heatmap | **0.866** | **0.820** | **0.842** | 0.0055 |

End to end on 163 discriminating parts it loses, and the interval excludes zero:

| arm | sketch IoU | skill | exact primitives |
|---|---|---|---|
| **approxPolyDP** (shipped) | **0.548** | **0.212** | **20%** |
| learned corners | 0.505 | 0.138 | 13% |
| approxPolyDP positions, learned filter | 0.505 | 0.138 | 20% |

Learned corners: -0.043 [-0.061, -0.025]. Filtered: -0.043 [-0.063, -0.024].

**The proxy table says why.** `approxPolyDP` localises a corner it finds almost twice as
tightly, and only fires too often. Over-firing looked free, because `merge_and_snap` recombines
collinear runs, so the obvious synthesis was to keep `approxPolyDP`'s precise positions and let
the model delete its false positives. That arm changed only 57 of 196 parts - and lost on 40 of
the 57 it touched.

Which settles what the label was wrong about. A corner in the STEP file is where one primitive
hands over to the next. A corner the fitter wants is **wherever a run becomes fittable by a
single primitive**, and those are not the same set: splitting a long spline into two arcs needs
a split point that is not a vertex of anything. `approxPolyDP`'s extra corners are not false
positives, they are the fitter doing its job, and a model trained to call them wrong makes the
drawing worse while scoring 0.84 on its own terms.

That is rule 8 - pose the problem so the labels are real - and it is the third time in this
project that a contour-labelling proxy has failed to transfer. Two different output
representations have now lost end to end from strong proxy scores, for the same underlying
reason: **the supervision is available from the STEP file, and the STEP file does not know what
the fitter needs.** A fourth variant on this theme is not worth running. If a learned component
is going to help here it has to be trained against the fitter's own objective, or replace the
fitter entirely rather than feed it.

`P2F_LEARNED_CORNERS=1` and `P2F_FILTERED_CORNERS=1` enable the losing paths; both are off.

## The capture path runs, measured without a camera

`carve.from_photos` detects the ChArUco target, solves each pose and carves. None of it had
ever been executed end to end, because measuring it needs photos with the board in them and
neither dataset has any. `capture_check.py` renders them instead - the board is planar, so a
homography maps it into any view exactly - which exercises detection, pose and the carve
together and leaves only optics and matting untested.

Two things it found:

**The board must be laid out by a rotation, not a reflection.** Its printed frame runs x right
and y down, which is left-handed in 2D, so placing it along world +x and +y and shooting from
above renders every view mirrored. The detector then finds three phantom markers out of
thirty-five and no pose at all. Turning the board over about its x axis fixes it and keeps the
part above the board where `carve` expects it. This is the same class of bug as the
left-handed camera basis already recorded here.

**Pose is not the limit; the board's detection floor is.** With that fixed:

| | |
|---|---|
| views solved | 16 of 16 |
| rotation error | 0.016 degrees median, 0.052 worst |
| position error | 0.064 mm median, 0.50 worst |

That is two orders of magnitude inside the 2 degrees the pose sweep above says we can afford,
so registration is a solved problem once the board is in frame. But the board stops being
detectable below about **15 degrees** of elevation, and a silhouette taken that low bounds a
part's height only to within `width / 2 * tan(elevation)`. For a 16 mm wide part that is 2.1 mm,
and the measured overshoot is 2.6 mm on a 7 mm tall box, while the two in-plane extents come
back to within 1.6 mm.

So silhouette carving from a board **systematically overestimates height**, by an amount set by
the part's width and the lowest angle the target survives. Shoot the lowest views the detector
will still accept - dropping the elevations from 30-68 degrees to 15-55 took the height error
from 5.4 mm to 2.6 mm.

**A correction for it was written, measured on real parts, and reverted.** Trimming the hull by
`w/2 * tan(e)` fixed seven synthetic shapes beautifully - mean absolute error 1.49 mm to 0.46 mm,
bias +1.49 mm to +0.11 mm. On 49 real PrintCAD parts carved through the same rendered board
photos it made things worse:

| | raw hull | with the correction |
|---|---|---|
| mean bias | +0.20 mm | -0.61 mm |
| mean absolute error | 0.30 mm | 0.65 mm |
| median absolute error | **0.11 mm** | 0.55 mm |
| within 1 mm | 44 of 49 | 39 of 49 |

It helped 11 parts and hurt 35, and splitting on how flat the hull's top is did not rescue it.

**The premise was wrong, not the arithmetic.** A 26x16x7 block centred on the board is the worst
case for this artifact: wide, flat topped, and short, so a 15 degree view really cannot see past
it. Real parts are none of those things, and sixteen views from 15 to 55 degrees already bound
their height to 0.11 mm median. There was no bias left to remove.

Seven hand-picked shapes are not a sample. This is the failure mode this repo already documents
for thresholds - a change that improves a statistic on a small set and breaks the real one - and
it survived a full test suite and a figure before the real measurement caught it.

This is also the exact quantity free-space depth carving removes, which is why depth was worth
0.063 on T-LESS: with a depth camera the correction is unnecessary, and without one it recovers
most of what depth would have given for the height.

## The metric has a baseline, and nobody had computed it

Region IoU is scale-invariant and dihedral-aligned, so **any circle matches any circle at
0.99**. A pipeline that draws one circle and never looks at the photo is therefore not a
strawman - it is a real answer with a real score. Over the 397 parts both paths were measured
on, that answer scores **0.555**, against the photo path's **0.600**.

The whole photo pipeline is worth **+0.045** over ignoring the photo.

`sketch_score.trivial_score` computes it, `difficulty` is one minus it, and `score_one` now
carries both on every part, so no future run can quote a mean without its baseline beside it.

Splitting on whether a part can discriminate at all (`difficulty >= 0.15`):

| subset | n | photo | carve, thinnest | carve, learned |
|---|---|---|---|---|
| everything | 397 | 0.600 | 0.668 | 0.757 |
| cannot discriminate | 76 | 0.667 | 0.695 | 0.888 |
| can discriminate | 321 | 0.584 | 0.662 | 0.726 |
| genuinely complex (>= 0.35) | 244 | 0.558 | 0.633 | 0.696 |

The parts that cannot discriminate score *highest* everywhere, which is the tell: they were
inflating every mean this project has ever quoted. On the 321 that can, the trivial answer
gets 0.455, so as a fraction of the margin actually available:

| | IoU | skill |
|---|---|---|
| photo | 0.584 | 0.236 |
| carve, thinnest axis | 0.662 | 0.380 |
| **carve, learned axis** | **0.726** | **0.498** |
| carve, oracle axis | 0.756 | 0.552 |

Aggregate skill is pooled, not a mean of per-part ratios - the ratio blows up wherever the
trivial answer is already near 1, which is exactly the subset being excluded.

None of the session's conclusions reverse under this metric: the ordering photo < thinnest <
learned < oracle holds on every subset. What changes is the size of everything. The photo path
in particular is much weaker than 0.600 suggested.

