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

## Viewpoint tilt has +0.09 in it and the silhouette does not know the tilt

The error budget blames 0.212 on capture and says all of it is viewpoint tilt. That was measured
against the orthographic silhouette, so the first question is what it is worth against the
sketch. Warping each photo's mask by a grid of 25 rectifying homographies and taking the best
per part, over 78 trusted parts:

| | all | discriminating |
|---|---|---|
| as shot | 0.528 | 0.539 |
| best rectification per part | 0.647 | 0.630 |
| **headroom** | **+0.119** | **+0.090** |

60 of 78 parts want some correction and the median chosen tilt is 8 degrees. The score is a
deterministic function of part and warp, so this is a real ceiling rather than a max over noise -
an earlier permutation test of mine was ill-posed, because shuffling parts mixes their difficulty.

**Nothing visible in a silhouette can find that warp.** Every image-only criterion, scored by
choosing the warp it likes best and then measuring the sketch:

| chooser | IoU | of the ceiling |
|---|---|---|
| as shot | 0.539 | - |
| most symmetric | 0.479 | -67% |
| most solid | 0.460 | -89% |
| most right-angled | 0.437 | -114% |
| most rectangular | 0.429 | -123% |
| most parallel edges | 0.404 | -151% |
| fewest elements | 0.354 | -206% |

All of them are worse than leaving the photo alone, and no single fixed correction beats as-shot
either. A random warp scores 0.447 against 0.528, so most warps hurt: this is not a search that
needs a better prior, it is a quantity that is not in the data. A tilted square and an untilted
trapezoid are the same picture.

**Choosing among the three photos is better posed, and still does not work.** The ceiling is the
same size and the choice is one of three rather than a continuum:

| | discriminating |
|---|---|
| what ships today | 0.554 |
| first photo | 0.574 |
| learned selector, held out by part | 0.577 |
| best of the three (oracle) | **0.642** |

The learned selector picks the best view 47% of the time against 33% for chance - and against 48%
for the pipeline as it already stands.

**The view model was checked for the failure that caught the axis model, and does not have it.**
Both were trained on PrintCAD alone, and PrintCAD shoots its good photo first: photo 1 scores 0.578
against 0.508 and 0.511 and is the best of three 43% of the time. A selector that learned that
convention rather than what a good view looks like would collapse on anyone else's photographs.

On 771 parts it never trained on, split by whether the convention holds:

| subset | n | agrees | model | first-photo rule | chance | oracle |
|---|---|---|---|---|---|---|
| all | 771 | 0.56 | 0.632 | 0.590 | - | 0.681 |
| photo 1 is the best view | 324 | 0.60 | 0.659 | 0.706 | - | 0.706 |
| **photo 1 is not the best** | 447 | 0.53 | **0.612** | 0.506 | 0.550 | 0.663 |

It answers "photo 1" 38% of the time against a true rate of 42%, so it is not defaulting to the
convention, and on the 447 parts where the convention fails it beats chance by 0.062 and the
first-photo rule by 0.106, taking 55% of the headroom available there. It learned something about
what a usable view looks like rather than which slot it sits in.

That is not proof it survives a different camera and lighting - every photograph here is still
PrintCAD's - but it rules out the specific way the axis model broke.
 Its apparent edge over what ships is not view selection
getting better. PrintCAD's first photo is systematically the good one (0.578 against 0.508 and
0.511, best of three 43% of the time against 29% and 28%), which is a capture convention of that
dataset and not something a user's own photos would obey.

**Measuring the tilt where it is genuinely measurable is not available either.** A circular hole
images as an ellipse whose squash is the tilt outright, rather than a correlate of it. 12% of
parts show one usable hole in some photo; requiring one in all three, so views can be compared,
leaves **1 part in 250**.

So the tilt term is real, large, and unreachable from the mask. Three independent attempts say
the same thing, which is the point at which to stop trying variants. What is left is either pose
from the board - already validated at 0.017 degrees, and the carve path's skill is 0.498 against
the photo path's 0.212 - or the photo's pixels rather than its mask, which is untested and is the
one place shading, highlights and hole ellipticity survive. Segmentation throws all of it away.

**One thing to fix regardless: what ships is worse than trivial alternatives.** It scores 0.554
where taking the first photo scores 0.574, and it draws from the worst of the three views 27% of
the time. That is not a tilt problem, it is routing.

**Fixing the routing is worth +0.038, and it is the only accuracy gain this line of work produced.**
`view_model.py` scores each photo the way `axis_model` scores each axis and takes the best. On 300
parts none of which the model was trained on:

| arm | sketch IoU | skill | exact primitives |
|---|---|---|---|
| shipped rules | 0.565 | 0.148 | 23% |
| **learned view choice** | **0.602** | **0.221** | **26%** |

+0.038 [+0.019, +0.056] over n=250 discriminating parts, changing the view on 109 of 295 and
winning on 65 of those against 40 losses. Through FreeCAD both arms build 44 of 45 valid solids
and fail on the same part, so the gain costs no null solids and no loose sketches. It is on by
default; `P2F_VIEW_MODEL=0` restores the rules.

The first run of this A/B reported exactly +0.0000 on every part. `P2F_VIEW_MODEL` was serving as
both the feature flag and the model-path override, so setting it to 1 pointed the loader at a file
named "1", `load()` returned None and the model silently fell back to the rules it was meant to
replace. A flag that reads as enabled while doing nothing is the failure mode to watch for here -
the A/B looked clean, it just measured the control twice.

## Pixels do not know the tilt either, and the thresholds have not moved

Two hypotheses followed from the view-choice result, and both lost.

**Pixels instead of the mask.** Segmentation discards shading across a face, specular highlights
and the ellipticity of an oblique hole, which is where foreshortening physically lives, so a CNN
on the photo crop should beat statistics of its silhouette. `pixel_data.py` builds masked
greyscale crops plus the mask as a second channel, `pixel_train.py` scores each of the three views
and takes the best - the same per-candidate framing that worked for the carve axis and the view
choice. 1014 parts, split by part into train/val/test, epoch chosen on val.

On the 157 test parts **neither** model was trained on:

| chooser | agrees with the best view | sketch IoU |
|---|---|---|
| chance | 0.33 | 0.564 |
| first photo | 0.40 | 0.589 |
| **silhouette statistics** (shipped) | 0.56 | **0.631** |
| pixels | 0.57 | 0.622 |
| oracle | 1.00 | 0.669 |

Pixels minus silhouette is -0.009 [-0.030, +0.011]. A dead heat: they disagree on 74 of 157 parts
and split them 36 to 35. Whatever the mask loses, the model cannot use.

An ensemble looked promising because the two disagree so often, and it is the one place the
agreement metric and the objective come apart cleanly: averaging the two scores lifts agreement
from 0.56 to **0.61** and moves sketch IoU by **+0.0015 [-0.0082, +0.0108]**. Picking the right
photo more often stops paying once the wrong photos being avoided are the ones that were nearly as
good anyway. Reporting the agreement alone would have looked like a win.

Read against the first 243-part run, where pixels scored 0.49 agreement and looked beaten, this is
also a reminder that a model on 182 training parts says nothing: the same architecture on 659 went
to 0.57 and drew level. The first run's conclusion was noise in both directions.

**The thresholds after the view change.** Every empirical constant was jointly tuned when the
pipeline drew from an oblique photo 27% of the time; the shipped selector changed which image
reaches the tracer on 109 of 295 parts, so those constants were fitted to a distribution that no
longer exists. Six of them, three values each, on the same 160 parts with both arms regenerating
specs:

| | |
|---|---|
| shipped | 0.645 |
| best single change (`HOLE_FRAC_VISIBLE` 0.08) | 0.644 |
| every other change | -0.004 to -0.006 |

Not one is positive. The constants were at a local optimum under the old input distribution and
they are still at one under the new, which is a stronger statement than the original tuning made.

## The curve gap survives four attempts, and the photo path is saturated

Real sketches average 5.13 curve edges; we draw 1.49. Only 21% of those real curves are shorter
than 5% of the sketch diagonal and the median is 12.8%, so about 3.26 per sketch are large enough
to see. Four different attempts to close that gap have now lost end to end:

| attempt | curves | exact primitives | sketch IoU |
|---|---|---|---|
| **shipped** | 1.49 | **25%** | **0.648** |
| loosen the arc gates | 3.07 | 19% | 0.633 |
| loosen further | 3.63 | 18% | 0.631 |
| fit arcs across consecutive runs | 1.64 | 24% | 0.635 |
| learned per-point curve labels | - | - | 0.452 |

The run-merging attempt was the best-motivated of them: `approxPolyDP` places corners at geometric
extrema, which on a curve means along it, so an arc arrives as several short flat chords no gate
can accept. On a synthetic circle chopped into 8 runs it recovers 2 arcs instead of 8 lines. On
real traces it fired on 64 of 213 parts and cost 0.033 of IoU on exactly those, -0.0125
[-0.021, -0.005] overall. The mechanism is real and the fix is not.

**Taken together the photo path is saturated.** In one session, on the same metric with the same
protocol:

| change | result |
|---|---|
| learned view choice | **+0.038, shipped** |
| pixels instead of the mask | -0.009, tie |
| ensembling both | +0.0015, nothing |
| re-sweeping six thresholds | nothing positive |
| loosening arc gates | -0.013 |
| merging runs into arcs | -0.013 |

One win in six, and the win was a routing bug rather than a modelling gain. The remaining error is
not reachable by thresholds, fitting heuristics or selection models - the three kinds of change
this codebase can express. Against a trivial circle at 0.455 the photo path scores 0.602 for a
skill of 0.221, where carving from known poses scores 0.726 for 0.498. The next real gain is a
different input, not a better estimator on this one.

## Carving's advantage is not an artifact of perfect input

Every carve number here comes from silhouettes rasterised straight from the truth mesh, which is
the right tool for testing geometry and the wrong number to compare against a real-photo pipeline.
Pose error was swept earlier. This is the other half of the capture term, and it needed no new
photographs: displace each silhouette's boundary by a spatially correlated random field - signed
distance to the edge, plus smooth noise, re-thresholded - which wanders along an edge the way a
matting model does rather than eroding uniformly.

Over 110 real-outline parts, 16 views, exact poses:

| boundary displacement | sketch IoU | cost | exact primitives |
|---|---|---|---|
| none | 0.734 | - | 25% |
| 1 px | 0.723 | -0.011 | 27% |
| 2 px | 0.716 | -0.019 | 25% |
| 4 px | 0.705 | -0.030 | 22% |
| 8 px | 0.661 | -0.074 | 15% |

**Carving is barely sensitive to matting.** Eight pixels of boundary wander - far worse than RMBG
produces - still leaves it at 0.661 against the photo path's 0.602 on comparable parts. Sixteen
views average the error out: a voxel survives only where every silhouette agrees, and independent
boundary noise rarely agrees.

That completes the carve error budget, and both halves are small at realistic magnitudes:

| term | measured cost |
|---|---|
| pose, from a detected ChArUco board (0.016 degrees) | negligible; 1 degree would cost 0.033 |
| matting, at a realistic 2 px | 0.019 |
| **expected on real photographs** | **about 0.715** |

Against the photo path's 0.602 that is **+0.11**, and it is now an estimate built from two measured
degradation curves rather than an extrapolation from clean synthetic input. The board rig is worth
building.

## Carving to a sketch from real photographs, and what it costs the axis model

T-LESS has real photographs with dataset poses, real segmentation and a CAD mesh, which is
everything the PrintCAD sets lack. It has no STEP sketch, so the reference is the mesh's own
cross-section, taken on the axis whose section area varies least - T-LESS parts are not prisms
like PrintCAD's and their variation runs 0.09 to 0.78 depending on direction, so the axis has to
be chosen rather than assumed. 21 of 30 objects have such an axis, median variation 0.08.

| | sketch IoU |
|---|---|
| a circle, ignoring the photographs | 0.493 |
| **carved from real photographs** | **0.605** |
| carved from renders of the mesh at the same poses | 0.576 |

Skill over the trivial answer is 0.221 on n=21. Real photographs appeared to beat renders by
0.029, which is the wrong sign for a capture penalty, and the render arm turned out to be simply
invalid: `carve_check.silhouette` rasterises at its own 900x900 canvas while T-LESS masks are
400x400, so the two arms were never comparable. **Only the real-photograph number stands.** The
matting sweep already answers the question that arm was meant to: capture is not what limits
carving.

**The axis model does not survive the change of dataset.** It picks the reference axis on 6 of 21
T-LESS objects, 29% against 33% for chance, where it reaches 89% on PrintCAD. n=21 cannot separate
29% from chance, but it separates both from 89% comfortably.

**And it is confidently wrong.** On the objects it gets right its top score averages 0.56 with a
margin of 0.31 over the runner-up; on the ones it gets wrong, 0.73 and 0.62. Confidence runs
backwards, so the obvious mitigation - abstain when unsure and fall back to the thinnest-extent
rule - would abstain on precisely the cases the model handles correctly. Any gate here has to come
from geometry, such as whether any axis of the carved volume has a near-constant section, not from
the classifier's own score.

That is a real limitation of a component shipped today, and the cause is visible in the training
set: every labelled example came from PrintCAD, whose parts are extrusions of a face. T-LESS parts
are industrial housings and connectors whose section changes along every axis - the median best
axis still varies by 0.08 and the worst by 0.78 - so "the axis a single sketch describes" is a
different question there, and often has no good answer at all. The classifier learned PrintCAD's
notion of a base face, not a general one.

The sketch numbers above were taken with the axis forced to the reference, precisely so that this
failure did not contaminate the capture measurement. In production on parts like these the axis
would be wrong most of the time, and the honest expectation is well below 0.605.

**A geometric gate catches what the confidence could not.** `section_constancy` is the median
slice population over the largest, measured off the carved volume, so 1.0 means a part of constant
cross-section - which is exactly the condition under which a base face exists at all:

| gate | PrintCAD kept / correct | T-LESS kept / correct |
|---|---|---|
| none | 100% / 89% | 100% / 36% |
| **constancy >= 0.8** | **94% / 89%** | **64% / 56%** |

In distribution it costs nothing: the parts it drops were not being got right anyway, and accuracy
on the rest is unchanged. Out of distribution it refuses the worst third and lifts the remainder
from 36% to 56%. Within PrintCAD alone the measure is useless as a discriminator - 0.947 when right
against 0.944 when wrong, because every part there is a prism - which is precisely why it had to be
checked on a second dataset to be seen at all.

It ships as a warning on the spec rather than a silent fallback, because when a part has no
constant-section axis there is no better rule to fall back to. The honest output is a drawing plus
a note saying no single sketch describes this part.

## Every model here was trained on PrintCAD, and it shows

The axis classifier's collapse on T-LESS prompted an audit of the other two learned components
against the same question: did it learn the task, or the dataset?

| component | in distribution | on T-LESS | verdict |
|---|---|---|---|
| view choice | 0.56 agreement, +0.038 IoU | not testable (T-LESS has no view triples) | survives its convention test |
| axis choice | 89% correct | 29% against 33% chance | **learned the dataset** |
| depth ratio | 0.435 median \|log\|, constant 1.045 | 0.348, constant **0.314** | **loses to a constant** |

**The depth model is beaten by the best constant on T-LESS**, 0.348 against 0.314 median absolute
log error, and 81% within 2x against 90%. On PrintCAD it beat a constant by a factor of two.

The cause is label shift rather than anything subtle. PrintCAD's depth-to-length ratios run 0.055
to 0.503 - thin plates and brackets - while T-LESS's run 0.377 to 1.561, because those are chunky
industrial housings. The model predicts 0.125 to 1.185 and systematically under-predicts, having
learned the range it was shown. A constant is hard to beat on a narrow target, and easy to beat on
a wide one, which is exactly why the PrintCAD comparison flattered it.

Its conformal band still covered the truth 100% of the time against a claimed 80%, but at a median
width of **19.8x** that is over-coverage by being uninformative, not by being robust. Conformal
guarantees hold under exchangeability, which a different dataset breaks outright; the coverage here
is luck, not the theorem.

**The view model is the one that survived, and it was checked deliberately rather than assumed.**
The general lesson is that split-by-part, which this repo has been careful about since the start,
protects against memorising a part and does nothing about memorising a dataset. Every learned
component here has exactly one dataset behind it.

## Carving recovers holes; it cannot recover pockets

Rendering the T-LESS failures as drawings showed smooth profiles where the truth has slots and
steps, which looked like carving returning an outline and no internal structure at all. That
inference was drawn from four pictures and three measurable objects, and it is wrong.

On 214 discriminating PrintCAD parts, with the axis chosen by the model:

| | carve | photo path |
|---|---|---|
| loops recovered against real | **2.15 / 2.42** | 1.48 / 1.99 |
| parts whose sketch has holes | 45% | - |
| of those, recovering at least one | **91%** | - |
| holes recovered against real | **2.48 / 3.12** | - |

Carving recovers 89% of the loops a sketch contains against the photo path's 74%, and 79% of the
individual holes. **It is better at internal structure, not worse.**

The distinction the T-LESS figure was actually showing is between a through-hole and a pocket. A
visual hull is the intersection of what every silhouette permits, so a hole that breaks the
outline from some direction is carved out exactly, and a cavity that never breaks any silhouette
is invisible in principle. PrintCAD parts are plates and brackets whose holes go through. T-LESS
connector housings have internal slots that do not, and on the three such objects measurable here
none of that structure came back - internal area 5.6% of the outer loop in truth, 0.0% recovered.

So the limitation is real and narrow: **carving cannot see a cavity that no viewpoint reveals**,
which is a property of the geometry rather than a tuning failure, and depth carving is the only
thing that would address it. It is not the broad claim that carving returns outlines only, and
three objects were never enough to support that.

## Snapping to 45 degrees: the prior is real and too small to use

Manufactured faces are mostly axis-aligned, so extending the rectilinear pass to multiples of 45
looks obviously right. The reference sketches say how much is actually there, over 5549 straight
edges from trusted parts:

| edges within 2 degrees of a multiple of | share |
|---|---|
| 90 | **72.6%** |
| 45 | 75.3% |
| 15 | 81.6% |

The 90-degree prior is strong and the shipped pass already exploits it. Adding 45 buys **2.7
points**, and **16.7% of edges sit more than 10 degrees from any multiple of 45** - chamfers and
tapers at genuinely odd angles that a loose tolerance would straighten into right angles that were
never in the part.

End to end on 198 discriminating parts, snapping each near-canonical edge onto its target by
rotating it about its own midpoint:

| arm | sketch IoU | exact primitives |
|---|---|---|
| **shipped, horizontal and vertical only** | **0.646** | 25% |
| plus 45 degrees, 6 degree tolerance | 0.640 | 26% |
| plus 45 degrees, 8 degree tolerance | 0.642 | 26% |
| plus 45 degrees, 11 degree tolerance | 0.645 | 26% |

-0.0059 [-0.0122, -0.0003] at the tight tolerance and indistinguishable from zero at the loose
ones, with exact primitives up half a point, which is one part in 198.

The interesting part is the activity: the pass changed **142 of 198 parts** and split them 64 to 78.
It is not doing nothing, it is doing as much harm as good - straightening a genuinely oblique edge
costs about what tidying a nearly-square one gains. The 2.7 points of extra edges it can help are
simply outnumbered by the 16.7% it can hurt. Reverted.

## Three CAD priors, all real in the truth, all losing when applied after tracing

Manufactured parts are axis-aligned, mirror-symmetric, and drilled with a small set of tools.
All three are true of the reference sketches:

| prior | how strong in the real sketches |
|---|---|
| edges within 2 degrees of a multiple of 90 | 72.6% (already exploited) |
| adding multiples of 45 | +2.7 points |
| outlines mirror-symmetric to within 0.95 | **69%**, median exactly 1.000 |
| curved edges sharing a radius with another | **49%**, all of them on 27% of parts |

Symmetry and shared radii are far stronger priors than the 45-degree one. All three lose end to
end, on 198 discriminating parts:

| arm | sketch IoU | exact primitives | verdict |
|---|---|---|---|
| **shipped** | **0.647** | **29%** | - |
| snap to multiples of 45 | 0.645 | 26% | -0.001 to -0.006 |
| enforce mirror symmetry | 0.626 | 24% | **-0.021 [-0.031, -0.012]** |
| unify near-equal arc radii | 0.627 | 28% | **-0.019 [-0.029, -0.011]** |

**The prior is about the part; the correction is applied to the tracing.** Symmetry changed 106
parts and split them 37 to 69. When one side of an outline traces well and the other badly - which
is the normal case, since the two sides face the camera differently - averaging them drags the good
side down to meet the bad one. The constraint is true of the object and false of the evidence, and
applying it blindly after the fact spends a correct side to repair an incorrect one.

Radius unification fails the same way: two arcs that share a radius in the part are fitted from
different numbers of pixels at different foreshortening, so the better fit gets pulled toward the
worse. Exact primitives barely moves (-1.0 points) because the radii were never what made a sketch
wrong, and IoU drops because the geometry moved.

None of this says the priors are useless. It says a prior has to enter where the evidence is
weighed - during fitting, with each side weighted by how well it was seen - not as a post-process
that treats both sides as equally trustworthy. That is a different piece of machinery from a
threshold, and it is the honest form of "the model should understand symmetry".

## Predicting constraints instead of snapping them: the labels are not learnable

Three CAD priors applied uniformly all lost, so the natural next step is to decide each constraint
per edge and per pair and then solve, weighting each edge by how well it was seen. That is
Vitruvion's second stage and it is the right shape of answer: a rule cannot tell a well-traced side
from a badly-traced one, and a model can.

`constraints.py` extracts every constraint that holds in a sketch - canonical angle per edge,
equal length, equal radius, parallel, perpendicular per pair - and aligns a traced sketch to its
reference so the labels transfer. The alignment took three attempts:

| aligner | elements matched within 0.05 | median distance |
|---|---|---|
| midpoint to midpoint | 43% | 0.063 |
| position along the loop | 21% | 0.190 |
| **dihedral pose, distance to the segment** | **62%** | **0.029** |

Midpoint matching fails whenever the tracer splits one ideal edge in two, because the halves' midpoints
sit far from the whole edge's midpoint. Arc-length position discards the geometry and does worse still.

At 62% matched the labels were below the bar set before starting, so the next check was whether
they are learnable at all. Held out 5-fold, split by part, against the majority baseline:

| constraint | n | majority | model |
|---|---|---|---|
| angle class | 1610 | 0.471 | 0.442 |
| equal length | 44440 | 0.586 | **0.614** |
| parallel | 44440 | 0.862 | 0.843 |
| perpendicular | 44440 | 0.912 | **0.920** |
| equal radius | 44440 | 0.937 | 0.916 |

Four of five sit at or below the baseline and the best gains 0.028. **The build stops here.**

The reason is not only label noise. Asking whether the ideal edge is horizontal, given the traced
edge's own angle, is asking the model to correct the tracer exactly where the tracer is wrong - and
the features available are the tracer's own output. Where the traced angle is close to right the
answer is already known and the model adds nothing; where it is wrong, nothing in the traced
geometry says so. It is the tilt result again in a different costume: the quantity is not in the
data we kept.

A constraint predictor worth having would have to see the evidence the tracing threw away - which
contour points supported each edge and how confidently - rather than the fitted primitives. That is
a change to what the tracer records, not a model that can be bolted onto its output.

**So the tracer was changed to record it, and it did not help.** `support_of` now stores, for every
element, how many contour points backed it, the RMS residual of the fit against those points, the
run's span and chord, and its straightness; `carry_support` follows each element through
regularisation so the numbers survive to the spec. It is pure metadata - all 174 tests pass
unchanged, no geometry moves. With those features and their neighbour ratios added:

| constraint | majority | geometry only | with support |
|---|---|---|---|
| angle class | 0.471 | 0.442 | 0.452 |
| equal length | 0.586 | **0.614** | **0.617** |
| parallel | 0.862 | 0.843 | 0.852 |
| perpendicular | 0.912 | 0.920 | 0.922 |
| equal radius | 0.937 | 0.916 | 0.934 |

Every constraint improves, by 0.002 to 0.018, and not one crosses its baseline. The evidence was
the obvious missing ingredient and it is not enough.

**Two readings survive this and the measurement cannot separate them.** Either the signal genuinely
is not in silhouette-derived data - consistent with the tilt result, where six criteria and two
learned models all failed to recover a quantity the mask does not contain - or 62% label alignment
leaves too much noise to detect a real effect. Both fit. What would separate them is better labels,
and the alignment fails precisely where our tracing and the truth disagree about how many edges the
part has, which is not a labelling bug but the pipeline being wrong.

The support metadata stays, recorded and unused by the pipeline. It costs a few numpy operations per
element, it is the foundation any future attempt needs, and it is now visible in the spec for anyone
debugging why an edge came out where it did.

## Architecture review of the learned layer

Eight learned components now exist. What each is worth, and whether it runs:

| component | job | on by default | measured value |
|---|---|---|---|
| `view_model` | which photo to draw from | **yes** | +0.038 [+0.019, +0.056], n=250 held out |
| `axis_model` | which way to look at a carved volume | yes (carve path) | 69% to 89% in distribution, **29% out** |
| `depth_model` (pixel) | depth from a DINOv3 embedding | **yes** | **never audited out of distribution** |
| `depth_model` (tabular) | depth from silhouette statistics | fallback | 0.435 here, **loses to a constant on T-LESS** |
| `mode_model` | which mode to build | no | worth ~0.001 |
| `view_rank` | which photo to draw from | no (`P2F_VIEW_PICK`) | superseded, see below |
| `curvenet` | per-point curve labels | no | 0.579 to 0.452, lost |
| `cornernet` | corner heatmap | no | 0.842 F1, -0.043 end to end, lost |

Four structural problems, in the order I would fix them.

**1. Two view selectors, and one silently shadows the other.** `pick_view` supports a learned
`view_rank`, and `outline_source` wraps its result with `view_model`. Since `view_model` is on by
default and always returns an answer, `P2F_VIEW_PICK=ranker` has no effect on the outline modes -
the ranker computes a view that is then discarded. Two models trained for the same decision, one
unreachable. They should be one component, or the wrapping should defer when the ranker is enabled.

**2. The depth model that ships has never been audited out of distribution.** `predict` tries the
DINOv3 pixel path first and falls back to the tabular one. The T-LESS audit built its views from
masks with no source image, so `embed.for_views` returned None and the pixel path never ran: the
"loses to a constant" result is about the **fallback**, not about what runs on a real photograph.
Auditing the shipped path needs T-LESS views carrying their image paths.

**3. Fourteen swallowed exceptions across the ML layer, none of which log.** Every `load` and
`predict` returns None on any failure and the caller quietly uses the geometric rule. That is the
right behaviour and the wrong silence: it is exactly how `P2F_VIEW_MODEL=1` pointing the loader at a
file named "1" produced an A/B that measured the control twice and reported +0.0000 on every part.
A one-line warning the first time a component falls back would have caught it in seconds.

**4. Every component has exactly one dataset behind it.** Split-by-part is enforced everywhere and
protects against memorising a part. Nothing protects against memorising PrintCAD, and two of the
three audited components turned out to have done so. The `section_constancy` gate is the only
out-of-distribution guard in the system, and it was found by testing on a second dataset rather than
by any in-distribution measurement.

**What the layer is actually worth.** One component earns its place on measured evidence
(`view_model`, +0.038). One is valuable but only inside its distribution (`axis_model`). One is
unaudited where it matters (`depth_model` pixel path). Two are off and measured losers, and their
5 MB of checkpoints are kept deliberately as the data pipeline for a better-posed attempt. The
honest summary is that this is a geometric pipeline with one reliable learned component in it, not
a learned system.

## The depth path that ships is the one that fails out of distribution

The earlier T-LESS depth audit built its views from bare masks, so `embed.for_views` found no image,
returned None, and the tabular fallback was measured. Giving the views their T-LESS RGB frames makes
the DINOv3 path run, and it changes the answer twice over. n=21 objects, embeddings obtained on all
21:

| | median absolute log error | within 2x | band coverage (claims 80%) |
|---|---|---|---|
| best constant, fitted on T-LESS | 0.314 | **90%** | - |
| **shipped: DINOv3 pixel path** | **1.117** | 24% | **24%** |
| tabular fallback | **0.236** | 71% | 100% |

**The shipped path is three and a half times worse than a constant**, and its conformal interval
collapses from a claimed 80% coverage to 24% - the guarantee assumes exchangeability and a new
dataset voids it. The tabular model, which the earlier audit maligned, is the better of the two
here: its median error beats the constant outright and its band over-covers rather than under-covers,
which is the safe direction to be wrong in.

The failure mode is visible in the predictions. The pixel path outputs 0.108 to 0.480 where the truth
runs 0.377 to 1.561 - it has collapsed onto PrintCAD's range of thin plates and brackets even harder
than the tabular model, which at least spans 0.103 to 1.469.

`P2F_DEPTH_PIXELS=0` selects the tabular path and is the safer setting for anything that is not a
PrintCAD-like part. Changing the default is a trade, not a fix: the pixel path presumably earns its
place in distribution, which is why it was built, and that comparison has not been re-run here.

**One diagnostic did not work.** Embedding norm was recorded to separate "the backbone is out of its
depth" from "the head is", and it is 1.7 for all 21 objects because the embeddings are L2 normalised.
The norm carries no information and that question remains open.

## A mode router that drew nothing, and the headroom it invented

Rendering the pipeline end to end on part 00911 showed a green box whose three photographs include
one taken square on, with the notched corners plainly visible - and that photograph produced no
sketch at all. The mask was correct; nothing downstream drew it.

`learned_mode` restricts the choice to profile, plan and revolve, because this project decided
always-outline on measurement and `stations` produces no face to draw. `mode_pixels` honours that
list and the per-mode regression head honours it. **The classifier path ignored it**, and the saved
classifier has `stations` among its four classes, so it returned a mode the caller had explicitly
excluded.

| | mode classifier as it was | honouring the list |
|---|---|---|
| 00911, square-on photograph | **0.00** | **0.97** |
| single-view specs drawing nothing | 158 of 780 (20%) | 0 |
| parts with at least one view blanked | 75 (29%) | 0 |
| parts whose best view was blanked | 19 | 0 |
| **full three-view specs** | 0.638 | 0.637, **0 mode changes** |

On the pipeline as it actually runs the bug is invisible: with three views the classifier already
lands on an allowed mode, so the fix is worth -0.0005 [-0.010, +0.008] and changes not one part.
It only bites when a single view is scored.

**Which is exactly how the view-selection labels are made.** A fifth of them recorded a view as
worthless when the mode router had simply declined to draw it, and on 19 parts the genuinely best
view was the one marked worthless. Rebuilding the labels with the fix in place, over 1033 parts:

| | contaminated labels | clean |
|---|---|---|
| what ships | 0.554 | **0.616** |
| best of three (oracle) | 0.642 | 0.659 |
| **headroom from view choice** | **+0.088** | **+0.043** |

**Half the headroom quoted earlier was an artifact.** The shipped +0.038 stands - that A/B compared
full three-view specs on both arms, where this bug does not fire - but the ceiling it was measured
against does not.

Retraining the selector on clean labels changes nothing: over 640 parts none of which the shipped
model was trained on, -0.0012 [-0.010, +0.007], with identical agreement at 0.51. The contamination
misled the analysis and not the model, which is the more embarrassing of the two outcomes and the
easier one to miss.

## The notches are lost before regularisation, not by it

Part 00911 is a square with four notched corners. The square-on photograph resolves them clearly -
they are plainly in the mask - and the drawing is a plain four-line square. `rectangularise`
replaces a nearly-rectangular loop with its bounding rectangle, and on this part it turns 8 elements
into 4, so it looked like the culprit.

It is not, and two measurements say so.

**Loosening it does not recover the notches.** With the fill threshold at 0.995 or a cap of five
elements, 00911 goes from 4 elements to 6 against a truth of **12**, and its IoU falls from 0.959 to
0.933. The tracer had already lost 12 to 8 before `rectangularise` ran; the pass takes the last
four, not the first four.

**And it earns its place elsewhere.** Over 235 discriminating parts, loosening it changes 8 and
costs 0.9 points of exact primitives:

| arm | sketch IoU | exact primitives | elements |
|---|---|---|---|
| **shipped, fill 0.92** | **0.643** | **26%** | 9.43 |
| fill 0.97 | 0.643 | 25% | 9.46 |
| fill 0.995 | 0.643 | 25% | 9.47 |
| at most 5 elements | 0.643 | 25% | 9.45 |
| the real sketches | - | 100% | 11.88 |

Of the eight parts that change, six get worse and two lose their exact-primitive match. A pass that
turns a nearly-rectangular outline into an exact rectangle is right more often than it is wrong.

**The sweep also missed the part that motivated it** - 00911 sits outside the 280 parts sampled - so
the threshold was measured on parts it barely touches. Checking the motivating part directly is what
showed the fix does not work, and it should have come first.

The real deficit is upstream, in corner detection: 9.43 elements against 11.88, and 1.49 curves
against 5.13. That is the same wall four attempts have already failed against, and `rectangularise`
is not a way around it.

## Choosing the corner tolerance per part: it buys overlap and costs the drawing

`approxPolyDP` takes one global tolerance, so it cannot be loose enough for a large outline and
tight enough for a small notch on it. Part 00911's four corner notches survive at 0.010 and are gone
at the shipped 0.015 - twelve corners against eight - and that single constant is behind both
standing deficits, 8.6 elements against 10.5 and 1.5 curves against 5.1.

The ceiling is real. Over 522 discriminating parts at five tolerances, choosing the best per part is
worth **+0.024 [+0.020, +0.028]**, the shipped value is best on only 13%, and the oracle lands at
10.68 elements against a truth of 11.53 - so the tolerance genuinely controls the deficit.

**Posing it as selection was the right call and measurably so.** The two learned components here
that work choose among a few candidates labelled by the end-to-end score; the five that failed
predict geometric properties labelled by correspondence to a reference, which is only 62% reliable.
Following the first pattern:

| formulation | picks the best tolerance | against the shipped constant |
|---|---|---|
| 5-way classifier, n=159 | 38% | -0.0008 |
| 5-way classifier, n=522 | 52% | +0.0047 [-0.0001, +0.0095] |
| **per-candidate scoring, n=522** | **57%** | **+0.0072 [+0.0024, +0.0121]** |

More data moved it from nothing to marginal; scoring candidates independently rather than as one
five-way choice moved it from marginal to significant, on the same data.

**And on held-out parts it still loses where it matters.** Over 296 parts the selector never saw:

| arm | sketch IoU | exact primitives | elements |
|---|---|---|---|
| **shipped, one tolerance** | 0.602 | **32%** | 8.61 |
| learned per part | **0.606** | 27% | 12.35 |
| the real sketches | - | 100% | 10.54 |

+0.0047 [+0.0010, +0.0085] of IoU for **-4.6 points of exact primitives**. The element count tells
the story: the shipped constant undershoots at 8.61 and the selector overshoots at 12.35. It does
not find the right number, it trades one error for the opposite one, and the extra elements buy
boundary overlap while breaking primitive matches. That is the arc-gate trade again - more elements,
wrong elements - and region IoU is the metric that cannot see the difference.

It changed 150 of 238 parts and split them 82 to 68, which is close to a coin flip on which parts it
helps. Kept behind `P2F_EPS_MODEL=1`, off by default: unlike the contour-labelling attempts this one
has a measured positive on one axis and a documented reason it is not enough, which is worth having
for anyone who returns to the element deficit.

## The capture path would have failed on the first real photograph

Building the preflight check meant running `capture.pose` on an image loaded the way the pipeline
loads one, and it raised immediately: `trace.load` returns float32 in 0..1 and the aruco detector
requires uint8. **`carve.from_photos` calls `load` and hands the result straight to `pose`**, so the
entire board path - the one thing this project has been recommending as its largest available gain -
would have thrown on the first photograph anyone took.

Nothing caught it because the datasets have no board photographs, so the path had only ever run on
arrays built in memory by `capture_check`, which are uint8. Every synthetic validation passed:
16 of 16 views solved, 0.016 degrees of pose error. The failure lives exactly in the gap those
tests could not reach.

`rectify.as_uint8` now coerces at the detector boundary, so every entry point is covered rather than
the one that happened to be found.

Two smaller things the same exercise turned up, both in code written the same hour:

- **elevation came out negative.** `capture.pose` solves the board's own frame, whose z axis points
  away from the camera because the printed frame is left-handed in 2D, so a camera above the table
  reads as below it. The same convention that mirrored every board render earlier.
- **reprojection was measured against a guessed focal length.** Calling `pose` without intrinsics
  uses 1.2 x the image's long edge; on renders whose true pose is accurate to 0.05 degrees that
  produced 4 to 6 px of reprojection and a spurious "the poses are unreliable". Calibrating across
  the set first, as `from_photos` already does, brings it to 0.1 px. A preflight that does not
  mirror the real path measures itself.

`photo2fcstd-preflight` now separates the cases it should, on rendered captures:

| capture | verdict |
|---|---|
| 16 views, 11 to 69 degrees, walked around | usable |
| 16 views but none below 36 degrees | reshoot: height is bounded by the lowest view |
| 5 views from one side | reshoot: too few, and a 274 degree arc never looked from |

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

