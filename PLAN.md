# Plan

## Where we are

The objective is **primitive F1**: drawn primitives matched against the ideal sketch's, type by
type, over a denominator that never moves. Region IoU is secondary - it rewards an outline that
agrees and is blind to a missing feature, which is how a circle scored 0.95 on a scalloped disc.

Measured on `data/test_ids.txt`, 246 parts with trustworthy truth that nothing has been fitted on:

| | value |
|---|---|
| primitive F1, current code (runs/chain_shipped, 2026-09-04) | **0.637 [0.594, 0.678]** |
| of which `chain_arcs` (A7) against the code before it | +0.0161 [+0.0048, +0.0298] |
| primitive F1, fitted thresholds before A7 | 0.612 |
| primitive F1, hand-picked thresholds | 0.566 |
| paired gain from fitting | +0.045 [+0.026, +0.065] |
| region IoU, full photographed set | 0.629 [0.611, 0.646] |
| what a perfect sketch and true depth score | 0.918 |

A sketch-only run over all 1908 parts takes five minutes; a full run with solids takes twenty-two.

### The curriculum, and where it actually fails

Measured on the frozen test set with current code, 242 parts with trustworthy truth:

| tier | n | primitive F1 | drawn/wanted |
|---|---|---|---|
| 1-4 primitives | 115 | 0.651 [0.578, 0.721] | 1.88 |
| 5-8 | 56 | 0.740 [0.677, 0.799] | 0.99 |
| 9-16 | 36 | 0.696 [0.620, 0.767] | 0.82 |
| **17 or more** | 39 | **0.391 [0.308, 0.475]** | 0.54 |
| no holes | 146 | 0.660 | |
| with holes | 100 | 0.602 | |

(runs/chain_shipped, current code. `chain_arcs` moved the bottom three tiers - over-drawing on the
1-4 tier fell from 2.21 to 1.88 - and the 17+ tier not at all: saturation on many-curve parts is
untouched, as its A/B's per-tier split predicted.)

The failure inverts with complexity: simple parts draw about twice too many primitives, and
complex parts draw barely half of what they need. The bottom of the ladder is the better of
the two - it was 3.78 times over before the thresholds were fitted, and an earlier version of
this table said so; measure current code before believing any of it.

A clean disc is one circle. Three of four scored 0.00 while a forced revolve gave 1.00, so a
view whose fitted ellipse is within 5% of circular now picks revolve before any learned
selector votes: that head is trained on per-mode voxel IoU, which cannot tell one circle from
fifty-three lines. Worth +0.001 on the test set, because clean discs are about 1% of it.

**The bottom rung is circles, and half of that is capture.** Splitting the 1-4 tier by what the
ideal is made of, on the 246-part test set: 38 parts whose ideal is all lines score **0.880** and
21 of 37 draw exactly four, so rectangles are finished. 54 parts whose ideal is nothing but circles
score **0.560** and draw 4.1 primitives where 1.6 are wanted. That one group is 22% of the trusted
test set and it is the whole of the tier's over-drawing.

Of those 54, 27 already draw exactly the right circles. The other 23 divide sharply:

| | n | why |
|---|---|---|
| chosen view's fitted aspect < 0.5 | **12** | the disc was photographed on its rim |
| a round view, outer circle found, holes drawn as 4-11 line loops | ~7 | the hole circle gate |
| other | 4 | |

**None of the 12 has a face-on sibling.** Checking the fitted aspect of all three photographs of each:
the best sibling beats the chosen view by 0.01 or less on 10 of them, and no part reaches 0.7. All
three frames show the rim, so no view choice, threshold or tracer change reaches them - they are
F1 0.000 and stay there. That is **4.9% of the trusted test set permanently at zero for want of one
photograph**, which is the capture wall of "86% have no good view" with a name and a count.

**The hole gate was the code half, and it is now closed.** `ellipse_ok` requires
`rms < 0.04 * a` before a loop may be drawn as one circle. Two forms of loosening it were measured
on 559 discriminating parts with specs regenerated for both arms and built (`tools/ab_hole_gate.py`):

| arm | primitive F1 | structure | region IoU | changed | builds |
|---|---|---|---|---|---|
| absolute escape, 5 px | **+0.0033 [+0.0001, +0.0072]** | +0.0023 [+0.0003, +0.0048] | -0.0000 | 6 | 551 valid, 6 unsolved - identical |
| absolute escape, 8 px | +0.0040 [+0.0002, +0.0084] | +0.0032 [+0.0004, +0.0069] | -0.0002 | 8 | **550 valid** |
| relative 0.05 | -0.0187 [-0.0314, -0.0075] | -0.0103 | -0.0001 | 21 | |
| relative 0.06 | -0.0162 [-0.0293, -0.0045] | -0.0089 | -0.0003 | 29 | |
| relative 0.07 | -0.0239 [-0.0392, -0.0095] | -0.0111 | +0.0009 | 46 | |

**Both reverted, and the pair is the finding.** The absolute escape gains, and it gains a lot per
firing - the six parts it touches move +0.30 F1 each, four of them to a perfect 1.00. But it admits
small rectangles: a 40x20 rectangle fits an ellipse at **rms 1.97 px**, so no absolute threshold
excludes it, and `test_every_loop_shares_the_part_frame` goes red. On real data two of its nine
firings at 8 px drew circles for parts whose ideal has none. It works only because it is
self-limiting to loops under `rms_px / 0.04` = 125 px, where PrintCAD's holes happen to be round -
a dataset prior, not a geometry.

Loosening the *relative* threshold is the form that cannot admit a rectangle, since rectangles sit
at rms/a 0.083 to 0.111 at every scale against a circle's 0.000. It loses outright at every setting,
because the same threshold governs a 400 px outer loop, where rms/a 0.05 is 20 px from circular and
flattening it destroys the part.

So the residual cannot separate a noisy small circle from a small rectangle, exactly as sweep and
sagitta cannot separate a shallow arc from a straight edge. Any fix needs evidence a polygon would
not have. 8 px additionally cost a valid solid (00061, which builds up to 5 px and fails at 6).

An **ellipse primitive** now exists, and it is inert on every real input. No part whose ideal
holds an ellipse or b-spline had ever been drawn with the right primitives, because the pipeline
emitted only `line`, `arc` and `circle`; a loop too eccentric to be a circle is now emitted as one
`ellipse`, through the scorer and into FreeCAD as `Part.Ellipse`. A two-ellipse spec builds a valid
solid of volume 18221.237 against a hand-computed 18221.2.

On perfect input it wins - structure +0.0028 [+0.0002, +0.0061] overall, +0.0174 [+0.0009, +0.0375]
on the 112 parts whose ideal holds one, and exact primitives there **0% to 4%**. On photographs it
never fires: the A/B is identical to four decimals, builds unchanged, and the T-LESS carve-to-sketch
is 0.649 either way. The reason is measured rather than assumed - an ellipse fitted to a
photo-traced loop leaves a residual of 0.073 of the semi-major axis on parts that really are
elliptical and 0.063 on parts that are not, with matching fire rates at every threshold. **The fit
carries no signal about whether the part is an ellipse**, so loosening would fire indiscriminately,
which is what sank the arc gates. Kept as neutral-not-harmful, and live the moment an outline is
cleaner than a photograph's.

**A `bsplinecurve` primitive now exists end to end** (2026-09-05): traced as a chain of four or
more same-turning pieces that no sub-chain can fit as a circle, scored by its sampled ring, built
as `Part.BSplineCurve` with interior poles bound through `params.scale` (the x8.000 scale test
passes with a spline in the sketch; end poles are pinned by the endpoint dims, binding them too
over-constrains, and tangent joins next to a spline must be plain or the solver conflicts - both
learned from 00061). On photographs it is measured neutral-not-harmful: 5 of 351 parts change,
curves +0.0011 [+0.0001, +0.0029], builds identical (`tools/ab_bspline.py`).

**And it does not touch the 58 parts whose ideal holds b-splines - zero changed - which is the
diagnosis.** Their mean F1 is 0.309 against 0.692 elsewhere, the largest bucket left (~0.055 of
overall F1), but the deficit is not curves-drawn-as-lines: their smooth boundaries are consumed
piecewise *as arcs* before any chain can form, so the loss is type accounting (arc drawn where
bsplinecurve is wanted) plus fragmentation (7 elements drawn where 14.6 are wanted). The identified
next mechanism was then attempted: `trace.tangent_merge` joins smooth chains (tangent break under
35 deg - strict "T" joins fire on only 2 of 7 junctions of a real freeform fit) holding two or
more arcs of scattered radii into one spline when no single circle explains the merged run. On the
target bucket it works: **F1 +0.0166 [+0.0067, +0.0285]** on the 58 bspline-ideal parts, region IoU
+0.0102, and the builds get *better* (342 valid against 340, unsolved 9 against 11), with 00026
going 0.00 to 1.00 as the single closed spline its STEP wants.

**But it cannot ship, and the reason is the same wall found twice already.** Off the target bucket
it merges *designed* multi-arc boundaries - ovals of four tangent arcs, arc-and-line profiles, even
hexagons whose traced arcs were spurious - and two principled gates (radius spread, an interleaved
line) leave the leak in the same class: overall F1 -0.0046 [-0.0118, +0.0014], losers -3.97 summed
against winners +1.34. Freeform-versus-designed-arcs is a *curvature-profile* question -
continuously varying against piecewise constant - and one matted contour buries exactly that signal
in wobble. A carve cross-section measures it directly. `P2F_TANGENT_MERGE=1` enables it
(off by default); `tools/ab_tangent.py` re-measures; the infrastructure and the unit tests stay.

Tried and removed on the way: an elliptical arc per contour run. `cv2.fitEllipse` scores 21 to 292
on runs the circle fits at 0.4 to 1.8, because `corner_runs` splits a curve into 20-50 degree pieces
and any smooth curve is locally circular. The ellipse spans several runs, so the run is the wrong
level to fit it at.

### What is shipped, and what it was worth

| component | held out | on by default |
|---|---|---|
| depth from frozen DINOv3 features | +0.040 [+0.024, +0.056] | yes |
| mode selection from the same features | +0.080 [+0.063, +0.097] | yes |
| fitted thresholds, eight of them | +0.045 [+0.026, +0.065] | yes |
| keeping detail on feature-bearing contours | +0.006 [+0.003, +0.008] | yes |

### What was measured and declined

Every one of these looked large as an oracle and collapsed when a selector had to find it without
the answer. They are listed so nobody pays for them twice.

| idea | oracle | what a selector actually got |
|---|---|---|
| choosing which photo to trace | +0.082, about half of it noise | +0.004 [-0.001, +0.009] end to end |
| carrying nine view-and-mode hypotheses | +0.081 | pipeline already picks the best 61% of the time |
| a per-part simplification tolerance | +0.059 | +0.004 [-0.010, +0.019], 7% of the headroom |
| undoing viewpoint tilt | +0.090 | 11% from pixels, and the target is mostly undetermined |
| rotational symmetry, five attempts | exact on synthetic gears | nothing on photographs |
| a bigger head on the same features | | +0.000 [-0.003, +0.004] |

### What actually limits it

Not modelling. Of the parts whose sketch scores under 0.4, **86% have no good view among the three
photographs** and **89% are still under 0.4 traced raw**, before any regularisation. Thirteen per
cent of parts cannot be reproduced by extruding their own ideal sketch, so their ceiling is not 1.0
and never was. The remaining work is capture, or accepting the scope.

## What comes next, ranked (2026-09-04)

The gaps ladder (`docs/gaps.md`) is complete: every code item is shipped, closed by measurement, or
measured-and-reverted, and the two person items are answered. What follows is the plan from here,
in the order the record supports - not the order the ideas are exciting in.

### 0. Do not add another selector to PrintCAD

The two slots where a learned component wins here - choosing among candidates the geometry already
produced - are exhausted: a perfect mode oracle is worth +0.0003, and view choice is captured.
Every model that replaced a geometric step lost, and 1,908 parts is not getting bigger. New
in-distribution PrintCAD models are mined out; anything proposed must name which of the two winning
shapes it takes, or why a third exists (`tilt` was one: a quantity with no geometric step at all).

### 1. One photo session unlocks three blocked components (C1/C2/C5)

The cheapest work left is not code and not a model - it is ~50 photographs at a desk:

- **~16 photos of one part on a strongly textured surface** gates SfM carving on real photographs
  (validated synthetically at 0.031 deg; carving doubles solid IoU and measures depth instead of
  guessing it).
- **A few dozen frames with the ChArUco board in view** are free tilt labels; the tilt head works
  (4.8 deg on T-LESS) and its gate refuses every PrintCAD photo until it is retrained on them.
  After that the reshoot check works board-free for good - and shooting square is worth +0.13,
  the largest single number on the board.
- The **twelve discs photographed on their rims** (4.9% of the test set at F1 0.000) need one
  square-on frame each.

### 2. Direct primitive prediction - closed at zero, and the closure is evidence-complete

Built as designed (2026-09-04): a 1.9M-parameter encoder-decoder (`seqnet.py`) that decides only
breakpoints and span types on the traced contour - corners are contour-point indices so
localisation inherits contour precision, fitting stays geometric, targets from Fusion 360 Gallery
plus the PrintCAD *tuning* split under homography, test split never entering, split by design.

**On its own distribution it clearly beats the tracer**: breakpoint F1 0.68-0.72 against 0.51-0.55,
span-type accuracy 0.89 against 0.79-0.83 (majority 0.80), held out by design. **End to end on
photographs it loses decisively** (n=350, specs both arms, built): primitive F1 **-0.067
[-0.087, -0.049]**, structure -0.051, valid solids 339 to 328, curves drawn 3.67 against the
geometric arm's 1.59 - on real contours it calls matting noise curved.

**Correction (same day): the "noise augmentation made it worse" claim was wrong.** The v3
augmentation patch failed silently in a backgrounded shell; contour roughness was identical between
the v2 and v3 datasets (mean |turn| 0.1910 vs 0.1911), so the -0.103 arm was a seed re-roll of
-0.067. Caught by measuring the data, the diff-the-snapshots rule applied one level down.

**The wobble was then actually run, and the whole arc is a ratchet that ends at zero.** Correlated
boundary wobble, amplitude calibrated to real photographs (mean |turn| 0.249 trained vs 0.258
measured), labels from the clean geometry - then two geometric acceptance gates, each targeted at a
failure class the previous stage exposed:

| stage | primitive F1 vs tracer | what it fixed |
|---|---|---|
| model, no augmentation | -0.067 [-0.087, -0.049] | |
| + calibrated wobble | -0.034 [-0.050, -0.019] | curve over-firing 3.67 to 1.95, builds undamaged |
| + parsimony gate (<= tracer + 2 elements) | -0.017 [-0.028, -0.008] | rectangles shattered into 18-25 arcs |
| + fit gate (residual <= 1.25x tracer) | **+0.0012 [+0.0000, +0.0027]** | right-primitives-wrong-geometry: 13 parts at F1 1.00 with IoU collapsed to 0.30 |

Each gate reclaimed score by trusting the model less; the endpoint accepts it on about five parts
of 351 and is indistinguishable from zero. Letting the model vote "this whole loop is round" was
also tried and it voted round on wobbly rectangles, so its whole-loop judgment fails the same way
its span judgment does. The conclusion is clean: the model's synthetic skill (breakpoint F1 0.71
against the tracer's 0.55, held out by design) does not survive real contours even with
measurement-calibrated noise, and geometric gates strong enough to block its errors block its wins
too. `P2F_SEQNET=1` enables the gated path; it stays off. The pipeline remains as infrastructure
(`tools/seq_data.py`, `seq_train.py`, `seq_eval.py`, `ab_seq.py`).

**"The archive cannot label real contours" was then screened, and it is false**
(`tools/label_screen.py`, `docs/figures/label_screen.png`). Aligning each part's ideal sketch onto
its real traced contour - rotation x per-axis scale x flip, chamfer-gated - transfers labels at:

| chamfer gate | yield (n=60 tune parts) | label error, median | p90 |
|---|---|---|---|
| cost <= 8 | 32% | 1.9 px of a 720 px part | 6.4 px |
| cost <= 12 | **55%** | **2.6 px (0.36%)** | 8.0 px |
| cost <= 18 | 68% | 3.4 px | 9.1 px |

The overlays verify it by eye: straight flanks inherit straight, fillets inherit curved. At 55%
yield that is roughly 350-400 tune parts x 3 photos of *real* labelled contours - the dataset the
seqnet closure said could not exist. Breakpoint tolerance in training is +-3 resampled points
(tens of raw pixels), so 2.6 px of label error is well inside it. The retrain was then run
(a rented 4090, four mixtures of synthetic and real, `tools/seq_train_remote.py`, ~$0.70):

| training mix | token acc on held-out real photographs |
|---|---|
| synthetic only | 0.4992 |
| + real x4 | **0.5085** |
| + real x12 | 0.4994 |
| + real x30 | 0.5040 |

Real labels help the proxy by less than its own noise (the real validation set is effectively ~33
photographs), and the best mixture, gated, scores **F1 -0.0010 [-0.0033, +0.0004] end to end** on
all 351 parts and **exactly 0.0000 on the 85 tune-excluded parts** - the gates accept its
proposals on six parts. So the same-distribution hypothesis is measured at the scale the archive
can deliver, and the scale is the problem: the alignment gate passes 14% of photographs, giving
2,028 samples from 338 photos, against 27,581 synthetic - enough to pull the model off the
synthetic distribution (its synthetic breakpoint F1 fell 0.71 to 0.66) and not enough to buy real
skill the gates would keep. **The fourteenth attempt closes at zero, like the thirteenth, and the
closure is now evidence-complete**: capacity, labels, representation, noise realism, and
same-distribution data have each been fixed in turn, and the end-to-end score never left zero.
What remains unexplored is scale itself - tens of thousands of real labelled contours, which means
photographing parts, which is the same conclusion every path in this file reaches.

### 3. If the goal is beyond PrintCAD: test-time render-and-compare (item 5 below)

The only idea that scales: a candidate model is right if it explains all the photographs, which
turns every unlabelled photo set into supervision at inference time. The crude version gave +0.010.
Done properly it is differentiable silhouette rendering over the sketch program's continuous
parameters, with pose as a nuisance variable - respecting the measured trap that intersecting
independently-registered views destroys material. This is the one research-grade bet in the file.

**Its cheapest version is now measured, and it is a null on the current capture protocol**
(`tools/depth_consistency.py`, n=92 parts with STEP-recorded depth). Choosing the depth whose
extruded prism best explains the other two photographs scores a median absolute log error of
**1.74** against the shipped predictor's 0.24 - and against a plain constant's 1.13, so it loses
to knowing nothing. Restricting to parts where the consistency score actually varies makes it
worse, so the variation is bias, not signal. This is the edge-on-estimator result re-derived from
the other direction: PrintCAD's three photographs are same-face dominated, and no amount of
rendering or comparing extracts a depth they do not contain. The bet stands only on top of the
sixteen-view capture - which moves its prerequisite into item 1 and removes any reason to attempt
it on the archive.

### The synthesis (2026-09-05): the capture rig is two instruments, and that resolves the strategy

Read the fourteenth zero for what it says. The final attempt removed every confound - capacity,
labels, representation, noise realism, distribution match - and landed at exactly 0.000 under gates
that only reject provably-worse drawings. That is not "models do not work here"; it is that the
geometric prior is already near the optimum *for this input*. Learner and tracer plateau at the
same wall because the wall is in the data: a shallow arc on a wobbly matted contour is genuinely
undetermined by that contour. You cannot classify what the input does not determine. The correct
response is to change the observable, not the estimator.

The sixteen-view capture does exactly that, twice over:

- **It changes the observable.** A cross-section of a carved solid has *measured* curvature at
  metric scale with matting noise averaged across sixteen silhouettes - the shallow-arc decision
  that is undecidable on one contour becomes trivial geometry on the carve. Already measured:
  0.649 carve-to-sketch on real T-LESS photographs, ~0.78 solid IoU against the photo path's 0.43.
- **It is a label factory.** The retrain died on scale: 14% alignment yield, 2,028 samples. With
  SfM poses in hand, aligning the known STEP to each frame stops being a fragile similarity search -
  the pose *is* the alignment - so yield goes to ~100% and every photographed part emits ~16
  exactly-labelled real contours. The dataset the fourteenth attempt lacked falls out of the same
  session that feeds carving.

So there is no choice to make between the geometric path and the learned path: build the rig, and
the choice resolves itself downstream with data neither side can currently claim. ML re-enters
afterward in the shapes that win here - the axis classifier retrained beyond PrintCAD (89%
in-domain, 29% out), selectors over carve outputs, and render-and-compare, which the null killed
on three same-face photographs but which becomes well-posed over sixteen registered views.

The week's fourteen zeros are a proof that the next model's first layer is a tripod.

**Measured addendum (2026-09-05, `tools/rig_value.py`): the rig's sketch claim is parity, not
victory - and the synthesis above overclaimed.** Rendering sixteen true-pose views of each part's
own truth mesh (the gated renderer; SfM equivalence 0.987-0.997), carving at 0.25 mm, slicing the
mid-section, extracting it with marching squares at sub-voxel precision, and tracing with the
production tracer, paired against the photo pipeline on the same 57 trusted parts:

| | photos | rig section |
|---|---|---|
| primitive F1 | 0.564 | 0.529, paired **-0.035 [-0.105, +0.031]** |
| exact primitives | 14% | 18% |
| curved drawn / wanted | ~1.6 | **1.60** / 4.30 |

The rig section draws the *same* 1.6 curves the photograph does. This was predictable from
`tracer_ceiling` (perfect drawings in: 1.76 curves) and the synthesis conflated two walls: the
noise floor explains the *marginal* photo-versus-perfect gap (1.62 vs 1.76), but the large curve
deficit is the decomposition's own saturation on any input, and better input does not fix the
decomposition. Getting here cost four harness bugs, each the kind that fakes a conclusion:
PrintCAD meshes stand on edge in their STLs and must be laid flat; `base_axis` guesses when the
axis is known by construction; a fixed carve box smaller than the part "keeps" everything; and
voxel staircase must be extracted with marching squares, not blurred.

**The fifteenth replacement was then screened on the rig sections themselves, and it is the
cleanest zero yet.** The reopened door - a model consuming carve sections, "trainable with no
domain gap since simulated sections are the deployment distribution" - was built
(`P2F_SECTION_NOISE=1` in `tools/seq_data.py` simulates the rig's own voxel-quantise-and-marching-
squares extraction; 17,200 samples, val 0.549) and run through the rig harness gated: **zero of 57
sections changed**. The diagnosis fired one gate-integrity fix on the way - `drawn_residual`
sampled long lines at six points, inflating residuals for exactly the fewer-longer-element drawings
a model proposes; it now samples by length (the earlier +0.001 photo-A/B gate numbers used the
biased version) - and the verdict *survived* the fix: the tracer fits real sections at 2-3 px while
the section-trained model misses by 7-18 px. The premise was false: quantisation was simulated but
the hull's geometry (skirt corners, elevation-dependent bulges) was not, and the domain gap
reappeared one level down, conserved. `data/seq_model.pt` currently holds this section model.

**The rig's re-priced value, all measured:** depth to 1-5% (against none), solid IoU ~0.43 to
~0.79, the twelve discs, the tilt labels - and for the sketch itself, parity under the current
tracer. The capture is still the only path that gains anything; it just gains the *model*, not
the drawing, until something better than the shipped decomposition consumes the section.

### The optimal decomposer: the ceiling explained, and greed exonerated (2026-09-05)

The last untried corner was the decomposition itself, replaced by better *geometry* rather than a
sixteenth model: `photo2fcstd/dptrace.py`, dynamic programming over every contour span with O(1)
incremental line and circle fits and an MDL cost - the same fits and the same arc gates as shipped,
applied to whole spans instead of greedy fragments. Four cells, all measured (`tools/ab_dp.py`,
`rig_value` with `P2F_DP_TRACE=1`):

| input | DP vs greedy, primitive F1 | curves drawn (~4.6-5.1 wanted) | builds |
|---|---|---|---|
| perfect rasterised ideals (n=414) | **+0.0160 [+0.0032, +0.0291]**; 4+-curve parts **+0.0643**, structure +0.0866, saturation broken | 1.70 -> 2.84 | |
| photographs, sigma 1.6 (n=351) | **-0.0664 [-0.0883, -0.0449]** | 1.61 -> 5.00, wobble tiled with arcs | 340 -> **310** valid |
| photographs, sigma 3.5 measured | -0.0318 [-0.0511, -0.0129] | 3.48 | 326 valid - halved, not closed |
| rig carve sections (n=57) | -0.0572 [-0.1018, -0.0142] | **4.35 of 4.30 wanted - right count, wrong places**; exact 18% -> 7% | |

Three conclusions, each worth keeping. **The clean-input ceiling was never fundamental** - the DP
breaks the saturation the moment the input is truly clean, so the curve deficit on perfect input
was the greedy decomposition after all, not information. **Greedy corner-first is an accidental,
load-bearing regulariser** - exact optimality under a mis-specified noise model faithfully fits the
noise, and greed's inability to see long spans is what protected the shipped tracer from wobble;
this is the third and strongest confirmation of the perfect-input law. **No available input is
clean enough**: matting wobble and hull-section noise are both structured, not Gaussian, so the
optimal estimator mis-models both; the DP+rig combination draws the right number of curves in the
wrong places. `P2F_DP_TRACE=1` enables it; it stays off; on genuinely clean input (a scanned
drawing, a DXF raster) it is the better tracer, measured.

### The Bayes route's cheapest form failed its screen (2026-09-05)

The proposed fix for the noise floor - a design prior over sketches settling decisions the contour
cannot - was screened at its cheapest (`tools/prior_screen.py`): a count-based prior over loop
compositions, fitted on 700 tuning-split ideals with the evaluated part left out, choosing between
the two candidate drawings every measured A/B left behind (38 pairs where the truth differs and the
compositions differ). It picks the truthfully better drawing **47% of the time overall and 41% on
the tangent-merge pairs - below both coin flip and the better fixed policy (61%)**. Where it looks
good (holegate, 83%) it merely matches the fixed policy. The failure mode is visible in the pairs:
a composition prior rewards *a-priori-common* drawings, and the truthfully better candidate is
often the less common composition that happens to match this part. Composition frequency alone
does not carry the discriminating signal, and the screen's own terms said that sinks the full
generative version too. What remains untested is a joint model where the prior is conditioned on
geometry rather than marginal over it - but that re-enters photograph territory, where fourteen
zeros stand. The noise floor holds; the fix is still the capture.

### The label factory exists, measured (2026-09-05): 60,000 real contours at 1.4-1.9 px

The missing axis every closure named - tens of thousands of real labelled contours - was priced by
downloading BOP-Industrial (ITODD, IPD, XYZ-IBD validation splits, ~14 GB; poses public) and
pooling with the T-LESS training split already on disk. `tools/bop_yield.py` counts instances,
uses the datasets' own `visib_fract` to answer the bin-clutter question by counting, and measures
label error as the real modal mask's boundary against the CAD silhouette projected through the
*given* pose:

| dataset | instances | objects | visib>95% | visib>99% | boundary, median / p90 |
|---|---|---|---|---|---|
| ITODD val | 123 | 28 | 105 | 78 | 1.7 / 2.3 px |
| IPD val | 2,264 | 10 | 723 | 282 | **1.4 / 2.1 px** |
| XYZ-IBD val | 59,850 | 15 | 26,707 | 19,645 | 1.7 / 2.4 px |
| T-LESS train | 37,584 | 30 | 32,587 | 11,156 | 1.9 / 3.7 px |
| **pooled** | **99,821** | **83** | **60,122** | **31,161** | |

Sixty thousand near-unoccluded real photographs of industrial parts whose contours label
themselves through the pose - against the archive alignment route's 338 photos at 2.6 px, a
178x scale-up at better label quality, acquired without a camera. The standing risk is diversity:
83 distinct objects, split-by-object mandatory, and whether 83 shapes teach a decomposer or get
memorised is exactly the question the sixteenth attempt would answer - on captured deployment
data this time, as the conservation law demands. The corpus also upgrades every OOD evaluation
in this file from T-LESS-only to four independent industrial sources.

### The sixteenth attempt: the first non-harmful learned replacement (2026-09-05)

Built on the conservation law's own prescription - inputs are real BOP contours (the deployment
distribution, unsimulated), labels are the production tracer run on each frame's *clean* CAD
silhouette projected through the given pose and transferred to the real contour across the measured
1.4-1.9 px gap. 20,468 real self-labelled contours, 79 industrial objects, trained split-by-object
on a rented 5090 (`tools/bop_seq_data.py`; held-out-object token acc 0.486).

End to end on 351 PrintCAD photographs, every one held out from the BOP training, gated and built:
**primitive F1 -0.0030 [-0.0089, +0.0027]** - straddling zero, and **curves drawn went slightly
down, not up** (1.61 to 1.65). It is the first of sixteen learned replacements that is not clearly
harmful: every prior one, the raw seqnet at -0.067 included, made photographs worse by inventing
curves on matting noise. Training on genuine deployment noise removed that failure mode.

But it gained nothing, for two measured reasons. **Cross-dataset transfer is flat** - 79 mostly-
straight industrial objects (8% curved spans) carry no curve knowledge to printed parts, and the
held-out-object proxy (0.486) sits below the in-distribution 0.55. **And there was little to learn**
- on BOP's own parts the greedy tracer already scores 0.795 breakpoint F1, near-optimal on straight
industrial edges. So the law resolves cleanly: **the domain gap was the harm mechanism, the
tracer's ceiling is the wall.** Deployment-matched data buys neutrality, the first non-negative
learned result here; a gain still needs real photographs of *curved* parts the tracer actually
fails on - neither the industrial corpus nor the printed archive, but the photo session pointed at
the right shelf. `data/bop16.pt` holds the model; `P2F_SEQNET=1` gates it; off by default.

### The pixel screen: light does not rescue the contour, in this form (2026-09-05)

The one axis no attempt had changed was the input's information content, and shading was the
measured candidate (the tilt result: pixels beat silhouettes by 3.6 deg). Screened for about a
dollar on a rented 5090 (`tools/pixel_screen_remote.py`): the seqnet architecture with seven grey
intensities sampled along each contour point's normal, against the identical architecture with
those channels zeroed - 10,869 real T-LESS contours labelled through the pose, held out by object,
two seeds:

| arm | held-out-object token acc |
|---|---|
| contour only | 0.4152 |
| contour + normal intensity profile | 0.4129 - **delta -0.0022, nothing** |

A local intensity profile carries no breakpoint or span-type signal beyond the contour. The caveat
is the encoding: this tested a 7-sample 1D profile, not a learned patch embedding - the tilt win
used DINOv3 on full crops - so a richer pixel representation is not excluded, but it no longer has
a cheap screen's presumption in its favour. The second expert move stands untouched by this null:
printing curve-heavy parts and photographing them targets the data deficit, not the input encoding,
and remains the one unexplored path to a drawing gain.

### Pose fusion is in the pipeline (2026-09-05)

`photo2fcstd/fuse.py` closes the loop the record kept pointing at: a posed multi-view capture now
produces a **buildable FCStd** through the normal spec and build path. `fused_spec(carved)` takes
the carve, extracts the mid-section by marching squares, traces it with the production tracer, and
emits an outline spec whose depth is the carve's measured interior-top-height - **the first spec
whose `depth_trusted` is True on the pipeline's own evidence**. `tools/sfm_real.py --gate` now runs
photographs-to-FCStd end to end on rendered frames: valid solid, sketch solved with zero free DoF,
depth 3.50 mm in the params sheet against 3.73 true. Unit-tested (`tests/test_fuse.py`).

Field-tested the same hour on the user's fan: three phone HEICs. The three-photo path chose the
cluttered side view (a laptop in frame - outside the capture protocol) and its sketch conflicted;
on the two face-on views it built a valid, fully-constrained square plate - and the grill and
blades are invisible because black-on-black defeats matting, and depth is guessed because both
photos show the same face. Which is the whole thesis in one object: the classic path gives the
outline; the fan's real geometry needs the sixteen-photo fused path it now has.

### Two general fixes the fan exposed (2026-09-05)

A user's box-fan, three phone photos, drove two archive-safe corrections - and one honest scope
limit. (1) `largest()` now drops a connected component that touches the image border unless it is
the only one: clutter (a laptop, a cable) that runs off-frame is never the part, the part is
centred. (2) `face_on()` gained a relative rectangularity veto - a view whose silhouette fills its
bounding box far less than a sibling's (rect 0.29 against 0.96) is edge-on and refused, which the
elongation veto missed because the stale pre-regularisation `select.elongation` read 1.14 on the
cluttered edge-on frame. A/B on 197 trusted archive parts: **-0.0018 [-0.0113, +0.0076], 5 changed**
- statistically zero, and the fan now auto-selects a face-on view and builds a valid, fully-
constrained square shroud plate from all three photos.

**The scope limit, kept honest.** The fan's central opening is a *grille* - dark wire, bright hub,
background through the gaps - not a segmentable hole, so a single photo yields the shroud outline
and nothing inside it. A dark-hole recovery that closed the grille into one bore was tried and
reverted: it was fan-overfitting, unproven on the archive, and the repo's own rule forbids it. The
bore, the blades and the depth are exactly what the sixteen-photo fused carve path (now in the
pipeline) exists to recover.

### Dark-opening recovery: real, measured, opt-in (2026-09-05)

The fan's face-on views segment as a solid square because the grille opening shows dark fan blades,
not the background - `recover_holes` (background-colour match) cannot see it. `recover_dark_holes`
does: it closes the dark grille fragments into one region and carves it, turning the face-on view
into a correct square-plate-plus-central-bore sketch that builds a valid solid **from the existing
photos, no reshoot**. But the archive A/B is decisive against shipping it on: **-0.0235
[-0.0458, -0.0031], 41 of 158 parts hurt** - PrintCAD's printed parts have shadows and dark recesses
that read as false holes. So it ships **off by default, `P2F_RECOVER_DARK=1` to enable**, for parts
whose openings are genuinely dark cavities (a bore, a grille, a counterbore). The learned view
ranker also does not prefer the newly-holed face-on view (it was trained without such parts), so the
face-on model is reached by naming the face-on views or `P2F_VIEW_MODEL=0`. This is the single-view
wall again, priced exactly: the fix that models the fan costs the general case, so it is a switch,
not a default.

### The LLM belongs in recognition, not tracing - and it is the winning shape (2026-09-05)

Sixteen learned attempts fed a contour to a small model and lost, every one replacing a geometric
step. A vision LLM in that role loses too - it cannot place a primitive to sub-pixel. But in the
*other* role, the only one that has ever won here (a selector / prior), it is decisive: it states
WHAT the part is - object class, which photo is face-on, what primitives the sketch holds, which
regions are openings - and the geometric engine measures WHERE. `tools/recognise.py` demonstrates
it on the fan: recognition says "fan shroud, square plate, central bore, four corner screws," and
the tracer measures the plate outline and bore radius from the recognised face photo. Result: a
valid solid with plate + bore + 4 screws, the correct CAD sketch, where the pure-geometry pipeline
needed two opt-in flags and still picked the wrong view.

**It is now a general module, not a fan script.** `photo2fcstd/recognise.py` defines the contract
- `recognise_prompt(n)` is the exact instruction a VLM gets with the photos; it returns a small JSON
(`face_photo_index`, `outline` in square/rect/disc/trace, `openings`, `screws`, `depth_ratio`) and
never coordinates. `build(rec, photos)` turns that answer into a buildable spec, the tracer
measuring every dimension from the recognised face. Tested end to end on a synthetic square-with-
bore (`tests/test_recognise.py`) and run on the fan with a real VLM answer: valid solid, plate +
bore + 4 screws, zero free DoF. Wiring a live vision model is one call that fills `rec`; today the
recognition is supplied by the assistant reading the photos, which is the same interface.

This is the resolution of the whole arc. The noise floor and the conservation law say the drawing
cannot be improved by a better estimator on the contour - both were proven. The LLM does not touch
the contour: it supplies the *identity* that no 2,000-part model and no matting network carries,
turning the hardest single-view failures (a grille read as solid, a tilted view preferred) into
trivial recognition. Recognition for structure, geometry for precision - the architecture the
sixteen nulls were pointing at all along.

### What not to do, measured

A fifteenth replacement at current data scale (expected value zero, tight CI), another local arc
statistic (three died on 2026-09-03/04; the family is exhausted), more augmentation knobs (two
measured), more acceptance gates (they converge to the tracer), threshold tuning (+0.007, inside
the noise floor), a bigger head on the same features (+0.000), trusting an oracle ceiling at face
value (they collapse to about a tenth here, six times measured), or any plan whose first step is
not a camera.

## 1. Fix the instrument first - done

Nothing else is worth running until the benchmark can detect what we are chasing.

- Evaluate on all 1907 photographed parts, not 200. On the box: ~15 min to segment 5731 photos on
  CUDA, ~15 min per run, about $0.60 for a full sweep. `watch_and_pull.sh` pulls each run as it
  finishes so a self-destruct cannot eat the results again.
- Report a paired bootstrap CI in `summary.txt` for every run, against a named baseline. A run that
  cannot say whether it changed anything is not a result.
- Group the split by geometry signature; PrintCAD contains near-duplicate parts.

**Done when:** `photo2fcstd-bench` prints `mean 0.4xx [lo, hi] vs <baseline> Δ+0.0xx [lo, hi]`, and
a +0.01 change is detectable.

**Status: reporting done, full-set run in flight.** `--baseline <run>` now prints a paired bootstrap
delta and the run's own resolution. Its first use caught a real +0.031 [+0.013, +0.050] from the
parallel commits between `v17` and `ci_check` — a change neither of us had measured.

## 2. Prove the metric path on real photographs - done on T-LESS, and the board is no longer required

**The printed target is not needed for pose.** Structure-from-motion recovers camera poses from the
scene, gated synthetically in `docs/sfm.md`: 24 of 24 images registered, camera rotation error 0.035
deg median against carving's 2 deg budget, and carving from SfM poses agrees with carving from true
poses at 0.995. Two hard requirements - a strongly textured surface, since at a quarter contrast 0
of 16 images register, and sixteen views rather than eight. Untested on real photographs.

So the ask is now sixteen photographs of a part on a newspaper, not a printed target in every frame.

**Carving to a sketch from real photographs measures 0.649**, not the 0.715 previously extrapolated:
`tools/tless_sketch.py` on 21 discriminating T-LESS objects, against a trivial circle's 0.493. It is
a different part set from PrintCAD, so it is not comparable with the photo path's numbers here.

## 2b. Prove the metric path on real photographs - done on T-LESS, board still unshot

`--rectify` and `photo2fcstd-carve` were validated only against synthetic boards and a simulated
box. Every millimetre figure quoted for them came from geometry we generated ourselves.

This does not need photos we shoot. [BOP](https://bop.felk.cvut.cz/datasets/) publishes exactly the
required thing: **T-LESS** is 30 industrial, textureless, largely symmetric parts — the hard case
for silhouettes — photographed on a calibrated turntable with CAD models in millimetres and a pose
per image. `photo2fcstd-bop` feeds those poses to the same carving code and reports the error
against the CAD model per dimension, over 30 real objects.

The printed target remains worth one session of shooting, for two things BOP cannot cover: the
ChArUco pose recovery end to end, and your own parts.

**Done when:** the carved model is compared against the caliper readings we already hold, and the
error is stated in millimetres per dimension. If it is worse than the two-photo path, say so and
keep the two-photo path.

## 3. Score sketches, not solids - done, and it is now the primary objective

Every headline number is voxel IoU — whether the *solid* matches. The requirement was an editable
sketch. `ideal_sketches.py` already holds the true primitives for all 1907 parts and
`sketch_score.py` can compare against them; neither is reported by the benchmark.

**Done when:** each run reports primitive agreement alongside IoU, so "the ring became two circles"
is a measured claim rather than an anecdote from a gallery.

**Status: done.** Every run now reports, and records per part in `events.jsonl`: how many parts emit
a sketch at all, the region IoU of that sketch against the ideal one with a CI, and how many
reproduce its exact primitive counts. On the 200-part set: 97 of 200 emit a sketch, region IoU
0.583 [0.503, 0.664] on the 55 with trustworthy ground truth, 16 exact.

## 4. Reformulate mode selection - done, +0.080 held out

Only after 1. Classification throws away most of the signal — a quarter of the labels are near-ties
and the model is punished for choosing between equally good answers.

- Regress IoU per mode (four targets per part) and take the argmax, or learn a pairwise ranking.
  Same labels, four times the supervision, ties stop being errors.
- Replace the sixty hand-crafted silhouette statistics with embeddings from a frozen pretrained
  backbone (DINOv2 is already on this machine) and k-NN against the training parts. With a few
  hundred labelled parts, retrieval beats fitting.
- Filter to decisive labels for training; keep ties in evaluation, where they cost nothing.

**Done when:** a held-out comparison on ≥426 parts shows a CI that excludes zero. The switch stays
off until then.

## 5. Self-supervision from the photos - not started, and the consistency signal measured weak

The photos are their own supervision: a candidate model is right if it explains all views. Crude
64-direction silhouette consistency already gave +0.010. Done properly — differentiable silhouette
rendering, or a better scoring function — it turns every unlabelled photo set into training signal
and works at inference time with no dataset at all.

This is the only item that scales beyond PrintCAD, so it is the one worth real research time once
the instrument and the metric are trustworthy.

## What the night of 31 August shipped

Two changes, each proven on parts the model had never seen, because it turned out that was the
only way to prove anything here.

| | held out | full set |
|---|---|---|
| pixel depth head | +0.040 [+0.024, +0.056] on 481 parts | -0.013, which is contamination |
| pixel mode selector | +0.080 [+0.063, +0.097] on 620 parts | +0.089 |
| both, against the evening's starting point | | **+0.077 [+0.066, +0.089]**, 0.463 -> 0.541 |

Sketches came through unharmed: region IoU 0.593 -> 0.630, exact primitive reproductions
260 -> 301, and 1871 of 1873 parts still emit a sketch.

### Where the remaining headroom is not

Two things were checked before stopping, and both came back negative in a useful way.

- **A stronger mode head buys nothing.** Gradient boosting on 128 principal components scores
  0.577 against ridge's 0.577, +0.000 [-0.003, +0.004]. The 0.038 still separating the selector
  from its oracle is not a modelling shortfall on these features; ridge already extracts what the
  embeddings contain. Closing it needs different information.
- **Tilt is mostly invisible, even in pixels.** With warp-oracle labels extended from 80 parts to
  699, ridge on the photo the warp acts on recovers +0.021 [+0.006, +0.036] of sketch IoU, which
  is 11% of the ceiling. That beats every silhouette criterion, all of which recovered 1% or less
  and most of which were negative, so pixels do see something the mask cannot. The other 89% is
  not visible from a single photograph and needs the board.

### Four invalid comparisons, in four different directions

The measurements were wrong more often than the code was, and each was wrong differently.

- **A second change landed mid-experiment.** The first depth A/B compared two arms whose code
  snapshots differed, because the parallel session shipped a view model between them. Diffing
  `runs/<name>/code` is now the first step after any surprising delta.
- **The metric had a blind spot.** A mode selector scored +0.115 by routing 23% of parts to
  stations, which wins voxel IoU with a staircase of widths and emits no sketch at all. A
  docstring had already forbidden it; eleven tests caught the violation.
- **The integration silently dropped a component.** Turning the selector on bypassed the outline
  view model, which showed up as a sketch loss on parts whose mode had not changed.
- **The bench scored models on their own training data.** Both depth heads are trained on all 1908
  photographed parts. Served in-sample the silhouette GBM recites - median x1.16 against its own
  out-of-fold x1.67 - and beats a head that generalises better. This one reversed a revert.

The rule that follows: a learned component is compared only on parts excluded from its training
set, using the held-out model files and id lists in `data/`. A full-set number for a learned
component measures memory, not skill.

## Where the remaining error actually lives

Measured on runs/final, worst quartile 467 parts averaging 0.213:

| cause | worst quartile | all parts |
|---|---|---|
| bad sketch, region IoU under 0.4 | 57% | 30% |
| mode choice costs more than 0.10 | 27% | 18% |
| depth off by more than 1.5x | 23% | 16% |
| misses two or more holes | 4% | 3% |
| no cause identified | 20% | |

The bad sketch dominates, so it was split twice more. Of the parts whose sketch scores
under 0.4, **86% have no good view at all** - all three photographs score under 0.4 - and
only 12% could be rescued by tracing a different one. Taking those parts back to the raw
contour before any regularisation: **89% are still under 0.4 unregularised**, and only 6%
are cases where regularisation destroys a good trace.

So the worst quartile is a capture problem. Not mode selection, not depth, not the
tracing code, and not view choice among the three photographs we have. The shape is not
in the photographs. That is the same wall the tilt work hit from the other side, and the
board is the answer to both.

## Verdicts, measured

| hypothesis | outcome |
|---|---|
| mode selection is worth chasing | **confirmed.** Oracle gap +0.094 [+0.070, +0.120]; routing the stations fallback to plan captured +0.048 offline on 1860 parts |
| a learned selector beats the rules | **confirmed, once powered.** +0.016 [+0.008, +0.024] against the strong "always plan" baseline on 949 held-out parts with folds grouped by geometry. The earlier "inconclusive" verdict came from 99 parts, not from the method |
| which photo we trace matters | **confirmed offline, not yet established end to end.** The pipeline consults the ranker on only 20% of parts - `profile` mode is hard-wired to the least rectangular view and `revolve` to the roundest, so only `plan` mode asks. On the 115 parts where the choice actually changed the paired gain is +0.022 [-0.003, +0.048]; on the other 477 it is exactly zero, and the whole-set delta is +0.004 [-0.001, +0.009]. The default stays on the old rule until the 1908-part run either resolves it or does not. |
| the oracle headline was honest | **refuted.** The +0.082 oracle gap is mostly luck: on 93 parts whose three photos are geometrically interchangeable an oracle still "gains" +0.045 [+0.029, +0.063], which is selection on scoring noise, not viewpoint. The real viewpoint effect is about +0.037, and the ranker takes +0.031 of it - roughly four fifths of what is actually there, not the 38% I first claimed |
| a cheap rule would do instead | **refuted.** Most rectangular is harmful (-0.048), largest area is neutral, least elongated gives +0.022 - a quarter of what the ranker takes |
| the model can predict its own reliability | **refuted.** Features give AUC 0.633, a direct failure classifier 0.622, and re-projecting the model onto its own photos 0.581. Only a low-recall warning is defensible: 60% precision at 21% recall |
| duplicates were inflating our numbers | **refuted.** 8.6% of parts have a geometric twin; grouped splits change the learned-selector gain by 0.003 |
| we overfitted to PrintCAD's photo style | **no evidence.** T-LESS through the same pipeline scores 0.428 [0.367, 0.489] against 0.462 on PrintCAD |

## The noise floor

Two photos of the same part that a silhouette cannot tell apart still score a median 0.039 and mean
0.067 IoU apart. That is the instrument's own scatter, and it sets three limits worth remembering
before quoting any per-view number:

- **Any max-over-views figure is inflated.** `tools/oracle_noise.py` measures the inflation directly,
  on parts where the views are interchangeable and the true gain must be zero.
- **A single part proves nothing.** A 0.04 difference between two runs on one part is the noise, not
  the change.
- **Gains below ~0.01 need the full set.** This is the same lesson as the 200-part instrument, one
  level down: per-view scores are noisier than per-part scores.

## Not doing, and why

- **More threshold tuning.** Measured at +0.007, inside the noise floor.
- **Training our own image→CAD generator.** cadrille is released, state of the art, and beat our
  fine-tune on every real part; it also emits code, not sketches.
- **Recovering perspective from a single uncalibrated photo.** Zhang–He on silhouette corners is
  ill-conditioned at phone distances — one part's three photos implied aspect ratios of 0.63, 0.22
  and 1.34. Use the target.

## Housekeeping

- Someone else is committing to this repo. Agree ownership of `modes.py` and `spec.py` before either
  of us edits them, or land changes through branches.
- `photo2fcstd-doctor` is the entry point on a new machine; keep it accurate as dependencies move.
