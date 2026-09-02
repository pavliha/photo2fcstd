# Work items

Every gap broken into something a person could pick up in one sitting: what it is, the measurement
behind it, and what "done" means. `n=` throughout; where something is unmeasured it says so instead
of estimating.

**Blocked by** is what matters for planning. `code` means nobody else is needed. `photos` means it
waits on a capture session. `person` means it waits on a judgement nobody has made.

## Where the drawing actually stands

Built end to end through FreeCAD (n=59): every document builds, 93% yield a real solid, 97% of
sketches solve, **none has a free degree of freedom**, and there are 38.8 constraints per 12.7
geometry - genuinely constrained, not fixed points. On the discriminating parts: region IoU 0.659,
structure 0.742, loops 0.999, curves 0.568, elements 0.659, 27% primitive-exact.

Holes are solved. What remains is arcs, and four diagnostics this session narrowed it a long way.

## What the arc investigation established

**The tracer does not fail at arcs. It saturates.** Pooled over 416 parts, given a *perfect*
rasterised face so nothing is capture:

| real curves | n | drawn | recovered | structure |
|---|---|---|---|---|
| 0 | 134 | 0.08 | - | 0.815 |
| 1 | 80 | 0.99 | **99%** | 0.842 |
| 2-3 | 107 | 1.88 | **89%** | 0.852 |
| 4-7 | 52 | 2.21 | 45% | 0.654 |
| 8-15 | 24 | 3.33 | 33% | 0.635 |
| >=16 | 19 | 6.37 | **23%** | 0.544 |

It finds essentially every curve on 321 of 416 parts and invents 0.08 false ones where there are
none. The defect is confined to the ~23% of parts with four or more curves.

Four things this rules out, so nobody re-runs them:

- **Capture is not the cause.** Perfect input recovers 1.76 curves against a photograph's 1.62.
- **Simplification is not the cause.** A 16x sweep of `trace.RUN_EPS` moves curve count by 0.17
  against a deficit of 3.5; it controls fragmentation (elements 17.4 to 11.9), not primitive type.
  Nothing beat shipped.
- **The arcs reach the decision intact.** Of 106 lost arcs, 29% collapse to one line; **71% survive
  as two or more pieces** and are refused by the fitter.
- **A lower gate cannot work.** Only 3% of real lines come out curved, and that precision is what
  the earlier loosening destroyed - it fired on 54% of parts with no curve at all. Sweep and
  sagitta are exactly the quantities that cannot separate a shallow arc from a straight edge.

**And the prize is modest.** If every complex non-b-spline part reached the simple group's level,
structure goes 0.756 to 0.798 on the trusted ceiling set - **+0.043**. For scale, `view_model`
shipped at +0.038 IoU. Worth one good attempt, not a campaign.

---

| # | item | blocked by | size |
|---|---|---|---|
| ~~A6~~ | ~~Arc chord gate relative to the run~~ - **measured, reverted** | - | - |
| ~~A8~~ | ~~Merge runs before fitting~~ - **measured, reverted** | - | - |
| **A7** | **Give the fitter evidence a straight line lacks** | code | large |
| A3 | Arc fitting before simplification - *demoted, addresses 29%* | code | medium |
| ~~B1~~ | ~~Two sketches that fail to solve~~ - **00011 fixed, 00039 remains** | - | - |
| ~~B2~~ | ~~Solids with volume that `isValid()` rejects~~ - **00061 fixed, 00086 is nested holes** | - | - |
| ~~B3~~ | ~~Build validity in the regression test~~ - **done** | - | - |
| ~~B4~~ | ~~One redundant constraint~~ - **gone with B1** | - | - |
| ~~D1~~ | ~~Verify the depth note~~ - **done, and the band is constant** | - | - |
| D2 | Refuse a solid when the depth band is uninformative | code | small |
| ~~D3~~ | ~~Use a second view's depth~~ - **already done, and edge-on does not help** | - | - |
| ~~F1~~ | ~~Audit the gates~~ - **done, neither distance gate is backwards** | - | - |
| ~~F2~~ | ~~Decide whether `stations` stays~~ - **kept, with a measured reason** | - | - |
| **C5** | **Shoot ~16 photos of one part on a patterned surface** | photos | small |
| C1 | Shoot 30-50 board frames for tilt labels | photos | small |
| C2 | Fit and gate the tilt head on them | code (after C1) | small |
| ~~C3~~ | ~~Run carve on a real capture~~ - **already done on T-LESS, 0.782 IoU** | - | - |
| C4 | Decide the no-board, no-scale case | code | small |
| E1 | Ask a person which drawing they would rather edit | person | small |
| E2 | Re-weight `structure_score` from the answers | code (after E1) | small |

Done: **A1** (which arcs are lost), **A2** (not simplification), **A4** (b-splines), **A5**
(saturation, not trust).

---

## A. Arcs

**A6. MEASURED AND REVERTED.** `trace.py:696` gated on `chord > ARC_MIN_CHORD_FRAC * length_px`, a
fraction of the whole part, so the hypothesis was that on complex parts every arc is too small to be
considered. The mechanism is real and far too small to matter.

On perfect input, n=418 (124 parts with four or more curves, 122 with none):

| arm | curves on complex parts | recovered | false curves/part | structure |
|---|---|---|---|---|
| shipped | 3.49 | 26% | 0.04 | 0.797 |
| frac/2 | 3.98 | 29% | 0.04 | +0.0022 [+0.0005, +0.0047] |
| **abs 8 px** | 4.15 | 31% | **0.04** | **+0.0030 [+0.0010, +0.0057]** |
| abs 5 px | 4.25 | 31% | 0.06 | +0.0024 [-0.0003, +0.0055] |
| really there | 13.57 | | | |

Removing the global gate entirely reaches 31%, not 90%: **the chord gate explains about 5 of the
65 missing points.** Precision holds, unlike the earlier loosening.

On real photographs, n=133 discriminating, specs regenerated for both arms and built:

| arm | IoU | structure | curves | valid solid | sketches unsolved |
|---|---|---|---|---|---|
| shipped | 0.603 | 0.695 | 1.96 | 125 (94%) | 3 |
| abs 8 px | 0.603 | 0.690 | 2.37 | **123 (92%)** | **6** |

It draws more curves and **doubles the unsolved sketches**, for -0.0045 structure [-0.0135, +0.0025]
and no IoU change. Reverted; `ARC_MIN_CHORD_PX` removed rather than left as a dead knob. This is the
documented pattern - a change that improves a statistic and breaks models.

**The next hypothesis, from A1 and this result together.** Of lost arcs, 71% survive as two or more
pieces. An arc split across three runs presents each run with a third of its sweep, so a 60 degree
arc arrives as three 20 degree runs and every one fails `ARC_MIN_SPAN_DEG = 40`. That would explain
why chord size is not the limit and why recovery tracks complexity - more corners, more splits.
Note run merging was tried and reverted once already (-0.013 IoU), but that merged *fitted arcs*;
this would merge runs *before* fitting. File as A8 and measure the split count against arc sweep
first.

**A8. MEASURED AND REVERTED.** A census of 3,384 contour runs on perfect input says where arcs
actually die:

| outcome | runs | share |
|---|---|---|
| **sweep under 40 deg** | 1425 | **42%** (64% of all rejections) |
| accepted as an arc | 1153 | 34% |
| circle fit too loose | 769 | 23% |
| too flat (sagitta) | 35 | 1% |
| chord too short for the part | **1** | **0%** |

That last row retro-explains A6: the chord gate fires on one outer-contour run in 3,384, so its
small measured effect must have come almost entirely from hole contours, where a chord is short
relative to the whole part.

Since sweep is the killer and 29% of lost arcs are split into pieces that each fall under it,
`merge_runs` joined consecutive runs when the union passed the **same** gates whole - loosening
nothing, just asking the question of the right span of contour. On perfect input, n=407:

| arm | curves | recovered | false/part | structure |
|---|---|---|---|---|
| shipped | 3.50 | 26% | 0.04 | 0.802 |
| merge 3 | 4.61 | **34%** | 0.15 | **+0.0132 [+0.0037, +0.0226]** |

Four times A6's effect. On real photographs, n=132, both arms regenerated and built:

| arm | IoU | structure | curves | valid solid | unsolved | build time |
|---|---|---|---|---|---|---|
| shipped | 0.605 | 0.696 | 1.95 | 124 (94%) | 3 | ~2 min |
| merge 3 | 0.592 | 0.713 | 2.61 | **121 (92%)** | **6** | **~1 hour** |

Structure +0.0170 [-0.0051, +0.0396] and IoU -0.0123 [-0.0314, +0.0090], neither significant and
pointing opposite ways. It loses three working solids, doubles the unsolved sketches, and makes
FreeCAD roughly **fifty times slower** - merged arcs produce geometry the solver labours over.
Reverted.

**The pattern across A6 and A8 is the finding.** Both improved perfect-input metrics with tight
confidence intervals, and both broke models on real photographs. A gain measured on rasterised
faces did not survive contact with a photograph either time. Any future arc work must carry a build
check, and perfect-input numbers should be treated as a *screen*, never as evidence to ship on.

**A7. Give the fitter evidence a straight line does not have.** If A6 fails or falls short, this is
what is left, and it follows from the ruled-out list above: the discriminant has to be something
other than local sweep and sagitta. Candidates - neighbouring geometry, symmetry between paired
fillets, continuity with an adjacent arc, fitting over a longer run than the corner split allows.
**Done when** one is measured. **Read the prior first**: twelve learned attempts, two wins, and
every loss replaced a geometric step. A model that *ranks candidate arc fits the geometry already
produced* is the winning shape; one that replaces `approxPolyDP` or the fitter is the losing one.

**A3. Arc fitting before polygon simplification** - *demoted by A2*. The premise was that
simplification commits to straight segments first. A2 shows 71% of lost arcs survive simplification
and are refused afterwards, so reordering addresses at most the 29% minority. **Done when**
measured, but expect little.

**A4/A5 policy, now settled**: quote the arc ceiling **excluding b-spline parts**. They are 15% of
the set, 94% of their curves are b-splines, and they score **0% exact** because there is no
b-spline primitive to emit. Excluding them the deficit is 1.66 against 3.83, not 1.76 against 5.19.
Adding a b-spline primitive is a separate and larger question nobody has asked for.

## B. Builds

n=59 through FreeCAD. The four failures are two different faults and should not be reported as one
number:

**Diagnosis so far, shared by B1 and B2.** Three of the four have self-intersecting or overlapping
traced loops: 00011 (two self-intersections, solver returns -2), 00086 (two self-intersections plus
two overlapping pairs), 00061 (one overlapping pair). 00039 has no geometric fault and is separate.
But 22% of all specs have such a fault and **10 of 13 still build a valid solid**, so refusing them
would sacrifice ten working parts to fix three - a guard is the wrong shape of fix.

Splitting further: 00011 is valid when every arc is replaced by its chord, so its arcs cross;
00086 is invalid even as chords, so its *polyline* self-intersects. Two different faults again.

**A genuine invariant violation was found and fixed on the way, and it explains none of them.**
`arc_from_run` projects each run's endpoints onto its own fitted circle, so a corner shared by two
arcs became two different points; regularisation keeps one, leaving the other arc's endpoint off its
own circle. **43% of emitted arcs were out by more than 1% of the radius, the worst by 21%** -
FreeCAD was being handed a centre, a radius and two endpoints that disagreed. `trace.reconcile_arcs`
moves the centre onto the chord's perpendicular bisector, so both endpoints lie on the circle
exactly and no endpoint moves.

Measured on 133 discriminating parts with both arms regenerated and built: region IoU **-0.0001
[-0.0008, +0.0005]**, structure **+0.0000** (an arc stays an arc, so primitive counts cannot move),
valid solids 125 either way, unsolved sketches 3 either way. **And all four faults persist
unchanged.** Kept as a correctness invariant, not as an improvement - it buys nothing measurable
and costs nothing measurable.

**The remaining cause is the polyline, not the arcs.** That is where B1 and B2 should resume:
regularisation moves points across one another, and no arc-level fix reaches it.

**B2. PARTLY DONE - 00061 fixed, and it was a third fault again.** Neither the arc invariant nor
self-intersection was the cause. `00061` had a 204 px hole lying **entirely across the outer
boundary** (100% of its area shared with the outer loop) and `00086` two holes crossing it at 14%
and 4%. A hole that straddles the boundary makes the padded face invalid, which is exactly how a
part reaches a build report as a solid with real volume that `isValid()` rejects.

`trace.drop_stray_holes` removes any loop not contained in the outer one, and `trace.keep_simple`
falls back to the unregularised elements when regularisation folds a loop through itself. Measured
on 133 discriminating parts, both arms regenerated and built:

| arm | valid solid | unsolved | IoU | structure | curves | elements t |
|---|---|---|---|---|---|---|
| before | 125 (94%) | 3 | 0.603 | 0.695 | 1.96 | 0.610 |
| **guards** | **129 (97%)** | 3 | 0.604 | 0.688 | 1.83 | 0.630 |

**Four more parts produce a usable solid.** IoU +0.0015 [-0.0003, +0.0037] and structure -0.0062
[-0.0174, +0.0040], both within noise; curves fall 0.13 per sketch because some dropped holes
carried arcs, and those holes were invalid anyway. **Shipped** - it is the first change this session
to move the number the product is actually judged on.

`00086` still fails, and not from a bug: its loops are **nested** - a circle hole containing two
smaller holes. The flat loop model has no way to say "material inside a hole", so that is a
representation limit, not a defect to fix here.

**B1. PARTLY DONE - 00011 fixed.** Its sketch carried `DistanceX` on 11 vertices and `DistanceY` on
11, pinning every coordinate, **and** Horizontal/Vertical on lines whose endpoints were already
pinned. FreeCAD flagged constraint 37, a `Vertical`, as redundant and returned -2. The builder
already skipped Horizontal/Vertical next to tangent joins; it now also skips them when both
endpoints of the line are coordinate-pinned in the relevant axis, which is the case that was
missed.

End to end on the same 59 parts as the recorded baseline:

| | before | after |
|---|---|---|
| documents built | 59/59 | 59/59 |
| **with a real solid** | 55 (93%) | **57 (97%)** |
| **sketches that solve** | 59/61 | **60/61** |
| free degrees of freedom | 0 | 0 |
| region IoU / structure / exact | 0.659 / 0.742 / 27% | 0.658 / 0.738 / 27% |

Quality unchanged, two more parts usable. Combined with B2 the sample goes 93% to 97% valid.

**00039 still returns -5** with nothing flagged redundant or conflicting: 6 lines and 4 arcs, 33
constraints against 44 degrees of freedom by hand count, so it is exactly determined and the solver
still fails - most likely singular rather than over-constrained. Not diagnosed.

**A correction to an earlier claim.** This repository's sketches were described as "genuinely
constrained, not fixed points" on the strength of 38.8 constraints per 12.7 geometry. That ratio is
high *because* most vertices carry both a DistanceX and a DistanceY from the origin: the sketches
are dimension-driven but coordinate-pinned, which is closer to fixed points than that phrasing
suggested. They are fully constrained and they are editable through the spreadsheet, but a person
would not call this a well-constrained sketch.

 00011 and 00039; the sketch solver returns -2 and -5.
**Done when** the cause is named and either fixed or refused loudly, as `traced_outline` already
does for a collapsed outline.

**B2. The two solids that are invalid.** 00061 and 00086 produce a solid *with volume* that
`isValid()` rejects. This is the more concerning kind: "valid" in a report means non-null with
volume, so these pass the repo's own check while being wrong. **Done when** the geometry fault is
identified and the report stops calling them valid.

**B3. DONE.** A build test already existed over three parts. It now also covers **00011 and 00061**,
the two fixed this session, so either regression turns the suite red. Added
`test_a_sample_of_parts_still_builds`: ten parts through one FreeCADCmd process, asserting every one
yields a real solid and no sketch keeps a free degree of freedom.

**The floor is 100%, not 90%.** At 90% a single regression in ten parts sits exactly on the line and
passes - which is what the first version of this test did, found only by running it with
`P2F_DROP_STRAY_HOLES=0` to check it could fail at all. Both directions are verified: green
normally, red with the guard disabled. Costs about 50 seconds; the suite is 269 tests in 171 s.

**B4. DONE - it was the same fault as B1.** Zero redundant and zero conflicting constraints across
61 sketches now, where there was one before. It was a Horizontal/Vertical duplicating a coordinate
pin, the same class the B1 fix removed.

## C. Needs photographs

**C5. Sixteen photographs on a patterned surface - the highest-value capture available.**
Structure-from-motion recovers camera poses from the scene, so carving no longer needs a printed
board. Validated synthetically in `docs/sfm.md`: 24 of 24 images registered, camera rotation error
**0.031 deg** against a 2 deg budget, and carving from SfM poses agrees with carving from true poses
at **0.987** (0.949-0.997 across parts).

That unblocks the largest measured gain in the project: **0.782 solid IoU against the photo path's
~0.43**.

Two requirements, both found by trying to break it: the surface must be strongly textured - at a
quarter of the contrast, **0 of 16 images registered on every part** - and sixteen views are needed,
since eight gave only five. Scale still needs one caliper reading, as it already does.

**Done when** sixteen photographs of one part on a newspaper or cutting mat have been carved and
scored.

**C1 can now be fed from public data instead.** BOP-Industrial - IPD (10 objects, RGB-D, 13
cameras), XYZ-IBD (15 objects, 273k real samples) - is real photographs of machined parts with exact
poses, in BOP format that `bop.py` already reads, so tilt labels no longer have to come from your
camera. Caveats in `docs/datasets.md`: these are cluttered bin-picking scenes needing a visibility
filter, and ITODD is grayscale, which is the wrong sensor for a head that reads shading. Your own
photographs remain the only way to test whether the *gate* admits them.

**C1 (original). Shoot 30-50 board frames.** The tilt head is good on real photographs (2.63 deg against a
6.98 constant, 87% within 5, over 0-30 deg) and useless trained on renders (7.6-8.0 against 8.81;
two renderers, the better one slightly worse). Recipe and assumptions in `docs/tilt-labels.md`;
`tools/tilt_board_data.py` turns the capture into labels. **Done when** `data/tilt_board.npz` spans
0-30 degrees.

**C2. Fit and gate the head.** `python tools/tilt_train.py data/tilt_board.npz`. **Done when**
held-out-by-part error beats the constant and the gate admits ordinary photographs - at which point
`square_check` stops abstaining and the +0.13 from shooting square becomes reachable board-free.

**C3. ALREADY DONE - and this entry was wrong.** Carving has run on real photographs since before
this session, and `docs/results.md` records it: T-LESS ships what the ChArUco board was for - 30
objects, 1,296 views each, `cam_K` and `cam_R`/`cam_t` per view, masks and CAD - so
`bop.carve_object` needs no capture session at all. On 30 objects, 22 views, 0.8 mm voxels, scored
at 1.5 mm against CAD: **0.719 mean IoU from silhouettes, 0.782 with depth free space**, against the
photo pipeline's ~0.43. The synthetic estimate of 0.736 was fair rather than flattering.

A dataset closed this, not a rig. See `docs/datasets.md`.

**C4. Decide the no-board, no-scale case.** Without a board or a known length the sketch is
dimensionless. **Done when** the behaviour is deliberate rather than incidental.

## D. Depth

0.435 median absolute log error against 1.045 for the best constant, 63% within 2x - but the
**median 80% band spans about 10x**. So "a sketch you can edit" is largely delivered; "a model you
can build from" is not. PrintCAD-specific: 0.348 on T-LESS against 0.314 for a constant there.

**D1. DONE, and it turned up something worse than a wording problem.** The note is honest in form -
47 of 54 specs carry, for example, *"depth predicted from the silhouettes: 57.5 px, 80% of the time
between 32.1 and 103.0 (3.2x spread, too wide to trust, put a caliper on it)"*, which states the
estimate, the coverage, the interval, the spread and a verdict.

But **46 of those 47 parts report exactly 3.20x**, because `depth_model.pixel_predict` - the shipped
path - has no quantile heads. It returns `exp(point ± offset)`, a fixed +/-0.576 in log space, so
every part gets the same multiplier. The interval is conformally calibrated for *marginal* coverage,
so "80% of the time" is honest on average, but it carries **no per-part information at all**: it
cannot tell you which parts are uncertain, which is exactly what a reader assumes it is for. The
tabular fallback does have real per-part quantiles (5x to 145x across four sampled parts), which is
why the one gate-refused part reads 40.2x.

Second consequence: the note's own threshold for "good enough to build from" is a spread under 2.0,
and 3.20 exceeds it. **The pixel path can never say a depth is usable** - 0 of 47 parts did.

**D1 is closed; the finding becomes D4.**

**D2.** Refuse when the band is uninformative. Partly moot: the note already says "too wide to
trust" and does so for 100% of parts on the shipped path. The open question is whether the pad
should still be built from a number the pipeline itself calls untrustworthy. **Done when** that is a
deliberate decision rather than the default.

**D4. DONE - real quantile heads measured and rejected; the phrasing is fixed instead.**

Fitting proper quantile heads on the same 1,437 embeddings, conformalised the CQR way, held out by
group:

| band | coverage | median width | p10 | p90 |
|---|---|---|---|---|
| shipped, constant | 80% | **3.20x** | 3.20x | 3.20x |
| quantile heads + CQR | 80% | 6.05x | 3.01x | 13.48x |

It does vary per part - 4.5x between p10 and p90 - but the median band **nearly doubles** at the
same coverage, and its width correlates only **0.22** with the actual error, the same figure that
made region IoU "nearly independent" of drawing quality. It does not beat its control, so it is not
shipped.

The phrasing is fixed instead. `depth_model.predict` now returns a fifth value saying whether the
band is this part's or a fixed calibration, and the note reads *"80% of parts land within 3.2x of
this - a fixed calibration, not this part's own uncertainty"* on the pixel path, keeping the
per-part wording only for the tabular model, which really does predict per-part quantiles.

Original statement of the item: A constant 3.20x band dressed as
a per-part prediction is worse than no band, because it invites a reader to compare parts by it.
Either fit quantile heads on the embedding as the tabular model does, or report the point estimate
with a single global caveat. **Done when** either the band varies per part with coverage checked on
held-out data, or the per-part phrasing is removed.

**D3. DONE - the premise was already satisfied, and the extra signal is not there.** Both heads
already read three views: the pixel head's 3,072 dims are three 1,024-vectors, and the tabular
features are computed over all the events. So "use a second view" was not open.

The real question was whether an *edge-on* view carries depth the model fails to use. Over 1,437
parts held out by group, against the thinnest silhouette in each capture:

| thinnest view | n | median abs log error | within 2x |
|---|---|---|---|
| 0.00-0.15 (most edge-on) | 59 | 0.329 | 88% |
| 0.15-0.30 | 175 | 0.377 | 86% |
| 0.30-0.50 | 372 | 0.324 | 81% |
| 0.50-0.70 | 419 | 0.289 | 83% |
| 0.70-1.01 (all face-on) | 412 | **0.214** | **93%** |

Captures **with** an edge-on view score 0.437 [0.385, 0.495] against 0.384 [0.362, 0.408] without -
worse, not better, and the trend runs monotonically the opposite way to the hypothesis. Correlation
between thinness and error is -0.13.

**This is correlational, not causal**: having an edge-on view is confounded with being a thin or
awkward part, which is harder anyway. Settling it properly needs the same part shot with and
without, which nobody has. But there is no evidence here to justify weighting an edge-on view, and
the hand-written estimator that tried lost to a constant.

## E. The objective itself

**E1. Ask a person.** `structure_score` weights `loops`, `curves` and `elements` **equally** and
that is arbitrary. Region IoU correlates 0.22 with structural agreement and 0.33 with exact
primitives, which is why the structural criterion exists - but it has never been checked against a
human. Every quality number here is a proxy its authors chose. **Done when** someone has seen twenty
pairs and said which they would rather edit.

**E2.** Re-weight from the answers, and re-judge the shipped and reverted changes under them.

## F. Robustness and tidying

**F1. DONE - neither distance gate is backwards.** A gate is sound when the quantity it thresholds
rises with the error it exists to catch. Measured as the correlation between gate score and
held-out error:

| gate | correlation | n | verdict |
|---|---|---|---|
| depth: `embedding_is_familiar` | +0.04 | 1,437 parts | flat, not backwards |
| tilt: `is_familiar` | +0.04 | 3,132 views | flat, not backwards |

Flat is the right answer for these two. They are cosine distance from the training centre, built to
detect being *outside* the distribution, and inside it there is nothing to rank - which is what +0.04
says. Neither shows the `axis_model` failure, whose own score runs backwards out of distribution
(0.73 when wrong against 0.56 when right) and which is already gated on geometry
(`section_constancy`) instead of on itself.

The audit tool was wrong before the gates were: it compared a 3,072-dim embedding against a
1,024-dim centre and reported "not measured". `embedding_is_familiar` reshapes into three views and
takes the median per-view distance, which is correct; `tools/gate_audit.py` now does the same.

**F2. DONE - kept, and it is not dead code.** Over 320 trusted parts with photos, 317 assemble
(plan 247, profile 63, revolve 7) and **3 raise** *"every view of this part traces to an outline with
no area"*: 00076, 00135, 00332. Those are the parts `stations` exists for, and forced onto them it
produces a valid solid for all three - volumes 217,980, 472,920 and 1,327,238, one solid each.

So it is a working capability covering 0.9% of parts, not dead code.

**It is deliberately not wired as the automatic fallback.** A `stations` document draws no sketch,
and the sketch is the product; emitting one would hand back a document that looks fine and contains
no drawing, and would swallow the "reshoot it square to the face" message that those three parts
currently get. That refusal was a deliberate decision recorded in CLAUDE.md, and reversing it is a
product call, not a measurement.

The option is now costed, should coverage ever be preferred to refusal: 3 parts in 320 gain a solid
and lose their reshoot advice.

## Ground truth, and the eight-fold option

Every number here is quoted on the ~55% of PrintCAD whose reference face can be inferred from its
STEP file. The [Fusion 360 Gallery Reconstruction dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset)
is now **built and verified**: 3,120 records passing `trustworthy()` against PrintCAD's 1,047, a
**3.0x** multiplier rather than the eight-fold a first reading suggested - half the designs are
multi-extrude timelines this pipeline does not model. Each record is checked against the dataset's
own mesh (median area x depth / volume of 1.000, 84% within 5%) rather than accepted on a heuristic,
and 94% pass where PrintCAD manages 55%. It carries **no b-splines**, so `exact` is reachable on
every record. It has no photographs, so it improves what a drawing is scored against rather than
what it is made from. `tools/fusion360.py`, and see `docs/datasets.md`.

## Closed - do not re-open

- **Tilt cannot be corrected after the fact.** At 15 degrees projective distortion costs -0.009
  [-0.026, +0.008]; side walls coming into view cost -0.084 [-0.114, -0.058]; rectifying by the
  *true* normal scores -0.013 (n=115).
- **Mode selection is finished.** A perfect oracle is worth +0.0003 region IoU [-0.022, +0.023].
- **Coverage is 100%.** `stations` is never chosen; the old "46% draw nothing" tradeoff predates the
  mode allowed-list fix.
- **Two independently-registered silhouettes cannot be intersected.** 0.332 against 0.433 from photo
  masks even at the best of 48 poses; 0.823 with truth silhouettes, so the loss is registration.
- **Renders cannot train a tilt head.** Two renderers, the better one slightly worse.
- **Ground truth covers about 55% of the dataset**, and the untrusted remainder is *simpler*, not
  harder - so the trusted subset is not a flattering sample for arcs.
