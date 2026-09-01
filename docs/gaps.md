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
| B1 | Two sketches that fail to solve | code | small |
| B2 | Two solids with volume that `isValid()` rejects | code | medium |
| B3 | Build validity in the regression test | code | small |
| B4 | One redundant constraint | code | small |
| D1 | Verify the `params` note states the depth band honestly | code | small |
| D2 | Refuse a solid when the depth band is uninformative | code | small |
| D3 | Use a second view's depth | code | medium |
| F1 | Audit every gate for backwards confidence | code | medium |
| F2 | Decide whether `stations` stays | code | small |
| C1 | Shoot 30-50 board frames for tilt labels | photos | small |
| C2 | Fit and gate the tilt head on them | code (after C1) | small |
| C3 | Run `carve.py` on a real capture | photos | medium |
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

**B1. The two that produce nothing.** 00011 and 00039; the sketch solver returns -2 and -5.
**Done when** the cause is named and either fixed or refused loudly, as `traced_outline` already
does for a collapsed outline.

**B2. The two solids that are invalid.** 00061 and 00086 produce a solid *with volume* that
`isValid()` rejects. This is the more concerning kind: "valid" in a report means non-null with
volume, so these pass the repo's own check while being wrong. **Done when** the geometry fault is
identified and the report stops calling them valid.

**B3. Build validity in the regression test.** The suite covers spec generation; nothing goes red
when a spec stops building. **Done when** a small sample builds through FreeCAD in CI and a null
solid fails the suite.

**B4. The redundant constraint.** One across 61 sketches. **Done when** traced to what emits it.

## C. Needs photographs

**C1. Shoot 30-50 board frames.** The tilt head is good on real photographs (2.63 deg against a
6.98 constant, 87% within 5, over 0-30 deg) and useless trained on renders (7.6-8.0 against 8.81;
two renderers, the better one slightly worse). Recipe and assumptions in `docs/tilt-labels.md`;
`tools/tilt_board_data.py` turns the capture into labels. **Done when** `data/tilt_board.npz` spans
0-30 degrees.

**C2. Fit and gate the head.** `python tools/tilt_train.py data/tilt_board.npz`. **Done when**
held-out-by-part error beats the constant and the gate admits ordinary photographs - at which point
`square_check` stops abstaining and the +0.13 from shooting square becomes reachable board-free.

**C3. Run `carve.py` on a real capture.** Carving reaches 0.736 volumetric IoU against truth (0.791
on parts at least four voxels thick, n=30) and an expected ~0.715 sketch IoU against the photo
path's 0.602, with depth *measured*. It has only ever run on synthetic views. **Done when** one real
capture is carved and scored, and the synthetic estimate is confirmed or corrected.

**C4. Decide the no-board, no-scale case.** Without a board or a known length the sketch is
dimensionless. **Done when** the behaviour is deliberate rather than incidental.

## D. Depth

0.435 median absolute log error against 1.045 for the best constant, 63% within 2x - but the
**median 80% band spans about 10x**. So "a sketch you can edit" is largely delivered; "a model you
can build from" is not. PrintCAD-specific: 0.348 on T-LESS against 0.314 for a constant there.

**D1.** Verify the `params` note states the range and says whether the number is worth building
from. **Done when** read on a real output rather than assumed.

**D2.** Refuse when the band is uninformative - a 10x band is not a dimension. **Done when** a
threshold exists, chosen by measurement.

**D3.** Use a second view. Depth is not in one face-on photograph, but a capture has three. **Done
when** it is known whether an edge-on view narrows the band. The hand-written edge-on estimator lost
to a constant, which is not the same test.

## E. The objective itself

**E1. Ask a person.** `structure_score` weights `loops`, `curves` and `elements` **equally** and
that is arbitrary. Region IoU correlates 0.22 with structural agreement and 0.33 with exact
primitives, which is why the structural criterion exists - but it has never been checked against a
human. Every quality number here is a proxy its authors chose. **Done when** someone has seen twenty
pairs and said which they would rather edit.

**E2.** Re-weight from the answers, and re-judge the shipped and reverted changes under them.

## F. Robustness and tidying

**F1. Audit the gates.** Nothing learned here survives a change of dataset: `depth_model` 0.435 to
1.117, `axis_model` 89% to 29% against 33% for chance, `tilt_model` 4.8 deg to 34. Each is gated so
it abstains, and the gates work - but `axis_model`'s confidence runs **backwards** out of
distribution, scoring 0.73 when wrong against 0.56 when right. **Done when** every gate has been
checked for that specific failure. A gate reading a broken signal is worse than no gate.

**F2. Decide whether `stations` stays.** Never selected on 197 trusted parts, reachable only when
forced, both enforced by `tests/test_regression.py`. **Done when** deliberately kept with a reason,
or removed.

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
