# Work items

Every gap broken into something a person could pick up in one sitting. Each item says what it is,
the measurement behind it, and what "done" means. `n=` throughout; where something is unmeasured it
says so instead of estimating.

**Blocked by** is the thing that matters for planning: `code` means nobody else is needed,
`photos` means it waits on a capture session, `person` means it waits on a judgement nobody has
made.

| # | item | blocked by | size |
|---|---|---|---|
| ~~A1~~ | ~~Characterise which arcs get split~~ - **done**, see below | - | - |
| A2 | Test whether simplification eps is the binding knob on perfect input | code | small |
| A3 | Try arc fitting before polygon simplification, not after | code | medium |
| A4 | Decide the `bsplinecurve` policy | code | small |
| A5 | Re-run the tracer ceiling on the 45% untrusted parts | code | small |
| B1 | Fix the two sketches that fail to solve | code | small |
| B2 | Fix the two solids with volume that `isValid()` rejects | code | medium |
| B3 | Put build validity in the regression test | code | small |
| B4 | Chase the one redundant constraint | code | small |
| C1 | Shoot 30-50 board frames for tilt labels | photos | small |
| C2 | Fit and gate the tilt head on them | code (after C1) | small |
| C3 | Shoot one carve set and run `carve.py` on real photographs | photos | medium |
| C4 | Decide what happens with no board and no scale | code | small |
| D1 | Verify the `params` note states the depth band honestly | code | small |
| D2 | Refuse to emit a solid when the depth band is uninformative | code | small |
| D3 | Use a second view's depth when the capture has one | code | medium |
| E1 | Ask a person which drawing they would rather edit | person | small |
| E2 | Re-weight `structure_score` from their answers | code (after E1) | small |
| F1 | Audit every gate for the same backwards-confidence failure | code | medium |
| F2 | Decide whether `stations` stays | code | small |

---

## A. The tracer draws arcs as straight pieces

The largest remaining defect and entirely in code. Given a *perfect* drawing of the true face, the
tracer recovers **1.76 curved primitives against a real 5.19** - barely more than the 1.62 it gets
from a photograph (n=180 matched parts, `tools/tracer_ceiling.py`). Element *count* comes out about
right (12.82 against 11.98); the *types* are wrong. Straight elements are already near-correct
(6.22 against 6.78), so `curves` (0.554) and `elements` (0.626) are one defect: ~3.5 missing arcs.

Already closed off, do not repeat: loosening the gates is not selective (fires on 54% of parts with
no curve at all, 57% of parts that have one; structure -0.036 [-0.070, -0.002]); per-point curve
classification lost 0.579 to 0.452; direct primitive prediction lost twice.

**A1. DONE.** `tools/arc_survival.py`, 555 matched elements over 180 parts. Circles survive 98% of
the time, b-splines 54%, arcs **26%**, and only 3% of real lines come out curved. The arc loss is
entirely on shallow ones: **0% below 30 degrees of sweep**, 13% at 30-60, rising to 68% past 240;
and by radius, 62% at 10-25 px falling to 5% past 150 px. The cliff is where `ARC_MIN_SPAN_DEG = 40`
puts it.

This is a precision/recall setting rather than a bug, and it reframes the rest of section A: sweep
and sagitta are exactly the quantities that cannot separate a shallow arc from a straight edge,
which is why loosening the gate fires on 54% of parts that have no curve at all. A fix must bring
evidence a straight line would not have - neighbouring geometry, symmetry, a longer run - not a
lower threshold on the same quantity.

**A2. Test whether simplification eps is the binding knob on perfect input.** An earlier sweep
patched `outline`'s eps rather than `corner_runs`' and all four arms returned an identical 0.639 -
an unwired knob. Redo it on the perfect-input harness, where capture noise is absent. **Done when**
the arms differ, or it is confirmed that eps is not what limits arc recovery.

**A3. Try arc fitting before polygon simplification.** `approxPolyDP` commits to straight segments
first and arcs are fitted to what survives, which is a plausible mechanism for an arc arriving as
chords. **Done when** measured on the same 180 parts with `verdict`, and shipped or reverted. Note
this modifies the geometric prior that beat twelve learned attempts - a null is the likely outcome.

**A4. Decide the `bsplinecurve` policy.** 328 of 2,163 ground-truth elements are b-splines; the
pipeline has no primitive for them and the scorer counts them as curves, so part of the "missing
3.5 arcs" may be unreachable by construction. Forcing them into "arc" cost 8 points when tried.
**Done when** the ceiling in A is restated separately for parts with and without b-splines.

**A5. Re-run the tracer ceiling on the untrusted 45%.** Everything is measured on the trusted
subset. **Done when** it is known whether the arc deficit is the same there or an artefact of which
parts `trustworthy()` admits.

## B. Builds

59 of 59 documents built, 55 (93%) with a real solid, 59 of 61 sketches solve, **0 with free
degrees of freedom**, 38.8 constraints per 12.7 geometry - genuinely constrained, not fixed points
(`tools/goal_check.py`, n=59).

**B1. The two that produce nothing.** 00011 and 00039; the sketch solver returns -2 and -5.
**Done when** the cause is named and either fixed or refused loudly, like `traced_outline` does for
a collapsed outline.

**B2. The two solids that are invalid.** 00061 and 00086 produce a solid *with* volume that
`isValid()` rejects - the more concerning kind, because "valid" in a report means non-null with
volume and these pass that. **Done when** the geometry fault is identified and the report stops
calling them valid.

**B3. Build validity in the regression test.** The suite covers spec generation; nothing fails when
a spec stops building. **Done when** a small sample builds through FreeCAD in CI and a null solid
turns the suite red.

**B4. The redundant constraint.** One across 61 sketches. Harmless, uninvestigated. **Done when**
it is traced to the constraint that emits it.

## C. Things that need photographs

**C1. Shoot 30-50 board frames.** The tilt head is good on real photographs (2.63 deg against a
6.98 constant, 87% within 5, over 0-30 deg) and useless when trained on renders (7.6-8.0 against
8.81; two renderers tried, the better one slightly worse). Real labelled photographs are the only
missing input. Recipe and assumptions in `docs/tilt-labels.md`; `tools/tilt_board_data.py` turns
the capture into labels. **Done when** `data/tilt_board.npz` exists spanning 0-30 degrees.

**C2. Fit and gate the head.** `python tools/tilt_train.py data/tilt_board.npz`. **Done when**
held-out-by-part error beats the constant and the gate admits ordinary photographs - at which point
`square_check` stops abstaining and the +0.13 from shooting square becomes reachable without a
board.

**C3. Run `carve.py` on a real capture.** Carving reaches 0.736 volumetric IoU against truth (0.791
on parts at least four voxels thick, n=30) and an expected ~0.715 sketch IoU against the photo
path's 0.602, with depth *measured*. It has only ever run on synthetic views. **Done when** one
real capture has been carved and scored, and the synthetic estimate is confirmed or corrected.

**C4. Decide the no-board, no-scale case.** Without a board or a known length the sketch is
dimensionless. **Done when** the behaviour is deliberate - refuse, or emit with an explicit unit
note - rather than whatever happens now.

## D. Depth

`depth_model` reaches 0.435 median absolute log error against 1.045 for the best constant, 63%
within 2x. But the **median 80% band spans about 10x**, and only a couple of percent of parts get a
band tighter than 2x. So "a sketch you can edit" is largely delivered; "a model you can build from"
is not. It is also PrintCAD-specific: 0.348 on T-LESS against 0.314 for a constant there.

**D1. Verify the `params` note.** It is supposed to state the range and say outright whether the
number is worth building from. **Done when** that has been read on a real output rather than
assumed.

**D2. Refuse when the band is uninformative.** A 10x band is not a dimension. **Done when** a
threshold exists above which the depth is emitted as a named guess rather than a number, decided by
measurement rather than taste.

**D3. Use a second view.** Depth is not in one face-on photograph, but a capture has three.
**Done when** it is known whether an edge-on view narrows the band - the hand-written edge-on
estimator lost to a constant, but that is not the same test.

## E. The objective itself

**E1. Ask a person.** `structure_score` weights `loops`, `curves` and `elements` **equally** and
that is arbitrary. Region IoU correlates 0.22 with structural agreement and 0.33 with exact
primitives, which is why the structural criterion exists - but the criterion has never been checked
against a human. Every quality number in this repository is a proxy its authors chose. **Done when**
someone has seen twenty pairs and said which they would rather edit.

**E2. Re-weight from the answers.** **Done when** the weights come from E1 and the shipped and
reverted changes have been re-judged under them.

## F. Robustness and tidying

**F1. Audit the gates.** Nothing learned here survives a change of dataset: `depth_model` 0.435 to
1.117, `axis_model` 89% to 29% against 33% for chance, `tilt_model` 4.8 deg to 34. Each is gated so
it abstains, and the gates work - but `axis_model`'s confidence runs *backwards* out of
distribution, scoring 0.73 when wrong against 0.56 when right. **Done when** every gate has been
checked for that specific failure, since a gate reading a broken signal is worse than none.

**F2. Decide whether `stations` stays.** Never selected on 197 trusted parts, reachable only when
forced, and `tests/test_regression.py` enforces both. Not dead code, but unexercised.
**Done when** it is deliberately kept with a reason, or removed.

## Not a gap, recorded so nobody re-opens it

- **Tilt cannot be corrected after the fact.** At 15 degrees the projective distortion a homography
  removes costs -0.009 [-0.026, +0.008]; the side walls coming into view cost -0.084 [-0.114,
  -0.058]; rectifying by the *true* normal scores -0.013 (n=115).
- **Mode selection is finished.** A perfect oracle is worth +0.0003 region IoU [-0.022, +0.023].
- **Two independently-registered silhouettes cannot be intersected.** 0.332 against 0.433 from
  photo masks even at the best of 48 poses; 0.823 with truth silhouettes, so the loss is
  registration.
- **Ground truth covers about 55% of the dataset** and every number here is on that subset.
