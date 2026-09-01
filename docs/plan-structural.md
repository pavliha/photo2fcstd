# Plan: the four structural problems in the learned layer

Ordered by what is at risk, not by effort. Each step says how it is verified, because three of the
four were only visible through measurement in the first place.

## 1. Make every fallback audible  (~30 minutes, do first)

**Problem.** Fourteen `except Exception: return None` sites across `axis_model`, `view_model`,
`depth_model` and `mode_pixels`. Every one degrades to a geometric rule without a word. This is how
`P2F_VIEW_MODEL=1` set the model path to a file named `"1"`, made `load()` return None, and produced
an A/B that measured its control twice and reported `+0.0000` on all 295 parts.

**Fix.** One shared helper, `photo2fcstd/fallback.py`:

```
note(component, reason)   # warns once per component per process, records into telemetry
```

Call it at every `return None`. Nothing changes behaviourally - the geometric fallback is correct -
but a run that silently lost its models now says so once per component.

**Verify.** A test that removes a model artefact, runs `assemble`, and asserts the warning appears
and the spec still builds. Then re-run the `P2F_VIEW_MODEL=1` scenario and confirm it is visible.

**Do it first** because it is cheap and it protects every measurement taken after it.

## 2. Audit the depth model that actually ships  (~2 hours)

**Problem.** `depth_model.predict` tries the DINOv3 pixel path, then the tabular one. The T-LESS
audit built views from bare masks, `embed.for_views` returned None, and the tabular fallback was
measured. The published "loses to a constant" result describes the fallback. **The path that runs on
a real photograph has never been tested outside PrintCAD.**

**Fix.** `analysis.view_from_mask` gains an optional `source` so a view can carry its image path;
T-LESS views are built from the dataset RGB frame with the dataset mask, so `embed` can reach the
image. Then re-run `tools/depth_ood.py` unchanged.

**Verify.** The same table as the tabular audit, n=21 objects: median absolute log error and
"within 2x" against the best constant fitted on T-LESS. Publish both paths side by side. If the
pixel path also loses to a constant, say so in `CLAUDE.md` against the shipped model, not the
fallback.

**Risk to watch.** DINOv3 on T-LESS crops is itself out of distribution - a null result may be the
backbone rather than the head. Report the embedding norm distribution on both datasets alongside, so
that alternative is visible rather than assumed away.

## 3. Resolve the two competing view selectors  (~2 hours)

**Problem.** `pick_view` can use a learned `view_rank`; `outline_source` then wraps the result with
`view_model`, which is on by default and always answers. `P2F_VIEW_PICK=ranker` therefore has no
effect on outline modes - the ranker's choice is computed and discarded. Two models, one decision,
one of them unreachable.

**Fix.** Measure before deleting. Run both on the 771 held-out parts already labelled in
`data/view_ceiling_big.json`: agreement with the best view, and sketch IoU of the view each picks.
Then either
- `view_rank` wins: `outline_source` defers to it and `view_model` is deleted, or
- `view_model` wins: `view_rank` and the `VIEW_PICK=ranker` branch are deleted, or
- they disagree usefully: combine them as one component, and prove the combination beats both.

Whichever way, **one component ends up owning the decision.**

**Verify.** Held-out A/B on the same parts with both arms regenerating specs, plus a FreeCAD build
check for null solids, exactly as the shipped view choice was measured.

## 4. Make out-of-distribution audit routine  (~1 day, then ongoing)

**Problem.** Every component has one dataset behind it. Split-by-part is enforced everywhere and
protects against memorising a part; nothing protects against memorising PrintCAD, and two of the
three components audited had. `section_constancy` is the only guard in the system and was only
findable by testing on a second dataset.

**Fix, in two parts.**

*(a) A standing audit.* `tools/ood_audit.py` runs every learned component against T-LESS and prints
in-distribution against out-of-distribution beside its majority or geometric baseline - the axis
classifier's 89/29, the depth model's both paths, the view model where triples exist. One command,
one table.

*(b) A regression gate.* `tests/test_ood.py` asserts each component's out-of-distribution score
stays within tolerance of its documented value, so a retrain that quietly improves PrintCAD and
breaks everything else fails the suite instead of shipping.

**Verify.** The audit is the verification. The gate's own value is proven by pointing it at
`axis_model` and confirming it fails on the ungated version and passes with `section_constancy`.

**What this does not fix.** A gate detects the failure; it does not repair it. Repair means either a
second labelled dataset for training, or components posed so the dataset cannot leak in - the axis
question stated as "does a constant-section axis exist and which is it" rather than "which of three",
so a part with no answer can be refused rather than guessed.

## Order and why

1 first: cheap, and it makes every later measurement trustworthy.
2 next: it is the only one where something shipping today is unknown.
3 then: a live ambiguity with no measured harm yet, but it makes 4's audit ill-defined until settled.
4 last: the deepest, and the only one that is ongoing work rather than a fix.


---

# Status: all four implemented

| step | outcome |
|---|---|
| 1. audible fallbacks | done - 14 silent returns annotated, 0 left, 4 tests |
| 2. audit the shipped depth path | done - **it loses to a constant, and to its own fallback** |
| 3. one view selector | done - the two are indistinguishable, so the shadowing was removed, not a winner picked |
| 4. routine OOD audit | done - `tools/ood_audit.py` and `tests/test_ood.py` |

**What each step actually found, as opposed to what it was expected to find.**

*Step 1* was meant to be housekeeping. It immediately surfaced a live fallback nobody had noticed:
scoring a single view makes `embed.for_views` fail every time, because it wants three image paths and
gets one, so any per-view experiment silently uses the tabular depth model.

*Step 2* was expected to confirm an earlier result and reversed it. The DINOv3 pixel path - the one
that runs on a real photograph - scores 1.117 median absolute log error against a T-LESS constant's
0.314, with its conformal band covering 24% where it claims 80%. The tabular fallback scores 0.236
and over-covers. The path we ship is the worse of the two off-distribution, and the earlier
"depth loses to a constant" note was about the fallback because bare masks gave `embed` nothing to
work with.

*Step 3* was expected to produce a winner and produced a tie: +0.008 [-0.018, +0.034] over 261
discriminating parts, agreeing with each other on 61%. The first attempt at this comparison was
invalid - it rebuilt view dicts from cached statistics, so `view_rank` saw a fabricated bounding box
and a symmetry count of zero, which are two of the eight features it reads.

*Step 4* is the only one whose value is preventative rather than immediate. `tests/test_ood.py`
pins each documented score within 0.06 and asserts two specific properties: that the axis classifier
still fails out of distribution and is still covered by `section_constancy`, and that the shipped
depth path is still worse than its fallback. If either stops being true the suite fails and someone
has to update the documentation deliberately.

**Left open.** Whether the depth pixel path's failure is the backbone or the head - the embedding
norm cannot say, being 1.7 for every T-LESS object because the vectors are L2 normalised. And
whether `view_model` survives a change of dataset, which needs three views of one object from one
setup, and T-LESS does not provide them.
