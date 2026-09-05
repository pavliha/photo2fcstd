# Architecture v2: a parametric template fitter

## Why

v1 promised "any part from any photo" and always returned *something* - usually a
valid-but-wrong solid. The session that produced it also measured why that can't work:
casual photos give class + a few robust proportions, never faithful geometry (tilt costs
0.11 IoU, single-view depth is a 10x band, matting eats thin walls, an asymmetric feature
or a hand in frame corrupts the silhouette). Real 3D form needs real capture (16-view
carve). Everything between is confident garbage.

So the app stops being a reconstructor and becomes a **template fitter with a hard
honesty gate**: recognize a known class, fit that template's parameters to the evidence,
verify the fit, emit a clean model with a confidence ledger - or refuse with the shot list.

## The contract

> An outline you can trust plus a ledger that says what else is real - or an honest no.
> Never a silent wrong.

Verify certifies the face silhouette only. Depth, hidden features and grille detail are
certified by the ledger (measured / inferred / default), not by the gate. Say exactly that.

## Inputs the pipeline cannot measure (required, not inferred)

- **Scale.** Every fit is in pixels. One caliper reading or a carve makes it metric.
  Without it the output is emitted in px with `scale` as a params cell - never guessed.
  **Exact size is not a goal and never a gate: the product is correct proportions**, and
  the user scales once.
- **Depth of a single face.** Inferred (ledger) until a carve / dense depth measures it.

## Acceptance is proportion agreement, not caliper agreement

A milestone is accepted when two *independent* measurements of the same proportion agree
within a stated tolerance, and the spread is written to the ledger. Current honest
precision, from the fan clip: depth/width is **0.373** from the tilted side photo's
silhouette and **0.447** from VGGT 3D - **20% apart**. The 3D number is the more
trustworthy (foreshortening shrinks depth in a tilted 2D view), so the target is: template
fit proportions agree with VGGT proportions within 10%, and the ledger reports the
2D-vs-3D spread when both exist. "Within 5% of a caliper" appears below only as an
optional stronger check when the user chooses to measure.

## Regimes (input quality decides, not the object)

| capture | yields | path |
|---|---|---|
| 1-3 casual photos, square-on | class + a few proportions | fit a template |
| 1-3 casual photos, tilted | class only, until T1 squares them | fit with tilt flagged, or refuse |
| 16-view orbit, textured bg | a real 3D hull | carve -> fit template/primitives to the hull |
| neither | nothing trustworthy | refuse + exact shot list |

## Pipeline

1. ASSESS CAPTURE -> capability level (casual-square / casual-tilted / carve / insufficient)
2. RECOGNIZE (LLM) -> part_class + which template. Never numbers. Cached per photo set;
   the app must not change its answer between two runs on the same input.
3. FIT -> template.fit(measurements) -> params, least-squares to the squared
   silhouette (casual) or the carve hull (orbit). Each param tagged
   measured | inferred | template-default.
4. VERIFY GATE (HARD, per path) -> rasterize the built model's real silhouette, align
   scale + rotation to the input mask, IoU. Below the path's threshold => refuse with
   reason. No unverified geometry is emitted.
5. EMIT -> clean parametric .FCStd + ledger + confidence + scale cell.

## Day-1 coverage (stated so nobody is surprised)

After M1+M3 the app models exactly the classes in the template library (one: fan_guard)
plus flat single-extrusion parts via `assemble`, and refuses everything else. Coverage grows
one template per PR (M5). A refusal with a shot list is the intended day-1 answer for an
unknown part, not a bug.

## Non-goals (measured dead, do not attempt)

- A general geometry-from-photo compiler (v1's `compile_program`). Always succeeds,
  usually wrong. Deleted, not demoted.
- Trusting single-view depth as metric (10x band).
- Replacing the tracer/decomposer with a learned model (16 attempts, 0 wins).
- Intersecting independently-posed silhouettes (measured loss).
- Accepting any milestone on synthetic/rendered input alone. Perfect-input gains have
  twice failed to survive a photograph; every acceptance below is on real photos.

---

## Milestones

Each milestone is independently shippable, has one acceptance check on real photos, and
reuses the v1 skeleton (registry, recognition, verify) that survives.

### M0 - A verify worth gating on  (S, prerequisite for M1)
Today's `verify` compares a convex hull of model vertices to the mask, normalized
independently, unaligned. It scored the correct bottle 0.195 because the neck is concave.
Replace with: rasterize the built model's actual silhouette in the photo's projection, align
scale and rotation (Procrustes on the outline, or upright both by PCA), then IoU.
- **Accept:** the correct yellow-fan template scores > 0.9; the correct bottle revolve
  scores > 0.85; the finger-held micro-fan and the wrong stacked-cylinder bottle both score
  < 0.5. Measured, not assumed, before any threshold is chosen.

### M1 - Hard verify gate, per path  (S)
Flip verify from warn to refuse. `design()` returns `{model: None, reason, reshoot: [...]}`
below threshold; no FCStd is written on refusal. **Thresholds are per path**, set from M0's
measured distribution: template fits high (~0.85), `assemble` at what its benchmark actually
scores on photographs (~0.6 region IoU) so the gate does not silently kill the one
benchmarked pipeline.
- **Accept:** a regression set of known-good inputs (yellow fan, the clean PrintCAD parts)
  is NOT refused - false-refusal rate reported; the micro-fan and the angled bottle ARE
  refused with a reshoot list.

### M2 - Templates fit, not fill  (M)
Give each template a `fit(measurements) -> (params, ledger)` that least-squares its params
to the observed silhouette rather than plugging one measured ratio. Convert `fan_guard`
first: fit frame_w, corner_r, bore_d, mount_pitch to the mask outline + hole.
Claims square-on shots only until T1 lands; a tilted input is fit with `tilt: flagged` in
the ledger, or refused by M1.
- **Accept:** on **real photos of the yellow fan**, fitted proportions (bore/frame,
  pitch/frame, corner/frame) reproduce the silhouette (verify IoU > 0.9) and agree with
  the VGGT proportions within 10% where both exist. Synthetic renders may be used to
  debug, never to accept.
- **Reuses:** `fan_guard.py` template, `count_rings`, `fit_ellipse`.
- **Status (2026-09-05): done.** `fit_fan_guard` measures corner radius (circle fit on
  boundary points with the straight edges excluded - unexcluded, the fit read 34% high on
  a synthetic truth), bore (ellipse), ring count (radial profile) and mount pitch (screw
  heads by low saturation in the corner windows). `design_fan` fits every face-on photo
  and takes the median per parameter, ledger recording the view count. Yellow fan, three
  views: corner 7.13 mm (2 views), bore 69.75 (3), pitch 59.2 (2), rings 4 (3); gate IoU
  0.927. The pitch is the proof of fit-over-fill: the ratio default would have plugged
  71.5 and put the screws 20% too far out.

### M3 - Delete the general tier  (S, deletion)
Remove `compile_program` and the `general` branch from `design()`. No template + no carve
=> refuse. Keep `assemble` (flat PrintCAD path, behind its own M1 threshold) and `revolve`
behind M4's fitting.
- **Accept:** `design()` has exactly {template, carve, assemble, refuse}; test suite green
  after removing the general test. M1+M3 land together as the honesty core.

### M4 - Carve -> template fit  (M, blocked on a reshoot)
When a 16-view capture exists, carve (existing `sfm_real`) then FIT a template or clean
primitives to the hull instead of tracing a raw section. Bottle -> cylinder+profile fit;
fan body -> box+bore fit. The hull is measurement; the template is the clean output.
External dependency: a slow, low (20-40 deg) orbit of the fan on the textured desk - the
existing 27 s handheld clip registered 11 of 27 frames and is not carve-grade.
- **Accept:** the enclosure template's depth/width proportion agrees with VGGT's within
  10%; output is a clean parametric solid, not a blob.
- **Reuses:** `sfm_real.py`, `fuse.measured_depth`, the template `fit()` from M2.
- **Status (2026-09-05): the fan did not need the carve.** The enclosure is the fan_guard
  template plus a skirt whose depth comes from V1b's dense-depth measurement - the
  object's thinnest 3D extent, orientation-free - so `design_fan(..., depth_json=)` builds
  80 x 80 x 39.8 with `box_depth = 35.78, measured (3D, VGGT)` in the ledger, gate 0.927.
  The carve stays for shapes no template covers (V2). Adding the skirt also exposed a
  verify bug: `_model_silhouette` took the *largest-volume object in the document*, which
  for the flat guard was the plate Pad before its pockets - M0's 0.927 had been comparing
  an un-pocketed plate. Verify now takes the document's final shape and fills interior
  holes on both sides (the contract is the outer outline); the knowns re-measure at flat
  0.925 / skirted 0.927 / bottle 0.951 against 0.427 / 0.32 for the wrong ones.

### M5 - Template library growth  (ongoing, one class per PR)
Add templates as `@register` + a parametric builder + `fit()`. Priority by what real parts
need: `enclosure` (box body, covers the fan's body), `revolve` (cylinder+neck, bottles/
knobs), `bracket` (L/plate + hole pattern). Each unlocks a whole class at designer quality.
- **Accept per template:** fitted proportions reproduce the silhouette (verify) and
  agree with an independent 3D measurement within 10% on one real physical example;
  refuses gracefully on out-of-class input.

### M6 - Ledger in the FCStd  (S)
Add a measured/inferred/default column to the params sheet, and a one-line confidence
summary in the doc. Depth is `inferred` until a caliper or carve makes it `measured`;
scale is `required` until set.
- **Accept:** opening any output FCStd shows, per dimension, whether it was measured.

---

## Learned track (vast.ai + HuggingFace available)

The session's 16 zeros were all *small models replacing a geometric step on synthetic
contours* - that family is exhausted, do not add to it. Two things were NEVER tested at
scale and are where compute + real data pay, because both attack the measured bottleneck
(capture costs 0.21 of the error, tracing only 0.11):

### T1 - Pose / tilt head  (highest value; pretrain on BOP, label your own domain)
The session showed DINOv3 on the part crop predicts the face normal to 4.8 deg (silhouette
stats got 16.7), and a tilt estimate is worth +0.13 by telling the user to reshoot square -
or by squaring the silhouette before a template fit (this is what unblocks M2 on tilted
input). **BOP (T-LESS, ITODD, IPD, XYZ-IBD) ships real photos with solved poses** - a free
pretraining label. But the record also says the T-LESS head **refused 100% of desk photos of
printed parts** - it does not transfer to the domain you actually shoot in. So: pretrain on
BOP, then finetune on **a few dozen of your own photos with the ChArUco board in frame**
(the board's solved pose is the label; the board is not needed at inference). Gate hard:
abstain out of distribution.
- **Ships as:** the ASSESS-CAPTURE tilt check + the squaring step feeding FIT.
- **Accept:** median tilt error < 5 deg on held-out **own-domain** photos (not just BOP);
  abstains rather than lies on unrecognized input.

### T2 - Image -> CAD-program prior  (highest risk; the data is the problem)
A modern image-to-CAD model (CAD-Recode / Img2CAD / DeepCAD-family) finetuned to PROPOSE a
template + rough params from a photo - a **prior/selector**, the one learned shape that has
won here, not a geometry replacement. Its output is a *proposal* the FIT stage refines and
the VERIFY gate can still reject, so a wrong proposal costs nothing.
- **Data reality:** ABC / DeepCAD / Fusion360 are CAD programs with **no photos**. Renders of
  them are not photographs (measured, repeatedly). The label factory built this session
  (BOP contours labelled through the pose) produced 60k real contours and still yielded
  zero gain for a geometry model - so real image-program pairs at scale are the unsolved
  input, and they come from photographing parts you own the CAD for. Budget T2 as a data
  project first, a training project second.
- **Accept:** on held-out REAL photos it picks the right template class more often than the
  LLM recognizer alone, AND end-to-end verify IoU is no worse. Otherwise it does not ship -
  same discipline that rejected the 16.
- **Explicitly not:** a monolithic photo->solid net with no geometric grounding or verify
  gate. That refights the domain gap with none of the defenses.

### T3 - Part isolation from clutter  (enabler)
The micro-fan failed because segmentation grabbed the hand. Finetune the matting/segmentation
on held-in-hand + cluttered real parts so ASSESS-CAPTURE can isolate the part or refuse.
- **Accept:** isolates the part (IoU vs hand-labeled) on a held-out clutter set; the
  finger-held fan either segments clean or is refused, never merged with the hand.

**Discipline for all three (non-negotiable):** split by part/object, report the free
baseline beside the score, validate on held-out REAL data not renders, and gate on the
end-to-end verify - never on the proxy metric. A trained model is a proposal the geometry
and the verify gate still police.

## Video -> CAD: what v2 above does and does not solve

Reviewed specifically against the video use case (a phone orbit of a part). v2 solves it
for **templated parts given a good orbit** (carve -> template fit, measured depth). It does
**not** solve it generally, and it under-uses video, for these measured reasons:

- "Extract 1 fps, hope 16 register" is what produced 11/27 and 10/32 registrations. Video's
  asset is continuity: sharpness and angular coverage are measurable per frame, so frames
  can be *selected*, not prayed over. v2 has no such stage.
- Exhaustive COLMAP on sparse frames needs a textured background and fails on blur.
  Feed-forward multi-view models (DUSt3R / MASt3R / VGGT) are pretrained, on HuggingFace,
  and built for casual phone capture: poses + dense depth from a handful of frames. This is
  the honest use of the GPU - inference of a geometry foundation model, not training.
- T1 (single-view tilt) is a photo fix. From video, every frame's pose is known after V1,
  so the square-on frame is *chosen by pose*, not estimated from one silhouette.
- Untemplated objects are refused by design; the only general path is fitting primitives
  to the hull, which is one clause in M4. It is honestly achievable (fitting measured 3D
  cannot produce confident garbage the way single-photo guessing does) and needs to be a
  real milestone.
- A visual hull is unforgiving: one blurred-frame mask deletes correct material forever.
  Frame selection and dense depth both mitigate it; v2 said nothing about per-frame masks.

### V1 - Video-native capture  (M, unblocks the carve regime from a phone)
Select frames by sharpness (variance of Laplacian) and angular coverage (tracked baseline),
not by fps. Poses + dense depth from a pretrained MASt3R / VGGT run on vast.ai; fall back to
COLMAP only if the model is unavailable. Per-frame masks from the app's matting, weighted by
the frame's sharpness in the carve (a soft hull, not a hard intersection).
- **Accept:** >= 80% of frames from a slow phone orbit registered (vs 41% today); carve
  depth of the yellow fan within 5% of a caliper reading.
- **Uses:** pretrained models only. No training.
- **Status (2026-09-05):** frame selection alone took COLMAP registration from 11/27 to
  35/36 on the fan clip. VGGT-1B on a rented 5090 with the app's per-frame masks (eroded,
  percentile footprint) measures, from the phone clip: fan standing on edge -
  height/width 1.008, depth/width 0.447; bottle upright - round footprint 0.998,
  height/diameter 2.42 (pump tip under-sampled). Unmasked, desk clutter inflated the
  footprint and gave a plausible-looking wrong number (0.389) by coincidence - masks are
  not optional. The 5% accept still needs a caliper reading; the poses are saved for the
  carve (`runs/fan/vggt/*_poses.npz`).

### V2 - Hull -> primitives  (M, the honest general path)
RANSAC planes / cylinders / spheres on the carve hull, fit each, assemble as pad / pocket /
revolve features, build clean. For an object with no template this is the only route that
does not guess, because every primitive is fit to measured 3D. Verify against the hull and
against the frames' silhouettes.
- **Accept:** the fan body and the bottle each fit to primitives within carve tolerance
  with no template; a refusal on a hull too noisy to fit (reported, not hidden).

- **V2 status (2026-09-06):** `tools/cloud_primitives.py` on the VGGT clouds, no template:
  the fan classifies box (roundness 0.43) and builds a clean box 1 : 0.875 : 0.378; the
  bottle classifies revolve (roundness 0.99, rim spread 0.07) and builds a revolved
  profile - body 82/81/82/81, shoulder 71, cap 58 - height/diameter 2.2. The fan's traced
  footprint gives depth/width **0.378 against the tilted photo's 0.373 (1.3%)**, which
  corrects V1b's 0.447: the 98th-percentile extents were inflated by flying pixels; the
  traced top-half footprint is the cleaner measurement and now feeds the enclosure. Height
  reads 12% low (the top-half selection trims the face edge) - noted, not hidden. A 3D
  containment check (cloud points inside the built solid) is the honest next verify for
  cloud-derived primitives; silhouette verify against a tilted frame is meaningless.

### V3 - Pose-aware template fit  (S, once V1 exists)
Pick the frame most square to the face by its pose and run the template fit there; for
revolves, take the profile from true side-on frames and average over azimuth.
- **Accept:** template fit from video matches the square-photo fit within 3%; the bottle
  profile is monotonic through the neck (no mid-body bulge like today's 166 px).
- **Status (2026-09-06):** the runner saves per-frame squareness. In this orbit the best
  frame scores **0.813 (~35 deg off square)** - no frame is square, so the 2D fit from it is
  poor (bore 38% off the photo consensus) though better than an arbitrary frame (48%). The
  useful output is the squareness number itself: below ~0.95 the 2D fit is refused and the
  cloud (V2) carries the geometry - exactly the tilted regime in the table. The bottle
  profile from the cloud is monotonic (82 -> 71 -> 58), the accept's bulge is gone.

**Priority change:** for the video path, V1 -> V2 -> V3 supersede T1. T1 stays for the
photo-only path. Revised order becomes M0 -> M1+M3 -> M2 -> **V1 -> V3** -> M4 -> **V2** ->
M5 -> M6, with T1 only if photo-only input turns out to matter.

## What survives v1

- The tier registry (`design.TEMPLATES`, `@register`).
- Recognition as class-selector (LLM emits class, not geometry) - now cached, deterministic
  per input.
- `verify` re-projection (M0 makes it real, M1 makes it a gate).
- `fan_guard` template, `count_rings`, `fit_ellipse`, `sfm_real`.

## What dies

- `compile_program` + the general tier (M3).
- "always return a solid" behavior (M1).
- Blind param-fill (M2 replaces with fit).
- Convex-hull verify (M0).

## Order

M0 (make verify true) -> M1+M3 (honesty core, together) -> M2 (fan fits, square shots) ->
T1 (squares tilted shots; unblocks M2 generally) -> M4 (carve fits, needs the reshoot) ->
M5 (grow library) -> M6 (ledger). T2 only after T1 and M5 have a real-photo set to train on.
