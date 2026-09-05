# Architecture v2: a parametric template fitter

## Why

v1 promised "any part from any photo" and always returned *something* — usually a
valid-but-wrong solid. The session that produced it also measured why that can't work:
casual photos give class + a few robust proportions, never faithful geometry (tilt costs
0.11 IoU, single-view depth is a 10x band, matting eats thin walls, an asymmetric feature
or a hand in frame corrupts the silhouette). Real 3D form needs real capture (16-view
carve). Everything between is confident garbage.

So the app stops being a reconstructor and becomes a **template fitter with a hard
honesty gate**: recognize a known class, fit that template's parameters to the evidence,
verify the fit, emit a clean model with a confidence ledger — or refuse with the shot list.

## The contract

> A model you can trust, or an honest no. Never a silent wrong.

## Regimes (input quality decides, not the object)

| capture | yields | path |
|---|---|---|
| 1-3 casual photos | class + a few proportions | fit a template |
| 16-view orbit, textured bg | a real 3D hull | carve -> fit template/primitives to the hull |
| neither | nothing trustworthy | refuse + exact shot list |

## Pipeline

1. ASSESS CAPTURE -> capability level (casual / carve / insufficient)
2. RECOGNIZE (LLM) -> part_class + which template. Never numbers.
3. FIT -> template.fit(measurements) -> params, least-squares to the squared
   silhouette (casual) or the carve hull (orbit). Each param tagged
   measured | inferred | template-default.
4. VERIFY GATE (HARD) -> re-project; IoU < tau => refuse, return reason. No unverified
   geometry is emitted.
5. EMIT -> clean parametric .FCStd + ledger + confidence.

## Non-goals (measured dead, do not attempt)

- A general geometry-from-photo compiler (v1's `compile_program`). Always succeeds,
  usually wrong. Deleted, not demoted.
- Trusting single-view depth as metric (10x band).
- Replacing the tracer/decomposer with a learned model (16 attempts, 0 wins).
- Intersecting independently-posed silhouettes (measured loss).

---

## Milestones

Each milestone is independently shippable, has one acceptance check, and reuses the v1
skeleton (registry, recognition, verify) that survives.

### M1 - Hard verify gate  (S)
Flip verify from warn to refuse. `design()` returns `{model: None, reason, reshoot: [...]}`
when silhouette IoU < tau (start tau = 0.85, tune on knowns). No FCStd written on refusal.
- **Accept:** the finger-held micro-fan and the angled bottle both return `model: None`
  with a reshoot list; the clean yellow fan still builds.
- **Reuses:** `design.verify` (exists). Just gate on it before `_build`.

### M2 - Templates fit, not fill  (M)
Give each template a `fit(measurements) -> (params, ledger)` that least-squares its params
to the observed silhouette rather than plugging one measured ratio. Convert `fan_guard`
first: fit frame_w, corner_r, bore_d, mount_pitch to the mask outline + hole.
- **Accept:** fan_guard params recovered within 5% on a synthetic rendered fan at known
  params; ledger labels every value measured/inferred/default.
- **Reuses:** `fan_guard.py` template, `count_rings`, `fit_ellipse`.

### M3 - Delete the general tier  (S, deletion)
Remove `compile_program` and the `general` branch from `design()`. No template + no carve
=> refuse. Keep `assemble` (flat PrintCAD path) and `revolve` behind M4's fitting.
- **Accept:** `design()` has exactly {template, carve, assemble, refuse}; test suite green
  after removing the general test.

### M4 - Carve -> template fit  (M)
When a 16-view capture exists, carve (existing `sfm_real`) then FIT a template or clean
primitives to the hull instead of tracing a raw section. Bottle -> cylinder+profile fit;
fan body -> box+bore fit. The hull is measurement; the template is the clean output.
- **Accept:** the good yellow-fan orbit (once reshot slow+low) fits the enclosure template
  to depth within carve's known 1-5%; output is a clean parametric solid, not a blob.
- **Reuses:** `sfm_real.py`, `fuse.measured_depth`, the template `fit()` from M2.

### M5 - Template library growth  (ongoing, one class per PR)
Add templates as `@register` + a parametric builder + `fit()`. Priority by what real parts
need: `enclosure` (box body, covers the fan's body), `revolve` (cylinder+neck, bottles/
knobs), `bracket` (L/plate + hole pattern). Each unlocks a whole class at designer quality.
- **Accept per template:** fits within 5% on a synthetic render of that class; refuses
  gracefully on out-of-class input.

### M6 - Ledger in the FCStd  (S)
Add a measured/inferred/default column to the params sheet, and a one-line confidence
summary in the doc. Depth is `inferred` until a caliper or carve makes it `measured`.
- **Accept:** opening any output FCStd shows, per dimension, whether it was measured.

---

## Learned track (vast.ai + HuggingFace available)

The session's 16 zeros were all *small models replacing a geometric step on synthetic
contours* - that family is exhausted, do not add to it. Two things were NEVER tested at
scale and are where compute + real data pay, because both attack the measured bottleneck
(capture costs 0.21 of the error, tracing only 0.11):

### T1 - Pose / tilt head on real photos  (highest value, lowest risk)
The session showed DINOv3 on the part crop predicts the face normal to 4.8 deg (silhouette
stats got 16.7), and that a tilt estimate is worth +0.13 by telling the user to reshoot
square - or by rectifying before a template fit. It failed only for lack of a real-photo
label set. **The data exists: BOP (T-LESS, ITODD, IPD, XYZ-IBD) ship real photos with
solved poses** - a free tilt/normal label. Train DINOv3 -> face normal, held out by object,
gate hard (abstain out of distribution, as the session's version already does).
- **Ships as:** the ASSESS-CAPTURE stage's tilt check + an optional rectifier feeding FIT.
- **Accept:** median tilt error < 5 deg on held-out BOP objects; refuses on unrecognized
  domains rather than lying.

### T2 - Image -> CAD-program prior  (higher risk, validate before trusting)
A modern image-to-CAD model (CAD-Recode / Img2CAD / DeepCAD-family) finetuned to PROPOSE a
template + rough params from a photo - a **prior/selector**, the one learned shape that has
won here, not a geometry replacement. Its output is a *proposal* the FIT stage refines and
the VERIFY gate can still reject, so a wrong proposal costs nothing.
- **Data:** ABC + Fusion360 Gallery + DeepCAD for CAD programs; render-to-real gap closed
  by finetuning on the real photos we capture (the session's law: deployment data must be
  in training). Objaverse / MVImgNet for real multi-view.
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

**Discipline for all three (from CLAUDE.md, non-negotiable):** split by part/object, report
the free baseline beside the score, validate on held-out REAL data not renders, and gate on
the end-to-end verify - never on the proxy metric. A trained model is a proposal the geometry
and the verify gate still police.

## What survives v1

- The tier registry (`design.TEMPLATES`, `@register`).
- Recognition as class-selector (LLM emits class, not geometry).
- `verify` re-projection (M1 makes it a gate).
- `fan_guard` template, `count_rings`, `fit_ellipse`, `sfm_real`.

## What dies

- `compile_program` + the general tier (M3).
- "always return a solid" behavior (M1).
- Blind param-fill (M2 replaces with fit).

## Order

M1 (honesty first) -> M2 (fan fits) -> M3 (delete trap) -> M4 (carve fits) ->
M5 (grow library) -> M6 (ledger). M1+M3 are the honesty core and can land first together.
