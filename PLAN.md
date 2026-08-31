# Plan

## Where we are

Measured on 200 PrintCAD parts (real hand-held photos, no fiducial), scale-free voxel IoU against
the STEP-derived truth:

| | value | 95 % CI |
|---|---|---|
| mean IoU | 0.432 | [0.400, 0.466] |
| oracle mode selection would add | +0.094 | [+0.070, +0.120] |
| learned mode selection, held out | −0.030 | [−0.063, +0.002] |
| paired A/B resolution at n=200 | ±0.007 | |

Every sketch is fully constrained, the sheet is in millimetres, a full benchmark run takes 172 s
locally and 88 s on a rented 4090 box.

Three facts drive everything below.

1. **The oracle gap is real and large.** Picking the best of the four modes per part is worth
   +0.094 — more than every geometry improvement of the last two days combined.
2. **The instrument is underpowered.** A 200-part paired comparison resolves ±0.007. Detecting a
   +0.01 improvement needs ~426 parts. Several experiments already run below that threshold were
   unfalsifiable, including the held-out learned-selector result, whose CI crosses zero.
3. **A third of the labels carry no signal.** Of 200 oracle labels, 9 % are unlearnable (no mode
   reaches 0.2) and 26 % are ties decided by less than 0.05. The honest ceiling is ~0.518, not the
   raw oracle 0.527.

## 1. Fix the instrument first

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

## 2. Prove the metric path on a real part

`--rectify` and `photo2fcstd-carve` are validated only against synthetic boards and a simulated box.
Every millimetre figure quoted for them comes from geometry we generated ourselves.

**Needs:** the printed ChArUco sheet, the OpenIPC camera on it, 12–20 photos including low grazing
angles.

**Done when:** the carved model is compared against the caliper readings we already hold, and the
error is stated in millimetres per dimension. If it is worse than the two-photo path, say so and
keep the two-photo path.

## 3. Score sketches, not solids

Every headline number is voxel IoU — whether the *solid* matches. The requirement was an editable
sketch. `ideal_sketches.py` already holds the true primitives for all 1907 parts and
`sketch_score.py` can compare against them; neither is reported by the benchmark.

**Done when:** each run reports primitive agreement alongside IoU, so "the ring became two circles"
is a measured claim rather than an anecdote from a gallery.

**Status: done.** Every run now reports, and records per part in `events.jsonl`: how many parts emit
a sketch at all, the region IoU of that sketch against the ideal one with a CI, and how many
reproduce its exact primitive counts. On the 200-part set: 97 of 200 emit a sketch, region IoU
0.583 [0.503, 0.664] on the 55 with trustworthy ground truth, 16 exact.

## 4. Reformulate mode selection

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

## 5. Self-supervision from the photos

The photos are their own supervision: a candidate model is right if it explains all views. Crude
64-direction silhouette consistency already gave +0.010. Done properly — differentiable silhouette
rendering, or a better scoring function — it turns every unlabelled photo set into training signal
and works at inference time with no dataset at all.

This is the only item that scales beyond PrintCAD, so it is the one worth real research time once
the instrument and the metric are trustworthy.

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
