# Plan

## Where we are

The objective is **primitive F1**: drawn primitives matched against the ideal sketch's, type by
type, over a denominator that never moves. Region IoU is secondary - it rewards an outline that
agrees and is blind to a missing feature, which is how a circle scored 0.95 on a scalloped disc.

Measured on `data/test_ids.txt`, 246 parts with trustworthy truth that nothing has been fitted on:

| | value |
|---|---|
| primitive F1, fitted thresholds | **0.612** |
| primitive F1, hand-picked thresholds | 0.566 |
| paired gain from fitting | +0.045 [+0.026, +0.065] |
| region IoU, full photographed set | 0.629 [0.611, 0.646] |
| what a perfect sketch and true depth score | 0.918 |

A sketch-only run over all 1908 parts takes five minutes; a full run with solids takes twenty-two.

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

## 2. Prove the metric path on real photographs - done on T-LESS, board still unshot

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
