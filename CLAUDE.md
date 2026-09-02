# photo2fcstd

Photos of a part in, a parametric FreeCAD **sketch** model out. The sketch is the
product. The solid is a consequence of it.

## The objective is the drawing, not the solid

Score with `sketch_score.py` against `data/printcad_ideal_sketches_all.json`, which
holds each part's real sketch extracted from its STEP file. `score.py` (voxel IoU on
solids) answers a different question and will point you the wrong way.

Only about 55% of the ground truth is usable. `sketch_score.trustworthy()` is the
filter: the reference face must be the true extrusion base (`prism`), not a sliver,
and its area times depth must actually equal the solid's volume. Quote numbers on the
trusted subset and say `n=`. The rest have no reliable reference and their scores mean
nothing.

## Region IoU is nearly independent of whether the drawing is right

It cannot see primitive type (an arc and the chords approximating it cover the same
area) and it cannot see small holes (they carry almost no area). Measured over 251
discriminating parts, region IoU correlates **0.22** with structural agreement and
**0.33** with reproducing the exact primitives - it explains about 5% and 11% of the
variance. Straight inversions are rarer than that sounds (4% of parts score in the top
quarter by IoU and the bottom quarter by structure, 2% the reverse), so IoU is not
lying; it is answering a different question from the one this project asks.

So judge a change with `sketch_score.verdict`, not with IoU alone. It reports region IoU
next to three structural terms, each a fraction of the real sketch reproduced:

- `loops` - how many real closed loops exist at all, which is the hole the tracer missed
- `curves` - how close the count of curved primitives is, which is the arc drawn as chords
- `elements` - how close the total primitive count is, which is fragmentation or over-smoothing

They are deliberately unweighted; weighting them needs a person saying which drawing they
would rather edit, and nobody has been asked. Sanity check on hand cases: the letter G at
IoU 0.41 scores 0.91 structurally, a rounded blob at IoU 0.81 scores 0.37, and a square
whose notches were smoothed away goes from 0.97 to 0.78.

Re-judging the shipped and reverted changes against it flipped nothing: view choice was
+0.038 IoU and +3.6 points exact, run merging -0.013 and -1.4, the mode allowed-list fix
neutral on both. The record survives because exact primitives were always reported
alongside - that is what rejected `eps_model`, which gained +0.005 IoU while losing 4.6
points of exact. Do not report IoU on its own again.

## Measure before changing a threshold, and measure builds too

Every empirical constant is in `thresholds.py`. They are jointly tuned and most sit at
a local optimum, so a lone change is usually neutral or worse. When you touch one:

- A/B on the **same** parts, regenerating specs for both arms.
- Report null solids and not-fully-constrained sketches, not just IoU. Several changes
  this repo has already rejected improved a statistic and broke models.
- 160 parts is enough to see a trend, too few to trust a 1-vs-2 difference.

Results already established, so nobody re-runs them: loosening the arc gates is closed,
having now lost on both metrics. It appeared to close the curve *fraction* gap (0.45 to
0.60 against an ideal 0.59), but that was an aggregate illusion - the looser gate fires
on 54% of parts whose real sketch has no curve at all and 57% of parts that do have one,
so it is not selective, and per-part curve *count* agreement gets worse (-0.074
[-0.141, -0.009]). Structure overall is -0.036 [-0.070, -0.002], IoU -0.013, exact 26%
to 19%, and it breaks 12 of 160 models. The tracer draws 1.61 curved primitives against
a real 5.17, and no threshold recovers the rest - but that is the tracer, not the photograph:
given a perfect drawing of the true face it still finds only 1.76. See the next section. Deriving arc direction correctly is neutral at the shipped gates.
Learned mode selection is worth ~0.001 of sketch IoU.

## The missing arcs are the tracer's, not the photograph's

The whole remaining structural gap is curves. Straight elements come out at 6.22 per sketch against
a real 6.78; curved ones at 1.61 against 5.17. So `curves` and `elements` are not two defects, they
are one: about three and a half missing curved primitives, which the tracer replaces with straight
pieces.

`tools/tracer_ceiling.py` settles where that comes from by handing the tracer a perfect drawing -
the ideal sketch rasterised from its own `xy` samples, square on, no walls, no tilt, no matting,
nothing to segment - on the same 180 parts the photo pipeline is measured on:

| | perfect drawing in | photographs in | really there |
|---|---|---|---|
| curved primitives | **1.76** | 1.62 | **5.19** |
| elements | 12.82 | 7.84 | 11.98 |
| region IoU | 0.694 | 0.645 | |
| exact primitives | 33% | 26% | |

**Perfect input recovers 1.76 curves against a photograph's 1.62.** The arcs are not being lost to
tilt, matting or resolution - the information is all there and the tracer does not use it. On
perfect input it produces roughly the right number of elements (12.82 against 11.98) with the wrong
*types*, splitting each arc into short straight pieces.

This is the largest identified headroom left in the pipeline and it is entirely in code. It also
corrects the arc-gate section above, which said arc recovery was capture-limited.

Note what it does *not* say. It is a ceiling on this decomposition only, and `approxPolyDP` plus arc
fitting is the same geometric prior that beat every learned replacement so far - so a fourth attempt
at replacing it needs a reason this case differs, and "there is headroom" is not one.

## Which arcs the tracer loses: shallow ones, and the gate says so

`tools/arc_survival.py` matches every ground-truth element to the primitives the tracer drew, in
the pixel frame of a perfect rasterised face, over 180 parts and 555 matched elements:

| ground truth | n | reproduced as a curve |
|---|---|---|
| circle | 33 | **98%** |
| bsplinecurve | 96 | 54% |
| arc | 149 | **26%** |
| line | 275 | 3% drawn curved |

Circles are fine. Arcs are the defect, and the loss is entirely on the shallow ones:

| arc sweep | n | kept | | arc radius | n | kept |
|---|---|---|---|---|---|---|
| 0-15 deg | 6 | **0%** | | 0-10 px | 12 | 0% |
| 15-30 | 14 | **0%** | | 10-25 px | 25 | **62%** |
| 30-60 | 40 | 13% | | 25-60 px | 40 | 34% |
| 60-120 | 53 | 31% | | 60-150 px | 39 | 20% |
| 120-240 | 33 | 45% | | >150 px | 33 | **5%** |
| >240 | 3 | 68% | | | | |

The cliff sits exactly where `ARC_MIN_SPAN_DEG = 40` puts it, and a large-radius short-sweep arc is
geometrically almost a straight line, which is the same fact from the other side.

Simplification is not what loses them. Sweeping `trace.RUN_EPS` over a 16x range on perfect input
(n=165) moves curve count by 0.17 against a deficit of 3.5, while moving element count from 17.4 to
11.9 - it controls fragmentation, not primitive type, and no arm beats shipped. Independently, of
106 lost arcs only 29% collapse to a single line; **71% survive as two or more pieces and are
refused by the fitter**. The arcs reach the decision intact.

The tracer does not fail at arcs so much as **saturate**. Pooling trusted and untrusted parts
(n=416), it recovers 99% of curves on parts that have one, 89% on parts with two or three - 321 of
416 parts - and draws 0.08 false curves on the 134 that have none. Recovery then falls to 45% at
4-7 real curves, 33% at 8-15 and 23% past 16, while drawn curves creep from 0.99 to only 6.37. The
defect lives entirely in the ~23% of parts with four or more curves, and complexity selects them,
not `trustworthy()`.

**The saturation replicates on an independent dataset.** On 1,185 Fusion 360 Gallery parts - a
different authoring tool, different parts, 6.6x the sample, no b-splines - recovery runs 99% at one
real curve, 87% at two or three, 57% at four to seven and 24% past eight, against PrintCAD's 99, 89,
45 and 23. Overall structure is 0.879 and 67% exact there against 0.756 and 33% here, which is what
simpler parts and no b-splines buy rather than a better tracer.

A likely mechanism, untested: `trace.py:696` requires `chord > ARC_MIN_CHORD_FRAC * length_px`
before a run is considered as an arc at all - a fraction of **the whole part**. On a complex part
every arc is a small share of it. The sagitta test beside it is relative to the local chord; only
this one is global. That is item A6.

Quote the ceiling **excluding b-spline parts**. They are 15% of the set, 94% of their curves are
b-splines, and they score 0% exact because there is no b-spline primitive to emit - so they inflate
the deficit without being fixable by any arc work. Excluding them it is 1.66 against 3.83, not 1.76
against 5.19.

**This is a precision/recall setting, not a bug.** Only 3% of real lines come out curved, and that
precision is what loosening the gate destroys: the looser gate fires on 54% of parts whose sketch
has no curve at all, because sweep and sagitta are exactly the quantities that cannot separate a
shallow arc from a straight edge. Any fix has to bring evidence a straight line would not have -
neighbouring geometry, symmetry, a longer run - rather than a lower threshold on the same
quantity.

It also bounds A4: b-splines are reproduced as curves 54% of the time, better than arcs, so they
are not the unreachable mass.

## Perfect-input gains have twice failed to survive a photograph

Two arc changes were measured on perfect rasterised faces, both improved it with a confidence
interval clear of zero, and both broke models when regenerated from real photographs and built:

| change | perfect input | on photographs | builds |
|---|---|---|---|
| chord gate absolute, not part-relative | +0.0030 structure [+0.0010, +0.0057] | -0.0045 [-0.0135, +0.0025] | valid 125 to 123, unsolved 3 to 6 |
| merge runs before fitting | +0.0132 [+0.0037, +0.0226] | +0.0170 [-0.0051, +0.0396] | valid 124 to 121, unsolved 3 to 6, **build ~50x slower** |

Both reverted. Treat a perfect-input measurement as a **screen** that says whether a mechanism
exists, never as evidence to ship on, and never run an arc A/B without regenerating specs from
photographs and building them. A census of 3,384 runs also shows where arcs actually die: 42% fail
`ARC_MIN_SPAN_DEG`, 23% fail the circle fit, and the chord gate fires on **one run in 3,384**.

## Mode selection and coverage are finished

Measured on 197 trusted parts with photos, every one draws a sketch and `stations` is never chosen,
so the "straight choice" described further down - 46% of parts drawing nothing - no longer binds; it
was written before the mode allowed-list fix. And scoring every allowed mode for every part, the
chosen mode is already the best for 79% of them: a perfect oracle is worth **+0.0003 region IoU
[-0.022, +0.023]** and +0.022 structure. There is nothing left in mode choice.

## Where the error actually is

Against the orthographic silhouette, on trusted parts whose photo shows the extrusion
face: capture costs 0.212, our tracing and regularisation 0.112, and the fact that a
silhouette includes side walls the base face does not costs 0.089. Threshold work lives
in the middle term only.

**On a thin ring, matting decides the score and nothing else does.** 01407 is a washer whose ideal
is two concentric circles with a wall 2% of the radius. The pipeline draws exactly two circles -
`exact` true, structure 1.000 - and scores region IoU **0.557**, with `extra` 0.747. The cause is
not the tracer: the RMBG mask itself has a radius ratio of 0.965 against an ideal 0.980, and the
traced circles reproduce that mask to within 0.001. Segmentation eats about **13 px** of the hole,
so the wall comes out 29.7 px where 16.8 is right, 1.77 times too thick.

Matting costs about 0.02 of sketch IoU on an ordinary part. On a thin annulus the same class of
error costs **0.41**, because the wall area is the whole of the shape. Before chasing a mode or a
threshold on a ring-shaped part, measure its mask.

**And there is nothing to turn.** The alpha threshold cannot move it: sweeping `> 128` from 96 to
240 takes the ratio from 0.964 to 0.971 against an ideal 0.980, because only **0.43% of pixels sit
in the transition band** - RMBG is confidently placing the boundary in the wrong place, not
hesitating about it. Eroding the mask does move the ratio arithmetically, but a 30 px ring wall
loses 40% of its area to the 6 px that would fix it, so any guard worth having refuses exactly the
parts that need it.

Nor is the bias general enough to erode globally. Over 175 discriminating parts, `extra` exceeds
`missing` on **53%** - a coin flip - and a three-pixel erosion measured end to end gives region IoU
+0.0079 [-0.0036, +0.0213], structure -0.0027 [-0.0287, +0.0227], and **valid solids 127 to 126
with unsolved sketches 1 to 2**. Reverted. Improving a thin ring needs a better matte, not a
post-process.

`tools/split_capture.py` breaks the capture term down further, and the answer is not
what it looks like:

| camera tilt off the face normal | orthographic | with perspective | real photo |
|---|---|---|---|
| 0 degrees | **0.946** | 0.930 | |
| 8 degrees | 0.834 | 0.835 | |
| 15 degrees | 0.799 | 0.795 | |
| 30 degrees | 0.801 | 0.798 | |
| measured photos | | | **0.814** |

**Perspective is not the problem** - it costs 0.003 to 0.016 at any tilt, because a
phone at arm's length from a 20 mm part is nearly orthographic already. **Matting is not
the problem either**: real photos score 0.814 where matting-free synthetic views at the
same apparent tilt score 0.834, so RMBG costs about 0.02.

The whole capture term is **viewpoint tilt**. Eight degrees off-axis already costs 0.11.
Shooting square to the face is worth about +0.13 with no code at all, and it is the
cheapest improvement available to this project. The value of the ChArUco target is not
undoing perspective, it is knowing the pose so the tilt can be corrected, plus giving
scale and a measured depth.

Depth cannot be a constant: the true depth/length ratio spans 0.055 to 0.503.

## Environment

- Interpreter: `~/3DPrint/.venv/bin/python`. `photo2fcstd` is installed editable.
- `FREECADCMD` defaults to `~/Code/FreeCAD/build/release/bin/FreeCADCmd`;
  `P2F_DATA` points at the PrintCAD dataset.
- Segmentation masks and voxels cache under `~/.cache/photo2fcstd`. All 5,713 masks are
  already built, so re-running the full set costs no GPU time.
- `FreeCADCmd` will not execute a script from outside the repo. Put it in the project
  directory and delete it afterwards.
- macOS spawns, so anything using `multiprocessing` must live in a real file with an
  `if __name__ == "__main__":` guard. A heredoc piped to python forks endlessly.
- Never add a module that shadows a stdlib name (`inspect.py` breaks numpy's import).

## Batch work

`bench.py` runs the four-stage benchmark; `build.py` batch-builds many specs in one
FreeCADCmd process via `P2F_LIST`. For sketch-level work you do not need FreeCAD at
all - generate specs with `--spec-only` and score them directly, which is ~50x faster.
Build a sample through FreeCAD anyway before shipping, to catch null solids and
unconstrained sketches.

`valid` in a build report means a non-null solid with volume; a shape can be
`isValid()` and still be nothing.

## Renting a GPU (vast.ai)

`vastai` is installed and authenticated. Budget is **$100/month with auto top-up**, so
cost is rarely the binding constraint - time is.

**The rule: estimate the runtime first, then decide.**

- Under ~30 minutes: run it locally, do not rent. Setup, upload and teardown cost more
  than the job saves.
- Over ~1 hour: rent. Do not make anyone sit through it.
- In between: local if it can run in the background while other work continues,
  rented if it blocks progress.

Estimate honestly by timing a small slice and extrapolating, and say the estimate out
loud before choosing. A GPU only helps the stage that is actually GPU-bound:

| job | bound by | rent a GPU? |
|---|---|---|
| RMBG segmentation of the photo set | GPU | yes - 71 min locally for 5,713 photos, minutes on a 4090/5090. Already cached, so only for a new dataset or a different matting model |
| `synth.py` data generation | CPU (OpenCV rasterise + trace) | only for the vCPU count - 133 s for 13.7k samples on 8 cores, ~6x faster on a 48-core box |
| `curvenet.py` training | trivial - 195k params, seconds an epoch | no |
| the benchmark, tracing, scoring, FreeCAD | CPU | no |

A bigger card does not speed up a CPU-bound stage. For synthetic data generation the
thing to buy is vCPU count, so sort on `cpu_cores`, not `dlperf`. For segmentation or a
real image-to-CAD model, buy the GPU.

**Pick on value, not on the top of the list.** `-o 'dlperf-'` puts a B200 at
$7.50/hr first; an RTX 5090 at $0.40/hr has roughly a third of the dlperf for a
nineteenth of the price, so it wins on `dlperf/$` by about 2x. Unless a job genuinely
needs 180 GB of VRAM, a 5090 or 4090 is the right machine.

```bash
vastai search offers 'reliability>0.95 num_gpus=1 gpu_ram>=16 dph_total<0.5 inet_down>100' \
  -o 'dlperf-' --limit 10
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
  --disk 40 --ssh --direct --label photo2fcstd
vastai show instances                     # wait for 'running'
vastai copy local:./data C.<ID>:/workspace/data
ssh -p <PORT> root@<HOST>                 # host/port from: vastai ssh-url <ID>
vastai copy C.<ID>:/workspace/out local:./out
vastai destroy instance <ID>              # ALWAYS - billing runs until destroyed
```

Use on-demand, not interruptible, for anything longer than a few minutes; a preempted
run costs more in wasted time than the spot discount saves. Ship the masks, not the
photos - `~/.cache/photo2fcstd/masks` is 13 MB against gigabytes of JPEGs.

## Learned curve segmentation was tried and lost

`synth.py` + `curvenet.py` train a 1D CNN to label each contour point straight or
curved, from silhouettes of the STEP sketches rasterised under random homographies
(exact supervision, median label error 0.24 px). It reaches **0.82** per-point test
accuracy against a 0.54 majority baseline, split by part.

End to end it is clearly worse than `corner_runs`:

| arm | sketch IoU | curve frac | elements/sketch |
|---|---|---|---|
| geometric (shipped) | **0.579** | 0.43 | 8.2 |
| learned | 0.461 | 0.38 | 16.6 |
| learned + minimum run length | 0.452 | 0.38 | 7.3 |

Enforcing a minimum run length fixed the fragmentation and made IoU slightly worse, so
the loss is not over-segmentation - the predicted boundaries are simply in the wrong
places. Per-point classification is the wrong output for this: `approxPolyDP` localises
a corner to a geometric extremum, while a per-point classifier gives a boundary fuzzy
by several points, and a corner off by a few points moves a line endpoint visibly.

Keep the modules - they are the data pipeline for a better-posed model - but the next
attempt should predict primitives directly (a sketch is a short program: DeepCAD,
Vitruvion) or regress corner positions as keypoints with sub-point offsets, not label
points. Enable the losing path with `P2F_LEARNED_CURVES=1`; it is off by default.

## Where a learned component pays, and where it does not

Twelve learned attempts, two wins, and the split is not about model capacity - it is about
what the model is asked to do:

- **Won**: `view_model` picks which of three photos to draw from; `mode_model` picks which
  mode to build. Both **select among candidates the geometry already produced**, and both
  are labelled by the end-to-end score itself.
- **Lost**: `curvenet`, `cornernet`, `eps_model`, `sketchnet` (twice), constraint
  prediction, per-element confidence. All of them **replace a geometric step** with a
  prediction, and every one lost to the code it replaced.

The geometric pipeline is a strong prior that a small model on a few thousand parts does
not beat. Before proposing a model, ask which of the two it is. If it replaces
`approxPolyDP`, arc fitting, or the tracer, the prior says it loses and it needs a reason
this case differs. If it chooses between things the tracer already built, it is worth a
run.

## How to work on the ML parts

Written for someone who is not an ML specialist. Each rule has the moment from this
repo that earned it.

**1. A number means nothing without a baseline.** Always compute what you get for free.
Here the majority class ("straight") is 54%, so a model at 0.66 is barely doing
anything, and 0.82 is the number that matters. Report them together, always.

**2. Split by part, never by sample.** Fourteen views of the same part are not fourteen
independent examples. Put every view of a part on one side of the split, or the model
memorises the part and the score is fiction.

**3. Get labels from geometry you already trust, not by hand.** The STEP files already
say which edge is a line and which is an arc. Rasterise that sketch, trace it with the
*same* `outline()` production uses, and label each traced point by point-to-segment
distance. Exact supervision, zero labelling, and the input distribution matches
production. Check the label quality (`median dist` should be well under a pixel).

**4. Augment for the failure you are trying to fix.** The geometry assumes orthographic
projection, which is why an angled photo of a hexagon traces as a pentagon. So the
synthetic views apply a random homography - the model sees perspective during training.
Augmentation is not decoration, it is where you inject the invariance you want.

**5. Drop labels that have no right answer.** A point exactly where a line meets an arc
belongs to both. Training on it teaches noise. Compute distance to each class
separately and mark a point ambiguous when they are within a couple of pixels, then
exclude it with `ignore_index=-100` - and compute the metrics on valid points only,
otherwise you are just hiding the hard cases from yourself. This was worth several
points of accuracy here.

**6. Read the training curve before touching the model.**

| what you see | what it means | what to do |
|---|---|---|
| train loss falling, test still rising at the end | underfit | more data, more epochs, more capacity |
| train loss near zero, test flat or falling | overfit | stop earlier, augment more, shrink the model |
| test score swinging wildly while train loss falls smoothly | training instability, not learning | usually normalisation - BatchNorm's running statistics were the culprit here, GroupNorm fixed it |

**7. Save the best checkpoint, not the last one.** This repo's first run peaked at 0.820
on epoch 28 and then overfit down to 0.800, and the saved file was the worse model.
Checkpoint whenever validation improves.

**8. Pose the problem so the labels are real.** The first attempt had three classes and
forced 348 b-spline edges into "arc", a category they do not belong to. Collapsing to
straight-vs-curved - and letting the existing fitter decide arc-vs-circle afterwards -
took accuracy from 0.664 to 0.743 on its own. If a label is arbitrary, the model cannot
learn it and you cannot trust the score.

**9. The proxy is not the objective. Always close the loop.** 0.82 per-point accuracy
lost end to end, 0.579 to 0.452 sketch IoU. Per-point labels cannot localise a corner
the way `approxPolyDP` can, and a corner is a line endpoint. Never ship on the proxy
metric; run the A/B on `sketch_score` with build checks, same as any threshold change.

**10. Suspect your harness as much as your model.** An eval loop here stepped 64 while
slicing 128, so test chunks overlapped and the predictions tensor came out twice too
long. It crashed, which was lucky - a subtler version would have quietly reported a
wrong accuracy. When a number surprises you, check the measurement first.

**11. Choosing the output representation beats adding capacity.** Every gain here came
from reformulating (binary labels, excluded boundaries, right normalisation), none from
a bigger network. When something plateaus, ask what the model is being asked to
predict before making it larger.

## Tilt is in the pixels, and the silhouette throws it away

The whole capture term is viewpoint tilt, and until now the only way to know a photo's tilt was a
ChArUco board in the frame. It turns out the photograph itself carries it. Measured on 3,132 real
T-LESS frames over 30 objects with dataset poses as truth, **held out by object**, predicting the
part's face normal in camera coordinates - which is exactly what a rectifying homography needs:

| arm | MAE | median | within 10 deg |
|---|---|---|---|
| constant (what you get free) | 22.6 deg | 20.7 | 20% |
| nine silhouette statistics | 16.7 | 15.0 | 34% |
| DINOv3 on the binary mask | 8.5 | 6.3 | 70% |
| **DINOv3 on the part's pixels** | **4.8** | **3.6** | **89%** |

Shading is worth **3.6 deg [3.4, 3.9]** beyond the silhouette. Both arms get the identical crop box
from the dataset mask, so the only difference is whether the crop holds the photograph or its
silhouette. The turntable background contributes nothing (+0.05 deg [-0.05, +0.15]), so this is
shading on the part, not scenery - which is what makes it plausible on a part sitting on a desk.

Note this is the third kind of learned component, and it is neither of the two in the section
above: it does not select among candidates, but nor does it replace a geometric step, because there
is no geometric step here to replace. The pipeline currently has no tilt estimate at all.

**Pose the target in camera coordinates.** Expressed in the object's own frame the azimuth is
arbitrary from one object to the next, and every arm ties the constant at 48 deg. That was a badly
posed target, not a negative result - the same trap as the three-class curve labels.

**But it cannot be cashed in by rectifying the photo.** Rendering 115 trusted parts square on and
at 15 degrees, as the full solid silhouette and as the bare base face, separates the two things
tilt does. Projective distortion of the face, which a homography removes, costs **-0.009
[-0.026, +0.008]** - indistinguishable from zero. The side walls coming into view, which no 2D warp
removes because the information is not in the image, cost **-0.084 [-0.114, -0.058]**. Rectifying
the solid silhouette by the *true* normal scores -0.013 [-0.028, +0.002]: no help even with a
perfect angle. This is the same fact as "perspective costs 0.003 to 0.016 at any tilt" in the table
above, followed through to its consequence.

So the value of a tilt estimate is **telling the photographer to reshoot**, not correcting the
photo - which is the +0.13 that shooting square was always worth, now available without a ChArUco
board in the frame. Its other plausible use is choosing among the photos already taken, which is
the shape of problem the two learned winners solve.

**It does not transfer, and the gate is what saves it.** On PrintCAD renders the T-LESS head scores
34 deg, worse than a constant, and its gate refuses 100% of PrintCAD images - real photographs and
renders alike. Training a second head on 2,989 PrintCAD renders instead beats a constant by only
**+2.5 deg [+2.2, +2.8]** (11.6 against 14.1), useless when the reshoot threshold is 8, and *its*
gate also refuses every real photograph.

The range is not the explanation, and neither is the shading. On the quantity the check actually
reads - tilt magnitude, not the full normal, which also carries an azimuth the check never uses -
restricted to the 0-30 deg regime it lives in:

| trained on | tilt error | constant | within 5 deg |
|---|---|---|---|
| **T-LESS photographs** | **2.63 deg** | 6.98 | **87%** |
| PrintCAD flat Lambertian renders | 7.62 | 8.81 | 45% |
| PrintCAD Blinn-Phong, specular, antialiased | 7.96 | 8.81 | 43% |

Rewriting the renderer with specular highlights, coloured albedo, several lights and supersampling
made it very slightly *worse*, so "the renders lack shading" was the wrong diagnosis and two
photometric attempts is enough - the same reason a lone threshold change is not worth chasing here.
Renders are not photographs and no knob tried closes that.

`data/tilt_printcad.npz` and its head are deleted; `tools/tilt_synth.py` stays as the harness that
established the null. So `tilt_model.square_check` ships gated and honest - it reports a reshoot when it recognises the
photograph and says "not checked" when it does not - but the artefact it needs does not exist yet.
**The way to get it is a few dozen real photographs with the ChArUco board in frame**, whose solved
pose is a free tilt label. The board is not needed to *use* the check; it is the cheapest way to
*train* it, after which the check works board-free for good.

**Still not established**: that rectifying by any estimate improves the drawing - see the section
above, where it does not.

## Carving is the strongest path, and it is validated synthetically

`carve.py` needs the ChArUco board in the photos, so it cannot be measured on the
PrintCAD sets at all - those photos have no target. `carve_check.py` drives it with
synthetic views of a truth mesh instead (exact poses, rendered silhouettes), which
tests the geometry without any capture. Run `python -m photo2fcstd.carve_check` for the
self-check, or with arguments `<views> <voxel_mm> <elevations>` for the sweep.

With 16 views and 0.25 mm voxels, over 40 trusted parts:

| | volumetric IoU vs truth | extent error |
|---|---|---|
| all parts | 0.736 mean, 0.810 median | +0.71 / +0.35 / +0.85 mm |
| at least 4 voxels thick (n=30) | **0.791** | +4.3% / +4.8% / +30.5% |
| thinner than 4 voxels (n=10) | median true thickness 0.43 mm | not resolvable |

For comparison the photo pipeline reaches about 0.43 solid IoU and its best possible
mode choice 0.527. Carving roughly doubles that, and the depth is measured rather than
guessed, so it is where the remaining accuracy is.

**The axis classifier is PrintCAD-specific, and confidently so.** It reaches 89% there and 29% on
T-LESS against 33% for chance, because every training example came from a dataset of extrusions and
T-LESS parts have no constant-section axis to find. Its confidence runs backwards out of
distribution - top score 0.73 when wrong against 0.56 when right - so gating on the model's own
score abstains on the cases it gets right. Gate on geometry instead, or retrain. Note also that
`carve_check.silhouette` renders on its own 900x900 canvas, so it cannot be compared against a
dataset's masks without matching the intrinsics first.

**Both halves of the capture term have now been measured, and neither is large.** Pose from
a detected board is good to 0.016 degrees where the budget allows 2, and eight pixels of
correlated boundary wander - far worse than RMBG - costs only 0.074 of sketch IoU, leaving
carving ahead of the photo path even then. Sixteen views average matting error out, because
a voxel survives only where every silhouette agrees.

**Measured, not expected: 0.649.** `tools/tless_sketch.py` carves 21 discriminating T-LESS objects
from real photographs and scores the *drawing* against the mesh's own cross-section: **0.649 sketch
IoU against a trivial circle's 0.493**. The 0.715 previously written here was an extrapolation from
synthetic carves plus a boundary-noise experiment, and it was optimistic. Note also that this is a
different part set from PrintCAD - its circle baseline is 0.493 - so 0.649 is **not** comparable
with the photo path's PrintCAD numbers without matching the parts.

One number in that run is not to be trusted: renders at the same poses score 0.586, *below* the real
photographs' 0.649, which is the wrong way round. `carve_check.silhouette` rasterises on its own
900x900 canvas and the dataset's masks do not share those intrinsics, which is already recorded
above as making the two incomparable. Treat the render arm as broken rather than the real arm as
flattered.

**Size the voxel to the smallest feature you care about - about a quarter of it.** The
default `VOXEL_MM = 0.4` cannot resolve anything under roughly 1.6 mm, and a quarter of
these parts are thinner than that. Relative extent error on the thin axis is a
misleading statistic when the feature is near the voxel size (a 0.11 mm plate carved at
0.5 mm reads as +350%); quote absolute millimetres there.

Two harness bugs worth remembering, both found here: an OpenCV camera basis must be
`[right, down, forward]` and right-handed (`det = +1`), and with every camera above the
horizon the space beneath the part is unobservable, so clip the grid at the board plane.

## Carving no longer needs the board: poses come from the scene

The ChArUco board existed to tell carving where each camera was. Structure-from-motion gets the same
thing from the scene, and `tools/sfm_check.py` gates it on renders with a textured ground plane and
exactly known poses:

| | 24 views, one part | 16 views, four parts |
|---|---|---|
| images registered | 24 of 24 | 16 of 16, all four |
| camera rotation error | median **0.031 deg**, worst 0.060 | 0.031 to 0.058 |
| carve agreement against true poses | **0.987** | 0.949 to 0.997 |

A detected board gives 0.016 deg against a 2 deg budget, so SfM is the same order with plenty of
headroom. One of the four parts carves to nothing **from true poses too** - it is 0.25 mm thin, 0.4
of a voxel - so it says nothing about SfM.

**Two hard requirements.** The background must be strongly textured: at a quarter contrast **0 of 16
images registered, on every part**, because the part is textureless and the scene does all the work.
And sixteen views, not eight - eight registered five. Scale is untouched by any of this; SfM is
scale-free and a metric solid still needs one caliper reading.

Untested: real photographs, where motion blur, exposure drift and a real desk's texture all differ.
The gate measures geometry, and geometry was not the risk it looked like.

## Do not intersect two independently-registered photo silhouettes

Twice measured, twice lost:

- A three-view visual hull from the photo masks scores 0.332 against 0.433 for the
  plain pipeline, even choosing the best of 48 poses. With *truth* silhouettes the same
  code reaches 0.823, so the loss is registration, not the idea.
- Replacing the front width-staircase with the traced outline and intersecting it with
  the side staircase - an attempt to get a real drawing and keep the two-view solid -
  scores 0.218 against 0.226 for two staircases and 0.280 for the outline alone.

Intersection is unforgiving: any misalignment between two views deletes correct
material and nothing restores it. The modes survive precisely because they never
combine two separately-framed silhouettes. Multi-view only pays once the views share a
pose, which is what `carve.py` gets from the ChArUco board.

So drawing every part **and** keeping the two-view solid is not available. It is a
straight choice - though as measured now, `stations` is never chosen and coverage is 100%, so this
paragraph describes the code before the mode allowed-list fix: `stations` kept solid IoU 0.433 with
46% of parts drawing nothing,
always-outline gives 100% coverage and sketch IoU 0.557 for solid IoU 0.364.
`tests/test_regression.py` currently encodes the first.

**Scoring one view at a time is not the same pipeline.** `assemble` with a single view took a
different mode from `assemble` with three, and until this was fixed the learned classifier ignored
the allowed-mode list and routed 20% of single-view specs to `stations`, which draws nothing. Any
per-view experiment - view selection labels above all - inherits that. Check what mode a single-view
spec chose before trusting its score.

## Depth is predicted, with an honest interval

`depth_model.py` predicts `log(depth / length)` from the silhouette features, trained on
the 1442 parts whose STEP file records a real extrusion depth against a non-sliver face.
Gradient boosting on tabular features - small wide data, so trees, not a network.
`tools/depth_data.py` builds the rows, `tools/depth_train.py` fits and calibrates.

Measured on a test slice that neither training nor calibration saw:

| | median \|log\| error | within 2x |
|---|---|---|
| best constant | 1.045 | 35% |
| geometric estimate (0.46 x edge-on aspect) | 1.118 | 28% |
| **learned point estimate** | **0.435** | **63%** |

End to end on 41 held-out outline parts the solid IoU rose from 0.404 to 0.483.

The band comes from conformalised quantile regression, so its coverage is guaranteed
rather than hoped for: **83% of true depths fall inside the 80% band**. Plain quantile
regression covered only 55%, which is why the calibration step matters.

**On the shipped pixel path the band is a constant.** `pixel_predict` has no quantile heads: it
returns `exp(point +/- offset)` with a fixed offset, so **46 of 47 measured parts report exactly a
3.20x spread**. Marginal coverage still holds, so "80% of the time between X and Y" is honest on
average, but the interval carries no per-part information and cannot say which parts are uncertain.
The tabular fallback does have real per-part quantiles (5x to 145x on four sampled parts); the 10x
median quoted below is that model's, not the shipped one's. A further consequence: the note calls a
depth usable only under a 2.0x spread, so the pixel path can **never** report a usable depth - 0 of
47 parts did.

**The bands are wide, and that is the real finding.** The median 80% band spans about
**10x**, and only a couple of percent of parts get a band tighter than 2x. Depth simply
is not in an uncalibrated photo of a part seen face-on, and the point estimate being
decent does not change that. The `params` note now states the range and says outright
whether the number is worth building from - which is the honest version of the old
blanket "measure it".

Relaxing the training filter from the strict `trustworthy` set (889) to any non-sliver
face (1442) improved both the point estimate (0.514 to 0.435) and the band (11.7x to
9.7x), as the learning curve predicted.

Permutation importance says the model leans on `ellipse_rms`, `stroke_frac` and the
*spread of rectangularity and solidity between views* - how much the three silhouettes
disagree - not on the thinnest view. That is why the hand-written edge-on estimator,
which used exactly that thinnest view, lost to a constant.

`data/depth_model.joblib` is gitignored like the other model artefacts, so a fresh clone
falls back to the geometric estimate until `tools/` rebuilds it.

**The shipped pixel path is the one that fails out of distribution.** With T-LESS views carrying
their RGB frames so DINOv3 actually runs: the pixel path scores 1.117 median absolute log error
against a T-LESS constant's 0.314, 24% within 2x, and its conformal band covers 24% where it claims
80%. The tabular fallback scores 0.236 and over-covers at 100%. `P2F_DEPTH_PIXELS=0` picks the
tabular path and is safer on anything that is not a PrintCAD-like plate or bracket. Embedding norm
cannot diagnose whether the backbone or the head is at fault - it is 1.7 for every T-LESS object
because the embeddings are L2 normalised.

**It does not survive a different dataset.** On T-LESS it scores 0.348 median absolute log error
against 0.314 for the best constant there, having beaten a constant two to one on PrintCAD. The
ratios differ outright - 0.055 to 0.503 here, 0.377 to 1.561 there - so the model under-predicts
having learned the range it was shown. The conformal band covered 100% against a claimed 80%, but
at 19.8x median width that is uninformative rather than robust, and conformal guarantees do not
survive a change of distribution anyway. Treat the point estimate as PrintCAD-specific.

## A collapsed outline falls back, then refuses

Regularising can flatten a thin silhouette to nothing: H/V snapping plus collinear
merging turn a slim curved bar into a single line, and the pad then yields a null solid
that the report used to call valid. About 3% of outline specs did this.

`spec.traced_outline` checks the traced loops enclose real area (`MIN_OUTLINE_FILL` of
`length_px` squared). If the chosen view collapses, the next most rectangular view is
tried; if every view collapses, `assemble` raises rather than writing a document that
looks fine and contains nothing. The views themselves look healthy in this case - 00238
has aspect 0.12 and rectangularity 0.4 - so the check has to be on the traced loops, not
on the silhouette statistics.

00238 recovers a real four-element outline from another photo; 00332 has no usable view
and now fails with a message telling you to reshoot it square to the face.

## The objective is primitive F1, over a denominator that never moves

`sketch_score.score_one` returns `primitive_f1`: drawn primitives matched against the ideal
sketch's, type by type, so over-drawing and under-drawing cost the same. Region IoU is
secondary and blind to missing features - it scored a scalloped disc 0.95 when the pipeline
drew a plain circle.

Every mean needs its denominator pinned to the set you meant, with a missing part scoring
zero rather than vanishing. This was got wrong four times in one session: a threshold search
found a configuration scoring 0.85 by making 235 of 269 parts fail to produce a sketch at
all, and F1 correlated -0.87 with how many parts were scored. `tools/tune_thresholds.py`
computes the denominator once, up front, from the ids file.

## Fit on the tuning split, confirm on the test split, never the reverse

`data/tune_ids.txt` and `data/test_ids.txt` are disjoint by geometry group with a fixed seed,
and `tools/freeze_split.py` refuses to redraw them. The eight thresholds in the pipeline were
fitted by random search on a slice of the tuning set and confirmed once on the test set:
+0.045 [+0.026, +0.065]. Every threshold they replaced had been chosen by eye on five or ten
parts, and several had to be walked back once the full set disagreed.

The sensitivity ranking from that search is worth as much as the gain: periodicity prominence
matters most, then the repeated-contour tolerance, then the peak floor. The angle snap
tolerance and the periodicity amplitude do not matter at all - do not spend time on them.

## Compare learned components only on parts they were not trained on

Both depth heads are fitted on all 1908 photographed parts, which is the set the bench scores.
Served in-sample the silhouette GBM reaches median x1.16 against its own out-of-fold x1.67 -
it is reciting - and beats a head that generalises better. That reading caused a good change to
be reverted for two hours. Retrain without a held-out third and bench on that third:
`data/depth_holdout_ids.txt` with the `_heldout` model files.

## An oracle here is worth about a tenth of its face value

Six ideas looked large as a best-of-K ceiling and collapsed when a selector had to find them
without the answer: view choice +0.082 became +0.004 end to end, nine view-and-mode hypotheses
+0.081 with the pipeline already picking the best 61% of the time, a per-part simplification
tolerance +0.059 became +0.004, tilt +0.090 became 11%, rotational symmetry exact on synthetic
gears and nothing on photographs, and a bigger head on the same features +0.000.

Size the prize with an oracle before building the machinery, and discount it heavily. Part of
every such ceiling is argmax picking up noise: on parts whose three photographs are
geometrically interchangeable, an oracle over them still "gains" +0.045.

## Diff the code snapshots before believing a delta

`runs/<name>/code` holds the source each run used. A depth A/B looked like a null until the
snapshots showed a second change had landed between the arms. Any surprising result gets
`diff -rq runs/a/code runs/b/code` before it gets an explanation.

## The dataset is a curriculum, and the bottom rung is not solved

PrintCAD is 1908 single-extrusion printed parts: median four primitives, no holes, 7.9 mm
thick. Work up the ladder, and check `tools/tiers.py` before choosing what to fix. As of this
writing the 1-4 primitive tier is 531 parts scoring 0.618 while drawing 3.78 times too many
primitives - a rectangle comes out as fifteen segments. Over-drawing on simple parts is the
largest single block of loss in the dataset and it is not a perception problem.
