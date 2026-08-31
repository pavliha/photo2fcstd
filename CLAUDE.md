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

## Region IoU is blind to things that matter

It cannot see primitive type (an arc and the chords approximating it cover the same
area) and it cannot see small holes (they carry almost no area). A change can be right
and move IoU by 0.000. Report `loops exact` and `curve fraction` alongside it, and
look at a render before believing either.

## Measure before changing a threshold, and measure builds too

Every empirical constant is in `thresholds.py`. They are jointly tuned and most sit at
a local optimum, so a lone change is usually neutral or worse. When you touch one:

- A/B on the **same** parts, regenerating specs for both arms.
- Report null solids and not-fully-constrained sketches, not just IoU. Several changes
  this repo has already rejected improved a statistic and broke models.
- 160 parts is enough to see a trend, too few to trust a 1-vs-2 difference.

Results already established, so nobody re-runs them: loosening the arc gates closes
the curve gap (0.45 to 0.60 against an ideal 0.59) but buys no accuracy and breaks 12
of 160 models. Deriving arc direction correctly is neutral at the shipped gates.
Learned mode selection is worth ~0.001 of sketch IoU.

## Where the error actually is

Against the orthographic silhouette, on trusted parts whose photo shows the extrusion
face: capture (viewpoint plus matting) costs 0.212, our tracing and regularisation
0.112, and the fact that a silhouette includes side walls the base face does not costs
0.089. Threshold work lives in the middle term only. Perspective is the biggest one and
only `capture.py` / `carve.py` address it - a photo of a bar seen edge-on does not
contain its width, and no fitting recovers it.

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
