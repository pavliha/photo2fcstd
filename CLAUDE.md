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

`vastai` is installed and authenticated. Check `vastai show user --raw` for the balance
before planning anything - it is small, and a top-end card empties it in under an hour.

**Know which part is actually slow before renting.** In this repo the GPU work is
narrow:

| job | bound by | rent a GPU? |
|---|---|---|
| RMBG segmentation of the photo set | GPU | yes - 71 min locally for 5,713 photos, minutes on a 4090/5090. Already cached, so only for a new dataset or a different matting model |
| `synth.py` data generation | CPU (OpenCV rasterise + trace) | only for the vCPU count - 133 s for 13.7k samples on 8 cores, ~6x faster on a 48-core box |
| `curvenet.py` training | trivial - 195k params, seconds an epoch | no |
| the benchmark, tracing, scoring, FreeCAD | CPU | no |

So a bigger card does not make the current work finish sooner. Rent when the model
grows into something real (an image-to-CAD sequence model), when re-segmenting a new
photo set, or when generating synthetic data at a scale where vCPU count dominates -
in that last case sort on `cpu_cores`, not `dlperf`.

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
