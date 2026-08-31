# Evaluating cadrille on these parts

[cadrille](https://github.com/col14m/cadrille) (ICLR 2026) is a 2B Qwen2-VL that emits
CadQuery from point clouds, images or text - state of the art on DeepCAD, Fusion360 and
CC3D, Apache 2.0, with SFT and RL checkpoints on HuggingFace. It is the "predict the
program, not the geometry" approach this project would otherwise have to build.

## What was run

Twenty trusted PrintCAD parts, three input arms, all in cadrille's image convention
(4 views, 128 px, 3 px black border):

- **control** - Open3D renders of the truth mesh, cadrille's own training distribution
- **mask** - RMBG silhouettes, pale yellow on white, styled to match
- **photo** - the real photographs, downscaled

## Results

SFT checkpoint, greedy:

| arm | executable solids |
|---|---|
| control | 17/20 generated, 12 executed |
| photo | 13 executed |
| mask | 12 executed |

RL checkpoint, 5 samples per part, control arm:

| | |
|---|---|
| samples that execute | **86%** (up from ~60% SFT - the RL training targets exactly this) |
| parts with at least one valid solid | 18/20 |
| first valid sample | mean IoU **0.024**, median 0.002 |
| oracle best of 5 (picks the best by comparing to truth - an upper bound) | mean IoU **0.034**, median 0.004 |
| parts above 0.5 IoU | **0** |

For comparison this project's pipeline reaches about 0.58 sketch IoU on the same data.

`figures/cadrille_fixed.png` shows the reconstructions: a flat plate becomes a rounded
slab, a cylinder fragments into disconnected sheets, a perforated bracket loses every
hole, a ring becomes a curved trough.

## The likely explanation

cadrille trains on DeepCAD and Fusion360, which are chunky multi-feature mechanical
solids. These parts are 20 mm flat plates, rings and rods - thin, simple, dominated by a
single extrusion. Its reconstructions are consistently too voluminous and too curved,
which is what a model biased toward blocky solids does with a thin plate.

## The caveat, stated plainly

**The control was never reproduced.** The proper check - run their `test.py` and
`evaluate.py` unmodified on their own DeepCAD test split and match their published
numbers - did not run: vast.ai stopped authorising instances with credit at $5.05
against a $5.00 autobill threshold and no payment method on file.

This matters because a harness bug was already found once here: cadrille's camera looks
at `(0.5, 0.5, 0.5)` (their meshes are normalised to the unit cube) and the first run
centred meshes at the origin, so the control arm was mis-framed and its first results
were void. A second such bug cannot be ruled out without the reproduction.

The judgement, not a verified fact: 0.024 is far too low to be a subtle harness
artefact, and the failures are coherent-but-wrong solids rather than garbage, which is
what genuine domain mismatch looks like. Treat "cadrille does not transfer to these
parts" as strongly indicated, control pending.

## Cost

$1.43 across four instances, all destroyed. Two offers went stale (`success: false`,
stuck at `intended_status: stopped`) and one had dead outbound bandwidth with git-lfs
pointers instead of meshes.

## If picking this up again

1. Run the DeepCAD reproduction first. Nothing else is worth doing until the harness is
   validated.
2. The bundle and runner are reproducible from `scratchpad/cadrille_bundle.py` in the
   session; the runner needs `PYTHONPATH` set to the cadrille repo, a `file_name` key in
   each batch item which must then be popped before `generate`, and `mesh_to_image`
   extracted out of `dataset.py` to avoid a pytorch3d dependency.
