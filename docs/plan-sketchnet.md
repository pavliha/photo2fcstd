# Plan: predict the sketch directly, as a set of primitives

## Why this and not the other things

Ten learned attempts in this repo have lost. Nine of them share one cause, and it took a long time
to see: they all needed a **correspondence** between what we traced and what the part actually is.
Per-point curve labels needed it. Corner heatmaps needed it. Constraint prediction needed it and
died at 62% alignment, and adding the evidence the tracer had discarded moved nothing.

A model that maps a raster straight to primitives never needs that correspondence. The STEP files
already hold exact primitive sequences for 1047 trusted parts - 5549 lines, 1778 arcs, 752 circles,
1782 b-splines - and `synth.py` already renders those sketches under random homographies to imitate
a photograph's foreshortening. The supervision is exact and free; only the output head is wrong.

This is what PICASSO (WACV 2025) does, and its finding that a **feed-forward set predictor beats an
autoregressive one** (0.751 against Vitruvion's 0.537) is the reason not to build a sequence model.

## What it predicts

A fixed set of 24 slots - 92% of trusted sketches have 24 primitives or fewer - each slot emitting

    presence, type (line / arc / circle), and its parameters, normalised to the sketch box

Set prediction, matched to the truth by Hungarian assignment on a geometric cost, so the model is
not punished for emitting the same primitives in a different order. Order is the thing an
autoregressive model has to learn and does not need to.

## Gates, in order, each one cheap enough to stop at

**Gate 1 - can it beat the tracer on rendered sketches?** Train and evaluate on `synth` renders,
where the input is clean and the labels are exact. Compare to running `trace.primitives` on the same
renders. If a model with perfect supervision and no capture noise cannot beat the geometric tracer
in its own best case, nothing downstream matters. **Stop if it loses here.**

**Gate 2 - does it survive the homography?** Same, with the tilt augmentation on. This is the
distribution shift we can actually simulate.

**Gate 3 - does it transfer to photographs?** Feed traced silhouettes from real PrintCAD photos and
score with `sketch_score` against the trusted subset, quoting the trivial-circle baseline and the
discriminating subset as everything else here does. This is where the two dataset-transfer collapses
measured today - the axis classifier at 89% to 29%, the depth model beaten by a constant - say the
risk is. **Expect this to be the one that fails.**

**Gate 4 - end to end.** A/B on full specs with both arms regenerating, FreeCAD null-solid check, an
out-of-distribution row in `tools/ood_audit.py`, and a `section_constancy`-style refusal if the model
is confident where it should not be.

## What would make it worth shipping

Beating 0.638 sketch IoU is not the bar. **28 to 32% exact primitives is the bar**, because that is
what a person editing the result feels, and because region IoU cannot see an arc drawn as chords -
the reason three separate attempts to close the curve gap looked neutral while making drawings
worse. A model that reaches 45% exact at the same IoU is a success; one that gains 0.02 IoU and
loses exact primitives is the arc-gate trade again and gets reverted like the others.

## Cost, honestly

Roughly 195k parameters was enough for the 1D models here; this needs a small CNN encoder and a set
head, so low millions, minutes an epoch on the laptop. Renders are CPU-bound and `synth.build`
already exists. No GPU rental: the earlier note stands that a GPU only helps the stage that is
GPU-bound, and this one is not - the bottleneck is generating renders on 6 cores.

The expensive part is not compute, it is that gates 1 to 3 each need a full evaluation loop, and
this repo's record says the honest prior is that it fails at gate 3.
