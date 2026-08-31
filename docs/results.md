# Measured state

Every number here is from this repository's own benchmarks, on the PrintCAD photo sets.
Quote the sample size with the number; several of these changed when the sample grew.

## The ground truth is only partly usable

`data/printcad_ideal_sketches_all.json` holds each part's real sketch, extracted from
its STEP file by `ideal_sketches.py`.

| | count |
|---|---|
| parts with photos | 1,907 |
| with 2D sketch geometry | 1,892 |
| genuinely a prism (face is the extrusion base) | 1,149 (61%) |
| passing `sketch_score.trustworthy` | 1,047 (55%) |

The prism count was **982 (52%)** until the extractor was fixed: it picked each part's
*largest planar face*, which for 167 parts is not the face the part is extruded from.
Those sketches were being scored against the wrong reference entirely.

## Where the pipeline stands (master)

On trusted parts, measured with `sketch_score.py`:

| | value |
|---|---|
| parts that emit a sketch at all | **100%** |
| region IoU over all trusted parts | **0.569** |
| region IoU over the parts that do draw | 0.569 |
| loop count exactly right | 81% |
| curve fraction, ours against ideal | 0.43 / 0.51 |
| solid IoU | ~0.44 |

Every part now draws; see `docs/decisions.md` for the trade that bought it.

### Two things the region metric cannot see

**Scrambled reference loops, now fixed.** A STEP wire lists its edges in topological
order and each may run either way, so concatenating them as stored can scramble the
ring. 698 of 1892 parts (37%) were affected, 315 of them trustworthy, and shapely filled
the scrambled ring into a blob - part 00294 is a thin wire staple whose reference read as
31% filled. `sketch_score.chain_edges` now walks the edges end to end. Correcting it
moved the headline from 0.553 to **0.569**, so the bias was mild, but it was real.

**Wire-thin parts.** Two thin outlines offset by one wire-width overlap almost nowhere,
so region IoU is dominated by alignment and cannot reward a correct-looking sketch.
00294 scores 0.055 with a faithful six-line chevron. A curve-distance measure would suit
those parts; region IoU does not.

## The error budget

Against the orthographic silhouette, on trusted parts whose photo shows the extrusion
face (n=94):

| stage | IoU | cost | fixable by |
|---|---|---|---|
| perfect | 1.000 | | |
| photo mask, before any tracing | 0.788 | **0.212** | capture |
| our tracing and regularisation | 0.675 | **0.112** | code |
| against the base face rather than the silhouette | 0.586 | **0.089** | multi-view |

### The capture term is viewpoint, not perspective

`tools/split_capture.py` renders each truth mesh orthographically and in perspective
from the same viewpoint and compares both against the real photo mask:

| camera tilt off the face normal | orthographic | with perspective | real photo |
|---|---|---|---|
| 0 degrees | **0.946** | 0.930 | |
| 8 degrees | 0.834 | 0.835 | |
| 15 degrees | 0.799 | 0.795 | |
| 30 degrees | 0.801 | 0.798 | |
| measured photos | | | **0.814** |

- **Perspective costs 0.003 to 0.016.** A phone at arm's length from a 20 mm part is
  already nearly orthographic.
- **Matting costs about 0.02.** Photos score 0.814 where matting-free synthetic views at
  the same apparent tilt score 0.834. RMBG is not the problem.
- **Viewpoint tilt is the whole term.** Eight degrees off-axis costs 0.11.

**Shooting square to the face is worth roughly +0.13 and requires no code.** It is the
cheapest improvement available to this project.

## Carving is roughly twice as accurate

`carve.py` needs the ChArUco board, so no PrintCAD photo can reach it.
`carve_check.py` drives it with synthetic views of a truth mesh instead - exact poses,
rendered silhouettes - which tests the geometry without any capture.

16 views at 0.25 mm voxels, 40 trusted parts:

| | volumetric IoU | extent error |
|---|---|---|
| all parts | 0.736 mean, **0.810 median** | +0.71 / +0.35 / +0.85 mm |
| at least 4 voxels thick (n=30) | **0.791** | +4.3% / +4.8% / +30.5% |
| thinner than 4 voxels (n=10) | median true thickness 0.43 mm | not resolvable |

For comparison the photo pipeline reaches about 0.43 solid IoU and its best possible
mode choice 0.527. Carving roughly doubles that **and measures the depth** instead of
predicting it.

Size the voxel to about a quarter of the smallest feature. The default `VOXEL_MM = 0.4`
cannot resolve anything under roughly 1.6 mm.

## Depth

Depth cannot be a constant: the true depth/length ratio spans 0.055 to 0.503. Measured
on a test slice that neither training nor calibration saw:

| | median \|log\| error | within 2x |
|---|---|---|
| best constant | 1.045 | 35% |
| geometric estimate (0.46 x edge-on aspect) | 1.118 | 28% |
| **learned regressor** | **0.435** | **63%** |

End to end on 41 held-out outline parts, solid IoU rose **0.404 to 0.483**.

The interval is conformally calibrated, so its coverage is guaranteed: **83% of true
depths fall inside the 80% band** (plain quantile regression covered only 55%). But the
median band spans about **10x**, and only a couple of percent of parts get a band
tighter than 2x. Depth is not in an uncalibrated photo of a part seen face-on. The
`params` note now states the range and whether the number is worth building from.

## Figures

- `figures/results.png` - photo, sketch and solid for four parts, including a `stations`
  part that produces no drawing.
- `figures/current_results.png` - six parts spanning the quality range, with what we draw
  beside what the STEP file contains.
- `figures/cadrille_fixed.png` - the cadrille evaluation, see `docs/cadrille.md`.
