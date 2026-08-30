# photo2fcstd

Photos of a physical part in, a parametric FreeCAD document out: constrained sketches bound to a
`params` spreadsheet, not a mesh. Every dimension is a named cell you can edit; change `mm_per_px`
and the whole body rescales.

    photo2fcstd IMG_3164.HEIC IMG_3166.HEIC --length-mm=26.57 --out=camera.FCStd

## How it works

1. **Cut out** — RMBG-2.0 mattes the part (shadows and same-hue backgrounds defeat colour rules).
   Masks are cached in `~/.cache/photo2fcstd/masks`, so a re-run costs nothing.
2. **Trace** — the silhouette is rotated onto its principal axis, symmetrised if it is nearly
   mirror-symmetric, and its contours are extracted with holes (inner contours over 0.2 % of the
   outline area: bolt holes on a pulley are 0.3 %).
3. **Regularise** — contours become ideal geometry, never pixel polygons. A loop that fits an
   ellipse becomes a `Circle` (an ellipse on a flat face is a circle seen at an angle); runs
   between corners that fit an arc become `ArcOfCircle`; everything else is a line snapped
   Horizontal/Vertical within 12° and merged when collinear. Concentric arcs share a centre;
   joins carry intent (Tangent, Perpendicular, Coincident). Dimensions round to 0.5 mm once a
   scale is known.
4. **Choose how the part is made** — see the modes below.
5. **Build** — `build.py` runs under `FreeCADCmd`, writes the `params` sheet, the sketches, the
   pad/revolve/pocket, and reports `solve`, redundant and conflicting constraints per sketch.
   Every mode produces fully constrained sketches (`solve=0`).

## Modes

| mode | chosen when | sketch |
|---|---|---|
| `stations` | two elevations of a stepped or turned part | two width-station polylines, padded symmetric and intersected (`Boolean Common`) |
| `profile` | one silhouette is clearly not a rectangle (bracket, bent sheet, arc) | that outline padded by a depth taken from the plain elevation |
| `plan` | every photo shows the same face: a plate | the outline padded by `depth`, which plan photos cannot show — set it with `--thickness-px` |
| `revolve` | the roundest photo fits an ellipse (aspect > 0.8, residual < 2 %) | a half profile revolved 360°, then the hole circles pocketed through |

`--mode=` forces one. Thresholds live in `thresholds.py`, each fitted on the 200-part benchmark.

## What the photos cannot tell you

A single uncalibrated photo has no scale and no metric perspective. Consequently:

- **Scale** comes from `--mm-per-px`, `--length-mm`, or one caliper reading typed into the sheet.
  Shooting on the ChArUco target and running `rectify.py` first gives an exact homography and
  0.05 mm/px.
- **Thickness of a plate seen face-on** is unobservable; the sheet says so in the parameter note
  and `--thickness-px` overrides the guess.
- **Perspective** cannot be undone from the part alone — the vanishing-point method on silhouette
  corners is ill-conditioned at phone distances (one part's three photos implied aspects 0.63,
  0.22 and 1.34). Use the target.

## Benchmark

    photo2fcstd-bench v11 --jobs 8        # 200 PrintCAD parts, real photos, ~4 min

Four stages: cached segmentation, parallel spec generation, one batched FreeCAD process for all
models, parallel scoring. Results land in `runs/<name>/` with the code snapshot that produced them.

Scale-free voxel IoU against the STEP-derived ground truth, 200 hand-held photo sets:

| | mean IoU |
|---|---|
| stations (103 parts) | 0.446 |
| plan (29) | 0.420 |
| revolve (43) | 0.437 |
| profile (25) | 0.387 |
| **all 200** | **0.433**, median 0.440, 85 above 0.5, 0 failures |

`photo2fcstd-score truth.stl model.stl` reports one pair; `overlay.py` draws the aligned overlay
(grey both, red missing, blue extra) which separates shape error from an unobservable-thickness
guess; `primitives_score.py` compares the sketch's primitives against the ideal ones extracted
from the STEP files by `ideal_sketches.py`.

## Layout

    src/photo2fcstd/
      analysis.py     photo -> one view record (mask, outline, ellipse, stations)
      modes.py        which modelling mode, and the geometry each needs
      spec.py         views -> the JSON spec the builder consumes
      build.py        runs under FreeCADCmd: sheet, sketches, pad/revolve/pocket
      trace.py        segmentation, contours, primitive fitting
      score.py        voxel IoU, 24 rotations, inertia-frame alignment
      overlay.py      aligned overlay with missing/extra
      bench.py        the four-stage benchmark
      thresholds.py   every empirical constant, one place
    tests/            pytest: trace, mode selection, scoring, FreeCAD build

## Install

    uv pip install -e .          # add --group dev for pytest
    pytest -q                    # 27 tests; dataset and FreeCAD tests skip if absent

`FREECADCMD` points at the FreeCAD binary, `P2F_DATA` at the PrintCAD dataset.
