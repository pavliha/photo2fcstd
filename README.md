# photo2fcstd

Photos of a physical part in, a parametric FreeCAD document out: constrained sketches bound to a
`params` spreadsheet, not a mesh. Every dimension is a named cell you can edit; change `mm_per_px`
and the whole body rescales.

    photo2fcstd IMG_3164.HEIC IMG_3166.HEIC --length-mm=26.57 --out=camera.FCStd

The spreadsheet it writes is in millimetres, one named cell per feature, each with a note saying
where the number came from and whether to trust it:

    scale       1.0     multiplies every dimension below; change it to rescale the whole part
    front_w1    12.0    front width of station 1 (mm)
    front_z5    26.5    front station boundary 5 (mm)
    depth       3.2     plate thickness: NOT visible when every photo shows the same face

Type a caliper reading straight into a cell. Without `--length-mm`, `--mm-per-px` or `--rectify`
there is no scale to apply, so the cells hold pixels and say so &mdash; set `scale` to mm/px later and
the whole body follows.

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

## Metric capture: the ChArUco target

Two commands use the printed target (`make_target.py`) to get what photographs alone cannot give.

    photo2fcstd part1.jpg part2.jpg --rectify --out=part.FCStd
    photo2fcstd-carve shot*.jpg --out=part.FCStd --voxel-mm=0.4

`--rectify` finds the board in each photo, flattens the perspective onto the board plane and takes
the scale from it (0.05 mm/px, ~0.03 mm reprojection error), so no `--length-mm` is needed.

`photo2fcstd-carve` is the multi-view path. The board gives every photo a pose *and* millimetres, so
the part is space-carved in real space: calibrate intrinsics across the views, solvePnP per view,
carve a voxel grid against the silhouettes, then the horizontal cross-section becomes the sketch and
the carved height becomes the depth &mdash; **measured, not guessed**. On a synthetic 20&times;12&times;6 mm box
from 12 views it builds 20.0 &times; 11.5 &times; 7.5 mm, fully constrained.

The visual hull is a superset by construction, so extents come out 0&ndash;2 mm large and shrink as views
are added; concave features (a bowl's interior) are invisible to silhouettes and still need the
sketch modes. Shoot 10&ndash;20 photos around the part including low, grazing angles &mdash; those are what
pin the height.

## Depth, when only one face is visible

`modes.outline_depth` takes the first answer that is actually evidence:

| in order | source | note in the sheet |
|---|---|---|
| `--thickness-px` | you measured it | "thickness from --thickness-px" |
| plan mode | a constant 7.5 % of length &mdash; every photo shows the same face, so nothing in them bounds it | says NOT visible, set from a caliper |
| profile with a plain elevation | that elevation's width, scaled | names the photo it came from |
| profile without one | 0.24 &times; the most edge-on view's aspect &times; length | says it is an estimate |

The constants are fitted, and the ordering is measured, not assumed. Replacing the whole table with
the edge-on estimate scores 0.327 on the 53 outline parts against 0.404 for the table; keeping it
only as the last-resort fallback gives 0.405 (200-part mean 0.432). `DEPTH_FROM_ASPECT` is 0.24
because that minimises both the log-error against true thickness (0.993 at 0.24, 1.112 at 0.46) and
the benchmark: 0.24 beats 0.46 by 0.020 on outline parts.

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

## Mode selection, learned (off by default)

Img2CAD-style factorisation: let a model choose the discrete *operation* and keep the continuous
*dimensions* deterministic. `P2F_LEARNED_MODES=1` routes selection through
`mode_model.predict` (RandomForest on per-view shape statistics, trained by
`photo2fcstd-train-modes` on oracle labels from forced-mode runs).

Measured, and the measurement is the point:

| | mean IoU |
|---|---|
| rules | 0.433 |
| learned, evaluated on its own training parts | 0.465 |
| learned, on 100 parts it never saw | **0.393** |

The in-sample gain is leakage. With 100 training parts the classifier does not generalise, so the
switch stays off. The oracle gap it targets is real (0.094); closing it needs an order of magnitude
more labels, which is what the full-dataset label run is for.

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
