# What the scores cannot see

Three times in one session a gap in the *measurement* looked like a fault in the
pipeline. This is the deliberate list, so the fourth one is found on purpose.

`sketch_score.region_iou` normalises both polygons to their own bounding box and takes
the best of eight dihedral poses. `score.best_iou` normalises by max extent and takes
the best of 24 rotations with and without a PCA frame. Both were built that way on
purpose - photographs carry no absolute scale or frame, so a pose-free, scale-free
comparison was the only fair one - and everything below follows from it.

| the score cannot see | does it matter | evidence |
|---|---|---|
| **absolute scale** | Yes, but it is handled elsewhere. `mm_per_px` comes from `--length-mm`, a rectified photo or a caliper, and the sheet says which. | by construction |
| **orientation** | No, as it turns out. The frame is arbitrary but *consistent*: 94% of view pairs of the same part land in the same pose. It disagrees with the STEP frame (34% of axes) because STEP files have no shared convention either - thinnest axis is spread 17/24/17 across parts. | `tools/` orientation check, n=58 |
| **position** | No. Everything is centred before comparison and a sketch has no meaningful origin here. | by construction |
| **primitive type** | Partly, and this bit does matter. An arc and the chords approximating it cover the same area, so region IoU is nearly identical - which is why loosening the arc gates raised curve fraction 0.45 to 0.60 and moved IoU by -0.001. Watch `curve_frac_mine` against `curve_frac_ideal` separately. | arc-gate sweep |
| **small holes** | Yes. A bolt hole carries almost no area, so recovering 20 of them on part 00061 moved region IoU by nothing. Watch `loops_mine` against `loops_ideal`. | `MIN_HOLE_FRAC` change |
| **wire-thin shapes** | Yes, badly. Two thin outlines offset by one wire-width barely overlap, so the score is dominated by alignment. 00294 scores 0.055 with a faithful six-line chevron. Parts under about 0.15 aspect need a curve-distance measure, not an area one. | 00294 |
| **which face was photographed** | Yes. Nine of 196 trusted parts are shot edge-on, so the silhouette is the part's millimetre of thickness. They average 0.104 against 0.592 for the rest, and no alternative view exists for any of them. Now flagged by `edge_on_warning`. | sliver audit |
| **whether the reference itself is right** | It was wrong for 37% of parts until `chain_edges` landed. A scrambled ring fills into a blob: 00171's reference read as a solid disc rather than a C-ring, costing that part 0.42. | `chain_edges` |
| **blind features** | Yes, and permanently. Pockets, bosses and counterbores never appear in a silhouette. Part 00239 draws 4 lines against a reference of 48 because its louvre slots are recesses, not holes. No amount of tracing recovers them; carving from grazing views can. | 00239 |

## How to use this

Quote region IoU with at least `loops exact` and `curve fraction` beside it, and say the
sample size. When a number surprises you, check this table before changing any code -
three of the surprises above were the measurement, not the pipeline.
