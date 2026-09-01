# Finishing the reshoot check

`tilt_model.square_check` reports how far off square a photograph was taken, with no ChArUco board
in the frame, and `preflight` calls it whenever no board posed. Shooting square is worth about
+0.13 of sketch IoU and tilt cannot be corrected afterwards, so telling the photographer is the
whole value.

It currently abstains on PrintCAD photographs, because the only head with real skill was fitted on
T-LESS and its gate does not recognise them. The missing input is not code. It is **a few dozen
real photographs whose tilt is known**, and the cheapest source of that is the board this project
already prints - a solved pose *is* an exact tilt label, free.

The board is not needed to use the check. It is needed once, to train it.

## What to shoot

One afternoon, one part, thirty to fifty frames:

1. `photo2fcstd-target` and print it. The centre is deliberately blank so the part sits in it
   without the pattern showing through the silhouette.
0. **Lie the part flat, face up.** The label is the board's own normal, so a part stood on edge is
   silently mislabelled and nothing downstream can detect it.
2. Put a part in the cleared middle. Any part with a flat face is fine; a few different parts is
   better than many frames of one, because the head is held out by part.
3. Shoot deliberately across the range that matters: square on, then roughly 5, 10, 15, 20, 25 and
   30 degrees off the face normal, several azimuths at each. The check's threshold is 8 degrees, so
   the frames near it are the ones that teach it most.
4. Keep the whole board in frame and in focus. `photo2fcstd-preflight <directory>` says whether the
   poses solved.

Repeat for two or three parts and stop. This is a labelling session, not a dataset.

## What happens then

```bash
python -m tools.tilt_board_data <directory>     # poses -> tilt labels, part-only crops, embeddings
python tools/tilt_train.py data/tilt_board.npz  # fit, gate, report held-out error
```

`tilt_train` prints held-out-by-part error against the constant baseline. The bar to clear is the
threshold it will be used at: the T-LESS head reaches 2.63 degrees against a 6.98 constant and 87%
within 5, which is comfortably good enough to warn at 8. Renders reach 7.6 and are not.

If the gate then admits your own photographs - which it should, since they will be the same camera
and the same kind of scene - the check works board-free from that point on, for every photograph
you ever take of a part.

## Why renders could not stand in

Two renderers were tried and both failed, which is why real photographs are the ask:

| trained on | tilt error | constant | within 5 deg |
|---|---|---|---|
| T-LESS photographs | **2.63 deg** | 6.98 | **87%** |
| flat Lambertian renders | 7.62 | 8.81 | 45% |
| Blinn-Phong, specular, several lights, antialiased | 7.96 | 8.81 | 43% |

The second renderer was written specifically because "the renders lack shading" looked like the
explanation. It made things very slightly worse, so that diagnosis was wrong and the gap is not a
photometric knob.
