# Shooting a part

Every number here was measured, and the measurement is named so you can argue with it.

## Once

Print the target and check it:

```
photo2fcstd-target                 # writes charuco_target.pdf, A4, at the nominal square size
photo2fcstd-target --verify        # confirms the printer did not scale it
```

If your printer scales the page, measure one square with calipers and set `P2F_SQUARE_MM` to what
you actually got. **A wrong square size makes every distance wrong and nothing will complain** - the
poses stay self-consistent, so the part comes out the wrong size with no error anywhere.

Tape the target flat. A curled sheet is the one failure the reprojection check catches for you.

## Each part

Stand the part near the middle of the target, on the board, not beside it.

**Take sixteen photographs, walking right around it.** Every measurement in `docs/results.md` used
sixteen views. Eight is the working minimum. What matters is not the count so much as leaving no arc
unwatched: a visual hull keeps whatever no silhouette contradicts, so a side you never photograph
stays solid.

**Vary the height, and get low.** Aim for a spread from about 15 degrees above the table up to 55.
The low frames are the ones that matter and they are the ones people skip:

| lowest camera | what the height of a 20 mm wide part is uncertain by |
|---|---|
| 15 degrees | 2.7 mm |
| 30 degrees | 5.8 mm |
| 45 degrees | 10.0 mm |

Height is bounded by the *lowest* view alone - `width/2 x tan(elevation)` - so twelve high frames
and no low ones give you a tall smear no matter how many you take. Below about 15 degrees the target
stops being detectable, which is the floor on this and the reason a depth camera is worth more here
than more photographs.

**Do not worry about the background, focus, or a shaky hand.** Matting is nearly free: eight pixels
of boundary wander - far worse than the segmenter produces - still leaves carving ahead of a single
photograph. Sixteen views average that error out.

**Do worry about the target being visible.** A pose is solved per photograph from the board alone.
A frame where the part hides the target is a frame that does not count.

## Before you carve

```
photo2fcstd-preflight photos/ --width-mm=20
```

It says whether the capture is usable and, when it is not, which of the above to fix. It exits
non-zero on a problem so you can put it in a script. Checking takes seconds; discovering the same
thing from a bad model takes minutes and tells you less.

Then:

```
photo2fcstd-carve photos/ --out part.FCStd
```

## What to expect

Carving from a board should reach about **0.72 sketch IoU** against **0.64** for the single-photo
path, on parts whose cross-section is roughly constant along one axis. Those are the parts a single
sketch describes at all; the spec carries a warning when yours is not one.

Two honest caveats. Pose is measured (0.016 degrees on rendered board photographs, against a budget
of 2) and matting is measured, but **no part of this has been run on a real photograph of a real
board** - the datasets have no such images, which is the whole reason the rig is worth building.
And a visual hull cannot see a cavity that never breaks the silhouette from any angle: through-holes
come out, blind pockets do not.
