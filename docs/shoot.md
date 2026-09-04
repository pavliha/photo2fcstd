# The photo session

One sitting, three jobs, about fifty frames. This is the top-ranked item in PLAN.md: every code
item is done, and these photographs unlock three components that are already built and waiting.
The constraints below are all measured, not guessed - the source is cited on each.

## Job 1: sixteen photos of one part on a patterned surface (C5)

Gates SfM carving on real photographs. Carving reaches 0.782 solid IoU where the photo pipeline
reaches ~0.43, and structure-from-motion removes the ChArUco board it used to need (docs/sfm.md).

- **Surface: strongly textured.** A newspaper, cutting mat or patterned cloth. Not a bare desk:
  at a quarter of the texture contrast, 0 of 16 synthetic images registered, on every part.
- **Sixteen views, not eight.** At 8, only 5 registered. Orbit the part so consecutive frames
  overlap; a full circle at one or two elevations.
- Part thicker than ~2 mm (a 0.25 mm plate carves to nothing from true poses too).
- Hold exposure steady if the phone allows it; motion blur is the untested risk, so brace or use
  a timer.
- Then: `python tools/sfm_real.py <directory> --length-mm <one caliper reading>` - written and
  gated: on rendered frames through its full path it recovers depth/length to 1% at 3.7 mm
  thickness, with the hull's own +0.5-0.8 mm overestimate on thinner parts. It reports the depth
  the archive's photographs cannot contain (tools/depth_consistency.py measured that null).

## Job 2: thirty to fifty frames with the board, tilt known (C1)

Trains the reshoot check so it works board-free forever after. Shooting square is worth +0.13 of
sketch IoU and tilt cannot be corrected afterwards (docs/tilt-labels.md has the full protocol).

- `photo2fcstd-target`, print it, part lies **flat, face up** in the blank centre - a part on its
  edge is silently mislabelled.
- Square on, then ~5, 10, 15, 20, 25, 30 degrees off the normal, several azimuths each. The
  check's threshold is 8 degrees, so frames near 8 teach it most.
- Whole board in frame and in focus. Two or three different parts beat many frames of one - the
  head is held out by part.
- Check as you go: `photo2fcstd-preflight <directory>` says whether the poses solved.
- Then (C2): `python -m tools.tilt_board_data <directory>` and `python tools/tilt_train.py
  data/tilt_board.npz`. The bar: beat the constant baseline held out by part, with a gate that
  accepts PrintCAD-style photographs.

## Feeding reshot photos back in

`P2F_RESHOOT_DIR=<dir>` makes `bench.photos_of(part)` prefer `<dir>/<part>/*.jpg` (or `.png`) over
the dataset. So a reshot 00141 goes in `<dir>/00141/`, and any bench, A/B or gallery command scores
it instead of the rim-shot original - no dataset edit. The sixteen-view carve path is separate:
`python tools/sfm_real.py <photo_dir> --length-mm <caliper>` takes its directory directly.

## Job 3: one square-on frame each for the twelve discs

These parts are at primitive F1 0.000 because all three photographs show the rim; no code change
reaches them (PLAN.md, "the bottom rung is circles"). One face-on photo each:

    00141  00208  00214  00235  00636  00720
    01088  01120  01298  01508  01665  01759

Lay the disc flat, shoot from directly above. That is the whole requirement.

## What this buys, in the plan's own numbers

| job | unlocks | measured value |
|---|---|---|
| 1 | carving without a board | solid IoU ~0.43 -> 0.78 on the parts it applies to |
| 2 | board-free reshoot warnings | shooting square is +0.13 sketch IoU |
| 3 | twelve parts off the floor | 4.9% of the test set at 0.000 -> scoreable |
