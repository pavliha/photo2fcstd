"""Check a set of photographs before carving from them, and say what to reshoot.

Every threshold here was measured rather than guessed:

- the board stops being detectable below about 15 degrees of elevation, and a silhouette taken that
  low bounds a part's height no tighter than `width/2 * tan(elevation)`
- pose error costs 0.033 of sketch IoU per degree and carving falls behind a single photograph at
  about 5 degrees; a detected board resolves pose to 0.016
- matting is nearly free: 8 px of correlated boundary wander still beats the photo path
- carving needs at least three posed views and was measured with sixteen
"""
import os

import numpy as np

MIN_VIEWS = 8
GOOD_VIEWS = 16
MIN_ELEVATION_DEG = 15.0
LOW_ELEVATION_DEG = 25.0
MAX_REPROJECTION_PX = 3.0
MAX_AZIMUTH_GAP_DEG = 120.0
MIN_MASK_FRACTION = 0.005


def elevation_of(view):
    """How far above the board the camera sits, in degrees.

    `capture.pose` solves the board's own frame, whose z axis points away from the camera - the
    printed frame runs x right and y down, which is left-handed in 2D - so the eye's z comes out
    negative for a camera above the table and the sign has to be flipped.
    """
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(view["rvec"], float))
    eye = -R.T @ np.asarray(view["tvec"], float).reshape(3)
    r = float(np.linalg.norm(eye))
    return float(-np.degrees(np.arcsin(np.clip(eye[2] / max(r, 1e-9), -1.0, 1.0))))


def azimuth_of(view):
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(view["rvec"], float))
    eye = -R.T @ np.asarray(view["tvec"], float).reshape(3)
    return float(np.degrees(np.arctan2(eye[1], eye[0])) % 360.0)


def largest_gap(angles):
    """The widest direction the cameras never look from.

    What matters for a visual hull is not how far apart the extremes are but whether any arc is
    unwatched: material the silhouettes never contradict survives. Four cameras at 90 degree
    spacing leave a 90 degree gap and are fine; four bunched on one side leave 300 and are not.
    """
    if len(angles) < 2:
        return 360.0
    a = np.sort(np.asarray(angles, float) % 360.0)
    return float(np.diff(np.concatenate([a, [a[0] + 360.0]])).max())


def inspect(paths, segment_fn=None):
    """Calibrate across the set first, exactly as `carve.from_photos` does.

    A pose solved with a guessed focal length reprojects at 4 to 6 px on images whose true pose is
    accurate to 0.05 degrees, so checking reprojection without calibrating first measures the guess
    rather than the capture.
    """
    from photo2fcstd import capture
    from photo2fcstd.trace import load, segment_photo
    segment_fn = segment_photo if segment_fn is None else segment_fn
    images = []
    for p in paths:
        try:
            images.append(np.asarray(load(p)))
        except Exception:
            images.append(None)
    try:
        cal = capture.calibrate([i for i in images if i is not None])
    except Exception:
        cal = None
    K = cal["K"] if cal else None
    dist = cal["dist"] if cal else None
    rows = []
    for p in paths:
        row = {"path": p, "board": False, "mask": None, "elevation": None,
               "azimuth": None, "reprojection_px": None, "corners": 0}
        try:
            image = images[paths.index(p)]
            got = capture.pose(image, K, dist)
            if got is not None:
                row.update(board=True, elevation=elevation_of(got), azimuth=azimuth_of(got),
                           reprojection_px=got.get("reprojection_px"), corners=got.get("corners", 0))
        except Exception:
            pass
        try:
            m = segment_fn(p)
            row["mask"] = float(m.sum()) / float(m.size) if m is not None else None
        except Exception:
            pass
        rows.append(row)
    return rows


def verdict(rows, part_width_mm=None):
    """What is wrong with this capture, most important first, and what to do about it."""
    posed = [r for r in rows if r["board"]]
    problems, notes = [], []

    if len(posed) < MIN_VIEWS:
        problems.append("only %d of %d photographs show the target well enough to solve a pose; "
                        "carving needs at least %d and works best with %d"
                        % (len(posed), len(rows), MIN_VIEWS, GOOD_VIEWS))
    elif len(posed) < GOOD_VIEWS:
        notes.append("%d posed views; %d were used for every measurement in docs/results.md"
                     % (len(posed), GOOD_VIEWS))

    thin = [r for r in rows if r["mask"] is not None and r["mask"] < MIN_MASK_FRACTION]
    if thin:
        problems.append("the part is tiny or missing in %d photographs; move closer or check the "
                        "background separates from it" % len(thin))

    if posed:
        elev = np.array([r["elevation"] for r in posed], float)
        low = elev.min()
        if low > LOW_ELEVATION_DEG:
            problems.append("the lowest camera is %.0f degrees above the table. Height is bounded by "
                            "the lowest view: at %.0f degrees a %s part's height is uncertain by "
                            "%s. Shoot some frames nearer %.0f degrees"
                            % (low, low, "%.0f mm wide" % part_width_mm if part_width_mm else "given",
                               "%.1f mm" % (part_width_mm / 2 * np.tan(np.radians(low)))
                               if part_width_mm else "width/2 x tan(elevation)", MIN_ELEVATION_DEG))
        elif low < MIN_ELEVATION_DEG:
            notes.append("lowest view %.0f degrees, below the %.0f the target usually survives - "
                         "those frames may not have posed" % (low, MIN_ELEVATION_DEG))
        if elev.max() - elev.min() < 20.0:
            notes.append("all cameras sit within %.0f degrees of the same height; a spread helps the "
                         "hull close" % (elev.max() - elev.min()))

        gap = largest_gap([r["azimuth"] for r in posed])
        if gap > MAX_AZIMUTH_GAP_DEG:
            problems.append("there is a %.0f degree arc the cameras never look from; a visual hull "
                            "keeps whatever no silhouette contradicts, so walk right around the part"
                            % gap)

        err = np.array([r["reprojection_px"] for r in posed if r["reprojection_px"]], float)
        if len(err) and np.median(err) > MAX_REPROJECTION_PX:
            problems.append("the target reprojects at %.1f px, so the poses are unreliable; check the "
                            "board is flat and the printed square size matches P2F_SQUARE_MM"
                            % np.median(err))
    return problems, notes


def report(paths, part_width_mm=None):
    rows = inspect(paths)
    posed = [r for r in rows if r["board"]]
    print("preflight on %d photographs\n" % len(rows))
    print("  %-34s %s" % ("target solved a pose in", "%d of %d" % (len(posed), len(rows))))
    if posed:
        elev = [r["elevation"] for r in posed]
        err = [r["reprojection_px"] for r in posed if r["reprojection_px"]]
        print("  %-34s %.0f to %.0f degrees" % ("camera elevation", min(elev), max(elev)))
        print("  %-34s %.0f degrees" % ("widest unwatched arc",
                                        largest_gap([r["azimuth"] for r in posed])))
        if err:
            print("  %-34s %.2f px median" % ("target reprojection", float(np.median(err))))
        print("  %-34s %.1f%% of frame median" % ("part occupies",
              100 * float(np.median([r["mask"] for r in rows if r["mask"] is not None] or [0]))))
    problems, notes = verdict(rows, part_width_mm)
    if not posed:
        from photo2fcstd import tilt_model
        extra_p, extra_n = tilt_model.square_check(paths)
        problems, notes = problems + extra_p, notes + extra_n
    print()
    for p in problems:
        print("  RESHOOT: %s" % p)
    for n in notes:
        print("  note:    %s" % n)
    if not problems:
        print("  usable. Carve with: photo2fcstd-carve %s" % os.path.dirname(paths[0] or "."))
    return rows, problems, notes


def run():
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if not args:
        raise SystemExit("photo2fcstd-preflight <photo|directory>... [--width-mm=25]\n"
                         "checks a capture before you carve from it")
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += [os.path.join(a, f) for f in sorted(os.listdir(a))
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))]
        else:
            paths.append(a)
    if not paths:
        raise SystemExit("no photographs found")
    _, problems, _ = report(paths, float(opt["width-mm"]) if "width-mm" in opt else None)
    raise SystemExit(1 if problems else 0)
