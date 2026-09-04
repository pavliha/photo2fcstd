import json
import os
import sys

import numpy as np
from scipy import ndimage
from PIL import Image, ImageFile, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

from photo2fcstd import thresholds as th
from photo2fcstd import thresholds as th_module
from photo2fcstd.rectify import PPMM

WARM = 0.005
MIN_STEP_MM = 0.7
STEP_GRAD = 1.6
TAIL = 0.25


def load(path):
    if not os.path.exists(path):
        from photo2fcstd.errors import CaptureError
        raise CaptureError("cannot find %s" % path)
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def largest(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("no object found")
    return lab == int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))) + 1


def segment(a):
    core = largest(ndimage.binary_opening(a.max(axis=2) < 0.32, np.ones((9, 9))))
    ys, xs = np.where(core)
    h, w = core.shape
    pad = int(0.12 * (xs.max() - xs.min()))
    span = ys.max() - ys.min()
    win = np.zeros_like(core)
    win[max(0, ys.min() - int(0.06 * span)):min(h, ys.max() + int(0.70 * span)),
        max(0, xs.min() - pad):min(w, xs.max() + pad)] = True
    m = ((a[..., 0] - a[..., 2]) < WARM) & win
    return largest(ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((13, 13)))))


def otsu(x):
    hist, edges = np.histogram(x, bins=256)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.cumsum((hist * mids)[::-1])[::-1] / np.maximum(w1, 1))
    between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[1:]) ** 2
    return float(mids[int(np.argmax(between))])


def segment_auto(a):
    h, w = a.shape[:2]
    border = np.concatenate([a[:h // 20].reshape(-1, 3), a[-h // 20:].reshape(-1, 3),
                             a[:, :w // 20].reshape(-1, 3), a[:, -w // 20:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(a - bg, axis=2)
    m = dist > otsu(dist)
    m = ndimage.binary_opening(m, np.ones((7, 7)))
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((15, 15))))
    m[:2, :] = m[-2:, :] = m[:, :2] = m[:, -2:] = False
    return largest(m)


_RMBG = {}


from photo2fcstd.settings import PACKAGE_ROOT, cache_dir


MASK_VERSION = "3"
TRIM_APPENDAGE = float(os.environ.get("P2F_TRIM_APPENDAGE", 0.0))


def cache_identity(path):
    """What names a photo for the cache: its place in the project, not its place on disk.

    Keying on the absolute path meant moving the dataset threw away every one of 5,713
    segmented masks, none of which had changed. Paths inside the project are recorded
    relative to it, so the tree can be moved or checked out anywhere and the cache follows.
    """
    full = os.path.realpath(path)
    root = os.path.realpath(PACKAGE_ROOT) + os.sep
    return os.path.relpath(full, root) if full.startswith(root) else full


def cached_mask(path, version=MASK_VERSION):
    import hashlib
    st = os.stat(path)
    raw = "%s|%d|%d" % (cache_identity(path), st.st_size, int(st.st_mtime))
    if version is not None:
        raw += "|v%s|t%.4f" % (version, TRIM_APPENDAGE)
    key = hashlib.sha1(raw.encode()).hexdigest()
    return os.path.join(cache_dir("masks"), key + ".npz")


def trim_appendages(mask, width_frac=None, keep_frac=0.55):
    """Remove wires, cables and tails: anything thinner than a fraction of the part.

    Erode to a core, then grow it back a bounded number of steps inside the mask. The body
    recovers its own outline, corners included, because every boundary pixel is within the
    erosion radius of the core; a cable running away from the body only grows back that far
    and the rest of it is gone. Full reconstruction was tried first and does the opposite -
    the tail is connected, so it fills straight back in - and a plain opening drops the tail
    but rounds every corner by the radius. Refused if it would eat the part.
    """
    width_frac = TRIM_APPENDAGE if width_frac is None else width_frac
    area = float(mask.sum())
    if area <= 0:
        return mask
    step = max(1, int(max(mask.shape) / 512))
    small = mask[::step, ::step]
    if not small.any():
        return mask
    ys, xs = np.nonzero(small)
    span = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    radius = int(round(width_frac * span / 2.0))
    if radius < 1:
        return mask
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (yy ** 2 + xx ** 2) <= radius ** 2
    core = ndimage.binary_erosion(small, disk)
    if not core.any():
        return mask
    kept_small = ndimage.binary_dilation(core, np.ones((3, 3), bool), iterations=radius, mask=small)
    if kept_small.sum() >= keep_frac * small.sum():
        grown = np.repeat(np.repeat(kept_small, step, axis=0), step, axis=1)[:mask.shape[0], :mask.shape[1]]
        pad = np.zeros_like(mask)
        pad[:grown.shape[0], :grown.shape[1]] = grown
        kept = mask & ndimage.binary_dilation(pad, np.ones((3, 3), bool), iterations=2 * step)
        if kept.sum() >= keep_frac * area:
            return kept
    return mask


def recover_holes(image, mask, tol=0.10, min_frac=0.01):
    """Carve interior regions whose colour matches the background seen around the part."""
    inner = ndimage.binary_erosion(mask, np.ones((9, 9)))
    outside = ~ndimage.binary_dilation(mask, np.ones((25, 25)))
    if outside.sum() < 500 or inner.sum() < 500:
        return mask
    background = np.median(image[outside], axis=0)
    distance = np.linalg.norm(image - background, axis=2)
    limit = tol * np.sqrt(3)
    candidates = (distance < limit) & inner
    body = inner & ~candidates
    if body.sum() < 0.2 * inner.sum() or float(np.median(distance[body])) < 3.0 * limit:
        return mask
    labelled, count = ndimage.label(candidates)
    edge = ~ndimage.binary_erosion(mask, np.ones((3, 3)))
    carved = np.zeros_like(mask)
    for i in range(1, count + 1):
        blob = labelled == i
        if blob.sum() < min_frac * mask.sum() or (blob & edge).any():
            continue
        carved |= blob
    return mask & ~carved


def segment_photo(path):
    f = cached_mask(path)
    if os.path.exists(f):
        return np.load(f)["mask"]
    legacy = cached_mask(path, version=None)
    image = load(path)
    base = np.load(legacy)["mask"] if os.path.exists(legacy) else segment_rmbg(image)
    mask = largest(trim_appendages(recover_holes(image, base)))
    os.makedirs(cache_dir("masks"), exist_ok=True)
    np.savez_compressed(f, mask=mask)
    return mask


def segment_rmbg(a):
    import torch
    from PIL import Image as _Image
    from torchvision import transforms
    if "model" not in _RMBG:
        from transformers import AutoModelForImageSegmentation
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        m = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True).to(dev).eval()
        _RMBG.update(model=m, dev=dev)
    im = _Image.fromarray((a * 255).astype(np.uint8))
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        pred = _RMBG["model"](tf(im).unsqueeze(0).to(_RMBG["dev"]))[-1].sigmoid().cpu()[0, 0]
    alpha = np.asarray(transforms.ToPILImage()(pred).resize(im.size)) > 128
    return largest(alpha)


def segment_any(a, mode="rmbg"):
    return {"dark": segment, "auto": segment_auto}.get(mode, segment_rmbg)(a)


DEGENERATE_PCA = float(os.environ.get("P2F_DEGENERATE_PCA", 0.85))


BOX_FILL_MIN = float(os.environ.get("P2F_BOX_FILL_MIN", 0.90))


def box_angle(mask):
    """The min-area box angle, but only for a shape that actually fills a box."""
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    (_, _), (w, h), deg = cv2.minAreaRect(contour)
    if w <= 0 or h <= 0 or float(cv2.contourArea(contour)) / (w * h) < BOX_FILL_MIN:
        return None
    if w < h:
        deg += 90.0
    return float(((deg + 90.0) % 180.0) - 90.0)


def upright_mask(mask):
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    major = vecs[:, int(np.argmax(vals))]
    angle = np.degrees(np.arctan2(major[0], major[1]))
    if float(min(vals) / max(max(vals), 1e-12)) > DEGENERATE_PCA:
        boxed = box_angle(mask)
        if boxed is not None:
            angle = -boxed
    rotated = ndimage.rotate(mask.astype(np.uint8), -angle, reshape=True, order=0) > 0
    return rotated, float(angle)


def boundary_period(pts, min_peaks=None, min_prominence=None, max_peaks=60, min_amplitude=None):
    min_peaks = int(os.environ.get("P2F_PERIOD_MIN_PEAKS", 7)) if min_peaks is None else min_peaks
    min_prominence = float(os.environ.get("P2F_PERIOD_PROMINENCE", 0.02492)) if min_prominence is None else min_prominence
    min_amplitude = float(os.environ.get("P2F_PERIOD_AMPLITUDE", 0.06429)) if min_amplitude is None else min_amplitude
    """How many repeated features sit on this contour, or None if it is not periodic.

    A gear, a scalloped disc or a perforated rim has a radius that rises and falls a fixed
    number of times around the centroid. Simplifying such a contour by arc length erases
    exactly the features that define the part.
    """
    from scipy.signal import find_peaks
    pts = np.asarray(pts, float)
    if len(pts) < 40:
        return None
    centre = pts.mean(axis=0)
    delta = pts - centre
    radius = np.hypot(delta[:, 0], delta[:, 1])
    if radius.max() <= 0:
        return None
    middle = float(np.median(radius))
    if middle <= 0 or (np.percentile(radius, 97) - np.percentile(radius, 3)) / middle < min_amplitude:
        return None
    order = np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))
    profile = radius[order] / radius.max()
    wrapped = np.concatenate([profile, profile[:len(profile) // 6]])
    peaks, _ = find_peaks(wrapped, prominence=min_prominence, distance=max(3, len(profile) // 60))
    peaks = [q for q in peaks if q < len(profile)]
    return len(peaks) if min_peaks <= len(peaks) <= max_peaks else None


CIRCLE_VETO_AMPLITUDE = float(os.environ.get("P2F_CIRCLE_VETO_AMPLITUDE", 0.19944))


def feature_amplitude(pts):
    """How far the contour departs from its mean radius, as a fraction of that radius."""
    pts = np.asarray(pts, float)
    delta = pts - pts.mean(axis=0)
    radius = np.hypot(delta[:, 0], delta[:, 1])
    middle = float(np.median(radius))
    if middle <= 0:
        return 0.0
    return float((np.percentile(radius, 97) - np.percentile(radius, 3)) / middle)


def outline(mask, eps_frac=0.008, min_hole=None):
    min_hole = th.MIN_HOLE_FRAC if min_hole is None else min_hole
    import cv2
    m = mask.astype(np.uint8) * 255
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer_i = max(range(len(cs)), key=lambda i: cv2.contourArea(cs[i]))
    c = cs[outer_i]
    area = cv2.contourArea(c)
    simplify = lambda k: cv2.approxPolyDP(k, eps_frac * cv2.arcLength(k, True), True).reshape(-1, 2).astype(float)
    holes = [simplify(cs[i]) for i in range(len(cs)) if hier[0][i][3] == outer_i and cv2.contourArea(cs[i]) > min_hole * area]
    hole_area = sum(cv2.contourArea(cs[i]) for i in range(len(cs)) if hier[0][i][3] == outer_i)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    x, y, w, h = cv2.boundingRect(c)
    raw_holes = [cs[i].reshape(-1, 2).astype(float).tolist() for i in range(len(cs)) if hier[0][i][3] == outer_i and cv2.contourArea(cs[i]) > min_hole * area]
    return simplify(c), {"rectangularity": float(area / max(w * h, 1)), "stroke_px": float(2 * (area - hole_area) / max(cv2.arcLength(c, True), 1)),
                         "solidity": float(area / max(hull_area, 1)), "bbox": (w, h), "holes": [h_.tolist() for h_ in holes],
                         "hole_frac": float(hole_area / max(area, 1)), "raw": c.reshape(-1, 2).astype(float).tolist(), "raw_holes": raw_holes}

def reflect_richer_half(shape, holes, axis):
    n = shape.shape[axis]
    half = n // 2
    take = lambda a, first: (a[:, :half] if first else a[:, n - half:]) if axis == 1 else (a[:half] if first else a[n - half:])
    first_area = int(take(holes, True).sum())
    second_area = int(take(holes, False).sum())
    keep_first = first_area >= second_area
    def rebuild(a):
        chosen = take(a, keep_first)
        mirrored = np.flip(chosen, axis=axis)
        out = a.copy()
        if axis == 1:
            out[:, :half], out[:, n - half:] = (chosen, mirrored) if keep_first else (mirrored, chosen)
        else:
            out[:half], out[n - half:] = (chosen, mirrored) if keep_first else (mirrored, chosen)
        return out
    return rebuild(shape), rebuild(holes)


def drop_speck_holes(holes, area, min_frac=None):
    """Reflection can leave slivers a few pixels wide; they are not features."""
    limit = (th.MIN_HOLE_FRAC if min_frac is None else min_frac) * max(area, 1)
    labelled, count = ndimage.label(holes)
    if count == 0:
        return holes
    keep = np.zeros_like(holes)
    for i in range(1, count + 1):
        blob = labelled == i
        if blob.sum() >= limit:
            keep |= blob
    return keep


def symmetrize(mask, min_iou=0.93):
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    ys, xs = np.nonzero(filled)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = filled[y0:y1, x0:x1]
    hole_crop = holes[y0:y1, x0:x1]
    axes = []
    for axis in (1, 0):
        f = np.flip(crop, axis=axis)
        if (crop & f).sum() / max((crop | f).sum(), 1) >= min_iou:
            crop = crop | f
            hole_crop = reflect_richer_half(hole_crop, hole_crop, axis)[0]
            axes.append("x" if axis == 1 else "y")
    out, out_holes = filled.copy(), holes.copy()
    out[y0:y1, x0:x1] = crop
    out_holes[y0:y1, x0:x1] = hole_crop
    return out & ~(drop_speck_holes(out_holes & out, out.sum())), axes


def fit_circle(pts):
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    (cx, cy, c), *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x ** 2 + y ** 2, rcond=None)
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-9)))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    w, h = np.ptp(x), np.ptp(y)
    return float(cx), float(cy), r, rms / r, float(min(w, h) / max(w, h, 1e-9))


SYMMETRY_TOL = float(os.environ.get("P2F_SYMMETRY", "0"))
RADIUS_TOL = float(os.environ.get("P2F_RADIUS_UNIFY", "0"))


def mirror_score(pts, axis, centre):
    """Region overlap between a loop and its own reflection, 1.0 when perfectly symmetric."""
    from shapely.geometry import Polygon
    try:
        p = Polygon(np.asarray(pts, float))
        if not p.is_valid or p.area <= 0:
            return 0.0
    except Exception:
        return 0.0
    q = np.asarray(pts, float).copy()
    q[:, axis] = 2 * centre - q[:, axis]
    try:
        m = Polygon(q)
        u = p.union(m).area
        return float(p.intersection(m).area / u) if u else 0.0
    except Exception:
        return 0.0


def symmetrise(pts, axis, centre, rounds=4):
    """Average every point with its reflected partner, so a near-symmetric loop becomes exact.

    Nearest-partner pairing is not an involution - a maps to b without b mapping to a - so one
    pass leaves the loop partly asymmetric and it has to be iterated to settle.
    """
    q = np.asarray(pts, float).copy()
    for _ in range(rounds):
        mirrored = q.copy()
        mirrored[:, axis] = 2 * centre - mirrored[:, axis]
        d = np.linalg.norm(q[:, None, :] - mirrored[None, :, :], axis=2)
        q = (q + mirrored[d.argmin(axis=1)]) / 2.0
    return q


def symmetrise_loops(loops, tol):
    """Make an almost-symmetric traced outline exactly symmetric.

    69% of real sketches are mirror-symmetric to within 0.95 region IoU and the median is exactly
    1.000, so a traced outline that is nearly symmetric almost certainly should be. Tracing noise
    breaks it: two fillets that are one radius in the part come back as two radii in the drawing.
    """
    if not loops:
        return loops
    outer = np.asarray(loops[0], float)
    if len(outer) < 8:
        return loops
    best = None
    for axis in (0, 1):
        centre = float(outer[:, axis].mean())
        sc = mirror_score(outer, axis, centre)
        if sc >= tol and (best is None or sc > best[0]):
            best = (sc, axis, centre)
    if best is None:
        return loops
    _, axis, centre = best
    return [symmetrise(np.asarray(lp, float), axis, centre).tolist() if len(lp) >= 8 else lp
            for lp in loops]


def unify_radii(els, tol):
    """Snap arc radii that are within tol of each other onto their shared mean.

    A part is drilled and filleted with a small set of tools: 49% of real curved edges share a
    radius with another edge of the same sketch, and on 27% of parts every curve does.
    """
    arcs = [e for e in els if e.get("type") == "arc" and e.get("r")]
    if len(arcs) < 2:
        return els
    r = np.array([e["r"] for e in arcs], float)
    order = np.argsort(r)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if abs(r[i] - r[cur[-1]]) <= tol * max(r[i], r[cur[-1]]):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    for g in groups:
        if len(g) < 2:
            continue
        mean = float(np.mean([r[i] for i in g]))
        for i in g:
            arcs[i]["r"] = mean
    return els


def dominant_frame(pts):
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 3:
        return 0.0
    weight, angle = [], []
    for i in range(n):
        d = pts[(i + 1) % n] - pts[i]
        length = float(np.hypot(*d))
        if length <= 0:
            continue
        weight.append(length)
        angle.append(np.degrees(np.arctan2(d[1], d[0])) % 90.0)
    if not weight:
        return 0.0
    weight, angle = np.array(weight), np.radians(np.array(angle) * 4.0)
    mean = np.arctan2((weight * np.sin(angle)).sum(), (weight * np.cos(angle)).sum())
    return float((np.degrees(mean) / 4.0) % 90.0)


def fit_directions(pts, frame=0.0, step=15.0, tol_deg=10.0, max_shift=0.08):
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 3:
        return pts
    edges = np.array([pts[(i + 1) % n] - pts[i] for i in range(n)])
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if not np.all(lengths > 0):
        return pts
    angles = np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))
    targets = np.round((angles - frame) / step) * step + frame
    keep = np.abs(((angles - targets + 180) % 360) - 180) > tol_deg
    targets[keep] = angles[keep]
    u = np.column_stack([np.cos(np.radians(targets)), np.sin(np.radians(targets))])
    prefix = np.zeros((n, n))
    for i in range(1, n):
        prefix[i, :i] = 1.0
    A = np.zeros((2 * n + 2, n))
    b = np.zeros(2 * n + 2)
    origin = pts[0]
    for axis in (0, 1):
        A[axis * n:(axis + 1) * n] = prefix * u[:, axis]
        b[axis * n:(axis + 1) * n] = pts[:, axis] - origin[axis]
    weight = 10.0 * float(np.sum(lengths))
    A[2 * n] = weight * u[:, 0]
    A[2 * n + 1] = weight * u[:, 1]
    solved, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.any(solved <= 0):
        return pts
    out = np.zeros_like(pts)
    out[0] = origin
    for i in range(1, n):
        out[i] = out[i - 1] + solved[i - 1] * u[i - 1]
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-6)
    if float(np.abs(out - pts).max()) > max_shift * span:
        return pts
    return out


def snap_angles(pts, tol_deg=None, step=15.0, lock=None, frame=0.0):
    tol_deg = float(os.environ.get("P2F_ANGLE_TOL", 3.36602)) if tol_deg is None else tol_deg
    pts = np.asarray(pts, float).copy()
    n = len(pts)
    lock = lock or [False] * n
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n
            if lock[i] or lock[j]:
                continue
            a, b = pts[i], pts[j]
            d = b - a
            length = float(np.hypot(*d))
            if length < 1e-6:
                continue
            ang = np.degrees(np.arctan2(d[1], d[0]))
            target = round((ang - frame) / step) * step + frame
            if abs(ang - target) > tol_deg:
                continue
            mid = (a + b) / 2
            t = np.radians(target)
            half = 0.5 * length * np.array([np.cos(t), np.sin(t)])
            pts[i], pts[j] = mid - half, mid + half
    return pts


def snap_rectilinear(pts, tol_deg=12.0, lock=None):
    pts = np.asarray(pts, float).copy()
    n = len(pts)
    lock = lock or [False] * n
    for _ in range(3):
        for i in range(n):
            if lock[i]:
                continue
            a, b = pts[i], pts[(i + 1) % n]
            ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180
            if ang < tol_deg or ang > 180 - tol_deg:
                pts[i][1] = pts[(i + 1) % n][1] = (a[1] + b[1]) / 2
            elif abs(ang - 90) < tol_deg:
                pts[i][0] = pts[(i + 1) % n][0] = (a[0] + b[0]) / 2
    return pts


def merge_collinear(pts, min_len, tol_deg=6.0):
    pts = [np.asarray(q, float) for q in pts]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            d1, d2 = b - a, c - b
            turn = abs((np.degrees(np.arctan2(d2[1], d2[0]) - np.arctan2(d1[1], d1[0])) + 180) % 360 - 180)
            if turn < tol_deg or (np.hypot(*d1) < min_len and turn < 45):
                pts.pop(i)
                changed = True
                break
    return np.array(pts)


def arc_span(run, cx, cy):
    ang = np.unwrap(np.arctan2(run[:, 1] - cy, run[:, 0] - cx))
    return float(np.degrees(ang[-1] - ang[0]))


RUN_EPS = float(os.environ.get("P2F_RUN_EPS", 0.01806))
RECONCILE_ARCS = os.environ.get("P2F_RECONCILE_ARCS", "1") != "0"
ELLIPSE_ARCS = os.environ.get("P2F_ELLIPSE_ARCS", "1") != "0"
ELLIPSE_MIN_ASPECT = 0.15
ELLIPSE_MAX_ASPECT = 0.92
ELLIPSE_FIT_TOL = 0.03
ELLIPSE_BETTER_THAN = 0.6
KEEP_SIMPLE = os.environ.get("P2F_KEEP_SIMPLE", "1") != "0"
DROP_STRAY_HOLES = os.environ.get("P2F_DROP_STRAY_HOLES", "1") != "0"
REPEATED_RUN_EPS = float(os.environ.get("P2F_REPEATED_RUN_EPS", 0.00836))
def corner_runs(raw, eps_frac=None):
    eps_frac = RUN_EPS if eps_frac is None else eps_frac
    import cv2
    pts = np.asarray(raw, np.float32)
    poly = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps_frac * cv2.arcLength(pts, True), True).reshape(-1, 2)
    corners = [int(np.argmin(np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1]))) for q in poly]
    corners = sorted(set(corners))
    runs = []
    for j, c in enumerate(corners):
        d = corners[(j + 1) % len(corners)]
        run = pts[c:d + 1] if d > c else np.vstack([pts[c:], pts[:d + 1]])
        runs.append(run.astype(float))
    return runs






MIN_ELLIPSE_POINTS = 12


def ellipse_ok(f, n_points=None):
    if f is None or (n_points is not None and n_points < MIN_ELLIPSE_POINTS):
        return False
    return f["rms"] < 0.04 * f["a"] and f["aspect"] > 0.5


def arc_from_run(run, ccw=None):
    f = fit_ellipse(run)
    cx, cy, r = (f["cx"], f["cy"], f["a"]) if ellipse_ok(f) else fit_circle(run)[:3]
    c = np.array([cx, cy])
    on = lambda q: (c + (np.asarray(q) - c) / max(np.hypot(*(np.asarray(q) - c)), 1e-9) * r).tolist()
    if ccw is None:
        ccw = arc_span(run, cx, cy) > 0
    return {"type": "arc", "p0": on(run[0]), "p1": on(run[-1]), "cx": float(cx), "cy": float(cy), "r": float(r), "ccw": bool(ccw), "_run": run}






def split_straight(seg, length_px, eps_frac=0.015):
    import cv2
    if len(seg) < 4:
        return [seg]
    pts = np.asarray(seg, np.float32)
    poly = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps_frac * cv2.arcLength(pts, False), False).reshape(-1, 2)
    cuts = sorted({0, len(seg) - 1} | {int(np.argmin(np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1]))) for q in poly})
    return [seg[a:b + 1] for a, b in zip(cuts, cuts[1:]) if b - a >= 1]


def absorb_short(spans, n, frac=0.04):
    lo = max(int(frac * n), 12)
    length = lambda s: (s[1] - s[0]) % n or n
    while len(spans) > 1:
        i = min(range(len(spans)), key=lambda j: length(spans[j]))
        if length(spans[i]) >= lo:
            break
        j = (i - 1) % len(spans)
        spans[j] = (spans[j][0], spans[i][1], spans[j][2])
        spans.pop(i)
    merged = [spans[0]]
    for a, b, c in spans[1:]:
        if c == merged[-1][2]:
            merged[-1] = (merged[-1][0], b, c)
        else:
            merged.append((a, b, c))
    if len(merged) > 1 and merged[0][2] == merged[-1][2]:
        merged[0] = (merged[-1][0], merged[0][1], merged[0][2])
        merged.pop()
    return merged






def merge_and_snap(els, length_px):
    def mergeable(a, b):
        if a["type"] != "arc" or b["type"] != "arc" or a["ccw"] != b["ccw"]:
            return False
        joint = np.vstack([a["_run"], b["_run"]])
        return ellipse_ok(fit_ellipse(joint)) or (np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) < 0.10 * a["r"] and abs(a["r"] - b["r"]) < 0.15 * a["r"])

    merged = []
    for e in els:
        if merged and mergeable(merged[-1], e):
            merged[-1] = arc_from_run(np.vstack([merged[-1]["_run"], e["_run"]]))
        else:
            merged.append(dict(e))
    if len(merged) > 1 and mergeable(merged[-1], merged[0]):
        merged[-1] = arc_from_run(np.vstack([merged[-1]["_run"], merged[0]["_run"]]))
        merged.pop(0)
    arcs = [i for i, e in enumerate(merged) if e["type"] == "arc"]
    for j, i in enumerate(arcs):
        e = merged[i]
        for k in arcs[:j]:
            o = merged[k]
            if o.get("centre_of") is None and np.hypot(o["cx"] - e["cx"], o["cy"] - e["cy"]) < 0.10 * max(o["r"], e["r"]):
                e["cx"], e["cy"], e["centre_of"] = o["cx"], o["cy"], k
                break
    for e in merged:
        if e["type"] != "arc":
            continue
        run = np.asarray(e["_run"], float)
        c = np.array([e["cx"], e["cy"]])
        e["ccw"] = bool(arc_span(run, c[0], c[1]) > 0)
        on = lambda q: (c + (np.asarray(q, float) - c) / max(np.hypot(*(np.asarray(q, float) - c)), 1e-9) * e["r"]).tolist()
        e["p0"], e["p1"] = on(run[0]), on(run[-1])
    for e in merged:
        if e.get("_run") is not None:
            e["support"] = support_of(e["_run"], e, length_px)
        e.pop("_run", None)
    return merged


def segment_res(pts, a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    L2 = float(d @ d)
    if L2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ d / L2, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * d), axis=1)


def support_of(run, element, length_px):
    """How much contour evidence stands behind one fitted element, and how well it fits.

    The tracer fits a line or an arc to a run of contour points and then keeps only the endpoints.
    That discards the one thing needed to tell a well-seen edge from a badly-seen one, which is why
    a constraint predictor built on the fitted geometry alone could not beat its majority baseline:
    where the trace is close the answer is already known, and where it is wrong nothing in the
    output says so.
    """
    run = np.asarray(run, float)
    n = len(run)
    if n < 2:
        return {"points": n, "residual": 0.0, "span_px": 0.0, "chord_px": 0.0, "straightness": 1.0}
    steps = np.linalg.norm(np.diff(run, axis=0), axis=1)
    span = float(steps.sum())
    chord = float(np.linalg.norm(run[-1] - run[0]))
    if element["type"] == "line":
        a, b = np.asarray(element["p0"], float), np.asarray(element["p1"], float)
        d = b - a
        L2 = float(d @ d)
        if L2 < 1e-12:
            res = 0.0
        else:
            t = np.clip((run - a) @ d / L2, 0.0, 1.0)
            res = float(np.sqrt(np.mean(np.linalg.norm(run - (a + t[:, None] * d), axis=1) ** 2)))
    elif element["type"] == "bsplinecurve":
        knots = np.asarray(element["xy"], float)
        seg = [segment_res(run, knots[i], knots[i + 1]) for i in range(len(knots) - 1)]
        res = float(np.sqrt(np.mean(np.min(np.stack(seg), axis=0) ** 2)))
    else:
        c = np.array([element["cx"], element["cy"]], float)
        res = float(np.sqrt(np.mean((np.linalg.norm(run - c, axis=1) - element["r"]) ** 2)))
    return {"points": int(n), "residual": float(res / max(length_px, 1e-9)),
            "span_px": span / max(length_px, 1e-9), "chord_px": chord / max(length_px, 1e-9),
            "straightness": float(chord / max(span, 1e-9))}


def carry_support(final, original):
    """Copy each element's evidence onto whatever regularisation turned it into."""
    if not original:
        return final
    mid = lambda e: (np.asarray(e.get("p0", [0, 0]), float) + np.asarray(e.get("p1", [0, 0]), float)) / 2
    src = np.array([mid(e) for e in original], float)
    for e in final:
        d = np.linalg.norm(src - mid(e), axis=1)
        e["support"] = dict(original[int(d.argmin())].get("support") or {})
    return final


BSPLINE_ARCS = os.environ.get("P2F_BSPLINE_ARCS", "1") != "0"
SEQNET = os.environ.get("P2F_SEQNET", "0") == "1"
CHAIN_ARCS = os.environ.get("P2F_CHAIN_ARCS", "1") != "0"
CHAIN_TURN_DEG = float(os.environ.get("P2F_CHAIN_TURN_DEG", 50.0))


def chord_turn(a, b):
    va = np.subtract(a["p1"], a["p0"])
    vb = np.subtract(b["p1"], b["p0"])
    return float(np.degrees(np.arctan2(va[0] * vb[1] - va[1] * vb[0], float(va @ vb))))


def chain_arcs(els, length_px):
    """Refit maximal chains of line pieces that all bend the same way as one arc.

    Of the arcs the tracer loses, 71% survive simplification as two or more pieces and are refused
    piecewise - each piece shows the gate a fraction of the true sweep. Three or more pieces each
    turning the same direction by a small angle is a sampled curve; two straight edges meet at one
    large corner, a zigzag alternates sign, and a single shallow corner is only one turn, so none
    can qualify. The merged run must still pass every shipped arc gate on its full sweep - this
    widens the fitter's support, not the gate.
    """
    n = len(els)
    if n < 3:
        return els
    def linked(i):
        a, b = els[i], els[(i + 1) % n]
        if a["type"] != "line" or b["type"] != "line" or a.get("_run") is None or b.get("_run") is None:
            return 0
        t = chord_turn(a, b)
        return (1 if t > 0 else -1) if 0.5 < abs(t) < CHAIN_TURN_DEG else 0
    signs = [linked(i) for i in range(n)]
    if all(signs) and len(set(signs)) == 1:
        return els
    start = next(i for i in range(n) if signs[i - 1] == 0 or signs[i - 1] != signs[i])
    order = [(start + k) % n for k in range(n)]
    def passes(chain):
        run = np.vstack([els[j]["_run"] for j in chain])
        chord = float(np.hypot(*(run[-1] - run[0])))
        if len(run) < th.ARC_MIN_POINTS or chord <= th.ARC_MIN_CHORD_FRAC * length_px:
            return None
        cx, cy, r, rel, _ = fit_circle(run)
        span = arc_span(run, cx, cy)
        cv = run[-1] - run[0]
        sag = float(np.max(np.abs(cv[0] * (run[:, 1] - run[0][1]) - cv[1] * (run[:, 0] - run[0][0])) / max(chord, 1e-9)))
        ok = (rel * r < max(th.ARC_FIT_TOL * r, 1.2) and th.ARC_MIN_SPAN_DEG < abs(span) < th.ARC_MAX_SPAN_DEG
              and sag > th.ARC_MIN_SAG_FRAC * chord)
        return arc_from_run(run) if ok else None
    def spline(chain):
        if not BSPLINE_ARCS or len(chain) < 4:
            return None
        turn = sum(abs(chord_turn(els[chain[k]], els[chain[k + 1]])) for k in range(len(chain) - 1))
        if turn < th.ARC_MIN_SPAN_DEG:
            return None
        run = np.vstack([els[j]["_run"] for j in chain])
        idx = np.linspace(0, len(run) - 1, 9).round().astype(int)
        return {"type": "bsplinecurve", "p0": run[0].tolist(), "p1": run[-1].tolist(),
                "xy": run[idx].tolist(), "_run": run}
    def resolve(chain):
        if len(chain) < 3:
            return [els[j] for j in chain]
        for size in range(len(chain), 2, -1):
            for lo in range(len(chain) - size + 1):
                arc = passes(chain[lo:lo + size])
                if arc is not None:
                    return resolve(chain[:lo]) + [arc] + resolve(chain[lo + size:])
        curved = spline(chain)
        if curved is not None:
            return [curved]
        return [els[j] for j in chain]
    out, chain, chain_sign = [], [], 0
    for k, i in enumerate(order):
        chain.append(i)
        junction = signs[i] if k + 1 < n else 0
        if junction != 0 and (chain_sign == 0 or junction == chain_sign):
            chain_sign = junction
            continue
        out.extend(resolve(chain))
        chain, chain_sign = [], 0
    out.extend(resolve(chain))
    return out


TANGENT_MERGE = os.environ.get("P2F_TANGENT_MERGE", "0") == "1"
TANGENT_RADIUS_SPREAD = float(os.environ.get("P2F_TANGENT_RADIUS_SPREAD", 1.6))
SMOOTH_JOIN_DEG = float(os.environ.get("P2F_SMOOTH_JOIN_DEG", 35.0))


def end_tangent(e, at_end):
    if e["type"] == "line":
        d = np.subtract(e["p1"], e["p0"])
    elif e["type"] == "bsplinecurve":
        q = np.asarray(e["xy"], float)
        d = q[-1] - q[-2] if at_end else q[1] - q[0]
    else:
        q = np.array(e["p1"] if at_end else e["p0"], float)
        rad = q - np.array([e["cx"], e["cy"]])
        d = np.array([-rad[1], rad[0]]) * (1.0 if e.get("ccw", True) else -1.0)
    return d / max(np.hypot(*d), 1e-9)


def smooth_break_deg(a, b):
    ta, tb = end_tangent(a, True), end_tangent(b, False)
    return float(np.degrees(np.arccos(np.clip(ta @ tb, -1.0, 1.0))))


def tangent_merge(els, length_px):
    """One freeform curve, drawn piecewise, comes back together.

    A smooth boundary that is not circular gets fitted as tangent-joined arcs of scattered radii
    with the odd line between - 7 elements drawn where the STEP wants one closed spline. A fillet
    pattern is also tangent-joined but its radii agree, which is the discriminant: merge only
    maximal tangent chains holding two or more arcs whose radii disagree by a real factor, and
    only into a spline when no single circle explains the merged run.
    """
    n = len(els)
    if not TANGENT_MERGE or n < 3:
        return els
    if any(e.get("_run") is None for e in els):
        return els
    linked = [smooth_break_deg(els[i - 1], els[i]) < SMOOTH_JOIN_DEG for i in range(n)]
    if all(linked):
        chains = [list(range(n))]
    else:
        start = next(i for i in range(n) if not linked[i])
        order = [(start + k) % n for k in range(n)]
        chains, cur = [], []
        for k, i in enumerate(order):
            cur.append(i)
            nxt = order[(k + 1) % n]
            if k + 1 == n or not linked[nxt]:
                chains.append(cur)
                cur = []
    out_idx = set()
    replacements = {}
    for chain in chains:
        radii = [els[j]["r"] for j in chain if els[j]["type"] == "arc"]
        inner = chain[1:-1] if len(chain) > 2 else []
        interleaved = any(els[j]["type"] == "line" for j in inner)
        if len(chain) < 3 or len(radii) < 2 or not interleaved:
            continue
        if max(radii) / max(min(radii), 1e-9) < TANGENT_RADIUS_SPREAD:
            continue
        run = np.vstack([els[j]["_run"] for j in chain])
        chord = float(np.hypot(*(run[-1] - run[0])))
        cx, cy, r, rel, _ = fit_circle(run)
        if rel * r < max(th.ARC_FIT_TOL * r, 1.2) and chord > 1e-6:
            replacements[chain[0]] = (chain, arc_from_run(run))
            continue
        k = max(9, min(2 + len(run) // 40, 17))
        idx = np.linspace(0, len(run) - 1, k).round().astype(int)
        replacements[chain[0]] = (chain, {"type": "bsplinecurve", "p0": run[0].tolist(),
                                          "p1": run[-1].tolist(), "xy": run[idx].tolist(),
                                          "_run": run})
    if not replacements:
        return els
    consumed = {j for chain, _ in replacements.values() for j in chain}
    out = []
    for i in range(n):
        if i in replacements:
            out.append(replacements[i][1])
        elif i not in consumed:
            out.append(els[i])
    return out if len(out) >= 1 else els


def elements(raw, length_px):
    repeated = boundary_period(raw)
    runs = corner_runs(raw, REPEATED_RUN_EPS if repeated else None)
    els = []
    for run in runs:
        chord = float(np.hypot(*(run[-1] - run[0])))
        if len(run) >= th.ARC_MIN_POINTS and chord > th.ARC_MIN_CHORD_FRAC * length_px:
            cx, cy, r, rel, _ = fit_circle(run)
            span = arc_span(run, cx, cy)
            cv = run[-1] - run[0]
            sag = float(np.max(np.abs(cv[0] * (run[:, 1] - run[0][1]) - cv[1] * (run[:, 0] - run[0][0])) / max(chord, 1e-9)))
            if rel * r < max(th.ARC_FIT_TOL * r, 1.2) and th.ARC_MIN_SPAN_DEG < abs(span) < th.ARC_MAX_SPAN_DEG and sag > th.ARC_MIN_SAG_FRAC * chord:
                els.append(arc_from_run(run))
                continue
        line = {"type": "line", "p0": run[0].tolist(), "p1": run[-1].tolist(), "_run": run}
        line["support"] = support_of(run, line, length_px)
        els.append(line)
    if CHAIN_ARCS:
        els = chain_arcs(els, length_px)
    els = tangent_merge(els, length_px)
    for e in els:
        if "support" not in e and e.get("_run") is not None:
            e["support"] = support_of(e["_run"], e, length_px)
    return merge_and_snap(els, length_px)


def square_quadrilateral(els, frame=0.0, convex_tol=0.97, taper_tol=0.35):
    if len(els) != 4 or any(e["type"] != "line" for e in els):
        return els
    pts = np.array([e["p0"] for e in els], float)
    area = abs(_signed_area(pts))
    if area <= 0:
        return els
    import cv2
    hull = cv2.convexHull(pts.astype(np.float32))
    if area / max(float(cv2.contourArea(hull)), 1e-9) < convex_tol:
        return els
    sides = np.array([np.hypot(*(pts[(i + 1) % 4] - pts[i])) for i in range(4)])
    pairs = [(sides[0], sides[2]), (sides[1], sides[3])]
    for a, b in pairs:
        if abs(a - b) / max(a, b, 1e-9) > taper_tol:
            return els
    width, height = float(np.mean(pairs[0])), float(np.mean(pairs[1]))
    centre = pts.mean(axis=0)
    t = np.radians(frame)
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-np.sin(t), np.cos(t)])
    corners = [centre - u * width / 2 - v * height / 2, centre + u * width / 2 - v * height / 2,
               centre + u * width / 2 + v * height / 2, centre - u * width / 2 + v * height / 2]
    if _signed_area(pts) < 0:
        corners = corners[::-1]
    return [{"type": "line", "p0": corners[i].tolist(), "p1": corners[(i + 1) % 4].tolist()}
            for i in range(4)]


def rectangularise(els, fill_tol=0.92, side_tol=0.06):
    if any(e["type"] != "line" for e in els) or len(els) < 4:
        return els
    pts = np.array([e["p0"] for e in els], float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    box = (hi - lo)
    if min(box) <= 0:
        return els
    area = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                         for i in range(len(pts))))
    if area / (box[0] * box[1]) < fill_tol:
        return els
    slack = side_tol * max(box)
    on_edge = np.minimum(np.abs(pts - lo), np.abs(pts - hi)).min(axis=1)
    if on_edge.max() > slack:
        return els
    corners = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    order = corners if _signed_area(pts) > 0 else corners[::-1]
    return [{"type": "line", "p0": order[i], "p1": order[(i + 1) % 4]} for i in range(4)]


def _signed_area(pts):
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


def merge_line_elements(els, min_len, tol_deg=6.0):
    if len(els) < 3:
        return els
    direction = lambda e: np.subtract(e["p1"], e["p0"])
    def turn(a, b):
        d1, d2 = direction(a), direction(b)
        return abs((np.degrees(np.arctan2(d2[1], d2[0]) - np.arctan2(d1[1], d1[0])) + 180) % 360 - 180)
    def joinable(a, b):
        if a["type"] != "line" or b["type"] != "line":
            return False
        t = turn(a, b)
        return t < tol_deg or (np.hypot(*direction(a)) < min_len and t < 45)
    merged = []
    for e in els:
        if merged and joinable(merged[-1], e):
            merged[-1] = {"type": "line", "p0": merged[-1]["p0"], "p1": e["p1"]}
        else:
            merged.append(dict(e))
    while len(merged) > 3 and joinable(merged[-1], merged[0]):
        merged[0] = {"type": "line", "p0": merged[-1]["p0"], "p1": merged[0]["p1"]}
        merged.pop()
    return merged


def regularise_lines(els, length_px, frame=None):
    pts = np.array([e["p0"] for e in els], float)
    lock = [els[i]["type"] != "line" or els[i - 1]["type"] != "line" for i in range(len(els))]
    keep = list(range(len(els)))
    if all(e["type"] == "line" for e in els):
        pts = merge_collinear(snap_rectilinear(pts), 0.015 * length_px)
        frame = dominant_frame(pts) if frame is None else frame
        pts = snap_angles(snap_rectilinear(pts), tol_deg=7.0, frame=frame)
        pts = fit_directions(pts, frame=frame)
        return [{"type": "line", "p0": pts[i].tolist(), "p1": pts[(i + 1) % len(pts)].tolist()} for i in range(len(pts))]
    pts = snap_rectilinear(pts, lock=lock)
    for i, e in enumerate(els):
        if e["type"] != "line":
            pts[i], pts[(i + 1) % len(els)] = e["p0"], e["p1"]
    for i, e in enumerate(els):
        e["p0"], e["p1"] = list(map(float, pts[i])), list(map(float, pts[(i + 1) % len(els)]))
    els = merge_line_elements(els, 0.015 * length_px)
    pts = np.array([e["p0"] for e in els], float)
    lock = [els[i]["type"] != "line" or els[i - 1]["type"] != "line" for i in range(len(els))]
    frame = dominant_frame(pts) if frame is None else frame
    pts = snap_angles(pts, tol_deg=7.0, lock=lock, frame=frame)
    pts = snap_angles(pts, tol_deg=7.0, lock=lock, frame=frame)
    for i, e in enumerate(els):
        if e["type"] != "line":
            pts[i], pts[(i + 1) % len(els)] = e["p0"], e["p1"]
    for i, e in enumerate(els):
        e["p0"], e["p1"] = list(map(float, pts[i])), list(map(float, pts[(i + 1) % len(els)]))
    return els


def full_ellipse_points(loop, n=96):
    c = np.array([loop["cx"], loop["cy"]], float)
    u = np.array([np.cos(loop["theta"]), np.sin(loop["theta"])])
    v = np.array([-np.sin(loop["theta"]), np.cos(loop["theta"])])
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [c + loop["a"] * np.cos(x) * u + loop["b"] * np.sin(x) * v for x in t]


def ellipse_points(e, steps=24):
    c = np.array([e["cx"], e["cy"]], float)
    u = np.array([np.cos(e["theta"]), np.sin(e["theta"])])
    v = np.array([-np.sin(e["theta"]), np.cos(e["theta"])])
    t0, t1 = e["t0"], e["t1"]
    span = ((t1 - t0) % (2 * np.pi)) if e.get("ccw", True) else -((t0 - t1) % (2 * np.pi))
    return [c + e["a"] * np.cos(t0 + span * f) * u + e["b"] * np.sin(t0 + span * f) * v
            for f in np.linspace(0, 1, steps)]


def loop_ring(loop, steps=16):
    """Sample a loop into points, whatever primitives it is made of."""
    if loop.get("type") == "ellipse":
        return np.array(full_ellipse_points(loop, 4 * steps))
    if loop.get("type") == "circle":
        a = np.linspace(0, 2 * np.pi, 4 * steps)
        return np.stack([loop["cx"] + loop["r"] * np.cos(a), loop["cy"] + loop["r"] * np.sin(a)], axis=1)
    pts = []
    for e in loop.get("elements", []):
        if e["type"] == "line":
            pts.append(np.array(e["p0"][:2], float))
            continue
        if e["type"] == "ellipse":
            pts += list(ellipse_points(e, steps))
            continue
        if e["type"] == "bsplinecurve":
            pts += [np.asarray(q, float) for q in e["xy"]]
            continue
        c = np.array([e["cx"], e["cy"]], float)
        a0 = np.arctan2(e["p0"][1] - c[1], e["p0"][0] - c[0])
        a1 = np.arctan2(e["p1"][1] - c[1], e["p1"][0] - c[0])
        if e.get("ccw", True) and a1 < a0:
            a1 += 2 * np.pi
        if not e.get("ccw", True) and a1 > a0:
            a1 -= 2 * np.pi
        a = np.linspace(a0, a1, steps)
        pts += list(np.stack([c[0] + e["r"] * np.cos(a), c[1] + e["r"] * np.sin(a)], axis=1))
    return np.array(pts, float) if pts else np.zeros((0, 2))


def drop_stray_holes(loops):
    """Remove holes that are not inside the outer loop.

    A hole straddling the boundary makes the padded face invalid, which is how a part reaches a
    build report as a solid with real volume that `isValid()` rejects - 00061 has a 204 px hole
    lying entirely across the outer edge, 00086 has two crossing it. The rest of the loop set is
    fine, so dropping the stray hole is the whole repair.
    """
    if not DROP_STRAY_HOLES or len(loops) < 2:
        return loops
    try:
        from shapely.geometry import Polygon
    except Exception:
        return loops
    rings = [loop_ring(l) for l in loops]
    polys = [Polygon(r) if len(r) >= 4 else None for r in rings]
    polys = [p if p is not None and p.is_valid else None for p in polys]
    outer = max((k for k in range(len(polys)) if polys[k] is not None),
                key=lambda k: polys[k].area, default=None)
    if outer is None:
        return loops
    kept, taken = [], []
    for k, loop in enumerate(loops):
        if k != outer and polys[k] is not None and not polys[k].within(polys[outer]):
            continue
        if k != outer and polys[k] is not None:
            twin = any(polys[k].equals(q) or (polys[k].intersection(q).area
                       > 0.98 * max(polys[k].area, q.area)) for q in taken)
            if twin:
                continue
            taken.append(polys[k])
        kept.append(loop)
    return kept


def is_simple(els):
    """Does this chain of elements enclose a region without crossing itself?"""
    pts = [e["p0"][:2] for e in els if "p0" in e]
    if len(pts) < 4:
        return True
    try:
        from shapely.geometry import Polygon
        return bool(Polygon(np.array(pts, float)).is_valid)
    except Exception:
        return True


def keep_simple(regularised, original):
    """Regularisation may fold a loop back through itself; when it does, keep what went in.

    H/V snapping and angle snapping move vertices independently, so a short feature can invert and
    the polyline cross itself two to four elements later - 00086 does exactly this. The fold is not
    detectable from the silhouette statistics, only from the loop, which is why the check is here.
    """
    return original if is_simple(original) and not is_simple(regularised) else regularised


def reconcile_arcs(els):
    """Make each arc's circle agree with the endpoints it actually ends up with.

    `arc_from_run` projects a run's endpoints onto its own fitted circle, so a corner shared by two
    arcs becomes two different points. Everything downstream enforces one point per corner, keeps
    one of them, and leaves the other arc's endpoint off its own circle: 43% of emitted arcs were
    out by more than 1% of the radius and the worst by 21%. FreeCAD is then handed a centre, a
    radius and two endpoints that disagree, which is where the self-intersecting loops, the -2
    solver returns and the solids that fail `isValid()` come from.

    The centre is moved onto the perpendicular bisector of the chord, keeping the radius as close to
    the fitted one as the chord allows and staying on the side the fit chose. No endpoint moves, so
    the chain stays closed, and both endpoints now lie on the circle exactly.
    """
    if not RECONCILE_ARCS:
        return els
    for e in els:
        if e.get("type") != "arc":
            continue
        p0, p1 = np.array(e["p0"][:2], float), np.array(e["p1"][:2], float)
        chord = p1 - p0
        half = float(np.hypot(*chord)) / 2
        if half < 1e-9:
            continue
        mid = (p0 + p1) / 2
        normal = np.array([-chord[1], chord[0]]) / (2 * half)
        r = max(float(e["r"]), half)
        h = float(np.sqrt(max(r * r - half * half, 0.0)))
        old = np.array([e["cx"], e["cy"]], float)
        candidates = [mid + normal * h, mid - normal * h]
        centre = min(candidates, key=lambda q: float(np.hypot(*(q - old))))
        e["cx"], e["cy"], e["r"] = float(centre[0]), float(centre[1]), r
    return els


def kinds_of(els):
    out = []
    for e in els:
        if e["type"] != "line":
            out.append("A")
            continue
        d = np.subtract(e["p1"], e["p0"])
        out.append("H" if abs(d[1]) < 1e-6 else "V" if abs(d[0]) < 1e-6 else "F")
    return out


def joins(els, tol_deg=10.0):
    out = []
    n = len(els)
    for i in range(n):
        a, b = els[i - 1], els[i]
        if a["type"] == "line" and b["type"] == "line":
            out.append("")
            continue
        if a["type"] == "bsplinecurve" or b["type"] == "bsplinecurve":
            out.append("")
            continue
        def tangent_dir(e, at_end):
            if e["type"] == "line":
                d = np.subtract(e["p1"], e["p0"])
            elif e["type"] == "bsplinecurve":
                q = np.asarray(e["xy"], float)
                d = q[-1] - q[-2] if at_end else q[1] - q[0]
                return d / max(np.hypot(*d), 1e-9)
            else:
                q = np.array(e["p1"] if at_end else e["p0"]); rad = q - np.array([e["cx"], e["cy"]])
                d = np.array([-rad[1], rad[0]]) * (1 if e["ccw"] else -1)
            return d / max(np.hypot(*d), 1e-9)
        da, db = tangent_dir(a, True), tangent_dir(b, False)
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(da, db)), 0, 1)))
        out.append("T" if ang < tol_deg else "P" if abs(ang - 90) < tol_deg else "")
    return out


def fit_ellipse(pts):
    import cv2
    pts = np.asarray(pts, np.float32)
    if len(pts) < 6:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    a, b = max(d1, d2) / 2, min(d1, d2) / 2
    theta = np.radians(ang if d1 >= d2 else ang + 90)
    c, s_ = np.cos(theta), np.sin(theta)
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    u, v = x * c + y * s_, -x * s_ + y * c
    rms = float(np.sqrt(np.mean((np.sqrt((u / max(a, 1e-6)) ** 2 + (v / max(b, 1e-6)) ** 2) - 1) ** 2)) * (a + b) / 2)
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "theta": float(theta), "rms": rms, "aspect": float(b / max(a, 1e-6))}


def primitives(raw_loops, length_px, circle_aspect=0.7, eps=None):
    out = []
    keep_eps = RUN_EPS
    if eps is not None:
        globals()["RUN_EPS"] = float(eps)
    try:
        return _primitives(raw_loops, length_px, circle_aspect)
    finally:
        globals()["RUN_EPS"] = keep_eps


def _primitives(raw_loops, length_px, circle_aspect=0.7):
    out = []
    if SYMMETRY_TOL > 0:
        raw_loops = symmetrise_loops(raw_loops, SYMMETRY_TOL)
    outer = fit_ellipse(np.asarray(raw_loops[0], float))
    hole_aspect = 0.7 * outer["aspect"] if ellipse_ok(outer) else 0.5
    for j, raw in enumerate(raw_loops):
        raw = np.asarray(raw, float)
        f = fit_ellipse(raw)
        toothed = boundary_period(raw) is not None and feature_amplitude(raw) >= CIRCLE_VETO_AMPLITUDE
        if (ellipse_ok(f, len(raw)) and f["aspect"] > (circle_aspect if j == 0 else hole_aspect)
                and not toothed):
            out.append({"type": "circle", "cx": f["cx"], "cy": f["cy"], "r": f["a"]})
            continue
        if (ELLIPSE_ARCS and f is not None and not toothed
                and len(raw) >= MIN_ELLIPSE_POINTS
                and f["rms"] < ELLIPSE_FIT_TOL * f["a"]
                and ELLIPSE_MIN_ASPECT < f["aspect"] < ELLIPSE_MAX_ASPECT):
            out.append({"type": "ellipse", "cx": f["cx"], "cy": f["cy"],
                        "a": f["a"], "b": f["b"], "theta": f["theta"]})
            continue
        traced = elements(raw, length_px)
        if SEQNET:
            from photo2fcstd import seq_infer
            if seq_infer.available():
                proposed = seq_infer.seq_elements(raw, length_px)
                gates_off = os.environ.get("P2F_SEQ_GATES") == "0"
                if proposed not in (None, "round") and (gates_off or (
                        len(proposed) <= len(traced) + 2
                        and seq_infer.drawn_residual(proposed, raw)
                            <= 1.25 * seq_infer.drawn_residual(traced, raw) + 0.002 * length_px)):
                    traced = proposed
        before = [dict(e) for e in traced]
        els = carry_support(rectangularise(regularise_lines(traced, length_px)), before)
        els = reconcile_arcs(keep_simple(els, traced) if KEEP_SIMPLE else els)
        if RADIUS_TOL > 0:
            els = unify_radii(els, RADIUS_TOL)
        out.append({"type": "loop", "elements": els, "kinds": kinds_of(els), "joins": joins(els)})
    return drop_stray_holes(out)

def concentric_edges(gray, mask, f, r_lo=0.15, r_hi=0.92, min_prom=0.25):
    from scipy.signal import find_peaks
    gy, gx = np.gradient(gray.astype(float))
    inner = ndimage.binary_erosion(mask, iterations=max(3, int(0.03 * f["b"])))
    g = np.hypot(gx, gy) * inner
    c, s_ = np.cos(f["theta"]), np.sin(f["theta"])
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    fr = np.linspace(r_lo, r_hi, 300)
    prof = []
    for t in fr:
        u, v = t * f["a"] * np.cos(ang), t * f["b"] * np.sin(ang)
        x = (f["cx"] + u * c - v * s_).round().astype(int)
        y = (f["cy"] + u * s_ + v * c).round().astype(int)
        ok = (x >= 0) & (x < g.shape[1]) & (y >= 0) & (y < g.shape[0])
        prof.append(g[y[ok], x[ok]].mean() if ok.any() else 0.0)
    prof = np.array(prof)
    prof = np.convolve(prof, np.ones(7) / 7, mode="same")
    prof = prof / max(prof.max(), 1e-9)
    peaks, props = find_peaks(prof, prominence=min_prom)
    order = np.argsort(props["prominences"])[::-1]
    return [float(fr[i]) for i in peaks[order][:2]]


def widths(mask):
    def run(row):
        idx = np.where(row)[0]
        return max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len) if len(idx) else None
    rows = [(y, run(mask[y])) for y in range(mask.shape[0])]
    a = np.array([(y, r[-1] - r[0] + 1) for y, r in rows if r is not None], dtype=float)
    return a[:, 0], a[:, 1]


def trim(y, w, tail=TAIL):
    sm = ndimage.uniform_filter1d(w, 9)
    keep = np.where(sm > tail * sm.max())[0]
    return y[keep.min():keep.max() + 1], w[keep.min():keep.max() + 1]


def stations(z, w, min_step, grad):
    dz = float(np.median(np.abs(np.diff(z)))) or 1.0
    size = max(3, int(round(min_step / dz)))
    sm = ndimage.uniform_filter1d(ndimage.uniform_filter1d(w, size), size)
    g = np.gradient(sm, z)
    edges = [z[0]]
    for i in np.argsort(-np.abs(g)):
        if abs(g[i]) < grad:
            break
        if all(abs(z[i] - e) > min_step for e in edges):
            edges.append(z[i])
    edges = sorted(edges + [z[-1]], reverse=True)
    spans = [(a, b, (z <= a) & (z > b)) for a, b in zip(edges[:-1], edges[1:])]
    return [{"z_from": round(float(a), 3), "z_to": round(float(b), 3),
             "width": round(float(np.median(w[sel])), 3)}
            for a, b, sel in spans if sel.sum() >= 8]


def trace(path, ppmm, min_step, grad, tail, mode="rmbg"):
    mask, _ = upright_mask(segment_any(load(path), mode))
    y, w = trim(*widths(mask), tail=tail)
    z = -(y - y[0]) / ppmm
    return stations(z, w / ppmm, min_step, grad), float(-z.min())


def demo():
    z = -np.arange(0, 30, 0.1)
    w = np.where(z > -10, 12.0, np.where(z > -20, 16.0, 6.0))
    st = stations(z, w, MIN_STEP_MM, STEP_GRAD)
    assert [round(s["width"]) for s in st] == [12, 16, 6], st
    y = np.arange(300, dtype=float)
    prof = np.where(y < 200, 16.0, np.where(y < 260, 5.0, 1.5))
    ty, tw = trim(y, prof)
    assert tw.min() >= 5.0 and abs(len(ty) - 260) <= 6, (len(ty), tw.min())
    print("photo2sketch self-check ok: 3 stations found, thin tail kept, wire cut")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    if not args:
        demo()
        raise SystemExit("\nphoto2sketch.py <front.png> [side.png] [--ppmm=20] [--min-step=0.7] "
                         "[--grad=1.6] [--tail=0.25] [--segment=rmbg|auto|dark] [--json=out.json]\n"
                         "images should be rectified by rectify.py; --ppmm skips the two-view check")
    ppmm = float(opt.get("ppmm", PPMM))
    p = (float(opt.get("min-step", MIN_STEP_MM)), float(opt.get("grad", STEP_GRAD)),
         float(opt.get("tail", TAIL)))
    result = {}
    for name, path in zip(("front", "side"), args):
        st, length = trace(path, ppmm, *p, mode=opt.get("segment", "rmbg"))
        result[name] = {"source": path, "length": round(length, 3), "stations": st}
        print("%s: %s  length %.2f mm, %d stations" % (name, path, length, len(st)))
        print("   z from    z to     width")
        print("\n".join("   %7.2f %7.2f   %7.2f" % (s["z_from"], s["z_to"], s["width"]) for s in st))
    if len(result) == 2 and "ppmm" in opt:
        print("\ncross-check SKIPPED: one --ppmm for both views cannot validate scale")
    elif len(result) == 2:
        a, b = result["front"]["length"], result["side"]["length"]
        print("\nlength agreement between views: %.2f vs %.2f mm  (%.1f%%)"
              % (a, b, 100 * abs(a - b) / max(a, b)))
    if "json" in opt:
        with open(opt["json"], "w") as fh:
            json.dump(result, fh, indent=2)
        print("wrote", opt["json"])
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
