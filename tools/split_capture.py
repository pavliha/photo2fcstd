"""Split the capture error: how much is viewpoint tilt, how much perspective, how much matting.

  ortho at an axis                      the target
  ortho tilted off-axis    -> tilt
  perspective tilted       -> tilt + perspective
  real photo mask (RMBG)   -> tilt + perspective + matting
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np, cv2, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carve_synth import K_of, W, H
from photo2fcstd.bench import photos_of, truth_of
from photo2fcstd.trace import segment_photo, upright_mask

N = 128
TILT = float(os.environ.get("TILT", 30.0))


def canon(mask, n=N):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    c = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    k = (n - 3) / max(c.shape)
    s = cv2.resize(c, (max(int(round(c.shape[1] * k)), 1), max(int(round(c.shape[0] * k)), 1)),
                   interpolation=cv2.INTER_AREA) > 127
    out = np.zeros((n, n), bool)
    y0, x0 = (n - s.shape[0]) // 2, (n - s.shape[1]) // 2
    out[y0:y0 + s.shape[0], x0:x0 + s.shape[1]] = s
    return out


def dih(a):
    return [f(np.rot90(a, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def iou(a, b):
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))


def render(mesh, eye, ortho, f=1400.0):
    target = np.zeros(3)
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0, 0, 1.0]) if abs(fwd[2]) < 0.95 else np.array([0, 1.0, 0])
    r = np.cross(fwd, up); r /= np.linalg.norm(r)
    d = np.cross(fwd, r)
    R = np.stack([r, d, fwd])
    v = (np.asarray(mesh.vertices, float) - eye) @ R.T
    if ortho:
        uv = v[:, :2] * (f / np.linalg.norm(eye)) + [W / 2, H / 2]
    else:
        uv = v[:, :2] / np.maximum(v[:, 2:3], 1e-6) * f + [W / 2, H / 2]
    img = np.zeros((H, W), np.uint8)
    for t in np.round(uv[np.asarray(mesh.faces)]).astype(np.int32):
        cv2.fillConvexPoly(img, t, 1)
    return img > 0


def axis_orthos(mesh):
    g = mesh.voxelized(max(np.max(mesh.extents) / 96.0, 1e-3)).fill()
    p = np.round(g.points / max(np.max(mesh.extents) / 96.0, 1e-3)).astype(int)
    p -= p.min(0)
    vol = np.zeros(p.max(0) + 1, bool)
    vol[p[:, 0], p[:, 1], p[:, 2]] = True
    return [canon(vol.any(axis=2)), canon(vol.any(axis=1)), canon(vol.any(axis=0))]


def one(part):
    try:
        m = trimesh.load(truth_of(part))
        m.apply_translation(-m.bounds.mean(axis=0))
        size = float(np.max(m.extents))
        ort = [o for o in axis_orthos(m) if o is not None]
        if not ort:
            return part, None
        best = lambda mask: max(iou(p, o) for p in dih(mask) for o in ort)
        out = {}
        el = np.radians(TILT)
        eye = np.array([np.cos(el), 0.0, np.sin(el)])
        eye = eye / np.linalg.norm(eye) * size * float(os.environ.get("DIST", 8.0))
        a = canon(render(m, eye, ortho=True))
        b = canon(render(m, eye, ortho=False))
        if a is None or b is None:
            return part, None
        out["tilt"] = best(a)
        out["tilt_persp"] = best(b)
        photo = canon(upright_mask(segment_photo(photos_of(part)[0])[0] if isinstance(segment_photo(photos_of(part)[0]), tuple) else segment_photo(photos_of(part)[0]))[0])
        out["photo"] = best(photo) if photo is not None else None
        return part, out
    except Exception as exc:
        return part, {"error": "%s: %s" % (type(exc).__name__, str(exc)[:60])}


def main():
    ideal = json.load(open("data/printcad_ideal_sketches_all.json"))
    from photo2fcstd.sketch_score import trustworthy
    parts = [p for p in open("data/printcad_all_ids.txt").read().split() if p in ideal and trustworthy(ideal[p])][:60]
    with Pool(6) as pool:
        res = dict(pool.map(one, parts))
    ok = [p for p in parts if res[p] and "error" not in res[p] and res[p].get("photo") is not None]
    errs = [res[p]["error"] for p in parts if res[p] and "error" in res[p]]
    m = lambda k: float(np.mean([res[p][k] for p in ok]))
    print("capture error split, n=%d, tilt %.0f deg (%d errors)" % (len(ok), TILT, len(errs)))
    if errs:
        print("  ", list(dict.fromkeys(errs))[:2])
    t, tp, ph = m("tilt"), m("tilt_persp"), m("photo")
    print("  orthographic, tilted off-axis   %.3f   <- viewpoint alone costs %.3f" % (t, 1 - t))
    print("  perspective,  same viewpoint    %.3f   <- perspective adds        %.3f" % (tp, t - tp))
    print("  real photo mask (RMBG)          %.3f   <- matting adds            %.3f" % (ph, tp - ph))


if __name__ == "__main__":
    main()
