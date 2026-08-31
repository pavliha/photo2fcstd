"""Re-read the session's conclusions through a metric that knows which parts can discriminate."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis_data import IDEAL  # noqa: E402
from photo2fcstd import sketch_score as SS  # noqa: E402


def diff_one(part):
    try:
        return part, SS.difficulty(IDEAL[part])
    except Exception:
        return part, None


def main():
    pvc = json.load(open(os.path.join(ROOT, "runs", "photo_vs_carve", "scores.json")))
    tab = np.array(pvc["table"], float)
    parts = pvc["parts"][:len(tab)]
    with Pool(6) as pool:
        d = dict(pool.map(diff_one, parts))
    keep = [(i, p) for i, p in enumerate(parts) if d.get(p) is not None]
    diff = np.array([d[p] for _, p in keep])
    photo, learned, thin, oracle = (np.array([tab[i][c] for i, _ in keep]) for c in range(4))
    json.dump({p: d[p] for _, p in keep}, open(os.path.join(ROOT, "data", "difficulty.json"), "w"))

    print("difficulty over %d parts: median %.2f, quartiles %.2f / %.2f\n"
          % (len(diff), np.median(diff), *np.percentile(diff, [25, 75])))
    print("  %-30s %5s %7s %9s %9s" % ("subset", "n", "photo", "thinnest", "learned"))
    for name, sel in (("everything", diff >= 0.0),
                      ("cannot discriminate (<0.15)", diff < 0.15),
                      ("can discriminate (>=0.15)", diff >= 0.15),
                      ("genuinely complex (>=0.35)", diff >= 0.35)):
        if sel.sum():
            print("  %-30s %5d %7.3f %9.3f %9.3f"
                  % (name, sel.sum(), photo[sel].mean(), thin[sel].mean(), learned[sel].mean()))

    base = 1.0 - diff
    m = diff >= 0.15
    pooled = lambda v: (v[m].mean() - base[m].mean()) / (1.0 - base[m].mean())
    med = lambda v: float(np.median([SS.skill(a, b) for a, b in zip(v[m], base[m])]))
    print("\n  the trivial answer (one circle) scores %.3f on the %d discriminating parts"
          % (base[m].mean(), m.sum()))
    print("  %-24s %8s %8s" % ("", "IoU", "skill"))
    for name, v in (("photo", photo), ("carve, thinnest axis", thin),
                    ("carve, learned axis", learned), ("carve, oracle axis", oracle)):
        print("    %-22s %8.3f %8.3f" % (name, v[m].mean(), pooled(v)))
    print("\n  a circle everywhere would score %.3f overall, against the photo path's %.3f"
          % (base.mean(), photo.mean()))


if __name__ == "__main__":
    main()
