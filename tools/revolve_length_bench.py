import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def truth_ld(mesh):
    e = np.sort(mesh.extents)
    return float(e[2] / e[0]) if e[1] / e[0] < 1.15 or e[2] / e[1] > 1.15 else float(e[0] / e[2])


def rows_for(parts):
    import trimesh
    from photo2fcstd import bench, recognise
    recs = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
    out = []
    for part in parts:
        rec = recs[part]
        if not rec.get("revolve"):
            continue
        photos = bench.photos_of(part)[:3]
        try:
            sp = recognise.revolve_spec(photos, rec, name=part)
            prof = np.array(sp["revolve"]["profile"], float); R = float(np.abs(prof[:, 0]).max()); L = float(prof[:, 1].max())
            m = trimesh.load(bench.truth_of(part)); e = np.sort(m.extents)[::-1]
            t_ld = float(e[0] / e[2]) if e[0] / e[1] > 1.1 else float(e[2] / e[0])
            out.append({"part": part, "ld_model": round(L / (2 * R), 3), "ld_truth": round(t_ld, 3), "err": round(float(np.log((L / (2 * R)) / t_ld)), 3),
                        "note": sp["revolve"]["length_note"][:80], "side_view": sp["revolve"].get("side_view")})
        except Exception as ex:
            out.append({"part": part, "error": str(ex)[:80]})
    return out


def main():
    parts = open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split()
    rows = rows_for(parts); ok = [r for r in rows if "err" in r]
    err = np.array([r["err"] for r in ok])
    print("revolve length: n=%d  |log ratio| median %.3f  within 10%%: %d  within 25%%: %d  >2x off: %d" % (len(ok), np.median(np.abs(err)), (np.abs(err) < 0.095).sum(), (np.abs(err) < 0.223).sum(), (np.abs(err) > 0.69).sum()))
    guess = [r for r in ok if "guess" in r["note"]]; side = [r for r in ok if "guess" not in r["note"]]
    print("  side view found n=%d: |err| median %.3f within 25%% %d | no side view (guess) n=%d: |err| median %.3f" % (len(side), np.median(np.abs([r["err"] for r in side])), sum(abs(r["err"]) < 0.223 for r in side), len(guess), np.median(np.abs([r["err"] for r in guess])) if guess else 0))
    print("  worst:", [(r["part"], r["ld_model"], r["ld_truth"], r["note"][:22]) for r in sorted(ok, key=lambda r: -abs(r["err"]))[:10]])
    json.dump(rows, open(os.path.join(ROOT, "runs", "revolve_length.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
