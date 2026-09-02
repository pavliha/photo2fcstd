import glob
import json
import os
import sys


def mine(spec):
    if spec.get("revolve"):
        v = spec["revolve"]
        return {"circle": len(v["holes"]) + 1}
    counts = {}
    for loop in spec["outline"]["loops"]:
        if loop["type"] in ("circle", "ellipse"):
            counts["circle"] = counts.get("circle", 0) + 1
        else:
            for e in loop["elements"]:
                counts[e["type"]] = counts.get(e["type"], 0) + 1
    return counts


def ideal(v):
    c = dict(v.get("counts", {}))
    return c


def jaccard(a, b):
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 1.0


def main(run_dir, ideal_path):
    ideal_all = json.load(open(ideal_path))
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*.spec.json"))):
        k = os.path.basename(p).split(".")[0]
        spec = json.load(open(p))
        if not (spec.get("outline") or spec.get("revolve")) or k not in ideal_all or "counts" not in ideal_all[k]:
            continue
        m, g = mine(spec), ideal(ideal_all[k])
        rows.append((k, jaccard(m, g), m.get("circle", 0) == g.get("circle", 0), m, g))
    if not rows:
        print("no outline/revolve parts to compare")
        return
    print("%d sketch parts: primitive-count Jaccard mean %.2f, circle count exact %d/%d" % (len(rows), sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows), len(rows)))
    for k, j, ok, m, g in sorted(rows, key=lambda r: r[1]):
        print("  %s  %.2f  mine %s  ideal %s" % (k, j, m, g))


def cli(argv):
    main(*argv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/printcad_ideal_sketches.json")
