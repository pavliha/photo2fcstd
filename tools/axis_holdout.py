import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis_data import IDEAL, one  # noqa: E402
from photo2fcstd import sketch_score as SS  # noqa: E402


def main():
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    seen = {r["part"] for r in json.load(open(os.path.join(ROOT, "data", "axis_rows.json")))}
    fresh = [p for p in parts if p not in seen][:400]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, fresh) if r]
    json.dump(rows, open(os.path.join(ROOT, "data", "axis_holdout.json"), "w"))
    print("%d fresh parts labelled" % len(rows))


if __name__ == "__main__":
    main()
