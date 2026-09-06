import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "data", "regression", "round_parts.json")
PHOTOS = os.path.join(ROOT, "data", "printcad", "PrintCAD", "captured_img")

pytestmark = pytest.mark.skipif(not os.path.isdir(PHOTOS), reason="PrintCAD photos not present")


def test_round_parts_do_not_regress():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import record_regression as RR
    from photo2fcstd import bench, recognise, sketch_score as SS
    golden = json.load(open(GOLDEN))
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    worse = {}
    for part, g in golden.items():
        rec = {"revolve": True, **g["rec"]}
        sp = recognise.revolve_spec(bench.photos_of(part)[:3], rec, name=part)
        sc = SS.score_one(sp, ideal[part])
        f1 = float((sc.get("primitive_f1") or {}).get("f1", 0.0))
        if f1 < g["f1"] - 1e-3:
            worse[part] = (g["f1"], round(f1, 4))
    assert not worse, "round parts regressed (golden -> now): %s" % worse
