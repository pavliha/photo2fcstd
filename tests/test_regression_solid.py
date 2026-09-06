import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "data", "regression", "solid_parts.json")
FREECAD = os.path.expanduser(os.environ.get("FREECADCMD", "~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
PHOTOS = os.path.join(ROOT, "data", "printcad", "PrintCAD", "captured_img")

pytestmark = pytest.mark.skipif(not (os.path.exists(FREECAD) and os.path.isdir(PHOTOS)), reason="FreeCAD or PrintCAD photos not present")


def test_built_solids_do_not_regress_against_step_truth(tmp_path, monkeypatch):
    import trimesh
    from photo2fcstd import bench, design, score
    monkeypatch.setenv("P2F_FACEPOSE", "0")
    golden = json.load(open(GOLDEN)); recs = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
    worse = {}
    for part, want in golden.items():
        out = str(tmp_path / (part + ".FCStd"))
        res = design.design(bench.photos_of(part)[:3], out, rec=recs[part])
        if res.get("model", 1) is None:
            worse[part] = (want, "refused: " + res["reason"]); continue
        subprocess.run([FREECAD, os.path.join(ROOT, "tools", "dump_shape.py")], capture_output=True, timeout=300,
                       env={**os.environ, "FC_IN": out, "FC_MESH": out + ".npz", "FC_INFO": out + ".json"})
        m = np.load(out + ".npz")
        got, _ = score.best_iou(trimesh.load(bench.truth_of(part)), trimesh.Trimesh(m["V"], m["T"]))
        if got < want - 0.05:
            worse[part] = (want, round(float(got), 3))
    assert not worse, "solids regressed against STEP truth (golden -> now): %s" % worse
