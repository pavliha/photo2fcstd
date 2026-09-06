import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "data", "regression", "img3222")
FREECAD = os.path.expanduser(os.environ.get("FREECADCMD", "~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
REC = {"face_photo_index": 2, "revolve": False, "part_class": "fan_guard", "single_extrusion": False,
       "openings": ["bore"], "grille": {"rings": 4, "spokes": 4}}

pytestmark = pytest.mark.skipif(not os.path.exists(FREECAD), reason="FreeCADCmd not present")


def test_fan_from_video_frames_with_hand_free_masks(tmp_path, monkeypatch):
    from photo2fcstd import design
    monkeypatch.setenv("P2F_MASK_DIR", CASE)
    photos = [os.path.join(CASE, f) for f in ("f028.jpg", "f001.jpg", "f027.jpg")]
    out = str(tmp_path / "fan.FCStd")
    res = design.design(photos, out, rec=REC, frame_w_mm=80.0, depth_json=os.path.join(CASE, "extents.json"))
    assert res.get("model", 1) is not None, res
    p = res["params"]
    assert 54 <= p["bore_d"] <= 60, p["bore_d"]
    assert 23 <= p["box_depth"] <= 26, p["box_depth"]
    assert p["rings"] in (4, 5), p["rings"]
    assert p["_ledger"]["box_depth"].startswith("measured (3D")
    assert res["verify"]["silhouette_iou"] >= 0.85, res["verify"]
