import json
import pytest


def test_prompt_names_the_contract_keys():
    from photo2fcstd import recognise
    p = recognise.recognise_prompt(3)
    for key in ("face_photo_index", "outline", "openings", "screws", "depth_ratio"):
        assert key in p


def test_recognition_builds_the_declared_structure(freecad, tmp_path):
    from photo2fcstd import cli, recognise
    import numpy as np, cv2
    img = np.zeros((700, 700), np.uint8)
    cv2.rectangle(img, (150, 150), (550, 550), 255, -1)
    cv2.circle(img, (350, 350), 120, 0, -1)   # a dark central bore
    p = str(tmp_path / "square_bore.png")
    cv2.imwrite(p, img)
    rec = {"face_photo_index": 0, "outline": "square", "openings": ["bore"],
           "screws": 4, "depth_ratio": 0.4}
    spec = recognise.build(rec, [p], name="t")
    kinds = [l["type"] for l in spec["outline"]["loops"]]
    assert kinds.count("loop") == 1 and kinds.count("circle") == 5   # plate + bore + 4 screws
    sp = str(tmp_path / "t.spec.json")
    json.dump(spec, open(sp, "w"))
    r = cli.freecad_build(sp, str(tmp_path / "t.FCStd"))
    assert r["valid"] and r["solids"] == 1
    for s in r["sketches"].values():
        assert s["solve"] == 0


def test_recognise_live_parses_the_model_json(monkeypatch, tmp_path):
    from photo2fcstd import recognise
    import subprocess
    class R:
        stdout = 'here is the answer\n{"face_photo_index": 0, "outline": "disc", "openings": [], "screws": 0, "depth_ratio": 0.3}\nthanks'
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    import numpy as np, cv2
    p = str(tmp_path / "x.png"); cv2.imwrite(p, np.zeros((50, 50), np.uint8))
    rec = recognise.recognise_live([p])
    assert rec["outline"] == "disc" and rec["face_photo_index"] == 0
