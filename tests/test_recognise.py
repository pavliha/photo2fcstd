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


def test_feature_program_builds_a_box_with_a_bore(freecad, tmp_path):
    import json
    from photo2fcstd import cli
    def sq(a):
        c = [(-a,-a),(a,-a),(a,a),(-a,a)]
        return {"type":"loop","elements":[{"type":"line","p0":list(c[i]),"p1":list(c[(i+1)%4])} for i in range(4)],
                "kinds":["H","V","H","V"],"joins":["","","",""]}
    spec = {"name":"enc","mode":"features","mm_per_px":1.0,"unit":"px","scale_note":"t",
            "views":{},"outline":None,"revolve":None,"stl":None,"measured":[],
            "features":[{"op":"pad","depth_px":40.0,"loops":[sq(50)]},
                        {"op":"pocket","through":True,"loops":[{"type":"circle","cx":0,"cy":0,"r":35}]}]}
    sp = str(tmp_path/"enc.spec.json"); json.dump(spec, open(sp,"w"))
    r = cli.freecad_build(sp, str(tmp_path/"enc.FCStd"))
    import math
    want = 100*100*40 - math.pi*35*35*40
    assert r["valid"] and r["solids"] == 1
    assert abs(r["volume"] - want) / want < 0.02


def test_feature_scale_cell_drives_all_features(freecad, tmp_path):
    import json, subprocess, pathlib
    from photo2fcstd import cli, spec as spec_mod
    def sq(a):
        c = [(-a,-a),(a,-a),(a,a),(-a,a)]
        return {"type":"loop","elements":[{"type":"line","p0":list(c[i]),"p1":list(c[(i+1)%4])} for i in range(4)],
                "kinds":["H","V","H","V"],"joins":["","","",""]}
    spec = {"name":"enc","mode":"features","mm_per_px":1.0,"unit":"px","scale_note":"t",
            "views":{},"outline":None,"revolve":None,"stl":None,"measured":[],
            "features":[{"op":"pad","depth_px":40.0,"loops":[sq(50)]},
                        {"op":"pocket","through":True,"loops":[{"type":"circle","cx":0,"cy":0,"r":35}]}]}
    out = str(tmp_path/"enc.FCStd"); sp = str(tmp_path/"enc.spec.json"); json.dump(spec, open(sp,"w"))
    r0 = cli.freecad_build(sp, out)
    inside = pathlib.Path(spec_mod.__file__).parents[2] / "_test_featscale.py"
    inside.write_text("import FreeCAD\n"
        "doc = FreeCAD.openDocument(%r)\n"
        "doc.getObject('params').set('B1','2.0')\n"
        "doc.recompute()\n"
        "b = next(o for o in doc.Objects if o.TypeId=='PartDesign::Body')\n"
        "print('VOL', b.Shape.Volume)\n" % out)
    try:
        rr = subprocess.run([freecad, str(inside)], capture_output=True, text=True, timeout=300)
    finally:
        inside.unlink()
    vol = float(next(l for l in rr.stdout.splitlines() if l.startswith("VOL")).split()[1])
    assert abs(vol - 8.0*r0["volume"]) / (8.0*r0["volume"]) < 0.02


def test_grille_is_a_valid_connected_web(freecad, tmp_path):
    import json
    from photo2fcstd import cli, recognise
    loops = recognise.grille_loops(0, 0, 100, n_rings=4, n_spokes=2, web=0.10)
    assert loops[0]["type"] == "circle" and len(loops) > 4   # outer disc + sector openings
    spec = {"name": "g", "mode": "features", "mm_per_px": 1.0, "unit": "px", "scale_note": "t",
            "views": {}, "outline": None, "revolve": None, "stl": None, "measured": [],
            "features": [{"op": "pad", "depth_px": 4.0, "loops": loops}]}
    sp = str(tmp_path / "g.spec.json"); json.dump(spec, open(sp, "w"))
    r = cli.freecad_build(sp, str(tmp_path / "g.FCStd"))
    assert r["valid"] and r["solids"] == 1                    # one connected web
    for s in r["sketches"].values():
        assert s["solve"] == 0


def test_fan_guard_params_measures_bore_from_photo(tmp_path):
    import numpy as np, cv2
    from photo2fcstd import recognise, trace
    trace.RECOVER_DARK = True
    img = np.zeros((700, 700), np.uint8)
    cv2.rectangle(img, (150, 150), (550, 550), 255, -1)
    cv2.circle(img, (350, 350), 150, 0, -1)
    p = str(tmp_path / "fan.png"); cv2.imwrite(p, img)
    f = trace.cached_mask(p)
    import os
    os.path.exists(f) and os.remove(f)
    rec = {"face_photo_index": 0, "outline": "square", "openings": ["bore"],
           "screws": 4, "grille": {"rings": 6, "spokes": 2}, "single_extrusion": False}
    par = recognise.fan_guard_params(rec, [p], frame_w_mm=80.0)
    assert par["rings"] == 6
    assert 0 < par["bore_d"] < par["frame_w"]
    assert abs(par["bore_d"] - 80.0 * 300 / 400) < 8   # bore 300px in a 400px frame


def test_design_refuses_when_no_path_applies(tmp_path):
    from photo2fcstd import design
    res = design.design(["/nonexistent.jpg"], str(tmp_path / "x.FCStd"),
                        rec={"part_class": None, "revolve": False, "single_extrusion": False})
    assert res["model"] is None and "no template" in res["reason"] and res["reshoot"]
    assert not (tmp_path / "x.FCStd").exists()


def test_verify_gate_refuses_a_wrong_silhouette(freecad, tmp_path):
    import numpy as np, cv2, os
    from photo2fcstd import design, trace
    tall = np.zeros((600, 400), np.uint8)
    cv2.rectangle(tall, (150, 300), (250, 560), 255, -1); cv2.rectangle(tall, (180, 120), (220, 300), 255, -1)
    p = str(tmp_path / "bottle.png"); cv2.imwrite(p, tall)
    sq = np.zeros((600, 600), np.uint8); cv2.rectangle(sq, (100, 100), (500, 500), 255, -1)
    q = str(tmp_path / "square.png"); cv2.imwrite(q, sq)
    for f in (p, q):
        os.path.exists(trace.cached_mask(f)) and os.remove(trace.cached_mask(f))
    ok = design.design([p], str(tmp_path / "ok.FCStd"), rec={"part_class": None, "revolve": True, "face_photo_index": 0})
    assert ok.get("model", 1) is not None and ok["verify"]["silhouette_iou"] >= 0.8
    v = design.verify(str(tmp_path / "ok.FCStd"), [q], {"face_photo_index": 0}, threshold=0.8)
    assert v["confidence"] == "low" and v["silhouette_iou"] < 0.8


def test_revolve_tier_builds_a_solid_of_revolution(freecad, tmp_path, monkeypatch):
    import numpy as np, cv2, os
    from photo2fcstd import design, recognise, trace
    img = np.zeros((600, 400), np.uint8)             # a tapered bottle silhouette
    cv2.rectangle(img, (150, 300), (250, 560), 255, -1)
    cv2.rectangle(img, (180, 120), (220, 300), 255, -1)
    p = str(tmp_path / "bottle.png"); cv2.imwrite(p, img)
    os.path.exists(trace.cached_mask(p)) and os.remove(trace.cached_mask(p))
    spec = recognise.revolve_spec([p], {"face_photo_index": 0}, name="b")
    prof = spec["revolve"]["profile"]
    assert len(prof) > 5 and prof[0][0] == 0.0 and prof[-1][0] == 0.0   # closes on the axis
    res = design.design([p], str(tmp_path / "b.FCStd"), rec={"part_class": None, "revolve": True, "face_photo_index": 0})
    assert res["tier"] == "revolve" and res["valid"] and res["solids"] == 1


def test_fit_fan_guard_measures_corner_bore_and_pitch(tmp_path):
    import numpy as np, cv2, os
    from photo2fcstd import recognise, trace
    img = np.zeros((700, 700, 3), np.uint8)
    yellow = (0, 200, 230)                                   # BGR: a saturated frame
    cv2.rectangle(img, (190, 150), (510, 550), yellow, -1)   # rounded 400x400 square, r=40
    cv2.rectangle(img, (150, 190), (550, 510), yellow, -1)
    for c in ((190, 190), (510, 190), (190, 510), (510, 510)):
        cv2.circle(img, c, 40, yellow, -1)
    cv2.circle(img, (350, 350), 120, (0, 0, 0), -1)          # bore, d = 240 px
    for c in ((206, 206), (494, 206), (206, 494), (494, 494)):
        cv2.circle(img, c, 12, (255, 255, 255), -1)          # screws at pitch 288 px
    p = str(tmp_path / "guard.png"); cv2.imwrite(p, img)
    os.path.exists(trace.cached_mask(p)) and os.remove(trace.cached_mask(p))
    params, ledger = recognise.fit_fan_guard(p, frame_w_mm=80.0)
    assert ledger["corner_r"] == "measured" and abs(params["corner_r"] - 8.0) < 1.5
    assert ledger["bore_d"] == "measured" and abs(params["bore_d"] - 48.0) < 4.0
    assert ledger["mount_pitch"] == "measured" and abs(params["mount_pitch"] - 57.6) < 5.0
