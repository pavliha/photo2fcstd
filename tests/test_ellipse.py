import json
import math
import os

import numpy as np
import pytest

from photo2fcstd import sketch_score as SS, trace


def ellipse_loop(cx=0.0, cy=0.0, a=40.0, b=16.0, theta=0.35):
    return {"type": "ellipse", "cx": cx, "cy": cy, "a": a, "b": b, "theta": theta}


def test_the_scorer_counts_an_ellipse_loop():
    spec = {"outline": {"loops": [ellipse_loop()]}}
    assert SS.counts_of_spec(spec) == {"ellipse": 1}


def test_the_ring_lies_on_the_ellipse():
    loop = ellipse_loop()
    ring = SS.full_ellipse_ring(loop)
    u = np.array([math.cos(loop["theta"]), math.sin(loop["theta"])])
    v = np.array([-math.sin(loop["theta"]), math.cos(loop["theta"])])
    q = ring - np.array([loop["cx"], loop["cy"]])
    r = (q @ u / loop["a"]) ** 2 + (q @ v / loop["b"]) ** 2
    assert np.allclose(r, 1.0, atol=1e-9)


def test_trace_and_score_sample_the_same_points():
    loop = ellipse_loop()
    assert np.allclose(np.array(trace.full_ellipse_points(loop, 96)), SS.full_ellipse_ring(loop, 96))


def test_a_traced_ellipse_becomes_one_primitive():
    import cv2
    from photo2fcstd import analysis, spec as spec_mod
    img = np.zeros((700, 700), np.uint8)
    cv2.ellipse(img, (350, 350), (240, 96), 20, 0, 360, 1, -1)
    view = analysis.view_from_mask(img)
    loops = spec_mod.traced_outline(view)
    assert loops and loops[0]["type"] == "ellipse", loops[0]["type"] if loops else None


def test_a_traced_circle_is_still_a_circle():
    import cv2
    from photo2fcstd import analysis, spec as spec_mod
    img = np.zeros((700, 700), np.uint8)
    cv2.circle(img, (350, 350), 220, 1, -1)
    view = analysis.view_from_mask(img)
    loops = spec_mod.traced_outline(view)
    assert loops and loops[0]["type"] == "circle"


def test_an_ellipse_builds_a_solid_of_the_right_volume(freecad, tmp_path):
    from photo2fcstd import cli
    outer, hole = ellipse_loop(), ellipse_loop(a=12.0, b=5.0)
    depth = 10.0
    spec = {"name": "ell", "unit": "px", "mm_per_px": 1.0, "scale_note": "test", "mode": "plan",
            "outline": {"source": "synthetic", "depth_px": depth, "depth_note": "test",
                        "loops": [outer, hole]}}
    sp = str(tmp_path / "ell.spec.json")
    json.dump(spec, open(sp, "w"))
    report = cli.freecad_build(sp, str(tmp_path / "ell.FCStd"))
    assert report["valid"] and report["solids"] == 1
    want = math.pi * (outer["a"] * outer["b"] - hole["a"] * hole["b"]) * depth
    assert report["volume"] == pytest.approx(want, rel=1e-3)
    for sk in report["sketches"].values():
        assert sk["solve"] == 0 and not sk["redundant"] and not sk["conflicting"]
        assert not sk["dof"]


def test_revolve_keeps_an_ellipse_hole():
    from photo2fcstd import modes
    loops = [{"type": "loop", "elements": []},
             {"type": "circle", "cx": 1.0, "cy": 2.0, "r": 5.0},
             ellipse_loop(cx=3.0, cy=4.0, a=7.0, b=2.0),
             {"type": "loop", "elements": []}]
    kept = modes.round_holes(loops)
    assert [h["type"] for h in kept] == ["circle", "ellipse"]


def test_an_ellipse_hole_reports_its_semi_major_as_the_wall_radius():
    from photo2fcstd import modes
    assert modes.hole_radius(ellipse_loop(a=7.0, b=2.0)) == 7.0
    assert modes.hole_radius({"type": "circle", "cx": 0, "cy": 0, "r": 5.0}) == 5.0


def test_a_revolve_with_an_elliptical_hole_builds(freecad, tmp_path):
    from photo2fcstd import cli
    R, depth, floor = 40.0, 12.0, 3.0
    prof = [[0, 0], [R, 0], [R, depth], [R * 0.8, depth], [R * 0.8, floor], [0, floor]]
    circ = {"type": "circle", "cx": 18.0, "cy": 0.0, "r": 4.0}
    oval = ellipse_loop(cx=-16.0, cy=6.0, a=9.0, b=3.5, theta=0.4)
    spec = {"name": "rev", "unit": "px", "mm_per_px": 1.0, "scale_note": "test", "mode": "revolve",
            "revolve": {"source": "synthetic", "profile": prof, "R": R, "rings": [],
                        "generic": True, "note": "test", "holes": [circ, oval]}}
    sp = str(tmp_path / "rev.spec.json")
    json.dump(spec, open(sp, "w"))
    report = cli.freecad_build(sp, str(tmp_path / "rev.FCStd"))
    assert report["valid"] and report["solids"] == 1
    body = math.pi * R ** 2 * floor + math.pi * (R ** 2 - (R * 0.8) ** 2) * (depth - floor)
    drilled = (math.pi * circ["r"] ** 2 + math.pi * oval["a"] * oval["b"]) * floor
    assert report["volume"] == pytest.approx(body - drilled, rel=2e-3)
    assert len(report["sketches"]["sk_holes"]["redundant"]) == 0


def test_a_bspline_loop_builds_a_valid_solid(freecad, tmp_path):
    import json, numpy as np
    from photo2fcstd import cli
    t = np.linspace(0, np.pi, 9)
    wave = [[float(x * 20), float(18 + 6 * np.sin(3 * x))] for x in t]
    els = [{"type": "bsplinecurve", "p0": wave[0], "p1": wave[-1], "xy": wave},
           {"type": "line", "p0": wave[-1], "p1": [wave[-1][0], 0.0]},
           {"type": "line", "p0": [wave[-1][0], 0.0], "p1": [0.0, 0.0]},
           {"type": "line", "p0": [0.0, 0.0], "p1": wave[0]}]
    from photo2fcstd.trace import joins, kinds_of
    loop = {"type": "loop", "elements": els, "kinds": kinds_of(els), "joins": joins(els)}
    spec = {"name": "spline", "unit": "px", "mm_per_px": 1.0, "scale_note": "test", "mode": "plan",
            "outline": {"source": "synthetic", "depth_px": 8.0, "depth_note": "test", "loops": [loop]}}
    sp = str(tmp_path / "spline.spec.json")
    json.dump(spec, open(sp, "w"))
    report = cli.freecad_build(sp, str(tmp_path / "spline.FCStd"))
    assert report["valid"] and report["solids"] == 1
    for sk in report["sketches"].values():
        assert sk["solve"] == 0 and not sk["conflicting"]
