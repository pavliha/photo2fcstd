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
