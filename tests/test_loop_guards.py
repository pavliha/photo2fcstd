import numpy as np

from photo2fcstd import trace


def square(cx=0.0, cy=0.0, s=10.0):
    p = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
    return {"type": "loop", "elements": [{"type": "line", "p0": list(p[i]), "p1": list(p[(i + 1) % 4])}
                                         for i in range(4)]}


def test_a_hole_straddling_the_boundary_is_dropped():
    loops = [square(0, 0, 100), square(100, 0, 10)]
    assert len(trace.drop_stray_holes(loops)) == 1


def test_a_hole_inside_the_boundary_is_kept():
    loops = [square(0, 0, 100), square(10, 10, 5)]
    assert len(trace.drop_stray_holes(loops)) == 2


def test_the_outer_loop_is_never_dropped():
    loops = [square(0, 0, 100)]
    assert trace.drop_stray_holes(loops) == loops


def test_a_folded_loop_falls_back_to_what_went_in():
    good = square(0, 0, 10)["elements"]
    folded = [{"type": "line", "p0": [0, 0], "p1": [10, 10]},
              {"type": "line", "p0": [10, 10], "p1": [10, 0]},
              {"type": "line", "p0": [10, 0], "p1": [0, 10]},
              {"type": "line", "p0": [0, 10], "p1": [0, 0]}]
    assert trace.keep_simple(folded, good) is good


def test_a_simple_loop_is_left_alone():
    good = square(0, 0, 10)["elements"]
    other = square(1, 1, 9)["elements"]
    assert trace.keep_simple(other, good) is other


def test_reconciled_arc_endpoints_lie_on_their_circle():
    els = [{"type": "arc", "p0": [10.0, 0.0], "p1": [0.0, 10.0], "cx": 3.0, "cy": 3.0, "r": 8.0}]
    trace.reconcile_arcs(els)
    e = els[0]
    c = np.array([e["cx"], e["cy"]])
    for k in ("p0", "p1"):
        assert abs(np.linalg.norm(np.array(e[k]) - c) - e["r"]) < 1e-9
