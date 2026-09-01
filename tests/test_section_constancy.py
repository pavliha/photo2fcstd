import numpy as np

from photo2fcstd.carve import PRISM_CONSTANCY, section_constancy, spec_from_carve


def carved(points):
    p = np.asarray(points, float)
    return {"points_mm": p, "voxel_mm": 1.0, "extents_mm": np.ptp(p, axis=0) + 1.0,
            "views": 16, "sources": []}


def prism():
    g = np.mgrid[0:30, 0:20, 0:12]
    return carved(np.column_stack([a.ravel() for a in g]))


def cone():
    pts = [(x, y, z) for z in range(20)
           for x in range(30 - z) for y in range(20 - z) if x < 30 - z and y < 20 - z]
    return carved(pts)


def test_a_prism_has_a_constant_section():
    assert section_constancy(prism(), 2) > 0.95


def test_a_tapering_part_does_not():
    assert section_constancy(cone(), 2) < PRISM_CONSTANCY


def test_the_spec_warns_only_when_the_section_changes():
    assert "warning" not in spec_from_carve(prism(), axis=2)["outline"]
    assert "warning" in spec_from_carve(cone(), axis=2)["outline"]


def test_constancy_is_always_reported():
    assert 0.0 <= spec_from_carve(prism(), axis=2)["outline"]["section_constancy"] <= 1.0
