import numpy as np
import pytest


def synthetic_carve(w=120, h=70, d=16, vox=0.5):
    xs, ys, zs = np.meshgrid(np.arange(w), np.arange(h), np.arange(d), indexing="ij")
    keep = np.ones(xs.shape, bool)
    keep[(xs - 60) ** 2 + (ys - 35) ** 2 < 12 ** 2] = False
    pts = np.stack([xs[keep], ys[keep], zs[keep]], axis=1).astype(float) * vox
    return {"points_mm": pts, "voxel_mm": vox, "views": 16,
            "extents_mm": np.ptp(pts, axis=0) + vox}


def test_fused_spec_measures_depth_and_keeps_the_hole():
    from photo2fcstd import fuse
    carved = synthetic_carve()
    spec = fuse.fused_spec(carved, name="prism", length_mm=60.0)
    ol = spec["outline"]
    assert ol["depth_trusted"] is True
    depth_mm = ol["depth_px"]
    assert depth_mm == pytest.approx(8.0, rel=0.15)
    assert len(ol["loops"]) == 2


def test_fused_spec_builds_a_valid_solid(freecad, tmp_path):
    import json
    from photo2fcstd import cli, fuse
    spec = fuse.fused_spec(synthetic_carve(), name="prism", length_mm=60.0)
    sp = str(tmp_path / "fused.spec.json")
    json.dump(spec, open(sp, "w"))
    r = cli.freecad_build(sp, str(tmp_path / "fused.FCStd"))
    assert r["valid"] and r["solids"] == 1
    for s in r["sketches"].values():
        assert s["solve"] == 0
