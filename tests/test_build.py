import json
import os
import pathlib

import pytest
import trimesh

from photo2fcstd import analysis, cli, spec
from photo2fcstd.settings import FREECADCMD


@pytest.fixture(scope="session")
def freecad():
    if not os.path.exists(FREECADCMD):
        pytest.skip("FreeCAD not found (set FREECADCMD)")
    return FREECADCMD


def build(part, tmp_path, photos_of, **kw):
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    views = [analysis.view(p) for p in photos_of(part)[:3]]
    out = str(tmp_path / (part + ".FCStd"))
    doc = spec.assemble(views, name=part, stl=str(tmp_path / (part + ".stl")), log=lambda *a: None, **kw)
    spec_path = os.path.splitext(out)[0] + ".spec.json"
    json.dump(doc, open(spec_path, "w"))
    return doc, cli.freecad_build(spec_path, out), out


@pytest.mark.parametrize("part", ["00476", "01407", "00523"])
def test_model_is_solid_and_fully_constrained(dataset, freecad, photos_of, tmp_path, part):
    doc, report, out = build(part, tmp_path, photos_of)
    assert report["valid"] and report["solids"] == 1
    assert os.path.getsize(out) > 0
    for name, sk in report["sketches"].items():
        assert sk["solve"] == 0, (name, sk)
        assert not sk["redundant"] and not sk["conflicting"], (name, sk)


def test_stl_matches_the_document(dataset, freecad, photos_of, tmp_path):
    doc, report, out = build("00476", tmp_path, photos_of)
    mesh = trimesh.load(str(tmp_path / "00476.stl"))
    assert mesh.volume > 0
    assert max(mesh.extents) == pytest.approx(max(report["bbox"]), rel=0.02)


def test_scale_rescales_the_body(dataset, freecad, photos_of, tmp_path):
    _, unscaled, _ = build("00476", tmp_path / "a", photos_of)
    _, scaled, _ = build("00476", tmp_path / "b", photos_of, length_mm=40.0)
    assert max(scaled["bbox"]) == pytest.approx(40.0, rel=0.05)
    assert max(unscaled["bbox"]) > 100
