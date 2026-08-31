import json
import os
import pathlib

import pytest
import trimesh

from photo2fcstd import analysis, cli, spec
from photo2fcstd.settings import FREECADCMD


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


def test_the_sheet_is_in_millimetres_when_the_scale_is_known(dataset, freecad, photos_of, tmp_path):
    doc, report, out = build("00476", tmp_path, photos_of, length_mm=40.0)
    assert doc["unit"] == "mm"
    assert doc["mm_per_px"] == 1.0
    biggest = max(abs(v) for loop in doc["outline"]["loops"] if loop["type"] == "loop"
                  for e in loop["elements"] for v in (e["p0"][0], e["p0"][1]))
    assert biggest < 40.0
    assert max(report["bbox"]) == pytest.approx(40.0, rel=0.05)


def test_the_sheet_says_pixels_when_no_scale_is_given(dataset, freecad, photos_of, tmp_path):
    doc, report, out = build("00476", tmp_path, photos_of)
    assert doc["unit"] == "px"
    assert "pixels" in doc["scale_note"]
    assert max(report["bbox"]) > 100


def test_missing_photos_and_freecad_say_what_to_do(tmp_path, monkeypatch):
    from photo2fcstd import cli
    from photo2fcstd.errors import BuildError, CaptureError
    with pytest.raises(CaptureError, match="cannot find"):
        cli.main(["/no/such/photo.jpg", "--out", str(tmp_path / "x.FCStd")])
    monkeypatch.setattr(cli, "FREECADCMD", "/nonexistent/FreeCADCmd")
    with pytest.raises(BuildError, match="set FREECADCMD"):
        cli.freecad_build(str(tmp_path / "spec.json"), str(tmp_path / "x.FCStd"))
