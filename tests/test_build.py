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


@pytest.mark.parametrize("part", ["00476", "01407", "00523", "00011", "00061"])
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


def test_editing_the_scale_cell_rescales_the_saved_document(dataset, freecad, photos_of, tmp_path):
    import subprocess
    _, report, out = build("00476", tmp_path, photos_of)
    v0 = report["volume"]
    inside = pathlib.Path(spec.__file__).parents[2] / "_test_rescale.py"
    inside.write_text(
        "import FreeCAD\n"
        "doc = FreeCAD.openDocument(%r)\n"
        "doc.getObject('params').set('B1', '2.0')\n"
        "doc.recompute()\n"
        "body = next(o for o in doc.Objects if o.TypeId == 'PartDesign::Body')\n"
        "print('VOLUME', body.Shape.Volume)\n" % out)
    try:
        r = subprocess.run([freecad, str(inside)], capture_output=True, text=True, timeout=300)
    finally:
        inside.unlink()
    line = next(l for l in r.stdout.splitlines() if l.startswith("VOLUME"))
    assert float(line.split()[1]) == pytest.approx(8.0 * v0, rel=1e-3), r.stdout[-500:]


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


BATCH = ["00476", "01407", "00523", "00011", "00061", "00133", "00171", "00308", "00359", "01289"]
VALID_FLOOR = 1.0


def test_a_sample_of_parts_still_builds(dataset, freecad, photos_of, tmp_path):
    """Every part in this batch must yield a real solid.

    Both wins in this area - dropping holes that straddle the boundary, and the Horizontal
    constraint duplicating a coordinate pin - showed up only in a build, never in a spec-level
    number, and nothing in the suite went red when they were broken. The floor is 100% rather than
    a fraction because at 90% a single regression in ten parts sits exactly on the line and passes.
    """
    import json
    import pathlib
    import subprocess

    from photo2fcstd import analysis, spec

    tmp_path = pathlib.Path(tmp_path)
    rows = []
    for part in BATCH:
        views = [analysis.view(p) for p in photos_of(part)[:3]]
        doc = spec.assemble(views, name=part, log=lambda *a: None)
        sp = tmp_path / (part + ".spec.json")
        json.dump(doc, open(sp, "w"))
        rows.append("%s\t%s" % (sp, tmp_path / (part + ".FCStd")))
    listing = tmp_path / "list.txt"
    listing.write_text("\n".join(rows))
    build_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "photo2fcstd", "build.py")
    proc = subprocess.run([FREECADCMD, build_py], env=dict(os.environ, P2F_LIST=str(listing)),
                          capture_output=True, text=True, timeout=1800)
    reports = {}
    for line in proc.stdout.splitlines():
        if line.startswith("BATCH "):
            _, name, blob = line.split(" ", 2)
            reports[name] = json.loads(blob)
    assert len(reports) == len(BATCH), (len(reports), proc.stderr[-400:])
    valid = [k for k, r in reports.items() if r.get("valid")]
    assert len(valid) >= VALID_FLOOR * len(BATCH), sorted(set(reports) - set(valid))
    free = [(k, n) for k, r in reports.items() for n, sk in r.get("sketches", {}).items() if sk.get("dof")]
    assert not free, free
