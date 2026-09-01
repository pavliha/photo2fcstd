import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORT = os.path.join(DATA, "ood_audit.json")

DOCUMENTED = {
    "axis_model": {"in": 0.892, "out": 0.286},
    "depth_model_shipped": {"out": 1.117},
    "depth_model_tabular": {"out": 0.236},
    "view_model": {"in": 0.560},
}
TOLERANCE = 0.06


def report():
    if not os.path.exists(REPORT):
        pytest.skip("run tools/ood_audit.py to produce %s" % REPORT)
    return json.load(open(REPORT))


@pytest.mark.parametrize("name", sorted(DOCUMENTED))
def test_scores_match_what_the_docs_claim(name):
    got = report().get(name)
    if got is None:
        pytest.skip("%s not in the audit" % name)
    for key, expected in DOCUMENTED[name].items():
        actual = got.get(key)
        if actual is None:
            continue
        assert abs(actual - expected) <= TOLERANCE, (
            "%s %s is %.3f, documented as %.3f - retrain, or update the docs and this test"
            % (name, key, actual, expected))


def test_the_gate_catches_a_dataset_specific_component():
    """The axis classifier is the worked example: at chance out of distribution, and only the
    section_constancy warning stands between it and a confidently wrong answer."""
    got = report().get("axis_model")
    if got is None:
        pytest.skip("no axis row")
    assert got["out"] <= got["baseline"] + 0.05, "axis_model no longer fails out of distribution"
    from photo2fcstd.carve import PRISM_CONSTANCY, section_constancy
    g = np.mgrid[0:30, 0:20, 0:12]
    prism = {"points_mm": np.column_stack([a.ravel() for a in g]).astype(float), "voxel_mm": 1.0}
    assert section_constancy(prism, 2) >= PRISM_CONSTANCY


def test_the_shipped_depth_path_is_worse_than_its_fallback_off_distribution():
    r = report()
    a, b = r.get("depth_model_shipped"), r.get("depth_model_tabular")
    if not (a and b and a.get("out") and b.get("out")):
        pytest.skip("both depth arms not recorded")
    assert a["out"] > b["out"], (
        "the pixel path no longer loses to the tabular one out of distribution - if this is a real "
        "improvement, update CLAUDE.md and DOCUMENTED")
