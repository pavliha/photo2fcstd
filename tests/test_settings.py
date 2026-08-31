import os

import pytest

from photo2fcstd import doctor, settings


def test_freecad_env_var_wins(monkeypatch):
    monkeypatch.setenv("FREECADCMD", "/somewhere/FreeCADCmd")
    assert settings.find_freecad() == "/somewhere/FreeCADCmd"


def test_freecad_is_found_on_the_path(monkeypatch):
    monkeypatch.delenv("FREECADCMD", raising=False)
    monkeypatch.setattr(settings.shutil, "which", lambda name: "/opt/bin/" + name if name == "FreeCADCmd" else None)
    assert settings.find_freecad() == "/opt/bin/FreeCADCmd"


def test_freecad_falls_back_to_known_install_locations(monkeypatch):
    monkeypatch.delenv("FREECADCMD", raising=False)
    monkeypatch.setattr(settings.shutil, "which", lambda name: None)
    monkeypatch.setattr(settings.os.path, "exists", lambda p: p == settings.FREECAD_GUESSES[1])
    assert settings.find_freecad() == settings.FREECAD_GUESSES[1]


def test_a_machine_without_freecad_still_reports_a_path(monkeypatch):
    monkeypatch.delenv("FREECADCMD", raising=False)
    monkeypatch.setattr(settings.shutil, "which", lambda name: None)
    monkeypatch.setattr(settings.os.path, "exists", lambda p: False)
    assert settings.find_freecad() in settings.FREECAD_GUESSES


def test_data_dir_prefers_the_env_then_the_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("P2F_DATA", str(tmp_path))
    assert settings.data_dir() == str(tmp_path)
    monkeypatch.delenv("P2F_DATA")
    monkeypatch.setattr(settings.os.path, "isdir", lambda p: p == settings.DATA_GUESSES[0])
    assert settings.data_dir() == settings.DATA_GUESSES[0]


def test_doctor_reports_a_missing_freecad_without_raising(monkeypatch):
    monkeypatch.setattr(doctor, "FREECADCMD", "/nonexistent/FreeCADCmd")
    rows = doctor.checks()
    freecad = [r for r in rows if r[0] == "FreeCAD"][0]
    assert not freecad[1]
    assert "set FREECADCMD" in freecad[2]


def test_doctor_passes_on_this_machine():
    rows = doctor.checks()
    broken = [name for name, ok, _ in rows if not ok and name in ("required packages", "FreeCAD")]
    assert not broken, broken
