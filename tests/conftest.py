import glob
import os

import pytest

from photo2fcstd.settings import data_dir


def photos(part):
    return sorted(glob.glob(os.path.join(data_dir(), "captured_img", "*", part + "_*.jpg")))


@pytest.fixture(scope="session")
def dataset():
    if not photos("00476"):
        pytest.skip("PrintCAD dataset not present (set P2F_DATA)")
    return data_dir()


@pytest.fixture(scope="session")
def photos_of():
    return photos


@pytest.fixture(scope="session")
def freecad():
    from photo2fcstd.settings import FREECADCMD
    if not os.path.exists(FREECADCMD):
        pytest.skip("FreeCAD not found (set FREECADCMD)")
    return FREECADCMD
