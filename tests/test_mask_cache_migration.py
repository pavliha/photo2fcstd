import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "tools")

from photo2fcstd import trace
import migrate_mask_cache as mig


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photos" / "shot_1.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not really a jpeg, but it has a size and an mtime")
    return path


def old_key(old_root, new_root, photo, version, trim):
    identity = os.path.join(old_root, os.path.relpath(photo, new_root))
    st = os.stat(photo)
    raw = "%s|%d|%d" % (identity, st.st_size, int(st.st_mtime))
    if version is not None:
        raw += "|v%s|t%.4f" % (version, trim)
    return hashlib.sha1(raw.encode()).hexdigest()


def test_a_mask_survives_the_photo_moving(tmp_path, photo, monkeypatch):
    masks = tmp_path / "masks"
    masks.mkdir()
    monkeypatch.setattr(mig, "cache_dir", lambda kind: str(masks))
    monkeypatch.setattr(trace, "cache_dir", lambda kind: str(masks))

    new_root = str(photo.parent)
    old_root = "/somewhere/it/used/to/live"
    stale = masks / (old_key(old_root, new_root, photo, trace.MASK_VERSION, trace.TRIM_APPENDAGE) + ".npz")
    np.savez_compressed(stale, mask=np.ones((2, 2), bool))

    scanned, moved = mig.migrate(old_root, new_root, [trace.MASK_VERSION], [trace.TRIM_APPENDAGE], False)

    assert (scanned, moved) == (1, 1)
    assert not stale.exists()
    assert os.path.exists(trace.cached_mask(str(photo)))


def test_a_dry_run_moves_nothing(tmp_path, photo, monkeypatch):
    masks = tmp_path / "masks"
    masks.mkdir()
    monkeypatch.setattr(mig, "cache_dir", lambda kind: str(masks))

    new_root = str(photo.parent)
    old_root = "/somewhere/it/used/to/live"
    stale = masks / (old_key(old_root, new_root, photo, trace.MASK_VERSION, trace.TRIM_APPENDAGE) + ".npz")
    np.savez_compressed(stale, mask=np.ones((2, 2), bool))

    assert mig.migrate(old_root, new_root, [trace.MASK_VERSION], [trace.TRIM_APPENDAGE], True) == (1, 1)
    assert stale.exists()


def test_an_entry_already_at_the_new_key_is_left_alone(tmp_path, photo, monkeypatch):
    masks = tmp_path / "masks"
    masks.mkdir()
    monkeypatch.setattr(mig, "cache_dir", lambda kind: str(masks))
    monkeypatch.setattr(trace, "cache_dir", lambda kind: str(masks))

    new_root = str(photo.parent)
    np.savez_compressed(trace.cached_mask(str(photo)), mask=np.ones((2, 2), bool))

    assert mig.migrate("/old", new_root, [trace.MASK_VERSION], [trace.TRIM_APPENDAGE], False)[1] == 0
