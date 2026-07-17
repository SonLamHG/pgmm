import json

import pytest

from kaggle_harness.chain import latest_checkpoint, push_checkpoint


def _touch(path, step):
    path.mkdir(parents=True, exist_ok=True)
    f = path / f"ckpt_step{step:06d}.pt"
    f.write_bytes(b"x")
    return f


def test_latest_checkpoint_picks_highest_step_across_dirs(tmp_path):
    a = tmp_path / "attached"
    b = tmp_path / "working"
    _touch(a, 1000)
    newest = _touch(b, 2000)
    _touch(a, 500)
    assert latest_checkpoint([a, b]) == newest


def test_latest_checkpoint_prefers_step_number_not_mtime(tmp_path):
    """A resumed session may rewrite an older checkpoint; step order is the
    truth, mtime is not."""
    a = tmp_path / "a"
    newest = _touch(a, 3000)
    stale = _touch(a, 100)
    import os
    import time

    time.sleep(0.01)
    os.utime(stale, None)  # make the OLD checkpoint the NEWEST file
    assert latest_checkpoint([a]) == newest


def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    assert latest_checkpoint([tmp_path]) is None


def test_latest_checkpoint_skips_missing_dirs(tmp_path):
    real = tmp_path / "real"
    ckpt = _touch(real, 10)
    assert latest_checkpoint([tmp_path / "nope", real]) == ckpt


def test_push_checkpoint_builds_version_command(tmp_path):
    ckpt = _touch(tmp_path, 100)
    argv = push_checkpoint(ckpt, slug="user/pgmm-ckpt", message="step 100",
                           dry_run=True)
    assert argv[:3] == ["kaggle", "datasets", "version"]
    assert "-m" in argv and "step 100" in argv
    meta = json.loads((ckpt.parent / "dataset-metadata.json").read_text())
    assert meta["id"] == "user/pgmm-ckpt"


def test_push_checkpoint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        push_checkpoint(tmp_path / "nope.pt", slug="user/x", message="m",
                        dry_run=True)
