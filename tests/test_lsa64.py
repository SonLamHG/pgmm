import numpy as np
import pytest

from pgmm.data.lsa64 import load_clip, split_clips


def _all_clip_ids():
    return [
        f"{sign:03d}_{signer:03d}_{rep:03d}"
        for sign in range(1, 65)
        for signer in range(1, 11)
        for rep in range(1, 6)
    ]


def test_random_split_matches_paper_counts():
    train, test = split_clips(_all_clip_ids(), mode="random", seed=0)
    assert len(train) == 2800
    assert len(test) == 400


def test_random_split_is_disjoint_and_total():
    ids = _all_clip_ids()
    train, test = split_clips(ids, mode="random", seed=0)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(ids)


def test_random_split_is_deterministic_given_seed():
    ids = _all_clip_ids()
    assert split_clips(ids, seed=0) == split_clips(ids, seed=0)
    assert split_clips(ids, seed=0) != split_clips(ids, seed=1)


def test_signer_split_holds_out_signers_entirely():
    ids = _all_clip_ids()
    train, test = split_clips(ids, mode="signer", seed=0)
    train_signers = {int(c.split("_")[1]) for c in train}
    test_signers = {int(c.split("_")[1]) for c in test}
    assert train_signers.isdisjoint(test_signers)


def test_unknown_split_mode_rejected():
    with pytest.raises(ValueError, match="unknown split mode"):
        split_clips(_all_clip_ids(), mode="bogus")


def test_load_clip_returns_unit_range_rgb(tmp_path):
    import cv2

    clip_dir = tmp_path / "001_001_001"
    clip_dir.mkdir()
    # pure red in BGR -> must come back as pure red in RGB
    red_bgr = np.zeros((128, 128, 3), np.uint8)
    red_bgr[..., 2] = 255
    for i in range(3):
        cv2.imwrite(str(clip_dir / f"frame_{i:05d}.jpg"), red_bgr)

    clip = load_clip(tmp_path, "001_001_001")
    assert clip.shape == (3, 128, 128, 3)
    assert clip.dtype == np.float32
    assert 0.0 <= clip.min() and clip.max() <= 1.0
    assert clip[0, 64, 64, 0] > 0.9  # red channel first => RGB
    assert clip[0, 64, 64, 2] < 0.1


def test_load_clip_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_clip(tmp_path, "999_999_999")
