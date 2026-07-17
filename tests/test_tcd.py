import numpy as np
import pytest

from pgmm.metrics.tcd import TCDVariant, tcd


def test_static_video_perfectly_reconstructed_scores_zero():
    frame = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    video = np.stack([frame] * 5)
    assert tcd(video, video) == 0.0


def test_all_pixels_exceeding_threshold_scores_one():
    target = np.zeros((3, 4, 4, 3), np.float32)
    generated = np.ones((3, 4, 4, 3), np.float32)
    assert tcd(generated, target) == 1.0


def test_known_fraction_of_deviating_pixels():
    target = np.zeros((3, 4, 4, 3), np.float32)
    generated = np.zeros((3, 4, 4, 3), np.float32)
    generated[1, 0, :, :] = 1.0  # one row of four -> 4/16 pixels
    assert tcd(generated, target) == 0.25


def test_penalises_perfect_reconstruction_of_fast_motion():
    """TCD is not reconstruction error: it compares against the temporal
    midpoint, so a perfectly reconstructed fast change still scores 1.0.
    This is why the paper's values sit near 0.13, not near 0."""
    target = np.zeros((3, 4, 4, 3), np.float32)
    target[1] = 1.0
    assert tcd(target.copy(), target) == 1.0


def test_averages_over_interior_frames_only():
    target = np.zeros((4, 4, 4, 3), np.float32)
    generated = np.zeros((4, 4, 4, 3), np.float32)
    generated[0] = 1.0  # first frame is not interior -> ignored
    generated[3] = 1.0  # last frame is not interior -> ignored
    assert tcd(generated, target) == 0.0


def test_variants_disagree_on_single_channel_deviation():
    """The point of D3: the channel reduction changes the answer, so the
    reading cannot be chosen arbitrarily."""
    target = np.zeros((3, 4, 4, 3), np.float32)
    generated = np.zeros((3, 4, 4, 3), np.float32)
    generated[1, :, :, 0] = 0.9  # red only
    assert tcd(generated, target, variant=TCDVariant.UNIT_MEAN) == 0.0  # 0.3 < 0.5
    assert tcd(generated, target, variant=TCDVariant.UNIT_MAX) == 1.0   # 0.9 > 0.5
    assert tcd(generated, target, variant=TCDVariant.UNIT_SUM) == 1.0   # 0.9 > 0.5
    assert tcd(generated, target, variant=TCDVariant.BYTE_MEAN) == 1.0  # 76.5 > 0.5


def test_rejects_videos_shorter_than_three_frames():
    v = np.zeros((2, 4, 4, 3), np.float32)
    with pytest.raises(ValueError, match="at least 3 frames"):
        tcd(v, v)


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        tcd(np.zeros((3, 4, 4, 3), np.float32), np.zeros((3, 8, 8, 3), np.float32))


def test_rejects_out_of_range_values():
    target = np.zeros((3, 4, 4, 3), np.float32)
    generated = np.full((3, 4, 4, 3), 255.0, np.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tcd(generated, target)
