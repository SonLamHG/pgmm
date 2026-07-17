import numpy as np
import pytest

from pgmm.metrics.basic import l1, lpips, ssim


@pytest.fixture
def video():
    return np.random.default_rng(0).random((4, 32, 32, 3)).astype(np.float32)


def test_l1_of_identical_is_zero(video):
    assert l1(video, video) == 0.0


def test_l1_of_known_offset(video):
    shifted = np.clip(video * 0 + 0.25, 0, 1)
    target = np.zeros_like(video)
    assert l1(shifted, target) == pytest.approx(0.25)


def test_ssim_of_identical_is_one(video):
    assert ssim(video, video) == pytest.approx(1.0, abs=1e-6)


def test_ssim_decreases_with_noise(video):
    noisy = np.clip(video + 0.2, 0, 1)
    assert ssim(noisy, video) < ssim(video, video)


def test_lpips_of_identical_is_zero(video):
    assert lpips(video, video) == pytest.approx(0.0, abs=1e-5)


def test_lpips_increases_with_corruption(video):
    corrupted = np.clip(video[:, ::-1], 0, 1)
    assert lpips(corrupted, video) > lpips(video, video)


def test_metrics_reject_out_of_range(video):
    bad = video * 255.0
    for fn in (l1, ssim, lpips):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            fn(bad, video)
