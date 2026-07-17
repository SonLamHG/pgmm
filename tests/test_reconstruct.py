import numpy as np
import pytest

from pgmm.eval.reconstruct import evaluate_clips
from pgmm.metrics.tcd import TCDVariant


@pytest.fixture
def prepared(tmp_path):
    import cv2

    rng = np.random.default_rng(0)
    for clip_id in ("001_001_001", "002_002_002"):
        d = tmp_path / clip_id
        d.mkdir()
        for i in range(5):
            frame = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"frame_{i:05d}.jpg"), frame)
    return tmp_path


def test_perfect_reconstruction_scores_ideally(prepared):
    scores = evaluate_clips(lambda clip: clip.copy(), prepared,
                            ["001_001_001", "002_002_002"])
    assert scores["l1"] == pytest.approx(0.0, abs=1e-6)
    assert scores["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert scores["lpips"] == pytest.approx(0.0, abs=1e-4)


def test_returns_all_expected_metrics(prepared):
    scores = evaluate_clips(lambda clip: clip.copy(), prepared, ["001_001_001"])
    assert set(scores) == {"l1", "ssim", "lpips", "tcd"}


def test_worse_reconstruction_scores_worse(prepared):
    good = evaluate_clips(lambda c: c.copy(), prepared, ["001_001_001"])
    bad = evaluate_clips(lambda c: np.zeros_like(c), prepared, ["001_001_001"])
    assert bad["l1"] > good["l1"]
    assert bad["ssim"] < good["ssim"]


def test_tcd_variant_is_threaded_through(prepared):
    """The harness must not silently pin one reading — M1 sweeps them."""
    mean = evaluate_clips(lambda c: c.copy(), prepared, ["001_001_001"],
                          tcd_variant=TCDVariant.UNIT_MEAN)
    byte = evaluate_clips(lambda c: c.copy(), prepared, ["001_001_001"],
                          tcd_variant=TCDVariant.BYTE_MEAN)
    assert byte["tcd"] > mean["tcd"]


def test_rejects_shape_changing_reconstruction(prepared):
    with pytest.raises(ValueError, match="same shape"):
        evaluate_clips(lambda c: c[:, :64, :64], prepared, ["001_001_001"])


def test_rejects_empty_clip_list(prepared):
    with pytest.raises(ValueError, match="no clips"):
        evaluate_clips(lambda c: c.copy(), prepared, [])
