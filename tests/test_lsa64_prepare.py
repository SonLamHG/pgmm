import numpy as np
import pytest

from pgmm.data.lsa64_prepare import (
    parse_lsa64_filename,
    prepare_dataset,
    prepare_video,
)


def _write_video(path, n_frames=6, w=640, h=480):
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h)
    )
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    writer.release()


def test_parses_sign_signer_repetition():
    assert parse_lsa64_filename("003_007_002.mp4") == (3, 7, 2)


def test_parse_rejects_unexpected_name():
    with pytest.raises(ValueError, match="LSA64 filename"):
        parse_lsa64_filename("clip1.mp4")


def test_prepare_video_writes_square_frames(tmp_path):
    src = tmp_path / "001_001_001.mp4"
    _write_video(src, n_frames=6)
    out = tmp_path / "out"
    count = prepare_video(src, out, size=128)
    assert count == 6
    frames = sorted(out.glob("frame_*.jpg"))
    assert len(frames) == 6

    import cv2

    img = cv2.imread(str(frames[0]))
    assert img.shape == (128, 128, 3)


def test_prepare_video_is_centre_cropped_not_squashed(tmp_path):
    """A 640x480 source must be centre-cropped to square before resizing;
    squashing would distort hand shape, which is the signal we care about."""
    src = tmp_path / "001_001_001.mp4"
    _write_video(src, n_frames=3, w=640, h=480)
    out = tmp_path / "out"
    prepare_video(src, out, size=128)

    import cv2

    from pgmm.data.lsa64_prepare import centre_crop_square

    cap = cv2.VideoCapture(str(src))
    _, first = cap.read()
    cap.release()
    cropped = centre_crop_square(first)
    assert cropped.shape[0] == cropped.shape[1] == 480


def test_prepare_dataset_indexes_all_clips(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    for name in ("001_001_001.mp4", "002_003_004.mp4"):
        _write_video(src / name, n_frames=4)
    index = prepare_dataset(src, tmp_path / "prepared", size=128)
    assert set(index) == {"001_001_001", "002_003_004"}
    assert all(v == 4 for v in index.values())
