"""Pre-extract LSA64 videos into 128x128 JPEG frames.

Kaggle gives ~4 CPU cores, which cannot decode video fast enough to keep two
T4s fed. Decoding once, offline, trades a one-time cost for a training run
that is GPU-bound rather than CPU-bound.
"""

import json
import re
from pathlib import Path

import cv2

_NAME_RE = re.compile(r"^(\d{3})_(\d{3})_(\d{3})$")


def parse_lsa64_filename(name: str) -> tuple[int, int, int]:
    """Parse LSA64's ``<sign>_<signer>_<repetition>.mp4`` convention.

    Returns 1-based (sign, signer, repetition).
    """
    stem = Path(name).stem
    match = _NAME_RE.match(stem)
    if not match:
        raise ValueError(
            f"not an LSA64 filename (expected NNN_NNN_NNN.mp4): {name!r}"
        )
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def centre_crop_square(frame):
    """Crop the largest centred square. Preserves aspect ratio, so hand shape
    is not distorted by the subsequent resize."""
    h, w = frame.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return frame[top : top + side, left : left + side]


def prepare_video(src: Path, dst_dir: Path, size: int = 128) -> int:
    """Decode ``src``, centre-crop to square, resize to ``size``, write JPEGs.

    Returns the number of frames written.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {src}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            square = centre_crop_square(frame)
            resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                str(dst_dir / f"frame_{count:05d}.jpg"),
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            count += 1
    finally:
        capture.release()
    if count == 0:
        raise ValueError(f"decoded zero frames from {src}")
    return count


def prepare_dataset(src_dir: Path, dst_dir: Path, size: int = 128) -> dict[str, int]:
    """Prepare every LSA64 clip under ``src_dir``; write ``index.json``."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, int] = {}
    for video in sorted(src_dir.glob("*.mp4")):
        parse_lsa64_filename(video.name)  # fail fast on unexpected names
        clip_id = video.stem
        index[clip_id] = prepare_video(video, dst_dir / clip_id, size=size)
    (dst_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index
