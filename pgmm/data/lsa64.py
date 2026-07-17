"""LSA64 split definition and frame loading.

The paper reports 2800 train / 400 test but never defines the partition
(decision D9). 87.5/12.5 factors against neither the 10 signers nor the 5
repetitions, so a seeded random split is the default reading; a
signer-held-out split stays available to test sensitivity.
"""

import random
from pathlib import Path

import cv2
import numpy as np

TEST_CLIPS = 400
TOTAL_CLIPS = 3200


def split_clips(
    clip_ids: list[str], mode: str = "random", seed: int = 0
) -> tuple[list[str], list[str]]:
    """Partition clip ids into (train, test).

    Args:
        clip_ids: every clip id, e.g. ``001_001_001``.
        mode: ``"random"`` (D9 default, 2800/400) or ``"signer"``
            (hold out whole signers — a harder, more honest generalisation
            test, kept for sensitivity analysis).
        seed: RNG seed for ``"random"``.
    """
    if mode == "random":
        shuffled = sorted(clip_ids)
        random.Random(seed).shuffle(shuffled)
        test = sorted(shuffled[:TEST_CLIPS])
        train = sorted(shuffled[TEST_CLIPS:])
        return train, test
    if mode == "signer":
        signers = sorted({int(c.split("_")[1]) for c in clip_ids})
        rng = random.Random(seed)
        held_out = set(rng.sample(signers, k=max(1, len(signers) // 8)))
        train = sorted(c for c in clip_ids if int(c.split("_")[1]) not in held_out)
        test = sorted(c for c in clip_ids if int(c.split("_")[1]) in held_out)
        return train, test
    raise ValueError(f"unknown split mode: {mode!r} (expected 'random' or 'signer')")


def load_clip(prepared_dir: Path, clip_id: str) -> np.ndarray:
    """Load a prepared clip as (N,H,W,C) float32 RGB in [0,1]."""
    clip_dir = Path(prepared_dir) / clip_id
    frame_paths = sorted(clip_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"no prepared frames for clip {clip_id!r} in {clip_dir}")
    frames = []
    for path in frame_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"cannot read frame: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames).astype(np.float32) / 255.0
