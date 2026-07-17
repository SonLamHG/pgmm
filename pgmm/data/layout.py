"""Materialise the D9 split as the train/ and test/ directories TPSMM expects.

This is not cosmetic plumbing. `FramesDataset` checks for `root_dir/train` and,
finding none, **silently falls back to its own random 80/20 split**:

    if os.path.exists(os.path.join(root_dir, 'train')):
        ...
    else:
        print("Use random train-test split.")
        train_videos, test_videos = train_test_split(self.videos, ..., test_size=0.2)

On LSA64 that fallback yields 2560/640 rather than the paper's 2800/400, and
quietly discards D9 in favour of an undocumented partition — with no error, just
a line of stdout nobody reads. Laying the split out on disk is what makes our
split the one that actually trains.
"""

import shutil
from pathlib import Path

from pgmm.data.lsa64 import split_clips


def materialise_split(
    prepared_dir: Path,
    out_dir: Path,
    mode: str = "random",
    seed: int = 0,
) -> dict[str, int]:
    """Copy prepared clips into ``out_dir/train`` and ``out_dir/test`` per D9.

    Args:
        prepared_dir: flat output of ``prepare_dataset`` — one directory of
            frames per clip id, plus ``index.json``.
        out_dir: destination root; must not already exist.
        mode: split mode, passed to :func:`pgmm.data.lsa64.split_clips`.
        seed: split seed.

    Returns:
        ``{"train": n, "test": n}`` — assert these against the paper's 2800/400.
    """
    prepared_dir, out_dir = Path(prepared_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists; refusing to merge into a possibly "
            f"stale layout — remove it first"
        )

    clip_ids = sorted(
        p.name for p in prepared_dir.iterdir()
        if p.is_dir() and any(p.glob("frame_*.jpg"))
    )
    if not clip_ids:
        raise ValueError(f"no prepared clips found in {prepared_dir}")

    train_ids, test_ids = split_clips(clip_ids, mode=mode, seed=seed)
    for name, ids in (("train", train_ids), ("test", test_ids)):
        dest = out_dir / name
        dest.mkdir(parents=True)
        for clip_id in ids:
            shutil.copytree(prepared_dir / clip_id, dest / clip_id)
    return {"train": len(train_ids), "test": len(test_ids)}
