"""Carry a training run across Kaggle sessions.

Kaggle caps a session at ~9h and wipes /kaggle/working between them, while a
100-epoch run needs more than one session. Each session ends by publishing its
checkpoint as a new version of a Kaggle Dataset; the next session attaches that
dataset and resumes from it.

Named ``kaggle_harness`` rather than ``kaggle``: on Kaggle the official
``kaggle`` package is installed, and a top-level ``kaggle`` directory in the
repo would shadow it — breaking the very API this module shells out to. The
clash is invisible locally, where that package is absent.
"""

import json
import re
import subprocess
from pathlib import Path

_STEP_RE = re.compile(r"ckpt_step(\d+)\.pt$")


def latest_checkpoint(search_dirs: list[Path]) -> Path | None:
    """Newest checkpoint by step number across ``search_dirs``.

    Ordered by the step encoded in the filename, never by mtime: a resumed
    session rewrites files, so mtime does not track training progress.
    """
    best: tuple[int, Path] | None = None
    for directory in search_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in directory.glob("ckpt_step*.pt"):
            match = _STEP_RE.search(path.name)
            if not match:
                continue
            step = int(match.group(1))
            if best is None or step > best[0]:
                best = (step, path)
    return best[1] if best else None


def push_checkpoint(
    ckpt: Path, slug: str, message: str, dry_run: bool = False
) -> list[str]:
    """Publish ``ckpt``'s directory as a new version of dataset ``slug``.

    Returns the argv used, so callers (and tests) can see exactly what ran.
    """
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint to push: {ckpt}")
    folder = ckpt.parent
    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": slug.split("/")[-1],
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    argv = [
        "kaggle", "datasets", "version",
        "-p", str(folder),
        "-m", message,
        "--dir-mode", "zip",
    ]
    if not dry_run:
        subprocess.run(argv, check=True)
    return argv
