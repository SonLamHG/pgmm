"""Drive one Kaggle training session of the vendored TPSMM.

A full run is 15.5 GPU-hours (D12) against a ~9h session cap, so every run spans
about two sessions. This module is what makes a session resumable, and it lives
in the repo rather than in the notebook so it can be tested off-Kaggle.

The handoff uses `kernel_sources`, not a download: Kaggle bundles each kernel's
output into a single `_output_.zip`, and the next session mounts the previous
session's zip alongside the prepared data (D20). Nothing transfers through the
workstation, and no kernel ever needs credentials (D16).

Resume semantics come from the vendored `train.py`:

    start_epoch = Logger.load_cpk(...) + 1
    scheduler = MultiStepLR(..., last_epoch=start_epoch - 1)
    for epoch in trange(start_epoch, train_params['num_epochs']):

so `num_epochs` is a **total**, not an increment, and the LR schedule resumes at
the right position. Sessions are therefore bounded by giving each a different
`num_epochs`.
"""

import re
from pathlib import Path

import yaml

KAGGLE_INPUT = Path("/kaggle/input")
_CKPT_RE = re.compile(r"(\d+)-checkpoint\.pth\.tar$")


def find_prepared_data(search_root: Path = KAGGLE_INPUT) -> Path:
    """Locate the extracted LSA64 root containing train/ and test/.

    Globs rather than hardcoding: the mount layout is
    /kaggle/input/datasets/<owner>/<slug>/... , not the commonly documented
    /kaggle/input/<slug>, and an earlier kernel died on that assumption.
    """
    for candidate in sorted(Path(search_root).rglob("lsa64_prepared")):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"no lsa64_prepared/{{train,test}} under {search_root}. Either "
        f"kernel_sources does not include the preprocess kernel, or its "
        f"_output_.zip was not extracted yet."
    )


def find_resume_checkpoint(search_root: Path) -> Path | None:
    """Newest TPSMM checkpoint under ``search_root``, or None to start fresh.

    Ordered by the epoch encoded in the filename, never by mtime: extraction
    rewrites every file's timestamp, so mtime says nothing about training order.
    """
    best: tuple[int, Path] | None = None
    for path in Path(search_root).rglob("*-checkpoint.pth.tar"):
        match = _CKPT_RE.search(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return best[1] if best else None


def build_session_config(
    base_config: Path,
    data_root: Path,
    out_path: Path,
    max_epochs: int,
) -> dict:
    """Write a session config: the committed config, pointed at the data and
    bounded to ``max_epochs`` total.

    Bounding matters. Left at 100, a session would be killed by the cap
    mid-epoch, and a killed session's output is not reliably saved — the work
    since the last checkpoint would be lost along with the run's own log. A
    session that exits cleanly hands its checkpoint on.
    """
    cfg = yaml.safe_load(Path(base_config).read_text())
    cfg["dataset_params"]["root_dir"] = str(data_root)
    cfg["train_params"]["num_epochs"] = max_epochs
    Path(out_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def steps_per_epoch(cfg: dict, n_train_clips: int = 2800) -> float:
    """Optimiser steps in one epoch.

    DatasetRepeater makes an epoch `num_repeats * n_clips` samples and each item
    is exactly one (source, driving) pair, so `num_repeats` is pairs-per-clip
    (D18).
    """
    t = cfg["train_params"]
    return n_train_clips * t["num_repeats"] / t["batch_size"]
