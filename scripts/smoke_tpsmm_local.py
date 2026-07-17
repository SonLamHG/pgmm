"""Run TPSMM end to end on CPU against a synthetic dataset.

Catches dependency-compatibility breakage in seconds. Without it, each such
error costs a full Kaggle cycle — push, queue, run, fetch log — at 5-10 minutes
apiece, and they only surface one at a time. TPSMM is 2021 code on a 2026 stack;
two of its dependencies have already broken it (see third_party/tpsmm/UPSTREAM.md).

Not a correctness test: the data is noise and one epoch of it means nothing. It
answers only "does the code path execute at all", which is exactly the question
that was costing kernel cycles.

    python scripts/smoke_tpsmm_local.py

Exits non-zero on failure. Run before pushing any change that touches the
vendored tree or the config.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
TPSMM = REPO / "third_party" / "tpsmm"
CONFIG = REPO / "pgmm" / "config" / "lsa64-tpsmm.yaml"


def build_tiny_dataset(root: Path) -> None:
    """Two train clips, one test clip, in the train/test layout TPSMM needs."""
    rng = np.random.default_rng(0)
    for split, clip_ids in (
        ("train", ["001_001_001", "001_001_002"]),
        ("test", ["002_002_002"]),
    ):
        for clip_id in clip_ids:
            d = root / split / clip_id
            d.mkdir(parents=True)
            for i in range(8):
                frame = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"frame_{i:05d}.jpg"), frame)


def build_tiny_config(data_root: Path, out: Path) -> None:
    """The real config, shrunk to run on a CPU in under a minute."""
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["dataset_params"]["root_dir"] = str(data_root)
    t = cfg["train_params"]
    t["num_epochs"] = 1
    t["num_repeats"] = 1
    t["batch_size"] = 2
    t["dataloader_workers"] = 0
    t["checkpoint_freq"] = 1
    t["scales"] = [1, 0.5]  # fewer perceptual scales: less VGG on a CPU
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tpsmm_smoke_"))
    try:
        data = tmp / "data"
        build_tiny_dataset(data)
        cfg = tmp / "tiny.yaml"
        build_tiny_config(data, cfg)

        print("running TPSMM on CPU ...")
        r = subprocess.run(
            [sys.executable, "run.py", "--config", str(cfg),
             "--log_dir", str(tmp / "log"), "--device_ids", "0"],
            cwd=TPSMM, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("FAILED\n--- stdout ---")
            print(r.stdout[-3000:])
            print("--- stderr ---")
            print(r.stderr[-3000:])
            return 1

        ckpts = list((tmp / "log").rglob("*.pth.tar"))
        if not ckpts:
            print("FAILED: ran, but wrote no checkpoint")
            return 1
        # the visualiser is the skimage.draw.circle path -- the shim's real test
        vis = list((tmp / "log").rglob("*-rec.png"))
        if not vis:
            print("FAILED: no visualisation written -- the circle shim path did not run")
            return 1

        print(f"OK: checkpoint {ckpts[0].stat().st_size/2**20:.0f} MiB, "
              f"visualisation rendered")
        for line in (tmp / "log").rglob("log.txt"):
            print("losses:", line.read_text().strip()[:120])
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
