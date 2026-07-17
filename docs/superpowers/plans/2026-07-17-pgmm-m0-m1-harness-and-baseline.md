# PGMM Replication — M0 (Harness) + M1 (Baseline Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Kaggle-resumable training/eval harness for LSA64 and use it to reproduce the TPSMM row of the paper's Table 1 — the hard gate that every later PGMM number depends on.

**Architecture:** Vendor the TPSMM repo (MIT) unmodified under `third_party/tpsmm/` and build a separate `pgmm/` package around it holding the data pipeline, the metric suite (L1/SSIM/LPIPS/FVD/TCD), and a checkpoint-chaining layer that lets a 100-epoch run survive Kaggle's ~9h session cap. The TCD metric ships with multiple candidate readings because the paper never defines its value range; M1 selects the correct one by calibrating against the paper's own published TPSMM number (0.130).

**Tech Stack:** Python (match Kaggle's interpreter), PyTorch, OpenCV, scikit-image, `lpips`, pytest, Kaggle API.

**Spec:** `docs/superpowers/specs/2026-07-17-pgmm-replication-design.md`

**Scope:** M0 and M1 only. M2–M4 (the PGMM modules and runs) are planned *after* the M1 gate passes, because their details depend on what M1 measures — the resolved TCD reading, the real step time, and whether the harness holds.

## Global Constraints

- **`third_party/tpsmm/` numerics are frozen.** No edit to its math. Any unavoidable change gets its own commit plus a `docs/decisions.md` entry.
- **Hardware target: Kaggle 2×T4 (Turing, 16GB each).** No bf16. No flash-attn. Memory-efficient SDPA backend only.
- **Kaggle quota bills session wall-clock, not GPU-hours** — use both GPUs per session.
- **Image convention:** float32 in `[0, 1]`. NHWC in `pgmm/metrics/*` and `pgmm/data/*`; NCHW inside torch models. Every metric function validates its input range.
- **Resolution: 128×128** (crop-and-resize), per the paper.
- **Every deviation from the paper gets a numbered `D<n>` entry in `docs/decisions.md`.** A number that cannot be traced to a decision is not reportable.
- **Never tune toward the paper's numbers.** Metric *definitions* are calibrated against the paper (D3); model results are not.
- **Determinism:** every training entry point takes `--seed` and seeds Python/NumPy/torch. Resume restores RNG state.

## Paper targets (LSA64, Table 1)

| Method | L1↓ | SSIM↑ | LPIPS↓ | FVD↓ | TCD↓ |
|---|---|---|---|---|---|
| TPSMM (this plan reproduces) | 0.01342 | 0.9208 | 0.02261 | 182.785 | 0.130 |
| PGMM (later plan) | 0.01144 | 0.9395 | 0.01638 | 154.726 | 0.125 |

Tolerance: ±5% relative on L1/LPIPS, ±0.005 absolute on SSIM. FVD exempt (ordering only).

## File Structure

| Path | Responsibility |
|---|---|
| `third_party/tpsmm/` | Vendored upstream fork, commit recorded. Frozen. |
| `pgmm/metrics/basic.py` | L1, SSIM, LPIPS over frame sequences |
| `pgmm/metrics/tcd.py` | TCD + its candidate readings (D3) |
| `pgmm/data/lsa64_prepare.py` | Video → 128×128 frame shards |
| `pgmm/data/lsa64.py` | Split definition (D9) + frame loading |
| `pgmm/train/state.py` | Checkpoint save/load: model, optim, sched, epoch/step, RNG |
| `pgmm/eval/reconstruct.py` | Reconstruction protocol → frames → metric row |
| `kaggle/chain.py` | Push/pull checkpoints as Kaggle Dataset versions |
| `kaggle/train.ipynb` | Session entry point |
| `docs/decisions.md` | Numbered decision log |
| `docs/results.md` | Our numbers vs the paper's, failures included |

`pgmm/metrics/fvd.py` appears in the spec's architecture but is **not built by
this plan** — FVD is exempt from the tolerance and so cannot fail the gate. It
is the first task of the next plan. See "Deferred to the next plan" below.

---

### Task 1: Scaffolding, vendored TPSMM, decision log

**Files:**
- Create: `pyproject.toml`, `pgmm/__init__.py`, `pgmm/metrics/__init__.py`, `pgmm/data/__init__.py`, `pgmm/train/__init__.py`, `pgmm/eval/__init__.py`, `docs/decisions.md`, `docs/results.md`
- Create: `third_party/tpsmm/` (vendored)
- Test: `tests/test_imports.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable `pgmm` package; `third_party/tpsmm` on `sys.path` for later tasks

- [ ] **Step 1: Vendor TPSMM and record the exact upstream commit**

Vendored as a plain directory rather than a submodule: later PGMM work must patch TPSMM's `model.py` to add `L_pd`/`L_align`, Kaggle clones are simpler without submodule recursion, and one clone contains everything.

```bash
git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git /tmp/tpsmm
cd /tmp/tpsmm && git rev-parse HEAD    # record this hash — do NOT invent one
```

Copy the working tree (minus `.git`) to `third_party/tpsmm/`, then write the real hash into `third_party/tpsmm/UPSTREAM.md`:

```markdown
# Vendored upstream
Repo: https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model
Commit: <paste the actual hash from git rev-parse>
License: MIT (see LICENSE in this directory)
Vendored: 2026-07-17

## Patch log
(no patches yet — numerics frozen)
```

Confirm `third_party/tpsmm/LICENSE` exists and is MIT. If absent, stop and resolve licensing before continuing.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "pgmm"
version = "0.1.0"
description = "Replication of Pose-Guided Fine-Grained Sign Language Video Generation (arXiv 2409.16709)"
requires-python = ">=3.10"
dependencies = [
    "torch", "torchvision", "numpy", "opencv-python-headless",
    "scikit-image", "lpips", "pyyaml", "imageio", "imageio-ffmpeg", "tqdm",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["pgmm*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Torch is intentionally unpinned: Kaggle ships its own build and pinning fights the image. Record the resolved version at M0 Step (Task 10).

- [ ] **Step 3: Seed the decision log**

`docs/decisions.md` — carry D1–D8 over from the spec verbatim and add the three found while planning:

```markdown
# Decision Log

Every deviation from the paper, and every choice the paper left open, is
recorded here. A result that cannot be traced to a decision is not reportable.

| # | Ambiguity | Decision | Status |
|---|---|---|---|
| D1 | Which 36 of COCO-WholeBody's 133 keypoints | Unresolved; ask authors, resolve empirically at M3 | open |
| D2 | `L_pd` argmax is non-differentiable | Use soft-argmax | decided |
| D3 | TCD value range / channel reduction for `T=0.5` | Calibrate against TPSMM=0.130 at M1 | open |
| D4 | Generator block count | Follow TPSMM `num_down_blocks=3` | decided |
| D5 | PFM attention may exceed 16GB | Memory-efficient SDPA; restrict scales if needed | open |
| D6 | `L_pd`/`L_align` weights | Start 1.0, log magnitudes at M2 | decided |
| D7 | `L_r` sub-loss weights | Inherited from vendored TPSMM | decided |
| D8 | FVD implementation sensitivity | Pin one I3D; report relative only | open |
| D9 | LSA64 2800/400 split undefined | Seeded random default; signer-held-out alternative | open |
| D10 | LPIPS backbone (alex vs vgg) undefined | `lpips` pkg, `net='alex'` | decided |
| D11 | SSIM implementation undefined | `skimage`, `data_range=1.0`, `channel_axis=-1` | decided |

## D9 — LSA64 split
The paper reports 2800 train / 400 test but never defines the partition.
LSA64 is 10 signers x 64 signs x 5 repetitions = 3200. A 2800/400 split is
87.5%/12.5%, which factors cleanly against neither 10 signers (would give
2560/640) nor 5 repetitions (also 2560/640). So the split is almost
certainly a plain random 87.5/12.5, not a structured hold-out.
**Decision:** default `split="random"`, seed 0. Keep `split="signer"`
selectable to test sensitivity. If M1 misses its target, re-examine this
first — a signer-overlapping split is *easier* than a held-out one and
would inflate all our numbers relative to the paper.

## D10 / D11 — metric implementations
Neither LPIPS backbone nor SSIM implementation is stated, and both shift
the third decimal — the precision at which we are claiming a match.
Pinned as above; revisit only if M1 misses.
```

- [ ] **Step 4: Write the import test**

```python
# tests/test_imports.py
import pathlib


def test_pgmm_package_imports():
    import pgmm  # noqa: F401


def test_vendored_tpsmm_present_with_license_and_pinned_commit():
    root = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "tpsmm"
    assert (root / "LICENSE").exists(), "vendored TPSMM must ship its MIT license"
    upstream = (root / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "Commit:" in upstream
    # a real 40-char sha, not a placeholder
    line = next(l for l in upstream.splitlines() if l.startswith("Commit:"))
    sha = line.split("Commit:")[1].strip()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), (
        f"UPSTREAM.md must record a real commit hash, got {sha!r}"
    )
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_imports.py -v`
Expected: 2 passed. If the sha assertion fails, the hash was invented — go back to Step 1 and paste the real one.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml pgmm/ third_party/ docs/decisions.md docs/results.md tests/test_imports.py
git commit -m "chore: scaffold pgmm package, vendor pinned TPSMM, seed decision log"
```

---

### Task 2: TCD metric with candidate readings

Done before the data pipeline because TCD is the paper's novel contribution, is pure and fully unit-testable, and its D3 ambiguity is the one M1 must resolve.

**Files:**
- Create: `pgmm/metrics/tcd.py`
- Test: `tests/test_tcd.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class TCDVariant(str, Enum)` with members `UNIT_MEAN`, `UNIT_MAX`, `UNIT_SUM`, `BYTE_MEAN`
  - `tcd(generated: np.ndarray, target: np.ndarray, threshold: float = 0.5, variant: TCDVariant = TCDVariant.UNIT_MEAN) -> float` — both arrays `(N,H,W,C)` float in `[0,1]`, `N>=3`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tcd.py
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_tcd.py -v`
Expected: all fail — `ModuleNotFoundError: No module named 'pgmm.metrics.tcd'`

- [ ] **Step 3: Implement**

```python
# pgmm/metrics/tcd.py
"""Temporal Consistency Difference (TCD), the metric introduced by
arXiv:2409.16709 Sec. 4.

    D   = |I_hat_t - 1/2 (I_{t-1} + I_{t+1})|
    TCD = mean over interior frames of the fraction of pixels with D > T,  T = 0.5

The paper never states the image value range the threshold applies to, nor
whether D is reduced across channels by mean, max, or sum (decision D3).
The reading is therefore selectable, and M1 picks it by calibrating the
TPSMM baseline against the paper's own published 0.130.
"""

from enum import Enum

import numpy as np


class TCDVariant(str, Enum):
    """Candidate readings of the paper's threshold semantics (D3)."""

    UNIT_MEAN = "unit_mean"  # [0,1], mean over channels
    UNIT_MAX = "unit_max"    # [0,1], max over channels
    UNIT_SUM = "unit_sum"    # [0,1], sum over channels (range [0,3])
    BYTE_MEAN = "byte_mean"  # [0,255], mean over channels


def _reduce_channels(diff: np.ndarray, variant: TCDVariant) -> np.ndarray:
    if variant is TCDVariant.UNIT_MEAN:
        return diff.mean(axis=-1)
    if variant is TCDVariant.UNIT_MAX:
        return diff.max(axis=-1)
    if variant is TCDVariant.UNIT_SUM:
        return diff.sum(axis=-1)
    if variant is TCDVariant.BYTE_MEAN:
        return diff.mean(axis=-1) * 255.0
    raise ValueError(f"unknown TCD variant: {variant}")


def _validate(generated: np.ndarray, target: np.ndarray) -> None:
    if generated.shape != target.shape:
        raise ValueError(
            f"generated and target must have the same shape, "
            f"got {generated.shape} and {target.shape}"
        )
    if generated.ndim != 4:
        raise ValueError(f"expected (N,H,W,C) arrays, got ndim={generated.ndim}")
    if generated.shape[0] < 3:
        raise ValueError(
            f"TCD needs at least 3 frames to have an interior frame, "
            f"got {generated.shape[0]}"
        )
    for name, arr in (("generated", generated), ("target", target)):
        if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
            raise ValueError(
                f"{name} must be in [0, 1], got range "
                f"[{arr.min():.4f}, {arr.max():.4f}]"
            )


def tcd(
    generated: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    variant: TCDVariant = TCDVariant.UNIT_MEAN,
) -> float:
    """Fraction of pixels deviating from the temporal midpoint by more than
    ``threshold``, averaged over interior frames. Lower is better.

    Args:
        generated: reconstructed frames, (N,H,W,C) float in [0,1].
        target: ground-truth frames, (N,H,W,C) float in [0,1].
        threshold: the paper's T, default 0.5.
        variant: which reading of the threshold semantics to apply (D3).
    """
    _validate(generated, target)
    midpoint = 0.5 * (target[:-2] + target[2:])          # (N-2,H,W,C)
    diff = np.abs(generated[1:-1] - midpoint)            # (N-2,H,W,C)
    reduced = _reduce_channels(diff, variant)            # (N-2,H,W)
    return float((reduced > threshold).mean())
```

The final `.mean()` over an `(N-2, H, W)` boolean array equals the paper's
`1/(N-2) * sum_t [ 1/(H*W) * sum_{h,w} ... ]` exactly, because every frame
contributes the same pixel count.

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_tcd.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add pgmm/metrics/tcd.py tests/test_tcd.py
git commit -m "feat: add TCD metric with candidate threshold readings (D3)"
```

---

### Task 3: Basic metrics (L1, SSIM, LPIPS)

**Files:**
- Create: `pgmm/metrics/basic.py`
- Test: `tests/test_basic_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `l1(generated: np.ndarray, target: np.ndarray) -> float`
  - `ssim(generated: np.ndarray, target: np.ndarray) -> float`
  - `lpips(generated: np.ndarray, target: np.ndarray, device: str = "cpu") -> float`

  All take `(N,H,W,C)` float `[0,1]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_basic_metrics.py
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_basic_metrics.py -v`
Expected: all fail — `ModuleNotFoundError: No module named 'pgmm.metrics.basic'`

- [ ] **Step 3: Implement**

```python
# pgmm/metrics/basic.py
"""Reconstruction metrics over frame sequences.

LPIPS backbone (D10) and SSIM implementation (D11) are pinned here: the
paper states neither, and both move the third decimal — which is the
precision at which we claim a match.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity

_lpips_model = None


def _validate(generated: np.ndarray, target: np.ndarray) -> None:
    if generated.shape != target.shape:
        raise ValueError(
            f"generated and target must have the same shape, "
            f"got {generated.shape} and {target.shape}"
        )
    for name, arr in (("generated", generated), ("target", target)):
        if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
            raise ValueError(
                f"{name} must be in [0, 1], got range "
                f"[{arr.min():.4f}, {arr.max():.4f}]"
            )


def l1(generated: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error. Lower is better."""
    _validate(generated, target)
    return float(np.abs(generated - target).mean())


def ssim(generated: np.ndarray, target: np.ndarray) -> float:
    """Structural similarity, averaged per frame (D11). Higher is better."""
    _validate(generated, target)
    scores = [
        structural_similarity(t, g, data_range=1.0, channel_axis=-1)
        for g, t in zip(generated, target)
    ]
    return float(np.mean(scores))


def _get_lpips_model(device: str):
    global _lpips_model
    if _lpips_model is None:
        import lpips as lpips_lib

        _lpips_model = lpips_lib.LPIPS(net="alex")  # D10
    return _lpips_model.to(device)


def lpips(generated: np.ndarray, target: np.ndarray, device: str = "cpu") -> float:
    """Learned perceptual distance, AlexNet backbone (D10). Lower is better."""
    _validate(generated, target)
    model = _get_lpips_model(device)
    # lpips expects NCHW in [-1, 1]
    g = torch.from_numpy(np.ascontiguousarray(generated)).permute(0, 3, 1, 2)
    t = torch.from_numpy(np.ascontiguousarray(target)).permute(0, 3, 1, 2)
    g = (g.to(device) * 2.0 - 1.0).float()
    t = (t.to(device) * 2.0 - 1.0).float()
    with torch.no_grad():
        distances = model(g, t)
    return float(distances.mean())
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_basic_metrics.py -v`
Expected: 7 passed. First run downloads AlexNet LPIPS weights — needs network.

- [ ] **Step 5: Commit**

```bash
git add pgmm/metrics/basic.py tests/test_basic_metrics.py
git commit -m "feat: add L1/SSIM/LPIPS metrics with pinned implementations (D10, D11)"
```

---

### Task 4: LSA64 preprocessing (video → 128×128 frames)

**Files:**
- Create: `pgmm/data/lsa64_prepare.py`
- Test: `tests/test_lsa64_prepare.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `parse_lsa64_filename(name: str) -> tuple[int, int, int]` returning `(sign, signer, repetition)`, all 1-based
  - `centre_crop_square(frame: np.ndarray) -> np.ndarray` — largest centred square crop
  - `prepare_video(src: Path, dst_dir: Path, size: int = 128) -> int` writing `frame_%05d.jpg` and returning the frame count
  - `prepare_dataset(src_dir: Path, dst_dir: Path, size: int = 128) -> dict[str, int]` mapping clip id → frame count

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lsa64_prepare.py
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_lsa64_prepare.py -v`
Expected: all fail — module not found

- [ ] **Step 3: Implement**

```python
# pgmm/data/lsa64_prepare.py
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_lsa64_prepare.py -v`
Expected: 5 passed

- [ ] **Step 5: Verify the filename convention against the real download**

The `NNN_NNN_NNN` convention above is taken from LSA64's documentation but has
not been checked against the actual archive. Download LSA64 (cut version) from
https://facundoq.github.io/datasets/lsa64/ and confirm:

```bash
ls <lsa64_dir> | head -5
ls <lsa64_dir> | wc -l     # expect 3200
```

Expected: names like `001_001_001.mp4`, and 3200 files. **If the convention
differs, fix `_NAME_RE` and the test together — do not make the test pass by
loosening the assertion.** Then confirm the parse holds across the whole set:

```bash
python -c "
from pathlib import Path
from pgmm.data.lsa64_prepare import parse_lsa64_filename
names = sorted(p.name for p in Path('<lsa64_dir>').glob('*.mp4'))
print(len(names))
signs, signers, reps = zip(*(parse_lsa64_filename(n) for n in names))
print('signs', min(signs), max(signs))       # expect 1 64
print('signers', min(signers), max(signers)) # expect 1 10
print('reps', min(reps), max(reps))          # expect 1 5
"
```

Record the observed counts in `docs/decisions.md` under D9 — the split
decision depends on this structure being what we think it is.

- [ ] **Step 6: Commit**

```bash
git add pgmm/data/lsa64_prepare.py tests/test_lsa64_prepare.py docs/decisions.md
git commit -m "feat: add LSA64 frame pre-extraction with centre-crop to 128px"
```

---

### Task 5: LSA64 split and frame loading

**Files:**
- Create: `pgmm/data/lsa64.py`
- Test: `tests/test_lsa64.py`

**Interfaces:**
- Consumes: `prepare_dataset`'s `index.json` layout from Task 4
- Produces:
  - `split_clips(clip_ids: list[str], mode: str = "random", seed: int = 0) -> tuple[list[str], list[str]]` returning `(train_ids, test_ids)`
  - `load_clip(prepared_dir: Path, clip_id: str) -> np.ndarray` returning `(N,H,W,C)` float32 in `[0,1]`, RGB

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lsa64.py
import numpy as np
import pytest

from pgmm.data.lsa64 import load_clip, split_clips


def _all_clip_ids():
    return [
        f"{sign:03d}_{signer:03d}_{rep:03d}"
        for sign in range(1, 65)
        for signer in range(1, 11)
        for rep in range(1, 6)
    ]


def test_random_split_matches_paper_counts():
    train, test = split_clips(_all_clip_ids(), mode="random", seed=0)
    assert len(train) == 2800
    assert len(test) == 400


def test_random_split_is_disjoint_and_total():
    ids = _all_clip_ids()
    train, test = split_clips(ids, mode="random", seed=0)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(ids)


def test_random_split_is_deterministic_given_seed():
    ids = _all_clip_ids()
    assert split_clips(ids, seed=0) == split_clips(ids, seed=0)
    assert split_clips(ids, seed=0) != split_clips(ids, seed=1)


def test_signer_split_holds_out_signers_entirely():
    ids = _all_clip_ids()
    train, test = split_clips(ids, mode="signer", seed=0)
    train_signers = {int(c.split("_")[1]) for c in train}
    test_signers = {int(c.split("_")[1]) for c in test}
    assert train_signers.isdisjoint(test_signers)


def test_unknown_split_mode_rejected():
    with pytest.raises(ValueError, match="unknown split mode"):
        split_clips(_all_clip_ids(), mode="bogus")


def test_load_clip_returns_unit_range_rgb(tmp_path):
    import cv2

    clip_dir = tmp_path / "001_001_001"
    clip_dir.mkdir()
    # pure red in BGR -> must come back as pure red in RGB
    red_bgr = np.zeros((128, 128, 3), np.uint8)
    red_bgr[..., 2] = 255
    for i in range(3):
        cv2.imwrite(str(clip_dir / f"frame_{i:05d}.jpg"), red_bgr)

    clip = load_clip(tmp_path, "001_001_001")
    assert clip.shape == (3, 128, 128, 3)
    assert clip.dtype == np.float32
    assert 0.0 <= clip.min() and clip.max() <= 1.0
    assert clip[0, 64, 64, 0] > 0.9  # red channel first => RGB
    assert clip[0, 64, 64, 2] < 0.1


def test_load_clip_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_clip(tmp_path, "999_999_999")
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_lsa64.py -v`
Expected: all fail — module not found

- [ ] **Step 3: Implement**

```python
# pgmm/data/lsa64.py
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_lsa64.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add pgmm/data/lsa64.py tests/test_lsa64.py
git commit -m "feat: add LSA64 split (D9) and unit-range RGB clip loading"
```

---

### Task 6: Checkpoint state save/load with RNG

The foundation of the M0 gate. A resume that silently restarts the LR schedule
would corrupt every downstream number invisibly — so state round-tripping is
tested before anything depends on it.

**Files:**
- Create: `pgmm/train/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `save_state(path: Path, *, model, optimizer, scheduler, epoch: int, step: int) -> None`
  - `load_state(path: Path, *, model, optimizer, scheduler) -> tuple[int, int]` returning `(epoch, step)` and restoring Python/NumPy/torch RNG state

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_state.py
import random

import numpy as np
import pytest
import torch

from pgmm.train.state import load_state, save_state


def _make():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 90], gamma=0.1
    )
    return model, optimizer, scheduler


def test_round_trip_restores_weights(tmp_path):
    model, optimizer, scheduler = _make()
    path = tmp_path / "ckpt.pt"
    save_state(path, model=model, optimizer=optimizer, scheduler=scheduler,
               epoch=3, step=42)

    fresh_model, fresh_opt, fresh_sched = _make()
    load_state(path, model=fresh_model, optimizer=fresh_opt, scheduler=fresh_sched)
    assert torch.allclose(model.weight, fresh_model.weight)


def test_round_trip_restores_epoch_and_step(tmp_path):
    model, optimizer, scheduler = _make()
    path = tmp_path / "ckpt.pt"
    save_state(path, model=model, optimizer=optimizer, scheduler=scheduler,
               epoch=3, step=42)

    fresh = _make()
    epoch, step = load_state(path, model=fresh[0], optimizer=fresh[1],
                             scheduler=fresh[2])
    assert (epoch, step) == (3, 42)


def test_round_trip_restores_lr_schedule_position(tmp_path):
    """The failure this whole task exists to prevent: a resume that restarts
    the LR schedule trains at the wrong LR and silently corrupts results."""
    model, optimizer, scheduler = _make()
    for _ in range(61):  # past the first milestone
        scheduler.step()
    expected_lr = optimizer.param_groups[0]["lr"]
    assert expected_lr == pytest.approx(2e-5)  # decayed once

    path = tmp_path / "ckpt.pt"
    save_state(path, model=model, optimizer=optimizer, scheduler=scheduler,
               epoch=61, step=100)

    fresh_model, fresh_opt, fresh_sched = _make()
    assert fresh_opt.param_groups[0]["lr"] == pytest.approx(2e-4)  # not yet restored
    load_state(path, model=fresh_model, optimizer=fresh_opt, scheduler=fresh_sched)
    assert fresh_opt.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_round_trip_restores_optimizer_moments(tmp_path):
    model, optimizer, scheduler = _make()
    loss = model(torch.ones(1, 4)).sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_state(path, model=model, optimizer=optimizer, scheduler=scheduler,
               epoch=0, step=1)

    fresh_model, fresh_opt, fresh_sched = _make()
    load_state(path, model=fresh_model, optimizer=fresh_opt, scheduler=fresh_sched)
    original = optimizer.state_dict()["state"]
    restored = fresh_opt.state_dict()["state"]
    assert set(original) == set(restored)
    for key in original:
        assert torch.allclose(original[key]["exp_avg"], restored[key]["exp_avg"])


def test_round_trip_restores_rng_streams(tmp_path):
    model, optimizer, scheduler = _make()
    path = tmp_path / "ckpt.pt"
    save_state(path, model=model, optimizer=optimizer, scheduler=scheduler,
               epoch=0, step=0)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())

    # perturb every stream
    random.random(); np.random.rand(); torch.rand(1)

    fresh = _make()
    load_state(path, model=fresh[0], optimizer=fresh[1], scheduler=fresh[2])
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_state.py -v`
Expected: all fail — module not found

- [ ] **Step 3: Implement**

```python
# pgmm/train/state.py
"""Training-state checkpointing.

Every Kaggle run is interrupted by the session cap, so every run resumes.
That makes resume correctness load-bearing for all reported numbers: a
resume that drops the LR schedule or the RNG stream produces plausible but
wrong results with no visible symptom.
"""

import random
from pathlib import Path

import numpy as np
import torch


def save_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
) -> None:
    """Write a resumable checkpoint atomically.

    Atomic because a session killed mid-write would otherwise leave a
    truncated checkpoint and lose the whole run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:
    """Restore a checkpoint in place. Returns ``(epoch, step)``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return payload["epoch"], payload["step"]
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_state.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pgmm/train/state.py tests/test_state.py
git commit -m "feat: add resumable training state with LR schedule and RNG restore"
```

---

### Task 7: Resume-equivalence test (the M0 exit criterion)

**Files:**
- Create: `tests/test_resume_equivalence.py`

**Interfaces:**
- Consumes: `save_state`/`load_state` from Task 6
- Produces: nothing importable — this is the executable form of the M0 gate

- [ ] **Step 1: Write the test**

This is the spec's M0 exit criterion made mechanical: a resumed run must
produce the identical loss curve to an uninterrupted one. It uses a tiny
model so it runs on CPU in seconds, but it exercises exactly the state that
breaks in practice — optimizer moments, LR position, RNG, and data order.

```python
# tests/test_resume_equivalence.py
"""M0 gate: an interrupted-and-resumed run must be bit-identical to an
uninterrupted one. If this fails, every number produced on Kaggle is suspect,
because every Kaggle run resumes."""

import numpy as np
import torch

from pgmm.train.state import load_state, save_state


def _build(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 8], gamma=0.1
    )
    return model, optimizer, scheduler


def _train_steps(model, optimizer, scheduler, n_steps: int) -> list[float]:
    """Batches drawn from the global RNG, so data order is part of the state
    under test — exactly as in the real loop."""
    losses = []
    for _ in range(n_steps):
        batch = torch.from_numpy(np.random.rand(16, 4)).float()
        target = batch.sum(dim=1, keepdim=True)
        loss = torch.nn.functional.mse_loss(model(batch), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses


def test_resumed_run_reproduces_uninterrupted_loss_curve(tmp_path):
    model, optimizer, scheduler = _build(seed=0)
    uninterrupted = _train_steps(model, optimizer, scheduler, 20)

    model, optimizer, scheduler = _build(seed=0)
    first_half = _train_steps(model, optimizer, scheduler, 10)
    ckpt = tmp_path / "ckpt.pt"
    save_state(ckpt, model=model, optimizer=optimizer, scheduler=scheduler,
               epoch=0, step=10)

    # a genuinely fresh process-like state: new objects, perturbed RNG
    model, optimizer, scheduler = _build(seed=999)
    np.random.rand(100)
    epoch, step = load_state(ckpt, model=model, optimizer=optimizer,
                             scheduler=scheduler)
    assert (epoch, step) == (0, 10)
    second_half = _train_steps(model, optimizer, scheduler, 10)

    resumed = first_half + second_half
    assert len(resumed) == len(uninterrupted)
    np.testing.assert_allclose(resumed, uninterrupted, rtol=0, atol=0)


def test_resume_without_rng_restore_would_diverge(tmp_path):
    """Guards the guard: proves the equivalence test above has teeth and is
    not passing for a trivial reason."""
    model, optimizer, scheduler = _build(seed=0)
    uninterrupted = _train_steps(model, optimizer, scheduler, 20)

    model, optimizer, scheduler = _build(seed=0)
    first_half = _train_steps(model, optimizer, scheduler, 10)
    ckpt = tmp_path / "ckpt.pt"
    save_state(ckpt, model=model, optimizer=optimizer, scheduler=scheduler,
               epoch=0, step=10)

    model, optimizer, scheduler = _build(seed=999)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    # deliberately skip RNG restore
    second_half = _train_steps(model, optimizer, scheduler, 10)

    assert first_half + second_half != uninterrupted
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_resume_equivalence.py -v`
Expected: 2 passed. If the first test fails, `load_state` is losing state — fix Task 6 rather than relaxing the tolerance.

- [ ] **Step 3: Commit**

```bash
git add tests/test_resume_equivalence.py
git commit -m "test: add M0 resume-equivalence gate"
```

---

### Task 8: Kaggle checkpoint chaining

**Files:**
- Create: `kaggle/chain.py`
- Test: `tests/test_chain.py`

**Interfaces:**
- Consumes: checkpoints written by `save_state` (Task 6)
- Produces:
  - `latest_checkpoint(search_dirs: list[Path]) -> Path | None` — newest `ckpt_step*.pt` across attached dataset dirs and the working dir
  - `push_checkpoint(ckpt: Path, slug: str, message: str, dry_run: bool = False) -> list[str]` — returns the kaggle CLI argv it ran (or would run)

- [ ] **Step 1: Write the failing tests**

The Kaggle API needs credentials and network, so `push_checkpoint` is tested
by asserting the command it constructs, and the selection logic — the part
that actually breaks — is tested directly.

```python
# tests/test_chain.py
import json

import pytest

from kaggle.chain import latest_checkpoint, push_checkpoint


def _touch(path, step):
    path.mkdir(parents=True, exist_ok=True)
    f = path / f"ckpt_step{step:06d}.pt"
    f.write_bytes(b"x")
    return f


def test_latest_checkpoint_picks_highest_step_across_dirs(tmp_path):
    a = tmp_path / "attached"
    b = tmp_path / "working"
    _touch(a, 1000)
    newest = _touch(b, 2000)
    _touch(a, 500)
    assert latest_checkpoint([a, b]) == newest


def test_latest_checkpoint_prefers_step_number_not_mtime(tmp_path):
    """A resumed session may rewrite an older checkpoint; step order is the
    truth, mtime is not."""
    a = tmp_path / "a"
    newest = _touch(a, 3000)
    stale = _touch(a, 100)
    import os, time

    time.sleep(0.01)
    os.utime(stale, None)  # make the OLD checkpoint the NEWEST file
    assert latest_checkpoint([a]) == newest


def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    assert latest_checkpoint([tmp_path]) is None


def test_latest_checkpoint_skips_missing_dirs(tmp_path):
    real = tmp_path / "real"
    ckpt = _touch(real, 10)
    assert latest_checkpoint([tmp_path / "nope", real]) == ckpt


def test_push_checkpoint_builds_version_command(tmp_path):
    ckpt = _touch(tmp_path, 100)
    argv = push_checkpoint(ckpt, slug="user/pgmm-ckpt", message="step 100",
                           dry_run=True)
    assert argv[:3] == ["kaggle", "datasets", "version"]
    assert "-m" in argv and "step 100" in argv
    meta = json.loads((ckpt.parent / "dataset-metadata.json").read_text())
    assert meta["id"] == "user/pgmm-ckpt"


def test_push_checkpoint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        push_checkpoint(tmp_path / "nope.pt", slug="user/x", message="m",
                        dry_run=True)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_chain.py -v`
Expected: all fail — `ModuleNotFoundError: No module named 'kaggle.chain'`

If the error is instead a name clash with the installed `kaggle` package, add
`kaggle/__init__.py` and ensure the repo root precedes site-packages on
`sys.path`; if the clash persists, rename the directory to `kaggle_harness/`
and update the imports and this plan's paths consistently.

- [ ] **Step 3: Implement**

```python
# kaggle/chain.py
"""Carry a training run across Kaggle sessions.

Kaggle caps a session at ~9h and wipes /kaggle/working between them, while a
100-epoch run needs more than one session. Each session ends by publishing its
checkpoint as a new version of a Kaggle Dataset; the next session attaches that
dataset and resumes from it.
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
        json.dumps({"title": slug.split("/")[-1], "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2),
        encoding="utf-8",
    )
    argv = ["kaggle", "datasets", "version", "-p", str(folder), "-m", message,
            "--dir-mode", "zip"]
    if not dry_run:
        subprocess.run(argv, check=True)
    return argv
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_chain.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add kaggle/chain.py tests/test_chain.py
git commit -m "feat: add Kaggle checkpoint chaining across sessions"
```

---

### Task 9: Reconstruction evaluation harness

**Files:**
- Create: `pgmm/eval/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: `load_clip` (Task 5); `l1`/`ssim`/`lpips` (Task 3); `tcd`/`TCDVariant` (Task 2)
- Produces:
  - `evaluate_clips(reconstruct_fn, prepared_dir: Path, clip_ids: list[str], tcd_variant: TCDVariant = TCDVariant.UNIT_MEAN, device: str = "cpu") -> dict[str, float]` returning keys `l1`, `ssim`, `lpips`, `tcd`

  `reconstruct_fn(clip: np.ndarray) -> np.ndarray` maps a ground-truth clip to its reconstruction, same shape. Injected rather than hard-wired so the harness is testable without a GPU and reusable for both TPSMM and PGMM.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reconstruct.py
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_reconstruct.py -v`
Expected: all fail — module not found

- [ ] **Step 3: Implement**

```python
# pgmm/eval/reconstruct.py
"""Reconstruction-protocol evaluation.

Produces the metric row we compare against the paper's Table 1. The model is
injected as ``reconstruct_fn`` so the harness is identical for the TPSMM
baseline and for PGMM — the comparison is only meaningful if both are scored
by the same code.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from pgmm.data.lsa64 import load_clip
from pgmm.metrics.basic import l1, lpips, ssim
from pgmm.metrics.tcd import TCDVariant, tcd

ReconstructFn = Callable[[np.ndarray], np.ndarray]


def evaluate_clips(
    reconstruct_fn: ReconstructFn,
    prepared_dir: Path,
    clip_ids: list[str],
    tcd_variant: TCDVariant = TCDVariant.UNIT_MEAN,
    device: str = "cpu",
) -> dict[str, float]:
    """Score ``reconstruct_fn`` over ``clip_ids``; means across clips."""
    if not clip_ids:
        raise ValueError("no clips to evaluate")
    rows: list[dict[str, float]] = []
    for clip_id in clip_ids:
        target = load_clip(prepared_dir, clip_id)
        generated = reconstruct_fn(target)
        if generated.shape != target.shape:
            raise ValueError(
                f"reconstruction must have the same shape as the target for "
                f"clip {clip_id!r}: got {generated.shape}, want {target.shape}"
            )
        rows.append(
            {
                "l1": l1(generated, target),
                "ssim": ssim(generated, target),
                "lpips": lpips(generated, target, device=device),
                "tcd": tcd(generated, target, variant=tcd_variant),
            }
        )
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_reconstruct.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add pgmm/eval/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: add reconstruction eval harness shared by baseline and PGMM"
```

---

### Task 10: Kaggle notebook, published dataset, and measured step time (M0 exit)

**Files:**
- Create: `kaggle/train.ipynb`, `kaggle/README.md`
- Modify: `docs/decisions.md`

**Interfaces:**
- Consumes: everything above
- Produces: a published LSA64 dataset slug; a measured `seconds_per_step`; a re-derived timeline

- [ ] **Step 1: Prepare and publish LSA64 as a Kaggle Dataset**

Locally, after downloading LSA64 (cut version):

```bash
python -c "
from pathlib import Path
from pgmm.data.lsa64_prepare import prepare_dataset
index = prepare_dataset(Path('<lsa64_dir>'), Path('data/lsa64_prepared'), size=128)
print(len(index), 'clips;', sum(index.values()), 'frames')
"
```

Expected: `3200 clips`. Then publish:

```bash
cd data/lsa64_prepared
kaggle datasets init -p .
# edit dataset-metadata.json: set title and id to <user>/lsa64-prepared-128
kaggle datasets create -p . --dir-mode zip
```

LSA64 is non-commercial with attribution required — **make the dataset private**
and record the attribution in `kaggle/README.md`. Publishing it publicly would
redistribute the authors' data under terms we were not granted.

- [ ] **Step 2: Write `kaggle/train.ipynb`**

A thin session driver — logic lives in the repo, not the notebook, so it stays
testable. Cells:

```python
# Cell 1 — environment
!git clone https://github.com/<user>/paper_pgmm.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -e .
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

```python
# Cell 2 — resume from the newest checkpoint, if any
from pathlib import Path
from kaggle.chain import latest_checkpoint

SEARCH = [Path("/kaggle/input/pgmm-ckpt"), Path("/kaggle/working/ckpt")]
resume_from = latest_checkpoint(SEARCH)
print("resuming from", resume_from or "scratch")
```

```python
# Cell 3 — train: fp32, DataParallel over both T4s.
# fp32 deliberately: AMP is unsafe around grid_sample/flow warping and is not
# revisited until after M1 passes, and then only with a numerics check.
import subprocess
argv = [
    "python", "run.py",
    "--config", "../../pgmm/config/lsa64-tpsmm.yaml",
    "--device_ids", "0,1",
]
if resume_from is not None:
    argv += ["--checkpoint", str(resume_from)]
subprocess.run(argv, cwd="/kaggle/working/repo/third_party/tpsmm", check=True)
```

Verify `--checkpoint` is the vendored `run.py`'s actual resume flag before
relying on it; if the flag differs, use the real one rather than editing
`run.py` (the vendored tree is frozen).

```python
# Cell 4 — publish the checkpoint before the session dies
from kaggle.chain import push_checkpoint
ckpt = latest_checkpoint([Path("/kaggle/working/ckpt")])
push_checkpoint(ckpt, slug="<user>/pgmm-ckpt", message=f"{ckpt.name}")
```

Record in `kaggle/README.md`: the dataset slugs, that quota bills session
wall-clock (so both GPUs should always be used), and that the checkpoint push
must be a separate cell so it can be run manually if training is interrupted.

- [ ] **Step 3: Verify both T4s are visible and used**

Run Cell 1.
Expected: `gpus 2`, both `Tesla T4`. If only one appears, the notebook's
accelerator is set to `GPU T4 x1` — fix it before measuring anything, or the
timeline will be wrong by 2×.

- [ ] **Step 4: Measure real step time on LSA64**

Run TPSMM training for 200 steps and time it:

```bash
!cd /kaggle/working/repo/third_party/tpsmm && python run.py \
    --config ../../pgmm/config/lsa64-tpsmm.yaml \
    --device_ids 0,1
```

Record `seconds_per_step`, steps per epoch, and peak memory per GPU.

- [ ] **Step 5: Re-derive the timeline from the measurement**

The spec's 8–12 weeks rests on an unmeasured guess. Replace it with arithmetic:

```
steps_per_epoch  = 2800 clips * 5 pairs / effective_batch_size
hours_per_run    = seconds_per_step * steps_per_epoch * 100 / 3600
runs_per_week    = 30 / hours_per_run          # 30h weekly quota
```

Write the measured numbers and the resulting estimate into `docs/decisions.md`
as **D12**. **If `hours_per_run` implies the 6-run matrix exceeds ~8 weeks of
quota, stop and raise it** — the options (fewer ablations, smaller batch,
AMP, paid compute) are the user's call, not the implementer's.

- [ ] **Step 6: Confirm resume works on Kaggle for real**

Run a session for ~200 steps, push the checkpoint, start a fresh session, and
confirm Cell 2 finds it and training continues from that step rather than 0.
The unit test in Task 7 proves the state logic; this proves the plumbing.

- [ ] **Step 7: Commit**

```bash
git add kaggle/train.ipynb kaggle/README.md docs/decisions.md
git commit -m "feat: add Kaggle session driver; record measured step time (D12)"
```

---

### Task 11: M1 — TPSMM baseline run and TCD calibration

The gate. Everything downstream depends on this passing.

**Files:**
- Create: `pgmm/config/lsa64-tpsmm.yaml`, `scripts/calibrate_tcd.py`
- Modify: `docs/results.md`, `docs/decisions.md`

**Interfaces:**
- Consumes: `evaluate_clips` (Task 9), `TCDVariant` (Task 2), `latest_checkpoint` (Task 8)
- Produces: a trained TPSMM checkpoint; a resolved D3; the baseline row of `docs/results.md`

- [ ] **Step 1: Write the LSA64 TPSMM config**

Copy TPSMM's `config/taichi-256.yaml` to `pgmm/config/lsa64-tpsmm.yaml` and change
only what the paper specifies — every other value stays at TPSMM's default so
`L_r` is inherited exactly (D7):

```yaml
dataset_params:
  root_dir: /kaggle/input/lsa64-prepared-128
  frame_shape: [128, 128, 3]        # paper: 128x128
  id_sampling: False
  augmentation_params:
    flip_param:
      horizontal_flip: True
      time_flip: True

model_params:
  common_params:
    num_tps: 10                      # TPSMM default; paper says 20 -> Step 2
    num_channels: 3
    bg: True

train_params:
  num_epochs: 100                    # paper
  num_repeats: 5                     # paper: 5 source/driving pairs per video
  epoch_milestones: [60, 90]         # paper: lr /10 at 60 and 90
  lr_generator: 2.0e-4               # paper
  batch_size: 28                     # provisional -> Task 10 measurement decides
```

- [ ] **Step 2: Set the TPS group count to the paper's value**

The paper states TPSMM is configured with **100 keypoints in 20 TPS groups
(K=20)**, and that MRAA is matched at K=20. TPSMM's default is `num_tps: 10`.
Set `num_tps: 20` and confirm against `third_party/tpsmm/config/*.yaml` that
`num_tps` is the knob that yields `5 * num_tps` keypoints (TPSMM uses 5 points
per TPS group → 20 × 5 = 100, matching the paper).

**If the vendored code does not produce 100 keypoints at `num_tps: 20`, stop
and record the discrepancy as a decision** rather than guessing — the keypoint
count changes the baseline, and the baseline is the whole gate.

- [ ] **Step 3: Train the baseline to completion**

Run across as many Kaggle sessions as needed, pushing the checkpoint each time.
Expected: 100 epochs; loss curve continuous across resumes (spot-check that no
discontinuity appears at a session boundary — that would mean Task 7's gate is
passing while the real loop still loses state).

- [ ] **Step 4: Write the TCD calibration script**

```python
# scripts/calibrate_tcd.py
"""Resolve D3 by calibration.

The paper never states the value range its T=0.5 applies to, nor the channel
reduction. But it does publish TPSMM's TCD on LSA64 = 0.130. So we score our
reproduced baseline under every candidate reading and adopt the one that lands
on the published value.

This calibrates the *metric's definition* against the paper — it does not tune
the *model*. The distinction matters: the model is never touched to chase a
number.
"""

import argparse
from pathlib import Path

from pgmm.data.lsa64 import split_clips
from pgmm.eval.reconstruct import evaluate_clips
from pgmm.metrics.tcd import TCDVariant

PAPER_TPSMM_TCD = 0.130


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from pgmm.eval.tpsmm_adapter import build_reconstruct_fn  # Step 5

    reconstruct_fn = build_reconstruct_fn(args.checkpoint, device=args.device)
    all_ids = sorted(p.name for p in args.prepared_dir.iterdir() if p.is_dir())
    _, test_ids = split_clips(all_ids, mode="random", seed=0)

    print(f"{'variant':<12} {'TCD':>8} {'|diff|':>8}")
    results = {}
    for variant in TCDVariant:
        scores = evaluate_clips(
            reconstruct_fn, args.prepared_dir, test_ids,
            tcd_variant=variant, device=args.device,
        )
        value = scores["tcd"]
        results[variant] = value
        print(f"{variant.value:<12} {value:>8.4f} {abs(value - PAPER_TPSMM_TCD):>8.4f}")

    best = min(results, key=lambda v: abs(results[v] - PAPER_TPSMM_TCD))
    print(f"\nclosest to the paper's {PAPER_TPSMM_TCD}: {best.value} "
          f"({results[best]:.4f})")
    print("Record this in docs/decisions.md under D3, including the full table "
          "above — the losing candidates are evidence too.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the TPSMM adapter**

```python
# pgmm/eval/tpsmm_adapter.py
"""Wrap a trained TPSMM checkpoint as a ``reconstruct_fn``.

Follows TPSMM's reconstruction protocol: the first frame is the source, every
frame is a driving frame in turn.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_TPSMM = Path(__file__).resolve().parents[2] / "third_party" / "tpsmm"
if str(_TPSMM) not in sys.path:
    sys.path.insert(0, str(_TPSMM))


def build_reconstruct_fn(checkpoint: Path, device: str = "cuda"):
    """Load TPSMM and return ``clip -> reconstruction`` (both (N,H,W,C) [0,1])."""
    # Construct the networks exactly as third_party/tpsmm/run.py does, load the
    # checkpoint, and set eval mode. Read run.py and demo.py in the vendored
    # tree for the current constructor signatures rather than assuming them —
    # they are version-specific and must not be guessed.
    raise NotImplementedError(
        "Implement against the vendored TPSMM's actual API; see "
        "third_party/tpsmm/demo.py for the load-and-animate reference path."
    )
```

**This is the one deliberate stub in the plan.** TPSMM's constructor signatures
vary across revisions, and the vendored commit is not pinned until Task 1 runs.
Writing invented signatures here would be worse than an explicit stop: read
`third_party/tpsmm/demo.py` in the vendored tree and implement against what is
actually there. The contract it must satisfy is fixed and tested by Task 9.

- [ ] **Step 6: Run the calibration**

Run: `python scripts/calibrate_tcd.py --prepared-dir <dir> --checkpoint <ckpt>`
Expected: a table of four TCD values, one close to 0.130.

**If no candidate lands near 0.130**, do not pick the least-bad one and move on.
It means the reading is something we have not enumerated, or the baseline itself
is off. Record the table, add candidates, and treat it as a gate failure.

- [ ] **Step 7: Score the baseline and compare against Table 1**

Score the full test set with the calibrated variant, then write `docs/results.md`:

```markdown
# Results

## M1 — TPSMM baseline, LSA64 (gate)

| Metric | Paper | Ours | Δ | Within tolerance |
|---|---|---|---|---|
| L1 ↓ | 0.01342 | ... | ... | ±5% |
| SSIM ↑ | 0.9208 | ... | ... | ±0.005 |
| LPIPS ↓ | 0.02261 | ... | ... | ±5% |
| FVD ↓ | 182.785 | not measured | — | deferred to the next plan |
| TCD ↓ | 0.130 | ... | ... | calibration target for D3, not a gate |

Decisions in force: D3=<variant>, D7, D9=random/seed0, D10=alex, D11=skimage.
TPS groups: num_tps=20 (100 keypoints), per the paper.
```

- [ ] **Step 8: Judge the gate**

- **Within tolerance on L1/SSIM/LPIPS → M1 passes.** Record D3's resolution and
  proceed to plan M2–M4.
- **Outside tolerance → M1 fails. Stop.** Do not start PGMM. Investigate in
  this order, because it runs cheapest-first and D9 is the likeliest culprit:
  1. **D9 (the split).** A random split lets the same signer appear in train and
     test, which is *easier* than a held-out split and would make our numbers
     look better than the paper's. If we are better than the paper, suspect this
     first — try `mode="signer"`.
  2. **num_tps / keypoint count** (Step 2).
  3. **D10/D11** metric implementations — these move the third decimal, so they
     matter only for a near-miss.
  4. **Preprocessing** — centre-crop vs. the paper's unstated crop.

  Record the investigation in `docs/results.md` **whether or not it succeeds**.
  A documented inability to reproduce the baseline is a legitimate replication
  finding and is reported as one.

- [ ] **Step 9: Commit**

```bash
git add pgmm/config/lsa64-tpsmm.yaml scripts/calibrate_tcd.py \
        pgmm/eval/tpsmm_adapter.py docs/results.md docs/decisions.md
git commit -m "feat: reproduce TPSMM baseline on LSA64; resolve TCD reading (D3)"
```

---

## Deferred to the next plan (post-gate)

- **FVD (D8).** The spec lists it, but it is exempt from the numeric tolerance
  and cannot fail the gate, so it does not block M1. It is genuinely needed for
  the M3 comparison and is the first task of the next plan.
- **M2–M4:** CMM, PFM, PoseNet, `L_pd` (soft-argmax, D2), `L_align`, the D1
  keypoint search, the D5 attention-memory resolution, and runs R1–R5.

These are planned after M1 reports, because their content depends on what M1
measures: the resolved TCD reading, the real step time (D12), and whether the
harness holds.

## Day-1 parallel actions (not blocking any task)

1. Submit the PHOENIX14T access request.
2. Submit the CSL-Daily access request.
3. Open an issue on `shitongkai/PGMM` asking about the code release **and D1
   (which 36 keypoints)**. If the authors answer D1, the largest unknown in the
   project collapses; if they release code, this plan changes substantially.
   Either way it is worth knowing in week 1 rather than week 8.
