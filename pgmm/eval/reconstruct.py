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
