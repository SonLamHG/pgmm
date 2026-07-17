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
