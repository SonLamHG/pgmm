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
