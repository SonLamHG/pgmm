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
