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
