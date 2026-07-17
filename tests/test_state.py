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
