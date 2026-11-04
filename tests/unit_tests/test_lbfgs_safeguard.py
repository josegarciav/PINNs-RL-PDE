"""The L-BFGS safeguard: a step that blows the loss up is rolled back, the learning rate halved,
and after repeated failures the trainer returns to Adam."""

import torch

from pinnrl.pdes.catalog import from_catalog


def _trainer(optimizer="lbfgs"):
    pde = from_catalog("heat")
    return pde, pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [8, 8]},
        optimizer=optimizer,
        learning_rate=1.0,
        num_collocation_points=64,
        batch_size=64,
        num_boundary_points=16,
        num_initial_points=16,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )


def test_normal_lbfgs_steps_lower_the_loss():
    pde, trainer = _trainer()
    assert trainer._is_lbfgs
    x, t = pde.generate_collocation_points(64)
    before = pde.compute_loss(trainer.model, x, t)["total"].item()
    out = trainer.train_steps(num_steps=3, resample_every=100)
    assert out["loss"][-1] < before and getattr(trainer, "_lbfgs_failures", 0) == 0


def test_blown_up_step_is_rolled_back_and_falls_back_to_adam(monkeypatch):
    pde, trainer = _trainer()
    x, t = pde.generate_collocation_points(64)
    original = pde.compute_loss
    calls = {"n": 0}

    def exploding(model, xx, tt):
        losses = original(model, xx, tt)
        calls["n"] += 1
        if calls["n"] > 1:  # the first call (the loss before the step) is honest; the rest explode
            losses = {k: v * 1e6 if k == "total" else v for k, v in losses.items()}
        return losses

    weights_before = [p.detach().clone() for p in trainer.model.parameters()]
    lr_before = trainer.optimizer.param_groups[0]["lr"]
    monkeypatch.setattr(pde, "compute_loss", exploding)
    trainer._lbfgs_step(x, t)
    assert trainer._lbfgs_failures == 1
    assert trainer.optimizer.param_groups[0]["lr"] == lr_before / 2
    for p, w in zip(trainer.model.parameters(), weights_before, strict=True):
        assert torch.equal(p.detach(), w)  # rolled back
    calls["n"] = 0
    trainer._lbfgs_step(x, t)
    calls["n"] = 0
    trainer._lbfgs_step(x, t)
    assert not trainer._is_lbfgs  # three failures: back to Adam
    assert type(trainer.optimizer).__name__ == "Adam"


def test_nan_loss_is_caught(monkeypatch):
    pde, trainer = _trainer()
    x, t = pde.generate_collocation_points(64)
    original = pde.compute_loss
    calls = {"n": 0}

    def nan_after_first(model, xx, tt):
        losses = original(model, xx, tt)
        calls["n"] += 1
        if calls["n"] > 1:
            losses = {k: v * float("nan") if k == "total" else v for k, v in losses.items()}
        return losses

    monkeypatch.setattr(pde, "compute_loss", nan_after_first)
    weights = [p.detach().clone() for p in trainer.model.parameters()]
    trainer._lbfgs_step(x, t)
    assert trainer._lbfgs_failures == 1
    for p, w in zip(trainer.model.parameters(), weights, strict=True):
        assert torch.equal(p.detach(), w)
