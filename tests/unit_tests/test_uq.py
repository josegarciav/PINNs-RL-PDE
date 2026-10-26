"""pinnrl.uq: metrics, swarm sampling, Laplace posterior, ensembles."""

import math

import numpy as np
import pytest
import torch

from pinnrl.pdes.catalog import from_catalog
from pinnrl.uq import (
    coverage,
    crps_gaussian,
    crps_samples,
    ensemble_summary,
    gaussian_interval,
    laplace_posterior,
    likelihood_weights,
    mean_width,
    misfit_tolerance,
    sample_equivalence_region,
    sample_interval,
    second_order_stable,
)


def test_metrics():
    lo, hi = gaussian_interval(0.0, 1.0, 0.9)
    assert lo == pytest.approx(-1.6449, abs=1e-3) and hi == pytest.approx(1.6449, abs=1e-3)
    assert coverage([0, 0, 0], [1, 1, 1], [0.5, 2.0, 1.0]) == pytest.approx(2 / 3)
    assert mean_width([0, 0], [1, 3]) == 2.0
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 20000)
    lo, hi = sample_interval(s, 0.9)
    assert lo == pytest.approx(-1.645, abs=0.05) and hi == pytest.approx(1.645, abs=0.05)
    # CRPS of a perfect Gaussian forecast equals its expectation sigma (1/sqrt(pi) - ... ) ≈ 0.234 sigma on average.
    assert crps_gaussian(0.0, 1.0, 0.0) == pytest.approx(
        2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), rel=1e-6
    )
    assert crps_samples(s, 0.0) == pytest.approx(crps_gaussian(0.0, 1.0, 0.0), abs=0.02)
    assert crps_gaussian(0.0, 1.0, 3.0) > crps_gaussian(0.0, 1.0, 0.5)


def test_stability_region():
    assert second_order_stable(0.7, 1.0)
    assert not second_order_stable(0.7, 5.0)
    assert not second_order_stable(1.2, 1.0)


def test_swarm_samples_a_banana_region():
    # y = a (1 - exp(-b x)) at x in [1, 30]: the nonlinear example of Pallero et al. (2018).
    xs = np.linspace(1, 30, 40)
    true = np.array([20.0, 0.1])
    rng = np.random.default_rng(1)
    sigma = 0.3
    y = true[0] * (1 - np.exp(-true[1] * xs)) + sigma * rng.normal(size=xs.size)

    def misfit(theta):
        pred = theta[:, :1] * (1 - np.exp(-theta[:, 1:2] * xs[None, :]))
        return ((pred - y[None, :]) ** 2).sum(axis=1) / sigma**2

    tol = misfit_tolerance(xs.size, 1.0, epsilon=0.25)  # misfit is already in sigma units
    out = sample_equivalence_region(
        misfit, [(5, 40), (0.02, 0.3)], tol, n_particles=40, n_iterations=150, seed=0
    )
    assert out.accepted.sum() > 100
    assert out.best_misfit < tol
    lo, hi = out.interval(0.9)
    assert lo[0] < true[0] < hi[0] and lo[1] < true[1] < hi[1]
    assert out.dispersion[0] == 1.0 and out.positions.shape[1] == 2
    # The accepted set is elongated along the a-b trade-off (positive correlation of a with 1/b).
    s = out.samples
    assert np.corrcoef(s[:, 0], s[:, 1])[0, 1] < 0


def test_likelihood_weights():
    w = likelihood_weights(n_data=200, sigma_data=0.05, n_residual=1000, sigma_residual=0.1)
    assert w["residual"] == 1.0 and w["data"] == pytest.approx((200 / 0.05**2) / (1000 / 0.1**2))


def test_ensemble_summary():
    e = [[0.1, 1.0], [0.11, 1.1], [0.09, 0.9], [0.1, 1.05]]
    s = ensemble_summary(e)
    assert s.mean.shape == (2,) and s.std[0] > 0 and 0 < s.dispersion < 1
    iv = s.interval(0.5)
    assert iv.shape == (2, 2) and iv[0, 0] <= 0.1 <= iv[1, 0]
    assert s.interval(0.9, gaussian=True)[1, 0] > s.mean[0]


def test_laplace_posterior_on_heat():
    torch.manual_seed(0)
    pde = from_catalog(
        "heat", trainable_parameters=["alpha"], parameter_initial_guesses={"alpha": 0.1}
    )
    pde.generate_synthetic_observations(n_points=100, noise_std=0.05, seed=0)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [32, 32, 32], "layer_norm": False},
        mode="data_augmented",
        num_collocation_points=1024,
        batch_size=1024,
        num_boundary_points=256,
        num_initial_points=256,
        learning_rate=2e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
    )
    # Hold alpha at the truth: data-augmented mode trains the weights only.
    trainer.train_steps(num_steps=1500, resample_every=100)
    post = laplace_posterior(pde, trainer.model, ["alpha"], 0.05)
    assert post.mean.shape == (1,) and post.std.shape == (1,)
    assert 0.001 < float(post.std[0]) < 0.02  # a few percent of alpha = 0.1 at 100 noisy points
    lo, hi = post.interval(0.9)
    assert lo[0] < 0.1 < hi[0]
    draws = post.sample(500)
    assert draws.shape == (500, 1) and abs(float(draws.std()) - float(post.std[0])) < 0.3 * float(
        post.std[0]
    )
