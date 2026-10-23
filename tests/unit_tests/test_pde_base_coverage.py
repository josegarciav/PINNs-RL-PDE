"""Tests for uncovered paths in src/pdes/pde_base.py."""

import pytest
import torch

from pinnrl.config import (
    AdaptiveWeightsConfig,
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    TrainingConfig,
)
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.pdes.pde_base import PDEBase, PDEConfig

DEVICE = torch.device("cpu")


# ── Helpers ──────────────────────────────────────────────────────────────


def _training_config():
    """Minimal TrainingConfig for PDEBase tests."""
    return TrainingConfig(
        num_epochs=1,
        batch_size=32,
        num_collocation_points=64,
        num_boundary_points=32,
        num_initial_points=32,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=1.0,
        early_stopping=EarlyStoppingConfig(enabled=False, patience=5, min_delta=1e-7),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=3
        ),
        adaptive_weights=AdaptiveWeightsConfig(enabled=False),
    )


def _make_heat_pde(**overrides):
    """Create a HeatEquation instance with sensible defaults."""
    defaults = dict(
        name="heat",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": 0.01},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        dimension=1,
        device=DEVICE,
        training=_training_config(),
    )
    defaults.update(overrides)
    return HeatEquation(config=PDEConfig(**defaults))


def _simple_model():
    """Tiny feedforward model for testing."""
    from pinnrl.config import Config, ModelConfig

    config = Config.__new__(Config)
    config.device = DEVICE
    config.model = ModelConfig(
        input_dim=2,
        hidden_dim=8,
        output_dim=1,
        num_layers=2,
        activation="tanh",
        architecture="feedforward",
    )
    config.training = _training_config()
    from pinnrl.neural_networks import PINNModel

    return PINNModel(config=config, device=DEVICE)


# ── PDEBase.__init__ domain handling ─────────────────────────────────────


def test_init_old_format_domain():
    """Old-style flat domain [xmin, xmax] is normalised to [(xmin, xmax)]."""
    pde = _make_heat_pde(domain=[0.0, 2.0])
    assert pde.domain == [(0.0, 2.0)]


def test_init_tuple_domain():
    """List-of-tuples domain passes through correctly."""
    pde = _make_heat_pde(domain=[(0.0, 1.0)])
    assert pde.domain == [(0.0, 1.0)]


def test_init_list_of_lists_domain():
    """List-of-lists domain is normalised to list-of-tuples."""
    pde = _make_heat_pde(domain=[[0.0, 1.0]])
    assert pde.domain == [(0.0, 1.0)]


def test_init_no_device():
    """When config has no device, falls back to CPU."""
    pde = _make_heat_pde(device=None)
    assert pde.device == torch.device("cpu")


def test_init_string_device():
    """String device is converted to torch.device."""
    cfg = PDEConfig(
        name="heat",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": 0.01},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={},
        dimension=1,
        device="cpu",
        training=_training_config(),
    )
    pde = HeatEquation(config=cfg)
    assert pde.device == torch.device("cpu")


def test_init_none_parameters():
    """None parameters are replaced with empty dict."""
    pde = _make_heat_pde(parameters=None)
    assert pde.config.parameters == {} or isinstance(pde.config.parameters, dict)


def test_init_input_output_dims():
    """input_dim and output_dim default to dimension+1 and 1."""
    pde = _make_heat_pde()
    assert pde.config.input_dim == 2  # 1D + time
    assert pde.config.output_dim == 1


# ── get_parameter ────────────────────────────────────────────────────────


def test_get_parameter_existing():
    """get_parameter returns the parameter value."""
    pde = _make_heat_pde()
    assert pde.get_parameter("alpha") == 0.01


def test_get_parameter_default():
    """get_parameter returns default for missing key."""
    pde = _make_heat_pde()
    assert pde.get_parameter("missing", default=42.0) == 42.0


def test_get_parameter_required_missing():
    """get_parameter raises ValueError when required param is missing."""
    pde = _make_heat_pde()
    with pytest.raises(ValueError, match="Required parameter"):
        pde.get_parameter("nonexistent", required=True)


def test_get_parameter_no_parameters():
    """get_parameter with no parameters dict returns default or raises."""
    pde = _make_heat_pde()
    pde.config.parameters = None
    assert pde.get_parameter("alpha", default=99.0) == 99.0
    with pytest.raises(ValueError):
        pde.get_parameter("alpha", required=True)


# ── Initial-condition factory (boundary semantics: test_physics_consistency) ──


def _ic(pde, **params):
    return PDEBase._create_initial_condition(pde, params)


def test_initial_condition_sine():
    x, t = torch.tensor([[0.5]]), torch.tensor([[0.0]])
    u0 = _ic(_make_heat_pde(), type="sine", amplitude=2.0, frequency=1.0)
    assert torch.allclose(u0(x, t), 2.0 * torch.sin(torch.pi * x))


def test_initial_condition_tanh():
    x, t = torch.tensor([[0.5]]), torch.tensor([[0.0]])
    assert torch.allclose(
        _ic(_make_heat_pde(), type="tanh", epsilon=0.1)(x, t), torch.tanh(x / 0.1)
    )


def test_initial_condition_gaussian():
    u0 = _ic(_make_heat_pde(), type="gaussian", mean=0.0, std=0.1)
    assert u0(torch.zeros(1, 1), torch.zeros(1, 1)).item() == pytest.approx(1.0)


def test_initial_condition_fixed():
    u0 = _ic(_make_heat_pde(), type="fixed", value=7.0)
    assert torch.allclose(u0(torch.randn(5, 1), torch.zeros(5, 1)), torch.full((5, 1), 7.0))


def test_initial_condition_random_is_bounded_and_reproducible():
    pde = _make_heat_pde()
    x = torch.rand(100, 1)
    a = _ic(pde, type="random", amplitude=0.5, seed=1)(x, None)
    b = _ic(pde, type="random", amplitude=0.5, seed=1)(x, None)
    assert a.shape == (100, 1) and torch.equal(a, b)
    assert a.abs().max() <= 0.5 + 1e-6


def test_initial_condition_empty_means_none():
    assert PDEBase._create_initial_condition(_make_heat_pde(), {}) is None


def test_initial_condition_unknown_type_raises():
    with pytest.raises(ValueError, match="unsupported initial condition"):
        _ic(_make_heat_pde(), type="totally_unknown")


# ── Sampling strategies ──────────────────────────────────────────────────


def test_sample_uniform_1d():
    """Uniform sampling returns correct shapes."""
    pde = _make_heat_pde()
    x, t = pde._sample_uniform(100)
    assert x.shape[1] == 1
    assert t.shape[1] == 1
    assert x.shape[0] == t.shape[0]


def test_sample_stratified_1d():
    """Stratified sampling returns correct shapes."""
    pde = _make_heat_pde()
    x, t = pde._sample_stratified(100)
    assert x.shape == (100, 1)
    assert t.shape == (100, 1)


def test_sample_residual_based_no_model():
    """Residual-based sampling without model falls back to uniform."""
    pde = _make_heat_pde()
    x, t = pde._sample_residual_based(100, model=None)
    assert x.shape[1] == 1
    assert t.shape[1] == 1


def test_sample_residual_based_with_model():
    """Residual-based sampling with a model produces valid points."""
    pde = _make_heat_pde()
    model = _simple_model()
    x, t = pde._sample_residual_based(100, model=model)
    assert x.shape[1] == 1
    assert t.shape[1] == 1


def test_generate_collocation_uniform():
    """generate_collocation_points with uniform strategy."""
    pde = _make_heat_pde()
    x, t = pde.generate_collocation_points(100, strategy="uniform")
    assert x.shape[1] == 1
    assert t.shape[1] == 1


def test_generate_collocation_stratified():
    """generate_collocation_points with stratified strategy."""
    pde = _make_heat_pde()
    x, t = pde.generate_collocation_points(100, strategy="stratified")
    assert x.shape == (100, 1)
    assert t.shape == (100, 1)


def test_generate_collocation_residual_based():
    """generate_collocation_points with residual_based strategy."""
    pde = _make_heat_pde()
    x, t = pde.generate_collocation_points(100, strategy="residual_based")
    assert x.shape[1] == 1


def test_generate_collocation_unknown_strategy():
    """Unknown sampling strategy raises ValueError."""
    pde = _make_heat_pde()
    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        pde.generate_collocation_points(100, strategy="banana")


def test_generate_collocation_adaptive_no_agent():
    """Adaptive strategy without rl_agent falls back to uniform."""
    pde = _make_heat_pde()
    pde.rl_agent = None
    x, t = pde.generate_collocation_points(100, strategy="adaptive")
    assert x.shape[1] == 1


# ── compute_loss ─────────────────────────────────────────────────────────


def test_compute_loss_basic():
    """compute_loss returns dict with expected keys."""
    pde = _make_heat_pde()
    model = _simple_model()
    x, t = pde.generate_collocation_points(64)
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    losses = pde.compute_loss(model, x, t)
    assert "total" in losses
    assert "residual" in losses
    assert "boundary" in losses
    assert "initial" in losses
    assert all(isinstance(v, torch.Tensor) for v in losses.values())


def test_compute_loss_with_fixed_weights():
    """compute_loss uses fixed weights from config."""
    tc = _training_config()
    tc.loss_weights = {"residual": 5.0, "boundary": 10.0, "initial": 10.0}
    tc.adaptive_weights = AdaptiveWeightsConfig(enabled=False)
    pde = _make_heat_pde(training=tc)
    model = _simple_model()
    x, t = pde.generate_collocation_points(64)
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    losses = pde.compute_loss(model, x, t)
    assert losses["total"].item() > 0


def test_compute_loss_adaptive_weights():
    """compute_loss with adaptive weights enabled."""
    tc = _training_config()
    tc.adaptive_weights = AdaptiveWeightsConfig(enabled=True, strategy="rbw")
    pde = _make_heat_pde(training=tc)
    model = _simple_model()
    x, t = pde.generate_collocation_points(64)
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    losses = pde.compute_loss(model, x, t)
    assert "total" in losses


# ── compute_derivatives ──────────────────────────────────────────────────


def test_compute_derivatives_temporal():
    """compute_derivatives returns temporal derivatives."""
    pde = _make_heat_pde()
    model = _simple_model()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    derivs = pde.compute_derivatives(
        model, x, t, temporal_derivatives=[1], spatial_derivatives=None
    )
    assert "dt" in derivs
    assert derivs["dt"].shape == (10, 1)


def test_compute_derivatives_spatial():
    """compute_derivatives returns spatial derivatives."""
    pde = _make_heat_pde()
    model = _simple_model()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    derivs = pde.compute_derivatives(
        model, x, t, temporal_derivatives=None, spatial_derivatives={1, 2}
    )
    assert "dx" in derivs
    assert "dx2" in derivs
    assert "laplacian" in derivs


def test_compute_derivatives_second_time():
    """compute_derivatives handles second-order temporal derivative."""
    pde = _make_heat_pde()
    model = _simple_model()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    derivs = pde.compute_derivatives(
        model, x, t, temporal_derivatives=[1, 2], spatial_derivatives=None
    )
    assert "dt" in derivs
    assert "dt2" in derivs


def test_compute_derivatives_invalid_temporal_order():
    """Temporal derivative > 2 raises ValueError."""
    pde = _make_heat_pde()
    model = _simple_model()
    x = torch.rand(5, 1, requires_grad=True)
    t = torch.rand(5, 1, requires_grad=True)
    with pytest.raises(ValueError, match="Temporal derivative order"):
        pde.compute_derivatives(model, x, t, temporal_derivatives=[3], spatial_derivatives=None)


def test_compute_derivatives_invalid_spatial_order():
    """Spatial derivative > 4 raises ValueError."""
    pde = _make_heat_pde()
    model = _simple_model()
    x = torch.rand(5, 1, requires_grad=True)
    t = torch.rand(5, 1, requires_grad=True)
    with pytest.raises(ValueError, match="Spatial derivative order"):
        pde.compute_derivatives(model, x, t, temporal_derivatives=None, spatial_derivatives={5})


# ── validate ─────────────────────────────────────────────────────────────


def test_validate():
    """validate returns error metrics dict."""
    pde = _make_heat_pde()
    model = _simple_model()
    metrics = pde.validate(model, num_points=64)
    assert "l2_error" in metrics
    assert "max_error" in metrics
    assert "mean_error" in metrics


# ── save/load state ──────────────────────────────────────────────────────


# ── 2D sampling ──────────────────────────────────────────────────────────


# ── PDE registry ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pde_type, ic",
    [
        ("heat", {"type": "sine", "amplitude": 1.0, "frequency": 1.0}),
        ("wave", {"type": "sine", "amplitude": 1.0, "frequency": 1.0}),
        ("burgers", {"type": "sine", "amplitude": 1.0, "frequency": 1.0}),
        ("kdv", {"type": "soliton", "amplitude": 2.0, "speed": 1.0}),
        ("convection", {"type": "sine", "amplitude": 1.0, "frequency": 1.0}),
        ("allen_cahn", {"type": "tanh", "epsilon": 0.1}),
        ("black_scholes", {"type": "option", "strike": 100.0, "option_type": "call"}),
        ("pendulum", {"type": "small_angle", "initial_angle": 0.5}),
    ],
)
def test_create_pde_by_key(pde_type, ic):
    from pinnrl.pdes import create_pde, get_pde_spec

    cfg = PDEConfig(
        name=pde_type,
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={},
        boundary_conditions={},
        initial_condition=ic,
        exact_solution={},
        dimension=1,
        device=DEVICE,
        training=_training_config(),
    )
    pde = create_pde(pde_type, cfg)
    assert isinstance(pde, get_pde_spec(pde_type).cls)


def test_create_pde_cahn_hilliard():
    """create_pde for cahn_hilliard (2D PDE)."""
    from pinnrl.pdes import CahnHilliardEquation, create_pde

    cfg = PDEConfig(
        name="cahn_hilliard",
        domain=[[0.0, 1.0], [0.0, 1.0]],
        time_domain=[0.0, 0.5],
        parameters={"M": 1.0, "epsilon": 0.01},
        boundary_conditions={},
        initial_condition={"type": "random", "amplitude": 0.1},
        exact_solution={},
        dimension=2,
        device=DEVICE,
        training=_training_config(),
    )
    assert isinstance(create_pde("cahn_hilliard", cfg), CahnHilliardEquation)


def test_pde_spec_accepts_key_and_display_name():
    from pinnrl.pdes import HeatEquation, get_pde_spec

    assert get_pde_spec("heat") is get_pde_spec("Heat Equation")
    assert get_pde_spec("heat equation").cls is HeatEquation
    assert get_pde_spec("Heat Equation 2D").key == "heat_2d"


def test_pde_spec_unknown_raises():
    from pinnrl.pdes import get_pde_spec

    with pytest.raises(KeyError, match="Unknown PDE 'nonexistent'"):
        get_pde_spec("nonexistent")


def test_registry_covers_config_yaml():
    """Every pde_configs entry in config.yaml resolves, by key and by its name field."""
    import yaml

    from pinnrl.config import DEFAULT_CONFIG_PATH
    from pinnrl.pdes import get_pde_spec

    with open(DEFAULT_CONFIG_PATH) as f:
        pde_configs = yaml.safe_load(f)["pde_configs"]
    for key, cfg in pde_configs.items():
        assert get_pde_spec(key).key == key
        if "name" in cfg:
            assert get_pde_spec(cfg["name"]).key == key


# ── 2D sampling ──────────────────────────────────────────────────────


def test_sample_uniform_2d():
    """2D uniform sampling returns multi-column x."""
    from pinnrl.pdes.cahn_hilliard import CahnHilliardEquation as CahnHilliard

    cfg = PDEConfig(
        name="cahn_hilliard",
        domain=[[0.0, 1.0], [0.0, 1.0]],
        time_domain=[0.0, 0.5],
        parameters={"M": 1.0, "epsilon": 0.01},
        boundary_conditions={"periodic": {}},
        initial_condition={"type": "random", "amplitude": 0.1},
        exact_solution={},
        dimension=2,
        device=DEVICE,
        training=_training_config(),
    )
    pde = CahnHilliard(config=cfg)
    x, t = pde._sample_uniform(100)
    assert x.shape[1] == 2
    assert t.shape[1] == 1


def test_sample_stratified_2d():
    """2D stratified sampling returns multi-column x."""
    from pinnrl.pdes.cahn_hilliard import CahnHilliardEquation as CahnHilliard

    cfg = PDEConfig(
        name="cahn_hilliard",
        domain=[[0.0, 1.0], [0.0, 1.0]],
        time_domain=[0.0, 0.5],
        parameters={"M": 1.0, "epsilon": 0.01},
        boundary_conditions={"periodic": {}},
        initial_condition={"type": "random", "amplitude": 0.1},
        exact_solution={},
        dimension=2,
        device=DEVICE,
        training=_training_config(),
    )
    pde = CahnHilliard(config=cfg)
    x, t = pde._sample_stratified(100)
    assert x.shape == (100, 2)
    assert t.shape == (100, 1)


# ── Multi-dimensional derivatives ────────────────────────────────────────


class _Quadratic2D(torch.nn.Module):
    """u = x1² + x2², so ∇²u = 4 and u_t = 0."""

    def forward(self, inp):
        return inp[:, 0:1] ** 2 + inp[:, 1:2] ** 2


@pytest.mark.parametrize(
    "pde_cls, params, ic, expected",
    [
        ("WaveEquation", {"c": 1.0}, {"type": "sine", "amplitude": 1.0, "frequency": 1.0}, -4.0),
        (
            "HeatEquation",
            {"alpha": 0.1},
            {"type": "sine", "amplitude": 1.0, "frequency": 1.0},
            -0.4,
        ),
    ],
)
def test_2d_residual_includes_laplacian(pde_cls, params, ic, expected):
    """Regression: n-D derivatives used to be taken w.r.t. a column slice and vanish."""
    import pinnrl.pdes as pdes

    pde = getattr(pdes, pde_cls)(
        PDEConfig(
            name="p",
            domain=[[0.0, 1.0], [0.0, 1.0]],
            time_domain=[0.0, 1.0],
            parameters=params,
            boundary_conditions={},
            initial_condition=ic,
            exact_solution={},
            dimension=2,
            device=DEVICE,
        )
    )
    x, t = torch.rand(6, 2), torch.rand(6, 1)
    residual = pde.compute_residual(_Quadratic2D(), x, t)
    assert torch.allclose(residual, torch.full_like(residual, expected), atol=1e-5)


def test_laplacian_of_derived_field():
    """laplacian() differentiates any field built from the input leaf."""
    pde = _make_heat_pde()
    x = torch.rand(5, 1, requires_grad=True)
    f = torch.sin(3 * x)
    assert torch.allclose(pde.laplacian(f, x), -9 * torch.sin(3 * x), atol=1e-5)
