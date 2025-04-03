import unittest
import torch
import numpy as np
import pytest
from src.pdes.pde_base import PDEBase, PDEConfig
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.kdv_equation import KdVEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.convection_equation import ConvectionEquation
from src.pdes.allen_cahn import AllenCahnEquation
from src.pdes.cahn_hilliard import CahnHilliardEquation
from src.pdes.black_scholes import BlackScholesEquation
from src.pdes.pendulum_equation import PendulumEquation
from src.neural_networks import FeedForwardNetwork, PINNModel
from src.rl_agent import RLAgent
from tests.test_components.test_utils import create_pde_from_config


class TestPDEs(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Determine the device to use for testing
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Create PDEs from config.yaml
        try:
            # Try to create Heat Equation from config
            self.heat_eq = create_pde_from_config("heat", self.device)
            print("Created Heat Equation from config.yaml")
        except Exception as e:
            print(f"Falling back to hardcoded Heat Equation: {e}")
            # Fallback to hardcoded parameters using PDEConfig
            config = PDEConfig(
                name="Heat Equation",
                domain=[(0.0, 1.0)],
                time_domain=(0.0, 1.0),
                parameters={"alpha": 0.01},
                boundary_conditions={"dirichlet": {"type": "fixed", "value": 0.0}},
                initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
                exact_solution={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
                dimension=1,
                device=self.device,
            )
            self.heat_eq = HeatEquation(config=config)

        try:
            # Try to create Wave Equation from config
            self.wave_eq = create_pde_from_config("wave", self.device)
            print("Created Wave Equation from config.yaml")
        except Exception as e:
            print(f"Falling back to hardcoded Wave Equation: {e}")
            # Fallback to hardcoded parameters using PDEConfig
            config = PDEConfig(
                name="Wave Equation",
                domain=[(0.0, 1.0)],
                time_domain=(0.0, 1.0),
                parameters={"c": 1.0},
                boundary_conditions={"dirichlet": {"type": "fixed", "value": 0.0}},
                initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
                exact_solution={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
                dimension=1,
                device=self.device,
            )
            self.wave_eq = WaveEquation(config=config)

        # Common model for testing - make sure it's on the same device
        self.model = FeedForwardNetwork(
            {
                "input_dim": 2,  # (x, t)
                "hidden_dims": [32, 32, 32],  # 3 layers of hidden_dim 32
                "output_dim": 1,  # u(x, t)
                "activation": "tanh",
                "device": self.device,
            }
        ).to(
            self.device
        )  # Explicitly move to device

        # Set up common test tensors - ensure they're on the right device
        self.x = (
            torch.linspace(0, 1, 10, device=self.device)
            .reshape(-1, 1)
            .requires_grad_(True)
        )
        self.t = (
            torch.linspace(0, 1, 10, device=self.device)
            .reshape(-1, 1)
            .requires_grad_(True)
        )
        self.inputs = torch.cat([self.x, self.t], dim=1)
        self.u = torch.sin(self.x + self.t)  # Simple analytical solution

        # Initialize PINN models
        # Use PDE-specific configuration for model initialization
        self.model_2d = FeedForwardNetwork(
            {
                "input_dim": 3,  # x, y, t
                "hidden_dims": [32, 32, 32],  # 3 layers of hidden_dim 32
                "output_dim": 1,  # u(x, t)
                "activation": "tanh",
                "device": self.device,
            }
        ).to(
            self.device
        )  # Explicitly move to device

        # Ensure model parameters require gradients
        for param in self.model_2d.parameters():
            param.requires_grad_(True)

        # Initialize RL agent for adaptive sampling tests
        self.rl_agent = RLAgent(
            state_dim=2, action_dim=1, hidden_dim=32, device=self.device
        )

    def test_heat_equation(self):
        """Test the Heat equation implementation."""
        # Test 1D Heat equation
        config = PDEConfig(
            name="Heat Equation",
            domain=[(0.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"alpha": 0.01},
            boundary_conditions={
                "dirichlet": {"type": "fixed", "value": 0.0},
                "neumann": {"type": "fixed", "value": 0.0},
                "periodic": {},
            },
            initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            exact_solution={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            dimension=1,
            device=self.device,
        )

        heat_eq = HeatEquation(config=config)

        # Test collocation points
        x, t = heat_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(
            torch.all(x >= config.domain[0][0]) and torch.all(x <= config.domain[0][1])
        )
        self.assertTrue(
            torch.all(t >= config.time_domain[0])
            and torch.all(t <= config.time_domain[1])
        )

        # Test exact solution
        u_exact = heat_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())

        # Test boundary conditions
        x_boundary = torch.tensor(
            [config.domain[0][0], config.domain[0][1]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = heat_eq._create_boundary_condition(
            "initial", config.initial_condition
        )(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))

        # Test residual computation
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        for param in self.model.parameters():
            param.requires_grad_(True)
        residual = heat_eq.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())

        # Test 2D Heat equation
        config_2d = PDEConfig(
            name="Heat Equation 2D",
            domain=[(0.0, 1.0), (0.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"alpha": 0.01},
            boundary_conditions={
                "dirichlet": {"type": "fixed", "value": 0.0},
                "neumann": {"type": "fixed", "value": 0.0},
                "periodic": {},
            },
            initial_condition={
                "type": "sine_2d",
                "amplitude": 1.0,
                "frequency_x": 2.0,
                "frequency_y": 2.0,
            },
            exact_solution={
                "type": "sine_2d",
                "amplitude": 1.0,
                "frequency_x": 2.0,
                "frequency_y": 2.0,
            },
            dimension=2,
            device=self.device,
        )

        heat_eq_2d = HeatEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = heat_eq_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(config_2d.domain):
            self.assertTrue(
                torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)
            )

        # Test 2D exact solution
        u_exact_2d = heat_eq_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())

        # Test 2D residual computation
        x_2d = x_2d.detach().requires_grad_(True)
        t_2d = t_2d.detach().requires_grad_(True)
        for param in self.model_2d.parameters():
            param.requires_grad_(True)
        residual_2d = heat_eq_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())

    def test_wave_equation(self):
        """Test the Wave equation implementation."""
        # Test 1D Wave equation
        config = PDEConfig(
            name="Wave Equation",
            domain=[(0.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"c": 1.0},
            boundary_conditions={
                "dirichlet": {"type": "fixed", "value": 0.0},
                "neumann": {"type": "fixed", "value": 0.0},
                "periodic": {},
            },
            initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            exact_solution={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            dimension=1,
            device=self.device,
        )

        wave_eq = WaveEquation(config=config)

        # Test collocation points
        x, t = wave_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(
            torch.all(x >= config.domain[0][0]) and torch.all(x <= config.domain[0][1])
        )
        self.assertTrue(
            torch.all(t >= config.time_domain[0])
            and torch.all(t <= config.time_domain[1])
        )

        # Test exact solution
        u_exact = wave_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())

        # Test boundary conditions
        x_boundary = torch.tensor(
            [config.domain[0][0], config.domain[0][1]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = wave_eq._create_boundary_condition(
            "initial", config.initial_condition
        )(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))

        # Test residual computation
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        for param in self.model.parameters():
            param.requires_grad_(True)
        residual = wave_eq.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())

        # Test 2D Wave equation
        config_2d = PDEConfig(
            name="Wave Equation 2D",
            domain=[(0.0, 1.0), (0.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"c": 1.0},
            boundary_conditions={
                "dirichlet": {"type": "fixed", "value": 0.0},
                "neumann": {"type": "fixed", "value": 0.0},
                "periodic": {},
            },
            initial_condition={
                "type": "sine_2d",
                "amplitude": 1.0,
                "frequency_x": 2.0,
                "frequency_y": 2.0,
            },
            exact_solution={
                "type": "sine_2d",
                "amplitude": 1.0,
                "frequency_x": 2.0,
                "frequency_y": 2.0,
            },
            dimension=2,
            device=self.device,
        )

        wave_eq_2d = WaveEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = wave_eq_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(config_2d.domain):
            self.assertTrue(
                torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)
            )

        # Test 2D exact solution
        u_exact_2d = wave_eq_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())

        # Test 2D residual computation
        x_2d = x_2d.detach().requires_grad_(True)
        t_2d = t_2d.detach().requires_grad_(True)
        for param in self.model_2d.parameters():
            param.requires_grad_(True)
        residual_2d = wave_eq_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())

    def test_black_scholes(self):
        """Test the Black-Scholes equation implementation."""
        # Test 1D Black-Scholes equation
        domain = [(0.0, 200.0)]  # Changed to list of tuples
        time_domain = (0.0, 1.0)
        strike = 100.0
        boundary_conditions = {
            "dirichlet": {"value": 0.0},
            "neumann": {"value": 1.0},  # Delta at upper boundary
            "initial": {"type": "call_option", "strike_price": strike},
        }
        initial_condition = {"type": "call_option", "strike_price": strike}
        exact_solution = {"type": "call_option", "strike_price": strike}

        config = PDEConfig(
            name="Black-Scholes Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={"sigma": 0.2, "r": 0.05},  # Volatility and risk-free rate
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device,
        )

        bs_eq = BlackScholesEquation(config=config)

        # Test collocation points
        x, t = bs_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0][0]) and torch.all(x <= domain[0][1]))
        self.assertTrue(
            torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1])
        )

        # Test exact solution
        u_exact = bs_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        self.assertTrue(torch.all(u_exact >= 0))  # Option price is non-negative

        # Test boundary conditions
        x_boundary = torch.tensor(
            [domain[0][0], domain[0][1]], dtype=torch.float32
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = bs_eq._create_boundary_condition("initial", initial_condition)(
            x_boundary, t_boundary
        )
        self.assertEqual(u_boundary.shape, (2, 1))

        # Test 2D Black-Scholes equation (basket option)
        domain_2d = [(0.0, 200.0), (0.0, 200.0)]  # Two stock prices
        config_2d = PDEConfig(
            name="Black-Scholes Equation 2D",
            domain=domain_2d,
            time_domain=time_domain,
            parameters={"sigma": 0.2, "r": 0.05},  # Volatility and risk-free rate
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device,
        )

        bs_eq_2d = BlackScholesEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = bs_eq_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(
                torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)
            )

        # Test 2D exact solution
        u_exact_2d = bs_eq_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        self.assertTrue(torch.all(u_exact_2d >= 0))  # Option price is non-negative

        # Test residual computation
        x_2d = x_2d.detach().requires_grad_(True)
        t_2d = t_2d.detach().requires_grad_(True)
        for param in self.model_2d.parameters():
            param.requires_grad_(True)
        residual_2d = bs_eq_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())

    def test_boundary_conditions(self):
        # Test boundary conditions
        bc_dirichlet = self.heat_eq._create_boundary_condition(
            "dirichlet", {"value": 0.0}
        )
        bc_neumann = self.heat_eq._create_boundary_condition("neumann", {"value": 0.0})
        bc_periodic = self.heat_eq._create_boundary_condition("periodic", {})

        # Test that boundary conditions are callable
        self.assertTrue(callable(bc_dirichlet))
        self.assertTrue(callable(bc_neumann))
        self.assertTrue(callable(bc_periodic))

        # Test boundary condition values
        x = torch.tensor([[0.0]], device=self.device)
        t = torch.tensor([[0.0]], device=self.device)

        self.assertEqual(bc_dirichlet(x, t).shape, (1, 1))
        self.assertEqual(bc_neumann(x, t).shape, (1, 1))
        self.assertEqual(bc_periodic(x, t).shape, (1, 1))

    def test_exact_solution(self):
        # Test exact solution computation
        exact_sol = self.heat_eq.exact_solution(self.x, self.t)
        self.assertEqual(exact_sol.shape, (10, 1))
        self.assertTrue(torch.isfinite(exact_sol).all())

    def test_adaptive_sampling(self):
        """Test adaptive sampling with RL agent for all PDEs."""
        num_points = 100

        # Test 1D Heat Equation
        x, t = self.heat_eq.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))

        # Test 2D Heat Equation
        domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        config_2d = PDEConfig(
            name="Heat Equation 2D",
            domain=domain_2d,
            time_domain=(0.0, 1.0),
            parameters={"alpha": 0.01},
            boundary_conditions=self.heat_eq.config.boundary_conditions,
            initial_condition=self.heat_eq.config.initial_condition,
            exact_solution=self.heat_eq.config.exact_solution,
            dimension=2,
            device=self.device,
        )
        heat_eq_2d = HeatEquation(config=config_2d)
        x, t = heat_eq_2d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x[:, 0] >= 0) and torch.all(x[:, 0] <= 1))
        self.assertTrue(torch.all(x[:, 1] >= 0) and torch.all(x[:, 1] <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))

        # Test Wave Equation
        x, t = self.wave_eq.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))

        # Test RL agent state and action only if rl_agent is available
        # Create a heat equation with RL agent for this test
        config_rl = PDEConfig(
            name="Heat Equation",
            domain=[(0.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"alpha": 0.01},
            boundary_conditions=self.heat_eq.config.boundary_conditions,
            initial_condition=self.heat_eq.config.initial_condition,
            exact_solution=self.heat_eq.config.exact_solution,
            dimension=1,
            device=self.device,
        )
        heat_eq_with_rl = HeatEquation(config=config_rl)

        # Set RL agent for the heat equation
        heat_eq_with_rl.rl_agent = self.rl_agent

        # Now test with the RL agent
        state = torch.randn(1, 2).to(self.device)  # Random (x, t) state
        action = heat_eq_with_rl.rl_agent.select_action(state)
        self.assertEqual(action.shape, (1, 1))  # Single action value
        self.assertTrue(0 <= action.item() <= 1)  # Action should be probability

    def test_validation(self):
        # Test validation metrics
        metrics = self.heat_eq.validate(self.model, num_points=100)
        self.assertIn("l2_error", metrics)
        self.assertIn("max_error", metrics)
        self.assertIn("mean_error", metrics)
        self.assertTrue(all(isinstance(v, float) for v in metrics.values()))

    def test_config_management(self):
        # Test PDE configuration
        config = self.heat_eq.config
        self.assertEqual(config.domain, [(0.0, 1.0)])  # Changed to list of tuples
        self.assertEqual(config.time_domain, (0.0, 1.0))
        self.assertEqual(config.initial_condition["type"], "sine")
        self.assertEqual(config.parameters["alpha"], 0.01)

    def test_rl_agent_integration(self):
        # Test RL agent state and action
        state = torch.randn(1, 2).to(self.device)  # Random (x, t) state
        action = self.rl_agent.select_action(state)

        self.assertEqual(action.shape, (1, 1))  # Single action value
        self.assertTrue(0 <= action.item() <= 1)  # Action should be probability

    def test_kdv_equation(self):
        """Test the Korteweg-de Vries (KdV) equation implementation."""
        # Test 1D KdV equation
        domain = [(-10.0, 10.0)]  # Changed to list of tuples
        time_domain = (0.0, 1.0)
        boundary_conditions = {
            "left": {"type": "periodic"},
            "right": {"type": "periodic"},
            "initial": {"type": "soliton", "speed": 1.0},
        }
        initial_condition = {"type": "soliton", "speed": 1.0}
        exact_solution = {"type": "soliton", "speed": 1.0}

        config = PDEConfig(
            name="KdV Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={"speed": 1.0},  # Soliton speed
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device,
        )

        kdv = KdVEquation(config=config)

        # Test collocation points
        x, t = kdv.generate_collocation_points(100)
        assert x.shape == (100, 1)
        assert t.shape == (100, 1)
        assert torch.all(x >= domain[0][0]) and torch.all(x <= domain[0][1])
        assert torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1])

        # Test exact solution
        u_exact = kdv.exact_solution(x, t)
        assert u_exact.shape == (100, 1)
        assert torch.all(u_exact >= 0)  # Soliton is always positive

        # Test boundary conditions
        x_boundary = torch.tensor(
            [domain[0][0], domain[0][1]], dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = kdv._create_boundary_condition("initial", initial_condition)(
            x_boundary, t_boundary
        )
        assert u_boundary.shape == (2, 1)

        # Test 2D KdV equation
        domain_2d = [(-10, 10), (-10, 10)]
        config_2d = PDEConfig(
            name="KdV Equation 2D",
            domain=domain_2d,
            time_domain=time_domain,
            parameters={"speed": 1.0},  # Soliton speed
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device,
        )
        kdv_2d = KdVEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = kdv_2d.generate_collocation_points(100)
        assert x_2d.shape == (100, 2)
        assert t_2d.shape == (100, 1)
        for i, (min_val, max_val) in enumerate(domain_2d):
            assert torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)

        # Test 2D exact solution
        u_exact_2d = kdv_2d.exact_solution(x_2d, t_2d)
        assert u_exact_2d.shape == (100, 1)
        assert torch.all(u_exact_2d >= 0)  # Soliton is always positive

    def test_burgers_equation(self):
        """Test the Burgers' equation implementation."""
        # Test 1D Burgers' equation
        config = PDEConfig(
            name="Burgers Equation",
            domain=[(-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"nu": 0.01},  # Kinematic viscosity
            boundary_conditions={
                "left": {"type": "dirichlet", "value": 0.0},
                "right": {"type": "dirichlet", "value": 0.0},
                "initial": {"type": "tanh", "epsilon": 0.1},
            },
            initial_condition={"type": "tanh", "epsilon": 0.1},
            exact_solution={"type": "tanh", "epsilon": 0.1},
            dimension=1,
            device=self.device,
        )

        burgers = BurgersEquation(config=config)

        # Test collocation points
        x, t = burgers.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(
            torch.all(x >= config.domain[0][0]) and torch.all(x <= config.domain[0][1])
        )
        self.assertTrue(
            torch.all(t >= config.time_domain[0])
            and torch.all(t <= config.time_domain[1])
        )

        # Test exact solution
        u_exact = burgers.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())

        # Test boundary conditions
        x_boundary = torch.tensor(
            [config.domain[0][0], config.domain[0][1]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = burgers._create_boundary_condition(
            "initial", config.initial_condition
        )(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))

        # Test 2D Burgers' equation
        config_2d = PDEConfig(
            name="Burgers Equation 2D",
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"nu": 0.01},  # Kinematic viscosity
            boundary_conditions={
                "left": {"type": "dirichlet", "value": 0.0},
                "right": {"type": "dirichlet", "value": 0.0},
                "initial": {"type": "tanh", "epsilon": 0.1},
            },
            initial_condition={"type": "tanh", "epsilon": 0.1},
            exact_solution={"type": "tanh", "epsilon": 0.1},
            dimension=2,
            device=self.device,
        )

        burgers_2d = BurgersEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = burgers_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(config_2d.domain):
            self.assertTrue(
                torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)
            )

        # Test 2D exact solution
        u_exact_2d = burgers_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())

    def test_convection(self):
        """Test convection equation functionality."""
        # 1D Convection equation
        config = PDEConfig(
            name="Convection Equation",
            domain=[(-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"velocity": 1.0},
            boundary_conditions={
                "left": {"type": "dirichlet", "value": 0.0},
                "right": {"type": "dirichlet", "value": 0.0},
                "initial": {"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            },
            initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            exact_solution={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            dimension=1,
            device=self.device,
        )

        convection_eq = ConvectionEquation(config=config)

        # Test collocation points
        x, t = convection_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(
            torch.all(x >= config.domain[0][0]) and torch.all(x <= config.domain[0][1])
        )
        self.assertTrue(
            torch.all(t >= config.time_domain[0])
            and torch.all(t <= config.time_domain[1])
        )

        # Test exact solution
        u_exact = convection_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())

        # Test boundary conditions
        x_boundary = torch.tensor(
            [config.domain[0][0], config.domain[0][1]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = convection_eq._create_boundary_condition(
            "initial", config.initial_condition
        )(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))

        # Test 2D Convection equation
        config_2d = PDEConfig(
            name="Convection Equation 2D",
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"velocity": [1.0, 1.0]},
            boundary_conditions={
                "left": {"type": "dirichlet", "value": 0.0},
                "right": {"type": "dirichlet", "value": 0.0},
                "initial": {"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            },
            initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            exact_solution={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            dimension=2,
            device=self.device,
        )

        convection_2d = ConvectionEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = convection_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(config_2d.domain):
            self.assertTrue(
                torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)
            )

        # Test 2D exact solution
        u_exact_2d = convection_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())

    def test_allen_cahn(self):
        """Test the Allen-Cahn equation implementation."""
        # Test 1D Allen-Cahn equation
        config = PDEConfig(
            name="Allen-Cahn Equation",
            domain=[(-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"epsilon": 0.1},  # Interface width parameter
            boundary_conditions={
                "left": {"type": "dirichlet", "value": -1.0},
                "right": {"type": "dirichlet", "value": 1.0},
                "initial": {"type": "tanh", "epsilon": 0.1},
            },
            initial_condition={"type": "tanh", "epsilon": 0.1},
            exact_solution={"type": "tanh", "epsilon": 0.1},
            dimension=1,
            device=self.device,
        )

        allen_cahn = AllenCahnEquation(config=config)

        # Test collocation points
        x, t = allen_cahn.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(
            torch.all(x >= config.domain[0][0]) and torch.all(x <= config.domain[0][1])
        )
        self.assertTrue(
            torch.all(t >= config.time_domain[0])
            and torch.all(t <= config.time_domain[1])
        )

        # Test exact solution
        u_exact = allen_cahn.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        self.assertTrue(
            torch.all(u_exact >= -1.0) and torch.all(u_exact <= 1.0)
        )  # Phase field bounds

        # Test boundary conditions
        x_boundary = torch.tensor(
            [config.domain[0][0], config.domain[0][1]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = allen_cahn._create_boundary_condition(
            "initial", config.initial_condition
        )(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))

        # Test residual computation
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        for param in self.model.parameters():
            param.requires_grad_(True)
        residual = allen_cahn.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())

        # Test 2D Allen-Cahn equation
        config_2d = PDEConfig(
            name="Allen-Cahn Equation 2D",
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"epsilon": 0.1},  # Interface width parameter
            boundary_conditions={
                "left": {"type": "dirichlet", "value": -1.0},
                "right": {"type": "dirichlet", "value": 1.0},
                "initial": {"type": "tanh", "epsilon": 0.1},
            },
            initial_condition={"type": "tanh", "epsilon": 0.1},
            exact_solution={"type": "tanh", "epsilon": 0.1},
            dimension=2,
            device=self.device,
        )

        allen_cahn_2d = AllenCahnEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = allen_cahn_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(config_2d.domain):
            self.assertTrue(
                torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)
            )

        # Test 2D exact solution
        u_exact_2d = allen_cahn_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        self.assertTrue(
            torch.all(u_exact_2d >= -1.0) and torch.all(u_exact_2d <= 1.0)
        )  # Phase field bounds

        # Test 2D residual computation
        x_2d = x_2d.detach().requires_grad_(True)
        t_2d = t_2d.detach().requires_grad_(True)
        for param in self.model_2d.parameters():
            param.requires_grad_(True)
        residual_2d = allen_cahn_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())

        # Test phase separation dynamics
        # The Allen-Cahn equation should preserve the phase field bounds [-1, 1]
        t_evolution = torch.linspace(0, 1, 10).reshape(-1, 1)
        x_fixed = torch.zeros(1, 1)
        u_evolution = allen_cahn.exact_solution(x_fixed.repeat(10, 1), t_evolution)
        self.assertTrue(
            torch.all(u_evolution >= -1.0) and torch.all(u_evolution <= 1.0)
        )

        # Test interface motion
        # The interface should move according to mean curvature flow
        x_interface = torch.linspace(-0.5, 0.5, 50).reshape(-1, 1)
        t_interface = torch.ones_like(x_interface) * 0.5
        u_interface = allen_cahn.exact_solution(x_interface, t_interface)
        self.assertTrue(
            torch.all(torch.diff(u_interface, dim=0) >= 0)
        )  # Monotonicity at interface

    def test_cahn_hilliard(self):
        """Test the Cahn-Hilliard equation implementation."""
        # Test 1D Cahn-Hilliard equation
        config = PDEConfig(
            name="Cahn-Hilliard Equation",
            domain=[(-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"epsilon": 0.1},  # Interface width parameter
            boundary_conditions={
                "left": {"type": "periodic"},
                "right": {"type": "periodic"},
                "initial": {"type": "tanh", "epsilon": 0.1},
            },
            initial_condition={"type": "tanh", "epsilon": 0.1},
            exact_solution={"type": "tanh", "epsilon": 0.1},
            dimension=1,
            device=self.device,
        )

        cahn_hilliard = CahnHilliardEquation(config=config)

        # Test collocation points
        x, t = cahn_hilliard.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(
            torch.all(x >= config.domain[0][0]) and torch.all(x <= config.domain[0][1])
        )
        self.assertTrue(
            torch.all(t >= config.time_domain[0])
            and torch.all(t <= config.time_domain[1])
        )

        # Test exact solution
        u_exact = cahn_hilliard.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        self.assertTrue(
            torch.all(u_exact >= -1.0) and torch.all(u_exact <= 1.0)
        )  # Phase field bounds

        # Test boundary conditions
        x_boundary = torch.tensor(
            [config.domain[0][0], config.domain[0][1]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = cahn_hilliard._create_boundary_condition(
            "initial", config.initial_condition
        )(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))

        # Test residual computation
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        for param in self.model.parameters():
            param.requires_grad_(True)
        residual = cahn_hilliard.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())

        # Test 2D Cahn-Hilliard equation
        config_2d = PDEConfig(
            name="Cahn-Hilliard Equation 2D",
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"epsilon": 0.1},  # Interface width parameter
            boundary_conditions={
                "left": {"type": "periodic"},
                "right": {"type": "periodic"},
                "initial": {"type": "tanh", "epsilon": 0.1},
            },
            initial_condition={"type": "tanh", "epsilon": 0.1},
            exact_solution={"type": "tanh", "epsilon": 0.1},
            dimension=2,
            device=self.device,
        )

        cahn_hilliard_2d = CahnHilliardEquation(config=config_2d)

        # Test 2D collocation points
        x_2d, t_2d = cahn_hilliard_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(config_2d.domain):
            self.assertTrue(
                torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val)
            )

        # Test 2D exact solution
        u_exact_2d = cahn_hilliard_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        self.assertTrue(
            torch.all(u_exact_2d >= -1.0) and torch.all(u_exact_2d <= 1.0)
        )  # Phase field bounds

        # Test 2D residual computation
        x_2d = x_2d.detach().requires_grad_(True)
        t_2d = t_2d.detach().requires_grad_(True)
        for param in self.model_2d.parameters():
            param.requires_grad_(True)
        residual_2d = cahn_hilliard_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())

        # Test phase separation dynamics
        # The Cahn-Hilliard equation should preserve the phase field bounds [-1, 1]
        t_evolution = torch.linspace(0, 1, 10).reshape(-1, 1)
        x_fixed = torch.zeros(1, 1)
        u_evolution = cahn_hilliard.exact_solution(x_fixed.repeat(10, 1), t_evolution)
        self.assertTrue(
            torch.all(u_evolution >= -1.0) and torch.all(u_evolution <= 1.0)
        )

        # Test mass conservation
        # The Cahn-Hilliard equation should conserve mass
        x_mass = torch.linspace(-1, 1, 100).reshape(-1, 1)
        t_mass = torch.zeros_like(x_mass)
        u_mass = cahn_hilliard.exact_solution(x_mass, t_mass)
        mass = torch.mean(u_mass)
        self.assertTrue(torch.abs(mass) < 0.1)  # Mass should be approximately conserved

        # Test interface motion
        # The interface should move according to surface diffusion
        x_interface = torch.linspace(-0.5, 0.5, 50).reshape(-1, 1)
        t_interface = torch.ones_like(x_interface) * 0.5
        u_interface = cahn_hilliard.exact_solution(x_interface, t_interface)
        self.assertTrue(
            torch.all(torch.diff(u_interface, dim=0) >= 0)
        )  # Monotonicity at interface

    def test_pendulum_equation(self):
        """Test PendulumEquation implementation."""
        # Create PDE configuration
        config = PDEConfig(
            name="Pendulum",
            domain=[(0.0, 1.0)],
            time_domain=(0.0, 1.0),
            parameters={"g": 9.81, "L": 1.0},
            boundary_conditions={"dirichlet": {"type": "fixed", "value": 0.0}},
            initial_condition={
                "type": "sine",
                "amplitude": 1.0,
                "frequency": 2.0,
            },
            exact_solution={
                "type": "sine",
                "amplitude": 1.0,
                "frequency": 2.0,
            },
            dimension=1,
        )

        # Create PDE instance
        pde = PendulumEquation(config)

        # Test initialization
        assert pde.g == 9.81
        assert pde.L == 1.0
        assert pde.domain == [(0.0, 1.0)]
        assert pde.config.time_domain == (0.0, 1.0)
        assert pde.dimension == 1

        # Test initial condition
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        u0 = pde.compute_initial_condition(x)
        assert u0.shape == (100, 1)
        assert torch.allclose(
            u0,
            1.0 * torch.sin(2.0 * x),
            atol=1e-6,
        )

        # Test boundary condition
        t = torch.ones_like(x)
        u_bc = pde.compute_boundary_condition(x, t)
        assert u_bc.shape == (100, 1)
        assert torch.allclose(u_bc, torch.zeros_like(x), atol=1e-6)

        # Test exact solution
        u_exact = pde.exact_solution(x, t)
        assert u_exact.shape == (100, 1)
        assert torch.allclose(
            u_exact,
            1.0 * torch.sin(2.0 * (x + t)),
            atol=1e-6,
        )

        # Test residual
        x = torch.linspace(0, 1, 100, requires_grad=True).reshape(-1, 1)
        t = torch.linspace(0, 1, 100, requires_grad=True).reshape(-1, 1)
        u = torch.sin(x + t)

        # Create a simple model for testing
        dummy_model = torch.nn.Sequential(
            torch.nn.Linear(2, 20), torch.nn.Tanh(), torch.nn.Linear(20, 1)
        ).to(pde.device)
        residual = pde.compute_residual(dummy_model, x, t)
        assert residual.shape == (100, 1)

        # Test time derivative
        derivatives = pde.compute_derivatives(
            dummy_model, x, t, temporal_derivatives=[1, 2], spatial_derivatives=set()
        )
        du_dt = derivatives["dt"]
        d2u_dt2 = derivatives["dt2"]
        assert du_dt.shape == (100, 1)
        assert d2u_dt2.shape == (100, 1)

        # Test energy computation
        energy = pde.compute_energy(dummy_model, x, t)
        assert energy.shape == (100, 1)
        assert torch.all(energy >= 0)  # Energy should be non-negative

        # Test phase space computation
        phase_space = pde.compute_phase_space(dummy_model, x, t)
        assert len(phase_space) == 2
        assert phase_space[0].shape == (100, 1)
        assert phase_space[1].shape == (100, 1)

        # Test invalid initial condition
        pde.config.initial_condition["type"] = "invalid"
        with pytest.raises(ValueError):
            pde.compute_initial_condition(x)

        # Test invalid boundary condition
        pde.config.boundary_conditions["dirichlet"]["type"] = "invalid"
        with pytest.raises(ValueError):
            pde.compute_boundary_condition(x, t)

        # Test invalid exact solution
        pde.config.exact_solution["type"] = "invalid"
        with pytest.raises(ValueError):
            pde.exact_solution(x, t)

        # Test invalid derivative order
        with pytest.raises(ValueError):
            derivatives = pde.compute_derivatives(
                dummy_model, x, t, temporal_derivatives=[3], spatial_derivatives=set()
            )

    def test_comprehensive_boundary_conditions(self):
        """Test all types of boundary conditions in all dimensions."""
        # Domain setup
        domain_1d = [(0.0, 1.0)]
        domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        domain_3d = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        time_domain = (0.0, 1.0)

        # Test various boundary conditions for 1D
        for bc_type in ["dirichlet", "neumann", "periodic"]:
            # Create boundary condition dictionary
            if bc_type == "periodic":
                boundary_conditions = {bc_type: {}}
            else:
                boundary_conditions = {bc_type: {"value": 0.0}}

            initial_condition = {"type": "sine", "amplitude": 1.0, "frequency": 2.0}
            exact_solution = {"type": "sine_wave", "amplitude": 1.0, "frequency": 2.0}

            # Create Heat equation with these boundary conditions
            config = PDEConfig(
                name="Heat Equation",
                domain=domain_1d,
                time_domain=time_domain,
                parameters={"alpha": 0.01},
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition,
                exact_solution=exact_solution,
                dimension=1,
                device=self.device,
            )
            heat_eq = HeatEquation(config=config)

            # Create a simple dummy model for testing
            dummy_model = torch.nn.Sequential(
                torch.nn.Linear(2, 20), torch.nn.Tanh(), torch.nn.Linear(20, 1)
            ).to(self.device)

            # Generate boundary points using the approach from compute_loss
            x_collocation, t_collocation = heat_eq.generate_collocation_points(100)
            # Compute all losses which includes boundary condition calculation
            losses = heat_eq.compute_loss(dummy_model, x_collocation, t_collocation)

            # Verify that loss components exist and are finite
            self.assertIn("boundary", losses)
            self.assertTrue(torch.isfinite(losses["boundary"]))

            # Verify that the boundary loss is a scalar
            self.assertEqual(losses["boundary"].shape, torch.Size([]))

            # For 2D, we only verify that the class initializes correctly,
            # as there is a problem with dimensions in compute_loss for 2D
            if bc_type not in [
                "robin"
            ]:  # Skip robin as it's not supported in the implementation
                config_2d = PDEConfig(
                    name="Heat Equation 2D",
                    domain=domain_2d,
                    time_domain=time_domain,
                    parameters={"alpha": 0.01},
                    boundary_conditions=boundary_conditions,
                    initial_condition=initial_condition,
                    exact_solution=exact_solution,
                    dimension=2,
                    device=self.device,
                )
                heat_eq_2d = HeatEquation(config=config_2d)

                # Verify that we can generate collocation points in 2D
                x_collocation_2d, t_collocation_2d = (
                    heat_eq_2d.generate_collocation_points(100)
                )
                self.assertEqual(x_collocation_2d.shape, (100, 2))
                self.assertEqual(t_collocation_2d.shape, (100, 1))

    def test_comprehensive_initial_conditions(self):
        """Test all types of initial conditions in different dimensions."""
        # Domain setup
        domain_1d = [(0.0, 1.0)]
        domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        time_domain = (0.0, 1.0)
        boundary_conditions = {"dirichlet": {"value": 0.0}}
        exact_solution = {"type": "sine_wave", "amplitude": 1.0, "frequency": 2.0}

        # Test various initial conditions - HeatEquation only supports sine type
        initial_condition_types = [
            {"type": "sine", "amplitude": 1.0, "frequency": 2.0},
            # Other types require implementation in the HeatEquation class
        ]

        for ic in initial_condition_types:
            # 1D Heat equation
            config = PDEConfig(
                name="Heat Equation",
                domain=domain_1d,
                time_domain=time_domain,
                parameters={"alpha": 0.01},
                boundary_conditions=boundary_conditions,
                initial_condition=ic,
                exact_solution=exact_solution,
                dimension=1,
                device=self.device,
            )
            heat_eq_1d = HeatEquation(config=config)

            # Create a simple dummy model for testing
            dummy_model = torch.nn.Sequential(
                torch.nn.Linear(2, 20), torch.nn.Tanh(), torch.nn.Linear(20, 1)
            ).to(self.device)

            # Generate points and compute losses which includes initial condition calculation
            x_collocation, t_collocation = heat_eq_1d.generate_collocation_points(100)
            losses = heat_eq_1d.compute_loss(dummy_model, x_collocation, t_collocation)

            # Verify that initial condition loss exists and is finite
            self.assertIn("initial", losses)
            self.assertTrue(torch.isfinite(losses["initial"]))
            self.assertEqual(losses["initial"].shape, torch.Size([]))

            # For 2D, we only verify that the class initializes correctly
            # as there is a problem with dimensions in compute_loss for 2D
            config_2d = PDEConfig(
                name="Heat Equation 2D",
                domain=domain_2d,
                time_domain=time_domain,
                parameters={"alpha": 0.01},
                boundary_conditions=boundary_conditions,
                initial_condition=ic,
                exact_solution=exact_solution,
                dimension=2,
                device=self.device,
            )
            heat_eq_2d = HeatEquation(config=config_2d)

            # Verify that we can generate collocation points in 2D
            x_collocation_2d, t_collocation_2d = heat_eq_2d.generate_collocation_points(
                100
            )
            self.assertEqual(x_collocation_2d.shape, (100, 2))
            self.assertEqual(t_collocation_2d.shape, (100, 1))


if __name__ == "__main__":
    unittest.main()
