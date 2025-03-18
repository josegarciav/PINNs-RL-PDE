import torch
import unittest
import numpy as np
from src.pinn import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.black_scholes import BlackScholesEquation
from src.pdes.kdv_equation import KdVEquation
from src.pdes.allen_cahn import AllenCahnEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.cahn_hilliard import CahnHilliardEquation
from src.pdes.convection_equation import ConvectionEquation
from src.rl_agent import RLAgent
from src.pdes.pde_base import PDEConfig


class TestPDEs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.x = torch.linspace(0, 1, 10).unsqueeze(1).to(self.device)
        self.t = torch.linspace(0, 1, 10).unsqueeze(1).to(self.device)
        self.xt = torch.cat([self.x, self.t], dim=1).requires_grad_(True)

        # Initialize 1D PDEs with parameters
        self.heat_eq = HeatEquation(
            alpha=0.01,
            domain=(0.0, 1.0),
            time_domain=(0.0, 1.0),
            boundary_conditions={
                'dirichlet': {'value': 0.0},
                'neumann': {'value': 0.0},
                'periodic': {}
            },
            initial_condition={
                'type': 'sine',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            exact_solution={
                'type': 'sine_wave',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            dimension=1,
            device=self.device
        )
        
        # Initialize 2D Heat Equation
        self.heat_eq_2d = HeatEquation(
            alpha=0.01,
            domain=[(0.0, 1.0), (0.0, 1.0)],  # 2D domain
            time_domain=(0.0, 1.0),
            boundary_conditions={
                'dirichlet': {'value': 0.0},
                'neumann': {'value': 0.0},
                'periodic': {}
            },
            initial_condition={
                'type': 'sine',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            exact_solution={
                'type': 'sine_wave',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            dimension=2,
            device=self.device
        )
        
        # Initialize Wave Equation with parameters
        self.wave_eq = WaveEquation(
            c=1.0,
            domain=(0.0, 1.0),
            time_domain=(0.0, 1.0),
            boundary_conditions={
                'dirichlet': {'value': 0.0},
                'neumann': {'value': 0.0},
                'periodic': {}
            },
            initial_condition={
                'type': 'sine',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            exact_solution={
                'type': 'sine_wave',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            dimension=1,
            device=self.device
        )
        
        # Initialize Black-Scholes Equation with parameters
        self.bs_eq = BlackScholesEquation(
            sigma=0.2,
            r=0.05,
            domain=(0.0, 1.0),
            time_domain=(0.0, 1.0),
            boundary_conditions={
                'dirichlet': {'value': 0.0},
                'neumann': {'value': 0.0},
                'periodic': {}
            },
            initial_condition={
                'type': 'call',
                'strike': 1.0
            },
            exact_solution={
                'type': 'call',
                'strike': 1.0
            },
            dimension=1,
            device=self.device
        )

        # Initialize PINN model with new architecture
        self.model = PINNModel(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            num_layers=4,
            activation="tanh",
            fourier_features=True,
            fourier_scale=10.0,
            dropout=0.1,
            layer_norm=True,
            device=self.device
        )

        # Initialize 2D PINN model
        self.model_2d = PINNModel(
            input_dim=3,  # (x, y, t)
            hidden_dim=64,
            output_dim=1,
            num_layers=4,
            activation="tanh",
            fourier_features=True,
            fourier_scale=10.0,
            dropout=0.1,
            layer_norm=True,
            device=self.device
        )

        # Initialize RL agent
        self.rl_agent = RLAgent(
            state_dim=2,
            action_dim=1,
            hidden_dim=64,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            device=self.device
        )

    def test_heat_equation_residual(self):
        residual = self.heat_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, (10, 1))
        self.assertTrue(torch.isfinite(residual).all())

    def test_wave_equation_residual(self):
        residual = self.wave_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, (10, 1))
        self.assertTrue(torch.isfinite(residual).all())

    def test_black_scholes_residual(self):
        residual = self.bs_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, (10, 1))
        self.assertTrue(torch.isfinite(residual).all())

    def test_boundary_conditions(self):
        # Test boundary conditions
        bc_dirichlet = self.heat_eq._create_boundary_condition("dirichlet", {"value": 0.0})
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
        x, t = self.heat_eq.generate_collocation_points(num_points, strategy='adaptive')
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))
        
        # Test 2D Heat Equation
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy='adaptive')
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x[:, 0] >= 0) and torch.all(x[:, 0] <= 1))
        self.assertTrue(torch.all(x[:, 1] >= 0) and torch.all(x[:, 1] <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))
        
        # Test Wave Equation
        x, t = self.wave_eq.generate_collocation_points(num_points, strategy='adaptive')
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))
        
        # Test Black-Scholes Equation
        x, t = self.bs_eq.generate_collocation_points(num_points, strategy='adaptive')
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))
        
        # Test RL agent state and action
        state = torch.randn(1, 2).to(self.device)  # Random (x, t) state
        action = self.heat_eq.rl_agent.select_action(state)
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
        self.assertEqual(config.domain, (0.0, 1.0))
        self.assertEqual(config.time_domain, (0.0, 1.0))
        self.assertEqual(config.initial_condition["type"], "sine")
        self.assertEqual(config.parameters["alpha"], 0.01)

    def test_rl_agent_integration(self):
        # Test RL agent state and action
        state = torch.randn(1, 2).to(self.device)  # Random (x, t) state
        action = self.rl_agent.select_action(state)
        
        self.assertEqual(action.shape, (1, 1))  # Single action value
        self.assertTrue(0 <= action.item() <= 1)  # Action should be probability

    def test_2d_heat_equation(self):
        """Test 2D heat equation functionality."""
        # Generate 2D collocation points
        num_points = 100
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy='latin_hypercube')
        
        # Check shapes
        self.assertEqual(x.shape, (num_points, 2))  # 2D spatial coordinates
        self.assertEqual(t.shape, (num_points, 1))  # Time coordinate
        
        # Check domain bounds
        self.assertTrue(torch.all(x[:, 0] >= 0) and torch.all(x[:, 0] <= 1))
        self.assertTrue(torch.all(x[:, 1] >= 0) and torch.all(x[:, 1] <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))
        
        # Test residual computation
        residual = self.heat_eq_2d.compute_residual(self.model_2d, x, t)
        self.assertEqual(residual.shape, (num_points, 1))
        self.assertTrue(torch.isfinite(residual).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=self.device)
        t_boundary = torch.zeros(2, 1, device=self.device)
        
        for bc_type, bc_func in self.heat_eq_2d.boundary_conditions.items():
            bc_value = bc_func(x_boundary, t_boundary)
            self.assertEqual(bc_value.shape, (2, 1))
            self.assertTrue(torch.isfinite(bc_value).all())

    def test_2d_wave_equation(self):
        """Test 2D wave equation functionality."""
        # Initialize 2D Wave Equation
        wave_eq_2d = WaveEquation(
            c=1.0,
            domain=[(0.0, 1.0), (0.0, 1.0)],  # 2D domain
            time_domain=(0.0, 1.0),
            boundary_conditions={
                'dirichlet': {'value': 0.0},
                'neumann': {'value': 0.0},
                'periodic': {}
            },
            initial_condition={
                'type': 'sine',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            exact_solution={
                'type': 'sine_wave',
                'amplitude': 1.0,
                'frequency': 2.0
            },
            dimension=2,
            device=self.device
        )
        
        # Generate 2D collocation points
        num_points = 100
        x, t = wave_eq_2d.generate_collocation_points(num_points, strategy='latin_hypercube')
        
        # Check shapes
        self.assertEqual(x.shape, (num_points, 2))  # 2D spatial coordinates
        self.assertEqual(t.shape, (num_points, 1))  # Time coordinate
        
        # Check domain bounds
        self.assertTrue(torch.all(x[:, 0] >= 0) and torch.all(x[:, 0] <= 1))
        self.assertTrue(torch.all(x[:, 1] >= 0) and torch.all(x[:, 1] <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))
        
        # Test residual computation
        residual = wave_eq_2d.compute_residual(self.model_2d, x, t)
        self.assertEqual(residual.shape, (num_points, 1))
        self.assertTrue(torch.isfinite(residual).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=self.device)
        t_boundary = torch.zeros(2, 1, device=self.device)
        
        for bc_type, bc_func in wave_eq_2d.boundary_conditions.items():
            bc_value = bc_func(x_boundary, t_boundary)
            self.assertEqual(bc_value.shape, (2, 1))
            self.assertTrue(torch.isfinite(bc_value).all())

    def test_2d_black_scholes(self):
        """Test 2D Black-Scholes equation functionality."""
        # Initialize 2D Black-Scholes Equation
        bs_eq_2d = BlackScholesEquation(
            sigma=0.2,
            r=0.05,
            domain=[(0.0, 1.0), (0.0, 1.0)],  # 2D domain
            time_domain=(0.0, 1.0),
            boundary_conditions={
                'dirichlet': {'value': 0.0},
                'neumann': {'value': 0.0},
                'periodic': {}
            },
            initial_condition={
                'type': 'call',
                'strike': 1.0
            },
            exact_solution={
                'type': 'call',
                'strike': 1.0
            },
            dimension=2,
            device=self.device
        )
        
        # Generate 2D collocation points
        num_points = 100
        x, t = bs_eq_2d.generate_collocation_points(num_points, strategy='latin_hypercube')
        
        # Check shapes
        self.assertEqual(x.shape, (num_points, 2))  # 2D spatial coordinates
        self.assertEqual(t.shape, (num_points, 1))  # Time coordinate
        
        # Check domain bounds
        self.assertTrue(torch.all(x[:, 0] >= 0) and torch.all(x[:, 0] <= 1))
        self.assertTrue(torch.all(x[:, 1] >= 0) and torch.all(x[:, 1] <= 1))
        self.assertTrue(torch.all(t >= 0) and torch.all(t <= 1))
        
        # Test residual computation
        residual = bs_eq_2d.compute_residual(self.model_2d, x, t)
        self.assertEqual(residual.shape, (num_points, 1))
        self.assertTrue(torch.isfinite(residual).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=self.device)
        t_boundary = torch.zeros(2, 1, device=self.device)
        
        for bc_type, bc_func in bs_eq_2d.boundary_conditions.items():
            bc_value = bc_func(x_boundary, t_boundary)
            self.assertEqual(bc_value.shape, (2, 1))
            self.assertTrue(torch.isfinite(bc_value).all())

    def test_kdv_equation(self):
        """Test the Korteweg-de Vries (KdV) equation implementation."""
        # Test 1D KdV equation
        domain = (-10, 10)
        time_domain = (0, 1)
        boundary_conditions = {
            'left': {'type': 'periodic'},
            'right': {'type': 'periodic'},
            'initial': {'type': 'soliton', 'speed': 1.0}
        }
        initial_condition = {'type': 'soliton', 'speed': 1.0}
        exact_solution = {'type': 'soliton', 'speed': 1.0}
        
        kdv = KdVEquation(
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1
        )
        
        # Test collocation points
        x, t = kdv.generate_collocation_points(100)
        assert x.shape == (100, 1)
        assert t.shape == (100, 1)
        assert torch.all(x >= domain[0]) and torch.all(x <= domain[1])
        assert torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1])
        
        # Test exact solution
        u_exact = kdv.exact_solution(x, t)
        assert u_exact.shape == (100, 1)
        assert torch.all(u_exact >= 0)  # Soliton is always positive
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = kdv._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        assert u_boundary.shape == (2, 1)
        
        # Test 2D KdV equation
        domain_2d = [(-10, 10), (-10, 10)]
        kdv_2d = KdVEquation(
            domain=domain_2d,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2
        )
        
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
        # ... existing code ...

    def test_convection_equation(self):
        """Test the Convection equation implementation."""
        # Test 1D Convection equation
        domain = (-1, 1)
        time_domain = (0, 1)
        boundary_conditions = {
            'left': {'type': 'periodic'},
            'right': {'type': 'periodic'},
            'initial': {'type': 'gaussian', 'amplitude': 1.0, 'width': 0.1}
        }
        initial_condition = {'type': 'gaussian', 'amplitude': 1.0, 'width': 0.1}
        exact_solution = {'type': 'gaussian', 'amplitude': 1.0, 'width': 0.1}
        
        convection = ConvectionEquation(
            c=1.0,  # Wave speed
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = convection.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = convection.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = convection._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test 2D Convection equation
        domain_2d = [(-1, 1), (-1, 1)]
        convection_2d = ConvectionEquation(
            c=1.0,
            domain=domain_2d,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device
        )
        
        # Test 2D collocation points
        x_2d, t_2d = convection_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val))
        
        # Test 2D exact solution
        u_exact_2d = convection_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())

if __name__ == "__main__":
    unittest.main()
