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


class TestPDEs(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Basic test points
        self.x = torch.linspace(0, 1, 100).reshape(-1, 1).to(self.device)
        self.t = torch.linspace(0, 1, 100).reshape(-1, 1).to(self.device)
        self.xt = torch.cat([self.x, self.t], dim=1)
        
        # Initialize PINN models
        self.model = PINNModel(
            input_dim=2,
            hidden_dim=32,
            output_dim=1,
            num_layers=2,
            device=self.device
        )

        self.model_2d = PINNModel(
            input_dim=3,
            hidden_dim=32,
            output_dim=1,
            num_layers=2,
            device=self.device
        )
        
        # Initialize RL agent for adaptive sampling tests
        self.rl_agent = RLAgent(
            state_dim=2,
            action_dim=1,
            hidden_dim=32,
            device=self.device
        )
        
        # Initialize heat equation for common tests
        domain = (0.0, 1.0)
        time_domain = (0.0, 1.0)
        boundary_conditions = {
            'dirichlet': {'value': 0.0},
            'neumann': {'value': 0.0},
            'periodic': {}
        }
        initial_condition = {'type': 'sine', 'amplitude': 1.0, 'frequency': 2.0}
        exact_solution = {'type': 'sine_wave', 'amplitude': 1.0, 'frequency': 2.0}
        
        self.heat_eq = HeatEquation(
            alpha=0.01,  # Thermal diffusivity
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )

        # Initialize wave equation for common tests
        self.wave_eq = WaveEquation(
            c=1.0,  # Wave speed
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )

    def test_heat_equation(self):
        """Test the Heat equation implementation."""
        # Test 1D Heat equation
        domain = (0.0, 1.0)
        time_domain = (0.0, 1.0)
        boundary_conditions = {
            'dirichlet': {'value': 0.0},
            'neumann': {'value': 0.0},
            'periodic': {}
        }
        initial_condition = {'type': 'sine', 'amplitude': 1.0, 'frequency': 2.0}
        exact_solution = {'type': 'sine_wave', 'amplitude': 1.0, 'frequency': 2.0}
        
        heat_eq = HeatEquation(
            alpha=0.01,  # Thermal diffusivity
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = heat_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = heat_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = heat_eq._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test residual computation
        residual = heat_eq.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())
        
        # Test 2D Heat equation
        domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        heat_eq_2d = HeatEquation(
            alpha=0.01,
            domain=domain_2d,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device
        )
        
        # Test 2D collocation points
        x_2d, t_2d = heat_eq_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val))
        
        # Test 2D exact solution
        u_exact_2d = heat_eq_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        
        # Test 2D residual computation
        residual_2d = heat_eq_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())

    def test_wave_equation(self):
        """Test the Wave equation implementation."""
        # Test 1D Wave equation
        domain = (0.0, 1.0)
        time_domain = (0.0, 1.0)
        boundary_conditions = {
            'dirichlet': {'value': 0.0},
            'neumann': {'value': 0.0},
            'periodic': {}
        }
        initial_condition = {'type': 'sine', 'amplitude': 1.0, 'frequency': 2.0}
        exact_solution = {'type': 'sine_wave', 'amplitude': 1.0, 'frequency': 2.0}
        
        wave_eq = WaveEquation(
            c=1.0,
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = wave_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = wave_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = wave_eq._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test residual computation
        residual = wave_eq.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())
        
        # Test 2D Wave equation
        domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        wave_eq_2d = WaveEquation(
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
        x_2d, t_2d = wave_eq_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val))
        
        # Test 2D exact solution
        u_exact_2d = wave_eq_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        
        # Test 2D residual computation
        residual_2d = wave_eq_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())

    def test_black_scholes(self):
        """Test the Black-Scholes equation implementation."""
        # Test 1D Black-Scholes equation
        domain = (0.0, 200.0)  # Stock price domain
        time_domain = (0.0, 1.0)  # Time to maturity
        strike = 100.0
        boundary_conditions = {
            'dirichlet': {'value': 0.0},
            'neumann': {'value': 1.0},  # Delta at upper boundary
            'initial': {'type': 'call_option', 'strike_price': strike}
        }
        initial_condition = {'type': 'call_option', 'strike_price': strike}
        exact_solution = {'type': 'call_option', 'strike_price': strike}
        
        bs_eq = BlackScholesEquation(
            sigma=0.2,  # Volatility
            r=0.05,    # Risk-free rate
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = bs_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = bs_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        self.assertTrue(torch.all(u_exact >= 0))  # Option price is non-negative
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = bs_eq._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test 2D Black-Scholes equation (basket option)
        domain_2d = [(0.0, 200.0), (0.0, 200.0)]  # Two stock prices
        bs_eq_2d = BlackScholesEquation(
            sigma=0.2,
            r=0.05,
            domain=domain_2d,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device
        )

        # Test 2D collocation points
        x_2d, t_2d = bs_eq_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val))
        
        # Test 2D exact solution
        u_exact_2d = bs_eq_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        self.assertTrue(torch.all(u_exact_2d >= 0))  # Option price is non-negative
        
        # Test residual computation
        residual = bs_eq_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual.shape, (100, 1))
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
        self.assertEqual(exact_sol.shape, (100, 1))
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
        domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        heat_eq_2d = HeatEquation(
            alpha=0.01,
            domain=domain_2d,
            time_domain=(0.0, 1.0),
            boundary_conditions=self.heat_eq.config.boundary_conditions,
            initial_condition=self.heat_eq.config.initial_condition,
            exact_solution=self.heat_eq.config.exact_solution,
            dimension=2,
            device=self.device
        )
        x, t = heat_eq_2d.generate_collocation_points(num_points, strategy='adaptive')
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
        # Test 1D Burgers' equation
        domain = (-1, 1)
        time_domain = (0, 1)
        boundary_conditions = {
            'left': {'type': 'dirichlet', 'value': 0.0},
            'right': {'type': 'dirichlet', 'value': 0.0},
            'initial': {'type': 'tanh', 'epsilon': 0.1}
        }
        initial_condition = {'type': 'tanh', 'epsilon': 0.1}
        exact_solution = {'type': 'tanh', 'epsilon': 0.1}
        
        burgers = BurgersEquation(
            nu=0.01,  # Kinematic viscosity
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = burgers.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = burgers.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = burgers._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test 2D Burgers' equation
        domain_2d = [(-1, 1), (-1, 1)]
        burgers_2d = BurgersEquation(
            nu=0.01,
            domain=domain_2d,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device
        )
        
        # Test 2D collocation points
        x_2d, t_2d = burgers_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val))
        
        # Test 2D exact solution
        u_exact_2d = burgers_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())

    def test_convection(self):
        """Test convection equation functionality."""
        # 1D Convection equation
        domain = (-1, 1)
        time_domain = (0, 1)
        velocity = 1.0  # Changed from c to velocity
        
        # Define boundary conditions
        boundary_conditions = {
            'left': {'type': 'dirichlet', 'value': 0.0},
            'right': {'type': 'dirichlet', 'value': 0.0},
            'initial': {'type': 'sine', 'amplitude': 1.0, 'frequency': 2.0}
        }
        
        # Define initial condition
        initial_condition = {
            'type': 'sine',
            'amplitude': 1.0,
            'frequency': 2.0
        }
        
        # Define exact solution
        exact_solution = {
            'type': 'sine',
            'amplitude': 1.0,
            'frequency': 2.0
        }
        
        # Create convection equation instance
        convection_eq = ConvectionEquation(
            velocity=velocity,  # Changed from c to velocity
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = convection_eq.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = convection_eq.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = convection_eq._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test 2D Convection equation
        domain_2d = [(-1, 1), (-1, 1)]
        convection_2d = ConvectionEquation(
            velocity=velocity,
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

    def test_allen_cahn(self):
        """Test the Allen-Cahn equation implementation."""
        # Test 1D Allen-Cahn equation
        domain = (-1, 1)
        time_domain = (0, 1)
        epsilon = 0.1  # Interface width parameter
        boundary_conditions = {
            'left': {'type': 'dirichlet', 'value': -1.0},
            'right': {'type': 'dirichlet', 'value': 1.0},
            'initial': {'type': 'tanh', 'epsilon': epsilon}
        }
        initial_condition = {'type': 'tanh', 'epsilon': epsilon}
        exact_solution = {'type': 'tanh', 'epsilon': epsilon}
        
        allen_cahn = AllenCahnEquation(
            epsilon=epsilon,  # Interface width parameter
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = allen_cahn.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = allen_cahn.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        self.assertTrue(torch.all(u_exact >= -1.0) and torch.all(u_exact <= 1.0))  # Phase field bounds
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = allen_cahn._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test residual computation
        residual = allen_cahn.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())
        
        # Test 2D Allen-Cahn equation
        domain_2d = [(-1, 1), (-1, 1)]
        allen_cahn_2d = AllenCahnEquation(
            epsilon=epsilon,
            domain=domain_2d,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device
        )
        
        # Test 2D collocation points
        x_2d, t_2d = allen_cahn_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val))
        
        # Test 2D exact solution
        u_exact_2d = allen_cahn_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        self.assertTrue(torch.all(u_exact_2d >= -1.0) and torch.all(u_exact_2d <= 1.0))  # Phase field bounds
        
        # Test 2D residual computation
        residual_2d = allen_cahn_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())
        
        # Test phase separation dynamics
        # The Allen-Cahn equation should preserve the phase field bounds [-1, 1]
        t_evolution = torch.linspace(0, 1, 10).reshape(-1, 1)
        x_fixed = torch.zeros(1, 1)
        u_evolution = allen_cahn.exact_solution(x_fixed.repeat(10, 1), t_evolution)
        self.assertTrue(torch.all(u_evolution >= -1.0) and torch.all(u_evolution <= 1.0))
        
        # Test interface motion
        # The interface should move according to mean curvature flow
        x_interface = torch.linspace(-0.5, 0.5, 50).reshape(-1, 1)
        t_interface = torch.ones_like(x_interface) * 0.5
        u_interface = allen_cahn.exact_solution(x_interface, t_interface)
        self.assertTrue(torch.all(torch.diff(u_interface, dim=0) >= 0))  # Monotonicity at interface

    def test_cahn_hilliard(self):
        """Test the Cahn-Hilliard equation implementation."""
        # Test 1D Cahn-Hilliard equation
        domain = (-1, 1)
        time_domain = (0, 1)
        epsilon = 0.1  # Interface width parameter
        boundary_conditions = {
            'left': {'type': 'periodic'},
            'right': {'type': 'periodic'},
            'initial': {'type': 'tanh', 'epsilon': epsilon}
        }
        initial_condition = {'type': 'tanh', 'epsilon': epsilon}
        exact_solution = {'type': 'tanh', 'epsilon': epsilon}
        
        cahn_hilliard = CahnHilliardEquation(
            epsilon=epsilon,  # Interface width parameter
            domain=domain,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=1,
            device=self.device
        )
        
        # Test collocation points
        x, t = cahn_hilliard.generate_collocation_points(100)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(t.shape, (100, 1))
        self.assertTrue(torch.all(x >= domain[0]) and torch.all(x <= domain[1]))
        self.assertTrue(torch.all(t >= time_domain[0]) and torch.all(t <= time_domain[1]))
        
        # Test exact solution
        u_exact = cahn_hilliard.exact_solution(x, t)
        self.assertEqual(u_exact.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact).all())
        self.assertTrue(torch.all(u_exact >= -1.0) and torch.all(u_exact <= 1.0))  # Phase field bounds
        
        # Test boundary conditions
        x_boundary = torch.tensor([domain[0], domain[1]], dtype=torch.float32).reshape(-1, 1)
        t_boundary = torch.zeros_like(x_boundary)
        u_boundary = cahn_hilliard._create_boundary_condition('initial', initial_condition)(x_boundary, t_boundary)
        self.assertEqual(u_boundary.shape, (2, 1))
        
        # Test residual computation
        residual = cahn_hilliard.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual).all())
        
        # Test 2D Cahn-Hilliard equation
        domain_2d = [(-1, 1), (-1, 1)]
        cahn_hilliard_2d = CahnHilliardEquation(
            epsilon=epsilon,
            domain=domain_2d,
            time_domain=time_domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=2,
            device=self.device
        )
        
        # Test 2D collocation points
        x_2d, t_2d = cahn_hilliard_2d.generate_collocation_points(100)
        self.assertEqual(x_2d.shape, (100, 2))
        self.assertEqual(t_2d.shape, (100, 1))
        for i, (min_val, max_val) in enumerate(domain_2d):
            self.assertTrue(torch.all(x_2d[:, i] >= min_val) and torch.all(x_2d[:, i] <= max_val))
        
        # Test 2D exact solution
        u_exact_2d = cahn_hilliard_2d.exact_solution(x_2d, t_2d)
        self.assertEqual(u_exact_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(u_exact_2d).all())
        self.assertTrue(torch.all(u_exact_2d >= -1.0) and torch.all(u_exact_2d <= 1.0))  # Phase field bounds
        
        # Test 2D residual computation
        residual_2d = cahn_hilliard_2d.compute_residual(self.model_2d, x_2d, t_2d)
        self.assertEqual(residual_2d.shape, (100, 1))
        self.assertTrue(torch.isfinite(residual_2d).all())
        
        # Test phase separation dynamics
        # The Cahn-Hilliard equation should preserve the phase field bounds [-1, 1]
        t_evolution = torch.linspace(0, 1, 10).reshape(-1, 1)
        x_fixed = torch.zeros(1, 1)
        u_evolution = cahn_hilliard.exact_solution(x_fixed.repeat(10, 1), t_evolution)
        self.assertTrue(torch.all(u_evolution >= -1.0) and torch.all(u_evolution <= 1.0))
        
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
        self.assertTrue(torch.all(torch.diff(u_interface, dim=0) >= 0))  # Monotonicity at interface

if __name__ == "__main__":
    unittest.main()
