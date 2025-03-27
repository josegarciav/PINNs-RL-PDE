import unittest
import torch
import numpy as np
from scipy.stats import qmc
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.kdv_equation import KdVEquation
from src.pdes.pde_base import PDEBase
from src.rl_agent import RLAgent, CollocationRLAgent

class TestPDESampling(unittest.TestCase):
    """Test PDE sampling strategies, especially for different dimensions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
        # Define common parameters for PDEs
        self.domain_1d = [(0.0, 1.0)]
        self.domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        self.time_domain = (0.0, 1.0)
        self.boundary_conditions = {
            "dirichlet": {"value": 0.0},
            "neumann": {"value": 0.0},
            "periodic": {},
        }
        self.initial_condition = {"type": "sine", "amplitude": 1.0, "frequency": 2.0}
        self.exact_solution = {"type": "sine_wave", "amplitude": 1.0, "frequency": 2.0}
        
        # Create 1D and 2D PDEs for testing
        self.heat_eq_1d = HeatEquation(
            alpha=0.01,
            domain=self.domain_1d,
            time_domain=self.time_domain,
            boundary_conditions=self.boundary_conditions,
            initial_condition=self.initial_condition,
            exact_solution=self.exact_solution,
            dimension=1,
            device=self.device,
        )
        
        self.heat_eq_2d = HeatEquation(
            alpha=0.01,
            domain=self.domain_2d,
            time_domain=self.time_domain,
            boundary_conditions=self.boundary_conditions,
            initial_condition=self.initial_condition,
            exact_solution=self.exact_solution,
            dimension=2,
            device=self.device,
        )
        
        self.wave_eq_1d = WaveEquation(
            c=1.0,
            domain=self.domain_1d,
            time_domain=self.time_domain,
            boundary_conditions=self.boundary_conditions,
            initial_condition=self.initial_condition,
            exact_solution=self.exact_solution,
            dimension=1,
            device=self.device,
        )
        
        self.wave_eq_2d = WaveEquation(
            c=1.0,
            domain=self.domain_2d,
            time_domain=self.time_domain,
            boundary_conditions=self.boundary_conditions,
            initial_condition=self.initial_condition,
            exact_solution=self.exact_solution,
            dimension=2,
            device=self.device,
        )
        
        # Create an RL agent for adaptive sampling
        self.rl_agent = RLAgent(
            state_dim=3,  # For 2D spatial + time
            action_dim=1,
            hidden_dim=32,
            device=self.device
        )
        
        self.coll_rl_agent = CollocationRLAgent(
            state_dim=3,
            action_dim=1,
            hidden_dim=32,
            device=self.device
        )

    def test_uniform_sampling_1d(self):
        """Test uniform sampling in 1D."""
        num_points = 100
        
        # Heat equation
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= self.domain_1d[0][0]) and torch.all(x <= self.domain_1d[0][1]))
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))
        
        # Wave equation
        x, t = self.wave_eq_1d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= self.domain_1d[0][0]) and torch.all(x <= self.domain_1d[0][1]))
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))

    def test_uniform_sampling_2d(self):
        """Test uniform sampling in 2D."""
        num_points = 100
        
        # Heat equation
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        
        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val))
        
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))
        
        # Wave equation
        x, t = self.wave_eq_2d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        
        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val))
        
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))

    def test_latin_hypercube_sampling_1d(self):
        """Test Latin Hypercube sampling in 1D."""
        num_points = 100
        
        # Heat equation
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="latin_hypercube")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= self.domain_1d[0][0]) and torch.all(x <= self.domain_1d[0][1]))
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))
        
        # Wave equation
        x, t = self.wave_eq_1d.generate_collocation_points(num_points, strategy="latin_hypercube")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= self.domain_1d[0][0]) and torch.all(x <= self.domain_1d[0][1]))
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))

    def test_latin_hypercube_sampling_2d(self):
        """Test Latin Hypercube sampling in 2D."""
        num_points = 100
        
        # Heat equation
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="latin_hypercube")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        
        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val))
        
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))
        
        # Wave equation
        x, t = self.wave_eq_2d.generate_collocation_points(num_points, strategy="latin_hypercube")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        
        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val))
        
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))

    def test_adaptive_sampling_with_rl_agent(self):
        """Test adaptive sampling with RL agent."""
        num_points = 100
        
        # Assign RL agents
        self.heat_eq_1d.rl_agent = self.rl_agent
        self.heat_eq_2d.rl_agent = self.rl_agent
        
        # Test 1D adaptive sampling
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= self.domain_1d[0][0]) and torch.all(x <= self.domain_1d[0][1]))
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))
        
        # Test 2D adaptive sampling
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        
        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val))
        
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))

    def test_adaptive_sampling_fallback(self):
        """Test that adaptive sampling falls back to uniform when no RL agent is provided."""
        num_points = 100
        
        # Ensure no RL agent is set
        self.heat_eq_1d.rl_agent = None
        
        # Should fall back to uniform
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= self.domain_1d[0][0]) and torch.all(x <= self.domain_1d[0][1]))
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))

    def test_different_num_points(self):
        """Test with different numbers of points."""
        num_points_list = [50, 100, 200, 500]  # Omit very small values that might cause rounding issues
        
        for num_points in num_points_list:
            # 1D uniform
            x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="uniform")
            # Allow for minor differences in point count due to implementation details
            self.assertTrue(abs(x.shape[0] - num_points) <= 1, f"Expected around {num_points} points, got {x.shape[0]}")
            self.assertEqual(x.shape[1], 1)
            self.assertEqual(t.shape[1], 1)
            
            # 2D uniform
            x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="uniform")
            self.assertTrue(abs(x.shape[0] - num_points) <= 1, f"Expected around {num_points} points, got {x.shape[0]}")
            self.assertEqual(x.shape[1], 2)
            self.assertEqual(t.shape[1], 1)
            
            # 1D latin hypercube
            x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="latin_hypercube")
            self.assertEqual(x.shape, (num_points, 1))  # Latin hypercube should be exact
            self.assertEqual(t.shape, (num_points, 1))
            
            # 2D latin hypercube
            x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="latin_hypercube")
            self.assertEqual(x.shape, (num_points, 2))  # Latin hypercube should be exact
            self.assertEqual(t.shape, (num_points, 1))

    def test_small_number_points(self):
        """Test with very small number of points which might be affected by rounding."""
        num_points = 10
        
        # For such small numbers, we should check that the shape is close to what we expect
        # but not necessarily exact
        
        # 1D uniform
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="uniform")
        self.assertTrue(abs(x.shape[0] - num_points) <= 1, f"Expected around {num_points} points, got {x.shape[0]}")
        self.assertEqual(x.shape[1], 1)
        self.assertEqual(t.shape[1], 1)
        
        # 2D uniform
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="uniform")
        self.assertTrue(abs(x.shape[0] - num_points) <= 1, f"Expected around {num_points} points, got {x.shape[0]}")
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(t.shape[1], 1)

    def test_collocation_history(self):
        """Test that collocation history is maintained correctly."""
        num_points = 100
        
        # Assign RL agent
        self.heat_eq_1d.rl_agent = self.rl_agent
        
        # Generate points with adaptive strategy to create history
        for _ in range(3):
            x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")
        
        # Check that history has been recorded
        self.assertEqual(len(self.heat_eq_1d.collocation_history), 3)
        
        # Each history entry should contain num_points data points
        for h in self.heat_eq_1d.collocation_history:
            self.assertEqual(h.shape[0], num_points)
            # For 1D problems, history has shape (num_points, 2) - x and t
            self.assertEqual(h.shape[1], 2)

    def test_invalid_strategy(self):
        """Test that an invalid strategy raises an error."""
        num_points = 100
        
        with self.assertRaises(ValueError):
            self.heat_eq_1d.generate_collocation_points(num_points, strategy="invalid_strategy")
            
if __name__ == "__main__":
    unittest.main() 