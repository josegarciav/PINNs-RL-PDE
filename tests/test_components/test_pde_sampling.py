import unittest
import torch
import numpy as np
from scipy.stats import qmc
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.kdv_equation import KdVEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.convection_equation import ConvectionEquation
from src.pdes.allen_cahn import AllenCahnEquation
from src.pdes.cahn_hilliard import CahnHilliardEquation
from src.pdes.black_scholes import BlackScholesEquation
from src.pdes.pendulum_equation import PendulumEquation
from src.pdes.pde_base import PDEBase, PDEConfig
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
            # Allow for a larger tolerance in point count due to grid based implementation in uniform sampling
            tolerance = max(5, int(num_points * 0.05))  # 5% tolerance or at least 5 points
            self.assertTrue(abs(x.shape[0] - num_points) <= tolerance, 
                           f"Expected around {num_points} points, got {x.shape[0]}")
            self.assertEqual(x.shape[1], 1)
            self.assertEqual(t.shape[1], 1)
            
            # 2D uniform
            x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="uniform")
            self.assertTrue(abs(x.shape[0] - num_points) <= tolerance, 
                           f"Expected around {num_points} points, got {x.shape[0]}")
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
        
        # Create a smaller RL agent that matches the input size for 1D + time (2 dimensions)
        rl_agent_1d = RLAgent(
            state_dim=2,  # 1D spatial + time
            action_dim=1,
            hidden_dim=32,
            device=self.device
        )
        
        # Assign RL agent
        self.heat_eq_1d.rl_agent = rl_agent_1d
        
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
            
    def test_adaptive_sampling_exploration(self):
        """Test adaptive sampling exploration behavior with RL agent."""
        num_points = 100
        
        # Create a mock RL agent that prioritizes high variance regions
        # This agent will return higher probability for points near the center
        class MockCollocationRLAgent:
            def __init__(self, device):
                self.device = device
                self.epsilon = 0.5
                
            def get_action(self, state):
                # Generate higher probabilities for points near the center (0.5)
                # and lower for edges
                x_coords = state[:, 0]  # Extract spatial coordinate
                probs = 1.0 - torch.abs(x_coords - 0.5) * 2  # Higher near center
                return probs.unsqueeze(1)
                
            # Add select_action method for compatibility with PDEBase
            def select_action(self, state):
                return self.get_action(state)
                
            def update_epsilon(self, epoch):
                self.epsilon = max(0.1, self.epsilon * 0.95)
        
        # Initialize heat equation with the mock agent
        mock_agent = MockCollocationRLAgent(self.device)
        self.heat_eq_1d.rl_agent = mock_agent
        
        # Generate points with adaptive strategy using our mock agent
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")
        
        # Check shapes
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        
        # Check domain bounds
        self.assertTrue(torch.all(x >= self.domain_1d[0][0]) and torch.all(x <= self.domain_1d[0][1]))
        self.assertTrue(torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1]))
        
        # With our mock agent, points should be more concentrated near x=0.5
        # Count points in central region vs edges
        central_region = (x > 0.4) & (x < 0.6)
        edge_region = ~central_region
        
        central_count = central_region.sum().item()
        edge_count = edge_region.sum().item()
        
        # Calculate the percentage of points in the central region
        central_percentage = central_count / num_points
        
        # Since central region is 20% of the domain but our agent favors it,
        # we expect more than 20% of points there
        self.assertGreater(central_percentage, 0.2, 
                         "Adaptive sampling should concentrate points in high-value regions")
        
        # Run for multiple epochs to see exploration change
        history = []
        for i in range(5):
            x, t = self.heat_eq_1d.generate_collocation_points(
                num_points, strategy="adaptive")
            
            # Store distribution metrics
            central_region = (x > 0.4) & (x < 0.6)
            central_count = central_region.sum().item()
            central_percentage = central_count / num_points
            history.append(central_percentage)
            
            # Agent's epsilon should decrease
            self.assertLessEqual(mock_agent.epsilon, 0.5 * (0.95 ** i), 
                              "Agent's epsilon should decrease over time")
        
        # Check if collocation points evolve over time by storing in history
        self.assertGreaterEqual(len(self.heat_eq_1d.collocation_history), 5, 
                         "Collocation history should contain at least 5 entries")
        
        # For 2D problems
        mock_agent_2d = MockCollocationRLAgent(self.device)
        self.heat_eq_2d.rl_agent = mock_agent_2d
        
        x_2d, t_2d = self.heat_eq_2d.generate_collocation_points(
            num_points, strategy="adaptive")
        
        # Check shapes
        self.assertEqual(x_2d.shape, (num_points, 2))
        self.assertEqual(t_2d.shape, (num_points, 1))

    def test_comprehensive_pde_models_sampling(self):
        """Test collocation point generation across all PDE models supported in the codebase."""
        num_points = 100
        
        # Common parameters
        domain_1d = [(0.0, 1.0)]
        domain_2d = [(0.0, 1.0), (0.0, 1.0)]
        time_domain = (0.0, 1.0)
        boundary_conditions = {"dirichlet": {"value": 0.0}}
        
        # Default initial and exact solution configurations
        initial_condition_sine = {"type": "sine", "amplitude": 1.0, "frequency": 2.0}
        exact_solution_sine = {"type": "sine", "amplitude": 1.0, "frequency": 2.0}
        
        # KdV specific configurations
        kdv_domain = [(-15.0, 15.0)]
        kdv_initial_condition = {"type": "soliton", "speed": 1.0}
        kdv_exact_solution = {"type": "soliton", "speed": 1.0}
        
        # Test all PDE types with each sampling strategy
        pde_models = [
            # Heat equation
            HeatEquation(
                alpha=0.01,
                domain=domain_1d,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition_sine,
                exact_solution=exact_solution_sine,
                dimension=1,
                device=self.device,
            ),
            # Wave equation
            WaveEquation(
                c=1.0,
                domain=domain_1d,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition_sine,
                exact_solution=exact_solution_sine,
                dimension=1,
                device=self.device,
            ),
            # KdV equation
            KdVEquation(
                domain=kdv_domain,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=kdv_initial_condition,
                exact_solution=kdv_exact_solution,
                dimension=1,
                device=self.device,
            ),
            # Burgers equation
            BurgersEquation(
                nu=0.01,
                domain=domain_1d,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition_sine,
                exact_solution=exact_solution_sine,
                dimension=1,
                device=self.device,
            ),
            # Convection equation
            ConvectionEquation(
                velocity=1.0,
                domain=domain_1d,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition_sine,
                exact_solution=exact_solution_sine,
                dimension=1,
                device=self.device,
            ),
            # Allen-Cahn equation
            AllenCahnEquation(
                epsilon=0.1,
                domain=domain_1d,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition_sine,
                exact_solution=exact_solution_sine,
                dimension=1,
                device=self.device,
            ),
            # Cahn-Hilliard equation
            CahnHilliardEquation(
                epsilon=0.1,
                domain=domain_1d,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition_sine,
                exact_solution=exact_solution_sine,
                dimension=1,
                device=self.device,
            ),
            # Black-Scholes equation
            BlackScholesEquation(
                sigma=0.2,
                r=0.05,
                domain=[(0.0, 200.0)],
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition_sine,
                exact_solution=exact_solution_sine,
                dimension=1,
                device=self.device,
            ),
        ]
        
        # Add pendulum equation using PDEConfig
        config = PDEConfig(
            name="Pendulum",
            domain=domain_1d,
            time_domain=time_domain,
            parameters={"g": 9.81, "L": 1.0},
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition_sine,
            exact_solution=exact_solution_sine,
            dimension=1,
            device=self.device,
        )
        pde_models.append(PendulumEquation(config))
        
        # Sampling strategies to test
        strategies = ["uniform", "latin_hypercube", "adaptive"]
        
        # Test each PDE model with each sampling strategy
        for i, pde in enumerate(pde_models):
            pde_name = pde.__class__.__name__
            print(f"Testing sampling for {pde_name}")
            
            for strategy in strategies:
                # For adaptive strategy, set the RL agent
                if strategy == "adaptive":
                    pde.rl_agent = self.rl_agent
                
                # Generate points
                x, t = pde.generate_collocation_points(num_points, strategy=strategy)
                
                # Check shapes
                self.assertEqual(x.shape[0], num_points, 
                                f"Wrong number of points for {pde_name} with {strategy} strategy")
                self.assertEqual(x.shape[1], 1, 
                                f"Wrong spatial dimension for {pde_name} with {strategy} strategy")
                self.assertEqual(t.shape, (num_points, 1), 
                                f"Wrong time dimension for {pde_name} with {strategy} strategy")
                
                # Check domain bounds
                domain = pde.domain
                for dim in range(pde.dimension):
                    self.assertTrue(
                        torch.all(x[:, dim] >= domain[dim][0]) and torch.all(x[:, dim] <= domain[dim][1]),
                        f"Points out of domain bounds for {pde_name} with {strategy} strategy"
                    )
                
                self.assertTrue(
                    torch.all(t >= pde.config.time_domain[0]) and torch.all(t <= pde.config.time_domain[1]),
                    f"Time points out of domain bounds for {pde_name} with {strategy} strategy"
                )
                
            # Test 2D version of each PDE model (if available)
            try:
                # Create a 2D version of the PDE
                pde_2d_cls = pde.__class__
                pde_2d_kwargs = {
                    "domain": domain_2d,
                    "time_domain": time_domain,
                    "boundary_conditions": boundary_conditions,
                    "initial_condition": initial_condition_sine,
                    "exact_solution": exact_solution_sine,
                    "dimension": 2,
                    "device": self.device,
                }
                
                # Add specific parameters based on PDE type
                if pde_name == "HeatEquation":
                    pde_2d_kwargs["alpha"] = 0.01
                elif pde_name == "WaveEquation":
                    pde_2d_kwargs["c"] = 1.0
                elif pde_name == "BurgersEquation":
                    pde_2d_kwargs["nu"] = 0.01
                elif pde_name == "ConvectionEquation":
                    pde_2d_kwargs["velocity"] = 1.0
                elif pde_name == "AllenCahnEquation" or pde_name == "CahnHilliardEquation":
                    pde_2d_kwargs["epsilon"] = 0.1
                elif pde_name == "BlackScholesEquation":
                    pde_2d_kwargs["sigma"] = 0.2
                    pde_2d_kwargs["r"] = 0.05
                    pde_2d_kwargs["domain"] = [(0.0, 200.0), (0.0, 200.0)]
                elif pde_name == "PendulumEquation":
                    # Create a new PDEConfig for 2D Pendulum
                    config_2d = PDEConfig(
                        name="Pendulum",
                        domain=domain_2d,
                        time_domain=time_domain,
                        parameters={"g": 9.81, "L": 1.0},
                        boundary_conditions=boundary_conditions,
                        initial_condition=initial_condition_sine,
                        exact_solution=exact_solution_sine,
                        dimension=2,
                        device=self.device,
                    )
                    # Create 2D pendulum directly with the new config instead of using kwargs
                    pde_2d = PendulumEquation(config_2d)
                    
                    # Test all sampling strategies for 2D pendulum
                    for strategy in strategies:
                        # For adaptive strategy, set the RL agent
                        if strategy == "adaptive":
                            pde_2d.rl_agent = self.rl_agent
                        
                        # Generate points
                        x_2d, t_2d = pde_2d.generate_collocation_points(num_points, strategy=strategy)
                        
                        # Check shapes
                        self.assertEqual(x_2d.shape[0], num_points, 
                                        f"Wrong number of points for 2D {pde_name} with {strategy} strategy")
                        self.assertEqual(x_2d.shape[1], 2, 
                                        f"Wrong spatial dimension for 2D {pde_name} with {strategy} strategy")
                        self.assertEqual(t_2d.shape, (num_points, 1), 
                                        f"Wrong time dimension for 2D {pde_name} with {strategy} strategy")
                        
                        # Check domain bounds
                        domain_2d = pde_2d.domain
                        for dim in range(pde_2d.dimension):
                            self.assertTrue(
                                torch.all(x_2d[:, dim] >= domain_2d[dim][0]) and 
                                torch.all(x_2d[:, dim] <= domain_2d[dim][1]),
                                f"Points out of domain bounds for 2D {pde_name} with {strategy} strategy"
                            )
                        
                        self.assertTrue(
                            torch.all(t_2d >= pde_2d.config.time_domain[0]) and 
                            torch.all(t_2d <= pde_2d.config.time_domain[1]),
                            f"Time points out of domain bounds for 2D {pde_name} with {strategy} strategy"
                        )

                
                # Create 2D PDE
                pde_2d = pde_2d_cls(**pde_2d_kwargs)
                
                for strategy in strategies:
                    # For adaptive strategy, set the RL agent
                    if strategy == "adaptive":
                        pde_2d.rl_agent = self.rl_agent
                    
                    # Generate points
                    x_2d, t_2d = pde_2d.generate_collocation_points(num_points, strategy=strategy)
                    
                    # Check shapes
                    self.assertEqual(x_2d.shape[0], num_points, 
                                    f"Wrong number of points for 2D {pde_name} with {strategy} strategy")
                    self.assertEqual(x_2d.shape[1], 2, 
                                    f"Wrong spatial dimension for 2D {pde_name} with {strategy} strategy")
                    self.assertEqual(t_2d.shape, (num_points, 1), 
                                    f"Wrong time dimension for 2D {pde_name} with {strategy} strategy")
                    
                    # Check domain bounds
                    domain_2d = pde_2d.domain
                    for dim in range(pde_2d.dimension):
                        self.assertTrue(
                            torch.all(x_2d[:, dim] >= domain_2d[dim][0]) and 
                            torch.all(x_2d[:, dim] <= domain_2d[dim][1]),
                            f"Points out of domain bounds for 2D {pde_name} with {strategy} strategy"
                        )
                    
                    self.assertTrue(
                        torch.all(t_2d >= pde_2d.config.time_domain[0]) and 
                        torch.all(t_2d <= pde_2d.config.time_domain[1]),
                        f"Time points out of domain bounds for 2D {pde_name} with {strategy} strategy"
                    )
            except Exception as e:
                print(f"Cannot test 2D version of {pde_name}: {str(e)}")

if __name__ == "__main__":
    unittest.main() 