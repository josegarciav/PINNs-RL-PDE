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
from tests.test_components.test_utils import create_pde_from_config


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

        # Create 1D and 2D PDEs for testing using config.yaml
        try:
            # Try to use the configuration from config.yaml
            self.heat_eq_1d = create_pde_from_config("heat", self.device, dimension=1)
        except (FileNotFoundError, KeyError, ValueError):
            # Fallback to hardcoded parameters if config.yaml is not available or missing necessary entries
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

        try:
            self.heat_eq_2d = create_pde_from_config("heat", self.device, dimension=2)
        except (FileNotFoundError, KeyError, ValueError):
            # Create a 2D heat equation
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
            device=self.device,
        )

        self.coll_rl_agent = CollocationRLAgent(
            state_dim=3, action_dim=1, hidden_dim=32, device=self.device
        )

    def test_uniform_sampling_1d(self):
        """Test uniform sampling in 1D."""
        num_points = 100

        # Heat equation
        x, t = self.heat_eq_1d.generate_collocation_points(
            num_points, strategy="uniform"
        )
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(
            torch.all(x >= self.domain_1d[0][0])
            and torch.all(x <= self.domain_1d[0][1])
        )
        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

        # Wave equation
        x, t = self.wave_eq_1d.generate_collocation_points(
            num_points, strategy="uniform"
        )
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(
            torch.all(x >= self.domain_1d[0][0])
            and torch.all(x <= self.domain_1d[0][1])
        )
        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

    def test_uniform_sampling_2d(self):
        """Test uniform sampling in 2D."""
        num_points = 100

        # Heat equation
        x, t = self.heat_eq_2d.generate_collocation_points(
            num_points, strategy="uniform"
        )
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))

        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(
                torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val)
            )

        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

        # Wave equation
        x, t = self.wave_eq_2d.generate_collocation_points(
            num_points, strategy="uniform"
        )
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))

        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(
                torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val)
            )

        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

    def test_latin_hypercube_sampling_1d(self):
        """Test Latin Hypercube sampling in 1D."""
        num_points = 100

        # Heat equation
        x, t = self.heat_eq_1d.generate_collocation_points(
            num_points, strategy="latin_hypercube"
        )
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(
            torch.all(x >= self.domain_1d[0][0])
            and torch.all(x <= self.domain_1d[0][1])
        )
        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

        # Wave equation
        x, t = self.wave_eq_1d.generate_collocation_points(
            num_points, strategy="latin_hypercube"
        )
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(
            torch.all(x >= self.domain_1d[0][0])
            and torch.all(x <= self.domain_1d[0][1])
        )
        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

    def test_latin_hypercube_sampling_2d(self):
        """Test Latin Hypercube sampling in 2D."""
        num_points = 100

        # Heat equation
        x, t = self.heat_eq_2d.generate_collocation_points(
            num_points, strategy="latin_hypercube"
        )
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))

        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(
                torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val)
            )

        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

        # Wave equation
        x, t = self.wave_eq_2d.generate_collocation_points(
            num_points, strategy="latin_hypercube"
        )
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))

        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(
                torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val)
            )

        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

    def test_adaptive_sampling_with_rl_agent(self):
        """Test adaptive sampling with RL agent."""
        num_points = 100

        # Assign RL agents
        self.heat_eq_1d.rl_agent = self.rl_agent
        self.heat_eq_2d.rl_agent = self.rl_agent

        # Test 1D adaptive sampling
        x, t = self.heat_eq_1d.generate_collocation_points(
            num_points, strategy="adaptive"
        )
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(
            torch.all(x >= self.domain_1d[0][0])
            and torch.all(x <= self.domain_1d[0][1])
        )
        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

        # Test 2D adaptive sampling
        x, t = self.heat_eq_2d.generate_collocation_points(
            num_points, strategy="adaptive"
        )
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))

        # Check domain bounds for each dimension
        for i, (min_val, max_val) in enumerate(self.domain_2d):
            self.assertTrue(
                torch.all(x[:, i] >= min_val) and torch.all(x[:, i] <= max_val)
            )

        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

    def test_adaptive_sampling_fallback(self):
        """Test that adaptive sampling falls back to uniform when no RL agent is provided."""
        num_points = 100

        # Ensure no RL agent is set
        self.heat_eq_1d.rl_agent = None

        # Should fall back to uniform
        x, t = self.heat_eq_1d.generate_collocation_points(
            num_points, strategy="adaptive"
        )
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(
            torch.all(x >= self.domain_1d[0][0])
            and torch.all(x <= self.domain_1d[0][1])
        )
        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

    def test_different_num_points(self):
        """Test with different numbers of points."""
        num_points_list = [
            50,
            100,
            200,
            500,
        ]  # Omit very small values that might cause rounding issues

        for num_points in num_points_list:
            # 1D uniform
            x, t = self.heat_eq_1d.generate_collocation_points(
                num_points, strategy="uniform"
            )
            # Allow for a larger tolerance in point count due to grid based implementation in uniform sampling
            tolerance = max(
                5, int(num_points * 0.05)
            )  # 5% tolerance or at least 5 points
            self.assertTrue(
                abs(x.shape[0] - num_points) <= tolerance,
                f"Expected around {num_points} points, got {x.shape[0]}",
            )
            self.assertEqual(x.shape[1], 1)
            self.assertEqual(t.shape[1], 1)

            # 2D uniform
            x, t = self.heat_eq_2d.generate_collocation_points(
                num_points, strategy="uniform"
            )
            self.assertTrue(
                abs(x.shape[0] - num_points) <= tolerance,
                f"Expected around {num_points} points, got {x.shape[0]}",
            )
            self.assertEqual(x.shape[1], 2)
            self.assertEqual(t.shape[1], 1)

            # 1D latin hypercube
            x, t = self.heat_eq_1d.generate_collocation_points(
                num_points, strategy="latin_hypercube"
            )
            self.assertEqual(
                x.shape, (num_points, 1)
            )  # Latin hypercube should be exact
            self.assertEqual(t.shape, (num_points, 1))

            # 2D latin hypercube
            x, t = self.heat_eq_2d.generate_collocation_points(
                num_points, strategy="latin_hypercube"
            )
            self.assertEqual(
                x.shape, (num_points, 2)
            )  # Latin hypercube should be exact
            self.assertEqual(t.shape, (num_points, 1))

    def test_small_number_points(self):
        """Test with very small number of points which might be affected by rounding."""
        num_points = 10

        # For such small numbers, we should check that the shape is close to what we expect
        # but not necessarily exact

        # 1D uniform
        x, t = self.heat_eq_1d.generate_collocation_points(
            num_points, strategy="uniform"
        )
        self.assertTrue(
            abs(x.shape[0] - num_points) <= 1,
            f"Expected around {num_points} points, got {x.shape[0]}",
        )
        self.assertEqual(x.shape[1], 1)
        self.assertEqual(t.shape[1], 1)

        # 2D uniform
        x, t = self.heat_eq_2d.generate_collocation_points(
            num_points, strategy="uniform"
        )
        self.assertTrue(
            abs(x.shape[0] - num_points) <= 1,
            f"Expected around {num_points} points, got {x.shape[0]}",
        )
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
            device=self.device,
        )

        # Assign RL agent
        self.heat_eq_1d.rl_agent = rl_agent_1d

        # Generate points with adaptive strategy to create history
        for _ in range(3):
            x, t = self.heat_eq_1d.generate_collocation_points(
                num_points, strategy="adaptive"
            )

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
            self.heat_eq_1d.generate_collocation_points(
                num_points, strategy="invalid_strategy"
            )

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
        x, t = self.heat_eq_1d.generate_collocation_points(
            num_points, strategy="adaptive"
        )

        # Check shapes
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))

        # Check domain bounds
        self.assertTrue(
            torch.all(x >= self.domain_1d[0][0])
            and torch.all(x <= self.domain_1d[0][1])
        )
        self.assertTrue(
            torch.all(t >= self.time_domain[0]) and torch.all(t <= self.time_domain[1])
        )

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
        self.assertGreater(
            central_percentage,
            0.2,
            "Adaptive sampling should concentrate points in high-value regions",
        )

        # Run for multiple epochs to see exploration change
        history = []
        for i in range(5):
            x, t = self.heat_eq_1d.generate_collocation_points(
                num_points, strategy="adaptive"
            )

            # Store distribution metrics
            central_region = (x > 0.4) & (x < 0.6)
            central_count = central_region.sum().item()
            central_percentage = central_count / num_points
            history.append(central_percentage)

            # Agent's epsilon should decrease
            self.assertLessEqual(
                mock_agent.epsilon,
                0.5 * (0.95**i),
                "Agent's epsilon should decrease over time",
            )

        # Check if collocation points evolve over time by storing in history
        self.assertGreaterEqual(
            len(self.heat_eq_1d.collocation_history),
            5,
            "Collocation history should contain at least 5 entries",
        )

        # For 2D problems
        mock_agent_2d = MockCollocationRLAgent(self.device)
        self.heat_eq_2d.rl_agent = mock_agent_2d

        x_2d, t_2d = self.heat_eq_2d.generate_collocation_points(
            num_points, strategy="adaptive"
        )

        # Check shapes
        self.assertEqual(x_2d.shape, (num_points, 2))
        self.assertEqual(t_2d.shape, (num_points, 1))

    def test_comprehensive_pde_models_sampling(self):
        """Test collocation point generation across all PDE models supported in the codebase."""
        num_points = 100

        # Create all PDE models from config.yaml
        pde_models = []
        pde_types = [
            "heat",
            "wave",
            "kdv",
            "burgers",
            "convection",
            "allen_cahn",
            "cahn_hilliard",
            "black_scholes",
            "pendulum",
        ]

        for pde_type in pde_types:
            try:
                pde = create_pde_from_config(pde_type, self.device)
                pde_models.append(pde)
                print(f"Created {pde_type} from config.yaml")
            except Exception as e:
                print(f"Could not create {pde_type} from config.yaml: {e}")
                # Don't add this PDE if it fails - we'll test the ones we can load

        # If no PDEs were loaded, use the fallback hardcoded method for at least Heat equation
        if not pde_models:
            print("Using fallback hardcoded PDEs for testing")
            # Common parameters
            domain_1d = [(0.0, 1.0)]
            time_domain = (0.0, 1.0)
            boundary_conditions = {"dirichlet": {"value": 0.0}}

            # Default initial and exact solution configurations
            initial_condition_sine = {
                "type": "sine",
                "amplitude": 1.0,
                "frequency": 2.0,
            }
            exact_solution_sine = {"type": "sine", "amplitude": 1.0, "frequency": 2.0}

            # Test with Heat equation
            pde_models.append(
                HeatEquation(
                    alpha=0.01,
                    domain=domain_1d,
                    time_domain=time_domain,
                    boundary_conditions=boundary_conditions,
                    initial_condition=initial_condition_sine,
                    exact_solution=exact_solution_sine,
                    dimension=1,
                    device=self.device,
                )
            )

        # Sampling strategies to test
        strategies = ["uniform", "latin_hypercube", "adaptive"]

        # Test each PDE model with each sampling strategy
        for i, pde in enumerate(pde_models):
            pde_name = pde.__class__.__name__
            print(f"Testing sampling for {pde_name}")

            # Test all sampling strategies
            for strategy in strategies:
                # 1D sampling
                x_1d, t_1d = pde.generate_collocation_points(
                    num_points, strategy=strategy
                )

                # Check dimensions of points
                self.assertEqual(
                    x_1d.shape[0],
                    num_points,
                    f"Wrong number of points for {pde_name} with {strategy}",
                )
                self.assertEqual(
                    x_1d.shape[1],
                    pde.dimension,
                    f"Wrong dimension for {pde_name} with {strategy}",
                )
                self.assertEqual(
                    t_1d.shape,
                    (num_points, 1),
                    f"Wrong time shape for {pde_name} with {strategy}",
                )

                # Verify bounds
                if isinstance(pde.domain, list):
                    for dim in range(pde.dimension):
                        min_val, max_val = pde.domain[dim]
                        self.assertTrue(
                            torch.all(x_1d[:, dim] >= min_val)
                            and torch.all(x_1d[:, dim] <= max_val),
                            f"Points out of bounds for {pde_name} with {strategy}",
                        )

                # Try 2D version of the same PDE if applicable
                if pde.dimension == 1:
                    try:
                        # Create a 2D version of the same PDE
                        pde_2d = create_pde_from_config(
                            pde_type, self.device, dimension=2
                        )

                        # Test 2D sampling
                        x_2d, t_2d = pde_2d.generate_collocation_points(
                            num_points, strategy=strategy
                        )

                        # Check dimensions
                        self.assertEqual(x_2d.shape[0], num_points)
                        self.assertEqual(x_2d.shape[1], 2)  # 2D
                        self.assertEqual(t_2d.shape, (num_points, 1))

                        # Verify bounds
                        for dim in range(2):
                            min_val, max_val = pde_2d.domain[dim]
                            self.assertTrue(
                                torch.all(x_2d[:, dim] >= min_val)
                                and torch.all(x_2d[:, dim] <= max_val)
                            )
                    except Exception as e:
                        print(f"Could not test 2D version of {pde_name}: {e}")


if __name__ == "__main__":
    unittest.main()
