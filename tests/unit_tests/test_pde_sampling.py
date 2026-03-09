import unittest

import torch
import torch.nn as nn

from src.pdes.heat_equation import HeatEquation
from src.rl.rl_agent import CollocationRLAgent, RLAgent
from tests.unit_tests.test_utils import create_pde_from_config


class TestPDESampling(unittest.TestCase):
    """Test all four sampling strategies: uniform, stratified, residual_based, adaptive."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

        # Create 1D and 2D PDEs for testing using config.yaml
        self.heat_eq_1d = create_pde_from_config("heat", self.device, dimension=1)
        self.heat_eq_2d = create_pde_from_config("heat", self.device, dimension=2)

        # Derive domain bounds from actual PDE config
        self.domain_1d = list(self.heat_eq_1d.domain)
        self.domain_2d = list(self.heat_eq_2d.domain)
        self.time_domain = self.heat_eq_1d.time_domain

        try:
            self.wave_eq_1d = create_pde_from_config("wave", self.device, dimension=1)
        except (FileNotFoundError, KeyError, ValueError):
            self.wave_eq_1d = None

        try:
            self.wave_eq_2d = create_pde_from_config("wave", self.device, dimension=2)
        except (FileNotFoundError, KeyError, ValueError):
            self.wave_eq_2d = None

        # Create RL agents for adaptive sampling
        self.rl_agent = RLAgent(state_dim=3, action_dim=1, hidden_dim=32, device=self.device)
        self.coll_rl_agent = CollocationRLAgent(
            state_dim=3, action_dim=1, hidden_dim=32, device=self.device
        )

    def _assert_bounds(self, x, t, pde):
        """Helper to assert points are within PDE domain bounds."""
        for dim in range(pde.dimension):
            x_min, x_max = pde.domain[dim]
            self.assertTrue(
                torch.all(x[:, dim] >= x_min) and torch.all(x[:, dim] <= x_max),
                f"Spatial dim {dim} out of bounds [{x_min}, {x_max}]",
            )
        t_min, t_max = pde.time_domain
        self.assertTrue(
            torch.all(t >= t_min) and torch.all(t <= t_max),
            f"Time out of bounds [{t_min}, {t_max}]",
        )

    def _make_dummy_model(self, input_dim, output_dim=1):
        """Create a simple model for residual-based sampling tests."""
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
        ).to(self.device)

    # ── Uniform sampling ──────────────────────────────────────────────

    def test_uniform_sampling_1d(self):
        """Test uniform sampling in 1D."""
        num_points = 100
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_1d)

        x, t = self.wave_eq_1d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.wave_eq_1d)

    def test_uniform_sampling_2d(self):
        """Test uniform sampling in 2D."""
        num_points = 100
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_2d)

        x, t = self.wave_eq_2d.generate_collocation_points(num_points, strategy="uniform")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.wave_eq_2d)

    # ── Stratified sampling ───────────────────────────────────────────

    def test_stratified_sampling_1d(self):
        """Test stratified (LHS-style) sampling in 1D."""
        num_points = 100
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="stratified")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_1d)

    def test_stratified_sampling_2d(self):
        """Test stratified sampling in 2D."""
        num_points = 100
        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="stratified")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_2d)

    def test_stratified_coverage(self):
        """Stratified sampling should cover the domain more evenly than pure random."""
        num_points = 200
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="stratified")
        x_min, x_max = self.heat_eq_1d.domain[0]
        x_mid = (x_min + x_max) / 2.0

        # Split domain in half — stratified should put ~50% in each half
        lower_half = (x < x_mid).sum().item()
        ratio = lower_half / num_points
        self.assertGreater(ratio, 0.35, "Stratified should cover lower half")
        self.assertLess(ratio, 0.65, "Stratified should cover upper half")

    # ── Residual-based (RAR) sampling ─────────────────────────────────

    def test_residual_based_no_model_fallback(self):
        """Residual-based sampling without model should fall back to uniform."""
        num_points = 100
        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="residual_based")
        # Should still return valid points (uniform fallback)
        self.assertEqual(x.shape[1], 1)
        self.assertEqual(t.shape[1], 1)
        self._assert_bounds(x, t, self.heat_eq_1d)

    def test_residual_based_with_model(self):
        """Residual-based sampling with a model should return valid points."""
        num_points = 100
        model = self._make_dummy_model(input_dim=2)  # x + t

        x, t = self.heat_eq_1d.generate_collocation_points(
            num_points, strategy="residual_based", model=model
        )
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_1d)

    def test_residual_based_2d(self):
        """Residual-based sampling in 2D."""
        num_points = 100
        model = self._make_dummy_model(input_dim=3)  # x1 + x2 + t

        x, t = self.heat_eq_2d.generate_collocation_points(
            num_points, strategy="residual_based", model=model
        )
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_2d)

    # ── RL adaptive sampling ──────────────────────────────────────────

    def test_adaptive_sampling_with_rl_agent(self):
        """Test adaptive sampling with RL agent."""
        num_points = 100

        self.heat_eq_1d.rl_agent = self.rl_agent
        self.heat_eq_2d.rl_agent = self.rl_agent

        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_1d)

        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 2))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_2d)

    def test_adaptive_sampling_fallback(self):
        """Adaptive sampling falls back to uniform when no RL agent is set."""
        num_points = 100
        self.heat_eq_1d.rl_agent = None

        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self._assert_bounds(x, t, self.heat_eq_1d)

    def test_adaptive_sampling_exploration(self):
        """Test adaptive sampling exploration behavior with mock RL agent."""
        num_points = 100

        class MockCollocationRLAgent:
            def __init__(self, device, domain_range):
                self.device = device
                self.epsilon = 0.5
                self.domain_range = domain_range

            def get_action(self, state):
                x_coords = state[:, 0]
                x_min, x_max = self.domain_range
                x_mid = (x_min + x_max) / 2.0
                x_half = (x_max - x_min) / 2.0
                probs = 1.0 - torch.abs(x_coords - x_mid) / x_half
                return probs.unsqueeze(1)

            def select_action(self, state):
                return self.get_action(state)

            def update_epsilon(self, epoch):
                self.epsilon = max(0.1, self.epsilon * 0.95)

        x_min, x_max = self.heat_eq_1d.domain[0]
        t_min, t_max = self.heat_eq_1d.time_domain
        x_mid = (x_min + x_max) / 2.0
        x_span = x_max - x_min
        mock_agent = MockCollocationRLAgent(self.device, domain_range=(x_min, x_max))
        self.heat_eq_1d.rl_agent = mock_agent

        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")

        self.assertEqual(x.shape, (num_points, 1))
        self.assertEqual(t.shape, (num_points, 1))
        self.assertTrue(torch.all(x >= x_min) and torch.all(x <= x_max))
        self.assertTrue(torch.all(t >= t_min) and torch.all(t <= t_max))

        # Central 20% of domain should have >20% of points due to agent bias
        central_lo = x_mid - 0.1 * x_span
        central_hi = x_mid + 0.1 * x_span
        central_count = ((x > central_lo) & (x < central_hi)).sum().item()
        self.assertGreater(
            central_count / num_points,
            0.2,
            "Adaptive sampling should concentrate points in high-value regions",
        )

        # Run multiple epochs — epsilon should decrease
        for i in range(5):
            self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")
            self.assertLessEqual(
                mock_agent.epsilon,
                0.5 * (0.95**i),
                "Agent's epsilon should decrease over time",
            )

        self.assertGreaterEqual(len(self.heat_eq_1d.collocation_history), 5)

        # Test 2D adaptive with mock agent
        mock_agent_2d = MockCollocationRLAgent(self.device, domain_range=self.heat_eq_2d.domain[0])
        self.heat_eq_2d.rl_agent = mock_agent_2d
        x_2d, t_2d = self.heat_eq_2d.generate_collocation_points(num_points, strategy="adaptive")
        self.assertEqual(x_2d.shape, (num_points, 2))
        self.assertEqual(t_2d.shape, (num_points, 1))

    # ── Collocation history ───────────────────────────────────────────

    def test_collocation_history(self):
        """Test that collocation history is maintained correctly with RL agent."""
        num_points = 100
        rl_agent_1d = RLAgent(state_dim=2, action_dim=1, hidden_dim=32, device=self.device)
        self.heat_eq_1d.rl_agent = rl_agent_1d

        for _ in range(3):
            self.heat_eq_1d.generate_collocation_points(num_points, strategy="adaptive")

        self.assertEqual(len(self.heat_eq_1d.collocation_history), 3)
        for h in self.heat_eq_1d.collocation_history:
            self.assertEqual(h.shape[0], num_points)
            self.assertEqual(h.shape[1], 2)  # 1D spatial + time

    # ── General / edge cases ──────────────────────────────────────────

    def test_invalid_strategy(self):
        """Test that an invalid strategy raises an error."""
        with self.assertRaises(ValueError):
            self.heat_eq_1d.generate_collocation_points(100, strategy="invalid_strategy")

    def test_different_num_points(self):
        """Test with different numbers of points."""
        for num_points in [50, 100, 200, 500]:
            tolerance = max(5, int(num_points * 0.05))

            x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="uniform")
            self.assertTrue(
                abs(x.shape[0] - num_points) <= tolerance,
                f"Expected around {num_points} points, got {x.shape[0]}",
            )
            self.assertEqual(x.shape[1], 1)
            self.assertEqual(t.shape[1], 1)

            # Stratified should always return exact count
            x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="stratified")
            self.assertEqual(x.shape[0], num_points)

    def test_small_number_points(self):
        """Test with very small number of points."""
        num_points = 10

        x, t = self.heat_eq_1d.generate_collocation_points(num_points, strategy="uniform")
        self.assertTrue(
            abs(x.shape[0] - num_points) <= 1,
            f"Expected around {num_points} points, got {x.shape[0]}",
        )
        self.assertEqual(x.shape[1], 1)
        self.assertEqual(t.shape[1], 1)

        x, t = self.heat_eq_2d.generate_collocation_points(num_points, strategy="uniform")
        self.assertTrue(
            abs(x.shape[0] - num_points) <= 1,
            f"Expected around {num_points} points, got {x.shape[0]}",
        )
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(t.shape[1], 1)

    def test_comprehensive_pde_models_sampling(self):
        """Test collocation point generation across all PDE models and strategies."""
        num_points = 100

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

        pde_models = []
        for pde_type in pde_types:
            try:
                pde = create_pde_from_config(pde_type, self.device)
                pde_models.append((pde_type, pde))
            except Exception as e:
                print(f"Could not create {pde_type}: {e}")

        if not pde_models:
            pde_models.append(
                (
                    "heat",
                    HeatEquation(
                        alpha=0.01,
                        domain=[(0.0, 1.0)],
                        time_domain=(0.0, 1.0),
                        boundary_conditions={"dirichlet": {"value": 0.0}},
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
                        device=self.device,
                    ),
                )
            )

        strategies = ["uniform", "stratified", "adaptive"]

        for pde_type, pde in pde_models:
            pde_name = pde.__class__.__name__
            for strategy in strategies:
                x, t = pde.generate_collocation_points(num_points, strategy=strategy)

                self.assertEqual(
                    x.shape[0],
                    num_points,
                    f"Wrong number of points for {pde_name} with {strategy}",
                )
                self.assertEqual(
                    x.shape[1],
                    pde.dimension,
                    f"Wrong dimension for {pde_name} with {strategy}",
                )
                self.assertEqual(
                    t.shape,
                    (num_points, 1),
                    f"Wrong time shape for {pde_name} with {strategy}",
                )
                self._assert_bounds(x, t, pde)


if __name__ == "__main__":
    unittest.main()
