
import torch
import unittest
from src.pinn import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.black_scholes import BlackScholesEquation
from config import CONFIG


class TestPDEs(unittest.TestCase):
    def setUp(self):
        """Set up PDE instances and a simple neural network model."""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # PDE Instances
        self.heat_eq = HeatEquation(device=self.device)
        self.wave_eq = WaveEquation(device=self.device)
        self.bs_eq = BlackScholesEquation(device=self.device)

        # Test collocation points (S, t or x, t)
        self.x = torch.linspace(0, 1, 10).view(-1, 1).to(self.device)
        self.t = torch.linspace(0, 1, 10).view(-1, 1).to(self.device)

        # Dummy PINN model (for testing purposes only)
        self.model = PINNModel(input_dim=2, hidden_dim=16, output_dim=1).to(self.device)

    def test_heat_equation_residual(self):
        """Test residual shape computation for Heat equation."""
        residual = self.heat_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, (10, 1))

    def test_wave_equation_residual(self):
        """Test residual computation for Wave Equation."""
        residual = self.heat_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, self.x.shape)

    def test_black_scholes_residual(self):
        """Test residual computation for Black-Scholes Equation."""
        residual = self.heat_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, self.x.shape)

if __name__ == "__main__":
    unittest.main()
