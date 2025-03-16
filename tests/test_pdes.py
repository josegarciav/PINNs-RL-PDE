
import torch
import unittest
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.black_scholes import BlackScholesEquation

class TestPDEs(unittest.TestCase):
    def setUp(self):
        """Initialize PDEs for testing."""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.heat_eq = HeatEquation(device=self.device)
        self.wave_eq = WaveEquation(device=self.device)
        self.bs_eq = BlackScholesEquation(device=self.device)

        # Generate test collocation points
        self.x = torch.linspace(0, 1, 10).view(-1, 1).to(self.device)
        self.t = torch.linspace(0, 1, 10).view(-1, 1).to(self.device)

    def test_heat_equation_residual(self):
        """Test if the Heat Equation residual has the correct shape."""
        model = lambda x: torch.sin(torch.pi * x)  # Dummy model
        residual = self.heat_eq.compute_residual(model, self.x, self.t)
        self.assertEqual(residual.shape, self.x.shape)

    def test_wave_equation_residual(self):
        """Test if the Wave Equation residual has the correct shape."""
        model = lambda x: torch.sin(torch.pi * x)  # Dummy model
        residual = self.wave_eq.compute_residual(model, self.x, self.t)
        self.assertEqual(residual.shape, self.x.shape)

    def test_black_scholes_residual(self):
        """Test if the Black-Scholes residual is computed correctly."""
        model = lambda x: torch.maximum(x - 1, torch.zeros_like(x))  # Call option model
        residual = self.bs_eq.compute_residual(model, self.x, self.t)
        self.assertEqual(residual.shape, self.x.shape)

if __name__ == "__main__":
    unittest.main()
