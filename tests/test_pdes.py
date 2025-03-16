
import torch
import unittest
from src.pinn import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.black_scholes import BlackScholesEquation


class TestPDEs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.x = torch.linspace(0, 1, 10).unsqueeze(1).to(self.device)
        self.t = torch.linspace(0, 1, 10).unsqueeze(1).to(self.device)
        self.xt = torch.cat([self.x, self.t], dim=1).requires_grad_(True)

        self.heat_eq = HeatEquation(device=self.device)
        self.wave_eq = WaveEquation(device=self.device)
        self.bs_eq = BlackScholesEquation(device=self.device)

        # Real PINN model
        self.model = PINNModel(input_dim=2, hidden_dim=16, output_dim=1, device=self.device)

    def test_heat_equation_residual(self):
        residual = self.heat_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, (10, 1))

    def test_wave_equation_residual(self):
        residual = self.wave_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, (10, 1))

    def test_black_scholes_residual(self):
        residual = self.bs_eq.compute_residual(self.model, self.x, self.t)
        self.assertEqual(residual.shape, (10, 1))

if __name__ == "__main__":
    unittest.main()
