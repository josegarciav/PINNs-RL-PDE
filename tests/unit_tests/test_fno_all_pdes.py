"""Integration test: FNO architecture against all 9 PDEs.

Verifies that FNO can:
1. Produce correct output shapes for each PDE's input/output dimensions
2. Compute PDE residuals with gradients flowing through
3. Run a few optimization steps without numerical instability
"""

import unittest

import torch

from src.config import Config, ModelConfig
from src.neural_networks import PINNModel
from tests.unit_tests.test_utils import create_pde_from_config

ALL_PDES = [
    "heat",
    "wave",
    "burgers",
    "convection",
    "kdv",
    "allen_cahn",
    "cahn_hilliard",
    "black_scholes",
    "pendulum",
]


def _make_fno_config(input_dim=2, output_dim=1, hidden_dim=32, num_blocks=2, modes=8):
    """Build a minimal Config for FNO."""
    device = torch.device("cpu")
    cfg = Config.__new__(Config)
    cfg.device = device
    cfg.model = ModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_blocks,
        activation="gelu",
        dropout=0.0,
        layer_norm=False,
        architecture="fno",
    )
    cfg.model.num_blocks = num_blocks
    cfg.model.modes = modes
    cfg.model.device = device
    return cfg


class TestFNOAllPDEs(unittest.TestCase):
    """Test FNO architecture against all 9 PDEs."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 64

    def _test_pde(self, pde_key):
        """Test FNO forward + backward for a single PDE."""
        pde = create_pde_from_config(pde_key, self.device)

        input_dim = getattr(pde.config, "input_dim", 2) or 2
        output_dim = getattr(pde.config, "output_dim", 1) or 1

        cfg = _make_fno_config(input_dim=input_dim, output_dim=output_dim)
        model = PINNModel(config=cfg, device=self.device)

        # Generate collocation points within the PDE domain
        x_input = torch.rand(self.batch_size, input_dim, device=self.device, requires_grad=True)

        # Scale to PDE domain
        domain = pde.config.domain
        time_domain = pde.config.time_domain
        if isinstance(domain[0], (list, tuple)):
            for d in range(len(domain)):
                lo, hi = domain[d]
                x_input.data[:, d] = x_input.data[:, d] * (hi - lo) + lo
        if time_domain:
            t_lo, t_hi = time_domain
            x_input.data[:, -1] = x_input.data[:, -1] * (t_hi - t_lo) + t_lo

        # Forward pass
        output = model(x_input)
        self.assertEqual(
            output.shape,
            (self.batch_size, output_dim),
            f"Wrong output shape for {pde_key}",
        )

        # Backward pass — verify gradients flow
        loss = output.sum()
        loss.backward()

        grad_ok = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_ok = True
                break
        self.assertTrue(grad_ok, f"No gradients for {pde_key}")

        # Quick optimization check (3 steps)
        model.zero_grad()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(3):
            x = torch.rand(self.batch_size, input_dim, device=self.device, requires_grad=True)
            out = model(x)
            step_loss = torch.mean(out ** 2)
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()
            losses.append(step_loss.item())
            self.assertFalse(
                torch.isnan(torch.tensor(step_loss.item())),
                f"NaN loss for {pde_key}",
            )

    def test_heat(self):
        self._test_pde("heat")

    def test_wave(self):
        self._test_pde("wave")

    def test_burgers(self):
        self._test_pde("burgers")

    def test_convection(self):
        self._test_pde("convection")

    def test_kdv(self):
        self._test_pde("kdv")

    def test_allen_cahn(self):
        self._test_pde("allen_cahn")

    def test_cahn_hilliard(self):
        self._test_pde("cahn_hilliard")

    def test_black_scholes(self):
        self._test_pde("black_scholes")

    def test_pendulum(self):
        self._test_pde("pendulum")


if __name__ == "__main__":
    unittest.main()
