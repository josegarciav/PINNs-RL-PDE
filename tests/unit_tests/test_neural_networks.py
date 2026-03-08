import unittest
import torch
import numpy as np
from src.neural_networks import (
    FeedForwardNetwork,
    ResNet,
    FourierNetwork,
    SIREN,
    AttentionNetwork,
    AutoEncoder,
    PINNModel,
    FourierFeatures,
    SIRENLayer,
    ResNetBlock,
    SelfAttention,
)
from src.config import Config, ModelConfig

# Import FeedForwardBlock from attention module instead of feedforward
from src.neural_networks.attention import FeedForwardBlock


def _make_config(input_dim=2, hidden_dim=32, output_dim=1, num_layers=3,
                 activation="tanh", architecture="feedforward", device=None):
    """Helper: build a minimal Config object for PINNModel tests."""
    if device is None:
        device = torch.device("cpu")
    cfg = Config.__new__(Config)
    cfg.device = device
    cfg.model = ModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        activation=activation,
        dropout=0.0,
        layer_norm=False,
        architecture=architecture,
    )
    # Ensure architecture-specific defaults are set
    if architecture == "fourier":
        cfg.model.mapping_size = 16
        cfg.model.scale = 10.0
    elif architecture == "siren":
        cfg.model.omega_0 = 30.0
    elif architecture == "attention":
        cfg.model.num_heads = 4
    elif architecture == "autoencoder":
        cfg.model.latent_dim = hidden_dim // 2
    elif architecture == "resnet":
        cfg.model.num_blocks = num_layers
    cfg.model.device = device
    return cfg


class TestNeuralNetworks(unittest.TestCase):
    """Test cases for different neural network architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 8
        self.input_dim = 2  # x, t
        self.hidden_dim = 32
        self.output_dim = 1
        self.sample_input = torch.rand(
            self.batch_size, self.input_dim, device=self.device
        )

    def test_feedforward_network(self):
        """Test FeedForwardNetwork architecture."""
        config = {
            "input_dim": self.input_dim,
            "hidden_dims": [self.hidden_dim, self.hidden_dim],
            "output_dim": self.output_dim,
            "activation": "relu",
            "dropout": 0.1,
            "device": self.device,
        }

        model = FeedForwardNetwork(config)
        output = model(self.sample_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        # Check if output requires grad
        self.assertTrue(output.requires_grad)
        # Check if model can be trained
        loss = torch.mean(output)
        loss.backward()
        # Check if gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_resnet(self):
        """Test ResNet architecture."""
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": 3,
            "output_dim": self.output_dim,
            "activation": "relu",  # String name, not the function
            "dropout": 0.1,
            "device": self.device,
        }

        model = ResNet(config)
        output = model(self.sample_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

        # Test gradients
        loss = torch.mean(output)
        loss.backward()

        # Check if gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

        # Verify ResNet architecture components
        self.assertEqual(len(model.blocks), config["num_blocks"])
        for block in model.blocks:
            self.assertIsInstance(block, ResNetBlock)

    def test_fourier_network(self):
        """Test FourierNetwork architecture."""
        config = {
            "input_dim": self.input_dim,
            "mapping_size": 16,
            "hidden_dim": self.hidden_dim,
            "hidden_dims": [self.hidden_dim, self.hidden_dim],
            "output_dim": self.output_dim,
            "activation": "relu",
            "scale": 10.0,
            "device": self.device,
        }

        model = FourierNetwork(config)
        output = model(self.sample_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        # Test gradients
        loss = torch.mean(output)
        loss.backward()
        # Make sure Fourier features layer has the expected parameters
        self.assertEqual(
            model.fourier.B.shape, (self.input_dim, config["mapping_size"])
        )

    def test_siren(self):
        """Test SIREN architecture."""
        config = {
            "input_dim": self.input_dim,
            "hidden_dims": [self.hidden_dim, self.hidden_dim],
            "output_dim": self.output_dim,
            "omega_0": 30.0,
            "device": self.device,
        }

        model = SIREN(config)
        output = model(self.sample_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        # Test gradients
        loss = torch.mean(output)
        loss.backward()
        # Verify omega_0 parameter - skip is_first check that doesn't exist
        for i, layer in enumerate(model.layers[:-1]):
            self.assertEqual(layer.omega_0, config["omega_0"])

    def test_attention_network(self):
        """Test AttentionNetwork architecture."""
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "activation": "gelu",
            "device": self.device,
        }

        model = AttentionNetwork(config)
        output = model(self.sample_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        # Test gradients
        loss = torch.mean(output)
        loss.backward()
        # Verify attention layers structure
        self.assertEqual(len(model.layers), config["num_layers"])
        for layer_pair in model.layers:
            self.assertIsInstance(layer_pair[0], SelfAttention)
            self.assertIsInstance(layer_pair[1], FeedForwardBlock)

    def test_autoencoder(self):
        """Test AutoEncoder architecture (PINN mode: encoder->latent->output_dim)."""
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "latent_dim": self.hidden_dim // 2,
            "hidden_dims": [self.hidden_dim, self.hidden_dim],
            "activation": "relu",
            "dropout": 0.0,
            "layer_norm": False,
            "device": self.device,
        }

        model = AutoEncoder(config)
        # Test encode function — should produce latent representation
        latent = model.encode(self.sample_input)
        self.assertEqual(latent.shape, (self.batch_size, config["latent_dim"]))

        # Test full forward pass — decoder outputs output_dim (solution value, not reconstruction)
        output = model(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

        # Test gradients
        loss = torch.mean(output)
        loss.backward()

    def test_pinn_model(self):
        """Test PINNModel with different architectures."""
        architectures = [
            "fourier",
            "resnet",
            "siren",
            "attention",
            "autoencoder",
            "feedforward",
        ]

        for arch in architectures:
            cfg = _make_config(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=3,
                activation="tanh",
                architecture=arch,
                device=self.device,
            )
            model = PINNModel(config=cfg, device=self.device)
            output = model(self.sample_input)

            self.assertEqual(
                output.shape,
                (self.batch_size, self.output_dim),
                f"Failed for architecture: {arch}",
            )

            loss = torch.mean(output)
            loss.backward()

            param_count = model.count_parameters()
            self.assertGreater(param_count, 0, f"No parameters in {arch} architecture")

            model.zero_grad()

    def test_save_load(self):
        """Test saving and loading models."""
        import tempfile
        import os

        cfg = _make_config(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=3,
            architecture="fourier",
            device=self.device,
        )
        model = PINNModel(config=cfg, device=self.device)

        before_output = model(self.sample_input).detach().clone()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name

        try:
            model.save_state(path)

            new_model = PINNModel(config=cfg, device=self.device)
            new_model.load_state(path)

            after_output = new_model(self.sample_input)

            self.assertTrue(
                torch.allclose(before_output, after_output, rtol=1e-5, atol=1e-5)
            )
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_fourier_features(self):
        """Test FourierFeatures layer."""
        mapping_size = 16
        scale = 10.0

        fourier = FourierFeatures(
            input_dim=self.input_dim,
            mapping_size=mapping_size,
            scale=scale,
            device=self.device,
        )

        output = fourier(self.sample_input)

        # Output should have shape [batch_size, 2*mapping_size]
        self.assertEqual(output.shape, (self.batch_size, 2 * mapping_size))

        # Test that the Fourier basis matrix B is properly initialized
        self.assertEqual(fourier.B.shape, (self.input_dim, mapping_size))

    def test_resnet_block(self):
        """Test ResNetBlock."""
        # Using the correct parameter names based on implementation
        block = ResNetBlock(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,  # Changed from out_dim to hidden_dim
            activation="relu",
            dropout=0.1,
        )

        x = torch.rand(self.batch_size, self.hidden_dim, device=self.device)
        output = block(x)

        # Output shape should be the same as input
        self.assertEqual(output.shape, x.shape)

        # Test that gradients flow through the residual connection
        loss = torch.mean(output)
        loss.backward()
        for param in block.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_gradient_stability(self):
        """Test that gradients are not infinite or unstable for each architecture."""
        architectures = [
            "fourier",
            "resnet",
            "siren",
            "attention",
            "autoencoder",
            "feedforward",
        ]

        for arch in architectures:
            cfg = _make_config(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=3,
                activation="tanh",
                architecture=arch,
                device=self.device,
            )
            model = PINNModel(config=cfg, device=self.device)

            # Forward pass with gradient tracking
            output = model(self.sample_input)

            # Compute gradients
            loss = torch.mean(output)
            loss.backward()

            # Check that gradients are finite and not too large
            for name, param in model.named_parameters():
                # Check gradients exist
                self.assertIsNotNone(param.grad, f"No gradients for {name} in {arch}")

                # Check no NaN values
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"NaN gradient detected for {name} in {arch}",
                )

                # Check no infinite values
                self.assertFalse(
                    torch.isinf(param.grad).any(),
                    f"Infinite gradient detected for {name} in {arch}",
                )

                # Check gradient magnitudes are reasonable (not exploding)
                max_grad = torch.max(torch.abs(param.grad)).item()
                self.assertLess(
                    max_grad,
                    100.0,
                    f"Exploding gradient detected for {name} in {arch}: {max_grad}",
                )

            # Reset gradients for next test
            model.zero_grad()

    def test_training_stability(self):
        """Test numerical stability during multiple training steps."""
        cfg = _make_config(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=3,
            activation="tanh",
            architecture="feedforward",
            device=self.device,
        )
        model = PINNModel(config=cfg, device=self.device)

        # Use a simple optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Run multiple training steps
        num_steps = 20
        loss_values = []

        for _ in range(num_steps):
            # Forward pass
            output = model(self.sample_input)
            loss = torch.mean(torch.square(output))  # MSE-like loss
            loss_values.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Check gradients before optimization step
            for param in model.parameters():
                self.assertFalse(
                    torch.isnan(param.grad).any(), "NaN gradient during training"
                )
                self.assertFalse(
                    torch.isinf(param.grad).any(), "Infinite gradient during training"
                )

            # Optimization step
            optimizer.step()

            # Check parameters after optimization step
            for param in model.parameters():
                self.assertFalse(
                    torch.isnan(param).any(), "NaN parameter during training"
                )
                self.assertFalse(
                    torch.isinf(param).any(), "Infinite parameter during training"
                )

        # Verify that loss is decreasing (generally)
        # Allow for some fluctuations but overall trend should be downward
        first_half_avg = sum(loss_values[: num_steps // 2]) / (num_steps // 2)
        second_half_avg = sum(loss_values[num_steps // 2 :]) / (
            num_steps - num_steps // 2
        )

        self.assertLess(
            second_half_avg,
            first_half_avg,
            "Loss not decreasing during training, possible instability",
        )


if __name__ == "__main__":
    unittest.main()
