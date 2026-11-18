"""Configuration branches of the networks: activations, ResNet defaults, layer-norm variants."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinnrl.neural_networks import AutoEncoder, ResNet
from pinnrl.neural_networks.base_network import BaseNetwork, ExactLayerNorm
from pinnrl.neural_networks.resnet import ResNetBlock


@pytest.mark.parametrize(
    "name, module",
    [
        ("relu", nn.ReLU),
        ("leaky_relu", nn.LeakyReLU),
        ("tanh", nn.Tanh),
        ("sigmoid", nn.Sigmoid),
        ("gelu", nn.GELU),
    ],
)
def test_resnet_block_activations(name, module):
    block = ResNetBlock(4, 8, activation=name, dropout=0.0)
    assert isinstance(block.activation_fn, module)
    y = block(torch.zeros(3, 4))
    assert y.shape == (3, 4)


def test_resnet_block_rejects_unknown_activation():
    with pytest.raises(ValueError, match="Unsupported activation"):
        ResNetBlock(4, 8, activation="swish")


def test_resnet_width_and_depth_fallbacks():
    base = {"input_dim": 2, "output_dim": 1, "dropout": 0.0}
    from_list = ResNet({**base, "hidden_dims": [16, 32], "num_layers": 2})
    assert from_list.hidden_dim == 16 and from_list.num_blocks == 2
    assert len(from_list.blocks) == 2
    defaults = ResNet(base)
    assert defaults.hidden_dim == 124 and defaults.num_blocks == 4
    explicit = ResNet({**base, "hidden_dim": 8, "num_blocks": 1, "activation": "gelu"})
    assert explicit.hidden_dim == 8 and len(explicit.blocks) == 1
    assert isinstance(explicit.activation_fn, nn.GELU)
    assert explicit(np.zeros((5, 2), dtype=np.float32)).shape == (5, 1)


@pytest.mark.parametrize("name, module", [("leaky_relu", nn.LeakyReLU), ("sigmoid", nn.Sigmoid)])
def test_base_network_activation_lookup(name, module):
    net = BaseNetwork({"device": torch.device("cpu")})
    assert isinstance(net._get_activation_module(name), module)


def test_base_network_rejects_unknown_activation_and_has_no_forward():
    net = BaseNetwork({"device": torch.device("cpu")})
    with pytest.raises(ValueError, match="Unsupported activation"):
        net._get_activation_module("swish")
    with pytest.raises(NotImplementedError):
        net(torch.zeros(1, 1))


def test_base_network_prepares_lists_scalars_and_devices():
    net = BaseNetwork({"device": torch.device("cpu")})
    from_list = net._prepare_input([[1.0, 2.0]])
    assert from_list.dtype == torch.float32 and from_list.shape == (1, 2)
    from_scalar = net._prepare_input(3.0)
    assert from_scalar.dtype == torch.float32 and float(from_scalar) == 3.0
    # torch distinguishes "cpu" from "cpu:0": the tensor is moved onto the network's device
    indexed = BaseNetwork({"device": torch.device("cpu:0")})
    moved = indexed._prepare_input(torch.zeros(2, 1))
    assert moved.shape == (2, 1) and moved.device.type == "cpu"


def test_exact_layer_norm_variants_match_torch():
    x = torch.randn(6, 4, 4, dtype=torch.float64)
    multi = ExactLayerNorm((4, 4)).double()  # more than one normalized dimension: the built-in
    assert torch.allclose(multi(x), nn.LayerNorm((4, 4)).double()(x))
    plain = ExactLayerNorm(4, elementwise_affine=False).double()
    assert torch.allclose(plain(x), nn.LayerNorm(4, elementwise_affine=False).double()(x))
    no_bias = ExactLayerNorm(4, bias=False).double()
    with torch.no_grad():
        no_bias.weight.mul_(2.0)
    reference = nn.LayerNorm(4, bias=False).double()
    with torch.no_grad():
        reference.weight.copy_(no_bias.weight)
    assert torch.allclose(no_bias(x), reference(x))


def test_autoencoder_with_layer_norm_and_dropout():
    net = AutoEncoder(
        {
            "input_dim": 2,
            "output_dim": 1,
            "hidden_dims": [8, 8],
            "latent_dim": 4,
            "layer_norm": True,
            "dropout": 0.1,
        }
    )
    kinds = {type(m) for m in net.encoder} | {type(m) for m in net.decoder}
    assert ExactLayerNorm in kinds and nn.Dropout in kinds
    net.eval()
    assert net(torch.zeros(5, 2)).shape == (5, 1)
    bare = AutoEncoder({"input_dim": 2, "hidden_dims": [8], "layer_norm": False})
    kinds = {type(m) for m in bare.encoder} | {type(m) for m in bare.decoder}
    assert ExactLayerNorm not in kinds and nn.Dropout not in kinds
