from pilotwimae import PilotWiMAE
import pytest
import torch


@pytest.fixture
def config():
    return {
        "input_shape": [14, 32, 32],
        "patch_size": [2, 4, 4],
        "embedding": {"type": "linear"},
        "positional_encoding": {
            "encoder": {"type": "sinusoidal_concat"},
            "decoder": {"type": "sinusoidal_concat"},
        },
        "masking": {"strategy": "random", "mask_ratio": 0.8},
        "encoder_dim": 64,
        "encoder_layers": 2,
        "encoder_nhead": 16,
        "decoder_layers": 4,
        "decoder_nhead": 8,
    }

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def input_channels(config: dict, batch_size: int):
    input_shape = config["input_shape"]
    h_real = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])
    h_imag = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])
    return torch.complex(h_real, h_imag)

@pytest.fixture
def num_patches(config: dict):
    patch_size = config["patch_size"]
    input_shape = config["input_shape"]
    return input_shape[0] // patch_size[0] * input_shape[1] // patch_size[1] * input_shape[2] // patch_size[2]

@pytest.fixture
def patch_dim(config: dict):
    return 2 * config["patch_size"][0] * config["patch_size"][1] * config["patch_size"][2]

@pytest.fixture
def model(config: dict):
    return PilotWiMAE(config, device="cpu")

def test_pilotwimae_forward(model: PilotWiMAE, config: dict, input_channels: torch.Tensor, num_patches: int, patch_dim: int, batch_size: int):
    mask_ratio = config["masking"]["mask_ratio"]
    num_patches_keep = int(num_patches * (1 - mask_ratio))
    num_patches_mask = num_patches - num_patches_keep
    output = model(input_channels, return_reconstruction=True)
    assert output["encoded_features"].shape == (batch_size, num_patches_keep, config["encoder_dim"])
    assert output["ids_keep"].shape == (batch_size, num_patches_keep)
    assert output["ids_mask"].shape == (batch_size, num_patches_mask)
    assert output["reconstructed_patches"].shape == (batch_size, num_patches, patch_dim)


def test_pilotwimae_encode(model: PilotWiMAE, config: dict, input_channels: torch.Tensor, num_patches: int, batch_size: int):
    """encode() is for downstream use: all patch tokens, no MAE masking."""
    encoded = model.encode(input_channels)
    assert encoded.shape == (batch_size, num_patches, config["encoder_dim"])


def test_pilotwimae_get_embeddings(model: PilotWiMAE, config: dict, input_channels: torch.Tensor, num_patches: int, batch_size: int):
    mean_pooled_embeddings = model.get_embeddings(input_channels, pooling="mean")
    assert mean_pooled_embeddings.shape == (batch_size, config["encoder_dim"])
    max_pooled_embeddings = model.get_embeddings(input_channels, pooling="max")
    assert max_pooled_embeddings.shape == (batch_size, config["encoder_dim"])
    # Expect the two pooling strategies to produce different tensors for random input
    assert not torch.allclose(mean_pooled_embeddings, max_pooled_embeddings)


def test_pilotwimae_get_model_info(model: PilotWiMAE, config: dict):
    model_info = model.get_model_info()
    # Model stores shapes as tuples
    assert model_info["input_shape"] == tuple(config["input_shape"])
    assert model_info["patch_size"] == tuple(config["patch_size"])
    # Embedding and positional encoding types should match config
    assert model_info["embedding_type"] == config["embedding"]["type"]
    assert model_info["encoder_positional_encoding"] == config["positional_encoding"]["encoder"]["type"]
    assert model_info["decoder_positional_encoding"] == config["positional_encoding"]["decoder"]["type"]
    assert model_info["encoder_layers"] == config["encoder_layers"]
    assert model_info["decoder_layers"] == config["decoder_layers"]
