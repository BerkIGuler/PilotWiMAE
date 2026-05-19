import pytest
import torch
from pilotwimae.models.modules import Decoder, SinusoidalConcat3D

@pytest.fixture
def batch_size() -> int:
    # Keep small to avoid OOM (SIGKILL) on CI / constrained hosts.
    return 8

@pytest.fixture
def encoded_tokens(
    batch_size: int,
    num_patches: int,
    d_model_decoder: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    mask_ratio = 0.9
    num_patches_keep = int(num_patches * (1 - mask_ratio))
    ids_keep = torch.randint(0, num_patches, (batch_size, num_patches_keep))
    orig_seq_len = num_patches
    tokens = torch.randn(batch_size, num_patches_keep, d_model_decoder)
    return tokens, ids_keep, orig_seq_len

@pytest.fixture
def num_patches() -> int:
    patch_size = (2, 4, 4)
    input_shape = (14, 32, 32)
    num_patches = input_shape[0] // patch_size[0] * input_shape[1] // patch_size[1] * input_shape[2] // patch_size[2]
    return num_patches

@pytest.fixture
def patch_dim() -> int:
    patch_size = (2, 4, 4)
    patch_dim = 2 * patch_size[0] * patch_size[1] * patch_size[2]
    return patch_dim

@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")

@pytest.fixture
def d_model_decoder() -> int:
    return 64

@pytest.fixture
def nhead_decoder() -> int:
    return 8

@pytest.fixture
def num_layers_decoder() -> int:
    return 4

@pytest.fixture
def grid_dims() -> tuple:
    patch_size = (2, 4, 4)
    input_shape = (14, 32, 32)
    return (input_shape[0] // patch_size[0], input_shape[1] // patch_size[1], input_shape[2] // patch_size[2])

@pytest.fixture
def pos_encoding(grid_dims: tuple, d_model_decoder: int) -> SinusoidalConcat3D:
    return SinusoidalConcat3D(grid_dims=grid_dims, d_model=d_model_decoder)

@pytest.fixture
def decoder(d_model_decoder: int, nhead_decoder: int, num_layers_decoder: int, device: torch.device, patch_dim: int, pos_encoding: SinusoidalConcat3D) -> Decoder:
    return Decoder(output_dim=patch_dim, pos_encoding=pos_encoding, d_model=d_model_decoder, nhead=nhead_decoder, num_layers=num_layers_decoder, device=device)

def test_decoder(
    decoder: Decoder,
    encoded_tokens: tuple[torch.Tensor, torch.Tensor, int],
    batch_size: int,
    patch_dim: int,
) -> None:
    tokens, ids_keep, orig_seq_len = encoded_tokens
    reconstructed_tokens = decoder(tokens, ids_keep, orig_seq_len)

    # Shape: (B, P, output_dim)
    assert reconstructed_tokens.shape == (batch_size, orig_seq_len, patch_dim)

    # Dtype
    assert reconstructed_tokens.dtype == tokens.dtype

    # No NaN/Inf
    assert reconstructed_tokens.isfinite().all(), "Decoder output should be finite"

    # Decoder is not identity (projection + transformer change the values)
    assert not torch.allclose(
        reconstructed_tokens, torch.zeros_like(reconstructed_tokens)
    ), "Decoder should produce non-zero output"