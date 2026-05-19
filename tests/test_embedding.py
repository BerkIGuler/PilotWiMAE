from pilotwimae.models.modules import LinearEmbedding, Conv3dEmbedding
import pytest
from typing import Tuple
import torch

@pytest.fixture
def patch_size() -> Tuple[int, int, int]:
    return (2, 4, 4)

@pytest.fixture
def patch_dim(patch_size: Tuple[int, int, int]) -> int:
    patch_dim = 2 * patch_size[0] * patch_size[1] * patch_size[2]
    return patch_dim

@pytest.fixture
def num_patches(patch_size: Tuple[int, int, int]) -> int:
    input_shape = (14, 32, 32)
    num_patches = input_shape[0] // patch_size[0] * input_shape[1] // patch_size[1] * input_shape[2] // patch_size[2]
    return num_patches

@pytest.fixture
def d_model() -> int:
    return 64

@pytest.fixture
def channels() -> torch.Tensor:
    input_shape = (14, 32, 32)
    batch_size = 512
    h_real = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])
    h_imag = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])
    return torch.complex(h_real, h_imag)

@pytest.fixture
def channel_patches(num_patches: int, patch_dim: int) -> torch.Tensor:
    batch_size = 512
    patches = torch.randn(batch_size, num_patches, patch_dim)
    return patches


@pytest.fixture
def linear_embedding(patch_dim: int, d_model: int) -> LinearEmbedding:
    return LinearEmbedding(patch_dim=patch_dim, d_model=d_model)

@pytest.fixture
def conv3d_embedding(patch_size: Tuple[int, int, int], d_model: int) -> Conv3dEmbedding:
    return Conv3dEmbedding(patch_size=patch_size, d_model=d_model)


def test_linear_embedding(
    linear_embedding: LinearEmbedding,
    channel_patches: torch.Tensor,
    num_patches: int,
    d_model: int,
) -> None:
    patches = linear_embedding(channel_patches)

    # Shape: (B, P, d_model)
    B = channel_patches.shape[0]
    assert patches.shape == (B, num_patches, d_model)

    # Dtype (linear preserves float by default)
    assert patches.dtype == channel_patches.dtype

    # Finite
    assert patches.isfinite().all(), "Linear embedding output should be finite"

    # Non-zero for random input
    assert not torch.allclose(patches, torch.zeros_like(patches))


def test_conv3d_embedding(
    conv3d_embedding: Conv3dEmbedding,
    channels: torch.Tensor,
    num_patches: int,
    d_model: int,
) -> None:
    patches = conv3d_embedding(channels)

    # Shape: (B, P, d_model)
    B = channels.shape[0]
    assert patches.shape == (B, num_patches, d_model)

    # Output is float (Conv3d on stacked real/imag)
    assert patches.dtype in (torch.float32, torch.float16)

    # Finite
    assert patches.isfinite().all(), "Conv3d embedding output should be finite"

    # Non-zero for random input
    assert not torch.allclose(patches, torch.zeros_like(patches))