from pathlib import Path
import pytest
import torch
from typing import Tuple
import numpy as np

from pilotwimae.models.modules import (
    LearnablePositionalEncoding,
    SinusoidalConcat3D,
)
    

@pytest.fixture
def patch_size() -> Tuple[int, int, int]:
    return (2, 4, 4)

@pytest.fixture
def input_shape() -> Tuple[int, int, int]:
    return (14, 32, 32)

@pytest.fixture
def grid_dims(patch_size: Tuple[int, int, int], input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(np.array(input_shape) // np.array(patch_size))

@pytest.fixture
def d_model() -> int:
    return 64

@pytest.fixture
def channel_embeddings(grid_dims: Tuple[int, int, int], d_model: int) -> torch.Tensor:
    embedding_dim = int(grid_dims[0] * grid_dims[1] * grid_dims[2])
    return torch.randn(512, embedding_dim, d_model)

@pytest.fixture
def learnable_pos_encoding(grid_dims: Tuple[int, int, int], d_model: int) -> LearnablePositionalEncoding:
    max_len = int(grid_dims[0] * grid_dims[1] * grid_dims[2])
    return LearnablePositionalEncoding(max_len=max_len, d_model=d_model)

@pytest.fixture
def sinusoidal_concat_pos_encoding(grid_dims: Tuple[int, int, int], d_model: int) -> SinusoidalConcat3D:
    return SinusoidalConcat3D(grid_dims=grid_dims, d_model=d_model)

def test_learnable_pos_encoding(channel_embeddings: torch.Tensor, learnable_pos_encoding: LearnablePositionalEncoding) -> None:
    encoded_channel_embeddings = learnable_pos_encoding(channel_embeddings)
    assert encoded_channel_embeddings.shape == channel_embeddings.shape
    assert not torch.allclose(encoded_channel_embeddings, channel_embeddings, atol=0.01)

def test_sinusoidal_concat_pos_encoding(channel_embeddings: torch.Tensor, sinusoidal_concat_pos_encoding: SinusoidalConcat3D) -> None:
    encoded_channel_embeddings = sinusoidal_concat_pos_encoding(channel_embeddings)
    assert encoded_channel_embeddings.shape == channel_embeddings.shape
    assert not torch.allclose(encoded_channel_embeddings, channel_embeddings, atol=1e-6)