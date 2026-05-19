from typing import Tuple
import pytest
import torch
from pilotwimae.models.modules import MaskGenerator


@pytest.fixture
def mask_ratio() -> float:
    return 0.9

@pytest.fixture
def d_model() -> int:
    return 64

@pytest.fixture
def grid_dims() -> Tuple[int, int, int]:
    patch_size = (2, 4, 4)
    return (14//patch_size[0], 32//patch_size[1], 32//patch_size[2])

@pytest.fixture
def random_mask_generator(grid_dims: Tuple[int, int, int], mask_ratio: float) -> MaskGenerator:    
    return MaskGenerator(device="cpu", mask_ratio=mask_ratio, strategy="random", grid_dims=grid_dims)

@pytest.fixture
def temporal_mask_generator(grid_dims: Tuple[int, int, int], mask_ratio: float) -> MaskGenerator:
    return MaskGenerator(device="cpu", mask_ratio=mask_ratio, strategy="temporal", grid_dims=grid_dims)

@pytest.fixture
def channel_embeddings(grid_dims: Tuple[int, int, int], d_model: int) -> torch.Tensor:
    num_patches = int(grid_dims[0] * grid_dims[1] * grid_dims[2])
    return torch.randn(512, num_patches, d_model)

def test_random_mask_generator(
    random_mask_generator: MaskGenerator, 
    channel_embeddings: torch.Tensor,
    mask_ratio: float,
    grid_dims: Tuple[int, int, int],
    d_model: int) -> None:

    num_patches = int(grid_dims[0] * grid_dims[1] * grid_dims[2])
    num_keep = int(num_patches * (1 - mask_ratio))
    unmasked, ids_keep, ids_mask = random_mask_generator(channel_embeddings)
    assert unmasked.shape == (512, num_keep, d_model)
    assert ids_keep.shape == (512, num_keep)
    assert ids_mask.shape == (512, num_patches - num_keep)
    # For each batch, kept and masked indices should be a disjoint partition of
    # {0, ..., num_patches-1}.
    for b in range(ids_keep.shape[0]):
        all_idx = torch.cat([ids_keep[b], ids_mask[b]])
        assert all_idx.numel() == num_patches
        # No duplicates
        assert torch.unique(all_idx).numel() == num_patches
        # Covers the full index set
        assert torch.equal(torch.sort(all_idx).values, torch.arange(num_patches))

def test_temporal_mask_generator(
    temporal_mask_generator: MaskGenerator, 
    channel_embeddings: torch.Tensor,
    mask_ratio: float,
    grid_dims: Tuple[int, int, int],
    d_model: int) -> None:

    nt, ns, nf = grid_dims
    patches_per_t = ns * nf
    num_t_keep = max(1, int(nt * (1 - mask_ratio)))
    expected_keep = num_t_keep * patches_per_t
    num_patches = nt * ns * nf

    unmasked, ids_keep, ids_mask = temporal_mask_generator(channel_embeddings)

    assert unmasked.shape == (512, expected_keep, d_model)
    assert ids_keep.shape == (512, expected_keep)
    assert ids_mask.shape == (512, num_patches - expected_keep)
    # For each batch, kept and masked indices should be a disjoint partition of
    # {0, ..., num_patches-1}.
    for b in range(ids_keep.shape[0]):
        all_idx = torch.cat([ids_keep[b], ids_mask[b]])
        assert all_idx.numel() == num_patches
        # No duplicates
        assert torch.unique(all_idx).numel() == num_patches
        # Covers the full index set
        assert torch.equal(torch.sort(all_idx).values, torch.arange(num_patches))