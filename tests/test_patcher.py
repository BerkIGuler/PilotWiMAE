from pathlib import Path
import pytest
import random
import torch
from pilotwimae.data import OptimizedPreloadedDataset, create_efficient_dataloader
from pilotwimae.models.modules import Patcher3D, InversePatcher3D


@pytest.fixture
def dataloader():
    """NPZ files to load (adjust paths if your data lives elsewhere)."""
    file_path = Path("/opt/shared/datasets/CSIGen/PilotWiMAE/boston_1")
    npz_files = list(file_path.glob("*.npz"))
    if not npz_files:
        pytest.skip(f"NPZ files not found under {file_path}.")
    random.shuffle(npz_files)
    npz_files = npz_files[:1]
    dataset = OptimizedPreloadedDataset(
        npz_files=npz_files,
        statistics=None,
    )
    return create_efficient_dataloader(
        dataset,
        batch_size=512,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )

@pytest.fixture
def patcher():
    patch_size = (2, 4, 4)
    return Patcher3D(patch_size)

@pytest.fixture
def inverse_patcher():
    patch_size = (2, 4, 4)
    original_shape = (14, 32, 32)
    return InversePatcher3D(original_shape, patch_size)


def test_patcher(dataloader, patcher, inverse_patcher):
    sample_batch = next(iter(dataloader))
    H = sample_batch.squeeze()
    patches = patcher(H)
    assert H.shape == (512, 14, 32, 32)
    assert patches.shape == (512, int(14/2 * 32/4 * 32/4), 2*2*4*4)
    inverse_H = inverse_patcher(patches)
    assert inverse_H.shape == (512, 14, 32, 32)
    assert torch.allclose(H, inverse_H)