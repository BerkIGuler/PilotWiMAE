from pathlib import Path
import pytest
import random
import torch
from pilotwimae.data import (
    OptimizedPreloadedDataset,
    create_efficient_dataloader,
    calculate_mean_power,
)


@pytest.fixture
def npz_file_list():
    """NPZ files to load (adjust paths if your data lives elsewhere)."""
    file_path = Path("/opt/shared/datasets/CSIGen/PilotWiMAE/boston_1")
    npz_files = list(file_path.glob("*.npz"))
    random.shuffle(npz_files)
    npz_files = npz_files[:2]  # select a few files for testing
    return npz_files


def test_dataset_stats(npz_file_list):
    """Calculate stats from raw data, reload normalized, and verify unit power / zero mean."""
    if not npz_file_list:
        pytest.skip("NPZ files not found (empty --opt/shared dataset directory).")
    missing = [f for f in npz_file_list if not Path(f).exists()]
    if missing:
        pytest.skip(f"NPZ files not found: {missing}")

    raw_dataset = OptimizedPreloadedDataset(
        npz_files=npz_file_list,
        statistics=None,
    )
    assert len(raw_dataset) > 0

    raw_loader = create_efficient_dataloader(
        raw_dataset,
        batch_size=512,
        num_workers=4,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )

    statistics = calculate_mean_power(raw_loader)

    norm_dataset = OptimizedPreloadedDataset(
        npz_files=npz_file_list,
        statistics=statistics,
    )

    norm_loader = create_efficient_dataloader(
        norm_dataset,
        batch_size=512,
        num_workers=4,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )

    avg_power = 0
    real_mean = 0
    imag_mean = 0

    for batch in norm_loader:
        avg_power += torch.mean(torch.abs(batch) ** 2)
        real_mean += torch.mean(batch.real)
        imag_mean += torch.mean(batch.imag)
    avg_power /= len(norm_loader)
    real_mean /= len(norm_loader)
    imag_mean /= len(norm_loader)

    assert torch.abs(avg_power - 1) < 5e-2
    assert torch.abs(real_mean) < 1e-2
    assert torch.abs(imag_mean) < 1e-2
