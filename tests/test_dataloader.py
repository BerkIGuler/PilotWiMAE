from pathlib import Path
import pytest
import random
import time
from pilotwimae.data import OptimizedPreloadedDataset, create_efficient_dataloader


@pytest.fixture
def npz_file_list():
    """NPZ files to load (adjust paths if your data lives elsewhere)."""
    file_path = Path("/opt/shared/datasets/CSIGen/PilotWiMAE/boston_1")
    npz_files = list(file_path.glob("*.npz"))
    random.shuffle(npz_files)
    npz_files = npz_files[:2]  # select a few files for testing
    return npz_files


def test_dataloader_load_and_iterate(npz_file_list):
    """Load dataset, wrap with create_efficient_dataloader, and iterate over batches."""
    if not npz_file_list:
        pytest.skip("NPZ files not found (empty --opt/shared dataset directory).")
    missing = [f for f in npz_file_list if not Path(f).exists()]
    if missing:
        pytest.skip(f"NPZ files not found: {missing}")

    dataset = OptimizedPreloadedDataset(
        npz_files=npz_file_list,
        statistics=None,
    )
    assert len(dataset) > 0

    # num_workers=0 so the first batch is fast (no worker spawn delay)
    dataloader = create_efficient_dataloader(
        dataset,
        batch_size=512,
        num_workers=4,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )

    time_start = time.time()
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        # batch shape: (batch_size, T, N, K)
        assert batch.dim() == 4
        assert batch.dtype.is_complex
    time_end = time.time()
    total_load_time = time_end - time_start
    load_time_per_batch = total_load_time / batch_count
    assert load_time_per_batch < 0.05  # seconds
    assert batch_count >= 1
