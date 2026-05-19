"""
Dataset implementations for PilotWiMAE.
"""

import gc
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OptimizedPreloadedDataset(Dataset):
    """Optimized dataset implementation for maximum training speed"""

    def __init__(self, npz_files: List[str],
                 statistics: Optional[Dict[str, float]] = None,
                 preload_batch_size: int = 500):
        """
        Args:
            npz_files: List of NPZ file paths
            statistics: Dict with normalization parameters. If provided, data will be normalized.
            preload_batch_size: Batch size when loading from each NPZ to limit memory spikes.
        """
        self.statistics = statistics
        self.normalize = statistics is not None
        self.preload_batch_size = preload_batch_size

        
        if not npz_files:
            raise ValueError("No NPZ files found")
        
        if self.statistics is not None:
            if "mean_power" not in self.statistics:
                raise ValueError("mean_power is required in statistics")
            elif self.statistics["mean_power"] <= 0:
                raise ValueError("mean_power must be positive")

        # shape is 1D: [num_samples, 1, 1, N, T, K]
        with np.load(npz_files[0]) as data:
            shape = data["shape"]
            assert shape.shape == (6,), "shape must be 1D with length 6"
            assert int(shape[1]) == 1, "second dimension must be 1"
            assert int(shape[2]) == 1, "third dimension must be 1"
            total_samples = int(shape[0])
            self.N, self.T, self.K = int(shape[3]), int(shape[4]), int(shape[5])

        for npz_file in npz_files[1:]:
            # Get channel counts from "shape" key (avoids loading large "h")
            with np.load(npz_file) as data:
                total_samples += int(data["shape"][0])

        logger.info(f"Total samples: {total_samples}, dimensions: {self.T}x{self.N}x{self.K}")

        self.all_data = torch.empty(
            (total_samples, self.T, self.N, self.K),  # (total_samples, time, num_tx, num_subcarriers)
            dtype=torch.complex64,
        )

        # Load all data into the pre-allocated tensor
        idx = 0
        for npz_file in tqdm(npz_files, desc="Preloading dataset files"):
            with np.load(npz_file) as data:
                file_samples = int(data["shape"][0])
                h_data = data["h"]

                # Process in batches to avoid large memory spikes
                for batch_start in range(0, file_samples, self.preload_batch_size):
                    batch_end = min(batch_start + self.preload_batch_size, file_samples)
                    batch_count = batch_end - batch_start

                    # ignore singleton axes; transpose (batch, N, T, K) -> (batch, T, N, K)
                    batch_data = torch.from_numpy(
                        h_data[batch_start:batch_end, 0, 0, :, :, :].transpose(0, 2, 1, 3)
                    )

                    if self.normalize:
                        scale = self.statistics['mean_power'] ** 0.5
                        # Division in 64-bit to avoid extra rounding when scale is very small (source may be complex64)
                        batch_data = batch_data.to(torch.complex128) / scale

                    # Store in pre-allocated tensor
                    self.all_data[idx:idx + batch_count] = batch_data
                    idx += batch_count

            # Force cleanup after each file
            gc.collect()

        logger.info(f"Successfully loaded all {total_samples} samples")

    def __len__(self) -> int:
        return self.all_data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            raise IndexError("Negative index is not supported.")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}.")
        return self.all_data[idx]
