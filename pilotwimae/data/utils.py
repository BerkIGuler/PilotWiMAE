"""
Data utilities for PilotWiMAE training pipeline.
"""

import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_efficient_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn=None,
) -> DataLoader:
    """
    Create an efficient dataloader with multiple workers and prefetching.

    Args:
        dataset: The dataset instance
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch

    Returns:
        DataLoader instance
    """
    # Base DataLoader arguments
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
    }
    if collate_fn is not None:
        dataloader_kwargs["collate_fn"] = collate_fn
    
    if num_workers > 0:
        dataloader_kwargs.update({
            'prefetch_factor': 2,  # Number of batches to prefetch from the dataloader
            'persistent_workers': True,  # Keep workers alive between iterations
        })
    
    return DataLoader(**dataloader_kwargs)


def calculate_mean_power(dataloader: DataLoader) -> Dict[str, float]:
    """
    Calculate the mean power E[|h|^2] of complex channel data.

    Since wireless channels are approximately zero-mean, normalizing by
    sqrt(mean_power) is roughly equivalent to normalizing by complex std.

    Args:
        dataloader: PyTorch DataLoader containing complex matrices

    Returns:
        dict containing mean_power
    """
    power_sum = 0.0  # accumulate in float64 to avoid overflow/precision loss
    total_elements = 0

    for batch in tqdm(dataloader, desc="Computing mean power"):
        # Compute in float64 for numerical stability with large batches or values
        re = batch.real.to(torch.float64)
        im = batch.imag.to(torch.float64)
        power_sum += torch.sum(re * re + im * im).item()
        total_elements += batch.numel()

    mean_power = power_sum / total_elements

    return {'mean_power': mean_power}