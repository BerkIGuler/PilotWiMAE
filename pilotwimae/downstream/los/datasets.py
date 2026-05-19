"""
LoS vs. nLoS labels from NPZ metadata, aligned with :class:`~pilotwimae.data.dataset.OptimizedPreloadedDataset`.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from torch.utils.data import Dataset

from pilotwimae.data.dataset import OptimizedPreloadedDataset


def _metadata_dict(path_str: str, data: NpzFile) -> dict[str, Any]:
    """Decode per-file NPZ ``metadata`` into a plain dict (same layout as channel NPZs with ``los_binary``)."""
    if "metadata" not in data.files:
        raise ValueError(f"{path_str}: missing 'metadata' key.")
    raw = data["metadata"]
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
        meta_obj = raw.item()
    elif isinstance(raw, dict):
        meta_obj = raw
    else:
        raise ValueError(
            f"{path_str}: 'metadata' must be a 0-d object array wrapping a dict (see NPZ writer convention)."
        )
    if not isinstance(meta_obj, dict):
        raise ValueError(f"{path_str}: metadata must decode to a dict, got {type(meta_obj).__name__}.")
    return meta_obj


def load_los_binary_labels(npz_files: Sequence[str]) -> torch.Tensor:
    """
    Load ``metadata['los_binary']`` from each NPZ in order, concatenated to match
    :class:`~pilotwimae.data.dataset.OptimizedPreloadedDataset` sample order.

    Returns:
        ``(N,)`` int64 tensor with values 0 (nLoS) and 1 (LoS).
    """
    chunks: List[np.ndarray] = []
    for path in npz_files:
        path_str = str(path)
        with np.load(path_str, allow_pickle=True) as data:
            if "shape" not in data.files:
                raise ValueError(f"{path_str}: missing 'shape' key.")
            file_samples = int(np.asarray(data["shape"]).reshape(-1)[0])
            meta = _metadata_dict(path_str, data)
            if "los_binary" not in meta:
                raise ValueError(f"{path_str}: metadata missing 'los_binary'.")
            los = np.asarray(meta["los_binary"]).astype(bool)
            if los.shape[0] != file_samples:
                raise ValueError(
                    f"{path_str}: los_binary length ({los.shape[0]}) != file_samples ({file_samples})."
                )
            chunks.append(los.astype(np.int64))
    if not chunks:
        raise ValueError("npz_files is empty.")
    stacked = np.concatenate(chunks, axis=0)
    return torch.from_numpy(stacked)


class LosBinaryLabelDataset(Dataset):
    """
    Pairs preloaded complex channels with integer LoS labels for kNN classification.

    ``__getitem__`` returns ``(channel, label)`` with ``label`` in ``{0, 1}`` (nLoS, LoS).
    """

    def __init__(self, base: OptimizedPreloadedDataset, labels: torch.Tensor) -> None:
        if len(base) != labels.shape[0]:
            raise ValueError(
                f"len(base)={len(base)} must match labels.shape[0]={labels.shape[0]}."
            )
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape {tuple(labels.shape)}.")
        self.base = base
        self.labels = labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            raise IndexError("Negative index is not supported.")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}.")
        return self.base[idx], self.labels[idx]
