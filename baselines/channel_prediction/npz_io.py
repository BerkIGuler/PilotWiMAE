"""NPZ discovery and channel iteration for channel-prediction baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


def build_sorted_npz_list(data_dir: Path) -> list[Path]:
    """Return NPZ paths under ``data_dir`` (recursive), sorted lexically."""
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {data_dir}")
    npz_files = sorted(data_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found under {data_dir}")
    return npz_files


def iter_channels(npz_files: Iterable[Path]) -> Iterable[np.ndarray]:
    """
    Yield channel tensors ``[T, N_a, N_f]`` (complex) from saved NPZ batches.

    Expects key ``h`` shaped ``[B, 1, 1, N, T, K]``.
    """
    for p in npz_files:
        with np.load(p) as data:
            h = data["h"][:, 0, 0, :, :, :].transpose(0, 2, 1, 3)
            for i in range(h.shape[0]):
                yield h[i]


def compute_dataset_mean_complex_power(npz_files: list[Path]) -> float:
    """Mean ``|h|^2`` over all entries and samples (used for noise-floor SNR)."""
    total = 0.0
    count = 0
    for target in iter_channels(npz_files):
        total += float(np.sum(np.abs(target) ** 2))
        count += int(target.size)
    if count == 0:
        raise ValueError("No channel samples found to compute dataset mean power")
    return total / float(count)


def compute_pref_reference_power(npz_files: list[Path], *, progress: bool = False) -> float:
    """
    Dataset-level reference power P_ref (TeX): mean over samples of (1/(T N_a N_f)) ||H||_F^2.

    Same as mean per-entry |h|^2 for each sample averaged over samples.
    """
    total = 0.0
    n_samples = 0
    ch_it = iter_channels(npz_files)
    if progress:
        ch_it = tqdm(
            ch_it,
            desc="P_ref (corr channels)",
            unit="ch",
            dynamic_ncols=True,
        )
    for H in ch_it:
        total += float(np.mean(np.abs(H) ** 2))
        n_samples += 1
    if n_samples == 0:
        raise ValueError("No channel samples for P_ref")
    return total / float(n_samples)
