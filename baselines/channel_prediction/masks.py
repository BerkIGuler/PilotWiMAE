"""Pilot-mask helpers for channel-prediction baselines."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def pilot_time_indices(indices: Iterable[int]) -> list[int]:
    """Return sorted unique pilot time indices."""
    uniq = sorted({int(i) for i in indices})
    if not uniq:
        raise ValueError("Pilot time index list is empty")
    return uniq


def expanded_subcarrier_indices(
    patch_frequency_indices: Iterable[int],
    *,
    freq_patch_size: int,
) -> list[int]:
    """
    Expand frequency patch indices into flat subcarrier indices.

    Example:
        patch_frequency_indices=[0,2,4,6], freq_patch_size=4
        -> [0..3, 8..11, 16..19, 24..27]
    """
    if int(freq_patch_size) <= 0:
        raise ValueError("freq_patch_size must be positive")
    expanded: list[int] = []
    for p in sorted({int(v) for v in patch_frequency_indices}):
        if p < 0:
            raise ValueError("Patch frequency indices must be non-negative")
        base = p * int(freq_patch_size)
        expanded.extend(range(base, base + int(freq_patch_size)))
    if not expanded:
        raise ValueError("Expanded subcarrier index list is empty")
    return expanded


def validate_index_bounds(
    indices: Iterable[int],
    *,
    upper_bound: int,
    name: str,
) -> None:
    """Raise if any index lies outside [0, upper_bound)."""
    if int(upper_bound) <= 0:
        raise ValueError("upper_bound must be positive")
    values = [int(i) for i in indices]
    if not values:
        raise ValueError(f"{name} is empty")
    lo = min(values)
    hi = max(values)
    if lo < 0 or hi >= int(upper_bound):
        raise ValueError(f"{name} out of range [0, {upper_bound}): min={lo}, max={hi}")


def non_pilot_mask_from_pilots(
    *,
    pilot_times: Iterable[int],
    known_subcarriers: Iterable[int],
    T: int,
    N_a: int,
    N_f: int,
) -> np.ndarray:
    """
    Boolean mask ``[T, N_a, N_f]``: ``True`` on non-pilot REs (False on pilot TF for every antenna).
    """
    if int(T) <= 0 or int(N_a) <= 0 or int(N_f) <= 0:
        raise ValueError("T, N_a, N_f must be positive")
    mask = np.ones((int(T), int(N_a), int(N_f)), dtype=bool)
    pt = np.asarray(list(pilot_times), dtype=np.int64)
    fs = np.asarray(list(known_subcarriers), dtype=np.int64)
    a_ix = np.arange(int(N_a), dtype=np.int64)
    mask[np.ix_(pt, a_ix, fs)] = False
    return mask
