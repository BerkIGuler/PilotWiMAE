"""Evaluation helpers for linear-interpolation channel-prediction baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .linear_interp import reconstruct_linear_from_pilots
from .metrics import mse, nmse


def evaluate_linear_interpolation(
    observed_grid: np.ndarray,
    target_grid: np.ndarray,
    *,
    pilot_times: list[int],
    known_subcarriers: list[int],
    frequency_axis: int = -1,
    time_axis: int = 0,
    frequency_outside_mode: str = "hold",
    time_outside_mode: str = "hold",
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Reconstruct and evaluate linear-interpolation baseline.

    Returns metrics and reconstructed grid.
    """
    recon = reconstruct_linear_from_pilots(
        observed_grid,
        pilot_times=pilot_times,
        known_subcarriers=known_subcarriers,
        frequency_axis=frequency_axis,
        time_axis=time_axis,
        frequency_outside_mode=frequency_outside_mode,
        time_outside_mode=time_outside_mode,
    )
    return {
        "mse": mse(recon, target_grid),
        "nmse": nmse(recon, target_grid, eps=eps),
        "reconstructed": recon,
    }
