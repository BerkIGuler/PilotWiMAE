"""
Custom loss functions for PilotWiMAE training.
"""

from typing import Tuple

import torch
import torch.nn as nn


class PerSampleNMSE(nn.Module):
    """Normalized MSE averaged across samples.

    For each sample, divide the sum of squared errors over the supplied
    elements by the sum of squared targets over the same elements, then
    average across the batch. This equalizes the loss contribution of
    samples with very different absolute powers (e.g. LoS near the BS vs.
    deep-shadow far from the BS), which matters when training a decoder
    against raw (unnormalized) channel patches whose dynamic range is
    large even after global pre-normalization.

    Inputs ``pred`` and ``target`` are expected to be shaped
    ``(B, N, D)``, where ``N`` is the number of selected items per sample
    (e.g. masked patches) and ``D`` is the feature dimension. Reduction
    is performed over both ``N`` and ``D`` per sample, then averaged
    across the batch ``B``. A small ``eps`` guards against zero-power
    samples.

    The returned scalar is dimensionless and directly comparable across
    runs. ``10 * log10(loss)`` gives NMSE in dB, which is the standard
    wireless-channel reconstruction metric.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"PerSampleNMSE expects matching shapes, got {tuple(pred.shape)} "
                f"and {tuple(target.shape)}"
            )
        if pred.dim() < 2:
            raise ValueError(
                f"PerSampleNMSE expects at least 2 dims (B, ..., D), got {pred.dim()}"
            )

        reduce_dims: Tuple[int, ...] = tuple(range(1, pred.dim()))
        # Upcast to float32: under AMP autocast (fp16/bf16) the per-sample sum
        # of squared errors over hundreds of patches with raw channel values
        # can overflow fp16's 65504 limit for high-power samples.
        pred_f = pred.float()
        target_f = target.float()
        sq_err = (pred_f - target_f).pow(2).sum(dim=reduce_dims)
        pow_target = target_f.pow(2).sum(dim=reduce_dims).clamp_min(self.eps)
        return (sq_err / pow_target).mean()
