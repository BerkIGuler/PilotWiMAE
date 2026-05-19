"""
Inverse of the per-patch normalization used with ``training.norm_patch_loss`` in
:class:`~pilotwimae.training.pilotwimae_trainer.PilotWiMAETrainer`.

The decoder is trained against targets that are standardized **per patch** along the
last dimension (concatenated real/imag of length ``2L``). Raw patch vectors from
``patcher`` must be denormalized with **each patch's** mean/std from the reference
(clean) patches before ``inverse_patcher`` or raw-space NMSE.
"""

from __future__ import annotations

import torch


def denormalize_norm_patch_patches(
    reconstructed: torch.Tensor,
    reference_patches: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Map decoder outputs back to raw patch space.

    Matches the trainer transform (for each ``(b, p, :)`` vector):

        target' = (target - mean) / sqrt(var + eps)

    Parameters
    ----------
    reconstructed:
        ``(B, P, 2L)`` model outputs (same layout as ``patcher``).
    reference_patches:
        ``(B, P, 2L)`` clean patches; mean/var are taken along ``-1`` per patch.
    eps:
        Same role as ``training.norm_patch_loss_eps`` in config.
    """
    if reconstructed.shape != reference_patches.shape:
        raise ValueError(
            f"Shape mismatch: recon {tuple(reconstructed.shape)} vs ref {tuple(reference_patches.shape)}"
        )
    mean = reference_patches.mean(dim=-1, keepdim=True)
    var = reference_patches.var(dim=-1, unbiased=False, keepdim=True)
    std = (var + float(eps)).sqrt()
    return reconstructed * std + mean
