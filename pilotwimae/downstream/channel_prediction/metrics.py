"""NMSE on masked patch tokens (MAE reconstruction targets)."""

from __future__ import annotations

import torch


def nmse_on_masked(
    reconstructed: torch.Tensor,
    patches_clean: torch.Tensor,
    ids_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    NMSE aggregated over all **masked** patch elements in the batch:

        sum((r - t)^2) / max(sum(t^2), eps)

    ``reconstructed`` and ``patches_clean`` are ``(B, P, D)``; ``ids_mask`` is
    ``(B, P_mask)`` long indices into ``P``.
    """
    if reconstructed.shape != patches_clean.shape:
        raise ValueError(
            f"Shape mismatch: recon {tuple(reconstructed.shape)} vs target {tuple(patches_clean.shape)}"
        )
    B, Pm = ids_mask.shape
    if B != reconstructed.shape[0]:
        raise ValueError("Batch size mismatch between ids_mask and patches")
    dev = reconstructed.device
    batch_idx = torch.arange(B, device=dev, dtype=torch.long).unsqueeze(-1).expand(-1, Pm)
    recon_m = reconstructed[batch_idx, ids_mask]
    tgt_m = patches_clean[batch_idx, ids_mask]
    err = (recon_m - tgt_m).double()
    num = (err * err).sum()
    den = (tgt_m.double() * tgt_m.double()).sum().clamp_min(float(eps))
    return num / den
