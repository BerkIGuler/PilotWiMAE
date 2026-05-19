"""
On-the-fly beam scoring and label extraction utilities.
"""

from typing import Tuple

import torch

from .types import LabelMode


def compute_beam_gains(
    channels: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    Compute average beam gains per OFDM symbol.

    Args:
        channels: Complex tensor with shape (T, N, K) or (B, T, N, K).
        codebook: Complex codebook tensor with shape (M, N),
            where M is the number of codewords (beams) and
            N is the codeword length (number of transmit antennas, typically N_h * N_v).

    Returns:
        Gain tensor with shape (T, M) or (B, T, M), where each value is:
            (1 / K) * sum_k |w_m^H h_{k,t}|^2
    """
    if channels.ndim == 3:
        channels = channels.unsqueeze(0)
        squeeze_batch = True
    elif channels.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError("channels must have shape (T, N, K) or (B, T, N, K).")

    if codebook.ndim != 2:
        raise ValueError("codebook must have shape (M, N).")

    if not channels.dtype.is_complex:
        raise ValueError("channels must be complex.")
    if not codebook.dtype.is_complex:
        raise ValueError("codebook must be complex.")

    _, _, n, _ = channels.shape
    _, n_codebook = codebook.shape
    if n != n_codebook:
        raise ValueError(f"Channel antenna count N={n} does not match codebook N={n_codebook}.")

    channels_btk_n = channels.permute(0, 1, 3, 2)
    # Compute per-beam complex inner products w_m^H h_{k,t} for every
    # batch item b, OFDM symbol t, and subcarrier k.
    #   codebook.conj(): (M, N) gives w_m^H along the N dimension
    #   channels_btk_n:  (B, T, K, N) gives h_{k,t}
    # Result:
    #   projections: (B, T, K, M), where projections[b,t,k,m] = w_m^H h_{k,t}.
    projections = torch.einsum("mn,btkn->btkm", codebook.conj(), channels_btk_n)

    # Convert complex projections to beamforming gains and average over
    # subcarriers (K) to obtain the wideband gain for each symbol:
    #   gain[b,t,m] = (1/K) * sum_k |w_m^H h_{k,t}|^2.
    gains = projections.abs().pow(2).mean(dim=2)

    if squeeze_batch:
        return gains.squeeze(0)
    return gains


def reduce_beam_gains(
    gains: torch.Tensor,
    *,
    label_mode: LabelMode = "sequence",
) -> torch.Tensor:
    """Reduce gain tensor to sequence-level or snapshot-level gains."""
    if gains.ndim == 2:
        gains = gains.unsqueeze(0)
        squeeze_batch = True
    elif gains.ndim == 3:
        squeeze_batch = False
    else:
        raise ValueError("gains must have shape (T, M) or (B, T, M).")

    if label_mode == "sequence":
        reduced = gains
    elif label_mode == "snapshot":
        reduced = gains.mean(dim=1)
    else:
        raise ValueError("label_mode must be one of {'sequence', 'snapshot'}.")

    if squeeze_batch:
        return reduced.squeeze(0)
    return reduced


def beam_targets_from_gains(
    gains: torch.Tensor,
    *,
    label_mode: LabelMode = "sequence",
    top_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract top-k beam indices and corresponding scores from gains."""
    reduced_gains = reduce_beam_gains(gains, label_mode=label_mode)

    if reduced_gains.ndim == 1:
        m = reduced_gains.size(0)
    elif reduced_gains.ndim in (2, 3):
        m = reduced_gains.size(-1)
    else:
        raise ValueError("Unexpected reduced gain shape.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_k > m:
        raise ValueError(f"top_k={top_k} cannot be larger than codebook size M={m}.")

    scores, indices = torch.topk(reduced_gains, k=top_k, dim=-1, largest=True, sorted=True)
    return indices.to(torch.int64), scores
