"""
Complex AWGN for noise-robust training.

SNR is defined per batch element relative to that sample's mean channel power
``mean(|h|^2)`` over (T, S, F).
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def snr_min_db_for_local_epoch(
    local_epoch: int, total_epochs: int, snr_start_db: float
) -> float:
    """
    Cosine curriculum for the lower bound of per-step SNR sampling.

    At local epoch 0, returns ``snr_start_db``. At the last local epoch
    (``total_epochs - 1``), returns 0 dB so sampling is over ``[0, snr_max]``
    when ``snr_max == snr_start_db``.

    Uses ``t = local_epoch / max(1, total_epochs - 1)`` so the endpoint is hit
    on the final discrete epoch.
    """
    if total_epochs <= 1:
        return float(snr_start_db)
    t = local_epoch / (total_epochs - 1)
    return (float(snr_start_db) / 2.0) * (1.0 + math.cos(math.pi * t))


def uniform_snr_db_per_sample(
    batch_size: int,
    low_db: float,
    high_db: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Shape (B,) uniform in [low_db, high_db] on ``device`` (float32)."""
    out = torch.empty(batch_size, device=device, dtype=torch.float32)
    out.uniform_(float(low_db), float(high_db), generator=generator)
    return out


def awgn_complex_channel(
    x: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    power_eps: float = 1e-12,
) -> torch.Tensor:
    """
    Add complex circular AWGN to ``x``.

    Args:
        x: Complex ``(B, T, S, F)``.
        snr_db: Float tensor ``(B,)`` — SNR in dB per sample.
        power_eps: Floor on mean power to avoid division by zero.

    Returns:
        ``x + n`` with same dtype and device as ``x``.
        Per sample ``b``, i.i.d. noise per element has
        ``E|n_{t,s,f}|^2 = mean(|x[b]|^2) / 10^(snr_db[b]/10)``.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x (B,T,S,F), got dim={x.dim()} shape={tuple(x.shape)}")
    if not x.is_complex():
        raise ValueError("x must be complex dtype")
    B = x.shape[0]
    if snr_db.shape != (B,):
        raise ValueError(f"snr_db must be (B,), got {tuple(snr_db.shape)} for B={B}")

    dtype_compute = torch.float32
    xr = x.real.to(dtype_compute)
    xi = x.imag.to(dtype_compute)
    mag2 = xr * xr + xi * xi
    P = mag2.view(B, -1).mean(dim=1).clamp_min(float(power_eps))

    snr_db_f = snr_db.to(device=x.device, dtype=dtype_compute)
    snr_lin = 10.0 ** (snr_db_f / 10.0)
    sigma2 = P / snr_lin
    # E|n|^2 = sigma2 per element: real/imag each variance sigma2/2
    sigma = torch.sqrt(sigma2 / 2.0).view(B, 1, 1, 1)

    nr = torch.randn_like(xr)
    ni = torch.randn_like(xi)
    noise = torch.complex(nr * sigma, ni * sigma)
    x_c = torch.complex(xr, xi)
    return (x_c + noise).to(x.dtype)
