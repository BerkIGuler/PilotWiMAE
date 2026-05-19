"""
Complex AWGN at a target SNR (beam-prediction / robustness tests).

SNR (dB) uses the usual linear ratio:

    SNR_lin = 10 ** (snr_db / 10) = P_s / P_n

where P_s is the **mean squared magnitude per complex element** of the signal the
SNR refers to. With ``noise_floor=True`` (default), P_s is the scalar
``signal_mean_power`` (e.g. dataset mean |h|^2). With ``noise_floor=False``, P_s is
the per-sample mean |h|^2 over all elements of that channel.

Circular complex Gaussian noise: n = n_r + j n_i with n_r, n_i i.i.d. N(0, P_n/2),
so E[|n|^2] = P_n.
"""

from __future__ import annotations

from typing import Optional

import torch

# --- AWGN: SNR(dB) → linear gain; circular complex noise ---
_SNR_DB_TO_LINEAR_BASE = 10.0
_COMPLEX_NOISE_VARIANCE_SPLIT = 2.0  # P_n split equally onto real and imaginary parts
_AWGN_RANDN_DTYPE = torch.float32


def add_complex_awgn_snr_db(
    h: torch.Tensor,
    snr_db: float,
    *,
    generator: Optional[torch.Generator] = None,
    signal_mean_power: float = 1.0,
    noise_floor: bool = True,
) -> torch.Tensor:
    """
    Add circular complex AWGN so that per-element SNR matches snr_db.

    Noise variance is chosen so that E[|n|^2] = P_s / SNR_lin with P_s the mean
    squared magnitude the SNR refers to.

    Parameters
    ----------
    h:
        Complex channel tensor (signal left at its current scale).
    snr_db:
        SNR in dB; linear SNR = 10^(snr_db/10) = P_s / P_n.
    generator:
        Optional torch.Generator for reproducible noise (device should match h.device
        when using CUDA for best reproducibility).
    signal_mean_power:
        Global P_s when ``noise_floor`` is True (strictly positive). Ignored when
        ``noise_floor`` is False.
    noise_floor:
        If True (default), use a single ``signal_mean_power`` for the whole batch
        (fixed noise floor vs. dataset/global calibration). If False, set P_s per
        batch row to the mean |h|^2 over all complex elements of that channel.
    """
    if not h.is_complex():
        raise TypeError(f"Expected complex tensor, got dtype={h.dtype}")

    snr_lin = _SNR_DB_TO_LINEAR_BASE ** (float(snr_db) / _SNR_DB_TO_LINEAR_BASE)
    if snr_lin <= 0:
        raise ValueError("snr_db must yield positive linear SNR")

    if noise_floor:
        p_s = float(signal_mean_power)
        if p_s <= 0:
            raise ValueError("signal_mean_power must be positive when noise_floor=True")
    else:
        dims = tuple(range(1, h.ndim))
        p_s = (h.abs() ** 2).mean(dim=dims, keepdim=True).clamp_min(1e-20)

    p_n = p_s / snr_lin
    sigma = (p_n / _COMPLEX_NOISE_VARIANCE_SPLIT) ** 0.5

    rand_kw: dict = {"device": h.device, "dtype": _AWGN_RANDN_DTYPE}
    if generator is not None:
        rand_kw["generator"] = generator

    noise_r = torch.randn(h.shape, **rand_kw)
    noise_i = torch.randn(h.shape, **rand_kw)
    noise = torch.complex(noise_r, noise_i) * sigma

    return h + noise.to(dtype=h.dtype)
