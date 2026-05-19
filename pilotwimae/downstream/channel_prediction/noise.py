"""AWGN on pilot patch tokens (real/imag concat patch vectors)."""

from __future__ import annotations

from typing import Optional

import torch

# Match pilotwimae.data.beam.channel_noise (SNR definition).
_SNR_DB_TO_LINEAR_BASE = 10.0
_COMPLEX_NOISE_VARIANCE_SPLIT = 2.0
_AWGN_RANDN_DTYPE = torch.float32


def corrupt_pilot_patches(
    patches: torch.Tensor,
    pilot_flat_keep: torch.Tensor,
    snr_db: float,
    signal_mean_power: Optional[float] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Copy ``patches`` and add circular complex AWGN to the **pilot** patch rows only.

    Each patch row is ``[Re(flat), Im(flat)]`` with ``flat`` length ``L`` (same as
    :class:`~pilotwimae.models.modules.patching.Patcher3D`).

    If ``signal_mean_power`` is provided, one global ``P_s`` is used for all pilot
    rows so ``P_n = P_s / SNR`` is identical everywhere (fixed noise floor).
    If ``signal_mean_power`` is None, ``P_s`` is one scalar per batch sample: the
    mean of ``|h|^2`` over all pilot patches and patch elements for that channel.

    Parameters
    ----------
    patches:
        ``(B, P, 2*L)`` real tensor.
    pilot_flat_keep:
        1D long indices into ``P`` (pilot patch indices).
    signal_mean_power:
        Optional global mean complex power ``P_s`` for fixed-floor AWGN.
    """
    if patches.dim() != 3:
        raise ValueError(f"Expected patches (B,P,2L), got shape {tuple(patches.shape)}")
    patch_dim = patches.shape[-1]
    if patch_dim % 2 != 0:
        raise ValueError(f"Last dim must be even (2*L), got {patch_dim}")
    L = patch_dim // 2

    out = patches.clone()
    idx = torch.unique(
        pilot_flat_keep.to(device=patches.device, dtype=torch.long).reshape(-1),
        sorted=False,
    )
    if idx.numel() == 0:
        return out
    p_max = int(patches.shape[1])
    if int(idx.min()) < 0 or int(idx.max()) >= p_max:
        raise ValueError(f"pilot_flat_keep out of range for P={p_max}")

    sel = out[:, idx, :]
    real = sel[..., :L]
    imag = sel[..., L:]
    c = torch.complex(real, imag)
    if signal_mean_power is not None:
        p_s_scalar = float(signal_mean_power)
        if p_s_scalar <= 0:
            raise ValueError("signal_mean_power must be positive when provided")
        p_s = torch.full(
            (c.shape[0], c.shape[1], 1),
            p_s_scalar,
            device=c.device,
            dtype=c.real.dtype,
        )
    else:
        # One P_s per batch row: mean |h|^2 over pilot patches and complex elements.
        p_s = (c.abs() ** 2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-20)
    snr_lin = _SNR_DB_TO_LINEAR_BASE ** (float(snr_db) / _SNR_DB_TO_LINEAR_BASE)
    p_n = p_s / snr_lin
    sigma = (p_n / _COMPLEX_NOISE_VARIANCE_SPLIT).sqrt()

    rand_kw: dict = {"device": c.device, "dtype": _AWGN_RANDN_DTYPE}
    if generator is not None:
        rand_kw["generator"] = generator
    nr = torch.randn(c.shape, **rand_kw)
    ni = torch.randn(c.shape, **rand_kw)
    n = torch.complex(nr, ni).to(dtype=c.dtype) * sigma.to(dtype=c.dtype)
    c_n = c + n

    out[:, idx, :L] = c_n.real
    out[:, idx, L:] = c_n.imag
    return out
