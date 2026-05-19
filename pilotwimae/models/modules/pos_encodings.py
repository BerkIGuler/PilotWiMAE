"""
Positional encoding modules for PilotWiMAE.

Provides 1D learnable encoding and a 3D sinusoidal (concat) variant that
exploits the (time, spatial, frequency) grid structure of the patch sequence.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import math


def _sinusoidal_table(
    max_len: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a (max_len, d) sinusoidal encoding table on ``device`` / ``dtype``."""
    position = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d)
    )
    pe = torch.zeros(max_len, d, device=device, dtype=dtype)
    # Even indices (0, 2, 4, ...) use all div_term entries
    pe[:, 0::2] = torch.sin(position * div_term)
    # Odd indices (1, 3, 5, ...) are fewer when d is odd; trim div_term accordingly
    num_odd = d // 2
    if num_odd > 0:
        pe[:, 1::2] = torch.cos(position * div_term[:num_odd])
    return pe


class LearnablePositionalEncoding(nn.Module):
    """Learnable 1D positional encoding indexed by flattened patch position."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        grid_dims: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """x: (B, seq_len, D) -> (B, seq_len, D) with positional encoding added.

        ``grid_dims`` is accepted for interface compatibility with sinusoidal
        PE modules but is ignored — learnable PE is indexed by flat position.
        """
        return x + self.position_embeddings[:, : x.size(1), :]


class SinusoidalConcat3D(nn.Module):
    """Three separate (D/3)-dim sinusoidal encodings, concatenated.

    When D is not divisible by 3, the frequency axis (last) gets the extra dims.

    Sinusoidal tables are built on the fly for the active ``(nt, ns, nf)`` so
    any grid size is supported (memory permitting).
    """

    def __init__(
        self,
        grid_dims: Tuple[int, int, int],
        d_model: int,
        init_scale: float = 0.01,
    ):
        """
        Args:
            grid_dims:  (nt, ns, nf) — default grid when ``forward`` omits override.
            d_model:    Embedding dimension.
            init_scale: Initial value of the learnable PE scale factor.
                        Starts small so the fixed sinusoidal values don't
                        overwhelm the embedding signal at initialization.
        """
        super().__init__()
        self.grid_dims = grid_dims
        self.d_model = d_model

        self.scale = nn.Parameter(torch.tensor(init_scale))

        d_base = d_model // 3
        self.d_t = d_base
        self.d_s = d_base
        self.d_f = d_model - d_base - d_base

    def _build_combined(
        self,
        nt: int,
        ns: int,
        nf: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build flat combined PE for the grid, shape (1, nt*ns*nf, D).

        Time-major flatten: ``idx = it * (ns*nf) + is_ * nf + if_``.
        """
        pe_t = _sinusoidal_table(nt, self.d_t, device, dtype)
        pe_s = _sinusoidal_table(ns, self.d_s, device, dtype)
        pe_f = _sinusoidal_table(nf, self.d_f, device, dtype)
        combined = torch.cat(
            [
                pe_t[:, None, None, :].expand(nt, ns, nf, -1),
                pe_s[None, :, None, :].expand(nt, ns, nf, -1),
                pe_f[None, None, :, :].expand(nt, ns, nf, -1),
            ],
            dim=-1,
        ).reshape(1, nt * ns * nf, self.d_model)
        return combined

    def forward(
        self,
        x: torch.Tensor,
        grid_dims: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, P, D) — token sequence.
            grid_dims: Override ``(nt, ns, nf)`` for this forward (any positive
                       sizes; must match ``P == nt*ns*nf``).
        """
        if grid_dims is None:
            nt, ns, nf = self.grid_dims
        else:
            nt, ns, nf = grid_dims
        pe = self._build_combined(nt, ns, nf, x.device, x.dtype)
        return x + self.scale.to(dtype=x.dtype) * pe
