"""
3D patching module for converting temporal wireless channel data to patches.

Input is a complex tensor of shape (B, T, S, F) where:
  T = time (OFDM symbols), S = spatial (TX antennas), F = frequency (subcarriers).

Each 3D sub-volume of size (pt, ps, pf) becomes one patch.
Real and imaginary parts are concatenated within each patch vector,
yielding patch vectors of length 2 * pt * ps * pf.
"""

from typing import Tuple

import torch
import numpy as np


class Patcher3D:
    """Convert a complex 3D channel tensor into a flat sequence of patch vectors."""

    def __init__(self, patch_size: Tuple[int, int, int] = (1, 4, 8)):
        if len(patch_size) != 3:
            raise ValueError(f"patch_size must be a 3-tuple (pt, ps, pf), got {patch_size}")
        self.patch_size = tuple(patch_size)

    def __call__(self, H):
        """
        Args:
            H: Complex tensor of shape (B, T, S, F) or (T, S, F).

        Returns:
            patches: Real tensor of shape (B, P, 2*L) where
                P = (T/pt)*(S/ps)*(F/pf) and L = pt*ps*pf.
        """
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H)

        squeeze_output = False
        if H.dim() == 3:
            H = H.unsqueeze(0)
            squeeze_output = True

        B, T, S, F = H.shape
        pt, ps, pf = self.patch_size

        if T % pt != 0 or S % ps != 0 or F % pf != 0:
            raise ValueError(
                f"Dimensions ({T}, {S}, {F}) must be divisible by "
                f"patch_size {self.patch_size}"
            )

        nt, ns, nf = T // pt, S // ps, F // pf
        num_patches = nt * ns * nf
        L = pt * ps * pf

        # (B, nt, pt, ns, ps, nf, pf)
        real = H.real.reshape(B, nt, pt, ns, ps, nf, pf)
        imag = H.imag.reshape(B, nt, pt, ns, ps, nf, pf)

        # -> (B, nt, ns, nf, pt, ps, pf) -> (B, P, L)
        real = real.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, num_patches, L)
        imag = imag.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, num_patches, L)

        # Concatenate real and imag within each patch: (B, P, 2*L)
        patches = torch.cat([real, imag], dim=-1)

        return patches.squeeze(0) if squeeze_output else patches


class InversePatcher3D:
    """Reconstruct a complex 3D channel tensor from a sequence of patch vectors."""

    def __init__(
        self,
        original_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int] = (1, 4, 8),
    ):
        if len(original_shape) != 3:
            raise ValueError(f"original_shape must be (T, S, F), got {original_shape}")
        if len(patch_size) != 3:
            raise ValueError(f"patch_size must be (pt, ps, pf), got {patch_size}")

        self.original_shape = tuple(original_shape)
        self.patch_size = tuple(patch_size)

        T, S, F = self.original_shape
        pt, ps, pf = self.patch_size
        if T % pt != 0 or S % ps != 0 or F % pf != 0:
            raise ValueError(
                f"original_shape ({T}, {S}, {F}) must be divisible by "
                f"patch_size {self.patch_size}"
            )

    def __call__(self, patches):
        """
        Args:
            patches: Real tensor of shape (B, P, 2*L) or (P, 2*L).

        Returns:
            H: Complex tensor of shape (B, T, S, F) or (T, S, F).
        """
        squeeze_output = False
        if patches.dim() == 2:
            patches = patches.unsqueeze(0)
            squeeze_output = True

        B, P, patch_dim = patches.shape
        T, S, F = self.original_shape
        pt, ps, pf = self.patch_size
        L = pt * ps * pf
        nt, ns, nf = T // pt, S // ps, F // pf

        if P != nt * ns * nf:
            raise ValueError(
                f"Number of patches ({P}) doesn't match expected "
                f"({nt * ns * nf})"
            )
        if patch_dim != 2 * L:
            raise ValueError(
                f"Patch dim ({patch_dim}) doesn't match expected "
                f"2*L = {2 * L}"
            )

        real = patches[:, :, :L]
        imag = patches[:, :, L:]

        # (B, P, L) -> (B, nt, ns, nf, pt, ps, pf)
        real = real.reshape(B, nt, ns, nf, pt, ps, pf)
        imag = imag.reshape(B, nt, ns, nf, pt, ps, pf)

        # -> (B, nt, pt, ns, ps, nf, pf) -> (B, T, S, F)
        real = real.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, S, F)
        imag = imag.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, S, F)

        H = torch.complex(real, imag)
        return H.squeeze(0) if squeeze_output else H
