"""
Patch embedding modules for PilotWiMAE.

Two strategies for projecting raw patch vectors into the model dimension:
  - LinearEmbedding: flatten patch -> nn.Linear  (standard ViT-style)
  - Conv3dEmbedding: nn.Conv3d with kernel_size=stride=patch_size
"""

from typing import Tuple

import torch
import torch.nn as nn


class LinearEmbedding(nn.Module):
    """Project flattened patch vectors to the model dimension with a linear layer."""

    def __init__(self, patch_dim: int, d_model: int):
        """
        Args:
            patch_dim: Dimensionality of each patch vector (2 * pt * ps * pf).
            d_model:   Target embedding dimension.
        """
        super().__init__()
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, P, patch_dim) from Patcher3D.

        Returns:
            (B, P, d_model)
        """
        return self.proj(patches)


class Conv3dEmbedding(nn.Module):
    """Project patches using a 3D convolution with kernel = stride = patch_size.

    The convolution operates on the raw complex tensor (real and imag as two
    input channels) rather than on pre-extracted patch vectors.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        d_model: int,
    ):
        """
        Args:
            patch_size: (pt, ps, pf).
            d_model:    Target embedding dimension (number of output channels).
        """
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.conv = nn.Conv3d(
            in_channels=2,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Complex tensor of shape (B, T, S, F).

        Returns:
            (B, P, d_model) where P = (T/pt)*(S/ps)*(F/pf).
        """
        # Stack real and imag as channel dim: (B, 2, T, S, F)
        x = torch.stack([H.real, H.imag], dim=1).float()
        # Conv3d -> (B, d_model, nt, ns, nf)
        x = self.conv(x)
        B, D = x.shape[0], x.shape[1]
        # Flatten spatial dims -> (B, P, d_model)
        x = x.reshape(B, D, -1).permute(0, 2, 1)
        return x
