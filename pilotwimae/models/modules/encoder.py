"""
Encoder module for PilotWiMAE.

Receives already-embedded and position-encoded patch tokens, optionally
masks them, and processes through a transformer encoder stack.

Includes:
  - Encoder: standard transformer encoder (random or temporal masking).
  - FactorizedEncoder: alternating temporal / spatial attention for factorized masking.
"""

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Transformer encoder for PilotWiMAE."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 16,
        activation: str = "gelu",
        dropout: float = 0.1,
        num_layers: int = 12,
        norm_first: bool = False,
        dim_feedforward: int = 0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        if self.device is None:
            logger.warning("No device provided for Encoder, using CPU")
            self.device = torch.device("cpu")

        ff_dim = dim_feedforward if dim_feedforward > 0 else 4 * d_model

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            activation=activation,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, d_model) — embedded + pos-encoded tokens.

        Returns:
            encoded_tokens (B, P, d_model)
        """
        return self.transformer(x)


class FactorizedEncoder(nn.Module):
    """
    Factorized transformer encoder with alternating temporal / spatial attention.

    Each of the ``num_blocks`` block-pairs applies temporal then spatial attention.
    With ``enable_cross_dim_mixing=True``, each block is: temporal attention,
    spatial softmax mixing, spatial attention, temporal softmax mixing (see init).

    Args:
        d_model:           Token embedding dimension.
        nhead:             Number of attention heads.
        num_blocks:        Number of (temporal, spatial) block-pairs.
        num_time_keep:     ``Tk`` — default kept time indices (masked MAE / tube path).
        num_spatial_keep:  ``Sk`` — default kept spatial patches. At inference with the
                           full patch grid, pass ``time_steps`` / ``spatial_steps`` to
                           :meth:`forward` instead (same weights, different ``T×S``).
        dim_feedforward:   Feed-forward hidden dimension (default: ``2 * d_model``).
        dropout:           Dropout rate.
        activation:        Activation function (``"relu"`` or ``"gelu"``).
        device:            Torch device.
        enable_cross_dim_mixing: If True, after each temporal attention apply spatial
            softmax mixing, and after each spatial attention apply temporal mixing (each
            block has its own pair of learned ``w`` vectors in ``ℝ^D``). If False,
            behavior matches the original temporal-then-spatial stack only.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        num_blocks: int = 3,
        num_time_keep: int = 4,
        num_spatial_keep: int = 12,
        dim_feedforward: int = 0,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
        device: Optional[torch.device] = None,
        enable_cross_dim_mixing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_time_keep = num_time_keep
        self.num_spatial_keep = num_spatial_keep
        self.enable_cross_dim_mixing = enable_cross_dim_mixing
        self.device = device
        if self.device is None:
            logger.warning("No device provided for FactorizedEncoder, using CPU")
            self.device = torch.device("cpu")

        ff_dim = dim_feedforward if dim_feedforward > 0 else 4 * d_model

        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(num_blocks)
        ])

        self.spatial_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(num_blocks)
        ])

        if enable_cross_dim_mixing:
            self.spatial_mix_weights = nn.ParameterList(
                [nn.Parameter(torch.empty(d_model)) for _ in range(num_blocks)]
            )
            self.temporal_mix_weights = nn.ParameterList(
                [nn.Parameter(torch.empty(d_model)) for _ in range(num_blocks)]
            )
            for w in self.spatial_mix_weights:
                nn.init.normal_(w, std=0.02)
            for w in self.temporal_mix_weights:
                nn.init.normal_(w, std=0.02)

        self.to(self.device)

    @staticmethod
    def _mix_along_spatial(h: torch.Tensor, w_vec: torch.Tensor) -> torch.Tensor:
        """Weighted mean over spatial axis Sk, residual broadcast to each position."""
        # h: (B, Tk, Sk, D), w_vec: (D,) -> scores (B, Tk, Sk)
        scores = torch.einsum("btsd,d->bts", h, w_vec)
        alpha = F.softmax(scores, dim=-1)
        h_bar = torch.einsum("bts,btsd->btd", alpha, h)
        return h + h_bar.unsqueeze(2)

    @staticmethod
    def _mix_along_temporal(h: torch.Tensor, w_vec: torch.Tensor) -> torch.Tensor:
        """Weighted mean over temporal axis Tk, residual broadcast to each time step."""
        scores = torch.einsum("btsd,d->bts", h, w_vec)
        alpha = F.softmax(scores, dim=1)
        h_bar = torch.einsum("bts,btsd->bsd", alpha, h)
        return h + h_bar.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_steps: Optional[int] = None,
        spatial_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, T*S, D)`` token sequence (typically ``Tk*Sk`` after MAE masking).
            time_steps, spatial_steps: If both set, reshape as ``(B, T, S, D)`` with these
                grid sizes (e.g. full patch grid ``nt``, ``ns*nf`` at inference). If omitted,
                use ``num_time_keep`` and ``num_spatial_keep`` from init (training / masked path).

        Returns:
            ``(B, T*S, D)`` — encoded tokens.
        """
        B, seq_len, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")
        explicit_grid = time_steps is not None and spatial_steps is not None
        if explicit_grid:
            Tk = int(time_steps)
            Sk = int(spatial_steps)
        elif time_steps is None and spatial_steps is None:
            Tk = self.num_time_keep
            Sk = self.num_spatial_keep
        else:
            raise ValueError("Pass both time_steps and spatial_steps, or neither.")

        if seq_len != Tk * Sk:
            if explicit_grid:
                if Tk > 0 and seq_len % Tk == 0:
                    sk_new = seq_len // Tk
                    warnings.warn(
                        f"FactorizedEncoder: explicit time_steps={Tk}, spatial_steps={Sk} "
                        f"gives Tk*Sk={Tk * Sk} but seq_len={seq_len}; using spatial_steps={sk_new} "
                        f"(seq_len / time_steps).",
                        UserWarning,
                        stacklevel=2,
                    )
                    Sk = sk_new
                elif Sk > 0 and seq_len % Sk == 0:
                    tk_new = seq_len // Sk
                    warnings.warn(
                        f"FactorizedEncoder: explicit time_steps={Tk}, spatial_steps={Sk} "
                        f"gives Tk*Sk={Tk * Sk} but seq_len={seq_len}; using time_steps={tk_new} "
                        f"(seq_len / spatial_steps).",
                        UserWarning,
                        stacklevel=2,
                    )
                    Tk = tk_new
                else:
                    raise ValueError(
                        f"Expected seq_len={Tk*Sk} (Tk*Sk), got {seq_len}; "
                        f"cannot infer factors from time_steps={Tk}, spatial_steps={Sk}"
                    )
            elif Tk > 0 and seq_len > 0 and seq_len % Tk == 0:
                Sk = seq_len // Tk
                warnings.warn(
                    f"FactorizedEncoder: seq_len={seq_len} differs from "
                    f"num_time_keep*num_spatial_keep={self.num_time_keep * self.num_spatial_keep}; "
                    f"using spatial_steps={Sk} = seq_len / num_time_keep (inference layout).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    f"Expected seq_len={Tk * Sk} (num_time_keep*num_spatial_keep) or "
                    f"a multiple of num_time_keep={Tk}, got seq_len={seq_len}"
                )

        # Reshape to structured grid: (B, Tk, Sk, D)
        h = x.view(B, Tk, Sk, self.d_model)

        for bi, (t_layer, s_layer) in enumerate(
            zip(self.temporal_layers, self.spatial_layers)
        ):
            # --- temporal attention ---
            # Merge spatial into batch: (B*Sk, Tk, D)
            h_t = h.permute(0, 2, 1, 3).reshape(B * Sk, Tk, self.d_model)
            h_t = t_layer(h_t)
            h = h_t.reshape(B, Sk, Tk, self.d_model).permute(0, 2, 1, 3)  # (B, Tk, Sk, D)

            if self.enable_cross_dim_mixing:
                h = self._mix_along_spatial(h, self.spatial_mix_weights[bi])

            # --- spatial attention ---
            # Merge time into batch: (B*Tk, Sk, D)
            h_s = h.reshape(B * Tk, Sk, self.d_model)
            h_s = s_layer(h_s)
            h = h_s.reshape(B, Tk, Sk, self.d_model)

            if self.enable_cross_dim_mixing:
                h = self._mix_along_temporal(h, self.temporal_mix_weights[bi])

        return h.reshape(B, Tk * Sk, self.d_model)
