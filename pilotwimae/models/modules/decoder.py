"""
Decoder module for PilotWiMAE.

Takes encoded visible tokens, inserts learnable mask tokens at masked
positions, adds a dedicated decoder positional encoding, runs a transformer,
and projects back to the patch dimension.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    """Transformer decoder for PilotWiMAE."""

    def __init__(
        self,
        output_dim: int,
        pos_encoding: nn.Module,
        d_model: int = 256,
        nhead: int = 8,
        activation: str = "gelu",
        dropout: float = 0.1,
        num_layers: int = 4,
        norm_first: bool = False,
        dim_feedforward: int = 0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            output_dim:     Patch vector dimension (2 * pt * ps * pf).
            pos_encoding:   Positional encoding module applied to the full
                            sequence (visible + masked) before the transformer.
            d_model:        Model dimension.
            nhead:          Number of attention heads.
            activation:     Activation function name.
            dropout:        Dropout probability.
            num_layers:     Number of transformer layers.
            norm_first:     If True, use Pre-LayerNorm (LN before attention/FFN).
                            If False (default), use Post-LayerNorm.
            dim_feedforward: FFN hidden dimension (default: 4 * d_model).
            device:         Torch device.
        """
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.device = device
        if self.device is None:
            logger.warning("No device provided for Decoder, using CPU")
            self.device = torch.device("cpu")

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_encoding = pos_encoding

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

        self.linear = nn.Linear(d_model, output_dim)
        self.to(self.device)

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        ids_keep: torch.Tensor,
        orig_seq_len: int,
        *,
        return_decoded_tokens: bool = False,
    ):
        """
        Args:
            encoded_tokens: (B, P_keep, d_model) — encoder outputs for
                            visible tokens.  These already carry the
                            encoder's positional encoding.
            ids_keep:       (B, P_keep) — flat patch indices of visible tokens.
            orig_seq_len:   Total number of patches P.

        Returns:
            (B, P, output_dim)
        """
        B = encoded_tokens.shape[0]
        num_visible = encoded_tokens.shape[1]

        batch_idx = (
            torch.arange(B, device=encoded_tokens.device)
            .view(-1, 1)
            .expand(-1, num_visible)
        )

        # Initialise all positions with the learnable mask token.
        full_sequence = self.mask_token.expand(B, orig_seq_len, -1).clone()

        # Place encoder outputs at visible positions.
        full_sequence[batch_idx, ids_keep] = encoded_tokens

        # Add the decoder's positional encoding to ALL positions
        # (visible + masked) before the transformer.
        full_sequence = self.pos_encoding(full_sequence)

        decoded = self.transformer(full_sequence)
        reconstructed = self.linear(decoded)
        if return_decoded_tokens:
            return reconstructed, decoded
        return reconstructed
