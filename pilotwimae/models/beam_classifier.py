"""
Supervised beam / LoS / nLoS classification: PilotWiMAE encoder (standard or
factorized), **mean-pooled** token embeddings, linear classifier (cross-entropy).

Training always uses the **full patch grid** (every token, no MAE masking): see
:func:`~pilotwimae.models.encoder_backbone.build_supervised_encoder`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from pilotwimae.data.beam import upa_2d_dft_num_beams

from .encoder_backbone import (
    build_supervised_encoder,
    build_tokenizer_modules,
    is_factorized_family,
    tokens_from_input,
)

logger = logging.getLogger(__name__)


class PilotWiMAEBeamClassifier(nn.Module):
    """
    Encoder matches the MAE tokenizer + transformer encoder + linear head trained
    on pooled token embeddings from the **full patch grid** (no masking).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.device = device
        if self.device is None:
            logger.warning("No device provided for PilotWiMAEBeamClassifier, defaulting to CPU")
            self.device = torch.device("cpu")

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        d_model = int(config.get("encoder_dim"))
        self.encoder_dim = d_model
        self.encoder_type = config.get("encoder_type")
        self.norm_first = bool(config.get("norm_first", False))
        ffn_factor = int(config.get("ffn_factor", 4))
        self.dim_feedforward = ffn_factor * d_model

        tok = build_tokenizer_modules(config)
        self.patcher = tok["patcher"]
        self.embedding = tok["embedding"]
        self.pos_encoding = tok["pos_encoding"]
        self.embedding_type = tok["embedding_type"]
        self.grid_dims = tok["grid_dims"]
        self.num_patches = tok["num_patches"]
        self.patch_dim = tok["patch_dim"]
        self.input_shape = tok["input_shape"]
        self.patch_size = tok["patch_size"]
        # Supervised beam classifier always trains on full-grid encoder outputs (no masking).
        self.encoder = build_supervised_encoder(
            config,
            self.device,
            dim_feedforward=self.dim_feedforward,
            norm_first=self.norm_first,
            d_model=d_model,
        )

        self.classifier = nn.Linear(d_model, num_classes)
        self.num_classes = num_classes

        pe_cfg = config["positional_encoding"]
        self.encoder_pos_encoding_type = pe_cfg["encoder"]["type"]
        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        grid_dims: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Complex (B,T,S,F) -> logits (B, num_classes)."""
        pooled = self.get_embeddings(x, pooling="mean", grid_dims=grid_dims)
        return self.classifier(pooled)

    def encode(
        self,
        x: torch.Tensor,
        grid_dims: Optional[Tuple[int, int, int]] = None,
        *,
        token_mode: str = "full_grid",
        pilot_flat_keep: Optional[torch.Tensor] = None,
        pilot_factorized_grid: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Encoder output ``(B, P', D)``.

        ``token_mode="full_grid"``: all patches (training layout).

        ``token_mode="pilot_visible"``: same gather + factorized layout as MAE
        :meth:`~pilotwimae.models.base.PilotWiMAE.encode` for kNN / eval.
        """
        if token_mode not in {"full_grid", "pilot_visible"}:
            if token_mode == "masked_visible":
                raise ValueError(
                    "token_mode='masked_visible' is only supported on MAE "
                    "(model.type=pilotwimae) checkpoints."
                )
            raise ValueError(
                f"Unsupported token_mode for supervised encoder: {token_mode}. "
                "Use 'full_grid' or 'pilot_visible'."
            )
        if token_mode == "pilot_visible" and pilot_flat_keep is None:
            raise ValueError("pilot_flat_keep is required when token_mode='pilot_visible'")
        if token_mode != "pilot_visible" and pilot_flat_keep is not None:
            raise ValueError("pilot_flat_keep is only valid when token_mode='pilot_visible'")
        if pilot_factorized_grid is not None and token_mode != "pilot_visible":
            raise ValueError("pilot_factorized_grid is only valid when token_mode='pilot_visible'")

        if token_mode == "pilot_visible":
            # Inference-style subset encoding (kNN eval); matches MAE pilot_visible path.
            with torch.no_grad():
                tokens = tokens_from_input(
                    self.embedding_type,
                    self.patcher,
                    self.embedding,
                    self.pos_encoding,
                    x,
                    grid_dims=grid_dims,
                )
                idx = pilot_flat_keep.to(device=self.device, dtype=torch.long).reshape(-1)
                B, P, D = tokens.shape
                if idx.numel() == 0:
                    raise ValueError("pilot_flat_keep must be non-empty")
                if int(idx.max()) >= P or int(idx.min()) < 0:
                    raise ValueError(
                        f"pilot_flat_keep indices out of range for P={P} tokens "
                        f"(min={int(idx.min())}, max={int(idx.max())})"
                    )
                gather_ix = idx.unsqueeze(0).expand(B, -1)
                visible = torch.gather(
                    tokens, dim=1, index=gather_ix.unsqueeze(-1).expand(B, -1, D)
                )
                seq_keep = visible.shape[1]
                if is_factorized_family(self.encoder_type):
                    if pilot_factorized_grid is None:
                        raise ValueError(
                            "pilot_factorized_grid=(Tk, Sk) is required for factorized "
                            "encoder with token_mode='pilot_visible'."
                        )
                    tk_e, sk_e = int(pilot_factorized_grid[0]), int(pilot_factorized_grid[1])
                    if tk_e * sk_e != seq_keep:
                        raise ValueError(
                            f"pilot_factorized_grid Tk*Sk={tk_e * sk_e} != gathered length {seq_keep}"
                        )
                    return self.encoder(
                        visible, time_steps=tk_e, spatial_steps=sk_e
                    )
                return self.encoder(visible)

        tokens = tokens_from_input(
            self.embedding_type,
            self.patcher,
            self.embedding,
            self.pos_encoding,
            x,
            grid_dims=grid_dims,
        )
        gd = grid_dims if grid_dims is not None else self.grid_dims
        nt, ns, nf = gd
        if is_factorized_family(self.encoder_type):
            return self.encoder(
                tokens, time_steps=int(nt), spatial_steps=int(ns * nf)
            )
        return self.encoder(tokens)

    def get_embeddings(
        self,
        x: torch.Tensor,
        pooling: str = "mean",
        grid_dims: Optional[Tuple[int, int, int]] = None,
        *,
        token_mode: str = "full_grid",
        pilot_flat_keep: Optional[torch.Tensor] = None,
        pilot_factorized_grid: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Mean (or max) pool over encoded patch tokens (full grid or pilot subset)."""
        encoded = self.encode(
            x,
            grid_dims=grid_dims,
            token_mode=token_mode,
            pilot_flat_keep=pilot_flat_keep,
            pilot_factorized_grid=pilot_factorized_grid,
        )
        if pooling == "mean":
            return encoded.mean(dim=1)
        if pooling == "max":
            return encoded.max(dim=1)[0]
        raise ValueError(f"Unsupported pooling: {pooling}")

    def save_checkpoint(self, filepath: str, **kwargs):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "num_classes": self.num_classes,
            **kwargs,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str,
        device: Optional[torch.device] = None,
        full_config: Optional[Dict[str, Any]] = None,
    ) -> "PilotWiMAEBeamClassifier":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        cfg = full_config or checkpoint.get("config")
        if isinstance(cfg, dict) and "model" in cfg and isinstance(cfg["model"], dict):
            model_config = cfg["model"]
        elif isinstance(cfg, dict) and "encoder_type" in cfg:
            model_config = cfg
        else:
            raise KeyError("Expected full config with ['model'] or model-only dict in checkpoint.")

        n_cls = checkpoint.get("num_classes")
        if n_cls is None and isinstance(cfg, dict):
            bp = cfg.get("task", {}).get("beam_prediction")
            if isinstance(bp, dict):
                n_cls = upa_2d_dft_num_beams(
                    int(bp["n_h"]),
                    int(bp["n_v"]),
                    o_h=int(bp.get("o_h", 1)),
                    o_v=int(bp.get("o_v", 1)),
                    u_h=int(bp.get("u_h", 1)),
                    u_v=int(bp.get("u_v", 1)),
                )
            else:
                tlos = cfg.get("task", {}).get("los")
                if isinstance(tlos, dict):
                    n_cls = int(tlos.get("num_classes", 2))
        num_classes = int(n_cls) if n_cls is not None else 0
        if num_classes < 2:
            raise KeyError(
                "Could not infer num_classes: add to checkpoint or provide "
                "task.beam_prediction / task.los."
            )

        model = cls(model_config, num_classes=num_classes, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
