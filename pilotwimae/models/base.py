"""
PilotWiMAE — Temporal Wireless Masked Autoencoder.

Config-driven 3D masked autoencoder for wireless channel data of shape
(B, T, S, F) complex, where T = time, S = spatial (TX antennas),
F = frequency (subcarriers).
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from .encoder_backbone import (
    build_mae_encoder_and_mask,
    build_pos_encoding,
    build_tokenizer_modules,
    encode_mae_tokens,
    is_factorized_family,
    tokens_from_input,
)
from .modules import Decoder
from .beam_classifier import PilotWiMAEBeamClassifier


class PilotWiMAE(nn.Module):
    """
    Temporal Wireless Masked Autoencoder constructed from a config dictionary.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.device = device
        if self.device is None:
            logger.warning("No device provided for PilotWiMAE, defaulting to CPU")
            self.device = torch.device("cpu")

        d_model = int(config.get("encoder_dim"))
        self.encoder_dim = d_model
        self.encoder_nhead = int(config.get("encoder_nhead"))
        self.encoder_layers = int(config.get("encoder_layers"))

        self.decoder_dim = d_model
        self.decoder_nhead = int(config.get("decoder_nhead"))
        self.decoder_layers = int(config.get("decoder_layers"))

        self.encoder_type = config.get("encoder_type")
        self.norm_first = bool(config.get("norm_first", False))
        ffn_factor = int(config.get("ffn_factor", 4))
        self.dim_feedforward = ffn_factor * d_model

        tok = build_tokenizer_modules(config)
        self.patcher = tok["patcher"]
        self.inverse_patcher = tok["inverse_patcher"]
        self.embedding = tok["embedding"]
        self.pos_encoding = tok["pos_encoding"]
        self.embedding_type = tok["embedding_type"]
        self.grid_dims = tok["grid_dims"]
        self.num_patches = tok["num_patches"]
        self.patch_dim = tok["patch_dim"]
        self.input_shape = tok["input_shape"]
        self.patch_size = tok["patch_size"]

        self.encoder, self.mask_generator, self.mask_ratio, self.masking_strategy = (
            build_mae_encoder_and_mask(
                config,
                self.device,
                dim_feedforward=self.dim_feedforward,
                norm_first=self.norm_first,
                d_model=d_model,
            )
        )

        pe_cfg = config["positional_encoding"]
        decoder_pe_module = build_pos_encoding(pe_cfg["decoder"], self.grid_dims, d_model)

        self.decoder = Decoder(
            output_dim=self.patch_dim,
            pos_encoding=decoder_pe_module,
            d_model=d_model,
            nhead=int(config.get("decoder_nhead")),
            num_layers=int(config.get("decoder_layers")),
            norm_first=self.norm_first,
            dim_feedforward=self.dim_feedforward,
            device=self.device,
        )

        self.encoder_pos_encoding_type = pe_cfg["encoder"]["type"]
        self.decoder_pos_encoding_type = pe_cfg["decoder"]["type"]

        # Optional auxiliary scale (mean/log-variance) prediction heads.
        # These are configured under model.scale_loss so checkpoints remain self-contained.
        sl = config.get("scale_loss", {})
        if sl is None:
            sl = {}
        if not isinstance(sl, dict):
            raise ValueError("model.scale_loss must be a dict when provided.")
        self.use_scale_loss = bool(sl.get("use_scale_loss", False))
        self.scale_loss_lambda_enc = float(sl.get("lambda_enc", 0.1))
        self.scale_loss_lambda_dec = float(sl.get("lambda_dec", 0.1))
        self.scale_loss_eps = float(sl.get("eps", 1e-8))
        if self.use_scale_loss:
            self.encoder_scale_head = nn.Linear(d_model, 2)
            self.decoder_scale_head = nn.Linear(d_model, 2)
        else:
            self.encoder_scale_head = None
            self.decoder_scale_head = None

        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstruction: bool = True,
        grid_dims: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = tokens_from_input(
            self.embedding_type,
            self.patcher,
            self.embedding,
            self.pos_encoding,
            x,
            grid_dims=grid_dims,
        )

        encoded, ids_keep, ids_mask = encode_mae_tokens(
            self.encoder_type,
            self.encoder,
            self.mask_generator,
            tokens,
            mask_ratio=mask_ratio,
            default_mask_ratio=self.mask_ratio,
            device=self.device,
        )

        output: Dict[str, torch.Tensor] = {
            "encoded_features": encoded,
            "ids_keep": ids_keep,
            "ids_mask": ids_mask,
        }

        if return_reconstruction:
            if self.use_scale_loss:
                reconstructed, decoded_tokens = self.decoder(
                    encoded, ids_keep, self.num_patches, return_decoded_tokens=True
                )
                output["decoded_tokens"] = decoded_tokens
            else:
                reconstructed = self.decoder(encoded, ids_keep, self.num_patches)
            output["reconstructed_patches"] = reconstructed

        if self.use_scale_loss:
            assert self.encoder_scale_head is not None
            assert self.decoder_scale_head is not None
            # Encoder predicts stats for visible tokens (same shape/order as encoded_features).
            output["pred_scale_encoder"] = self.encoder_scale_head(encoded)
            # Decoder predicts stats for all token positions (masked + visible); trainer can gather masked only.
            if "decoded_tokens" in output:
                output["pred_scale_decoder"] = self.decoder_scale_head(output["decoded_tokens"])

        return output

    def encode(
        self,
        x: torch.Tensor,
        grid_dims: Optional[Tuple[int, int, int]] = None,
        token_mode: str = "full_grid",
        pilot_flat_keep: Optional[torch.Tensor] = None,
        pilot_factorized_grid: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Encode input channels into token embeddings for downstream tasks.

        Training always applies masking through ``forward``. This method can either:
        - ``token_mode="full_grid"``: encode *all* patch tokens (no masking).
        - ``token_mode="masked_visible"``: encode only the *visible subset* produced by
          the model's mask generator (MAE-style visibility / sparse-pilot layout).
        - ``token_mode="pilot_visible"``: encode a fixed subset of patches given by
          ``pilot_flat_keep`` (1D indices into the time-major token sequence, same layout
          as :class:`~pilotwimae.models.modules.masking.FactorizedMaskGenerator`).
          For a **factorized** encoder, pass ``pilot_factorized_grid=(Tk, Sk)`` so that
          ``Tk * Sk == len(pilot_flat_keep)``; this is forwarded as ``time_steps`` /
          ``spatial_steps`` and may differ from pretraining ``num_time_keep`` /
          ``num_spatial_keep``.

        Internally, this is implemented via :func:`~pilotwimae.models.encoder_backbone.encode_mae_tokens`
        with ``mask_ratio=0.0`` for ``full_grid`` and ``mask_ratio=None`` for
        ``masked_visible`` (so it uses the model's configured ``self.mask_ratio``).
        ``pilot_visible`` gathers tokens then runs the encoder on ``(B, P_keep, D)``.

        For the standard encoder: ``full_grid`` runs on the full ``(B, P, D)`` sequence,
        while ``masked_visible`` runs on ``(B, P_keep, D)``.

        For the factorized encoder: ``full_grid`` may use explicit
        ``time_steps`` / ``spatial_steps`` to represent the full ``nt × (ns*nf)`` grid,
        and ``masked_visible`` runs on the factorized visible tube subset.
        """
        if token_mode not in {"full_grid", "masked_visible", "pilot_visible"}:
            raise ValueError(
                f"Unsupported token_mode: {token_mode}. "
                "Use 'full_grid', 'masked_visible', or 'pilot_visible'."
            )
        if token_mode == "pilot_visible" and pilot_flat_keep is None:
            raise ValueError("pilot_flat_keep is required when token_mode='pilot_visible'")
        if token_mode != "pilot_visible" and pilot_flat_keep is not None:
            raise ValueError("pilot_flat_keep is only valid when token_mode='pilot_visible'")
        if pilot_factorized_grid is not None and token_mode != "pilot_visible":
            raise ValueError("pilot_factorized_grid is only valid when token_mode='pilot_visible'")

        with torch.no_grad():
            tokens = tokens_from_input(
                self.embedding_type,
                self.patcher,
                self.embedding,
                self.pos_encoding,
                x,
                grid_dims=grid_dims,
            )
            if token_mode == "pilot_visible":
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
                            "encoder with token_mode='pilot_visible' (Tk*Sk must equal "
                            "number of kept patches)."
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

            encoded, _, _ = encode_mae_tokens(
                self.encoder_type,
                self.encoder,
                self.mask_generator,
                tokens,
                mask_ratio=0.0 if token_mode == "full_grid" else None,
                default_mask_ratio=self.mask_ratio,
                device=self.device,
            )
            return encoded

    def reconstruct_pilot_masked(
        self,
        x: torch.Tensor,
        pilot_flat_keep: torch.Tensor,
        pilot_factorized_grid: Optional[Tuple[int, int]] = None,
        grid_dims: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        MAE-style reconstruction with a **fixed** pilot mask (visible patches only).

        Same token gather / factorized layout rules as ``encode(..., token_mode="pilot_visible")``,
        then runs the pretrained decoder on encoder outputs.

        Returns:
            ``reconstructed_patches`` (B, P, patch_dim), ``ids_keep`` (B, P_keep),
            ``ids_mask`` (B, P_mask) — complement of pilots in ``[0, P)``.
        """
        idx = torch.unique(
            pilot_flat_keep.to(device=x.device, dtype=torch.long).reshape(-1),
            sorted=False,
        )
        tokens = tokens_from_input(
            self.embedding_type,
            self.patcher,
            self.embedding,
            self.pos_encoding,
            x,
            grid_dims=grid_dims,
        )
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
                    "encoder (Tk*Sk must equal number of kept patches)."
                )
            tk_e, sk_e = int(pilot_factorized_grid[0]), int(pilot_factorized_grid[1])
            if tk_e * sk_e != seq_keep:
                raise ValueError(
                    f"pilot_factorized_grid Tk*Sk={tk_e * sk_e} != gathered length {seq_keep}"
                )
            encoded = self.encoder(
                visible, time_steps=tk_e, spatial_steps=sk_e
            )
        else:
            encoded = self.encoder(visible)

        full_arange = torch.arange(P, device=x.device, dtype=torch.long)
        mask_1d = full_arange[~torch.isin(full_arange, idx)]
        ids_mask = mask_1d.unsqueeze(0).expand(B, -1)
        ids_keep = gather_ix
        reconstructed = self.decoder(encoded, ids_keep, self.num_patches)

        return {
            "reconstructed_patches": reconstructed,
            "ids_keep": ids_keep,
            "ids_mask": ids_mask,
        }

    def get_embeddings(
        self,
        x: torch.Tensor,
        pooling: str = "mean",
        token_mode: str = "full_grid",
        pilot_flat_keep: Optional[torch.Tensor] = None,
        pilot_factorized_grid: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Get pooled embeddings from encoder output."""
        encoded = self.encode(
            x,
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
            **kwargs,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def from_checkpoint(
        cls, filepath: str, device: Optional[torch.device] = None
    ) -> "PilotWiMAE":
        checkpoint = torch.load(filepath, map_location=device)
        cfg = checkpoint.get("config")
        if not isinstance(cfg, dict) or "model" not in cfg or not isinstance(
            cfg["model"], dict
        ):
            raise KeyError(
                "Checkpoint config must be a full training config with a "
                "'model' section (config['model'])."
            )

        model_config = cfg["model"]
        mkind = str(model_config.get("type", "")).lower()
        if mkind in ("temporalenc_beam", "temporalenc_los"):
            # Keep evaluate/evaluation scripts API-compatible:
            # PilotWiMAE.from_checkpoint(...) can return the supervised
            # encoder + head models, which expose get_embeddings().
            return PilotWiMAEBeamClassifier.from_checkpoint(
                filepath, device=device, full_config=cfg
            )

        model = cls(config=model_config, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "model_type": "PilotWiMAE",
            "input_shape": self.input_shape,
            "patch_size": self.patch_size,
            "patch_dim": self.patch_dim,
            "grid_dims": self.grid_dims,
            "num_patches": self.num_patches,
            "embedding_type": self.embedding_type,
            "encoder_positional_encoding": self.encoder_pos_encoding_type,
            "decoder_positional_encoding": self.decoder_pos_encoding_type,
            "masking_strategy": self.masking_strategy,
            "mask_ratio": self.mask_ratio,
            "norm_first": self.norm_first,
            "encoder_type": self.encoder_type,
            "encoder_dim": self.encoder_dim,
            "encoder_nhead": self.encoder_nhead,
            "encoder_layers": self.encoder_layers,
            "decoder_dim": self.decoder_dim,
            "decoder_nhead": self.decoder_nhead,
            "decoder_layers": self.decoder_layers,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
