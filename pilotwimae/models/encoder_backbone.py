"""
Shared encoder / tokenizer construction for PilotWiMAE and supervised variants.

MAE training uses masking (standard or factorized). Supervised training on beam labels always
uses :func:`build_supervised_encoder` (full grid, no masking).

Standard supervised uses the regular :class:`Encoder` on the full ``(B, P, D)`` token
sequence (``P = nt·ns·nf``). Factorized supervised feeds the full patch grid
(``nt x ns*nf`` tokens) to :class:`FactorizedEncoder` with ``Tk=nt`` and ``Sk=ns*nf``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .modules import (
    Patcher3D,
    InversePatcher3D,
    LinearEmbedding,
    Conv3dEmbedding,
    LearnablePositionalEncoding,
    SinusoidalConcat3D,
    MaskGenerator,
    FactorizedMaskGenerator,
    Encoder,
    FactorizedEncoder,
)

logger = logging.getLogger(__name__)


def is_factorized_family(encoder_type: Optional[Any]) -> bool:
    """True for factorized backbone variants (tube masking, T×S encoder layout)."""
    return str(encoder_type or "") in ("factorized", "factorized_mixing")


def build_embedding(config: Dict[str, Any], patch_size: Tuple[int, ...], patch_dim: int, d_model: int) -> nn.Module:
    emb_type = config["embedding"]["type"]
    if emb_type == "linear":
        return LinearEmbedding(patch_dim, d_model)
    if emb_type == "conv3d":
        return Conv3dEmbedding(patch_size, d_model)
    raise ValueError(f"Unknown embedding type: {emb_type!r}")


def build_pos_encoding(pe_cfg: Dict[str, Any], grid_dims: Tuple[int, int, int], d_model: int) -> nn.Module:
    pe_type = pe_cfg["type"]
    if pe_type == "sinusoidal_concat":
        return SinusoidalConcat3D(grid_dims, d_model)
    if pe_type == "learnable":
        nt, ns, nf = grid_dims
        return LearnablePositionalEncoding(nt * ns * nf, d_model)
    raise ValueError(f"Unknown positional encoding type: {pe_type!r}")


def tokenizer_geometry(config: Dict[str, Any]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int]:
    """Return input_shape, patch_size, num_patches, patch_dim."""
    input_shape = tuple(config["input_shape"])
    patch_size = tuple(config["patch_size"])
    T, S, F = input_shape
    pt, ps, pf = patch_size
    grid_dims = (T // pt, S // ps, F // pf)
    num_patches = grid_dims[0] * grid_dims[1] * grid_dims[2]
    patch_dim = 2 * pt * ps * pf
    return input_shape, patch_size, num_patches, patch_dim


def build_tokenizer_modules(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build patcher (for linear embedding), embedding, encoder-side positional encoding.

    Returns a dict with keys: patcher, inverse_patcher, embedding, pos_encoding,
    embedding_type, grid_dims, num_patches, patch_dim, input_shape, patch_size.
    """
    input_shape, patch_size, num_patches, patch_dim = tokenizer_geometry(config)
    d_model = int(config.get("encoder_dim"))

    patcher = Patcher3D(patch_size)
    inverse_patcher = InversePatcher3D(input_shape, patch_size)

    embedding_type = config["embedding"]["type"]
    embedding = build_embedding(config, patch_size, patch_dim, d_model)

    pe_cfg = config["positional_encoding"]
    if "encoder" not in pe_cfg:
        raise ValueError("'positional_encoding' must contain 'encoder' sub-config.")
    T, S, F = input_shape
    pt, ps, pf = patch_size
    grid_dims = (T // pt, S // ps, F // pf)
    pos_encoding = build_pos_encoding(pe_cfg["encoder"], grid_dims, d_model)

    return {
        "patcher": patcher,
        "inverse_patcher": inverse_patcher,
        "embedding": embedding,
        "pos_encoding": pos_encoding,
        "embedding_type": embedding_type,
        "grid_dims": grid_dims,
        "num_patches": num_patches,
        "patch_dim": patch_dim,
        "input_shape": input_shape,
        "patch_size": patch_size,
    }


def build_mae_encoder_and_mask(
    config: Dict[str, Any],
    device: torch.device,
    *,
    dim_feedforward: int,
    norm_first: bool,
    d_model: int,
) -> Tuple[nn.Module, Optional[FactorizedMaskGenerator], float, str]:
    """
    Encoder + mask generator for MAE (masked pretraining).

    Returns:
        encoder, mask_generator (FactorizedMaskGenerator instance for factorized; MaskGenerator for standard),
        mask_ratio (effective), masking_strategy string.
    """
    masking_cfg = config.get("masking")
    if not isinstance(masking_cfg, dict) or "strategy" not in masking_cfg:
        raise ValueError(f"'masking' must be a dict with 'strategy'. Got: {masking_cfg!r}")

    encoder_type = config.get("encoder_type")
    masking_strategy = masking_cfg["strategy"]
    mask_ratio = float(masking_cfg.get("mask_ratio", 0.0))

    _, patch_size, _, _ = tokenizer_geometry(config)
    T, S, F = tuple(config["input_shape"])
    pt, ps, pf = patch_size
    grid_dims = (T // pt, S // ps, F // pf)
    num_patches = grid_dims[0] * grid_dims[1] * grid_dims[2]

    if is_factorized_family(encoder_type):
        num_time_keep = int(masking_cfg["num_time_keep"])
        spatial_mask_ratio = float(masking_cfg["spatial_mask_ratio"])
        mask_generator = FactorizedMaskGenerator(
            grid_dims=grid_dims,
            num_time_keep=num_time_keep,
            spatial_mask_ratio=spatial_mask_ratio,
            device=device,
        )
        num_spatial_keep = mask_generator.num_spatial_keep
        effective_mask_ratio = 1.0 - (num_time_keep * num_spatial_keep) / num_patches
        encoder = FactorizedEncoder(
            d_model=d_model,
            nhead=int(config.get("encoder_nhead")),
            num_blocks=int(config.get("encoder_layers")),
            num_time_keep=num_time_keep,
            num_spatial_keep=num_spatial_keep,
            dim_feedforward=dim_feedforward,
            norm_first=norm_first,
            device=device,
            enable_cross_dim_mixing=(encoder_type == "factorized_mixing"),
        )
    else:
        mask_generator = MaskGenerator(
            device=device,
            mask_ratio=mask_ratio,
            strategy=masking_strategy,
            grid_dims=grid_dims,
        )
        effective_mask_ratio = mask_ratio
        encoder = Encoder(
            d_model=d_model,
            nhead=int(config.get("encoder_nhead")),
            num_layers=int(config.get("encoder_layers")),
            norm_first=norm_first,
            dim_feedforward=dim_feedforward,
            device=device,
        )

    return encoder, mask_generator, effective_mask_ratio, masking_strategy


def build_supervised_encoder(
    config: Dict[str, Any],
    device: torch.device,
    *,
    dim_feedforward: int,
    norm_first: bool,
    d_model: int,
) -> nn.Module:
    """
    Encoder for supervised tasks: **no masking**. All tokens are encoded.

    Factorized: Tk = nt, Sk = ns * nf (full grid)
    Standard: Pk = P (full grid)

    Returns:
        encoder module configured for full-grid supervised encoding.
    """
    encoder_type = config.get("encoder_type")
    _, patch_size, _, _ = tokenizer_geometry(config)
    T, S, F = tuple(config["input_shape"])
    pt, ps, pf = patch_size
    grid_dims = (T // pt, S // ps, F // pf)
    nt, ns, nf = grid_dims
    ns_nf = ns * nf

    if is_factorized_family(encoder_type):
        encoder = FactorizedEncoder(
            d_model=d_model,
            nhead=int(config.get("encoder_nhead")),
            num_blocks=int(config.get("encoder_layers")),
            num_time_keep=nt,
            num_spatial_keep=ns_nf,
            dim_feedforward=dim_feedforward,
            norm_first=norm_first,
            device=device,
            enable_cross_dim_mixing=(encoder_type == "factorized_mixing"),
        )
    else:
        encoder = Encoder(
            d_model=d_model,
            nhead=int(config.get("encoder_nhead")),
            num_layers=int(config.get("encoder_layers")),
            norm_first=norm_first,
            dim_feedforward=dim_feedforward,
            device=device,
        )
    return encoder


def tokens_from_input(
    embedding_type: str,
    patcher: Patcher3D,
    embedding: nn.Module,
    pos_encoding: nn.Module,
    x: torch.Tensor,
    grid_dims: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """Complex (B,T,S,F) -> embedded position-encoded tokens (B,P,D)."""
    if embedding_type == "conv3d":
        tokens = embedding(x)
    else:
        patches = patcher(x)
        tokens = embedding(patches)
    return pos_encoding(tokens, grid_dims=grid_dims)


def encode_mae_tokens(
    encoder_type: Optional[str],
    encoder: nn.Module,
    mask_generator: Any,
    tokens: torch.Tensor,
    *,
    mask_ratio: Optional[float],
    default_mask_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MAE encode path with optional masking. Returns encoded, ids_keep, ids_mask.

    When ``mask_ratio`` is 0.0 (or ``mask_ratio`` is None only if
    ``default_mask_ratio == 0``), the **standard** encoder runs on all ``P`` tokens.
    **Factorized**: if ``P == nt * (ns*nf)`` (from the factorized mask generator), run
    the encoder with ``time_steps=nt``, ``spatial_steps=ns*nf`` (full grid; same weights).
    Else if ``P == Tk*Sk`` from init, run without extra kwargs (tube-shaped sequence).
    Otherwise fall back to the tube mask (subset). Full grid is checked first so the
    correct ``(nt, ns*nf)`` reshape is used when ``Tk*Sk`` equals ``P`` numerically but
    factors differ.
    """
    def _ids_keep_all(*, B_: int, P_: int, device_: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ids_keep_ = (
            torch.arange(P_, device=device_, dtype=torch.long)
            .unsqueeze(0)
            .expand(B_, -1)
        )
        ids_mask_ = torch.empty(B_, 0, dtype=torch.long, device=device_)
        return ids_keep_, ids_mask_

    use_mask = (mask_ratio is None and default_mask_ratio > 0) or (
        mask_ratio is not None and mask_ratio > 0
    )

    if is_factorized_family(encoder_type):
        if use_mask:
            visible, ids_keep, ids_mask = mask_generator(tokens)
            encoded = encoder(visible)
            return encoded, ids_keep, ids_mask
        B, P, _ = tokens.shape
        Tk = int(encoder.num_time_keep)
        Sk = int(encoder.num_spatial_keep)
        mg = mask_generator
        if hasattr(mg, "nt") and hasattr(mg, "ns_nf"):
            nt = int(mg.nt)
            ns_nf = int(mg.ns_nf)
            if P == nt * ns_nf:
                encoded = encoder(tokens, time_steps=nt, spatial_steps=ns_nf)
                ids_keep, ids_mask = _ids_keep_all(B_=B, P_=P, device_=device)
                return encoded, ids_keep, ids_mask
        if P == Tk * Sk:
            encoded = encoder(tokens)
            ids_keep, ids_mask = _ids_keep_all(B_=B, P_=P, device_=device)
            return encoded, ids_keep, ids_mask
        if not getattr(encode_mae_tokens, "_logged_mae_factorized_subset", False):
            logger.info(
                "Factorized encode: P=%s does not match full grid nt*ns_nf or Tk*Sk=%s; "
                "using tube mask subset.",
                P,
                Tk * Sk,
            )
            encode_mae_tokens._logged_mae_factorized_subset = True  # type: ignore[attr-defined]
        visible, ids_keep, ids_mask = mask_generator(tokens)
        encoded = encoder(visible)
        return encoded, ids_keep, ids_mask

    if use_mask:
        mg = mask_generator
        if mask_ratio is not None and mask_ratio != default_mask_ratio:
            mg = MaskGenerator(
                device=device,
                mask_ratio=mask_ratio,
                strategy=mask_generator.strategy,
                grid_dims=mask_generator.grid_dims,
            )
        visible, ids_keep, ids_mask = mg(tokens)
        encoded = encoder(visible)
        return encoded, ids_keep, ids_mask

    encoded = encoder(tokens)
    B, P, _ = tokens.shape
    ids_keep, ids_mask = _ids_keep_all(B_=B, P_=P, device_=device)
    return encoded, ids_keep, ids_mask
