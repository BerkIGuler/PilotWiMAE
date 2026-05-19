"""
Training utility helpers for PilotWiMAE trainers.
"""

from typing import Any, Dict

import torch


def safe_torch_load(path: str, map_location):
    """Load a checkpoint dict (full training state, not weights_only)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_model_type(model_cfg: Dict[str, Any]) -> str:
    """Normalize required ``model.type`` for dispatch and logging."""
    if "type" not in model_cfg:
        raise KeyError("Missing required config field: model.type")
    mt = model_cfg.get("type")
    if not isinstance(mt, str):
        raise TypeError(f"Expected model.type to be a string, got: {type(mt).__name__}")
    return mt.lower()


def fmt_float(v: float) -> str:
    """Format a float compactly for experiment names (e.g. 0.001 -> '1e-3')."""
    if v == 0:
        return "0"
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    s = f"{v:.1e}"
    mantissa, exp = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = str(int(exp))
    if mantissa == "1":
        return f"1e{exp}"
    return f"{mantissa}e{exp}"


def generate_exp_name(config: Dict[str, Any]) -> str:
    """Build a descriptive experiment name from the full training config."""
    m = config["model"]
    t = config["training"]

    parts = []

    # encoder type
    enc_type = m.get("encoder_type", "standard")
    parts.append(enc_type)

    # embedding
    parts.append(m["embedding"]["type"])

    # positional encoding (encoder side)
    pe = m["positional_encoding"]["encoder"]["type"]
    pe_short = {"sinusoidal_concat": "sincat", "learnable": "learnable"}
    parts.append(pe_short.get(pe, pe))

    # norm_first
    if m.get("norm_first", False):
        parts.append("preln")

    # patch size
    ps = m["patch_size"]
    parts.append("x".join(str(p) for p in ps))

    # encoder architecture
    d = m["encoder_dim"]
    el = m["encoder_layers"]
    eh = m["encoder_nhead"]
    ffn_f = m.get("ffn_factor", 4)
    parts.append(f"{d}d_ffn{ffn_f}x")
    if enc_type in ("factorized", "factorized_mixing"):
        parts.append(f"{el}blocks{eh}h")
    else:
        parts.append(f"{el}L{eh}h")

    # decoder architecture
    dl = m["decoder_layers"]
    dh = m["decoder_nhead"]
    parts.append(f"{dl}L{dh}h")

    # optimizer
    opt = t["optimizer"]
    parts.append(opt["type"])

    # scheduler
    parts.append(t["scheduler"]["type"])

    # loss
    parts.append(t["loss"])
    if t.get("norm_patch_loss", False):
        parts.append("normpatch")

    # training hyperparams
    parts.append(f"{t['epochs']}ep")
    parts.append(f"bs{t['batch_size']}")
    parts.append(f"lr{fmt_float(opt['lr'])}")
    parts.append(f"wd{fmt_float(opt['weight_decay'])}")

    # masking / task-specific suffixes
    model_type = resolve_model_type(m)
    if model_type == "temporalenc_beam":
        bp = config.get("task", {}).get("beam_prediction") or {}
        parts.append(f"beam_oh{bp.get('o_h', 1)}_ov{bp.get('o_v', 1)}")
        uh, uv = int(bp.get("u_h", 1)), int(bp.get("u_v", 1))
        if uh != 1 or uv != 1:
            parts.append(f"uh{uh}_uv{uv}")
    elif model_type == "temporalenc_los":
        lc = config.get("task", {}).get("los") or {}
        parts.append(f"los_nc{int(lc.get('num_classes', 2))}")
    else:
        masking = m["masking"]
        strategy = masking["strategy"]
        if strategy == "factorized":
            parts.append(f"tkeep{masking['num_time_keep']}")
            parts.append(f"smask{masking['spatial_mask_ratio']}")
        else:
            parts.append(strategy)
            parts.append(f"mask{masking['mask_ratio']}")

    return "_".join(parts)
