"""Framework-agnostic baseline metrics."""

from __future__ import annotations

import math

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error for real or complex arrays."""
    p = np.asarray(pred)
    t = np.asarray(target)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs target {t.shape}")
    d = p - t
    return float(np.mean(np.abs(d) ** 2))


def nmse(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-12) -> float:
    """Normalized MSE: sum(|pred-target|^2) / max(sum(|target|^2), eps)."""
    p = np.asarray(pred)
    t = np.asarray(target)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs target {t.shape}")
    num = float(np.sum(np.abs(p - t) ** 2))
    den = max(float(np.sum(np.abs(t) ** 2)), float(eps))
    return num / den


def nmse_masked(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """NMSE with numerator/denominator summed only where ``mask`` is True (broadcast-safe)."""
    p = np.asarray(pred)
    t = np.asarray(target)
    m = np.asarray(mask, dtype=bool)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs target {t.shape}")
    if m.shape != p.shape:
        raise ValueError(f"mask shape {m.shape} must match pred {p.shape}")
    diff = np.abs(p - t) ** 2
    tgt = np.abs(t) ** 2
    num = float(np.sum(diff[m]))
    den = max(float(np.sum(tgt[m])), float(eps))
    return num / den


def nmse_non_pilot_tf(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    tf_mask: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """NMSE over non-pilot TF positions (full ``[T,N_a,N_f]`` mask); one scalar per channel tensor."""
    return nmse_masked(pred, target, tf_mask, eps=eps)


def nmse_to_db(nmse_linear: float, *, eps: float = 1e-12) -> float:
    """Convert linear NMSE to dB: ``10 log10(max(nmse, eps))``."""
    return 10.0 * math.log10(max(float(nmse_linear), float(eps)))
