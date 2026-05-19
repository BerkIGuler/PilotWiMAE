"""Pilot masking and pilot-only AWGN for channel-prediction baselines."""

from __future__ import annotations

import math
import re
import numpy as np


def _parse_int_list(part: str) -> list[int]:
    out: list[int] = []
    for chunk in part.split(","):
        c = chunk.strip()
        if not c:
            continue
        if not re.fullmatch(r"-?\d+", c):
            raise ValueError(f"Invalid integer in pilot_pattern: {c!r}")
        out.append(int(c))
    return out


def parse_pilot_pattern(pattern: str) -> tuple[list[int], list[int]]:
    """Parse pilot pattern ``t:<t0>,<t1>;f:<f0>,<f1>,...``."""
    text = (pattern or "").strip()
    if not text:
        raise ValueError("pilot_pattern is empty")
    parts = [p.strip() for p in text.split(";")]
    if len(parts) != 2:
        raise ValueError("pilot_pattern must contain exactly one ';'")
    t_seg, f_seg = parts
    if not t_seg.lower().startswith("t:"):
        raise ValueError("pilot_pattern first segment must start with 't:'")
    if not f_seg.lower().startswith("f:"):
        raise ValueError("pilot_pattern second segment must start with 'f:'")
    t_idx = _parse_int_list(t_seg[2:])
    f_idx = _parse_int_list(f_seg[2:])
    if not t_idx or not f_idx:
        raise ValueError("pilot_pattern must contain both non-empty t and f index lists")
    return t_idx, f_idx


def make_observed_from_target(
    target: np.ndarray,
    *,
    pilot_times: list[int],
    known_subcarriers: list[int],
) -> np.ndarray:
    """Zero grid with pilot REs copied from ``target``."""
    obs = np.zeros_like(target)
    t_ix, n_ix, f_ix = np.ix_(
        np.asarray(pilot_times, dtype=np.int64),
        np.arange(target.shape[1], dtype=np.int64),
        np.asarray(known_subcarriers, dtype=np.int64),
    )
    obs[t_ix, n_ix, f_ix] = target[t_ix, n_ix, f_ix]
    return obs


def complex_awgn_like(
    x: np.ndarray,
    *,
    snr_db: float,
    signal_mean_power: float | None,
    noise_floor: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex Gaussian noise so average SNR matches ``snr_db`` (linear scale vs signal power)."""
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    if snr_lin <= 0:
        raise ValueError("snr_db must map to positive linear SNR")
    if noise_floor:
        if signal_mean_power is None or float(signal_mean_power) <= 0:
            raise ValueError("signal_mean_power must be positive when noise_floor=True")
        p_s = float(signal_mean_power)
    else:
        p_s = float(np.mean(np.abs(x) ** 2))
    p_n = p_s / snr_lin
    sigma = math.sqrt(p_n / 2.0)
    noise = rng.normal(0.0, sigma, size=x.shape) + 1j * rng.normal(0.0, sigma, size=x.shape)
    return x + noise.astype(x.dtype, copy=False)


def apply_pilot_awgn(
    observed: np.ndarray,
    *,
    pilot_times: list[int],
    known_subcarriers: list[int],
    snr_db: float,
    signal_mean_power: float | None,
    noise_floor: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add AWGN only on pilot REs."""
    out = observed.copy()
    t_ix, n_ix, f_ix = np.ix_(
        np.asarray(pilot_times, dtype=np.int64),
        np.arange(observed.shape[1], dtype=np.int64),
        np.asarray(known_subcarriers, dtype=np.int64),
    )
    pilots = out[t_ix, n_ix, f_ix]
    out[t_ix, n_ix, f_ix] = complex_awgn_like(
        pilots,
        snr_db=snr_db,
        signal_mean_power=signal_mean_power,
        noise_floor=noise_floor,
        rng=rng,
    )
    return out
