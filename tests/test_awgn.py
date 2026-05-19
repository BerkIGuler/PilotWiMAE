"""Tests for noise-robust training AWGN helpers."""

import pytest
import torch

from pilotwimae.training.awgn import (
    awgn_complex_channel,
    snr_min_db_for_local_epoch,
    uniform_snr_db_per_sample,
)


def test_snr_min_db_curriculum_endpoints():
    assert snr_min_db_for_local_epoch(0, 100, 40.0) == pytest.approx(40.0)
    assert snr_min_db_for_local_epoch(99, 100, 40.0) == pytest.approx(0.0, abs=1e-9)
    assert snr_min_db_for_local_epoch(0, 1, 40.0) == pytest.approx(40.0)


def test_uniform_snr_db_shape_and_range():
    low, high = 10.0, 30.0
    s = uniform_snr_db_per_sample(500, low, high, torch.device("cpu"))
    assert s.shape == (500,)
    assert (s >= low - 1e-6).all() and (s <= high + 1e-6).all()


def test_awgn_preserves_shape_dtype_device():
    x = torch.randn(4, 3, 4, 4, dtype=torch.complex64)
    snr = torch.full((4,), 20.0, dtype=torch.float32)
    y = awgn_complex_channel(x, snr)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_awgn_empirical_snr_matches_target():
    torch.manual_seed(0)
    B, T, S, F = 8, 14, 32, 32
    x = torch.randn(B, T, S, F, dtype=torch.complex64) * 0.5
    snr_db = 15.0
    snr = torch.full((B,), snr_db, dtype=torch.float32)
    n_trials = 400
    ratios = []
    for _ in range(n_trials):
        n = awgn_complex_channel(x, snr) - x
        P_sig = (x.abs() ** 2).view(B, -1).mean(dim=1)
        P_noise = (n.abs() ** 2).view(B, -1).mean(dim=1)
        ratio = (P_noise / P_sig.clamp(min=1e-12)).mean().item()
        ratios.append(ratio)
    mean_ratio = sum(ratios) / len(ratios)
    expected = 10.0 ** (-snr_db / 10.0)
    assert mean_ratio == pytest.approx(expected, rel=0.08)


def test_awgn_complex128():
    x = torch.randn(2, 4, 4, 4, dtype=torch.complex128)
    snr = torch.tensor([10.0, 20.0], dtype=torch.float32)
    y = awgn_complex_channel(x, snr)
    assert y.dtype == torch.complex128
