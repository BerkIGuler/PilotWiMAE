"""Unit tests for custom training losses."""

import math

import pytest
import torch

from pilotwimae.training import PerSampleNMSE


def test_per_sample_nmse_zero_error_is_zero():
    pred = torch.randn(4, 16, 8)
    loss = PerSampleNMSE()(pred, pred.clone())
    assert torch.allclose(loss, torch.zeros(()))


def test_per_sample_nmse_unit_target_matches_mean_sq_err():
    """When sum(target^2) per sample == 1, NMSE reduces to total sq-error."""
    target = torch.zeros(2, 4, 1)
    target[:, 0, 0] = 1.0
    pred = torch.zeros_like(target)
    pred[:, 0, 0] = 0.5
    loss = PerSampleNMSE()(pred, target)
    assert torch.allclose(loss, torch.tensor(0.25))


def test_per_sample_nmse_invariant_to_per_sample_scale():
    """Multiplying both pred and target of one sample by a constant must not
    change the per-sample NMSE contribution. This is the key property that
    motivates NMSE for raw-target training: high-power samples don't dominate.
    """
    torch.manual_seed(0)
    target = torch.randn(4, 16, 8)
    pred = target + 0.1 * torch.randn_like(target)

    scales = torch.tensor([1.0, 100.0, 0.01, 1000.0])
    target_scaled = target * scales.view(-1, 1, 1)
    pred_scaled = pred * scales.view(-1, 1, 1)

    loss_unscaled = PerSampleNMSE()(pred, target)
    loss_scaled = PerSampleNMSE()(pred_scaled, target_scaled)
    assert torch.allclose(loss_unscaled, loss_scaled, atol=1e-6)


def test_per_sample_nmse_handles_zero_power_target():
    """Zero-power target falls back to eps; loss must be finite."""
    target = torch.zeros(2, 4, 8)
    pred = torch.full_like(target, 0.1)
    loss = PerSampleNMSE(eps=1e-8)(pred, target)
    assert torch.isfinite(loss)


def test_per_sample_nmse_db_conversion_matches_wireless_convention():
    """NMSE in dB = 10*log10(loss). Sanity check at a known value."""
    target = torch.ones(1, 1, 1)
    pred = target + math.sqrt(0.1)  # sq err = 0.1, target power = 1 -> NMSE = 0.1
    loss = PerSampleNMSE()(pred, target)
    db = 10.0 * torch.log10(loss)
    assert torch.allclose(db, torch.tensor(-10.0), atol=1e-5)


def test_per_sample_nmse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        PerSampleNMSE()(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))


def test_per_sample_nmse_upcasts_under_fp16():
    """Inputs in fp16 must not overflow during the squared-sum reduction."""
    pred = torch.full((2, 800, 32), 10.0, dtype=torch.float16)
    target = torch.full((2, 800, 32), 9.0, dtype=torch.float16)
    loss = PerSampleNMSE()(pred, target)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32
