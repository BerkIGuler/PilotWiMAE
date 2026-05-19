"""Fixed-pilot MAE reconstruction helpers and NMSE."""

import pytest
import torch

from pilotwimae import PilotWiMAE
from pilotwimae.downstream.channel_prediction.metrics import nmse_on_masked
from pilotwimae.downstream.channel_prediction.noise import corrupt_pilot_patches
from pilotwimae.downstream.channel_prediction.norm_patch import denormalize_norm_patch_patches


def _tiny_config() -> dict:
    return {
        "type": "pilotwimae",
        "encoder_type": "standard",
        "input_shape": [4, 8, 8],
        "patch_size": [1, 2, 2],
        "embedding": {"type": "linear"},
        "positional_encoding": {
            "encoder": {"type": "sinusoidal_concat"},
            "decoder": {"type": "sinusoidal_concat"},
        },
        "masking": {"strategy": "random", "mask_ratio": 0.5},
        "encoder_dim": 32,
        "encoder_layers": 1,
        "encoder_nhead": 4,
        "decoder_layers": 1,
        "decoder_nhead": 4,
    }


def test_nmse_on_masked_zero_when_equal() -> None:
    B, P, D = 2, 5, 8
    t = torch.randn(B, P, D)
    ids_mask = torch.tensor([[1, 3], [0, 4]], dtype=torch.long)
    n = nmse_on_masked(t, t, ids_mask, eps=1e-12)
    assert n.item() == 0.0


def test_denormalize_norm_patch_roundtrip() -> None:
    torch.manual_seed(0)
    ref = torch.randn(2, 5, 16)
    eps = 1e-6
    mean = ref.mean(dim=-1, keepdim=True)
    var = ref.var(dim=-1, unbiased=False, keepdim=True)
    std = (var + eps).sqrt()
    normed = (ref - mean) / std
    back = denormalize_norm_patch_patches(normed, ref, eps=eps)
    assert torch.allclose(back, ref, atol=1e-5, rtol=1e-4)


def test_nmse_on_masked_known_ratio() -> None:
    B, Pm, D = 1, 1, 2
    recon = torch.zeros(B, 3, D)
    tgt = torch.zeros(B, 3, D)
    tgt[0, 0, :] = 1.0
    recon[0, 0, :] = 2.0
    ids_mask = torch.tensor([[0]], dtype=torch.long)
    n = nmse_on_masked(recon, tgt, ids_mask, eps=1e-12)
    assert abs(float(n.item()) - 1.0) < 1e-5


def test_corrupt_pilot_patches_shape() -> None:
    patches = torch.randn(2, 10, 16)
    idx = torch.tensor([0, 2, 9], dtype=torch.long)
    out = corrupt_pilot_patches(patches, idx, 30.0)
    assert out.shape == patches.shape
    assert torch.allclose(out[:, [1, 3, 4, 5, 6, 7, 8]], patches[:, [1, 3, 4, 5, 6, 7, 8]])


def test_corrupt_pilot_patches_fixed_floor_uses_same_noise_scale() -> None:
    torch.manual_seed(0)
    patches = torch.zeros(1, 2, 16)
    patches[:, 1, :] = 10.0  # very different signal level across pilot rows
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = corrupt_pilot_patches(
        patches,
        idx,
        snr_db=20.0,
        signal_mean_power=1.5,
    )
    diff = out - patches
    # With fixed-floor AWGN both pilot rows should have similar noise std.
    std0 = diff[:, 0, :].std(unbiased=False)
    std1 = diff[:, 1, :].std(unbiased=False)
    ratio = (std0 / std1).item()
    assert 0.6 <= ratio <= 1.4


def test_corrupt_pilot_patches_fixed_floor_requires_positive_power() -> None:
    patches = torch.zeros(1, 3, 8)
    idx = torch.tensor([0, 2], dtype=torch.long)
    with pytest.raises(ValueError, match="signal_mean_power must be positive"):
        _ = corrupt_pilot_patches(
            patches,
            idx,
            snr_db=10.0,
            signal_mean_power=0.0,
        )


def test_corrupt_pilot_patches_per_channel_mean_power_matches_snr() -> None:
    """Without signal_mean_power, P_s is mean |h|^2 over pilot patches (per batch row)."""
    torch.manual_seed(3)
    L = 4
    B = 4
    patches = torch.randn(B, 10, 2 * L)
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    snr_db = 10.0
    snr_lin = 10.0 ** (snr_db / 10.0)

    gen = torch.Generator(device=patches.device)
    gen.manual_seed(7)
    out = corrupt_pilot_patches(patches, idx, snr_db, signal_mean_power=None, generator=gen)
    clean_sel = patches[:, idx, :]
    obs_sel = out[:, idx, :]
    p_clean = torch.complex(clean_sel[..., :L], clean_sel[..., L:])
    p_obs = torch.complex(obs_sel[..., :L], obs_sel[..., L:])
    noise = p_obs - p_clean
    for b in range(B):
        p_s = (p_clean[b].abs() ** 2).mean().item()
        p_n = p_s / snr_lin
        emp = (noise[b].abs() ** 2).mean().item()
        assert abs(emp - p_n) / p_n < 0.2


def test_reconstruct_pilot_masked_shapes_and_ids_mask_len() -> None:
    cfg = _tiny_config()
    model = PilotWiMAE(cfg, device=torch.device("cpu"))
    B = 2
    T, S, F = cfg["input_shape"]
    x = torch.complex(torch.randn(B, T, S, F), torch.randn(B, T, S, F))
    P = model.num_patches
    pilot = torch.tensor([0, 1, 3], dtype=torch.long)
    with torch.no_grad():
        out = model.reconstruct_pilot_masked(x, pilot, pilot_factorized_grid=None)
    assert out["reconstructed_patches"].shape == (B, P, model.patch_dim)
    assert out["ids_keep"].shape == (B, pilot.numel())
    assert out["ids_mask"].shape[1] == P - torch.unique(pilot).numel()


def test_reconstruct_pilot_masked_factorized() -> None:
    cfg = _tiny_config()
    cfg["encoder_type"] = "factorized"
    cfg["masking"] = {
        "strategy": "factorized",
        "mask_ratio": 0.5,
        "num_time_keep": 2,
        "spatial_mask_ratio": 0.5,
    }
    model = PilotWiMAE(cfg, device=torch.device("cpu"))
    B = 1
    T, S, F = cfg["input_shape"]
    x = torch.complex(torch.randn(B, T, S, F), torch.randn(B, T, S, F))
    P = model.num_patches
    pilot = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    with torch.no_grad():
        out = model.reconstruct_pilot_masked(x, pilot, pilot_factorized_grid=(2, 2))
    assert out["reconstructed_patches"].shape == (B, P, model.patch_dim)
