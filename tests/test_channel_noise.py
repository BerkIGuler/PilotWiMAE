import torch

from pilotwimae.data.beam.channel_noise import add_complex_awgn_snr_db


def test_add_complex_awgn_mean_noise_power_matches_snr():
    torch.manual_seed(0)
    h = torch.randn(200, 4, 4, 8, dtype=torch.complex64)
    h = h / (h.abs() ** 2).mean().sqrt()

    snr_db = 10.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_n = 1.0 / snr_lin

    gen = torch.Generator(device=h.device)
    gen.manual_seed(123)
    noisy = add_complex_awgn_snr_db(h, snr_db, generator=gen, signal_mean_power=1.0)
    noise = noisy - h
    emp_pn = (noise.abs() ** 2).mean().item()
    assert abs(emp_pn - p_n) / p_n < 0.15


def test_add_complex_awgn_uses_signal_mean_power_for_snr_scale():
    """P_n = P_s / SNR_lin with P_s from signal_mean_power (not from sample magnitudes)."""
    torch.manual_seed(1)
    h = torch.randn(100, 3, 3, 4, dtype=torch.complex64) * 0.5  # arbitrary scale
    p_s = 0.73
    snr_db = 6.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_n = p_s / snr_lin

    gen = torch.Generator(device=h.device)
    gen.manual_seed(42)
    noisy = add_complex_awgn_snr_db(h, snr_db, generator=gen, signal_mean_power=p_s)
    noise = noisy - h
    emp_pn = (noise.abs() ** 2).mean().item()
    assert abs(emp_pn - p_n) / p_n < 0.15


def test_add_complex_awgn_per_sample_power_when_noise_floor_false():
    """noise_floor=False: P_s[b] = mean |h|^2 over elements of batch row b."""
    torch.manual_seed(2)
    h = torch.randn(32, 4, 4, 6, dtype=torch.complex64)
    scales = torch.linspace(0.25, 4.0, 32).view(32, 1, 1, 1).to(dtype=h.real.dtype)
    h = h * scales.to(dtype=h.dtype)

    snr_db = 10.0
    snr_lin = 10.0 ** (snr_db / 10.0)

    gen = torch.Generator(device=h.device)
    gen.manual_seed(99)
    noisy = add_complex_awgn_snr_db(h, snr_db, generator=gen, noise_floor=False)
    noise = noisy - h

    for b in range(h.shape[0]):
        p_s = (h[b].abs() ** 2).mean().item()
        p_n = p_s / snr_lin
        emp = (noise[b].abs() ** 2).mean().item()
        assert abs(emp - p_n) / p_n < 0.4
