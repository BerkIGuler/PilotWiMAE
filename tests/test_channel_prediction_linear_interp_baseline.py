import numpy as np

from baselines.channel_prediction.evaluate import evaluate_linear_interpolation
from baselines.channel_prediction.linear_interp import (
    fill_frame_linear_frequency,
    interpolate_time_linear,
    reconstruct_linear_from_pilots,
)
from baselines.channel_prediction.masks import (
    expanded_subcarrier_indices,
    pilot_time_indices,
)
from baselines.channel_prediction.metrics import mse, nmse


def test_expanded_subcarrier_indices_for_patch_size_4() -> None:
    got = expanded_subcarrier_indices([0, 2, 4, 6], freq_patch_size=4)
    assert got == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]


def test_fill_frame_linear_frequency_linear_ramp() -> None:
    # Two rows with the same linear ramp across frequency.
    x = np.tile(np.arange(32, dtype=np.float64), (2, 1))
    known = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    out = fill_frame_linear_frequency(
        x,
        known_subcarriers=known,
        frequency_axis=1,
        frequency_outside_mode="linear",
    )
    assert np.allclose(out, x)


def test_fill_frame_linear_frequency_outside_linear_extrapolation() -> None:
    # Known points only in the middle; edge behavior differs by mode.
    x = np.tile(np.arange(32, dtype=np.float64), (1, 1))
    known = [8, 9, 10, 11, 16, 17, 18, 19]
    out = fill_frame_linear_frequency(
        x, known_subcarriers=known, frequency_axis=1, frequency_outside_mode="linear"
    )
    assert np.allclose(out, x)


def test_interpolate_time_linear_midpoint() -> None:
    a = np.ones((2, 2), dtype=np.float64) * 2.0
    b = np.ones((2, 2), dtype=np.float64) * 10.0
    mid = interpolate_time_linear(a, b, t0=2, t1=10, t_query=6)
    assert np.allclose(mid, np.ones((2, 2)) * 6.0)


def test_reconstruct_linear_from_pilots_recovers_affine_signal() -> None:
    # target[t, n, f] = t + 2*f is linear in time and frequency.
    T, N, F = 16, 3, 32
    tt = np.arange(T, dtype=np.float64)[:, None, None]
    ff = np.arange(F, dtype=np.float64)[None, None, :]
    target = tt + 2.0 * ff
    target = np.repeat(target, N, axis=1)

    t_pilots = pilot_time_indices([2, 11])
    known = expanded_subcarrier_indices([0, 2, 4, 6], freq_patch_size=4)

    observed = np.zeros_like(target)
    t_ix, n_ix, f_ix = np.ix_(np.asarray(t_pilots), np.arange(N), np.asarray(known))
    observed[t_ix, n_ix, f_ix] = target[t_ix, n_ix, f_ix]
    recon = reconstruct_linear_from_pilots(
        observed,
        pilot_times=t_pilots,
        known_subcarriers=known,
        time_axis=0,
        frequency_axis=2,
        frequency_outside_mode="linear",
    )
    # Exact on [2,11]; clamped outside this range by design.
    assert np.allclose(recon[2:12], target[2:12], atol=1e-10)


def test_reconstruct_linear_from_pilots_linear_extrapolation_outside_range() -> None:
    T, N, F = 16, 1, 32
    tt = np.arange(T, dtype=np.float64)[:, None, None]
    ff = np.arange(F, dtype=np.float64)[None, None, :]
    target = tt + 2.0 * ff

    t_pilots = [2, 11]
    known = expanded_subcarrier_indices([0, 2, 4, 6], freq_patch_size=4)
    observed = np.zeros_like(target)
    t_ix, n_ix, f_ix = np.ix_(np.asarray(t_pilots), np.arange(N), np.asarray(known))
    observed[t_ix, n_ix, f_ix] = target[t_ix, n_ix, f_ix]
    recon = reconstruct_linear_from_pilots(
        observed,
        pilot_times=t_pilots,
        known_subcarriers=known,
        time_axis=0,
        frequency_axis=2,
        frequency_outside_mode="linear",
        time_outside_mode="linear",
    )
    assert np.allclose(recon, target, atol=1e-10)


def test_metrics_sanity() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 5.0], dtype=np.float64)
    assert mse(a, b) == (0.0 + 0.0 + 4.0) / 3.0
    assert abs(nmse(a, b) - (4.0 / (1.0 + 4.0 + 25.0))) < 1e-12


def test_evaluate_linear_interpolation_returns_metrics() -> None:
    T, N, F = 12, 2, 32
    target = np.zeros((T, N, F), dtype=np.complex64)
    for t in range(T):
        for f in range(F):
            target[t, :, f] = (t + f) + 1j * (2 * t - f)
    pilots_t = [2, 11]
    known = expanded_subcarrier_indices([0, 2, 4, 6], freq_patch_size=4)
    observed = np.zeros_like(target)
    t_ix, n_ix, f_ix = np.ix_(np.asarray(pilots_t), np.arange(N), np.asarray(known))
    observed[t_ix, n_ix, f_ix] = target[t_ix, n_ix, f_ix]

    out = evaluate_linear_interpolation(
        observed,
        target,
        pilot_times=pilots_t,
        known_subcarriers=known,
        time_axis=0,
        frequency_axis=2,
    )
    assert "mse" in out and "nmse" in out and "reconstructed" in out
    assert out["reconstructed"].shape == target.shape
