"""Tests for Kronecker LMMSE core (layout + equivalence to brute force)."""

from __future__ import annotations

import numpy as np

from baselines.channel_prediction.lmmse_core import (
    brute_force_lmmse_weights,
    compute_lmmse_weights_W,
    compute_lmmse_weights_W_full,
    estimate_R_hh,
    kronecker_blocks,
    pilot_observation_matrix,
    prepare_correlations,
    prepare_full_covariance,
    selection_matrix_pilot_freqs,
    selection_matrix_pilot_times,
    stack_pilot_vector,
    S_f_matrix,
    vec_time_ant_freq,
    estimate_R_t_R_sf,
)


def test_vec_roundtrip_small() -> None:
    rng = np.random.default_rng(0)
    T, Na, Nf = 5, 3, 7
    H = rng.standard_normal((T, Na, Nf)) + 1j * rng.standard_normal((T, Na, Nf))
    v = vec_time_ant_freq(H)
    assert v.shape == (T * Na * Nf,)
    from baselines.channel_prediction.lmmse_core import unvec_time_ant_freq

    H2 = unvec_time_ant_freq(v, T=T, N_a=Na, N_f=Nf)
    assert np.allclose(H2, H)


def test_stack_pilot_matches_kron_selector() -> None:
    rng = np.random.default_rng(1)
    T, Na, Nf = 4, 2, 8
    pilot_times = [1, 3]
    known_subcarriers = [0, 2, 5]  # arbitrary subset
    H = rng.standard_normal((T, Na, Nf)) + 1j * rng.standard_normal((T, Na, Nf))

    P_t = selection_matrix_pilot_times(pilot_times, T=T)
    P_f = selection_matrix_pilot_freqs(known_subcarriers, N_f=Nf)
    Sf = S_f_matrix(P_f, N_a=Na)
    Ph = np.kron(P_t, Sf)

    z1 = Ph @ vec_time_ant_freq(H)
    z2 = stack_pilot_vector(H, pilot_times=pilot_times, known_subcarriers=known_subcarriers)
    assert z1.shape == z2.shape
    assert np.allclose(z1, z2)


def test_kronecker_W_matches_brute_force_small() -> None:
    rng = np.random.default_rng(2)
    T, Na, Nf = 4, 2, 4
    Nfull = T * Na * Nf
    pilot_times = [1, 3]
    known_subcarriers = [0, 2]

    # PSD factors
    A = rng.standard_normal((T, T)) + 1j * rng.standard_normal((T, T))
    R_t = A @ A.conj().T + 0.5 * np.eye(T)
    B = rng.standard_normal((Na * Nf, Na * Nf)) + 1j * rng.standard_normal((Na * Nf, Na * Nf))
    R_sf = B @ B.conj().T + 0.5 * np.eye(Na * Nf)

    R_t, R_sf = prepare_correlations(R_t, R_sf, T=T, N_a=Na, N_f=Nf, eps_scale=1e-6)
    R_hh = np.kron(R_t, R_sf)
    assert R_hh.shape == (Nfull, Nfull)

    P_t = selection_matrix_pilot_times(pilot_times, T=T)
    P_f = selection_matrix_pilot_freqs(known_subcarriers, N_f=Nf)
    Sf = S_f_matrix(P_f, N_a=Na)
    Ph = np.kron(P_t, Sf)

    sigma_sq = 0.17
    W_bf = brute_force_lmmse_weights(R_hh, Ph, sigma_sq=sigma_sq)

    R_t_pp, R_t_p, R_sf_pp, R_sf_p = kronecker_blocks(
        R_t, R_sf, pilot_times=pilot_times, known_subcarriers=known_subcarriers, T=T, N_a=Na, N_f=Nf
    )
    W_k = compute_lmmse_weights_W(R_t_pp, R_t_p, R_sf_pp, R_sf_p, sigma_sq=sigma_sq)

    assert W_bf.shape == W_k.shape
    rel = np.linalg.norm(W_bf - W_k, ord="fro") / max(np.linalg.norm(W_bf, ord="fro"), 1e-12)
    assert rel < 1e-8


def test_pilot_observation_matrix_matches_kron() -> None:
    T, Na, Nf = 4, 2, 8
    pilot_times = [1, 3]
    known_subcarriers = [0, 2, 5]
    P_t = selection_matrix_pilot_times(pilot_times, T=T)
    P_f = selection_matrix_pilot_freqs(known_subcarriers, N_f=Nf)
    Sf = S_f_matrix(P_f, N_a=Na)
    Ph = np.kron(P_t, Sf)
    Ph2 = pilot_observation_matrix(pilot_times, known_subcarriers, T=T, N_a=Na, N_f=Nf)
    assert Ph.shape == Ph2.shape == (len(pilot_times) * len(known_subcarriers) * Na, T * Na * Nf)
    assert np.allclose(Ph, Ph2)


def test_estimate_R_hh_shapes() -> None:
    rng = np.random.default_rng(11)
    T, Na, Nf = 3, 2, 4
    chans = [
        rng.standard_normal((T, Na, Nf)) + 1j * rng.standard_normal((T, Na, Nf)) for _ in range(5)
    ]
    R_hh = estimate_R_hh(chans, T=T, N_a=Na, N_f=Nf)
    d = T * Na * Nf
    assert R_hh.shape == (d, d)
    assert np.allclose(R_hh, np.conjugate(R_hh.T))


def test_compute_lmmse_weights_W_full_matches_brute_force() -> None:
    rng = np.random.default_rng(12)
    T, Na, Nf = 3, 2, 4
    pilot_times = [0, 2]
    known_subcarriers = [1, 3]
    A = rng.standard_normal((T * Na * Nf, T * Na * Nf)) + 1j * rng.standard_normal((T * Na * Nf, T * Na * Nf))
    R_hh = A @ A.conj().T + 0.3 * np.eye(T * Na * Nf)
    R_hh = prepare_full_covariance(R_hh, dim=T * Na * Nf, eps_scale=1e-6)
    Ph = pilot_observation_matrix(pilot_times, known_subcarriers, T=T, N_a=Na, N_f=Nf)
    sigma_sq = 0.05
    W1 = brute_force_lmmse_weights(R_hh, Ph, sigma_sq=sigma_sq)
    W2 = compute_lmmse_weights_W_full(R_hh, Ph, sigma_sq=sigma_sq)
    assert np.allclose(W1, W2)


def test_estimate_correlations_shapes() -> None:
    rng = np.random.default_rng(3)
    T, Na, Nf = 3, 2, 4
    chans = [
        rng.standard_normal((T, Na, Nf)) + 1j * rng.standard_normal((T, Na, Nf)) for _ in range(4)
    ]
    R_t, R_sf = estimate_R_t_R_sf(chans, T=T, N_a=Na, N_f=Nf)
    assert R_t.shape == (T, T)
    assert R_sf.shape == (Na * Nf, Na * Nf)
    assert np.allclose(R_t, np.conjugate(R_t.T))
    assert np.allclose(R_sf, np.conjugate(R_sf.T))
