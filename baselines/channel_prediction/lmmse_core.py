"""LMMSE channel estimator: Kronecker factor prior and full sample covariance (spec in ``context/LMMSE_implementation/lmmse_baseline.tex``)."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def vec_time_ant_freq(H: np.ndarray) -> np.ndarray:
    """
    Column vector ``h = vec(H)`` with time slowest, antenna fastest within each time slice.

    ``H`` shape ``(T, N_a, N_f)`` (complex). Flatten order matches C-order on ``(T, N_a, N_f)``:
    index ``t * (N_a * N_f) + a * N_f + f``.
    """
    H = np.asarray(H)
    if H.ndim != 3:
        raise ValueError(f"H must be 3D [T,N_a,N_f], got shape {H.shape}")
    return np.reshape(H, (-1,), order="C")


def unvec_time_ant_freq(h: np.ndarray, *, T: int, N_a: int, N_f: int) -> np.ndarray:
    """Inverse of :func:`vec_time_ant_freq`."""
    v = np.asarray(h).reshape(-1)
    expected = int(T) * int(N_a) * int(N_f)
    if v.size != expected:
        raise ValueError(f"vec length {v.size} != T*N_a*N_f={expected}")
    return v.reshape((int(T), int(N_a), int(N_f)), order="C")


def vec_slab_ant_freq(Hslab: np.ndarray) -> np.ndarray:
    """Vec one time slice ``[N_a, N_f]`` with antenna varying fastest: length ``N_a * N_f``."""
    s = np.asarray(Hslab)
    if s.ndim != 2:
        raise ValueError("slab must be [N_a, N_f]")
    Na, Nf = s.shape
    return np.reshape(s, (Na * Nf,), order="C")


def selection_matrix_pilot_times(pilot_times: list[int], *, T: int) -> np.ndarray:
    """Matrix ``P_t`` shape ``(|T_p|, T)`` — each row selects one pilot symbol index."""
    pt = np.asarray(pilot_times, dtype=np.int64)
    P = np.zeros((pt.size, int(T)), dtype=np.float64)
    for i, t in enumerate(pt):
        if t < 0 or t >= int(T):
            raise ValueError(f"pilot time {t} out of range for T={T}")
        P[i, int(t)] = 1.0
    return P


def selection_matrix_pilot_freqs(known_subcarriers: list[int], *, N_f: int) -> np.ndarray:
    """Matrix ``P_f`` shape ``(|F_p|, N_f)`` — each row selects one subcarrier index."""
    fs = np.asarray(known_subcarriers, dtype=np.int64)
    P = np.zeros((fs.size, int(N_f)), dtype=np.float64)
    for i, f in enumerate(fs):
        if f < 0 or f >= int(N_f):
            raise ValueError(f"pilot subcarrier {f} out of range for N_f={N_f}")
        P[i, int(f)] = 1.0
    return P


def S_f_matrix(P_f: np.ndarray, *, N_a: int) -> np.ndarray:
    """``S_f = I_{N_a} ⊗ P_f``, shape ``(N_a*|F_p|, N_a*N_f)``."""
    return np.kron(np.eye(int(N_a), dtype=P_f.dtype), P_f)


def pilot_observation_matrix(
    pilot_times: list[int],
    known_subcarriers: list[int],
    *,
    T: int,
    N_a: int,
    N_f: int,
) -> np.ndarray:
    """
    Selector ``Ph`` with ``Ph @ vec(H)`` equal to :func:`stack_pilot_vector` output.

    Shape ``(n_p, T N_a N_f)`` with ``n_p = |T_p| |F_p| N_a``.
    """
    P_t = selection_matrix_pilot_times(pilot_times, T=T)
    P_f = selection_matrix_pilot_freqs(known_subcarriers, N_f=N_f)
    Sf = S_f_matrix(P_f, N_a=int(N_a))
    return np.kron(P_t, Sf)


def stack_pilot_vector(H: np.ndarray, *, pilot_times: list[int], known_subcarriers: list[int]) -> np.ndarray:
    """
    Stack noisy LS pilots ``ĥ_p`` length ``|T_p||F_p|N_a``.

    Uses the same ordering as ``(P_t ⊗ S_f) vec(H)`` with :func:`pilot_observation_matrix`.
    """
    H = np.asarray(H)
    T, Na, Nf = H.shape
    Ph = pilot_observation_matrix(pilot_times, known_subcarriers, T=T, N_a=Na, N_f=Nf)
    return Ph @ vec_time_ant_freq(H)


def estimate_R_t_R_sf(
    channels: Iterable[np.ndarray],
    *,
    T: int,
    N_a: int,
    N_f: int,
    dtype: np.dtype = np.complex128,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample-average estimates of ``R_t`` (TeX §4.1) and ``R_sf`` (§4.2)."""
    R_t = np.zeros((int(T), int(T)), dtype=dtype)
    Nsf = int(N_a) * int(N_f)
    R_sf = np.zeros((Nsf, Nsf), dtype=dtype)
    n_t = 0
    n_sf = 0
    for H in channels:
        H = np.asarray(H, dtype=dtype)
        if H.shape != (int(T), int(N_a), int(N_f)):
            raise ValueError(f"Expected shape ({T},{N_a},{N_f}), got {H.shape}")
        for a in range(int(N_a)):
            for f in range(int(N_f)):
                v = H[:, a, f]
                R_t += np.outer(v, np.conjugate(v))
                n_t += 1
        for t in range(int(T)):
            v = vec_slab_ant_freq(H[t, :, :])
            R_sf += np.outer(v, np.conjugate(v))
            n_sf += 1
    if n_t == 0:
        raise ValueError("No samples for correlation estimation")
    R_t /= float(n_t)
    R_sf /= float(n_sf)
    return R_t, R_sf


def estimate_R_hh(
    channels: Iterable[np.ndarray],
    *,
    T: int,
    N_a: int,
    N_f: int,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """Sample covariance ``R_hh = E[h h^H]`` with ``h = vec(H)`` (:func:`vec_time_ant_freq`)."""
    d = int(T) * int(N_a) * int(N_f)
    R = np.zeros((d, d), dtype=dtype)
    n = 0
    for H in channels:
        H = np.asarray(H, dtype=dtype)
        if H.shape != (int(T), int(N_a), int(N_f)):
            raise ValueError(f"Expected shape ({T},{N_a},{N_f}), got {H.shape}")
        h = vec_time_ant_freq(H)
        R += np.outer(h, np.conjugate(h))
        n += 1
    if n == 0:
        raise ValueError("No samples for correlation estimation")
    R /= float(n)
    return R


def hermitian_symmetrize(R: np.ndarray) -> np.ndarray:
    return 0.5 * (R + np.conjugate(R.T))


def trace_normalize(R: np.ndarray, *, target_trace: float) -> np.ndarray:
    tr = float(np.real(np.trace(R)))
    if tr <= 0:
        raise ValueError("trace normalization failed: non-positive trace")
    return R * (float(target_trace) / tr)


def relative_diagonal_load(R: np.ndarray, *, dim: int, eps_scale: float = 1e-6) -> np.ndarray:
    tr = float(np.real(np.trace(R)))
    eps = float(eps_scale) * tr / float(dim)
    out = R + eps * np.eye(R.shape[0], dtype=R.dtype)
    return out


def prepare_correlations(
    R_t: np.ndarray,
    R_sf: np.ndarray,
    *,
    T: int,
    N_a: int,
    N_f: int,
    eps_scale: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Hermitize, trace-normalize (``tr(R_t)=T``, ``tr(R_sf)=N_a N_f``), diagonal load."""
    R_t = hermitian_symmetrize(R_t)
    R_sf = hermitian_symmetrize(R_sf)
    R_t = trace_normalize(R_t, target_trace=float(T))
    R_sf = trace_normalize(R_sf, target_trace=float(N_a * N_f))
    R_t = relative_diagonal_load(R_t, dim=int(T), eps_scale=eps_scale)
    R_sf = relative_diagonal_load(R_sf, dim=int(N_a * N_f), eps_scale=eps_scale)
    return R_t, R_sf


def prepare_full_covariance(
    R_hh: np.ndarray,
    *,
    dim: int,
    eps_scale: float = 1e-6,
) -> np.ndarray:
    """Hermitize, trace-normalize to ``tr(R)=dim``, diagonal load (same spirit as :func:`prepare_correlations`)."""
    R = hermitian_symmetrize(R_hh)
    R = trace_normalize(R, target_trace=float(dim))
    R = relative_diagonal_load(R, dim=int(dim), eps_scale=eps_scale)
    return R


def kronecker_blocks(
    R_t: np.ndarray,
    R_sf: np.ndarray,
    *,
    pilot_times: list[int],
    known_subcarriers: list[int],
    T: int,
    N_a: int,
    N_f: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``R_{t,pp}, R_{t,p}, R_{sf,pp}, R_{sf,p}``."""
    P_t = selection_matrix_pilot_times(pilot_times, T=T)
    P_f = selection_matrix_pilot_freqs(known_subcarriers, N_f=N_f)
    Sf = S_f_matrix(P_f, N_a=int(N_a))
    R_t_pp = P_t @ R_t @ P_t.T
    R_t_p = R_t @ P_t.T
    R_sf_pp = Sf @ R_sf @ Sf.conj().T
    R_sf_p = R_sf @ Sf.conj().T
    return R_t_pp, R_t_p, R_sf_pp, R_sf_p


def compute_lmmse_weights_W(
    R_t_pp: np.ndarray,
    R_t_p: np.ndarray,
    R_sf_pp: np.ndarray,
    R_sf_p: np.ndarray,
    *,
    sigma_sq: float,
) -> np.ndarray:
    """
    ``W = B C^{-1}`` with ``C = kron(R_t_pp, R_sf_pp) + σ² I``, ``B = kron(R_t_p, R_sf_p)``.

    Hermitian solve ``C X = B^H``, ``W = X^H``.
    """
    C = np.kron(R_t_pp, R_sf_pp) + float(sigma_sq) * np.eye(R_t_pp.shape[0] * R_sf_pp.shape[0], dtype=R_t_pp.dtype)
    B = np.kron(R_t_p, R_sf_p)
    C = hermitian_symmetrize(C)
    BH = B.conj().T
    X = np.linalg.solve(C, BH)
    return X.conj().T


def apply_W_to_pilots(W: np.ndarray, pilot_vec: np.ndarray) -> np.ndarray:
    """Full-channel vec ``ĥ = W ĥ_p`` (length ``T N_a N_f``)."""
    z = np.asarray(pilot_vec).reshape(-1)
    return (W @ z.astype(W.dtype, copy=False)).reshape(-1)


def lmmse_estimate_channel(
    H_obs_pilots: np.ndarray,
    *,
    W: np.ndarray,
    pilot_times: list[int],
    known_subcarriers: list[int],
    T: int,
    N_a: int,
    N_f: int,
) -> np.ndarray:
    """Form pilot vector from noisy grid ``H_obs`` (zeros elsewhere OK) and apply ``W``."""
    z = stack_pilot_vector(H_obs_pilots, pilot_times=pilot_times, known_subcarriers=known_subcarriers)
    h_hat = apply_W_to_pilots(W, z)
    return unvec_time_ant_freq(h_hat, T=T, N_a=N_a, N_f=N_f)


def brute_force_lmmse_weights(
    R_hh: np.ndarray,
    pilot_rows: np.ndarray,
    *,
    sigma_sq: float,
) -> np.ndarray:
    """
    Reference ``W_bf`` such that ``ĥ_full = W_bf ĥ_p`` under vec ordering of full grid.

    ``pilot_rows`` is ``(|p|, TN_aNf)`` selector ``P`` with one 1 per row (Kronecker pilot layout).
    """
    Ph = pilot_rows.astype(np.float64)
    n_p = Ph.shape[0]
    R_pp = Ph @ R_hh @ Ph.T
    R_fp = R_hh @ Ph.T
    C = R_pp + float(sigma_sq) * np.eye(n_p, dtype=R_pp.dtype)
    C = hermitian_symmetrize(C)
    return (np.linalg.solve(C, R_fp.conj().T)).conj().T


def compute_lmmse_weights_W_full(
    R_hh: np.ndarray,
    Ph: np.ndarray,
    *,
    sigma_sq: float,
) -> np.ndarray:
    """Full-covariance LMMSE ``W`` with ``ĥ = W ĥ_p``; alias for :func:`brute_force_lmmse_weights`."""
    return brute_force_lmmse_weights(R_hh, Ph, sigma_sq=sigma_sq)


def vec_brute_force_order(H: np.ndarray) -> np.ndarray:
    """Same layout as :func:`vec_time_ant_freq` (for brute-force tests)."""
    return vec_time_ant_freq(H)
