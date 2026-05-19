#!/usr/bin/env python3
"""Evaluate LMMSE baselines (Kronecker or full sample covariance) on NPZ data (see lmmse_baseline.tex)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from baselines.channel_prediction.lmmse_core import (
    compute_lmmse_weights_W,
    compute_lmmse_weights_W_full,
    estimate_R_hh,
    estimate_R_t_R_sf,
    kronecker_blocks,
    lmmse_estimate_channel,
    pilot_observation_matrix,
    prepare_correlations,
    prepare_full_covariance,
    stack_pilot_vector,
    unvec_time_ant_freq,
    vec_time_ant_freq,
)
from baselines.channel_prediction.metrics import nmse_non_pilot_tf
from baselines.channel_prediction.masks import (
    expanded_subcarrier_indices,
    non_pilot_mask_from_pilots,
    pilot_time_indices,
)
from baselines.channel_prediction.npz_io import (
    build_sorted_npz_list,
    compute_pref_reference_power,
    iter_channels,
)
from baselines.channel_prediction.pilot_noise import (
    apply_pilot_awgn,
    make_observed_from_target,
    parse_pilot_pattern,
)


def _parse_snrs(s: str) -> list[float]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("--snrs must include at least one value")
    return [float(v) for v in vals]


def _nmse_to_db(nmse_linear: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(nmse_linear), float(eps)))


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must be non-empty")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std}


def _say(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[lmmse] {msg}", flush=True)


def collect_npz_files(data_dirs: list[Path]) -> list[Path]:
    """Sorted NPZ paths across directories (de-duplicated by resolved path)."""
    seen: set[Path] = set()
    out: list[Path] = []
    for d in data_dirs:
        for p in build_sorted_npz_list(d):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return sorted(out, key=lambda x: str(x.resolve()))


def _normalize_channel(H: np.ndarray, pref_sqrt: float) -> np.ndarray:
    return H / float(pref_sqrt)


def _mean_pilot_power(obs: np.ndarray, *, pilot_times: list[int], known_subcarriers: list[int]) -> float:
    """Mean |·|² over pilot REs (same indexing as :func:`apply_pilot_awgn`)."""
    t_ix, n_ix, f_ix = np.ix_(
        np.asarray(pilot_times, dtype=np.int64),
        np.arange(obs.shape[1], dtype=np.int64),
        np.asarray(known_subcarriers, dtype=np.int64),
    )
    pilots = obs[t_ix, n_ix, f_ix]
    return float(np.mean(np.abs(pilots) ** 2))


def _resolve_torch_device(device_arg: str) -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Use --solver_backend=numpy or install torch.")
    d = str(device_arg).strip().lower()
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --torch_device=cuda but CUDA is not available.")
        return torch.device("cuda")
    if d.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested --torch_device={device_arg} but CUDA is not available.")
        try:
            idx = int(d.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Unsupported --torch_device: {device_arg}. Expected cuda, cuda:<index>, or cpu."
            ) from exc
        if idx < 0:
            raise ValueError(f"Unsupported --torch_device: {device_arg}. CUDA index must be >= 0.")
        num_devices = int(torch.cuda.device_count())
        if idx >= num_devices:
            raise RuntimeError(
                f"Requested --torch_device={device_arg}, but only {num_devices} CUDA device(s) are available."
            )
        return torch.device(f"cuda:{idx}")
    if d == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported --torch_device: {device_arg}. Expected cuda, cuda:<index>, or cpu.")


def _compute_lmmse_weights_torch(
    R_t_pp: np.ndarray,
    R_t_p: np.ndarray,
    R_sf_pp: np.ndarray,
    R_sf_p: np.ndarray,
    *,
    sigma_sq: float,
    device: torch.device,
) -> torch.Tensor:
    tdtype = torch.complex128
    R_t_pp_t = torch.as_tensor(R_t_pp, dtype=tdtype, device=device)
    R_t_p_t = torch.as_tensor(R_t_p, dtype=tdtype, device=device)
    R_sf_pp_t = torch.as_tensor(R_sf_pp, dtype=tdtype, device=device)
    R_sf_p_t = torch.as_tensor(R_sf_p, dtype=tdtype, device=device)
    C = torch.kron(R_t_pp_t, R_sf_pp_t)
    C = C + float(sigma_sq) * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    B = torch.kron(R_t_p_t, R_sf_p_t)
    C = 0.5 * (C + C.conj().transpose(0, 1))
    X = torch.linalg.solve(C, B.conj().transpose(0, 1))
    return X.conj().transpose(0, 1)


def _compute_lmmse_weights_torch_full(
    R_hh_t: torch.Tensor,
    Ph_t: torch.Tensor,
    *,
    sigma_sq: float,
) -> torch.Tensor:
    """``W`` such that ``vec(Ĥ) = W z`` with ``z`` the stacked pilot vector (matches numpy full LMMSE)."""
    tdtype = torch.complex128
    Ph_c = Ph_t.to(tdtype)
    R_pp = Ph_c @ R_hh_t @ Ph_c.transpose(0, 1)
    R_fp = R_hh_t @ Ph_c.transpose(0, 1)
    p = int(R_pp.shape[0])
    C = R_pp + float(sigma_sq) * torch.eye(p, dtype=R_pp.dtype, device=R_pp.device)
    C = 0.5 * (C + C.conj().transpose(0, 1))
    X = torch.linalg.solve(C, R_fp.conj().transpose(0, 1))
    return X.conj().transpose(0, 1)


def _compute_lmmse_weights_torch_full_batched(
    R_hh_t: torch.Tensor,
    Ph_t: torch.Tensor,
    sigma_sq: torch.Tensor,
) -> torch.Tensor:
    """Stack of ``W`` for batch size ``B``; ``sigma_sq`` shape ``(B,)``."""
    tdtype = torch.complex128
    Ph_c = Ph_t.to(tdtype)
    R_pp = Ph_c @ R_hh_t @ Ph_c.transpose(0, 1)
    R_fp = R_hh_t @ Ph_c.transpose(0, 1)
    p = int(R_pp.shape[0])
    B = int(sigma_sq.shape[0])
    eye = torch.eye(p, dtype=R_pp.dtype, device=R_pp.device)
    sig = sigma_sq.to(dtype=R_pp.real.dtype).reshape(B, 1, 1)
    C_stack = R_pp.unsqueeze(0) + sig * eye.unsqueeze(0)
    C_stack = 0.5 * (C_stack + C_stack.conj().transpose(-2, -1))
    rh = R_fp.conj().transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
    X = torch.linalg.solve(C_stack, rh)
    return X.conj().transpose(-2, -1)


def _estimate_R_t_R_sf_torch(
    corr_npz: list[Path],
    *,
    T: int,
    N_a: int,
    N_f: int,
    pref_sqrt: float,
    show_bar: bool,
    max_channels: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tdtype = torch.complex128
    nsf = int(N_a) * int(N_f)
    R_t_acc = torch.zeros((int(T), int(T)), dtype=tdtype, device=device)
    R_sf_acc = torch.zeros((nsf, nsf), dtype=tdtype, device=device)
    n = 0
    ch_it = iter_channels(corr_npz)
    if show_bar:
        ch_it = tqdm(ch_it, desc="Sample corr R_t, R_sf", unit="ch", dynamic_ncols=True)
    for H in ch_it:
        if max_channels > 0 and n >= int(max_channels):
            break
        Hn = _normalize_channel(H, pref_sqrt)
        H2d = np.asarray(Hn, dtype=np.complex128).reshape(int(T), nsf, order="C")
        H2d_t = torch.as_tensor(H2d, dtype=tdtype, device=device)
        R_t_acc = R_t_acc + (H2d_t @ H2d_t.conj().transpose(0, 1))
        R_sf_acc = R_sf_acc + (H2d_t.transpose(0, 1) @ H2d_t.conj())
        n += 1
    if n == 0:
        raise ValueError("No samples for correlation estimation")
    R_t = (R_t_acc / float(n * nsf)).detach().cpu().numpy()
    R_sf = (R_sf_acc / float(n * int(T))).detach().cpu().numpy()
    return R_t, R_sf


def _estimate_R_hh_torch(
    corr_npz: list[Path],
    *,
    T: int,
    N_a: int,
    N_f: int,
    pref_sqrt: float,
    show_bar: bool,
    max_channels: int,
    device: torch.device,
) -> np.ndarray:
    tdtype = torch.complex128
    d = int(T) * int(N_a) * int(N_f)
    R_acc = torch.zeros((d, d), dtype=tdtype, device=device)
    n = 0
    ch_it = iter_channels(corr_npz)
    if show_bar:
        ch_it = tqdm(ch_it, desc="Sample corr R_hh (full)", unit="ch", dynamic_ncols=True)
    for H in ch_it:
        if max_channels > 0 and n >= int(max_channels):
            break
        Hn = _normalize_channel(H, pref_sqrt)
        hv = vec_time_ant_freq(Hn).astype(np.complex128, copy=False)
        h_col = torch.as_tensor(hv, dtype=tdtype, device=device).reshape(d, 1)
        R_acc = R_acc + (h_col @ h_col.conj().transpose(0, 1))
        n += 1
    if n == 0:
        raise ValueError("No samples for correlation estimation")
    return (R_acc / float(n)).detach().cpu().numpy()


def _prepare_torch_per_channel_solver(
    R_t_pp: np.ndarray,
    R_t_p: np.ndarray,
    R_sf_pp: np.ndarray,
    R_sf_p: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tdtype = torch.complex128
    R_t_pp_t = torch.as_tensor(R_t_pp, dtype=tdtype, device=device)
    R_t_p_t = torch.as_tensor(R_t_p, dtype=tdtype, device=device)
    R_sf_pp_t = torch.as_tensor(R_sf_pp, dtype=tdtype, device=device)
    R_sf_p_t = torch.as_tensor(R_sf_p, dtype=tdtype, device=device)
    Rt_pp_h = 0.5 * (R_t_pp_t + R_t_pp_t.conj().transpose(0, 1))
    Rsf_pp_h = 0.5 * (R_sf_pp_t + R_sf_pp_t.conj().transpose(0, 1))
    eval_t, U_t = torch.linalg.eigh(Rt_pp_h)
    eval_sf, U_sf = torch.linalg.eigh(Rsf_pp_h)
    eval_t = torch.clamp(eval_t, min=0.0)
    eval_sf = torch.clamp(eval_sf, min=0.0)
    eval_kron = torch.outer(eval_t, eval_sf).reshape(-1)
    U_kron = torch.kron(U_t, U_sf)
    B_kron = torch.kron(R_t_p_t, R_sf_p_t)
    return {"U_kron": U_kron, "eval_kron": eval_kron, "B_kron": B_kron}


def _lmmse_apply_torch_per_channel_batch(
    Z: torch.Tensor, sigma_sq: torch.Tensor, *, solver: dict[str, torch.Tensor]
) -> torch.Tensor:
    U_kron = solver["U_kron"]
    eval_kron = solver["eval_kron"]
    B_kron = solver["B_kron"]
    z_proj = U_kron.conj().transpose(0, 1) @ Z
    inv = 1.0 / (eval_kron[:, None] + sigma_sq[None, :])
    x = U_kron @ (z_proj * inv.to(dtype=z_proj.real.dtype))
    return B_kron @ x


def main() -> None:
    parser = argparse.ArgumentParser(description="LMMSE baseline evaluation on NPZ channels.")
    parser.add_argument(
        "--corr_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories with NPZ channels for P_ref and correlation estimation.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory with NPZ test channels (same layout as training NPZs).",
    )
    parser.add_argument(
        "--pilot_pattern",
        type=str,
        default="t:2,11;f:0,2,4,6",
        help="Pilot layout in patch coordinates.",
    )
    parser.add_argument(
        "--freq_patch_size",
        type=int,
        default=4,
        help="Frequency patch size for expanding f-patch indices.",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory for result JSON.")
    parser.add_argument(
        "--output_stem",
        type=str,
        default="lmmse_baseline",
        help="Output JSON stem.",
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default="0,5,10,15,20,25,30",
        help='Comma-separated SNR(dB) sweep; interpretation depends on --pilot_snr_mode (see there).',
    )
    parser.add_argument(
        "--pilot_snr_mode",
        type=str,
        choices=("per_channel", "dataset"),
        default="per_channel",
        help=(
            "per_channel (default): mean pilot |h|^2 per channel defines signal power; AWGN variance "
            "is chosen so mean pilot SNR equals the sweep value, and LMMSE uses the matching σ² per "
            "channel (recomputes W each channel). "
            "dataset: legacy fixed reference — noise vs signal_mean_power=1 after P_ref and single σ² "
            "from the nominal SNR for all channels (one W per SNR)."
        ),
    )
    parser.add_argument(
        "--lmmse_model",
        type=str,
        choices=("kronecker", "full"),
        default="kronecker",
        help=(
            "kronecker: Kronecker prior R≈R_t⊗R_sf from separate time / space–freq marginals. "
            "full: sample covariance R_hh over vec(H) (same estimator as brute-force reference in tests)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for pilot AWGN.")
    parser.add_argument(
        "--num_folds",
        type=int,
        default=1,
        help="Split test samples into this many disjoint folds (by order); stats across folds.",
    )
    parser.add_argument(
        "--debug_size",
        type=int,
        default=0,
        help="If > 0, evaluate only the first debug_size test channels per SNR.",
    )
    parser.add_argument(
        "--debug_corr_size",
        type=int,
        default=0,
        help="If > 0, use only the first debug_corr_size correlation channels for fitting stats.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm bars.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable phase logging (tqdm still runs unless --no_progress).",
    )
    parser.add_argument(
        "--solver_backend",
        type=str,
        choices=("numpy", "torch"),
        default="numpy",
        help="Linear solve backend for LMMSE weights; use 'torch' for GPU/CPU torch solve.",
    )
    parser.add_argument(
        "--torch_device",
        type=str,
        default="cuda",
        help=(
            "Device when --solver_backend=torch: cpu, cuda (device 0), or cuda:N (e.g. cuda:0)."
        ),
    )
    parser.add_argument(
        "--torch_batch_size",
        type=int,
        default=256,
        help="Batch size for torch batched LMMSE apply.",
    )
    args = parser.parse_args()
    quiet = bool(args.quiet)
    show_bar = not bool(args.no_progress)
    per_channel_snr = str(args.pilot_snr_mode).strip().lower() == "per_channel"
    lmmse_model = str(args.lmmse_model).strip().lower()
    use_kronecker = lmmse_model == "kronecker"
    use_torch_backend = str(args.solver_backend).strip().lower() == "torch"
    debug_size = max(0, int(args.debug_size))
    debug_corr_size = max(0, int(args.debug_corr_size))
    torch_batch_size = max(1, int(args.torch_batch_size))
    torch_device = _resolve_torch_device(args.torch_device) if use_torch_backend else None
    _say(
        "Backend="
        + ("torch" if use_torch_backend else "numpy")
        + (f" device={torch_device}" if use_torch_backend else ""),
        quiet=quiet,
    )

    t_idx, f_patch_idx = parse_pilot_pattern(args.pilot_pattern)
    pilots_t = pilot_time_indices(t_idx)
    known_sc = expanded_subcarrier_indices(
        f_patch_idx,
        freq_patch_size=int(args.freq_patch_size),
    )

    corr_dirs = [Path(p).expanduser().resolve() for p in args.corr_dirs]
    test_dir = Path(args.test_dir).expanduser().resolve()
    _say(f"Collecting NPZ paths under corr_dirs: {', '.join(str(d) for d in corr_dirs)}", quiet=quiet)
    corr_npz = collect_npz_files(corr_dirs)
    _say(f"Indexing test NPZ under {test_dir}", quiet=quiet)
    test_npz = build_sorted_npz_list(test_dir)
    _say(
        f"Found {len(corr_npz)} correlation NPZ files, {len(test_npz)} test NPZ files",
        quiet=quiet,
    )

    _say("Computing reference power P_ref from correlation channels...", quiet=quiet)
    pref = compute_pref_reference_power(corr_npz, progress=show_bar)
    pref_sqrt = math.sqrt(pref)
    _say(f"P_ref={pref:.6e} (channels normalized by sqrt(P_ref))", quiet=quiet)

    snr_list = _parse_snrs(args.snrs)
    rng = np.random.default_rng(int(args.seed))
    K = max(1, int(args.num_folds))

    # Probe shape + NMSE mask
    _say("Probing one test channel for grid shape...", quiet=quiet)
    T = Na = Nf = None
    for H0 in iter_channels(test_npz):
        T, Na, Nf = H0.shape
        break
    if T is None:
        raise RuntimeError("No test channels found.")
    _say(f"Grid shape T={T}, N_a={Na}, N_f={Nf}; pilots t={pilots_t}, |F_p|={len(known_sc)}", quiet=quiet)
    tf_mask = non_pilot_mask_from_pilots(
        pilot_times=pilots_t,
        known_subcarriers=known_sc,
        T=int(T),
        N_a=int(Na),
        N_f=int(Nf),
    )

    torch_per_ch_solver = None
    R_hh: np.ndarray | None = None
    Ph_np: np.ndarray | None = None
    R_t_pp = R_t_p = R_sf_pp = R_sf_p = None

    if use_kronecker:
        _say("Accumulating sample Kronecker correlations on normalized correlation set...", quiet=quiet)
        if use_torch_backend:
            assert torch_device is not None
            R_t, R_sf = _estimate_R_t_R_sf_torch(
                corr_npz,
                T=int(T),
                N_a=int(Na),
                N_f=int(Nf),
                pref_sqrt=pref_sqrt,
                show_bar=show_bar,
                max_channels=debug_corr_size,
                device=torch_device,
            )
        else:

            def _corr_streams():
                ch_it = iter_channels(corr_npz)
                if show_bar:
                    ch_it = tqdm(
                        ch_it,
                        desc="Sample corr R_t, R_sf",
                        unit="ch",
                        dynamic_ncols=True,
                    )
                n_corr = 0
                for H in ch_it:
                    if debug_corr_size > 0 and n_corr >= debug_corr_size:
                        break
                    yield _normalize_channel(H, pref_sqrt)
                    n_corr += 1

            R_t, R_sf = estimate_R_t_R_sf(_corr_streams(), T=int(T), N_a=int(Na), N_f=int(Nf))
        _say("Regularizing correlations and building Kronecker pilot blocks...", quiet=quiet)
        R_t, R_sf = prepare_correlations(
            R_t, R_sf, T=int(T), N_a=int(Na), N_f=int(Nf), eps_scale=1e-6
        )
        R_t_pp, R_t_p, R_sf_pp, R_sf_p = kronecker_blocks(
            R_t,
            R_sf,
            pilot_times=pilots_t,
            known_subcarriers=known_sc,
            T=int(T),
            N_a=int(Na),
            N_f=int(Nf),
        )
        _say("Pilot-side covariance blocks ready.", quiet=quiet)
        if use_torch_backend:
            assert torch_device is not None
            torch_per_ch_solver = _prepare_torch_per_channel_solver(
                R_t_pp, R_t_p, R_sf_pp, R_sf_p, device=torch_device
            )
    else:
        _say("Accumulating full sample covariance R_hh on normalized correlation set...", quiet=quiet)
        if use_torch_backend:
            assert torch_device is not None
            R_hh = _estimate_R_hh_torch(
                corr_npz,
                T=int(T),
                N_a=int(Na),
                N_f=int(Nf),
                pref_sqrt=pref_sqrt,
                show_bar=show_bar,
                max_channels=debug_corr_size,
                device=torch_device,
            )
        else:

            def _corr_streams_full():
                ch_it = iter_channels(corr_npz)
                if show_bar:
                    ch_it = tqdm(
                        ch_it,
                        desc="Sample corr R_hh (full)",
                        unit="ch",
                        dynamic_ncols=True,
                    )
                n_corr = 0
                for H in ch_it:
                    if debug_corr_size > 0 and n_corr >= debug_corr_size:
                        break
                    yield _normalize_channel(H, pref_sqrt)
                    n_corr += 1

            R_hh = estimate_R_hh(_corr_streams_full(), T=int(T), N_a=int(Na), N_f=int(Nf))
        dim_full = int(T) * int(Na) * int(Nf)
        assert R_hh is not None
        _say(f"Regularizing full covariance (dim={dim_full})...", quiet=quiet)
        R_hh = prepare_full_covariance(R_hh, dim=dim_full, eps_scale=1e-6)
        Ph_np = pilot_observation_matrix(
            pilots_t, known_sc, T=int(T), N_a=int(Na), N_f=int(Nf)
        )
        _say("Pilot observation matrix Ph ready.", quiet=quiet)

    R_hh_t: torch.Tensor | None = None
    Ph_t: torch.Tensor | None = None
    if use_torch_backend and not use_kronecker:
        assert torch_device is not None and R_hh is not None and Ph_np is not None
        R_hh_t = torch.as_tensor(R_hh, dtype=torch.complex128, device=torch_device)
        Ph_t = torch.as_tensor(Ph_np, dtype=torch.float64, device=torch_device)

    # Materialize test channels (fold indexing)
    _say("Loading and normalizing all test channels...", quiet=quiet)
    _tc_it = iter_channels(test_npz)
    if show_bar:
        _tc_it = tqdm(_tc_it, desc="Load test channels", unit="ch", dynamic_ncols=True)
    test_channels = [_normalize_channel(H, pref_sqrt) for H in _tc_it]
    if debug_size > 0:
        test_channels = test_channels[:debug_size]
    n_samples = len(test_channels)
    if n_samples == 0:
        raise RuntimeError("Empty test set.")
    mode_note = "effective per-channel pilot SNR" if per_channel_snr else "dataset reference power + nominal σ²"
    _say(
        f"{n_samples} test channels in memory; evaluating {len(snr_list)} SNRs, K={K} folds "
        f"({mode_note})",
        quiet=quiet,
    )

    nmse_by_snr_linear: dict[str, dict[str, float]] = {}
    nmse_db_by_snr: dict[str, dict[str, float]] = {}
    fold_means_by_snr: dict[str, list[float]] = {}

    pilot_pwr_eps = 1e-30

    for snr_db in snr_list:
        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        if not per_channel_snr:
            _say(f"SNR {snr_db} dB: single LMMSE weight matrix W (dataset reference σ²)...", quiet=quiet)
            sigma_sq_fixed = 1.0 / snr_lin
            if use_torch_backend:
                assert torch_device is not None
                if use_kronecker:
                    assert R_t_pp is not None and R_t_p is not None and R_sf_pp is not None and R_sf_p is not None
                    W_fixed_t = _compute_lmmse_weights_torch(
                        R_t_pp,
                        R_t_p,
                        R_sf_pp,
                        R_sf_p,
                        sigma_sq=float(sigma_sq_fixed),
                        device=torch_device,
                    )
                else:
                    assert R_hh_t is not None and Ph_t is not None
                    W_fixed_t = _compute_lmmse_weights_torch_full(
                        R_hh_t, Ph_t, sigma_sq=float(sigma_sq_fixed)
                    )
            else:
                if use_kronecker:
                    assert R_t_pp is not None and R_t_p is not None and R_sf_pp is not None and R_sf_p is not None
                    W_fixed = compute_lmmse_weights_W(
                        R_t_pp, R_t_p, R_sf_pp, R_sf_p, sigma_sq=float(sigma_sq_fixed)
                    )
                else:
                    assert R_hh is not None and Ph_np is not None
                    W_fixed = compute_lmmse_weights_W_full(
                        R_hh, Ph_np, sigma_sq=float(sigma_sq_fixed)
                    )
        else:
            _say(
                f"SNR {snr_db} dB: per-channel σ² from pilot power / {snr_lin:.6g} "
                f"(recomputing W each channel)...",
                quiet=quiet,
            )

        # Per-fold mean NMSE (linear), then aggregate across folds
        fold_means: list[float] = []
        for fold_id in range(K):
            nmse_vals: list[float] = []
            indices = [i for i in range(n_samples) if (i % K) == fold_id]
            if use_torch_backend:
                assert torch_device is not None
                if use_kronecker:
                    assert torch_per_ch_solver is not None
                batch_iter = range(0, len(indices), torch_batch_size)
                if show_bar:
                    batch_iter = tqdm(
                        batch_iter,
                        total=(len(indices) + torch_batch_size - 1) // torch_batch_size,
                        desc=f"LMMSE estimate SNR={snr_db}dB fold {fold_id + 1}/{K}",
                        leave=True,
                        dynamic_ncols=True,
                    )
                for start in batch_iter:
                    idx_batch = indices[start : start + torch_batch_size]
                    z_cols: list[np.ndarray] = []
                    sigma_vals: list[float] = []
                    for idx in idx_batch:
                        target = test_channels[idx]
                        obs = make_observed_from_target(
                            target,
                            pilot_times=pilots_t,
                            known_subcarriers=known_sc,
                        )
                        if per_channel_snr:
                            p_s = max(
                                _mean_pilot_power(obs, pilot_times=pilots_t, known_subcarriers=known_sc),
                                pilot_pwr_eps,
                            )
                            sigma_sq_ch = float(p_s / snr_lin)
                            obs_noisy = apply_pilot_awgn(
                                obs,
                                pilot_times=pilots_t,
                                known_subcarriers=known_sc,
                                snr_db=float(snr_db),
                                signal_mean_power=None,
                                noise_floor=False,
                                rng=rng,
                            )
                        else:
                            sigma_sq_ch = float(sigma_sq_fixed)
                            obs_noisy = apply_pilot_awgn(
                                obs,
                                pilot_times=pilots_t,
                                known_subcarriers=known_sc,
                                snr_db=float(snr_db),
                                signal_mean_power=1.0,
                                noise_floor=True,
                                rng=rng,
                            )
                        z_noisy = stack_pilot_vector(
                            obs_noisy, pilot_times=pilots_t, known_subcarriers=known_sc
                        ).astype(np.complex128, copy=False)
                        z_cols.append(z_noisy)
                        sigma_vals.append(sigma_sq_ch)

                    Z_np = np.stack(z_cols, axis=1)
                    Z_t = torch.as_tensor(Z_np, dtype=torch.complex128, device=torch_device)
                    if use_kronecker:
                        if per_channel_snr:
                            sigma_t = torch.as_tensor(sigma_vals, dtype=torch.float64, device=torch_device)
                            H_hat_t = _lmmse_apply_torch_per_channel_batch(
                                Z_t, sigma_t, solver=torch_per_ch_solver
                            )
                        else:
                            H_hat_t = W_fixed_t @ Z_t
                    else:
                        assert R_hh_t is not None and Ph_t is not None
                        if per_channel_snr:
                            sigma_t = torch.as_tensor(sigma_vals, dtype=torch.float64, device=torch_device)
                            W_batch = _compute_lmmse_weights_torch_full_batched(R_hh_t, Ph_t, sigma_t)
                            Zp = Z_t.transpose(0, 1).unsqueeze(-1)
                            H_hat_t = torch.bmm(W_batch, Zp).squeeze(-1).transpose(0, 1)
                        else:
                            H_hat_t = W_fixed_t @ Z_t
                    H_hat_np = H_hat_t.detach().cpu().numpy()
                    for j, idx in enumerate(idx_batch):
                        target = test_channels[idx]
                        hat = unvec_time_ant_freq(H_hat_np[:, j], T=int(T), N_a=int(Na), N_f=int(Nf))
                        nmse_vals.append(float(nmse_non_pilot_tf(hat, target, tf_mask=tf_mask)))
            else:
                ch_iter = (test_channels[i] for i in indices)
                if show_bar:
                    ch_iter = tqdm(
                        ch_iter,
                        total=len(indices),
                        desc=f"LMMSE estimate SNR={snr_db}dB fold {fold_id + 1}/{K}",
                        leave=True,
                        dynamic_ncols=True,
                    )
                for target in ch_iter:
                    obs = make_observed_from_target(
                        target,
                        pilot_times=pilots_t,
                        known_subcarriers=known_sc,
                    )
                    if per_channel_snr:
                        p_s = max(_mean_pilot_power(obs, pilot_times=pilots_t, known_subcarriers=known_sc), pilot_pwr_eps)
                        sigma_sq_ch = float(p_s / snr_lin)
                        if use_kronecker:
                            assert R_t_pp is not None and R_t_p is not None and R_sf_pp is not None and R_sf_p is not None
                            W = compute_lmmse_weights_W(
                                R_t_pp, R_t_p, R_sf_pp, R_sf_p, sigma_sq=sigma_sq_ch
                            )
                        else:
                            assert R_hh is not None and Ph_np is not None
                            W = compute_lmmse_weights_W_full(R_hh, Ph_np, sigma_sq=sigma_sq_ch)
                        obs_noisy = apply_pilot_awgn(
                            obs,
                            pilot_times=pilots_t,
                            known_subcarriers=known_sc,
                            snr_db=float(snr_db),
                            signal_mean_power=None,
                            noise_floor=False,
                            rng=rng,
                        )
                    else:
                        W = W_fixed
                        obs_noisy = apply_pilot_awgn(
                            obs,
                            pilot_times=pilots_t,
                            known_subcarriers=known_sc,
                            snr_db=float(snr_db),
                            signal_mean_power=1.0,
                            noise_floor=True,
                            rng=rng,
                        )
                    hat = lmmse_estimate_channel(
                        obs_noisy,
                        W=W,
                        pilot_times=pilots_t,
                        known_subcarriers=known_sc,
                        T=int(T),
                        N_a=int(Na),
                        N_f=int(Nf),
                    )
                    nmse_vals.append(float(nmse_non_pilot_tf(hat, target, tf_mask=tf_mask)))
            if nmse_vals:
                fold_means.append(float(np.mean(np.asarray(nmse_vals, dtype=np.float64))))
        stats = _mean_std(fold_means)
        _say(
            f"SNR {snr_db} dB done: fold mean NMSE (linear) = {stats['mean']:.6e}",
            quiet=quiet,
        )

        key = str(int(snr_db) if float(snr_db).is_integer() else snr_db)
        nmse_by_snr_linear[key] = stats
        fold_means_by_snr[key] = fold_means

        mean_db = _nmse_to_db(stats["mean"])
        lo_db = _nmse_to_db(max(stats["mean"] - stats["std"], 0.0))
        hi_db = _nmse_to_db(stats["mean"] + stats["std"])
        nmse_db_by_snr[key] = {
            "mean": mean_db,
            "std_minus": mean_db - lo_db,
            "std_plus": hi_db - mean_db,
            "std": 0.5 * ((mean_db - lo_db) + (hi_db - mean_db)),
        }

    noise_mode_json = (
        "effective_per_channel_mean_pilot_power"
        if per_channel_snr
        else "dataset_reference_power_fixed_noise_floor_after_pref"
    )
    payload = {
        "method": f"lmmse_{lmmse_model}",
        "lmmse_model": lmmse_model,
        "metric_version": 3,
        "nmse_eval_non_pilot_only": True,
        "pilot_snr_mode": str(args.pilot_snr_mode),
        "solver_backend": "torch" if use_torch_backend else "numpy",
        "torch_device": str(args.torch_device) if use_torch_backend else None,
        "torch_batch_size": int(args.torch_batch_size),
        "noise_mode": noise_mode_json,
        "pref_reference_power": pref,
        "corr_dirs": [str(d) for d in corr_dirs],
        "test_dir": str(test_dir),
        "pilot_pattern": args.pilot_pattern,
        "freq_patch_size": int(args.freq_patch_size),
        "snrs_db": snr_list,
        "seed": int(args.seed),
        "num_folds": K,
        "num_test_samples": n_samples,
        "debug_size": int(args.debug_size),
        "debug_corr_size": int(args.debug_corr_size),
        "nmse_db_by_snr": nmse_db_by_snr,
        "nmse_by_snr_linear": nmse_by_snr_linear,
        "nmse_fold_mean_linear_by_snr": fold_means_by_snr,
    }

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.output_stem}.json"
    _say(f"Writing JSON to {out_path}", quiet=quiet)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    for snr_db in snr_list:
        key = str(int(snr_db) if float(snr_db).is_integer() else snr_db)
        d = nmse_db_by_snr[key]
        print(
            f"SNR={snr_db:>5} dB -> NMSE={d['mean']:.4f} dB "
            f"(lin mean over folds={nmse_by_snr_linear[key]['mean']:.6e})"
        )


if __name__ == "__main__":
    main()
