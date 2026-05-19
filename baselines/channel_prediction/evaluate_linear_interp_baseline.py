#!/usr/bin/env python3
"""Evaluate linear-interpolation channel-prediction baseline on NPZ data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from baselines.channel_prediction import (
    evaluate_linear_interpolation,
    expanded_subcarrier_indices,
    nmse_non_pilot_tf,
    non_pilot_mask_from_pilots,
    pilot_time_indices,
)
from baselines.channel_prediction.npz_io import (
    build_sorted_npz_list,
    compute_dataset_mean_complex_power,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear interpolation baseline evaluation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with NPZ files.")
    parser.add_argument(
        "--pilot_pattern",
        type=str,
        default="t:2,11;f:0,2,4,6",
        help="Pilot layout in patch coordinates, e.g. t:2,11;f:0,2,4,6",
    )
    parser.add_argument(
        "--freq_patch_size",
        type=int,
        default=4,
        help="Patch size on frequency axis for mapping f-patch indices to subcarriers.",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory for result JSON.")
    parser.add_argument(
        "--output_stem",
        type=str,
        default="linear_interp_baseline",
        help="Output JSON stem.",
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default="0,5,10,15,20,25,30",
        help='Comma-separated SNRs in dB (default "0,5,10,15,20,25,30").',
    )
    parser.add_argument(
        "--noise_floor",
        action="store_true",
        help=(
            "Use fixed global signal power (dataset mean |h|^2) to set pilot AWGN power. "
            "If omitted, use per-sample signal power for fixed-SNR mode."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for AWGN generation.")
    parser.add_argument(
        "--frequency_outside_mode",
        type=str,
        default="hold",
        choices=["hold", "linear"],
        help="How to fill frequencies outside pilot subcarrier span: hold or linear extrapolation.",
    )
    parser.add_argument(
        "--time_outside_mode",
        type=str,
        default="hold",
        choices=["hold", "linear"],
        help="How to fill times outside pilot interval: hold or linear extrapolation.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    t_idx, f_patch_idx = parse_pilot_pattern(args.pilot_pattern)
    pilots_t = pilot_time_indices(t_idx)
    known_subcarriers = expanded_subcarrier_indices(
        f_patch_idx,
        freq_patch_size=int(args.freq_patch_size),
    )

    data_dir = Path(args.data_dir).expanduser().resolve()
    npz_files = build_sorted_npz_list(data_dir)
    snr_list = _parse_snrs(args.snrs)
    rng = np.random.default_rng(int(args.seed))
    dataset_mean_power = (
        compute_dataset_mean_complex_power(npz_files) if args.noise_floor else None
    )

    T = N = F = None
    for _probe in iter_channels(npz_files):
        T, N, F = _probe.shape
        break
    if T is None:
        raise RuntimeError("No channels loaded")
    tf_mask = non_pilot_mask_from_pilots(
        pilot_times=pilots_t,
        known_subcarriers=known_subcarriers,
        T=int(T),
        N_a=int(N),
        N_f=int(F),
    )

    nmse_by_snr_linear: dict[str, dict[str, float]] = {}
    nmse_db_by_snr: dict[str, dict[str, float]] = {}
    mse_by_snr: dict[str, dict[str, float]] = {}
    n_samples = 0

    for snr_db in snr_list:
        nmse_vals: list[float] = []
        mse_vals: list[float] = []
        channels = iter_channels(npz_files)
        if not args.no_progress:
            channels = tqdm(
                channels,
                total=None,
                desc=f"SNR {snr_db} dB ({'noise_floor' if args.noise_floor else 'fixed_snr'})",
                leave=True,
                dynamic_ncols=True,
            )
        for target in channels:
            observed = make_observed_from_target(
                target,
                pilot_times=pilots_t,
                known_subcarriers=known_subcarriers,
            )
            observed_noisy = apply_pilot_awgn(
                observed,
                pilot_times=pilots_t,
                known_subcarriers=known_subcarriers,
                snr_db=float(snr_db),
                signal_mean_power=dataset_mean_power,
                noise_floor=bool(args.noise_floor),
                rng=rng,
            )
            out = evaluate_linear_interpolation(
                observed_noisy,
                target,
                pilot_times=pilots_t,
                known_subcarriers=known_subcarriers,
                time_axis=0,
                frequency_axis=2,
                frequency_outside_mode=args.frequency_outside_mode,
                time_outside_mode=args.time_outside_mode,
            )
            recon = out["reconstructed"]
            nmse_vals.append(float(nmse_non_pilot_tf(recon, target, tf_mask=tf_mask)))
            mse_vals.append(float(np.mean(np.abs(recon - target) ** 2)))
            if snr_db == snr_list[0]:
                n_samples += 1

        key = str(int(snr_db) if float(snr_db).is_integer() else snr_db)
        nmse_lin_stats = _mean_std(nmse_vals)
        mse_stats = _mean_std(mse_vals)
        nmse_by_snr_linear[key] = nmse_lin_stats
        mse_by_snr[key] = mse_stats
        mean_db = _nmse_to_db(nmse_lin_stats["mean"])
        lo_db = _nmse_to_db(max(nmse_lin_stats["mean"] - nmse_lin_stats["std"], 0.0))
        hi_db = _nmse_to_db(nmse_lin_stats["mean"] + nmse_lin_stats["std"])
        nmse_db_by_snr[key] = {
            "mean": mean_db,
            "std_minus": mean_db - lo_db,
            "std_plus": hi_db - mean_db,
            "std": 0.5 * ((mean_db - lo_db) + (hi_db - mean_db)),
        }

    payload = {
        "method": "linear_interpolation",
        "metric_version": 2,
        "nmse_eval_non_pilot_only": True,
        "data_dir": str(data_dir),
        "pilot_pattern": args.pilot_pattern,
        "freq_patch_size": int(args.freq_patch_size),
        "snrs_db": snr_list,
        "noise_floor": bool(args.noise_floor),
        "noise_mode": "fixed_noise_power" if args.noise_floor else "fixed_snr",
        "mean_complex_power_dataset": dataset_mean_power,
        "seed": int(args.seed),
        "frequency_outside_mode": str(args.frequency_outside_mode),
        "time_outside_mode": str(args.time_outside_mode),
        "num_samples": n_samples,
        "nmse_db_by_snr": nmse_db_by_snr,
        "nmse_by_snr_linear": nmse_by_snr_linear,
        "mse_by_snr": mse_by_snr,
    }

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.output_stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Noise mode: {payload['noise_mode']}")
    for snr_db in snr_list:
        key = str(int(snr_db) if float(snr_db).is_integer() else snr_db)
        d = nmse_db_by_snr[key]
        print(
            f"SNR={snr_db:>5} dB -> NMSE={d['mean']:.4f} dB "
            f"(lin mean={nmse_by_snr_linear[key]['mean']:.6e}, mse mean={mse_by_snr[key]['mean']:.6e})"
        )


if __name__ == "__main__":
    main()
