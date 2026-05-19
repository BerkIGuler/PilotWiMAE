from __future__ import annotations

"""
Plot NMSE (dB) vs SNR (dB) from flat JSTSP result JSONs.

Examples:
  python -m pilotwimae.plot.plot_channel_prediction_paper_figures --results_dir results/channel_prediction/jstsp_figures/t211f0246 --png --pdf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_METHODS = (
    "supervised_fst",
    "supervised_jst",
    "pilotwimae_fst",
    "linear_interp",
    "lmmse_practical",
    "lmmse_gold",
)
_METHOD_LABEL = {
    "supervised_fst": "Supervised FST",
    "supervised_jst": "Supervised JST",
    "pilotwimae_fst": "PilotWiMAE (FST)",
    "linear_interp": "Linear interpolation",
    "lmmse_practical": "LMMSE practical",
    "lmmse_gold": "LMMSE gold",
}
_METHOD_COLOR = {
    "supervised_fst": "tab:red",
    "supervised_jst": "tab:green",
    "pilotwimae_fst": "tab:blue",
    "linear_interp": "tab:brown",
    "lmmse_practical": "tab:orange",
    "lmmse_gold": "tab:purple",
}
_METHOD_MARKER = {
    "supervised_fst": "^",
    "supervised_jst": "s",
    "pilotwimae_fst": "o",
    "linear_interp": "P",
    "lmmse_practical": "D",
    "lmmse_gold": "X",
}

_MARKER_SIZE = 6.5
_LABEL_FONT_SIZE = 15
_TITLE_FONT_SIZE = 15
_TICK_FONT_SIZE = 13
_LEGEND_FONT_SIZE = 12
_FIG_SIZE = (10, 6.5)


def _snr_dict_key(s: float) -> str:
    return str(int(s)) if float(s).is_integer() else str(s)


def _linear_to_db(x_lin: np.ndarray) -> np.ndarray:
    x_safe = np.maximum(x_lin, np.finfo(float).tiny)
    return 10.0 * np.log10(x_safe)


def _detect_method_from_content(data: dict) -> str:
    method = str(data.get("method", ""))
    if method == "linear_interpolation":
        return "linear_interp"
    if method == "lmmse_kronecker":
        corr_dirs = [str(x) for x in data.get("corr_dirs", [])]
        corr_text = " ".join(corr_dirs).replace("\\", "/").lower()
        if "ood_test_35" in corr_text:
            return "lmmse_gold"
        return "lmmse_practical"

    ck_kind = str(data.get("checkpoint_kind", "")).lower()
    ck = str(data.get("checkpoint", "")).lower()
    if ck_kind == "pilotwimae":
        return "pilotwimae_fst"
    if ck_kind == "temporalenc_ce" and "factorized" in ck:
        return "supervised_fst"
    if ck_kind == "temporalenc_ce" and "standard" in ck:
        return "supervised_jst"
    raise ValueError("Could not map result JSON to one of the 6 methods.")


def _extract_fold_linear_matrix(data: dict) -> np.ndarray:
    folds_linear = data.get("folds_nmse_linear")
    if isinstance(folds_linear, list) and folds_linear and isinstance(folds_linear[0], list):
        arr = np.asarray(folds_linear, dtype=float)
        if arr.ndim != 2:
            raise ValueError("folds_nmse_linear must be 2D [n_folds, n_snrs].")
        return arr

    fold_means = data.get("nmse_fold_mean_linear_by_snr")
    if isinstance(fold_means, dict):
        snrs = [float(x) for x in data["snrs_db"]]
        cols: List[np.ndarray] = []
        for s in snrs:
            k = _snr_dict_key(s)
            vals = fold_means.get(k)
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"Missing nmse_fold_mean_linear_by_snr[{k}]")
            cols.append(np.asarray(vals, dtype=float))
        n_folds = cols[0].shape[0]
        for c in cols:
            if c.shape[0] != n_folds:
                raise ValueError("Inconsistent fold count across SNR keys.")
        return np.stack(cols, axis=1)

    raise ValueError("Result JSON must contain folds_nmse_linear or nmse_fold_mean_linear_by_snr.")


def _curve_from_result(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snrs = np.asarray([float(x) for x in data["snrs_db"]], dtype=float)
    fold_mat = _extract_fold_linear_matrix(data)
    if fold_mat.shape[1] != snrs.shape[0]:
        raise ValueError("Fold matrix SNR dimension does not match snrs_db.")

    mean_lin = fold_mat.mean(axis=0)
    std_lin = fold_mat.std(axis=0, ddof=1) if fold_mat.shape[0] > 1 else np.zeros_like(mean_lin)
    lo_lin = np.maximum(mean_lin - std_lin, np.finfo(float).tiny)
    hi_lin = np.maximum(mean_lin + std_lin, np.finfo(float).tiny)
    mean_db = _linear_to_db(mean_lin)
    lo_db = _linear_to_db(lo_lin)
    hi_db = _linear_to_db(hi_lin)
    return snrs, mean_db, lo_db, hi_db


def _load_results_flat(results_dir: Path, pilot_pattern: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    files = sorted(
        p for p in results_dir.glob("result_*.json") if p.is_file() and p.stem.split("_")[-1].isdigit()
    )
    if not files:
        raise FileNotFoundError(f"No numbered result_*.json files found in: {results_dir}")

    curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        pp = str(data.get("pilot_pattern", ""))
        if pilot_pattern and pp != pilot_pattern:
            continue
        method = _detect_method_from_content(data)
        if method in curves:
            raise ValueError(f"Duplicate JSONs mapped to method={method}: {p}")
        curves[method] = _curve_from_result(data)

    missing = [m for m in _METHODS if m not in curves]
    if missing:
        raise ValueError(f"Missing methods after loading flat results: {missing}")
    return curves


def _plot(curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], out_dir: Path, *, write_png: bool, write_pdf: bool, no_title: bool) -> List[Path]:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    wrote: List[Path] = []

    ref_snr = None
    for method in _METHODS:
        snr, mean_db, lo_db, hi_db = curves[method]
        if ref_snr is None:
            ref_snr = snr
        elif list(ref_snr) != list(snr):
            raise ValueError("Inconsistent SNR grid across methods.")

        ax.plot(
            snr,
            mean_db,
            color=_METHOD_COLOR[method],
            linestyle="-",
            linewidth=2.2,
            marker=_METHOD_MARKER[method],
            markersize=_MARKER_SIZE,
        )
        ax.fill_between(snr, lo_db, hi_db, color=_METHOD_COLOR[method], alpha=0.14)

    ax.set_xlabel("SNR (dB)", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel("NMSE (dB)", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
    if not no_title:
        ax.set_title("OOD @ 3.5GHz: channel NMSE vs SNR", fontsize=_TITLE_FONT_SIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.8, linewidth=1)
    ax.tick_params(axis="both", labelsize=_TICK_FONT_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=_METHOD_COLOR[m],
            marker=_METHOD_MARKER[m],
            linestyle="-",
            linewidth=2.2,
            markersize=_MARKER_SIZE,
            label=_METHOD_LABEL[m],
        )
        for m in _METHODS
    ]
    leg = ax.legend(handles=handles, loc="best", fontsize=_LEGEND_FONT_SIZE, frameon=False, handlelength=2.6)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    fig.tight_layout()
    base = "nmse_vs_snr__ood__3p5ghz__jstsp_flat"
    if write_png:
        p = out_dir / f"{base}.png"
        fig.savefig(p, dpi=220)
        wrote.append(p)
    if write_pdf:
        p = out_dir / f"{base}.pdf"
        fig.savefig(p)
        wrote.append(p)
    plt.close(fig)
    return wrote


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot JSTSP flat channel NMSE results.")
    p.add_argument(
        "--results_dir",
        type=str,
        default="results/channel_prediction/jstsp_figures/t211f0246",
        help="Flat directory containing numbered result_1.json ... result_6.json.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: RESULTS_DIR).",
    )
    p.add_argument(
        "--pilot_pattern",
        type=str,
        default="t:2,11;f:0,2,4,6",
        help="Only include JSONs with this pilot_pattern (empty string disables filter).",
    )
    p.add_argument("--png", action="store_true", help="Write PNG output.")
    p.add_argument("--pdf", action="store_true", help="Write PDF output.")
    p.add_argument("--no_title", action="store_true", help="Disable plot title.")
    p.add_argument("--validate_only", action="store_true", help="Validate input mapping only.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.validate_only and not args.png and not args.pdf:
        raise SystemExit("Select --png and/or --pdf, or use --validate_only.")

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"--results_dir is not a directory: {results_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = _load_results_flat(results_dir, pilot_pattern=str(args.pilot_pattern) if args.pilot_pattern else "")
    print("Validated method mapping:")
    for m in _METHODS:
        print(f"  - {_METHOD_LABEL[m]}")

    if args.validate_only:
        print("Validation OK.")
        return

    written = _plot(
        curves,
        out_dir,
        write_png=bool(args.png),
        write_pdf=bool(args.pdf),
        no_title=bool(args.no_title),
    )
    print(f"Wrote {len(written)} plot file(s) to: {out_dir}")
    for p in written:
        print(str(p))


if __name__ == "__main__":
    main()
