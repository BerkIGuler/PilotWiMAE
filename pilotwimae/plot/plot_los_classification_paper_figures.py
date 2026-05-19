from __future__ import annotations

"""
LOS classification accuracy vs SNR (fixed-SNR AWGN) for paper-style comparison plots.

Methods (folder under --results_root, same color in every figure):
  fst_scaleaux_noiserobust/  -> FST noise scale
  scaleaux_fst/              -> FST scale
  supervised/                -> FST supervised
  self_supervised/           -> JST / FST / FST noise (selected by checkpoint)

Four figures: (3.5GHz | 28GHz) x (in-distribution | OOD), inferred from path and/or data_dir.
Recursive `result_*.json` scan. Only fixed-SNR rows are used: `awgn.noise_floor` false or 0.

Examples:
  python -m pilotwimae.plot.plot_los_classification_paper_figures --png --out_dir results/knn_los/paper_figures

  python3 pilotwimae/plot/plot_los_classification_paper_figures.py --validate_only
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_BANDS = ("3.5GHz", "28GHz")
_SPLITS = ("in_dist", "ood")
_BASE_METHODS = (
    "fst_scaleaux_noiserobust",
    "scaleaux_fst",
    "supervised",
    "ckpt1",
    "ckpt2",
    "ckpt3",
)
_TOKEN_MODES = ("full", "pilot")

_BASE_METHOD_LABEL = {
    "fst_scaleaux_noiserobust": "FST noise scale",
    "scaleaux_fst": "FST scale",
    "supervised": "FST supervised",
    "ckpt1": "JST",
    "ckpt2": "FST",
    "ckpt3": "FST noise",
}
_BASE_METHOD_COLOR = {
    "fst_scaleaux_noiserobust": "tab:gray",
    "scaleaux_fst": "tab:brown",
    "supervised": "tab:red",
    "ckpt1": "tab:purple",
    "ckpt2": "tab:blue",
    "ckpt3": "tab:cyan",
}
_BASE_METHOD_MARKER = {
    "fst_scaleaux_noiserobust": "X",
    "scaleaux_fst": "D",
    "supervised": "^",
    "ckpt1": "s",
    "ckpt2": "o",
    "ckpt3": "*",
}

_ID_EXPECTED_CITIES = 4
_OOD_EXPECTED_CITIES = 1

_MARKER_SIZE = 6.5
_LABEL_FONT_SIZE = 15
_TITLE_FONT_SIZE = 15
_TICK_FONT_SIZE = 13
_LEGEND_FONT_SIZE = 12
_Y_LIM = (0.71, 0.96)
_FIG_SIZE = (10, 6.5)

_CKPT1_RUN_NAME = (
    "pilotwimae_standard_linear_sincat_1x4x4_128d_ffn4x_6L8h_2L4h_"
    "adamw_cosine_mse_normpatch_100ep_bs512_lr1e-3_wd5e-3_random_mask0.95"
)
_CKPT2_RUN_NAME = (
    "pilotwimae_factorized_linear_sincat_1x4x4_128d_ffn4x_3blocks8h_2L4h_"
    "adamw_cosine_mse_normpatch_300ep_bs512_lr5e-4_wd5e-3_tkeep2_smask0.9"
)
_CKPT3_PATH_SUFFIX = (
    "runs/pilotwimae_factorized_linear_sincat_1x4x4_128d_ffn4x_3blocks8h_2L4h_"
    "adamw_cosine_mse_normpatch_200ep_bs512_lr1e-4_wd5e-3_tkeep2_smask0.9_"
    "noiserobust_snr40/last_checkpoint.pt"
)


def _slug(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _require_key(d: dict, key: str, ctx: str):
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {ctx}")
    return d[key]


def _json_files(results_root: Path) -> List[Path]:
    files = sorted(p for p in results_root.rglob("result_*.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No result_*.json files found under: {results_root}")
    return files


def _is_fixed_snr_row(data: dict) -> bool:
    awgn = data.get("awgn")
    if not isinstance(awgn, dict):
        return False
    v = awgn.get("noise_floor", object())
    if v is False:
        return True
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
        return True
    return False


def _infer_band_split_from_data_dir(data_dir: str) -> Tuple[Optional[str], Optional[str]]:
    d = data_dir.replace("\\", "/")
    if "ood_test_35" in d:
        return "3.5GHz", "ood"
    if "ood_test_28" in d:
        return "28GHz", "ood"
    if "test_35" in d:
        return "3.5GHz", "in_dist"
    if "test_28" in d:
        return "28GHz", "in_dist"
    return None, None


def _band_split_for_row(path: Path, data: dict) -> Tuple[Optional[str], Optional[str]]:
    parts = path.parts
    band_p = next((b for b in _BANDS if b in parts), None)
    split_p = next((s for s in _SPLITS if s in parts), None)
    data_dir = str(data.get("experiment", {}).get("data_dir", ""))
    band_d, split_d = _infer_band_split_from_data_dir(data_dir)
    band = band_p or band_d
    split = split_p or split_d
    if band_p is not None and band_d is not None and band_p != band_d:
        raise ValueError(f"Band mismatch path={band_p} vs data_dir={band_d}: {path}")
    if split_p is not None and split_d is not None and split_p != split_d:
        raise ValueError(f"Split mismatch path={split_p} vs data_dir={split_d}: {path}")
    return band, split


def _pilot_pattern(data: dict) -> str:
    model = data.get("model", {})
    if isinstance(model, dict) and "pilot_pattern" in model:
        return str(model["pilot_pattern"])
    return ""


def _token_mode(data: dict) -> Optional[str]:
    model = data.get("model", {})
    mode = str(model.get("inference_token_mode", "full_grid"))
    if mode == "full_grid":
        return "full"
    if mode == "pilot_visible":
        return "pilot"
    return None


def _dataset_id(data: dict) -> str:
    exp = data.get("experiment", {})
    if isinstance(exp, dict) and exp.get("dataset_id"):
        return str(exp["dataset_id"])
    data_dir = str(exp.get("data_dir", "")).rstrip("/")
    return Path(data_dir).name


def _extract_accuracy_curve(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    snrs = np.array([float(x) for x in data["snrs_db"]], dtype=float)
    by_snr: Dict[float, Tuple[float, float]] = {}
    for row in data["snr_robustness"]:
        if str(row.get("snr_db", "")).lower() == "clean":
            continue
        acc = row["accuracy"]
        by_snr[float(row["snr_db"])] = (float(acc["mean"]), float(acc["std"]))
    means: List[float] = []
    stds: List[float] = []
    for s in snrs:
        if float(s) not in by_snr:
            raise ValueError(f"Missing snr={s} in snr_robustness")
        m, sd = by_snr[float(s)]
        means.append(m)
        stds.append(sd)
    return snrs, np.array(means, dtype=float), np.array(stds, dtype=float)


def _validate_los_schema(data: dict, path: Path) -> None:
    ctx = str(path)
    if data.get("task") != "los_binary_classification":
        raise ValueError(f"Expected task=los_binary_classification in {ctx}")
    _require_key(data, "snrs_db", ctx)
    _require_key(data, "snr_robustness", ctx)
    exp = _require_key(data, "experiment", ctx)
    _require_key(exp, "dataset_id", ctx)
    _require_key(exp, "checkpoint_run_name", ctx)
    _require_key(exp, "data_dir", ctx)
    model = _require_key(data, "model", ctx)
    _require_key(model, "checkpoint_path", ctx)
    awgn = _require_key(data, "awgn", ctx)
    _require_key(awgn, "noise_floor", ctx)
    for row in data["snr_robustness"]:
        if not isinstance(row, dict):
            raise TypeError(f"snr_robustness rows must be objects in {ctx}")
        if str(row.get("snr_db", "")).lower() == "clean":
            continue
        acc = _require_key(row, "accuracy", ctx)
        _require_key(acc, "mean", ctx)
        _require_key(acc, "std", ctx)


def _from_self_supervised_bucket(data: dict) -> Optional[str]:
    exp = data["experiment"]
    model = data["model"]
    run_name = str(exp.get("checkpoint_run_name", ""))
    checkpoint_path = str(model.get("checkpoint_path", "")).replace("\\", "/")

    if run_name == _CKPT1_RUN_NAME:
        return "ckpt1"
    if run_name == _CKPT2_RUN_NAME:
        return "ckpt2"
    if checkpoint_path.endswith(_CKPT3_PATH_SUFFIX):
        return "ckpt3"
    return None


def _base_method_for_row(path: Path, data: dict) -> Optional[str]:
    parts = path.parts
    if "fst_scaleaux_noiserobust" in parts:
        return "fst_scaleaux_noiserobust"
    if "scaleaux_fst" in parts:
        return "scaleaux_fst"
    if "supervised" in parts:
        return "supervised"
    if "self_supervised" in parts:
        return _from_self_supervised_bucket(data)
    return None


def _run_signature(data: dict, base_method: str) -> str:
    if base_method in ("ckpt1", "ckpt2", "ckpt3"):
        return str(data["model"]["checkpoint_path"])
    return str(data["experiment"]["checkpoint_run_name"])


def _aggregate_city_curves(
    rows: Iterable[Tuple[Path, dict]],
    *,
    pilot_pattern: str,
) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    """
    (band, split, method) -> {snr, city_to_mean, city_to_std}
    """
    out: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    seen_sig: Dict[Tuple[str, str, str], set[str]] = {}

    for path, data in rows:
        if not _is_fixed_snr_row(data):
            continue
        _validate_los_schema(data, path)
        mode = _token_mode(data)
        if mode is None:
            continue
        if mode == "pilot" and pilot_pattern and _pilot_pattern(data) != pilot_pattern:
            continue
        base_method = _base_method_for_row(path, data)
        if base_method is None:
            continue
        method = f"{base_method}__{mode}"
        band, split = _band_split_for_row(path, data)
        if band is None or split is None:
            continue

        city = _dataset_id(data)
        sig = _run_signature(data, base_method)
        gkey = (band, split, method)
        seen_sig.setdefault(gkey, set()).add(sig)

        snr, mean, std = _extract_accuracy_curve(data)
        if gkey not in out:
            out[gkey] = {"snr": snr, "city_to_mean": {}, "city_to_std": {}}
        else:
            ref = out[gkey]["snr"]  # type: ignore[index]
            if list(ref) != list(snr):
                raise ValueError(f"Inconsistent snr grid for {gkey}: {path}")

        city_to_mean: Dict[str, np.ndarray] = out[gkey]["city_to_mean"]  # type: ignore[index]
        city_to_std: Dict[str, np.ndarray] = out[gkey]["city_to_std"]  # type: ignore[index]
        if city in city_to_mean or city in city_to_std:
            raise ValueError(
                f"Duplicate city={city} for {gkey}: {path}. "
                "Use a tighter filter or remove duplicates."
            )
        city_to_mean[city] = mean
        city_to_std[city] = std

    for k, sigs in seen_sig.items():
        if len(sigs) > 1:
            raise ValueError(
                f"Multiple checkpoint signatures for {k}: {', '.join(sorted(sigs))}. "
                "Expected one checkpoint variant per method/mode."
            )
    return out


def _panel_curve(
    agg: Dict[Tuple[str, str, str], Dict[str, object]],
    *,
    band: str,
    split: str,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (band, split, method)
    if key not in agg:
        raise ValueError(f"Missing curve for {key}")
    snr = agg[key]["snr"]  # type: ignore[index]
    city_to_mean: Dict[str, np.ndarray] = agg[key]["city_to_mean"]  # type: ignore[index]
    city_to_std: Dict[str, np.ndarray] = agg[key]["city_to_std"]  # type: ignore[index]

    n_expected = _ID_EXPECTED_CITIES if split == "in_dist" else _OOD_EXPECTED_CITIES
    if len(city_to_mean) != n_expected or len(city_to_std) != n_expected:
        raise ValueError(
            f"{key}: expected {n_expected} city curves, got {len(city_to_mean)} "
            f"({sorted(city_to_mean.keys())})"
        )

    cities = sorted(city_to_mean.keys())
    mean_stack = np.stack([city_to_mean[c] for c in cities], axis=0)
    std_stack = np.stack([city_to_std[c] for c in cities], axis=0)
    mean = mean_stack.mean(axis=0)
    std = std_stack.mean(axis=0)
    return snr, mean, std


def _shared_ylim_across_panels(agg: Dict[Tuple[str, str, str], Dict[str, object]]) -> Tuple[float, float]:
    _ = agg
    return _Y_LIM


def _plot_panel(
    agg: Dict[Tuple[str, str, str], Dict[str, object]],
    *,
    band: str,
    split: str,
    y_lim: Tuple[float, float],
    out_dir: Path,
    write_png: bool,
    write_pdf: bool,
    no_title: bool,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    wrote: List[Path] = []

    ref_snr: Optional[np.ndarray] = None
    for base_method in _BASE_METHODS:
        for mode in _TOKEN_MODES:
            method = f"{base_method}__{mode}"
            snr, mean, std = _panel_curve(agg, band=band, split=split, method=method)
            if ref_snr is None:
                ref_snr = snr
            elif list(ref_snr) != list(snr):
                raise ValueError(f"Inconsistent snr axis for panel {(band, split)}")

            color = _BASE_METHOD_COLOR[base_method]
            marker = _BASE_METHOD_MARKER[base_method]
            line_style = "-" if mode == "full" else "--"
            ax.plot(
                snr,
                mean,
                color=color,
                linestyle=line_style,
                linewidth=2.2,
                marker=marker,
                markersize=_MARKER_SIZE,
            )
            lo = np.clip(mean - std, 0.0, 1.0)
            hi = np.clip(mean + std, 0.0, 1.0)
            ax.fill_between(snr, lo, hi, color=color, alpha=0.14)

    ax.set_ylim(y_lim[0], y_lim[1])
    split_name = "ID" if split == "in_dist" else "OOD"
    ax.set_xlabel("SNR (dB)", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
    if not no_title:
        ax.set_title(
            f"{split_name} @ {band}: LOS classification accuracy vs SNR",
            fontsize=_TITLE_FONT_SIZE,
            fontweight="bold",
        )
    ax.grid(True, linestyle="--", alpha=0.8, linewidth=1)
    ax.tick_params(axis="both", labelsize=_TICK_FONT_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    handles: List[Line2D] = []
    for base_method in _BASE_METHODS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_BASE_METHOD_COLOR[base_method],
                marker=_BASE_METHOD_MARKER[base_method],
                linestyle="-",
                linewidth=2.2,
                markersize=_MARKER_SIZE,
                label=_BASE_METHOD_LABEL[base_method],
            )
        )
    handles.extend(
        [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="Full channel"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="Pilot-only"),
        ]
    )
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=_LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=2.6,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    fig.tight_layout()

    band_slug = _slug(band)
    base = f"los_accuracy_vs_snr__{'id' if split == 'in_dist' else 'ood'}__{band_slug}__los_paper"
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
    p = argparse.ArgumentParser(
        description="Plot LOS classification accuracy vs SNR for paper figures."
    )
    p.add_argument(
        "--results_root",
        type=str,
        default="results/knn_los",
        help="Root scanned recursively for result_*.json.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: RESULTS_ROOT/paper_figures).",
    )
    p.add_argument(
        "--pilot_pattern",
        type=str,
        default="t:2,11;f:0,2,4,6",
        help="Only include JSON whose model.pilot_pattern matches (empty string disables filter).",
    )
    p.add_argument("--png", action="store_true", help="Write PNG output.")
    p.add_argument("--pdf", action="store_true", help="Write PDF output.")
    p.add_argument("--no_title", action="store_true", help="Disable plot titles.")
    p.add_argument(
        "--validate_only",
        action="store_true",
        help="Run coverage checks only; do not write plots.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.validate_only and not args.png and not args.pdf:
        raise SystemExit("Select --png and/or --pdf, or use --validate_only.")

    root = Path(args.results_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--results_root is not a directory: {root}")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else (root / "paper_figures").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_pat = str(args.pilot_pattern) if args.pilot_pattern else ""
    rows: List[Tuple[Path, dict]] = []
    for p in _json_files(root):
        with p.open("r", encoding="utf-8") as f:
            rows.append((p, json.load(f)))

    agg = _aggregate_city_curves(rows, pilot_pattern=pilot_pat)

    for band in _BANDS:
        for split in _SPLITS:
            for base_method in _BASE_METHODS:
                for mode in _TOKEN_MODES:
                    _ = _panel_curve(
                        agg,
                        band=band,
                        split=split,
                        method=f"{base_method}__{mode}",
                    )

    if args.validate_only:
        print("Validation OK: all four LOS panels have all method curves.")
        for band in _BANDS:
            for split in _SPLITS:
                print(f"  panel: {split} @ {band} -> {len(_BASE_METHODS) * len(_TOKEN_MODES)} curves")
        return

    y_lim = _shared_ylim_across_panels(agg)
    written: List[Path] = []
    for band in _BANDS:
        for split in _SPLITS:
            paths = _plot_panel(
                agg,
                band=band,
                split=split,
                y_lim=y_lim,
                out_dir=out_dir,
                write_png=bool(args.png),
                write_pdf=bool(args.pdf),
                no_title=bool(args.no_title),
            )
            written.extend(paths)
            print(f"Wrote panel: {split} @ {band}")

    print(f"Wrote {len(written)} plot file(s) to: {out_dir}")
    for pth in written:
        print(str(pth))


if __name__ == "__main__":
    main()
