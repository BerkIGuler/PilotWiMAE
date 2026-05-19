from __future__ import annotations

"""
Top-K beam accuracy vs SNR (fixed-SNR AWGN) for paper-style comparison plots.

Curves (one figure per codebook size M ∈ {32, 128}; all curves solid; family colors fixed across figures):
  FST full / pilot — factorized self-supervised runs (tkeep2, 2L4h) without scaleaux in
    `checkpoint_run_name` (typically under .../self_supervised/...).
  FST + scale full / pilot — same FST recipe but run names containing `scaleaux`
    (typically .../self_supervised_scale_loss/...).
  FST noise full / pilot — factorized FST with training-time noise robustness (`noiserobust` in
    `checkpoint_run_name`; typically .../self_supervised_noise_robust/...).
  JST full / pilot — standard self-supervised (random_mask0.95, 2L4h); pilot uses
    `--pilot_pattern` (default t:2,11;f:0,2,4,6).
  FST supervised — supervised encoder, full-grid only.

Eight figures: in-distribution vs OOD × 3.5GHz vs 28GHz × M ∈ {32, 128} (band/split/M in filenames).
Nine curves per figure (one per model family). Same color = same family on every saved figure.

Data: recursive `result_*.json` under `--results_root`. Only fixed-SNR sweeps: ``awgn.noise_floor``
must be JSON false or numeric 0; true, null/omitted, and other values are skipped (same convention
as channel-prediction paper plots vs. ``noise_floor``).

Examples:
  python -m pilotwimae.plot.plot_beam_prediction_paper_figures --results_root results/knn_beam_batch --top_k 1 --png

  python -m pilotwimae.plot.plot_beam_prediction_paper_figures --results_root results/knn_beam_batch --top_k 5 --png --out_dir results/knn_beam_batch/paper_figures

  python -m pilotwimae.plot.plot_beam_prediction_paper_figures --results_root results/knn_beam_batch --top_k 5 --png --pdf --out_dir results/knn_beam_batch/paper_figures

  python3 pilotwimae/plot/plot_beam_prediction_paper_figures.py --results_root results/knn_beam_batch --validate_only

The last form runs the file directly and avoids importing the full `pilotwimae` package (no PyTorch).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pilotwimae.data.beam.beam_codebook import num_beams_from_saved_codebook

_BANDS = ("3.5GHz", "28GHz")
_SPLITS = ("in_dist", "ood")
_FAMILIES = (
    "fst_full",
    "fst_pilot",
    "fst_scale_full",
    "fst_scale_pilot",
    "fst_noise_full",
    "fst_noise_pilot",
    "fst_scale_noise_full",
    "fst_scale_noise_pilot",
    "jst_full",
    "jst_pilot",
    "supervised_full",
    "supervised_pilot",
)
_M_SIZES = (32, 128)
_ID_EXPECTED_CITIES = 4
_OOD_EXPECTED_CITIES = 1

_FAMILY_LABEL = {
    "fst_full": "FST",
    "fst_pilot": "FST",
    "fst_scale_full": "FST scale",
    "fst_scale_pilot": "FST scale",
    "fst_noise_full": "FST noise",
    "fst_noise_pilot": "FST noise",
    "fst_scale_noise_full": "FST noise scale",
    "fst_scale_noise_pilot": "FST noise scale",
    "jst_full": "JST",
    "jst_pilot": "JST",
    "supervised_full": "FST supervised",
    "supervised_pilot": "FST supervised",
}
_FAMILY_COLOR = {
    "fst_full": "tab:blue",
    "fst_pilot": "tab:blue",
    "fst_scale_full": "tab:brown",
    "fst_scale_pilot": "tab:brown",
    "fst_noise_full": "tab:cyan",
    "fst_noise_pilot": "tab:cyan",
    "fst_scale_noise_full": "tab:gray",
    "fst_scale_noise_pilot": "tab:gray",
    "jst_full": "tab:purple",
    "jst_pilot": "tab:purple",
    "supervised_full": "tab:red",
    "supervised_pilot": "tab:red",
}
_FAMILY_MARKER = {
    "fst_full": "o",
    "fst_pilot": "o",
    "fst_scale_full": "v",
    "fst_scale_pilot": "v",
    "fst_noise_full": "*",
    "fst_noise_pilot": "*",
    "fst_scale_noise_full": ">",
    "fst_scale_noise_pilot": ">",
    "jst_full": "D",
    "jst_pilot": "D",
    "supervised_full": "^",
    "supervised_pilot": "^",
}
_MODEL_REP_FAMILIES = (
    "fst_full",
    "fst_scale_full",
    "fst_noise_full",
    "fst_scale_noise_full",
    "jst_full",
    "supervised_full",
)
_MARKER_SIZE = 6.5

_LABEL_FONT_SIZE = 15
_TITLE_FONT_SIZE = 15
_TICK_FONT_SIZE = 13
_LEGEND_FONT_SIZE = 12
_Y_LIM = (0.1, 1.00)
_FIG_SIZE = (10, 6.5)


def _validate_topk(topk: int) -> None:
    if topk < 1 or topk > 5:
        raise ValueError(f"--top_k must be in [1, 5], got {topk}")


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


def _check_band_split_from_path(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = path.parts
    band = next((b for b in _BANDS if b in parts), None)
    split = next((s for s in _SPLITS if s in parts), None)
    # Path segments are exact names: e.g. self_supervised_scale_loss is not equal to self_supervised.
    if (
        "self_supervised_scale_loss" in parts
        or "self_supervised_noise_robust" in parts
        or "self_supervised_noise_robust_scale_loss" in parts
        or "self_supervised" in parts
    ):
        branch: Optional[str] = "self_supervised"
    elif "supervised" in parts:
        branch = "supervised"
    else:
        branch = None
    return band, split, branch


def _codebook_size(data: dict) -> int:
    return num_beams_from_saved_codebook(data["codebook"])


def _validate_schema(data: dict, path: Path, top_k: int) -> None:
    ctx = str(path)
    _require_key(data, "experiment", ctx)
    _require_key(data, "model", ctx)
    _require_key(data, "codebook", ctx)
    _require_key(data, "awgn", ctx)
    _require_key(data, "snrs_db", ctx)
    _require_key(data, "snr_robustness", ctx)
    _require_key(data["experiment"], "dataset_id", ctx)
    _require_key(data["experiment"], "checkpoint_run_name", ctx)
    _require_key(data["model"], "model_type", ctx)
    _require_key(data["awgn"], "noise_floor", ctx)
    _require_key(data["awgn"], "mean_complex_power", ctx)
    top_key = f"top_{top_k}"
    for row in data["snr_robustness"]:
        if not isinstance(row, dict):
            raise TypeError(f"snr_robustness rows must be objects in {ctx}")
        if str(row.get("snr_db", "")).lower() == "clean":
            continue
        tk = _require_key(row, top_key, ctx)
        _require_key(tk, "mean", ctx)
        _require_key(tk, "std", ctx)


def _extract_curve(data: dict, top_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    snrs = np.array([float(x) for x in data["snrs_db"]], dtype=float)
    top_key = f"top_{top_k}"
    by_snr: Dict[float, Tuple[float, float]] = {}
    for row in data["snr_robustness"]:
        if str(row["snr_db"]).lower() == "clean":
            continue
        by_snr[float(row["snr_db"])] = (
            float(row[top_key]["mean"]),
            float(row[top_key]["std"]),
        )
    vals_mean = []
    vals_std = []
    for s in snrs:
        if float(s) not in by_snr:
            raise ValueError(f"Missing snr={s} in snr_robustness")
        m, sd = by_snr[float(s)]
        vals_mean.append(m)
        vals_std.append(sd)
    return snrs, np.array(vals_mean, dtype=float), np.array(vals_std, dtype=float)


def _is_fixed_snr_result(data: dict) -> bool:
    """True only for fixed-SNR eval JSON (``noise_floor`` false or 0, never true)."""
    v = data["awgn"]["noise_floor"]
    if v is False:
        return True
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
        return True
    return False


def _is_factorized_fst_run(run_name: str) -> bool:
    rn = run_name.lower()
    return ("tkeep2" in rn) and bool(re.search(r"_2l4h_", rn))


def _is_fst_scale_loss_run(run_name: str) -> bool:
    """FST trained with scale auxiliary loss (run directory tag scaleaux)."""
    return _is_factorized_fst_run(run_name) and "scaleaux" in run_name.lower()


def _is_fst_noise_robust_run(run_name: str) -> bool:
    """FST with training-time channel noise (run directory tag noiserobust)."""
    return _is_factorized_fst_run(run_name) and "noiserobust" in run_name.lower()


def _is_fst_scale_noise_run(run_name: str) -> bool:
    """FST with both scale auxiliary and training-time noise robustness."""
    rn = run_name.lower()
    return _is_factorized_fst_run(run_name) and "scaleaux" in rn and "noiserobust" in rn


def _is_fst_baseline_run(run_name: str) -> bool:
    """FST without scale or noise-robust auxiliary variants (plain self-supervised FST)."""
    rn = run_name.lower()
    return (
        _is_factorized_fst_run(run_name)
        and "scaleaux" not in rn
        and "noiserobust" not in rn
    )


def _is_jst_target_run(run_name: str) -> bool:
    rn = run_name.lower()
    return bool(re.search(r"random_mask0\.95", rn)) and bool(re.search(r"_2l4h_", rn))


def _family_for_row(data: dict, branch: str, pilot_pattern: str) -> Optional[str]:
    model = data["model"]
    mode = str(model.get("inference_token_mode", ""))
    if not mode:
        mode = "full_grid"
    mtype = str(model.get("model_type", ""))
    if branch == "self_supervised":
        run_name = str(data["experiment"]["checkpoint_run_name"])
        if mtype == "factorized" and _is_fst_scale_noise_run(run_name):
            if mode == "full_grid":
                return "fst_scale_noise_full"
            if mode == "pilot_visible" and str(model.get("pilot_pattern", "")) == pilot_pattern:
                return "fst_scale_noise_pilot"
            return None
        if mtype == "factorized" and _is_fst_scale_loss_run(run_name):
            if mode == "full_grid":
                return "fst_scale_full"
            if mode == "pilot_visible" and str(model.get("pilot_pattern", "")) == pilot_pattern:
                return "fst_scale_pilot"
            return None
        if mtype == "factorized" and _is_fst_noise_robust_run(run_name):
            if mode == "full_grid":
                return "fst_noise_full"
            if mode == "pilot_visible" and str(model.get("pilot_pattern", "")) == pilot_pattern:
                return "fst_noise_pilot"
            return None
        if mtype == "factorized" and _is_fst_baseline_run(run_name):
            if mode == "full_grid":
                return "fst_full"
            if mode == "pilot_visible" and str(model.get("pilot_pattern", "")) == pilot_pattern:
                return "fst_pilot"
            return None
        if mtype == "standard" and _is_jst_target_run(run_name):
            if mode == "full_grid":
                return "jst_full"
            if mode == "pilot_visible" and str(model.get("pilot_pattern", "")) == pilot_pattern:
                return "jst_pilot"
            return None
        return None
    if branch == "supervised":
        if mode == "full_grid":
            return "supervised_full"
        if mode == "pilot_visible" and str(model.get("pilot_pattern", "")) == pilot_pattern:
            return "supervised_pilot"
        return None
    return None


def _aggregate_city_curves(
    rows: Iterable[Tuple[Path, dict]],
    *,
    top_k: int,
    pilot_pattern: str,
) -> Dict[Tuple[str, str, str, int], Dict[str, object]]:
    """
    Return map:
      (band, split, family, M) -> {snr, city_to_curve}
    where city_to_curve maps dataset_id -> np.ndarray(mean over SNR points).
    """
    out: Dict[Tuple[str, str, str, int], Dict[str, object]] = {}
    seen_checkpoint_names: Dict[Tuple[str, str, str, int], set[str]] = {}

    for path, data in rows:
        band, split, branch = _check_band_split_from_path(path)
        if band is None or split is None or branch is None:
            continue
        awgn = data.get("awgn", {})
        if not isinstance(awgn, dict):
            continue
        if "noise_floor" not in awgn:
            # Older schemas without this flag cannot be reliably classified as fixed_snr.
            continue
        if not _is_fixed_snr_result(data):
            continue
        _validate_schema(data, path, top_k)

        family = _family_for_row(data, branch, pilot_pattern)
        if family is None:
            continue

        m = _codebook_size(data)
        if m not in _M_SIZES:
            continue

        city = str(data["experiment"]["dataset_id"])
        run_name = str(data["experiment"]["checkpoint_run_name"])
        group_key = (band, split, family, m)
        seen_checkpoint_names.setdefault(group_key, set()).add(run_name)

        snr, mean, std = _extract_curve(data, top_k)
        key = (band, split, family, m)
        if key not in out:
            out[key] = {"snr": snr, "city_to_mean": {}, "city_to_std": {}}
        else:
            ref = out[key]["snr"]  # type: ignore[index]
            if list(ref) != list(snr):
                raise ValueError(f"Inconsistent snrs for {key}: {path}")

        city_to_mean: Dict[str, np.ndarray] = out[key]["city_to_mean"]  # type: ignore[index]
        city_to_std: Dict[str, np.ndarray] = out[key]["city_to_std"]  # type: ignore[index]
        if city in city_to_mean or city in city_to_std:
            raise ValueError(
                f"Duplicate city curve for {key} city={city}: {path}. "
                "Please deduplicate results_root or tighten run filters."
            )
        city_to_mean[city] = mean
        city_to_std[city] = std

    # Guard against accidental mixing of multiple checkpoint variants per family/band/split.
    for k, names in seen_checkpoint_names.items():
        if len(names) > 1:
            names_disp = ", ".join(sorted(names))
            raise ValueError(
                f"Multiple checkpoint_run_name values matched {k}: {names_disp}. "
                "Expected one target model variant."
            )

    return out


def _panel_curve(
    agg: Dict[Tuple[str, str, str, int], Dict[str, object]],
    *,
    band: str,
    split: str,
    family: str,
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (band, split, family, m)
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

    # Shading must reflect fold variability from each run JSON (top_k.std), not city spread.
    # For ID, aggregate by averaging city means and averaging city stds.
    # For OOD (single city), this naturally returns that city's mean/std.
    cities = sorted(city_to_mean.keys())
    mean_stack = np.stack([city_to_mean[c] for c in cities], axis=0)
    std_stack = np.stack([city_to_std[c] for c in cities], axis=0)
    mean = mean_stack.mean(axis=0)
    std = std_stack.mean(axis=0)
    return snr, mean, std


def _shared_ylim_across_panels(agg: Dict[Tuple[str, str, str, int], Dict[str, object]]) -> Tuple[float, float]:
    _ = agg
    return _Y_LIM


def _plot_panel(
    agg: Dict[Tuple[str, str, str, int], Dict[str, object]],
    *,
    band: str,
    split: str,
    m: int,
    top_k: int,
    y_lim: Tuple[float, float],
    out_dir: Path,
    write_png: bool,
    write_pdf: bool,
    no_title: bool,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    wrote: List[Path] = []

    ref_snr: Optional[np.ndarray] = None
    for family in _FAMILIES:
        snr, mean, std = _panel_curve(agg, band=band, split=split, family=family, m=m)
        if ref_snr is None:
            ref_snr = snr
        elif list(ref_snr) != list(snr):
            raise ValueError(f"Panel {(band, split, m)} has inconsistent snr axes")
        color = _FAMILY_COLOR[family]
        marker = _FAMILY_MARKER[family]
        ax.plot(
            snr,
            mean,
            color=color,
            linestyle="--" if family.endswith("_pilot") else "-",
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
    ax.set_ylabel(f"Top-{top_k} Accuracy", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
    if not no_title:
        ax.set_title(
            f"{split_name} @ {band} (M = {m}): FST supervised vs FST/JST (full/pilot, incl. FST+scale)",
            fontsize=_TITLE_FONT_SIZE,
            fontweight="bold",
        )
    ax.grid(True, linestyle="--", alpha=0.8, linewidth=1)
    ax.tick_params(axis="both", labelsize=_TICK_FONT_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    family_handles = [
        Line2D(
            [0],
            [0],
            color=_FAMILY_COLOR[f],
            marker=_FAMILY_MARKER[f],
            linestyle="-",
            linewidth=2.2,
            markersize=_MARKER_SIZE,
            label=_FAMILY_LABEL[f],
        )
        for f in _MODEL_REP_FAMILIES
    ]
    mode_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="Full channel"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="Pilot-only"),
    ]
    leg = ax.legend(
        handles=family_handles + mode_handles,
        loc="lower right",
        fontsize=_LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=2.6,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    fig.tight_layout()

    band_slug = _slug(band)
    base = (
        f"top_{top_k}_vs_snr__{'id' if split == 'in_dist' else 'ood'}__{band_slug}"
        f"__M{m}__supervised_vs_fst_jst_full_pilot"
    )
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
        description="Plot supervised vs FST/JST (full/pilot) Top-K vs SNR for 3.5/28 GHz and ID/OOD."
    )
    p.add_argument(
        "--results_root",
        type=str,
        default="results/knn_beam_batch",
        help="Root directory recursively scanned for result_*.json",
    )
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: RESULTS_ROOT/plots)")
    p.add_argument("--top_k", type=int, default=1, help="Top-K metric to plot (1..5)")
    p.add_argument(
        "--pilot_pattern",
        type=str,
        default="t:2,11;f:0,2,4,6",
        help="Pilot pattern used to identify pilot-only FST/JST runs.",
    )
    p.add_argument("--png", action="store_true", help="Write PNG output.")
    p.add_argument("--pdf", action="store_true", help="Write PDF output.")
    p.add_argument("--no_title", action="store_true", help="Disable plot titles.")
    p.add_argument(
        "--validate_only",
        action="store_true",
        help="Run coverage/consistency checks only; do not write plots.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    _validate_topk(int(args.top_k))
    if not args.validate_only and not args.png and not args.pdf:
        raise SystemExit("Select at least one output format via --png and/or --pdf (or use --validate_only).")

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.is_dir():
        raise NotADirectoryError(f"--results_root is not a directory: {results_root}")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else (results_root / "plots").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[Path, dict]] = []
    for p in _json_files(results_root):
        with p.open("r") as f:
            d = json.load(f)
        rows.append((p, d))

    agg = _aggregate_city_curves(rows, top_k=int(args.top_k), pilot_pattern=str(args.pilot_pattern))

# Validate all required 8 figures × 12 curves (one per family at each M).
    for band in _BANDS:
        for split in _SPLITS:
            for family in _FAMILIES:
                for m in _M_SIZES:
                    _ = _panel_curve(agg, band=band, split=split, family=family, m=m)

    if args.validate_only:
        print("Validation OK: all required panel curves are present and consistent.")
        for m in _M_SIZES:
            for band in _BANDS:
                for split in _SPLITS:
                    print(f"  figure: {split} @ {band} M={m} -> 12 curves")
        return

    y_lim = _shared_ylim_across_panels(agg)
    written: List[Path] = []
    for m in _M_SIZES:
        for band in _BANDS:
            for split in _SPLITS:
                paths = _plot_panel(
                    agg,
                    band=band,
                    split=split,
                    m=m,
                    top_k=int(args.top_k),
                    y_lim=y_lim,
                    out_dir=out_dir,
                    write_png=bool(args.png),
                    write_pdf=bool(args.pdf),
                    no_title=bool(args.no_title),
                )
                written.extend(paths)
                print(f"Wrote panel: {split} @ {band} (M={m})")

    print(f"Wrote {len(written)} plot file(s) to: {out_dir}")
    for p in written:
        print(str(p))


if __name__ == "__main__":
    main()
