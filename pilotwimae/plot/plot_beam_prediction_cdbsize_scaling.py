"""
Top-K beam accuracy vs SNR for codebook-size scaling experiments (28 GHz).

Reads flat ``result_*.json`` files (e.g. under ``results/knn_bp_cdb_size_scaling/fst_noise_scale/``)
produced by ``scripts/beam_prediction/batch_evaluate_single_ckpt.sh``: each file is one
``(city × codebook quadruple u_h:u_v:o_h:o_v × observation mode)`` run.

Produces one figure (axes/legend styling aligned with ``plot_beam_prediction_paper_figures.py``;
no plot title — OOD is indicated by the output filename):
  - **OOD**: ``la_1`` only (``ood_test_28``); linear Top-K accuracy in ``[0, 1]`` with a ±1-std band
    (linear), clipped to ``[0, 1]``.
  - **Full channel** (``model.inference_token_mode == full_grid``): **solid** line.
  - **Pilot only** (``pilot_visible``): **dashed** line.

Per codebook configuration, both modes are required in the results directory.

Legend: one entry per codebook (mathtext as below). Linestyle (solid vs dashed) is explained once
via a second mini-legend with **black** proxy lines (full channel / pilot only).

Legend format (``M`` = codebook size; mathtext for angular params):
  - ``M > 32``: ``cdb size M, $o_h$ = …, $o_v$ = …``
  - ``M < 32``: ``cdb size M, $u_h$ = …, $u_v$ = …``
  - ``M == 32``: ``cdb size M``

Examples:
  python -m pilotwimae.plot.plot_beam_prediction_cdbsize_scaling --results_dir results/knn_bp_cdb_size_scaling/fst_noise_scale --top_k 1 --png

  python pilotwimae/plot/plot_beam_prediction_cdbsize_scaling.py --results_dir results/knn_bp_cdb_size_scaling/fst_noise_scale --top_k 5 --png --pdf

  python -m pilotwimae.plot.plot_beam_prediction_cdbsize_scaling --results_dir results/knn_bp_cdb_size_scaling/fst_noise_scale --top_k 1 --png --omit_uhuv '(4,4),(4,2)'
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pilotwimae.data.beam.beam_codebook import num_beams_from_saved_codebook

_OOD_CITY = "la_1"
_MODES_PLOT = ("full_grid", "pilot_visible")  # plot order: full channel first, then pilot
_MODE_LINESTYLE = {"full_grid": "-", "pilot_visible": "--"}

_LABEL_FONT_SIZE = 15
_TICK_FONT_SIZE = 13
_LEGEND_FONT_SIZE = 12
_FIG_SIZE = (10, 6.5)
_MARKER_SIZE = 6.5
_Y_LIM = (0.1, 1.03)  # small headroom above 1.0 so markers / shading do not touch the frame
_LEGEND_HANDLELENGTH = 3.2  # longer handles show linestyle differences (legend)

# Matplotlib tab10 — repeat hue cycle after 10 curves; linestyle is fixed by pilot vs full channel.
_TAB10_HEX = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "8")


def _curve_style(idx: int) -> Tuple[str, str]:
    """(color, marker) — hue and marker shape vary by codebook index."""
    i = int(idx)
    color = _TAB10_HEX[i % len(_TAB10_HEX)]
    marker = _MARKERS[i % len(_MARKERS)]
    return color, marker


def _slug(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _json_files(results_dir: Path) -> List[Path]:
    files = sorted(p for p in results_dir.glob("result_*.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No result_*.json files found in: {results_dir}")
    return files


def _is_28ghz_run(data: dict) -> bool:
    dd = str(data.get("experiment", {}).get("data_dir", ""))
    return "test_28" in dd or "ood_test_28" in dd


def _is_fixed_snr_result(data: dict) -> bool:
    awgn = data.get("awgn")
    if not isinstance(awgn, dict) or "noise_floor" not in awgn:
        return False
    v = awgn["noise_floor"]
    if v is False:
        return True
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
        return True
    return False


def _dataset_city(data: dict) -> str:
    return str(data["experiment"]["dataset_id"])


def _is_ood_28_la(data: dict) -> bool:
    if _dataset_city(data) != _OOD_CITY:
        return False
    dd = str(data.get("experiment", {}).get("data_dir", ""))
    return "ood_test_28" in dd


def _inference_token_mode(data: dict) -> str:
    m = data.get("model", {}).get("inference_token_mode")
    return str(m) if m is not None else ""


def _codebook_quadruple(cb: dict) -> Tuple[int, int, int, int]:
    return (
        int(cb.get("u_h", 1)),
        int(cb.get("u_v", 1)),
        int(cb.get("o_h", 1)),
        int(cb.get("o_v", 1)),
    )


def _legend_label(M: int, cb: dict) -> str:
    """
    Legend strings with mathtext. Angular symbols use ``\\mathbf`` inside math mode —
    ``Legend`` bold weight does not propagate into ``$...$`` mathtext substrings.
    """
    u_h, u_v, o_h, o_v = _codebook_quadruple(cb)
    if M == 32:
        return f"cdb size {M}"
    if M > 32:
        return (
            rf"cdb size {M}, $\mathbf{{o}}_{{\mathrm{{h}}}}$ = {o_h}, "
            rf"$\mathbf{{o}}_{{\mathrm{{v}}}}$ = {o_v}"
        )
    return (
        rf"cdb size {M}, $\mathbf{{u}}_{{\mathrm{{h}}}}$ = {u_h}, "
        rf"$\mathbf{{u}}_{{\mathrm{{v}}}}$ = {u_v}"
    )


def _linestyle_legend_handles() -> List[Line2D]:
    """Black proxy lines: solid = full channel, dashed = pilot (shown once, separate mini-legend)."""
    return [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2.2,
            label="Full channel",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="Pilot only",
        ),
    ]


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
    vals_mean: List[float] = []
    vals_std: List[float] = []
    for s in snrs:
        if float(s) not in by_snr:
            raise ValueError(f"Missing snr={s} in snr_robustness")
        m, sd = by_snr[float(s)]
        vals_mean.append(m)
        vals_std.append(sd)
    return snrs, np.array(vals_mean, dtype=float), np.array(vals_std, dtype=float)


def _parse_omit_uhuv(s: Optional[str]) -> Set[Tuple[int, int]]:
    """
    Parse comma-separated (u_h, u_v) pairs, e.g. ``(4,4),(4,2)``.
    Whitespace around numbers and commas is allowed.
    """
    if s is None:
        return set()
    t = str(s).strip()
    if not t:
        return set()
    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", t)
    if not pairs:
        raise ValueError(
            f"Could not parse any (u_h, u_v) pairs from --omit_uhuv={s!r}; "
            "expected e.g. '(4,4),(4,2)'."
        )
    return {(int(a), int(b)) for a, b in pairs}


def _sort_config_keys(
    keys: Iterable[Tuple[int, int, int, int]],
    codebook_by_key: Dict[Tuple[int, int, int, int], dict],
) -> List[Tuple[int, int, int, int]]:
    def key_fn(k: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        M = num_beams_from_saved_codebook(codebook_by_key[k])
        return (M, k[0], k[1], k[2], k[3])

    return sorted(keys, key=key_fn)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot Top-K vs SNR for codebook scaling (28 GHz OOD LA): full channel (solid) "
            "and pilot only (dashed), linear accuracy in [0, 1]."
        )
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="results/knn_bp_cdb_size_scaling/fst_noise_scale",
        help="Directory containing result_*.json (non-recursive glob).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: RESULTS_DIR/plots_cdbsize_scaling).",
    )
    p.add_argument("--top_k", type=int, default=1, help="Top-K metric (1..5).")
    p.add_argument("--png", action="store_true", help="Write PNG.")
    p.add_argument("--pdf", action="store_true", help="Write PDF.")
    p.add_argument(
        "--pilot_pattern",
        type=str,
        default=None,
        help=(
            "If set, keep only pilot_visible JSONs whose model.pilot_pattern equals this string. "
            "full_grid runs are unaffected by this filter."
        ),
    )
    p.add_argument(
        "--omit_uhuv",
        type=str,
        default=None,
        help=(
            "Exclude codebooks whose undersampling (u_h, u_v) matches any listed pair. "
            "Comma-separated parenthesized pairs, e.g. '(4,4),(4,2)'. "
            "Quote the argument in the shell so commas/parentheses are preserved."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    top_k = int(args.top_k)
    if top_k < 1 or top_k > 5:
        raise SystemExit("--top_k must be in [1, 5]")
    if not args.png and not args.pdf:
        raise SystemExit("Select at least one of --png / --pdf")

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"--results_dir is not a directory: {results_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else (results_dir / "plots_cdbsize_scaling").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_filter = args.pilot_pattern
    omit_uhuv = _parse_omit_uhuv(args.omit_uhuv)

    # (u_h,u_v,o_h,o_v) -> { inference_token_mode -> (path, data) }
    bucket: Dict[Tuple[int, int, int, int], Dict[str, Tuple[Path, dict]]] = {}

    for path in _json_files(results_dir):
        with path.open("r") as f:
            data = json.load(f)
        if not _is_28ghz_run(data):
            continue
        if not _is_fixed_snr_result(data):
            continue
        if not _is_ood_28_la(data):
            continue
        mode = _inference_token_mode(data)
        if mode not in _MODES_PLOT:
            continue
        if pilot_filter is not None and mode == "pilot_visible":
            pp = data.get("model", {}).get("pilot_pattern")
            if str(pp) != pilot_filter:
                continue
        cb = data.get("codebook")
        if not isinstance(cb, dict):
            continue
        gk = _codebook_quadruple(cb)
        if omit_uhuv and (gk[0], gk[1]) in omit_uhuv:
            continue
        bucket.setdefault(gk, {})
        if mode in bucket[gk]:
            raise ValueError(f"Duplicate mode {mode} for codebook {gk}: {path} vs {bucket[gk][mode][0]}")
        bucket[gk][mode] = (path, data)

    if not bucket:
        raise SystemExit("No valid JSON rows after filters.")

    meta: Dict[Tuple[int, int, int, int], dict] = {}
    ood_curves_lin: Dict[
        Tuple[int, int, int, int], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {}

    for gk, by_mode in bucket.items():
        need = set(_MODES_PLOT)
        have = set(by_mode.keys())
        if need - have:
            raise ValueError(
                f"Codebook {gk}: need both {sorted(need)} for OOD LA; missing {sorted(need - have)}. "
                f"Have modes: {sorted(have)}"
            )

        any_data = next(iter(by_mode.values()))[1]
        meta[gk] = any_data["codebook"]

        curves_m: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        snr_ref: Optional[np.ndarray] = None
        for mode in _MODES_PLOT:
            _, d = by_mode[mode]
            snr, m_o, s_o = _extract_curve(d, top_k)
            if snr_ref is None:
                snr_ref = snr
            elif list(snr_ref) != list(snr):
                raise ValueError(f"SNR axis mismatch for codebook {gk}, mode={mode}")
            lo_o = np.clip(m_o - s_o, 0.0, 1.0)
            hi_o = np.clip(m_o + s_o, 0.0, 1.0)
            curves_m[mode] = (snr, m_o, lo_o, hi_o)
        assert snr_ref is not None
        ood_curves_lin[gk] = curves_m

    sorted_keys = _sort_config_keys(bucket.keys(), meta)

    def _plot_ood(
        curves: Dict[
            Tuple[int, int, int, int],
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        ],
        fname_suffix: str,
    ) -> List[Path]:
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        written: List[Path] = []
        for idx, gk in enumerate(sorted_keys):
            color, marker = _curve_style(idx)
            cb = meta[gk]
            M = num_beams_from_saved_codebook(cb)
            by_mode_curves = curves[gk]
            for mode in _MODES_PLOT:
                snr, mean_lin, lo_lin, hi_lin = by_mode_curves[mode]
                linestyle = _MODE_LINESTYLE[mode]
                # One codebook entry in the main legend (full channel curve); pilot hidden.
                label = _legend_label(M, cb) if mode == "full_grid" else "_nolegend_"
                ax.plot(
                    snr,
                    mean_lin,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.2,
                    marker=marker,
                    markersize=_MARKER_SIZE,
                    markeredgecolor=color,
                    markeredgewidth=0.6,
                    markerfacecolor=color,
                    label=label,
                )
                ax.fill_between(snr, lo_lin, hi_lin, color=color, alpha=0.14)

        ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
        ax.set_xlabel("SNR (dB)", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
        ax.set_ylabel(f"Top-{top_k} Accuracy", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.8, linewidth=1)
        ax.tick_params(axis="both", labelsize=_TICK_FONT_SIZE)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        h_cdb, lab_cdb = ax.get_legend_handles_labels()
        leg_cdb = ax.legend(
            h_cdb,
            lab_cdb,
            loc="lower right",
            fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            ncol=2,
            columnspacing=1.15,
            handlelength=_LEGEND_HANDLELENGTH,
        )
        leg_cdb.set_zorder(102)
        for txt in leg_cdb.get_texts():
            txt.set_fontweight("bold")
        ax.add_artist(leg_cdb)

        h_style = _linestyle_legend_handles()
        leg_style = ax.legend(
            handles=h_style,
            loc="lower left",
            fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            ncol=1,
            handlelength=_LEGEND_HANDLELENGTH,
        )
        leg_style.set_zorder(103)
        for txt in leg_style.get_texts():
            txt.set_fontweight("bold")
        fig.tight_layout()

        base = f"top_{top_k}_vs_snr__28GHz__{_slug(fname_suffix)}__cdb_scaling"
        if args.png:
            p = out_dir / f"{base}.png"
            fig.savefig(p, dpi=220)
            written.append(p)
        if args.pdf:
            p = out_dir / f"{base}.pdf"
            fig.savefig(p)
            written.append(p)
        plt.close(fig)
        return written

    all_paths = _plot_ood(ood_curves_lin, fname_suffix="ood_la")

    print(f"Wrote {len(all_paths)} plot file(s) to: {out_dir}")
    for p in all_paths:
        print(str(p))


if __name__ == "__main__":
    main()
