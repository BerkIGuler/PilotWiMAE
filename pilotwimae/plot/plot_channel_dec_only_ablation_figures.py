from __future__ import annotations

"""
NMSE (dB) vs SNR (dB) for dec-only ablations: one curve per ``dec_params``.

Reads ``result_*.json`` under a single results folder (e.g.
``results/channel_prediction/dec_only/t211f0246``). Parses:

- ``dec_params`` from ``checkpoint`` parent dir:
  ``.../pilotwimae_dec_only_fst_<dec_params>_from_tk2_sm09_.../best_checkpoint.pt``
  (boundary token ``_from_tk2_sm09`` so ``normpatch_mse_from_tk2`` does not truncate early).
  Legend entries are ``num. dec. layers = N`` (parsed from ``_decN`` in the run slug).
  ``--dec_params`` may still use the shortened alias form (see ``_dec_params_legend_label``).
- City from ``data_dir`` basename (``boston_1``, ``la_1``, ...).

Optional ``--dec_params`` (comma-separated, e.g. ``tk2_sm05_dec4,tk4_sm075_dec6``)
selects which ``dec_params`` curves to draw; default is all runs in the folder.

Use ``--split ood`` (default), ``id``, or ``both`` to control which figure(s) are written.

ID (3.5 GHz ``test_35``): average linear NMSE over boston, sf, nyc, chicago;
plot ``10*log10(mean_lin)``. With ``--shade``: band from cross-city std in linear
domain mapped to dB (same delta rule as ``plot_channel_prediction_paper_figures``).

OOD (``ood_test_35``): LA only — ``nmse_by_snr_linear`` mean per SNR; with ``--shade``,
fold std in linear domain → dB band.

Plot style (fonts, grid, linewidth, markers, fig size) matches
``plot_channel_prediction_paper_figures.py``.

Example:
  python3 pilotwimae/plot/plot_channel_dec_only_ablation_figures --results_dir results/channel_prediction/dec_only/t211f0246 --png

  python3 pilotwimae/plot/plot_channel_dec_only_ablation_figures --shade --png

  python3 pilotwimae/plot/plot_channel_dec_only_ablation_figures --snrs_db 0,5,10,15,20,25,30 --png

  python3 pilotwimae/plot/plot_channel_dec_only_ablation_figures --dec_params tk2_sm05_dec4,tk4_sm075_dec4 --png

  python3 pilotwimae/plot/plot_channel_dec_only_ablation_figures --split both --png

  python3 pilotwimae/plot/plot_channel_dec_only_ablation_figures --png --no-title
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_DEFAULT_SNRS_DB_STR = "0,5,10,15,20,25"

_PREFIX = "pilotwimae_dec_only_fst_"
# Run dir tail is ``..._from_tk2_sm09_fst_scaleaux...``; use full token so ``mse_from_tk2``
# inside ``normpatch_mse_from_tk2`` is not mistaken for the decoder-slug boundary.
_DEC_PARAMS_BOUNDARY = "_from_tk2_sm09"
# Suffix only in some run names; stripped for legend (grouping still uses full parsed slug).
_LEGEND_STRIP_SUFFIXES = ("_normpatch_mse",)
_ID_CITIES_ORDER = ("boston", "sf", "nyc", "chicago")

_MARKER_SIZE = 6.5
_LABEL_FONT_SIZE = 15
_TITLE_FONT_SIZE = 15
_TICK_FONT_SIZE = 13
_LEGEND_FONT_SIZE = 15
_FIG_SIZE = (10, 6.5)

# Match paper figures panel y-range.
_Y_LIM = (-31.0, 5.0)

_CURVE_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)
_CURVE_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*", "<", ">")


def _slug(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _snr_dict_key(s: float) -> str:
    return str(int(s)) if float(s).is_integer() else str(s)


def _parse_snrs_db_arg(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --snrs_db")
    return np.array([float(p) for p in parts], dtype=float)


def _snr_file_tag(snrs: np.ndarray) -> str:
    bits = []
    for x in snrs:
        xf = float(x)
        bits.append(str(int(xf)) if xf.is_integer() else str(xf))
    return "__snr_" + "_".join(bits)


def _parse_dec_params_list(s: str) -> Optional[List[str]]:
    """Comma/semicolon-separated ``dec_params`` slugs; empty string -> plot all."""
    t = str(s).strip()
    if not t:
        return None
    parts = [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]
    return parts if parts else None


def _dec_file_tag(dec_keys: List[str]) -> str:
    """Filename fragment when plotting a ``dec_params`` subset (order preserved)."""
    return "__dec_" + "+".join(_slug(k) for k in dec_keys)


def _resolve_dec_key(available_keys: List[str], token: str) -> str:
    """Map user ``--dec_params`` token to a single available key (raw slug or legend alias)."""
    if token in available_keys:
        return token
    legend_hits = [k for k in available_keys if _dec_params_legend_label(k) == token]
    if len(legend_hits) == 1:
        return legend_hits[0]
    if len(legend_hits) > 1:
        raise ValueError(
            f"--dec_params {token!r} is ambiguous (multiple runs legend-match): {legend_hits!r}"
        )
    raise KeyError(token)


def _resolve_dec_keys(
    available_keys: List[str], want: Optional[List[str]]
) -> List[str]:
    available = sorted(available_keys)
    if want is None:
        return available
    out: List[str] = []
    seen: set[str] = set()
    for w in want:
        try:
            k = _resolve_dec_key(available, w)
        except KeyError:
            raise ValueError(
                f"--dec_params not found in results: {w!r}. "
                f"Available (raw): {available}. "
                f"You may also pass the shortened legend form (e.g. tk4_sm05_dec1)."
            ) from None
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _subset_curve_by_snrs(
    snr: np.ndarray,
    mean_db: np.ndarray,
    want_snrs: np.ndarray,
    std_db: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Keep only SNR points listed in ``want_snrs`` (order preserved)."""
    idx: List[int] = []
    for w in want_snrs:
        matches = np.nonzero(np.isclose(snr, float(w), rtol=0.0, atol=1e-6))[0]
        if matches.size != 1:
            raise ValueError(
                f"SNR {w!r} must match exactly one entry in data snrs_db; "
                f"got {int(matches.size)} match(es); grid={list(snr)}"
            )
        idx.append(int(matches[0]))
    take = np.asarray(idx, dtype=int)
    out_std = None if std_db is None else std_db[take]
    return snr[take], mean_db[take], out_std


def _linear_to_db(x_lin: np.ndarray) -> np.ndarray:
    x_safe = np.maximum(x_lin, np.finfo(float).tiny)
    return 10.0 * np.log10(x_safe)


def _std_db_from_linear(mean_lin: float, std_lin: float) -> float:
    """First-order delta std for Y = 10*log10(X)."""
    if mean_lin <= 0.0 or std_lin < 0.0:
        return 0.0
    return float((10.0 / np.log(10.0)) * (float(std_lin) / float(mean_lin)))


def _parse_dec_params(checkpoint: str) -> str:
    parent = Path(checkpoint).parent.name
    if _PREFIX not in parent:
        raise ValueError(
            f"Cannot parse dec_params from checkpoint parent name: {parent!r} "
            f"(expected {_PREFIX!r}...)"
        )
    i = parent.index(_PREFIX) + len(_PREFIX)
    if _DEC_PARAMS_BOUNDARY in parent:
        j = parent.index(_DEC_PARAMS_BOUNDARY, i)
    elif "_from_tk2" in parent:
        j = parent.index("_from_tk2", i)
    else:
        raise ValueError(
            f"Cannot find decoder slug boundary in checkpoint parent name: {parent!r}"
        )
    return parent[i:j]


def _dec_params_legend_label(dec_slug: str) -> str:
    """Short slug for ``--dec_params`` alias matching (e.g. strip ``_normpatch_mse``)."""
    out = dec_slug
    for suf in _LEGEND_STRIP_SUFFIXES:
        if out.endswith(suf):
            out = out[: -len(suf)]
    return out


_DEC_LAYER_COUNT_RE = re.compile(r"_dec(\d+)")


def _legend_num_dec_layers_line(dec_slug: str) -> str:
    """Plot legend text: ``num. dec. layers = N`` from ``..._decN...`` in the run slug."""
    m = _DEC_LAYER_COUNT_RE.search(dec_slug)
    if not m:
        raise ValueError(
            f"Cannot infer decoder layer count from dec_params slug {dec_slug!r} "
            "(expected a substring like '_dec4')."
        )
    return f"num. dec. layers = {int(m.group(1), 10)}"


def _city_from_data_dir(data_dir: str) -> Tuple[str, str]:
    """Returns (split, city) with split in {'id','ood'}."""
    d = data_dir.replace("\\", "/")
    base = Path(d).name.lower()
    if "ood_test_35" in d:
        split = "ood"
    elif "test_35" in d:
        split = "id"
    else:
        raise ValueError(f"Unrecognized data_dir (expected test_35 or ood_test_35): {data_dir}")

    for city in ("boston", "sf", "nyc", "chicago", "la"):
        if base.startswith(city + "_") or base == city:
            return split, city
    raise ValueError(f"Cannot infer city from data_dir basename: {base!r}")


def _pilot_pattern(data: dict) -> str:
    if "pilot_pattern" in data:
        return str(data["pilot_pattern"])
    meta = data.get("pilot_meta")
    if isinstance(meta, dict) and "pilot_pattern" in meta:
        return str(meta["pilot_pattern"])
    return ""


def _json_files(results_dir: Path) -> List[Path]:
    files = sorted(p for p in results_dir.glob("result_*.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No result_*.json under {results_dir}")
    return files


def _linear_nmse_at_snrs(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """SNR grid and linear NMSE means from ``nmse_by_snr_linear``."""
    snrs = np.array([float(x) for x in data["snrs_db"]], dtype=float)
    lookup_lin = data.get("nmse_by_snr_linear")
    if not isinstance(lookup_lin, dict):
        raise KeyError("Missing or invalid 'nmse_by_snr_linear'")
    means: List[float] = []
    for s in snrs:
        k = _snr_dict_key(float(s))
        block = lookup_lin.get(k)
        if not isinstance(block, dict):
            raise KeyError(f"Missing nmse_by_snr_linear[{k!r}]")
        means.append(float(block["mean"]))
    return snrs, np.asarray(means, dtype=float)


def _linear_nmse_lin_mean_std(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SNR grid and linear NMSE mean/std per SNR from ``nmse_by_snr_linear``."""
    snrs = np.array([float(x) for x in data["snrs_db"]], dtype=float)
    lookup_lin = data.get("nmse_by_snr_linear")
    if not isinstance(lookup_lin, dict):
        raise KeyError("Missing or invalid 'nmse_by_snr_linear'")
    means: List[float] = []
    stds: List[float] = []
    for s in snrs:
        k = _snr_dict_key(float(s))
        block = lookup_lin.get(k)
        if not isinstance(block, dict):
            raise KeyError(f"Missing nmse_by_snr_linear[{k!r}]")
        means.append(float(block["mean"]))
        stds.append(float(block["std"]))
    return snrs, np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def _aggregate_id_across_cities(
    city_to_data: Dict[str, dict],
    *,
    shade: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Per-SNR: mean linear NMSE over cities, then ``10*log10``; optional cross-city std band in dB."""
    ref_snrs: Optional[np.ndarray] = None
    mean_lins: List[np.ndarray] = []
    for city in _ID_CITIES_ORDER:
        if city not in city_to_data:
            raise KeyError(f"Missing ID city {city!r}; have {sorted(city_to_data)}")
        snr, m_lin = _linear_nmse_at_snrs(city_to_data[city])
        if ref_snrs is None:
            ref_snrs = snr
        elif not np.allclose(ref_snrs, snr):
            raise ValueError(f"Inconsistent snrs_db for city {city}")
        mean_lins.append(m_lin)
    assert ref_snrs is not None
    stack = np.stack(mean_lins, axis=0)
    mean_lin = stack.mean(axis=0)
    mean_db = _linear_to_db(mean_lin)
    if not shade:
        return ref_snrs, mean_db, None
    std_lin = stack.std(axis=0, ddof=1)
    std_db = np.array(
        [_std_db_from_linear(float(ml), float(sl)) for ml, sl in zip(mean_lin, std_lin)]
    )
    return ref_snrs, mean_db, std_db


def _aggregate_ood_single(data: dict, *, shade: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if shade:
        snr, mean_lin, std_lin = _linear_nmse_lin_mean_std(data)
        mean_db = _linear_to_db(mean_lin)
        std_db = np.array(
            [_std_db_from_linear(float(ml), float(sl)) for ml, sl in zip(mean_lin, std_lin)]
        )
        return snr, mean_db, std_db
    snr, mean_lin = _linear_nmse_at_snrs(data)
    mean_db = _linear_to_db(mean_lin)
    return snr, mean_db, None


def _load_grouped(
    results_dir: Path, pilot_pattern: str
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, dict], np.ndarray]:
    id_by_dec: Dict[str, Dict[str, dict]] = {}
    ood_by_dec: Dict[str, dict] = {}
    ref_snr: Optional[np.ndarray] = None

    for path in _json_files(results_dir):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        pp = _pilot_pattern(data)
        if pilot_pattern and pp != pilot_pattern:
            continue
        ck = str(data.get("checkpoint", ""))
        dd = str(data.get("data_dir", ""))
        dec = _parse_dec_params(ck)
        split, city = _city_from_data_dir(dd)
        snr, _ = _linear_nmse_at_snrs(data)
        if ref_snr is None:
            ref_snr = snr
        elif not np.allclose(ref_snr, snr):
            raise ValueError(f"Inconsistent snrs_db vs other files: {path}")

        if split == "id":
            if city not in _ID_CITIES_ORDER:
                raise ValueError(f"Unexpected ID city {city!r} in {path}")
            id_by_dec.setdefault(dec, {})
            if city in id_by_dec[dec]:
                raise ValueError(f"Duplicate (dec, city)=({dec}, {city}): {path}")
            id_by_dec[dec][city] = data
        else:
            if city != "la":
                raise ValueError(f"Expected OOD city la, got {city!r} in {path}")
            if dec in ood_by_dec:
                raise ValueError(f"Duplicate OOD row for dec={dec}: {path}")
            ood_by_dec[dec] = data

    if ref_snr is None:
        raise RuntimeError("No JSON rows after filtering.")

    for dec, cmap in id_by_dec.items():
        if len(cmap) != len(_ID_CITIES_ORDER):
            raise ValueError(
                f"ID dec={dec!r}: expected {len(_ID_CITIES_ORDER)} cities, "
                f"got {len(cmap)} {sorted(cmap)}"
            )

    return id_by_dec, ood_by_dec, ref_snr


def _plot_split(
    curves: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    *,
    title: str,
    out_path_stem: Path,
    write_png: bool,
    write_pdf: bool,
    no_title: bool,
    shade: bool,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    wrote: List[Path] = []

    for i, (_, snr, mean_db, std_db) in enumerate(curves):
        color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
        marker = _CURVE_MARKERS[i % len(_CURVE_MARKERS)]
        ax.plot(
            snr,
            mean_db,
            color=color,
            linestyle="-",
            linewidth=2.2,
            marker=marker,
            markersize=_MARKER_SIZE,
        )
        if shade and std_db is not None:
            lo = mean_db - std_db
            hi = mean_db + std_db
            ax.fill_between(snr, lo, hi, color=color, alpha=0.14)

    ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
    ax.set_xlabel("SNR (dB)", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel("NMSE (dB)", fontsize=_LABEL_FONT_SIZE, fontweight="bold")
    if not no_title:
        ax.set_title(title, fontsize=_TITLE_FONT_SIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.8, linewidth=1)
    ax.tick_params(axis="both", labelsize=_TICK_FONT_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=_CURVE_COLORS[i % len(_CURVE_COLORS)],
            marker=_CURVE_MARKERS[i % len(_CURVE_MARKERS)],
            linestyle="-",
            linewidth=2.2,
            markersize=_MARKER_SIZE,
            label=lab,
        )
        for i, (lab, _, _, _) in enumerate(curves)
    ]
    leg = ax.legend(
        handles=handles,
        loc="best",
        fontsize=_LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=2.6,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    fig.tight_layout()

    if write_png:
        p = Path(str(out_path_stem) + ".png")
        fig.savefig(p, dpi=220)
        wrote.append(p)
    if write_pdf:
        p = Path(str(out_path_stem) + ".pdf")
        fig.savefig(p)
        wrote.append(p)
    plt.close(fig)
    return wrote


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot dec-only ablation NMSE vs SNR (ID vs OOD).")
    p.add_argument(
        "--results_dir",
        type=str,
        default="results/channel_prediction/dec_only/t211f0246",
        help="Directory containing result_*.json for one ablation grid.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: RESULTS_DIR/figures).",
    )
    p.add_argument(
        "--pilot_pattern",
        type=str,
        default="t:2,11;f:0,2,4,6",
        help="Only include JSON whose pilot_pattern matches (empty disables).",
    )
    p.add_argument(
        "--snrs_db",
        type=str,
        default=_DEFAULT_SNRS_DB_STR,
        help=(
            "Comma-separated SNR values (dB) to plot; each must appear in result JSON "
            f"snrs_db. Default: {_DEFAULT_SNRS_DB_STR}."
        ),
    )
    p.add_argument(
        "--dec_params",
        type=str,
        default="",
        help=(
            "Comma-separated dec_params to plot: raw slug from the checkpoint path "
            "(e.g. tk4_sm05_dec1_normpatch_mse) or the shortened legend form when unique "
            "(e.g. tk4_sm05_dec1). Omit or empty to plot all. Order is preserved."
        ),
    )
    p.add_argument(
        "--shade",
        action="store_true",
        help=(
            "Draw semi-transparent NMSE (dB) uncertainty bands: ID = std across cities "
            "in linear NMSE then delta dB; OOD = linear std from nmse_by_snr_linear per SNR."
        ),
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("ood", "id", "both"),
        default="ood",
        help="Which figure(s) to write: OOD (LA) only, ID (four-city mean) only, or both.",
    )
    p.add_argument("--png", action="store_true", help="Write PNG output.")
    p.add_argument("--pdf", action="store_true", help="Write PDF output.")
    p.add_argument(
        "--no_title",
        "--no-title",
        action="store_true",
        dest="no_title",
        help="Disable plot titles.",
    )
    p.add_argument(
        "--validate_only",
        action="store_true",
        help="Load and check coverage only; do not write plots.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.validate_only and not args.png and not args.pdf:
        raise SystemExit("Select --png and/or --pdf, or use --validate_only.")

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"--results_dir is not a directory: {results_dir}")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (results_dir / "figures").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_pat = str(args.pilot_pattern) if args.pilot_pattern else ""
    id_by_dec, ood_by_dec, _ = _load_grouped(results_dir, pilot_pat)

    want_snrs = _parse_snrs_db_arg(str(args.snrs_db))
    want_dec = _parse_dec_params_list(str(args.dec_params))
    id_keys = sorted(id_by_dec.keys())
    ood_keys = sorted(ood_by_dec.keys())
    if args.split == "id":
        available_keys = id_keys
    elif args.split == "ood":
        available_keys = ood_keys
    else:
        available_keys = sorted(set(id_keys) & set(ood_keys))
        only_id = sorted(set(id_keys) - set(ood_keys))
        only_ood = sorted(set(ood_keys) - set(id_keys))
        if only_id or only_ood:
            print(
                "Warning: dec_params mismatch between ID and OOD for --split both; "
                f"using intersection only. only_id={only_id} only_ood={only_ood}"
            )
    dec_keys = _resolve_dec_keys(available_keys, want_dec)
    if not dec_keys:
        raise ValueError("No dec_params to plot after filtering.")
    shade = bool(args.shade)
    id_curves: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    ood_curves: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    for dec in dec_keys:
        leg = _legend_num_dec_layers_line(dec)
        if args.split in ("id", "both"):
            snr_i, m_i, s_i = _aggregate_id_across_cities(id_by_dec[dec], shade=shade)
            snr_i, m_i, s_i = _subset_curve_by_snrs(snr_i, m_i, want_snrs, s_i)
            id_curves.append((leg, snr_i, m_i, s_i))
        if args.split in ("ood", "both"):
            snr_o, m_o, s_o = _aggregate_ood_single(ood_by_dec[dec], shade=shade)
            snr_o, m_o, s_o = _subset_curve_by_snrs(snr_o, m_o, want_snrs, s_o)
            ood_curves.append((leg, snr_o, m_o, s_o))
        if args.split == "both":
            if not np.allclose(snr_i, snr_o):
                raise ValueError(f"SNR mismatch ID vs OOD after subset for dec={dec}")

    if args.validate_only:
        print("Validation OK.")
        print(
            "  dec_params in folder: "
            f"id({len(id_keys)}): {', '.join(id_keys)} | "
            f"ood({len(ood_keys)}): {', '.join(ood_keys)}"
        )
        print(f"  dec_params plotted ({len(dec_keys)}): {', '.join(dec_keys)}")
        print("  plot legend: " + ", ".join(_legend_num_dec_layers_line(d) for d in dec_keys))
        print(
            "  --dec_params short aliases (optional): "
            + ", ".join(f"{_dec_params_legend_label(d)} -> {d}" for d in dec_keys)
        )
        print(f"  SNR points in JSON: {list(_linear_nmse_at_snrs(id_by_dec[dec_keys[0]]['boston'])[0])}")
        print(f"  SNR points plotted (--snrs_db): {list(want_snrs)}")
        print(f"  --shade: {shade}")
        print(f"  --split: {args.split}")
        return

    slug_dir = _slug(results_dir.name)
    snr_tag = _snr_file_tag(want_snrs)
    dec_tag = _dec_file_tag(dec_keys) if want_dec is not None else ""
    shade_tag = "__shade" if shade else ""
    written: List[Path] = []
    if args.split in ("id", "both"):
        written.extend(
            _plot_split(
                id_curves,
                title="ID @ 3.5 GHz: dec-only NMSE vs SNR (mean over cities)",
                out_path_stem=out_dir / f"nmse_vs_snr__id__3_5GHz__dec_only__{slug_dir}{dec_tag}{snr_tag}{shade_tag}",
                write_png=bool(args.png),
                write_pdf=bool(args.pdf),
                no_title=bool(args.no_title),
                shade=shade,
            )
        )
    if args.split in ("ood", "both"):
        written.extend(
            _plot_split(
                ood_curves,
                title="OOD @ 3.5 GHz (LA): dec-only NMSE vs SNR",
                out_path_stem=out_dir / f"nmse_vs_snr__ood__3_5GHz__dec_only__{slug_dir}{dec_tag}{snr_tag}{shade_tag}",
                write_png=bool(args.png),
                write_pdf=bool(args.pdf),
                no_title=bool(args.no_title),
                shade=shade,
            )
        )
    print(f"Wrote {len(written)} file(s) to {out_dir}")
    for pth in written:
        print(str(pth))


if __name__ == "__main__":
    main()
