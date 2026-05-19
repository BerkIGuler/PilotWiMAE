"""
CLI to evaluate kNN classification using PilotWiMAE embeddings.

This script:
1) Loads a PilotWiMAE checkpoint.
2) Builds K-fold train/test splits (default K=10, disjoint 10% test folds) or a
   single random split when `--n_folds 1`.
3) Generates per-sample beam classification labels on-the-fly using the
   beam_prediction utilities.
4) Each fold: fits kNN on noiseless train embeddings and evaluates on the test set.
5) Aggregates noiseless top-1..top-5 and per-SNR top-1..top-5 (complex AWGN) as
   mean and sample std across folds. Training-style normalization (checkpoint
   ``mean_power`` → sqrt scaling for embeddings) is separate from AWGN calibration:
   with ``--noise_floor``, P_s is the dataset mean |h|² (fixed noise floor); without
   it, P_s is the mean |h|² of each channel sample. Results JSON keeps
   ``normalization.mean_power``, ``awgn.mean_complex_power``, and ``awgn.noise_floor``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import yaml

from pilotwimae.data import OptimizedPreloadedDataset, BeamLabelDatasetWrapper, add_complex_awgn_snr_db
from pilotwimae.downstream.models.knn import kNNforClassification
from pilotwimae.models import PilotWiMAE
from pilotwimae.models.encoder_backbone import is_factorized_family
from pilotwimae.downstream.beam_prediction.pilot_pattern import (
    parse_pilot_pattern,
    pilot_visible_flat_keep,
)
from torch.utils.data import DataLoader, Subset

DEFAULT_SNRS_DB = [0, 5, 10, 15, 20, 25, 30]

_DATASET_MEAN_POWER_CHUNK_SAMPLES = 2048

# Reproducible per-fold DataLoader shuffle (train / test disjoint subsets unchanged).
_FOLD_SHUFFLE_STRIDE = 1_000_003
_TRAIN_SHUFFLE_SEED_OFFSET = 17
_TEST_SHUFFLE_SEED_OFFSET = 29

# Deterministic AWGN generator seeds per (base_seed, snr_db, fold_idx).
_AWGN_SEED_MULT_BASE = 1_000_003
_AWGN_SEED_MULT_FOLD = 97_621_831
_AWGN_SNR_KEY_SCALE = 1_000
_GENERATOR_SEED_MODULUS = 2**31

# DataLoader (used when num_workers > 0).
_DATALOADER_PREFETCH_FACTOR = 2


def _read_checkpoint_parent_config(checkpoint_path: Path) -> dict:
    parent_dir = checkpoint_path.parent
    cfg_path = parent_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Expected config.yaml next to checkpoint: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _infer_mean_power(cfg: dict) -> Optional[float]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    stats = data_cfg.get("statistics", {}) if isinstance(data_cfg, dict) else {}
    mean_power = stats.get("mean_power", None) if isinstance(stats, dict) else None
    if mean_power is None:
        return None
    return float(mean_power)


def _infer_model_type(cfg: dict) -> Optional[str]:
    """Encoder backbone: ``model.encoder_type`` (``standard`` | ``factorized`` | ``factorized_mixing``).

    This matches MAE and supervised-beam checkpoints: both store ``encoder_type``
    under ``model``. It is **not** the same as ``model.type``
    (``pilotwimae`` vs ``temporalenc_beam``).
    """
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    enc_type = model_cfg.get("encoder_type", None) if isinstance(model_cfg, dict) else None
    if enc_type is None:
        return None
    return str(enc_type)


def _infer_checkpoint_model_kind(cfg: dict) -> str:
    """Training recipe: ``model.type`` (default ``pilotwimae`` for legacy YAML)."""
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        return "pilotwimae"
    mt = model_cfg.get("type", "pilotwimae")
    if not isinstance(mt, str):
        return "pilotwimae"
    return mt.lower()


def _build_sorted_npz_list(data_dir: Path) -> list[str]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"--data_dir must be a directory: {data_dir}")
    npz_files = sorted([str(p) for p in data_dir.rglob("*.npz")])
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found under data_dir: {data_dir}")
    return npz_files


def _compute_dataset_mean_complex_power(
    dataset: OptimizedPreloadedDataset,
    *,
    chunk_samples: int = _DATASET_MEAN_POWER_CHUNK_SAMPLES,
) -> float:
    """Mean |h|^2 over all tensor elements (accumulated in FP64, chunked along batch)."""
    total_sq = 0.0
    total_el = 0
    h = dataset.all_data
    for i in range(0, h.shape[0], chunk_samples):
        batch = h[i : i + chunk_samples]
        total_sq += (batch.real.to(torch.float64) ** 2 + batch.imag.to(torch.float64) ** 2).sum().item()
        total_el += batch.numel()
    return total_sq / total_el


def _parse_snrs(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--snrs must list at least one SNR (dB), comma-separated.")
    return [float(p) for p in parts]


def _make_noisy_transform(
    snr_db: float,
    generator: torch.Generator,
    *,
    signal_mean_power: float,
    noise_floor: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _t(x: torch.Tensor) -> torch.Tensor:
        return add_complex_awgn_snr_db(
            x,
            snr_db,
            generator=generator,
            signal_mean_power=signal_mean_power,
            noise_floor=noise_floor,
        )

    return _t


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    vals = [float(x) for x in values]
    n = len(vals)
    if n == 0:
        raise ValueError("empty values for mean/std")
    mean = sum(vals) / n
    if n < 2:
        return {"mean": mean, "std": 0.0}
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return {"mean": mean, "std": float(var**0.5)}


def _balanced_fold_segments(n: int, n_folds: int, generator: torch.Generator) -> list[torch.Tensor]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if n < n_folds:
        raise ValueError(f"Dataset size {n} must be >= n_folds ({n_folds})")
    perm = torch.randperm(n, generator=generator)
    base = n // n_folds
    rem = n % n_folds
    segments: list[torch.Tensor] = []
    start = 0
    for i in range(n_folds):
        sz = base + (1 if i < rem else 0)
        segments.append(perm[start : start + sz])
        start += sz
    return segments


def _make_fold_train_test_subsets(
    base_dataset: torch.utils.data.Dataset,
    *,
    n_folds: int,
    test_split: float,
    seed: int,
) -> list[Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]]:
    """Disjoint K-fold test sets (default behavior) or one random 90/10-style split when n_folds==1."""
    n = len(base_dataset)
    gen = torch.Generator().manual_seed(int(seed))
    if n_folds == 1:
        test_size = int(n * test_split)
        train_size = n - test_size
        if train_size <= 0 or test_size <= 0:
            raise ValueError(
                f"Invalid train/test sizes for n={n}, test_split={test_split}: train={train_size}, test={test_size}"
            )
        train_b, test_b = torch.utils.data.random_split(
            base_dataset, [train_size, test_size], generator=gen
        )
        return [(train_b, test_b)]

    if abs(n_folds * test_split - 1.0) > 1e-5:
        raise ValueError(
            "For n_folds > 1, require n_folds * test_split ≈ 1 "
            f"(got {n_folds} * {test_split} = {n_folds * test_split})"
        )

    segs = _balanced_fold_segments(n, n_folds, gen)
    pairs: list[Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]] = []
    for i in range(n_folds):
        test_ix = segs[i].tolist()
        train_ix = torch.cat([segs[j] for j in range(n_folds) if j != i]).tolist()
        pairs.append((Subset(base_dataset, train_ix), Subset(base_dataset, test_ix)))
    return pairs


def _awgn_generator_seed(base_seed: int, snr_db: float, fold_idx: int) -> int:
    snr_key = int(round(float(snr_db) * _AWGN_SNR_KEY_SCALE))
    mixed = (
        int(base_seed) * _AWGN_SEED_MULT_BASE
        + snr_key
        + int(fold_idx) * _AWGN_SEED_MULT_FOLD
    )
    return mixed % _GENERATOR_SEED_MODULUS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate kNN classification with PilotWiMAE embeddings.")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory with NPZ files.")
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test fraction per fold. For n_folds>1 must equal 1/n_folds (e.g. 0.1 for 10 folds).",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (disjoint test sets). Use 1 for a single random train/test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and (with SNR) noise.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers.")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to a PilotWiMAE .pt checkpoint. A config.yaml must exist next to it.",
    )

    parser.add_argument("--mean_power", type=float, default=None, help="Override mean power for channel normalization.")

    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["factorized", "factorized_mixing", "standard"],
        help=(
            "Optional override/sanity-check for model.encoder_type in the run's "
            "config.yaml (standard vs factorized / factorized_mixing). Applies to both MAE and "
            "supervised-beam checkpoints; it does not select temporalenc_beam—"
            "loading uses PilotWiMAE.from_checkpoint, which picks the right class."
        ),
    )
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"], help="Pooling over encoded tokens.")
    parser.add_argument(
        "--inference_token_mode",
        type=str,
        default="full_grid",
        choices=["full_grid", "masked_visible", "pilot_visible"],
        help=(
            "Embedding token mode: 'full_grid' encodes all patch tokens; "
            "'masked_visible' uses MAE masking visibility (pilotwimae only); "
            "'pilot_visible' uses fixed pilot indices from --pilot_pattern "
            "(MAE or supervised temporalenc_beam)."
        ),
    )
    parser.add_argument(
        "--pilot_pattern",
        type=str,
        default=None,
        help=(
            "Required when --inference_token_mode=pilot_visible. "
            "Format: t:<t0>,<t1>,...;f:<f0>,<f1>,... (0-based grid indices; "
            "f lists frequency indices, expanded over all spatial antenna indices ns)."
        ),
    )

    parser.add_argument("--n_h", type=int, required=True, help="UPA horizontal elements.")
    parser.add_argument("--n_v", type=int, required=True, help="UPA vertical elements.")
    parser.add_argument("--o_h", type=int, default=1, help="Horizontal oversampling factor.")
    parser.add_argument("--o_v", type=int, default=1, help="Vertical oversampling factor.")
    parser.add_argument(
        "--u_h",
        type=int,
        default=1,
        help="Horizontal undersampling factor (K_h=N_h/U_h); use with o_h=1; must divide N_h.",
    )
    parser.add_argument(
        "--u_v",
        type=int,
        default=1,
        help="Vertical undersampling factor (K_v=N_v/U_v); use with o_v=1; must divide N_v.",
    )
    parser.add_argument("--antenna_order", type=str, default="hv", choices=["hv", "vh"], help="Antenna linearization order.")

    parser.add_argument("--k", type=int, default=50, help="k for kNN.")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="Similarity metric.")

    parser.add_argument(
        "--snrs",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SNRS_DB),
        help='Comma-separated test SNRs in dB (default "0,5,...,30").',
    )
    parser.add_argument(
        "--noise_floor",
        action="store_true",
        help=(
            "Use dataset mean |h|² as global P_s for AWGN (fixed noise floor). "
            "If omitted, P_s is the per-sample mean |h|² over all complex elements."
        ),
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory for the result JSON (see --output_stem).",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help=(
            "Base filename (no extension) for the result JSON under save_dir "
            "(default: parent directory name of --checkpoint_path)."
        ),
    )

    args = parser.parse_args()
    if args.n_folds < 1:
        raise SystemExit("error: --n_folds must be >= 1")

    data_dir = Path(args.data_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    snr_list = _parse_snrs(args.snrs)

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_cfg = _read_checkpoint_parent_config(checkpoint_path)

    inferred_mean_power = _infer_mean_power(ckpt_cfg)
    if args.mean_power is not None:
        mean_power = float(args.mean_power)
    else:
        if inferred_mean_power is None:
            raise ValueError("mean_power not found in checkpoint config.yaml and was not provided via --mean_power")
        mean_power = inferred_mean_power

    inferred_model_type = _infer_model_type(ckpt_cfg)
    if args.model_type is not None:
        if inferred_model_type is not None and args.model_type != inferred_model_type:
            raise ValueError(
                f"--model_type={args.model_type} does not match checkpoint config encoder_type={inferred_model_type}"
            )
        model_type = args.model_type
    else:
        model_type = inferred_model_type

    npz_files = _build_sorted_npz_list(data_dir)
    base_dataset = OptimizedPreloadedDataset( 
        npz_files=npz_files,
        statistics={"mean_power": mean_power},
    )
    ds_power = _compute_dataset_mean_complex_power(base_dataset)
    if ds_power <= 0:
        raise ValueError(f"Dataset mean complex power must be positive, got {ds_power}")
    if args.noise_floor:
        print(
            f"Mean |h|^2 on data after sqrt(mean_power) scaling: {ds_power:.6f} "
            "(AWGN: fixed noise floor P_s = this value)"
        )
    else:
        print(
            f"Mean |h|^2 on data after sqrt(mean_power) scaling: {ds_power:.6f} "
            "(AWGN: per-channel P_s = mean |h|² per sample; dataset mean is diagnostic only)"
        )

    fold_pairs = _make_fold_train_test_subsets(
        base_dataset,
        n_folds=args.n_folds,
        test_split=args.test_split,
        seed=args.seed,
    )
    n_folds_eff = len(fold_pairs)

    pin_memory = device.type == "cuda"
    base_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0:
        base_loader_kwargs.update(
            {"prefetch_factor": _DATALOADER_PREFETCH_FACTOR, "persistent_workers": True}
        )

    model = PilotWiMAE.from_checkpoint(str(checkpoint_path), device=device)
    model.eval()
    ckpt_kind = _infer_checkpoint_model_kind(ckpt_cfg)

    if args.pilot_pattern is not None and args.inference_token_mode != "pilot_visible":
        raise ValueError(
            "--pilot_pattern is only valid with --inference_token_mode=pilot_visible"
        )
    if args.inference_token_mode == "pilot_visible" and not args.pilot_pattern:
        raise ValueError(
            "--pilot_pattern is required when --inference_token_mode=pilot_visible"
        )

    if args.inference_token_mode == "masked_visible" and ckpt_kind != "pilotwimae":
        raise ValueError(
            "--inference_token_mode=masked_visible is only valid for MAE checkpoints "
            "(model.type=pilotwimae)."
        )
    if args.inference_token_mode == "pilot_visible" and ckpt_kind not in (
        "pilotwimae",
        "temporalenc_beam",
        "temporalenc_los",
    ):
        raise ValueError(
            f"--inference_token_mode=pilot_visible is not supported for model.type={ckpt_kind}. "
            "Use pilotwimae (MAE) or supervised temporalenc_beam / temporalenc_los."
        )

    pilot_flat_keep: Optional[torch.Tensor] = None
    pilot_factorized_grid: Optional[Tuple[int, int]] = None
    pilot_meta: dict = {}
    if args.inference_token_mode == "pilot_visible":
        t_nums, f_nums = parse_pilot_pattern(args.pilot_pattern)
        nt, ns, nf = model.grid_dims
        Tk = len(sorted(set(t_nums)))
        Sk = len(sorted(set(f_nums))) * ns
        if is_factorized_family(model.encoder_type):
            pilot_factorized_grid = (Tk, Sk)
            nt_keep = int(model.encoder.num_time_keep)
            nsp_keep = int(model.encoder.num_spatial_keep)
            if Tk != nt_keep or Sk != nsp_keep:
                warnings.warn(
                    f"pilot_visible: pilot geometry Tk={Tk}, Sk={Sk} does not match factorized "
                    f"encoder num_time_keep={nt_keep}, num_spatial_keep={nsp_keep} from pretraining. "
                    f"Using pilot (Tk, Sk) as FactorizedEncoder time_steps/spatial_steps.",
                    UserWarning,
                    stacklevel=1,
                )
        pilot_batch = pilot_visible_flat_keep(nt, ns, nf, t_nums, f_nums, device=device)
        pilot_flat_keep = pilot_batch.squeeze(0)
        p_keep = int(pilot_flat_keep.numel())
        if Tk * Sk != p_keep:
            raise ValueError(f"Internal error: Tk*Sk={Tk*Sk} != pilot_flat_keep length {p_keep}")
        p_tot = int(model.num_patches)
        pilot_meta = {
            "pilot_pattern": args.pilot_pattern,
            "pilot_num_keep": p_keep,
            "pilot_factorized_tk": Tk,
            "pilot_factorized_sk": Sk,
            "num_patches": p_tot,
            "pilot_effective_mask_ratio": (p_tot - p_keep) / p_tot if p_tot else 0.0,
        }

    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        if ckpt_kind == "pilotwimae":
            if args.inference_token_mode == "pilot_visible":
                return model.get_embeddings(
                    x,
                    pooling=args.pooling,
                    token_mode="pilot_visible",
                    pilot_flat_keep=pilot_flat_keep,
                    pilot_factorized_grid=pilot_factorized_grid,
                )
            return model.get_embeddings(
                x,
                pooling=args.pooling,
                token_mode=args.inference_token_mode,
            )
        if args.inference_token_mode == "pilot_visible":
            return model.get_embeddings(
                x,
                pooling=args.pooling,
                token_mode="pilot_visible",
                pilot_flat_keep=pilot_flat_keep,
                pilot_factorized_grid=pilot_factorized_grid,
            )
        return model.get_embeddings(x, pooling=args.pooling)

    fold_accs: list[float] = []
    fold_noiseless: list[dict[str, float]] = []
    fold_snr_tables: list[list[dict]] = []

    for fold_idx, (train_base, test_base) in enumerate(fold_pairs):
        train_ds = BeamLabelDatasetWrapper(
            train_base,
            n_h=args.n_h,
            n_v=args.n_v,
            o_h=args.o_h,
            o_v=args.o_v,
            u_h=args.u_h,
            u_v=args.u_v,
            antenna_order=args.antenna_order,
            label_mode="snapshot",
            return_format="class_index",
            top_k=1,
        )
        test_ds = BeamLabelDatasetWrapper(
            test_base,
            n_h=args.n_h,
            n_v=args.n_v,
            o_h=args.o_h,
            o_v=args.o_v,
            u_h=args.u_h,
            u_v=args.u_v,
            antenna_order=args.antenna_order,
            label_mode="snapshot",
            return_format="class_index",
            top_k=1,
        )
        # Shuffle batch order within each fold’s train/test subsets (same disjoint split as before).
        g_train = torch.Generator()
        g_train.manual_seed(
            int(args.seed) + fold_idx * _FOLD_SHUFFLE_STRIDE + _TRAIN_SHUFFLE_SEED_OFFSET
        )
        g_test = torch.Generator()
        g_test.manual_seed(
            int(args.seed) + fold_idx * _FOLD_SHUFFLE_STRIDE + _TEST_SHUFFLE_SEED_OFFSET
        )
        train_loader = DataLoader(
            train_ds,
            collate_fn=train_ds.make_collate_fn(),
            shuffle=True,
            generator=g_train,
            **base_loader_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            collate_fn=test_ds.make_collate_fn(),
            shuffle=True,
            generator=g_test,
            **base_loader_kwargs,
        )

        knn = kNNforClassification(
            k=args.k,
            metric=args.metric,
            encode_fn=encode_fn,
            device=device,
        )
        knn.fit(train_loader)
        metrics = knn.test(test_loader)
        noiseless_topk = knn.test_topk(test_loader, max_k=5)

        fold_accs.append(float(metrics["accuracy"]))
        fold_noiseless.append(dict(noiseless_topk))

        rows_one_fold: list[dict] = []
        row_clean = {"snr_db": "clean", **{f"top_{i}": noiseless_topk[f"top_{i}"] for i in range(1, 6)}}
        rows_one_fold.append(row_clean)

        for snr_db in snr_list:
            noise_gen = torch.Generator(device=device)
            noise_gen.manual_seed(_awgn_generator_seed(args.seed, snr_db, fold_idx))
            noisy_t = _make_noisy_transform(
                snr_db, noise_gen, signal_mean_power=ds_power, noise_floor=args.noise_floor
            )
            m = knn.test_topk(test_loader, max_k=5, input_transform=noisy_t)
            s = float(snr_db)
            row = {"snr_db": str(int(s)) if s == int(s) else str(s)}
            for i in range(1, 6):
                row[f"top_{i}"] = m[f"top_{i}"]
            rows_one_fold.append(row)
        fold_snr_tables.append(rows_one_fold)

    noiseless_agg: dict[str, dict[str, float]] = {}
    for ki in range(1, 6):
        key = f"top_{ki}"
        noiseless_agg[key] = _mean_std([d[key] for d in fold_noiseless])

    n_snr_rows = 1 + len(snr_list)
    snr_robustness_agg: list[dict] = []
    for ri in range(n_snr_rows):
        snr_db_key = fold_snr_tables[0][ri]["snr_db"]
        row_agg: dict = {"snr_db": snr_db_key}
        for ti in range(1, 6):
            col = f"top_{ti}"
            row_agg[col] = _mean_std([fold_snr_tables[f][ri][col] for f in range(n_folds_eff)])
        snr_robustness_agg.append(row_agg)

    output_stem = args.output_stem if args.output_stem is not None else checkpoint_path.parent.name
    experiment = {
        "data_dir": str(data_dir),
        "dataset_id": data_dir.name,
        "checkpoint_run_name": checkpoint_path.parent.name,
    }

    out = {
        "accuracy": _mean_std(fold_accs),
        "noiseless_top_k": noiseless_agg,
        "experiment": experiment,
        "snr_robustness": snr_robustness_agg,
        "snrs_db": snr_list,
        "cv": {"n_folds": args.n_folds, "test_split": args.test_split, "seed": args.seed},
        "k": args.k,
        "metric": args.metric,
        "pooling": args.pooling,
        "codebook": {
            "n_h": args.n_h,
            "n_v": args.n_v,
            "o_h": args.o_h,
            "o_v": args.o_v,
            "u_h": args.u_h,
            "u_v": args.u_v,
            "antenna_order": args.antenna_order,
        },
        "normalization": {"mean_power": mean_power},
        "awgn": {"mean_complex_power": ds_power, "noise_floor": bool(args.noise_floor)},
        "model": {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "model_type": model_type,
            "checkpoint_kind": ckpt_kind,
            "inference_token_mode": args.inference_token_mode,
            **pilot_meta,
        },
    }

    output_path = save_dir / f"{output_stem}.json"

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
