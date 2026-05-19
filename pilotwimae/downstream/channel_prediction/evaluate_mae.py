"""
Evaluate fixed-pilot MAE reconstruction (channel estimation / prediction by pilot pattern).

Loads a full PilotWiMAE checkpoint (encoder + decoder), including supervised
channel-estimation runs saved as ``model.type=temporalenc_ce`` (same module class as MAE).
For each test fold and SNR, adds AWGN only on pilot patch tokens, runs ``reconstruct_pilot_masked``, and reports
mean NMSE on masked patch indices. Reports/saves NMSE in dB (``10*log10(NMSE)``),
while retaining linear NMSE fields for backward compatibility. Progress uses tqdm unless
``--no_progress`` is set.

JSON output schema (top-level keys):
  ``checkpoint``, ``data_dir``, ``pilot_pattern``, ``pilot_meta``, ``snrs_db``,
  ``nmse_eps``, ``mean_complex_power_dataset``, ``noise_floor``, ``norm_patch_loss_training``,
  ``nmse_uses_norm_patch_denorm``, ``n_folds``, ``folds_nmse_db``, ``nmse_db_by_snr``.
  Linear fields are also included as ``folds_nmse_linear`` and ``nmse_by_snr_linear``.

  When the checkpoint used ``training.norm_patch_loss``, decoder outputs are
  denormalized per patch (see :func:`~pilotwimae.downstream.channel_prediction.norm_patch.denormalize_norm_patch_patches`)
  before NMSE so metrics match raw patch space.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pilotwimae.data import OptimizedPreloadedDataset
from pilotwimae.downstream.beam_prediction.pilot_pattern import (
    parse_pilot_pattern,
    pilot_visible_flat_keep,
)
from pilotwimae.models import PilotWiMAE
from pilotwimae.models.encoder_backbone import is_factorized_family

from .metrics import nmse_on_masked
from .noise import corrupt_pilot_patches
from .norm_patch import denormalize_norm_patch_patches

DEFAULT_SNRS_DB = [0, 5, 10, 15, 20, 25, 30]

_DATASET_MEAN_POWER_CHUNK_SAMPLES = 2048
# Reproducible per-fold DataLoader shuffle (same recipe as beam kNN eval).
_FOLD_SHUFFLE_STRIDE = 1_000_003
_TEST_SHUFFLE_SEED_OFFSET = 29
_DATALOADER_PREFETCH_FACTOR = 2
_AWGN_SEED_MULT_BASE = 1_000_003
_AWGN_SEED_MULT_FOLD = 97_621_831
_AWGN_SNR_KEY_SCALE = 1_000
_GENERATOR_SEED_MODULUS = 2**31

# MAE pretraining and supervised CE both use :class:`~pilotwimae.models.base.PilotWiMAE`.
_CHANNEL_MAE_EVAL_CKPT_KINDS = frozenset({"pilotwimae", "temporalenc_ce"})


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


def _infer_norm_patch_loss(cfg: dict) -> tuple[bool, float]:
    train = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    if not isinstance(train, dict):
        return False, 1e-6
    use = bool(train.get("norm_patch_loss", False))
    eps = float(train.get("norm_patch_loss_eps", 1e-6))
    return use, eps


def _infer_checkpoint_model_kind(cfg: dict) -> str:
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


def _nmse_to_db(nmse_linear: float, eps: float) -> float:
    # Clamp to eps so log10 is well-defined and consistent with NMSE denominator guarding.
    x = max(float(nmse_linear), float(eps))
    return 10.0 * float(torch.log10(torch.tensor(x, dtype=torch.float64)).item())


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
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-pilot MAE channel reconstruction (NMSE on masked patches)."
    )
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
        help="CV folds over the dataset (disjoint test sets). Use 1 for a single random split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and noise generators.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="PilotWiMAE .pt checkpoint (model.type pilotwimae or temporalenc_ce). config.yaml must sit beside it.",
    )
    parser.add_argument("--mean_power", type=float, default=None, help="Override mean_power for normalization.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--pilot_pattern",
        type=str,
        required=True,
        help='Pilot layout, e.g. t:2,11;f:0,2,4,6',
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SNRS_DB),
        help='Comma-separated pilot SNRs in dB (default "0,5,...,30").',
    )
    parser.add_argument(
        "--nmse_eps",
        type=float,
        default=1e-12,
        help="Epsilon for NMSE denominator clamp.",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory for the result JSON.")
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help="JSON filename stem under save_dir (default: checkpoint parent folder name).",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars (e.g. for batch logs).",
    )
    parser.add_argument(
        "--noise_floor",
        action="store_true",
        help=(
            "Use dataset mean |h|² as global P_s on pilot tokens (fixed noise floor). "
            "If omitted, P_s is the per-sample mean |h|² over pilot patch elements."
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
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    ckpt_cfg = _read_checkpoint_parent_config(checkpoint_path)
    norm_patch_loss, norm_patch_eps = _infer_norm_patch_loss(ckpt_cfg)
    inferred_mean_power = _infer_mean_power(ckpt_cfg)
    if args.mean_power is not None:
        mean_power = float(args.mean_power)
    else:
        if inferred_mean_power is None:
            raise ValueError("mean_power not in config.yaml; pass --mean_power")
        mean_power = inferred_mean_power

    npz_files = _build_sorted_npz_list(data_dir)
    base_dataset = OptimizedPreloadedDataset(npz_files=npz_files, statistics={"mean_power": mean_power})
    ds_power = _compute_dataset_mean_complex_power(base_dataset)
    if ds_power <= 0:
        raise ValueError(f"Dataset mean complex power must be positive, got {ds_power}")
    if args.noise_floor:
        print(
            f"Mean |h|^2 on data after sqrt(mean_power) scaling: {ds_power:.6f} "
            "(pilot AWGN: fixed noise floor P_s = this value)"
        )
    else:
        print(
            f"Mean |h|^2 on data after sqrt(mean_power) scaling: {ds_power:.6f} "
            "(pilot AWGN: per-channel P_s = mean |h|² over pilot patches; dataset mean is diagnostic only)"
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
    if not isinstance(model, PilotWiMAE):
        raise ValueError(
            "Channel MAE reconstruction requires a PilotWiMAE checkpoint "
            "(pilotwimae or temporalenc_ce). Beam/LoS supervised checkpoints load a different class."
        )
    ckpt_kind = _infer_checkpoint_model_kind(ckpt_cfg)
    if ckpt_kind not in _CHANNEL_MAE_EVAL_CKPT_KINDS:
        allowed = ", ".join(sorted(_CHANNEL_MAE_EVAL_CKPT_KINDS))
        raise ValueError(
            f"Unsupported checkpoint model.type={ckpt_kind!r} for this eval (allowed: {allowed})."
        )

    if str(model.embedding_type).lower() != "linear":
        raise ValueError(
            f"Pilot-patch AWGN + inverse_patcher path requires embedding.type=linear; got {model.embedding_type!r}"
        )

    model.eval()
    nt, ns, nf = model.grid_dims
    t_nums, f_nums = parse_pilot_pattern(args.pilot_pattern)
    Tk = len(sorted(set(t_nums)))
    Sk = len(sorted(set(f_nums))) * ns
    pilot_factorized_grid: Optional[Tuple[int, int]] = None
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

    folds_nmse_linear: list[list[float]] = []
    disable_pbar = bool(args.no_progress)

    for fold_idx, (_train_base, test_base) in enumerate(fold_pairs):
        g_test = torch.Generator()
        g_test.manual_seed(
            int(args.seed) + fold_idx * _FOLD_SHUFFLE_STRIDE + _TEST_SHUFFLE_SEED_OFFSET
        )
        test_loader = DataLoader(
            test_base,
            shuffle=True,
            generator=g_test,
            **base_loader_kwargs,
        )
        fold_row: list[float] = []
        for snr_db in snr_list:
            gen = torch.Generator(device=device)
            gen.manual_seed(_awgn_generator_seed(args.seed, snr_db, fold_idx))
            sum_nmse = 0.0
            n_batches = 0
            desc = f"Fold {fold_idx + 1}/{n_folds_eff} SNR {snr_db} dB"
            pbar = (
                None
                if disable_pbar
                else tqdm(
                    test_loader,
                    desc=desc,
                    leave=True,
                    dynamic_ncols=True,
                )
            )
            iterator = pbar if pbar is not None else test_loader
            with torch.no_grad():
                for batch in iterator:
                    x = batch.to(device)
                    patches_clean = model.patcher(x)
                    patches_obs = corrupt_pilot_patches(
                        patches_clean,
                        pilot_flat_keep,
                        snr_db,
                        signal_mean_power=ds_power if args.noise_floor else None,
                        generator=gen,
                    )
                    x_obs = model.inverse_patcher(patches_obs)
                    out = model.reconstruct_pilot_masked(
                        x_obs,
                        pilot_flat_keep,
                        pilot_factorized_grid=pilot_factorized_grid,
                    )
                    recon_p = out["reconstructed_patches"]
                    if norm_patch_loss:
                        recon_p = denormalize_norm_patch_patches(
                            recon_p, patches_clean, eps=norm_patch_eps
                        )
                    nmse_b = nmse_on_masked(
                        recon_p,
                        patches_clean,
                        out["ids_mask"],
                        eps=float(args.nmse_eps),
                    )
                    sum_nmse += float(nmse_b.item())
                    n_batches += 1
                    if pbar is not None:
                        nmse_lin_running = sum_nmse / max(n_batches, 1)
                        nmse_db_running = _nmse_to_db(nmse_lin_running, float(args.nmse_eps))
                        pbar.set_postfix(
                            nmse_db_mean=f"{nmse_db_running:.3f}",
                            nmse_lin_mean=f"{nmse_lin_running:.6f}",
                        )
            fold_row.append(sum_nmse / max(n_batches, 1))
        folds_nmse_linear.append(fold_row)

    nmse_by_snr_linear: dict[str, dict[str, float]] = {}
    nmse_db_by_snr: dict[str, dict[str, float]] = {}
    for j, snr_db in enumerate(snr_list):
        col_lin = [folds_nmse_linear[i][j] for i in range(n_folds_eff)]
        key = str(int(snr_db) if snr_db == int(snr_db) else snr_db)
        stats_lin = _mean_std(col_lin)
        nmse_by_snr_linear[key] = stats_lin
        mean_lin = float(stats_lin["mean"])
        std_lin = float(stats_lin["std"])
        # Report in dB after aggregating in linear scale.
        mean_db = _nmse_to_db(mean_lin, float(args.nmse_eps))
        lo_db = _nmse_to_db(max(mean_lin - std_lin, 0.0), float(args.nmse_eps))
        hi_db = _nmse_to_db(mean_lin + std_lin, float(args.nmse_eps))
        nmse_db_by_snr[key] = {
            "mean": mean_db,
            # Asymmetric spread in dB induced by log transform.
            "std_minus": mean_db - lo_db,
            "std_plus": hi_db - mean_db,
            # Keep symmetric proxy for compatibility with code that expects one std value.
            "std": 0.5 * ((mean_db - lo_db) + (hi_db - mean_db)),
        }

    stem = args.output_stem if args.output_stem else checkpoint_path.parent.name
    out_path = save_dir / f"{stem}.json"
    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_kind": ckpt_kind,
        "data_dir": str(data_dir),
        "pilot_pattern": args.pilot_pattern,
        "pilot_meta": pilot_meta,
        "snrs_db": snr_list,
        "nmse_eps": float(args.nmse_eps),
        "mean_complex_power_dataset": ds_power,
        "noise_floor": bool(args.noise_floor),
        "norm_patch_loss_training": norm_patch_loss,
        "nmse_uses_norm_patch_denorm": norm_patch_loss,
        "nmse_scale_primary": "db",
        "n_folds": n_folds_eff,
        "folds_nmse_db": [
            [_nmse_to_db(v, float(args.nmse_eps)) for v in row] for row in folds_nmse_linear
        ],
        "nmse_db_by_snr": nmse_db_by_snr,
        # Backward-compatible linear fields:
        "folds_nmse_linear": folds_nmse_linear,
        "nmse_by_snr_linear": nmse_by_snr_linear,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("NMSE(dB) by SNR [aggregate in linear, then convert to dB]:")
    for snr_db in snr_list:
        key = str(int(snr_db) if snr_db == int(snr_db) else snr_db)
        stats = nmse_db_by_snr[key]
        print(
            f"  SNR={snr_db:>5} dB: {stats['mean']:.3f} dB "
            f"(+{stats['std_plus']:.3f} / -{stats['std_minus']:.3f})"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
