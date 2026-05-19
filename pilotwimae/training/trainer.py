"""
Base trainer class for PilotWiMAE.
"""

import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split

from pilotwimae.models import PilotWiMAE, PilotWiMAEBeamClassifier
from pilotwimae.data import (
    OptimizedPreloadedDataset,
    create_efficient_dataloader,
    calculate_mean_power,
)
from pilotwimae.data.beam import upa_2d_dft_num_beams
from .losses import PerSampleNMSE
from .utils import (
    safe_torch_load as _safe_torch_load,
    resolve_model_type,
    generate_exp_name,
)

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer providing common training functionality for PilotWiMAE."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.config = config
        if device is not None:
            self.device = device
            self.config["training"]["device"] = str(device)
            logger.info("Device overridden to %s", device)
        else:
            self.device = torch.device(config["training"]["device"])

        self.model = self.setup_model()
        self._apply_encoder_freeze()
        self.optimizer = self.setup_optimizer(self.model)
        self.scheduler = self.setup_scheduler(self.optimizer)
        self.criterion = self.setup_criterion()
        # Optional mixed-precision training (AMP)
        self.use_mixed_precision = bool(
            self.config["training"].get("mixed_precision", False)
        )
        self.scaler = GradScaler(enabled=self.use_mixed_precision)
        self.writer = None  # tensorboard writer

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        # Cumulative epoch index for TensorBoard when continuing training.
        self._tensorboard_epoch_offset = 0

        self.setup_logging()
        self._maybe_resume_from_checkpoint()

    def setup_logging(self):
        log_dir = self.config["logging"]["log_dir"]
        model_type = resolve_model_type(self.config["model"])

        exp_name = self.config["logging"].get("exp_name", None)
        if not exp_name:
            exp_name = generate_exp_name(self.config)
        self.log_dir = Path(log_dir) / f"{model_type}_{exp_name}"

        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(self.log_dir / "config.yaml", "w") as f:
            # device field of config could be overridden by the command line!!
            yaml.dump(self.config, f, default_flow_style=False)

        # Remove any previously attached handlers to avoid duplicate log entries
        # when multiple trainer instances are created in the same process.
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Add file handler so logs are written to train.log
        log_file = self.log_dir / "train.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        if self.config["logging"].get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                logger.warning(
                    "tensorboard not available. Install with: pip install tensorboard"
                )
                self.writer = None

    def _maybe_resume_from_checkpoint(self) -> None:
        """
        Load weights / optimizer / scheduler / TB step from training.resume config.

        TensorBoard x-axis continues from ``last_global_epoch + 1`` in the checkpoint
        unless you set ``tensorboard_epoch_offset`` explicitly.

        ``checkpoint_path`` may be absolute, CWD-relative, or a filename only; if the
        file is missing, we also try ``<log_dir>/<filename>`` (same run folder).

        ``load_optimizer`` and ``load_scheduler`` default to True (full resume).
        For a fresh optimizer + cosine from the YAML, use::

            resume:
              checkpoint_path: runs/.../last_checkpoint.pt
              load_optimizer: false
              load_scheduler: false

        Set ``reset_best_val_loss: true`` to ignore the checkpoint's ``best_val_loss`` and
        treat the next validation as the first candidate for a new best (e.g. after
        changing the loss or starting a new fine-tuning phase).
        """
        resume = (self.config.get("training") or {}).get("resume") or {}
        path = resume.get("checkpoint_path")
        if not path:
            return

        path_p = Path(path).expanduser()
        if not path_p.is_file():
            # Allow `best_checkpoint.pt` / `last_checkpoint.pt` next to this run's log_dir
            alt = self.log_dir / path_p.name
            if alt.is_file():
                path_p = alt
            else:
                raise FileNotFoundError(
                    f"resume.checkpoint_path not found: {path} (also tried {alt})"
                )
        path = str(path_p)

        ckpt = _safe_torch_load(path, map_location=self.device)
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint missing model_state_dict: {path}")

        encoder_only = bool(resume.get("encoder_only", False))
        if encoder_only:
            # Load only encoder-side weights; decoder is kept randomly initialized.
            # All keys starting with "decoder." are excluded from the filtered dict
            # so load_state_dict(strict=False) leaves the new decoder untouched.
            full_sd = ckpt["model_state_dict"]
            encoder_sd = {
                k: v for k, v in full_sd.items() if not k.startswith("decoder.")
            }
            skipped = [k for k in full_sd if k.startswith("decoder.")]
            missing, unexpected = self.model.load_state_dict(encoder_sd, strict=False)
            # "missing" will contain the current decoder.* keys — that is expected.
            decoder_missing = [k for k in missing if k.startswith("decoder.")]
            other_missing = [k for k in missing if not k.startswith("decoder.")]
            logger.info(
                "Resume (encoder_only): loaded %d encoder keys, "
                "skipped %d decoder keys from checkpoint, "
                "%d decoder keys randomly initialized.",
                len(encoder_sd),
                len(skipped),
                len(decoder_missing),
            )
            if other_missing:
                logger.warning(
                    "Resume (encoder_only): non-decoder keys missing from checkpoint "
                    "(randomly initialized): %s",
                    other_missing,
                )
            if unexpected:
                logger.warning(
                    "Resume (encoder_only): unexpected keys ignored: %s",
                    list(unexpected),
                )
        else:
            # Allow non-strict resume so new parameters (e.g. auxiliary heads) can be
            # randomly initialized while loading preexisting encoder/decoder weights.
            strict_model = bool(resume.get("strict", True))
            if strict_model:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                missing, unexpected = self.model.load_state_dict(
                    ckpt["model_state_dict"], strict=False
                )
                if missing:
                    logger.warning("Resume: missing model keys (randomly initialized): %s", list(missing))
                if unexpected:
                    logger.warning("Resume: unexpected model keys (ignored): %s", list(unexpected))
                if not missing and not unexpected:
                    logger.info("Resume: all model weights loaded successfully")

        load_opt = resume.get("load_optimizer", True)
        # Default True so full resume restores LR schedule (set false for fresh cosine, etc.)
        load_sched = resume.get("load_scheduler", True)

        if load_opt and self.optimizer and ckpt.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif not load_opt:
            logger.info("Resume: skipped loading optimizer state (fresh optimizer)")

        if load_sched and self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif not load_sched:
            logger.info("Resume: skipped loading scheduler state (fresh scheduler)")

        if resume.get("reset_patience", False):
            self.patience_counter = 0
        else:
            self.patience_counter = int(ckpt.get("patience_counter", 0))

        if resume.get("reset_best_val_loss", False):
            self.best_val_loss = float("inf")
            prev = ckpt.get("best_val_loss")
            if prev is not None:
                logger.info("Resume: reset best_val_loss (was %s)", prev)
            else:
                logger.info("Resume: reset best_val_loss")
        else:
            self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.global_step = int(ckpt.get("global_step", 0))

        last_ge = ckpt.get("last_global_epoch")
        if last_ge is None:
            last_ge = ckpt.get("epoch", -1)
        last_ge = int(last_ge)

        explicit = resume.get("tensorboard_epoch_offset")
        if explicit is not None:
            self._tensorboard_epoch_offset = int(explicit)
        else:
            self._tensorboard_epoch_offset = last_ge + 1

        if self.use_mixed_precision and ckpt.get("scaler_state_dict"):
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as exc:
                logger.warning("Could not load GradScaler state: %s", exc)

        logger.info(
            "Resumed from %s | TB epochs start at %d (last_global_epoch in ckpt was %d)",
            path,
            self._tensorboard_epoch_offset,
            last_ge,
        )

    def _apply_encoder_freeze(self) -> None:
        """Freeze encoder parameters based on ``training.freeze_encoder`` config.

        When ``training.freeze_encoder: true``, every parameter whose name does
        **not** start with ``"decoder."`` has ``requires_grad`` set to ``False``.
        This must be called after ``setup_model()`` and before ``setup_optimizer()``
        so that frozen parameters are excluded from the optimizer automatically.

        For a soft freeze (encoder trained at a lower LR rather than not at all),
        leave ``freeze_encoder`` unset or ``false`` and instead set
        ``training.optimizer.encoder_lr`` to the desired smaller learning rate.
        """
        if not self.config.get("training", {}).get("freeze_encoder", False):
            return

        frozen = 0
        for name, param in self.model.named_parameters():
            if not name.startswith("decoder."):
                param.requires_grad_(False)
                frozen += 1

        trainable = sum(
            1 for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "Encoder frozen: %d parameter tensors set requires_grad=False. "
            "%d trainable tensors remain (decoder only).",
            frozen,
            trainable,
        )

    def setup_model(self) -> nn.Module:
        model_config = self.config["model"]
        # Ensure optional scale-loss config is stored under model.scale_loss so it is checkpointed.
        # Users may specify either:
        #   - training.scale_loss.{use_scale_loss,lambda_enc,lambda_dec,eps}
        #   - training.use_scale_loss + training.lambda_enc/lambda_dec/scale_loss_eps (flat keys)
        train_cfg = self.config.get("training", {})
        if isinstance(train_cfg, dict):
            sl_from_training = train_cfg.get("scale_loss", None)
            if sl_from_training is None:
                if (
                    "use_scale_loss" in train_cfg
                    or "lambda_enc" in train_cfg
                    or "lambda_dec" in train_cfg
                    or "scale_loss_eps" in train_cfg
                ):
                    sl_from_training = {
                        "use_scale_loss": bool(train_cfg.get("use_scale_loss", False)),
                        "lambda_enc": float(train_cfg.get("lambda_enc", 0.1)),
                        "lambda_dec": float(train_cfg.get("lambda_dec", 0.1)),
                        "eps": float(train_cfg.get("scale_loss_eps", 1e-8)),
                    }
            if isinstance(sl_from_training, dict) and "scale_loss" not in model_config:
                model_config["scale_loss"] = copy.deepcopy(sl_from_training)
        model_type = resolve_model_type(model_config)

        if model_type == "pilotwimae":
            model = PilotWiMAE(config=model_config, device=self.device)
        elif model_type == "temporalenc_beam":
            bp = self.config.get("task", {}).get("beam_prediction")
            if not isinstance(bp, dict):
                raise KeyError(
                    "task.beam_prediction is required for model.type temporalenc_beam "
                    "(n_h, n_v, o_h, o_v, optional u_h, u_v, ...)."
                )
            num_classes = upa_2d_dft_num_beams(
                int(bp["n_h"]),
                int(bp["n_v"]),
                o_h=int(bp.get("o_h", 1)),
                o_v=int(bp.get("o_v", 1)),
                u_h=int(bp.get("u_h", 1)),
                u_v=int(bp.get("u_v", 1)),
            )
            model = PilotWiMAEBeamClassifier(
                config=model_config,
                num_classes=num_classes,
                device=self.device,
            )
        elif model_type == "temporalenc_los":
            los_cfg = self.config.get("task", {}).get("los")
            if not isinstance(los_cfg, dict):
                raise KeyError(
                    "task.los is required for model.type temporalenc_los "
                    "(optional key: num_classes, default 2)."
                )
            num_classes = int(los_cfg.get("num_classes", 2))
            if num_classes < 2:
                raise ValueError(f"task.los.num_classes must be >= 2, got {num_classes}")
            model = PilotWiMAEBeamClassifier(
                config=model_config,
                num_classes=num_classes,
                device=self.device,
            )
        elif model_type == "temporalenc_ce":
            ce_cfg = self.config.get("task", {}).get("channel_estimation")
            if not isinstance(ce_cfg, dict):
                raise KeyError(
                    "task.channel_estimation is required for model.type temporalenc_ce "
                    "(required key: pilot_pattern)."
                )
            if not isinstance(ce_cfg.get("pilot_pattern"), str):
                raise KeyError(
                    "task.channel_estimation.pilot_pattern is required for temporalenc_ce."
                )
            model = PilotWiMAE(config=model_config, device=self.device)
        else:
            raise NotImplementedError(
                f"Model type {model_type} not implemented. "
                f"Supported: 'pilotwimae', 'temporalenc_beam', "
                f"'temporalenc_los', 'temporalenc_ce'."
            )

        return model

    def _build_param_groups(self, model: nn.Module, opt_config: Dict[str, Any]):
        """Return parameter groups for the optimizer.

        When ``optimizer.encoder_lr`` is set, two groups are created:
        - *decoder* — parameters whose name starts with ``"decoder."``, using
          the main ``optimizer.lr``.
        - *encoder* — all other trainable parameters, using ``optimizer.encoder_lr``.

        This enables a soft-freeze: the encoder is updated at a much smaller
        learning rate than the decoder, preserving representation quality while
        allowing slight fine-tuning.

        When ``freeze_encoder: true`` is used instead, encoder parameters already
        have ``requires_grad=False`` so the encoder group will be empty and the
        optimizer only ever updates the decoder — no separate groups are needed,
        but they are constructed here for consistency and logged accordingly.

        When neither option is active, all trainable parameters are returned as a
        single flat iterable (original behaviour).
        """
        encoder_lr = opt_config.get("encoder_lr")
        if encoder_lr is None:
            return [p for p in model.parameters() if p.requires_grad]

        encoder_lr = float(encoder_lr)
        decoder_params = [
            p for n, p in model.named_parameters()
            if n.startswith("decoder.") and p.requires_grad
        ]
        encoder_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("decoder.") and p.requires_grad
        ]
        logger.info(
            "Parameter groups — decoder: %d tensors (lr=%.2e), encoder: %d tensors (lr=%.2e)",
            len(decoder_params),
            opt_config["lr"],
            len(encoder_params),
            encoder_lr,
        )
        return [
            {"params": decoder_params, "lr": opt_config["lr"]},
            {"params": encoder_params, "lr": encoder_lr},
        ]

    def setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        opt_config = self.config["training"]["optimizer"]
        opt_type = opt_config["type"].lower()
        params = self._build_param_groups(model, opt_config)

        if opt_type == "adam":
            return optim.Adam(
                params,
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.0),
                betas=tuple(opt_config.get("betas", (0.9, 0.999))),
            )
        elif opt_type == "adamw":
            return optim.AdamW(
                params,
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.0),
                betas=tuple(opt_config.get("betas", (0.9, 0.999))),
            )
        elif opt_type == "sgd":
            return optim.SGD(
                params,
                lr=opt_config["lr"],
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=opt_config.get("weight_decay", 0.0),
            )
        else:
            raise NotImplementedError(f"Optimizer type {opt_type} not implemented. Only 'adam', 'adamw', and 'sgd' are supported.")
 
    def setup_scheduler(
        self, optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        if "scheduler" not in self.config["training"]:
            return None

        sched_config = self.config["training"]["scheduler"]
        sched_type = sched_config["type"].lower()
        warmup_epochs = int(sched_config.get("warmup_epochs", 0))

        if sched_type == "cosine":
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, sched_config["T_max"] - warmup_epochs),
                eta_min=sched_config.get("eta_min", 0.000003),
            )
        elif sched_type == "step":
            main_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config["step_size"],
                gamma=sched_config.get("gamma", 0.1),
            )
        elif sched_type == "exponential":
            main_scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=sched_config["gamma"]
            )
        else:
            raise NotImplementedError(
                f"Scheduler type '{sched_type}' not implemented. "
                "Supported types: 'cosine', 'step', 'exponential'."
            )

        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-4, total_iters=warmup_epochs,
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        return main_scheduler

    def setup_criterion(self) -> nn.Module:
        loss_type = self.config["training"].get("loss", "mse").lower()
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss()
        elif loss_type == "nmse":
            # Per-sample normalized MSE; intended for raw-target decoder
            # pretraining where samples have very different absolute power.
            eps = float(self.config["training"].get("nmse_eps", 1e-8))
            return PerSampleNMSE(eps=eps)
        elif loss_type in ("cross_entropy", "ce"):
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                f"Loss type {loss_type} not implemented. "
                f"Supported: 'mse', 'l1', 'huber', 'nmse', 'cross_entropy (or ce)'."
            )

    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        data_config = self.config["data"]
        training_config = self.config["training"]
        normalize = data_config.get("normalize")
        calculate_stats = data_config.get("calculate_statistics")
        debug_size = data_config.get("debug_size", None)  # use full dataset if debug size is not provided

        statistics = self._setup_statistics(data_config, normalize, calculate_stats)
        train_dataset, val_dataset = self._create_datasets(
            data_config, normalize, calculate_stats, statistics, debug_size
        )

        # Compute statistics from training data only when needed (normalize=True) and not pre-computed
        if normalize and calculate_stats and statistics is None:
            statistics = self._calculate_statistics(train_dataset, training_config)
            train_dataset, val_dataset = self._create_datasets(
                data_config, normalize, False, statistics, debug_size
            )

        train_loader, val_loader = self._create_dataloaders(
            train_dataset, val_dataset, training_config
        )
        return train_loader, val_loader

    def _create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        training_config: Dict,
        *,
        collate_fn_train=None,
        collate_fn_val=None,
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = create_efficient_dataloader(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn_train,
        )
        val_loader = create_efficient_dataloader(
            val_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_val,
        )
        logger.info("Train samples: %d", len(train_dataset))
        logger.info("Validation samples: %d", len(val_dataset))
        return train_loader, val_loader

    def _setup_statistics(
        self, data_config: Dict, normalize: bool, calculate_stats: bool
    ) -> Optional[Dict]:
        if normalize and not calculate_stats:
            statistics = data_config.get("statistics")
            if statistics:
                logger.info("Using pre-computed statistics: %s", statistics)
                return statistics
        return None

    def _create_datasets(
        self,
        data_config: Dict,
        normalize: bool,
        calculate_stats: bool,
        statistics: Optional[Dict],
        debug_size: Optional[int],
    ) -> Tuple[Dataset, Dataset]:
        data_root = Path(data_config["data_dir"])
        npz_files = [str(p) for p in data_root.rglob("*.npz")]  # recursively find all NPZ files in the data directory

        if not npz_files:
            raise ValueError(f"No NPZ files found in {data_config['data_dir']}")

        dataset = OptimizedPreloadedDataset(
            npz_files=npz_files,
            statistics=statistics if (normalize and not calculate_stats) else None,
        )

        if debug_size is not None:
            dataset, _ = random_split(
                dataset,
                [debug_size, len(dataset) - debug_size],
                generator=torch.Generator().manual_seed(42),
            )

        val_split = data_config.get("val_split")
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, val_dataset

    def _calculate_statistics(
        self, train_dataset: Dataset, training_config: Dict
    ) -> Dict:
        logger.info("Computing statistics from training dataset...")
        batch_size = int(training_config["batch_size"])
        num_workers = int(training_config.get("num_workers", 0))
        temp_loader = create_efficient_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=max(0, num_workers),
            shuffle=False,
        )
        statistics = calculate_mean_power(temp_loader)
        logger.info("Calculated statistics: %s", statistics)
        return statistics

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        raise NotImplementedError

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        raise NotImplementedError

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "epoch": self.current_epoch,
            "last_global_epoch": self.current_epoch,
            "global_step": self.global_step,
            "patience_counter": self.patience_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.use_mixed_precision else None
            ),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_best:
            checkpoint_path = self.log_dir / "best_checkpoint.pt"
        else:
            checkpoint_path = self.log_dir / "last_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
        self, checkpoint_path: str, model_only: bool = True, strict: bool = True
    ):
        checkpoint = _safe_torch_load(
            str(Path(checkpoint_path).expanduser()), map_location=self.device
        )

        if strict:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            missing, unexpected = self.model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            if missing:
                logger.warning("Missing keys (randomly initialized): %s", list(missing))
            if unexpected:
                logger.warning("Unexpected keys (ignored): %s", list(unexpected))
            if not missing and not unexpected:
                logger.info("All model weights loaded successfully")

        if not model_only:
            if "optimizer_state_dict" in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if (
                "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"]
                and self.scheduler
            ):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            last_ge = checkpoint.get("last_global_epoch", checkpoint.get("epoch", -1))
            self.current_epoch = int(last_ge)
            self._tensorboard_epoch_offset = int(last_ge) + 1
            self.global_step = checkpoint.get("global_step", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.patience_counter = int(checkpoint.get("patience_counter", 0))
            if self.use_mixed_precision and checkpoint.get("scaler_state_dict"):
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception as exc:
                    logger.warning("Could not load GradScaler state: %s", exc)
            logger.info(
                "Loaded full training state; last_global_epoch=%s, next TB epoch=%s",
                last_ge,
                self._tensorboard_epoch_offset,
            )
        else:
            logger.info("Loaded model weights only (training state not restored)")

    def train(self):
        train_loader, val_loader = self.setup_dataloaders()

        epochs_this_run = int(self.config["training"]["epochs"])
        patience = self.config["training"]["patience"]
        epoch_offset = int(self._tensorboard_epoch_offset)

        logger.info(
            "Training for %d epochs (TensorBoard x-axis: %d .. %d)",
            epochs_this_run,
            epoch_offset,
            epoch_offset + epochs_this_run - 1,
        )
        logger.info("Model: %s", resolve_model_type(self.config["model"]))
        logger.info("Device: %s", self.device)
        logger.info("Log directory: %s", self.log_dir)

        for local_i in range(epochs_this_run):
            global_epoch = epoch_offset + local_i
            self.current_epoch = global_epoch

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.log_metrics(train_metrics, val_metrics, global_epoch)

            if self.scheduler:
                self.scheduler.step()

            min_delta = self.config["training"]["min_delta"]
            is_best = val_metrics["val_loss"] < (self.best_val_loss - min_delta)

            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            save_every_n = self.config["training"].get("save_checkpoint_every_n", 10)
            save_best = self.config["training"].get("save_best", False)
            # When save_best is True, still run periodic last_checkpoint saves; otherwise a
            # resumed run that never beats the loaded best_val_loss would write no .pt files.
            if save_best and is_best:
                self.save_checkpoint(is_best=True)
            if global_epoch % save_every_n == 0:
                self.save_checkpoint(is_best=False)

            if self.patience_counter >= patience:
                logger.info(
                    "Early stopping after %d epochs without improvement", patience
                )
                break

        logger.info("Training completed!")

    def log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
    ):
        logger.info("Epoch %d: %s %s", epoch, train_metrics, val_metrics)

        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
            lr = self.optimizer.param_groups[0].get("lr")
            if lr is not None:
                self.writer.add_scalar("optim/lr", float(lr), epoch)

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: Optional[torch.device] = None,
    ) -> "BaseTrainer":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config, device=device)
