"""
Trainer for supervised LoS vs. nLoS binary classification.

Uses :class:`~pilotwimae.models.beam_classifier.PilotWiMAEBeamClassifier`
(``temporalenc_los``): **full patch grid** through the encoder, **mean-pooled**
embeddings, then a linear head — same forward path as supervised beam (no MAE masking).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from pilotwimae.data import OptimizedPreloadedDataset, create_efficient_dataloader
from pilotwimae.downstream.los.datasets import LosBinaryLabelDataset, load_los_binary_labels

from .trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LosClassifierTrainer(BaseTrainer):
    """Cross-entropy training with (channel, los_label) batches; labels from NPZ metadata."""

    def _create_los_datasets(
        self,
        data_config: Dict,
        normalize: bool,
        calculate_stats: bool,
        statistics: Optional[Dict],
        debug_size: Optional[int],
    ) -> Tuple[Dataset, Dataset]:
        """
        Same split logic as :meth:`BaseTrainer._create_datasets`, but:

        - NPZ paths are **sorted** so channel preload order matches ``load_los_binary_labels``.
        - Full dataset is wrapped with :class:`~pilotwimae.downstream.los.datasets.LosBinaryLabelDataset`
          **before** train/val split so labels stay aligned.
        """
        data_root = Path(data_config["data_dir"])
        npz_files = sorted(str(p) for p in data_root.rglob("*.npz"))

        if not npz_files:
            raise ValueError(f"No NPZ files found in {data_config['data_dir']}")

        base = OptimizedPreloadedDataset(
            npz_files=npz_files,
            statistics=statistics if (normalize and not calculate_stats) else None,
        )
        labels = load_los_binary_labels(npz_files)
        dataset: Dataset = LosBinaryLabelDataset(base, labels)

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

    def _calculate_statistics(self, train_dataset: Dataset, training_config: Dict) -> Dict:
        """Mean power over training channels only (dataloader yields ``(channel, label)`` batches)."""
        logger.info("Computing statistics from training dataset...")
        batch_size = int(training_config["batch_size"])
        num_workers = int(training_config.get("num_workers", 0))
        temp_loader = create_efficient_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=max(0, num_workers),
            shuffle=False,
        )
        power_sum = 0.0
        total_elements = 0
        for batch in tqdm(temp_loader, desc="Computing mean power"):
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            re = data.real.to(torch.float64)
            im = data.imag.to(torch.float64)
            power_sum += torch.sum(re * re + im * im).item()
            total_elements += data.numel()
        mean_power = power_sum / total_elements
        statistics = {"mean_power": mean_power}
        logger.info("Calculated statistics: %s", statistics)
        return statistics

    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        los = self.config.get("task", {}).get("los")
        if not isinstance(los, dict):
            raise KeyError("task.los is required for LosClassifierTrainer (may be empty {}).")

        data_config = self.config["data"]
        training_config = self.config["training"]
        normalize = data_config.get("normalize")
        calculate_stats = data_config.get("calculate_statistics")
        debug_size = data_config.get("debug_size", None)

        statistics = self._setup_statistics(data_config, normalize, calculate_stats)
        train_dataset, val_dataset = self._create_los_datasets(
            data_config, normalize, calculate_stats, statistics, debug_size
        )

        if normalize and calculate_stats and statistics is None:
            # Rebuild with stats computed on training channels only (subset indices).
            statistics = self._calculate_statistics(train_dataset, training_config)
            train_dataset, val_dataset = self._create_los_datasets(
                data_config, normalize, False, statistics, debug_size
            )

        return self._create_dataloaders(
            train_dataset,
            val_dataset,
            training_config,
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)

        for batch_idx, (data, targets) in enumerate(
            tqdm(train_loader, desc=f"Train epoch {self.current_epoch}")
        ):
            data = data.to(self.device)
            targets = targets.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                logits = self.model(data)
                loss = self.criterion(logits, targets)

            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            gradient_clip_val = self.config["training"].get("gradient_clip_val", 1.0)
            if gradient_clip_val > 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip_val
                )

            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())
            self.global_step += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "accuracy": correct / max(1, total),
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation"):
                data = data.to(self.device)
                targets = targets.to(self.device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    logits = self.model(data)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += int((pred == targets).sum().item())
                total += int(targets.numel())

        val_loss = total_loss / max(1, num_batches)
        val_acc = correct / max(1, total)
        return {"val_loss": val_loss, "val_accuracy": val_acc}

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
            "num_classes": int(getattr(self.model, "num_classes", 0)),
        }

        if is_best:
            checkpoint_path = self.log_dir / "best_checkpoint.pt"
        else:
            checkpoint_path = self.log_dir / "last_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
