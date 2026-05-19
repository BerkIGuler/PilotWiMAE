"""
Trainer for supervised beam prediction (PilotWiMAEBeamClassifier).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pilotwimae.data import BeamLabelDatasetWrapper

from .trainer import BaseTrainer

logger = logging.getLogger(__name__)


class BeamClassifierTrainer(BaseTrainer):
    """Cross-entropy training with (channel, beam_label) batches."""

    # override the base class method
    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        bp = self.config.get("task", {}).get("beam_prediction")
        if not isinstance(bp, dict):
            raise KeyError("task.beam_prediction is required for BeamClassifierTrainer.")

        data_config = self.config["data"]
        training_config = self.config["training"]
        normalize = data_config.get("normalize")
        calculate_stats = data_config.get("calculate_statistics")
        debug_size = data_config.get("debug_size", None)

        statistics = self._setup_statistics(data_config, normalize, calculate_stats)
        train_dataset, val_dataset = self._create_datasets(
            data_config, normalize, calculate_stats, statistics, debug_size
        )

        if normalize and calculate_stats and statistics is None:
            statistics = self._calculate_statistics(train_dataset, training_config)
            train_dataset, val_dataset = self._create_datasets(
                data_config, normalize, False, statistics, debug_size
            )

        train_wrapped = BeamLabelDatasetWrapper(
            train_dataset,
            n_h=int(bp["n_h"]),
            n_v=int(bp["n_v"]),
            o_h=int(bp.get("o_h", 1)),
            o_v=int(bp.get("o_v", 1)),
            u_h=int(bp.get("u_h", 1)),
            u_v=int(bp.get("u_v", 1)),
            antenna_order=bp.get("antenna_order", "hv"),
            label_mode=bp.get("label_mode", "snapshot"),
            return_format="class_index",
            top_k=int(bp.get("top_k", 1)),
        )
        val_wrapped = BeamLabelDatasetWrapper(
            val_dataset,
            n_h=int(bp["n_h"]),
            n_v=int(bp["n_v"]),
            o_h=int(bp.get("o_h", 1)),
            o_v=int(bp.get("o_v", 1)),
            u_h=int(bp.get("u_h", 1)),
            u_v=int(bp.get("u_v", 1)),
            antenna_order=bp.get("antenna_order", "hv"),
            label_mode=bp.get("label_mode", "snapshot"),
            return_format="class_index",
            top_k=int(bp.get("top_k", 1)),
        )

        return self._create_dataloaders(
            train_wrapped,
            val_wrapped,
            training_config,
            collate_fn_train=train_wrapped.make_collate_fn(),
            collate_fn_val=val_wrapped.make_collate_fn(),
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
