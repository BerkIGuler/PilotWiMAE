#!/usr/bin/env python3
"""Train supervised LoS vs. nLoS classifier (PilotWiMAE encoder + linear head)."""

import argparse

import torch
import yaml

from pilotwimae.training import LosClassifierTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train PilotWiMAE LoS / nLoS binary classifier"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to training config YAML (model.type: temporalenc_los)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cuda:0 (overrides config training.device)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("model", {}).get("type") != "temporalenc_los":
        raise NotImplementedError(
            f"Model type {config.get('model', {}).get('type')} not implemented. "
            f"Supported: 'temporalenc_los'."
        )

    device = torch.device(args.device) if args.device is not None else None
    trainer = LosClassifierTrainer(config, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
