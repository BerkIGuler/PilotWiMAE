"""Entry point for MAE / masked pretraining (PilotWiMAETrainer)."""

import argparse

import torch
import yaml

from pilotwimae import PilotWiMAETrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train PilotWiMAE masked autoencoder (MAE)",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to training config YAML (model.type: pilotwimae)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda:0). Overrides config if provided",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("model", {}).get("type") != "pilotwimae":
        raise NotImplementedError(
            "Config model.type must be 'pilotwimae'. "
            f"Got: {config.get('model', {}).get('type')}"
        )

    device = torch.device(args.device) if args.device is not None else None
    trainer = PilotWiMAETrainer(config, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
