import copy
from pathlib import Path

import pytest
from pilotwimae import PilotWiMAETrainer
from pilotwimae.training.utils import resolve_model_type


@pytest.fixture
def config() -> dict:
    """
    Minimal test config based on configs/default_training.yaml,
    with smaller batch/epochs and CPU device for fast tests.
    """
    return {
        "model": {
            "type": "pilotwimae",
            "input_shape": [14, 32, 32],
            "patch_size": [2, 4, 4],
            "embedding": {"type": "linear"},
            "positional_encoding": {"encoder": {"type": "sinusoidal_concat"}, "decoder": {"type": "sinusoidal_concat"}},
            "masking": {"strategy": "random", "mask_ratio": 0.9},
            "encoder_dim": 32,
            "encoder_layers": 2,
            "encoder_nhead": 4,
            "decoder_layers": 1,
            "decoder_nhead": 4,
        },
        "data": {
            # Point to a small directory or fake path; test will skip if empty.
            "data_dir": "/opt/shared/datasets/CSIGen/PilotWiMAE/boston_1",
            "normalize": True,
            "val_split": 0.1,
            "debug_size": None,
            "calculate_statistics": False,
            "statistics": {
                "mean_power": 1.4218511784624965e-11,
            },
        },
        "training": {
            "batch_size": 512,
            "epochs": 1,
            "num_workers": 0,
            # Use CPU in tests (CI/sandbox may not support CUDA initialization).
            "device": "cpu",
            "optimizer": {
                "type": "adam",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "cosine",
                "T_max": 1,
                "eta_min": 1e-5,
            },
            "loss": "mse",
            "patience": 1,
            "min_delta": 0.0,
            "gradient_clip_val": 0.0,
            "save_checkpoint_every_n": 1,
            "save_best": True,
        },
        "logging": {
            "log_dir": "runs_test",
            "tensorboard": False,
            "log_every_n_steps": 10,
            "exp_name": "pilotwimae_trainer_test",
        },
    }


def test_pilotwimae_trainer_init(config: dict) -> None:
    """Smoke test: trainer constructs with config and basic attributes exist."""
    trainer = PilotWiMAETrainer(config)
    assert trainer.model is not None
    assert trainer.optimizer is not None
    # Ensure model type is correct
    assert config["model"]["type"] == "pilotwimae"


def test_resolve_model_type_requires_model_type() -> None:
    with pytest.raises(KeyError):
        resolve_model_type({})
    assert resolve_model_type({"type": "pilotwimae"}) == "pilotwimae"


def test_pilotwimae_trainer_init_without_model_type_key(config: dict) -> None:
    """model.type is required; config without it should fail fast."""
    cfg = copy.deepcopy(config)
    cfg["model"].pop("type", None)
    with pytest.raises(KeyError):
        PilotWiMAETrainer(cfg)

def test_pilotwimae_trainer_train(config: dict) -> None:
    data_dir = Path(config["data"]["data_dir"])
    if not any(data_dir.rglob("*.npz")):
        pytest.skip(f"NPZ files not found under {data_dir}.")
    trainer = PilotWiMAETrainer(config)
    trainer.train()
    