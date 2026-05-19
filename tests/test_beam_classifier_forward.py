import torch

from pilotwimae.models.beam_classifier import PilotWiMAEBeamClassifier


def _tiny_model_config(encoder_type: str):
    return {
        "encoder_type": encoder_type,
        "input_shape": [4, 8, 8],
        "patch_size": [1, 4, 4],
        "embedding": {"type": "linear"},
        "positional_encoding": {
            "encoder": {"type": "sinusoidal_concat"},
            "decoder": {"type": "sinusoidal_concat"},
        },
        "masking": {"strategy": "random", "mask_ratio": 0.5},
        "norm_first": False,
        "encoder_dim": 32,
        "encoder_layers": 1 if encoder_type == "standard" else 1,
        "encoder_nhead": 4,
        "decoder_layers": 1,
        "decoder_nhead": 4,
        "ffn_factor": 2,
    }


def test_beam_classifier_forward_standard():
    cfg = _tiny_model_config("standard")
    num_classes = 32
    m = PilotWiMAEBeamClassifier(cfg, num_classes=num_classes, device=torch.device("cpu"))
    B, T, S, F = 2, 4, 8, 8
    x = torch.randn(B, T, S, F, dtype=torch.complex64)
    logits = m(x)
    assert logits.shape == (B, num_classes)


def test_beam_classifier_forward_factorized():
    cfg = _tiny_model_config("factorized")
    cfg["masking"] = {
        "strategy": "factorized",
        "num_time_keep": 2,
        "spatial_mask_ratio": 0.5,
    }
    num_classes = 16
    m = PilotWiMAEBeamClassifier(cfg, num_classes=num_classes, device=torch.device("cpu"))
    B, T, S, F = 2, 4, 8, 8
    x = torch.randn(B, T, S, F, dtype=torch.complex64)
    logits = m(x)
    assert logits.shape == (B, num_classes)
