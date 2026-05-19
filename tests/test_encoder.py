import pytest
import torch
from pilotwimae.models.modules import Encoder, FactorizedEncoder

@pytest.fixture
def input_embeddings(num_patches: int, d_model_encoder: int) -> torch.Tensor:
    batch_size = 128
    return torch.randn(batch_size, num_patches, d_model_encoder)

@pytest.fixture
def num_patches() -> int:
    patch_size = (2, 4, 4)
    input_shape = (14, 32, 32)
    num_patches = input_shape[0] // patch_size[0] * input_shape[1] // patch_size[1] * input_shape[2] // patch_size[2]
    return num_patches

@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")

@pytest.fixture
def d_model_encoder() -> int:
    return 64

@pytest.fixture
def nhead_encoder() -> int:
    return 8

@pytest.fixture
def num_layers_encoder() -> int:
    return 2

@pytest.fixture
def encoder(d_model_encoder: int, nhead_encoder: int, num_layers_encoder: int, device: torch.device) -> Encoder:
    return Encoder(d_model=d_model_encoder, nhead=nhead_encoder, num_layers=num_layers_encoder, device=device)

def test_encoder(
    encoder: Encoder,
    input_embeddings: torch.Tensor,
    num_patches: int,
    d_model_encoder: int,
) -> None:
    encoded_tokens = encoder(input_embeddings)

    # Shape
    assert encoded_tokens.shape == input_embeddings.shape

    # Dtype preserved (or at least float)
    assert encoded_tokens.dtype == input_embeddings.dtype

    # No NaN/Inf
    assert encoded_tokens.isfinite().all(), "Encoder output should be finite"

    # Encoder is not identity (output differs from input for random embeddings)
    assert not torch.allclose(encoded_tokens, input_embeddings), "Encoder should transform the input"


def _reference_factorized_forward_no_mixing(
    enc: FactorizedEncoder,
    x: torch.Tensor,
    Tk: int,
    Sk: int,
) -> torch.Tensor:
    """Explicit temporal→spatial stack without cross-dim mixing (legacy behavior)."""
    B, _, D = x.shape
    h = x.view(B, Tk, Sk, D)
    for t_layer, s_layer in zip(enc.temporal_layers, enc.spatial_layers):
        h_t = h.permute(0, 2, 1, 3).reshape(B * Sk, Tk, D)
        h_t = t_layer(h_t)
        h = h_t.reshape(B, Sk, Tk, D).permute(0, 2, 1, 3)
        h_s = h.reshape(B * Tk, Sk, D)
        h_s = s_layer(h_s)
        h = h_s.reshape(B, Tk, Sk, D)
    return h.reshape(B, Tk * Sk, D)


def test_factorized_encoder_mixing_off_matches_reference_forward() -> None:
    torch.manual_seed(0)
    d, nhead, nb = 24, 4, 2
    enc = FactorizedEncoder(
        d_model=d,
        nhead=nhead,
        num_blocks=nb,
        num_time_keep=2,
        num_spatial_keep=3,
        enable_cross_dim_mixing=False,
        device=torch.device("cpu"),
    )
    enc.eval()
    B, Tk, Sk = 2, 2, 3
    x = torch.randn(B, Tk * Sk, d)
    y = enc(x)
    y_ref = _reference_factorized_forward_no_mixing(enc, x, Tk, Sk)
    assert torch.allclose(y, y_ref, atol=0.0, rtol=0.0)
    assert not enc.enable_cross_dim_mixing


def test_factorized_encoder_mixing_on_variable_grid() -> None:
    d = 16
    enc = FactorizedEncoder(
        d_model=d,
        nhead=4,
        num_blocks=2,
        num_time_keep=2,
        num_spatial_keep=3,
        enable_cross_dim_mixing=True,
        device=torch.device("cpu"),
    )
    B, Tk, Sk = 2, 5, 7
    x = torch.randn(B, Tk * Sk, d)
    y = enc(x, time_steps=Tk, spatial_steps=Sk)
    assert y.shape == (B, Tk * Sk, d)
    assert y.isfinite().all()
    assert len(enc.spatial_mix_weights) == 2
    assert len(enc.temporal_mix_weights) == 2


def test_factorized_encoder_full_grid_kwarg() -> None:
    """Tube-sized init but forward with larger nt×(ns*nf) grid (inference checkpoint path)."""
    d = 32
    enc = FactorizedEncoder(
        d_model=d,
        nhead=4,
        num_blocks=2,
        num_time_keep=2,
        num_spatial_keep=3,
        device=torch.device("cpu"),
    )
    B, Tk, Sk = 2, 4, 6
    x = torch.randn(B, Tk * Sk, d)
    y = enc(x, time_steps=Tk, spatial_steps=Sk)
    assert y.shape == (B, Tk * Sk, d)
    assert y.isfinite().all()
    y_default = enc(torch.randn(B, 2 * 3, d))
    assert y_default.shape == (B, 6, d)
