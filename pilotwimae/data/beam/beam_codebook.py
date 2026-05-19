"""
Beam codebook generation utilities for downstream tasks.
"""

import math
from typing import Any, Literal, Mapping, Tuple, Union

import torch


AntennaOrder = Literal["hv", "vh"]


def upa_axis_dft_codewords(n_elems: int, o: int, u: int) -> int:
    """
    Number of DFT angular bins along one UPA dimension.

    Oversampled regime (``u == 1``): ``K = o * N``.
    Undersampled regime (``u > 1``): ``K = N / u`` with ``o == 1`` and ``u | N``.
    ``o > 1`` and ``u > 1`` on the same axis are not permitted.
    """
    if n_elems <= 0:
        raise ValueError("n_elems must be positive.")
    if o <= 0 or u <= 0:
        raise ValueError("o and u must be positive integers.")
    if o > 1 and u > 1:
        raise ValueError(
            "Cannot combine oversampling and undersampling on the same axis: "
            "use oversampling only (u=1) or undersampling only (o=1)."
        )
    if u > 1:
        if o != 1:
            raise ValueError("When undersampling (u > 1), oversampling factor o must be 1.")
        if n_elems % u != 0:
            raise ValueError(
                f"Undersampling factor u={u} must divide antenna count n_elems={n_elems}."
            )
        return n_elems // u
    return o * n_elems


def upa_2d_dft_num_beams(
    n_h: int,
    n_v: int,
    *,
    o_h: int = 1,
    o_v: int = 1,
    u_h: int = 1,
    u_v: int = 1,
) -> int:
    """Total beams ``M = K_h * K_v`` for ``generate_upa_2d_dft_codebook`` with the same args."""
    k_h = upa_axis_dft_codewords(n_h, o_h, u_h)
    k_v = upa_axis_dft_codewords(n_v, o_v, u_v)
    return k_h * k_v


def num_beams_from_saved_codebook(cb: Mapping[str, Any]) -> int:
    """
    Recover ``M`` from a persisted ``codebook`` object (eval JSON ``codebook`` or similar).

    Missing ``u_h`` / ``u_v`` default to ``1`` (oversampled-only legacy records).
    """
    return upa_2d_dft_num_beams(
        int(cb["n_h"]),
        int(cb["n_v"]),
        o_h=int(cb.get("o_h", 1)),
        o_v=int(cb.get("o_v", 1)),
        u_h=int(cb.get("u_h", 1)),
        u_v=int(cb.get("u_v", 1)),
    )


def flatten_beam_index(m_h: int, m_v: int, n_h_total: int) -> int:
    """
    Flatten 2D beam index (horizontal, vertical) into a 1D index.

    ``n_h_total`` is ``K_h`` (horizontal DFT grid size), not the physical element count ``N_h``.
    """
    return m_v * n_h_total + m_h


def unflatten_beam_index(m: int, n_h_total: int) -> Tuple[int, int]:
    """Recover (m_h, m_v) from flattened index; ``n_h_total`` is ``K_h`` (see ``flatten_beam_index``)."""
    m_v = m // n_h_total
    m_h = m % n_h_total
    return m_h, m_v


def _steering_vector(num_ant: int, num_codewords: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(num_ant, device=device, dtype=torch.float32)
    m = torch.arange(num_codewords, device=device, dtype=torch.float32)
    phase = 2.0 * math.pi * m / float(num_codewords)
    exponent = idx[:, None] * phase[None, :]
    return torch.exp(1j * exponent).to(dtype=dtype) / math.sqrt(num_ant)


def generate_upa_2d_dft_codebook(
    n_h: int,
    n_v: int,
    *,
    o_h: int = 1,
    o_v: int = 1,
    u_h: int = 1,
    u_v: int = 1,
    antenna_order: AntennaOrder = "hv",
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Generate oversampled or undersampled 2D DFT UPA codebook (Kronecker DFT grids).

    Per horizontal / vertical dimension: oversampling uses ``K = o * N`` (``u`` = 1);
    undersampling uses ``K = N / u`` (``o`` = 1). Mixing ``o > 1`` and ``u > 1`` on the
    same axis is rejected; independent axes may use different regimes.

    Args:
        n_h: Number of horizontal antenna elements (columns).
        n_v: Number of vertical antenna elements (rows).
        o_h: Horizontal oversampling factor (effective when ``u_h == 1``).
        o_v: Vertical oversampling factor (effective when ``u_v == 1``).
        u_h: Horizontal undersampling factor; ``u_h`` must divide ``n_h``.
        u_v: Vertical undersampling factor; ``u_v`` must divide ``n_v``.
        antenna_order:
            - "hv": antenna index layout (v * N_h + h), equivalent to a_v ⊗ a_h.
            - "vh": antenna index layout (h * N_v + v), equivalent to a_h ⊗ a_v.
        device: Output tensor device.
        dtype: Complex dtype for the codebook.

    Returns:
        Codebook tensor with shape (M, N_h * N_v), ``M = K_h * K_v``.
    """
    if n_h <= 0 or n_v <= 0:
        raise ValueError("n_h and n_v must be positive.")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be a complex dtype.")

    device = torch.device(device)
    k_h = upa_axis_dft_codewords(n_h, o_h, u_h)
    k_v = upa_axis_dft_codewords(n_v, o_v, u_v)

    a_h = _steering_vector(
        num_ant=n_h,
        num_codewords=k_h,
        device=device,
        dtype=dtype,
    )
    a_v = _steering_vector(
        num_ant=n_v,
        num_codewords=k_v,
        device=device,
        dtype=dtype,
    )

    codewords = []
    for m_v in range(k_v):
        for m_h in range(k_h):
            if antenna_order == "hv":
                w = torch.kron(a_v[:, m_v], a_h[:, m_h])
            elif antenna_order == "vh":
                w = torch.kron(a_h[:, m_h], a_v[:, m_v])
            else:
                raise ValueError("antenna_order must be one of {'hv', 'vh'}.")
            codewords.append(w)

    return torch.stack(codewords, dim=0).to(dtype=dtype)
