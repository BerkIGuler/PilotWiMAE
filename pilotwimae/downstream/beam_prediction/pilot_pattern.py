"""Parse pilot strings and build factorized flat_keep indices (kNN pilot_visible)."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import torch

from pilotwimae.models.modules.masking import factorized_flat_keep_from_t_s


def parse_pilot_pattern(s: str) -> Tuple[List[int], List[int]]:
    """
    Format: ``t:<t0>,<t1>,...;f:<f0>,<f1>,...`` (spaces allowed).

    ``t`` entries are time grid indices; ``f`` entries are frequency grid indices
    ``if_``. For each ``f``, all spatial indices ``is`` in ``[0, ns)`` are included
    when building spatial keep slots (``Sk = len(f_uniq) * ns``).
    """
    text = (s or "").strip()
    if not text:
        raise ValueError("pilot_pattern is empty")

    parts = [p.strip() for p in text.split(";")]
    if len(parts) != 2:
        raise ValueError(
            "pilot_pattern must have exactly one ';' separating t:... and f:... "
            f"(got {len(parts) - 1} semicolons)"
        )
    t_seg, f_seg = parts
    tl = t_seg.lower()
    fl = f_seg.lower()
    if not tl.startswith("t:"):
        raise ValueError("pilot_pattern first segment must start with 't:'")
    if not fl.startswith("f:"):
        raise ValueError("pilot_pattern second segment must start with 'f:'")

    t_list = _parse_int_list(t_seg[2:].strip())
    f_list = _parse_int_list(f_seg[2:].strip())
    if not t_list:
        raise ValueError("pilot_pattern t: list must contain at least one index")
    if not f_list:
        raise ValueError("pilot_pattern f: list must contain at least one index")
    return t_list, f_list


def _parse_int_list(part: str) -> List[int]:
    if not part:
        return []
    out: List[int] = []
    for chunk in part.split(","):
        c = chunk.strip()
        if not c:
            continue
        if not re.fullmatch(r"-?\d+", c):
            raise ValueError(f"Invalid integer in pilot_pattern: {c!r}")
        out.append(int(c))
    return out


def pilot_visible_flat_keep(
    nt: int,
    ns: int,
    nf: int,
    t_list: Sequence[int],
    f_list: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Deterministic ``(1, Tk*Sk)`` flat indices matching :class:`FactorizedMaskGenerator`
    geometry (same as pretraining).

    Spatial slots: for each distinct ``if_`` in ``f_list`` (sorted), for each
    ``is`` in ``range(ns)``, append ``is * nf + if_``.
    """
    if nt < 1 or ns < 1 or nf < 1:
        raise ValueError(f"Invalid grid_dims nt={nt}, ns={ns}, nf={nf}")

    t_uniq = sorted(set(int(x) for x in t_list))
    f_uniq = sorted(set(int(x) for x in f_list))

    for it in t_uniq:
        if it < 0 or it >= nt:
            raise ValueError(f"Time index {it} out of range for nt={nt}")
    for if_ in f_uniq:
        if if_ < 0 or if_ >= nf:
            raise ValueError(f"Frequency index {if_} out of range for nf={nf}")

    spatial: List[int] = []
    for if_ in f_uniq:
        for is_ in range(ns):
            spatial.append(is_ * nf + if_)

    ns_nf = ns * nf
    for sidx in spatial:
        if sidx < 0 or sidx >= ns_nf:
            raise ValueError(f"Derived spatial index {sidx} out of range for ns_nf={ns_nf}")

    ids_t = torch.tensor([t_uniq], device=device, dtype=torch.long)
    ids_s = torch.tensor([spatial], device=device, dtype=torch.long)
    return factorized_flat_keep_from_t_s(ids_t, ids_s, ns_nf)
