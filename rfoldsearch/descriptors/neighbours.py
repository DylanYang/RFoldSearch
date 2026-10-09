"""Spatial-neighbour utilities for FRAME features.

`find_top_k_nn_with_virtual_center` returns the top-k nearest
non-sequential neighbours scored by a virtual atom anchored slightly
off the glycosidic axis (the convention used by Foldseek for
proteins). A `min_seq_sep` cutoff (default 4 nt) excludes trivially
adjacent neighbours along the chain.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F


def _vc_weights():
    """Virtual-centre (alpha, beta, gamma) weights.

    Read from environment variables RFS_VC_ALPHA/BETA/GAMMA if set
    (used by the virtual-centre weight ablation), else default 0.5 each.
    """

    def _g(name, default=0.5):
        v = os.environ.get(name)
        return float(v) if v is not None else default

    return _g("RFS_VC_ALPHA"), _g("RFS_VC_BETA"), _g("RFS_VC_GAMMA")


def compute_virtual_center_from_coords(
    coords: torch.Tensor,
    mask: torch.Tensor,
    c4p_idx: int = 1,
    base_idx: int | None = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Virtual center per residue.

    vc = C4' + alpha * u + beta * v + gamma * w
    where u points from C4' to a base-representative atom, v is the
    local backbone direction (C4'(i) -> C4'(i+1)), and w = u x v.

    Args:
        coords: (B, L, A, 3) atom positions; need at least C4' and a
            base proxy atom.
        mask: (B, L) valid-residue flag.
        c4p_idx: index of C4' in the atom dimension (default 1).
        base_idx: index of the base proxy atom. Auto-pick = 4 if
            A > 4 else 2.
    Returns:
        vc: (B, L, 3) — zero at padded positions.
    """
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(f"coords should have shape (B, L, A, 3), got {tuple(coords.shape)}")

    B, L, A, _ = coords.shape
    if c4p_idx >= A:
        raise ValueError(f"c4p_idx={c4p_idx} is out of range for A={A}")
    if base_idx is None:
        base_idx = 4 if A > 4 else 2
    if base_idx >= A:
        raise ValueError(f"base_idx={base_idx} is out of range for A={A}")

    C4p = coords[:, :, c4p_idx, :]
    B_i = coords[:, :, base_idx, :]

    u = F.normalize(B_i - C4p, dim=-1, eps=eps)

    if L == 1:
        v = torch.zeros_like(C4p)
    else:
        v_raw = torch.zeros_like(C4p)
        v_raw[:, :-1, :] = C4p[:, 1:, :] - C4p[:, :-1, :]
        v_raw[:, -1, :] = v_raw[:, -2, :]
        v = F.normalize(v_raw, dim=-1, eps=eps)

    w = F.normalize(torch.cross(u, v, dim=-1), dim=-1, eps=eps)
    vc = C4p + alpha * u + beta * v + gamma * w
    vc = vc.masked_fill(~mask[..., None].bool(), 0.0)
    return vc


def find_top_k_nn_with_virtual_center(
    coords: torch.Tensor,
    mask: torch.Tensor,
    k: int = 1,
    min_seq_sep: int = 4,
    c4p_idx: int = 1,
    base_idx: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> torch.Tensor:
    """Top-k nearest non-sequential virtual-center neighbours.

    If (alpha, beta, gamma) are left as None, the weights are taken
    from the RFS_VC_ALPHA/BETA/GAMMA environment variables (default
    0.5 each), enabling the virtual-centre weight ablation.

    Returns:
        (B, L, k) — slot m is the m-th nearest valid partner, or -1
        when fewer than m valid partners exist.
    """
    if alpha is None or beta is None or gamma is None:
        a, b, g = _vc_weights()
        alpha = a if alpha is None else alpha
        beta = b if beta is None else beta
        gamma = g if gamma is None else gamma
    vc = compute_virtual_center_from_coords(
        coords=coords,
        mask=mask,
        c4p_idx=c4p_idx,
        base_idx=base_idx,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    B, L, _ = vc.shape
    device = vc.device
    nn_idx = torch.full((B, L, k), -1, dtype=torch.long, device=device)

    for b in range(B):
        valid_pos = mask[b].bool().nonzero(as_tuple=True)[0]
        n = valid_pos.numel()
        if n < 2:
            continue
        pts = vc[b, valid_pos]
        seq_sep = (valid_pos[:, None] - valid_pos[None, :]).abs()
        allowed = seq_sep >= min_seq_sep
        dist2 = torch.cdist(pts, pts).pow(2).masked_fill(~allowed, float("inf"))
        k_eff = min(k, n)
        best_dist, best_j = dist2.topk(k_eff, dim=1, largest=False)
        for ki in range(k_eff):
            has_neighbor = torch.isfinite(best_dist[:, ki])
            nn_idx[b, valid_pos[has_neighbor], ki] = valid_pos[best_j[has_neighbor, ki]]
    return nn_idx
