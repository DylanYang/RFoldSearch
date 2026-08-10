"""SCALAR features — low-dimensional pairwise / density summaries.

The 7-D g_ctx block is a concatenation of three sub-blocks:
    * seq_sep × {nn, bp, st}   (3D) — signed log-scaled sequence
      separations to spatial-neighbour / base-pair / stacking partners
    * pair_orient              (2D) — cis / trans flags
      (Leontis & Westhof 2001), zeroed when no base pair exists
    * contact_density          (2D) — clipped non-sequential C4'
      neighbour counts within 8 and 12 Angstroms
"""
from __future__ import annotations

import torch


def seq_sep_feature(tgt_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Signed log-scaled sequence separation to a partner. (B, L, 1)."""
    B, L = mask.shape
    device = mask.device
    idx = torch.arange(L, device=device).view(1, L).expand(B, L)
    in_range = (tgt_idx >= 0) & (tgt_idx < L)
    safe_idx = tgt_idx.clamp(min=0, max=max(L - 1, 0))
    tgt_valid = torch.gather(mask, 1, safe_idx) & in_range
    valid = mask & tgt_valid
    sep = torch.zeros(B, L, device=device, dtype=torch.float32)
    raw = (tgt_idx - idx).float()
    denom = torch.log1p(torch.tensor(float(max(1, L - 1)), device=device))
    sep[valid] = torch.sign(raw[valid]) * torch.log1p(raw[valid].abs()) / denom
    return sep.unsqueeze(-1)


def pair_orient_features(
    pair_orient: torch.Tensor | None,
    bp_valid: torch.Tensor,
) -> torch.Tensor:
    """One-hot cis (=1) / trans (=2) flags for base-pair orientation.

    bp_valid: (B, L) gates the output so that non-paired residues are 0.
    Returns: (B, L, 2).
    """
    if pair_orient is None:
        return torch.zeros(*bp_valid.shape, 2, device=bp_valid.device)
    if pair_orient.ndim == 3:
        pair_orient = pair_orient[..., 0]
    pair_orient = pair_orient.to(bp_valid.device)
    cis = ((pair_orient == 1) & bp_valid).float()
    trans = ((pair_orient == 2) & bp_valid).float()
    return torch.stack([cis, trans], dim=-1)


def contact_density(
    coords: torch.Tensor,
    mask: torch.Tensor,
    min_seq_sep: int = 4,
    radii: tuple[float, float] = (8.0, 12.0),
    norm_count: float = 12.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Non-sequential C4' contact counts at the requested radii, clipped to [0, 1].

    Returns (B, L, len(radii))."""
    C4p = coords[..., 1, :]
    B, L, _ = C4p.shape
    device = C4p.device
    out = torch.zeros(B, L, len(radii), device=device, dtype=coords.dtype)

    for b in range(B):
        valid_pos = mask[b].bool().nonzero(as_tuple=True)[0]
        if valid_pos.numel() < 2:
            continue
        pts = C4p[b, valid_pos]
        n = pts.shape[0]
        counts_by_radius = [torch.zeros(n, device=device, dtype=torch.float32) for _ in radii]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = pts[start:end]
            dist2 = (chunk[:, None, :] - pts[None, :, :]).pow(2).sum(dim=-1)
            seq_sep = (valid_pos[start:end, None] - valid_pos[None, :]).abs()
            allowed = seq_sep >= min_seq_sep
            for r_i, radius in enumerate(radii):
                counts_by_radius[r_i][start:end] = (
                    ((dist2 <= radius * radius) & allowed).sum(dim=1).float()
                )
        for r_i, counts in enumerate(counts_by_radius):
            out[b, valid_pos, r_i] = (counts / norm_count).clamp(max=1.0)
    return out
