"""FRAME features — residue-to-partner 6D rigid transforms.

The same primitive (`rel_features_for_pairs`) is used for:
  * g_nn (spatial nearest neighbour, with multi-k variants)
  * g_bp (base-pair partner)
  * g_st (stacking partner)

Each output is 10D = 1D distance + 3D local direction + 6D rotation
(first two columns of `T_src^{-1} . T_tgt`). For g_bp / g_st a binary
mask is returned alongside, since not every residue has a partner.
"""
from __future__ import annotations

import torch

from ._utils import build_local_frames, compute_rel_transform_10d


def rel_features_for_pairs(
    coords: torch.Tensor,
    mask: torch.Tensor,
    tgt_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """10D rigid transform for each residue to its target partner.

    Args:
        coords: (B, L, 3, 3) [C3', C4', O4'] coordinates.
        mask: (B, L) residue validity.
        tgt_idx: (B, L) index of the partner, -1 if none.
    Returns:
        features: (B, L, 10) — zeros where no partner exists.
        valid: (B, L) boolean — True where the partner is valid.
    """
    B, L = mask.shape
    device = coords.device

    in_range = (tgt_idx >= 0) & (tgt_idx < L)
    safe_idx = tgt_idx.clamp(min=0, max=max(L - 1, 0))
    tgt_valid = torch.gather(mask, 1, safe_idx) & in_range
    valid = mask & tgt_valid
    features = torch.zeros(B, L, 10, device=device)

    rot, origin = build_local_frames(coords)

    for b in range(B):
        valid_pos = valid[b].nonzero(as_tuple=True)[0]
        for idx in valid_pos:
            i = idx.item()
            j = tgt_idx[b, i].item()
            features[b, i] = compute_rel_transform_10d(
                rot[b, i],
                origin[b, i],
                rot[b, j],
                origin[b, j],
            )
    return features, valid
