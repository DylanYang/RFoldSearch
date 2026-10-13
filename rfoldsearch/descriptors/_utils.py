"""Low-level geometric primitives shared by all descriptor families."""
from __future__ import annotations

import torch


def build_local_frames(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-residue local coordinate frames from (C3', C4', O4').

    Uses Gram-Schmidt on the C3'->C4' and C4'->O4' vectors. C4' is the
    frame origin; (e1, e2, e3) are the local x, y, z axes.

    Args:
        coords: (..., 3, 3) atom positions [C3', C4', O4']
    Returns:
        rot: (..., 3, 3) rotation whose columns are (e1, e2, e3)
        origin: (..., 3) C4' position
    """
    C3p = coords[..., 0, :]
    C4p = coords[..., 1, :]
    O4p = coords[..., 2, :]

    e1 = C4p - C3p
    e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-10)

    v = O4p - C4p
    proj = (v * e1).sum(dim=-1, keepdim=True) * e1
    e2 = v - proj
    e2 = e2 / (e2.norm(dim=-1, keepdim=True) + 1e-10)

    e3 = torch.cross(e1, e2, dim=-1)
    rot = torch.stack([e1, e2, e3], dim=-1)
    return rot, C4p


def compute_rel_transform_10d(
    rot_src: torch.Tensor,
    origin_src: torch.Tensor,
    rot_tgt: torch.Tensor,
    origin_tgt: torch.Tensor,
) -> torch.Tensor:
    """10D residue-to-partner rigid transform.

    Layout: [distance (1D), local direction (3D), 6D rotation].

    The 6D rotation is the first two columns of `T_src^{-1} . T_tgt`,
    following Zhou et al. 2019, On the Continuity of Rotation
    Representations in Neural Networks (CVPR).
    """
    delta = origin_tgt - origin_src
    local_delta = torch.einsum("...ij,...j->...i", rot_src.transpose(-1, -2), delta)

    dist = local_delta.norm(dim=-1, keepdim=True)
    direction = local_delta / (dist + 1e-10)

    R_rel = torch.einsum("...ij,...jk->...ik", rot_src.transpose(-1, -2), rot_tgt)
    rot6d = R_rel[..., :2].reshape(*R_rel.shape[:-2], 6)

    return torch.cat([dist, direction, rot6d], dim=-1)
