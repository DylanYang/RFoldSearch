"""Assemble the 49-D per-residue structural descriptor.

The descriptor encodes the three forces that determine RNA tertiary
structure (spatial proximity, base pairing, base stacking) as
relative-pose FRAME blocks, plus a low-dimensional SCALAR context
block:

    [FRAME]   0..10    g_nn (k=1)     nearest spatial neighbour       (10D)
    [FRAME]  10..20    g_nn (k=2)     second-nearest spatial neighbour (10D)
    [FRAME]  20..30    g_bp           base-pair partner xform          (10D)
    [FRAME]  30..31    m_bp           base-pair availability mask       (1D)
    [FRAME]  31..41    g_st           stacking partner xform           (10D)
    [FRAME]  41..42    m_st           stacking availability mask        (1D)
    [SCALAR] 42..49    g_ctx          seq-sep × 3 + cis/trans + density (7D)

Total width: 49.
"""
from __future__ import annotations

import torch

from .frame import rel_features_for_pairs
from .neighbours import find_top_k_nn_with_virtual_center
from .scalar import contact_density, pair_orient_features, seq_sep_feature

DESCRIPTOR_WIDTH = 49
K_NEIGHBORS = 2


def compute_descriptor(
    coords: torch.Tensor,
    mask: torch.Tensor,
    pair_idx: torch.Tensor | None = None,
    stack_idx: torch.Tensor | None = None,
    pair_orient: torch.Tensor | None = None,
    min_seq_sep: int = 4,
) -> tuple[torch.Tensor, dict]:
    """Compute the 49-D descriptor for each residue.

    Args:
        coords: (B, L, 3, 3) backbone [C3', C4', O4'] coordinates.
        mask:   (B, L) residue validity.
        pair_idx, stack_idx: (B, L) partner indices, -1 if none.
        pair_orient: (B, L), -1 / 1 / 2 = none / cis / trans, used
            inside g_ctx; safe to leave None.
        min_seq_sep: minimum sequence separation for "non-sequential"
            neighbours (default 4 nt).

    Returns:
        descriptor: (B, L, 49) tensor.
        info: dict of named sub-blocks (for loss decomposition / debug).
    """
    B, L = mask.shape
    device = coords.device

    if pair_idx is not None and pair_idx.ndim == 3:
        pair_idx = pair_idx[..., 0]
    if stack_idx is not None and stack_idx.ndim == 3:
        stack_idx = stack_idx[..., 0]
    if pair_idx is None:
        pair_idx = torch.full((B, L), -1, dtype=torch.long, device=device)
    if stack_idx is None:
        stack_idx = torch.full((B, L), -1, dtype=torch.long, device=device)

    # ── FRAME · top-2 spatial neighbours (virtual-centre kNN) ──────
    nn_idx_top2 = find_top_k_nn_with_virtual_center(
        coords,
        mask,
        k=K_NEIGHBORS,
        min_seq_sep=min_seq_sep,
    )
    nn1_idx = nn_idx_top2[..., 0]
    nn2_idx = nn_idx_top2[..., 1]
    g_nn1, _ = rel_features_for_pairs(coords, mask, nn1_idx)
    g_nn2, _ = rel_features_for_pairs(coords, mask, nn2_idx)

    # ── FRAME · base-pair and stacking partners ────────────────────
    g_bp, bp_valid = rel_features_for_pairs(coords, mask, pair_idx)
    m_bp = bp_valid.float().unsqueeze(-1)
    g_st, st_valid = rel_features_for_pairs(coords, mask, stack_idx)
    m_st = st_valid.float().unsqueeze(-1)

    # ── SCALAR · 7-D context block ─────────────────────────────────
    g_ctx = torch.cat(
        [
            seq_sep_feature(nn1_idx, mask),
            seq_sep_feature(pair_idx, mask),
            seq_sep_feature(stack_idx, mask),
            pair_orient_features(pair_orient, bp_valid),
            contact_density(coords, mask, min_seq_sep=min_seq_sep),
        ],
        dim=-1,
    )

    # ── Concatenate into the canonical 49-D layout ─────────────────
    descriptor = torch.cat([g_nn1, g_nn2, g_bp, m_bp, g_st, m_st, g_ctx], dim=-1)
    descriptor = descriptor * mask.unsqueeze(-1).float()

    info = {
        "g_nn1": g_nn1,
        "g_nn2": g_nn2,
        "g_bp": g_bp,
        "m_bp": m_bp,
        "g_st": g_st,
        "m_st": m_st,
        "g_ctx": g_ctx,
        "nn1_idx": nn1_idx,
        "nn2_idx": nn2_idx,
        "pair_idx": pair_idx,
        "stack_idx": stack_idx,
        "valid_mask": mask,
    }
    return descriptor, info


def parse_descriptor(f: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the 49-D descriptor into named slots.

    The layout is fixed (see module docstring); only width 49 is
    accepted.
    """
    D = f.shape[-1]
    if D != DESCRIPTOR_WIDTH:
        raise ValueError(
            f"Expected descriptor width {DESCRIPTOR_WIDTH}, got {D}. "
            "The canonical descriptor was simplified to a fixed 49-D "
            "layout (g_nn k=1, g_nn k=2, g_bp+m_bp, g_st+m_st, g_ctx)."
        )
    return {
        "g_nn1": f[..., 0:10],
        "g_nn2": f[..., 10:20],
        "g_bp": f[..., 20:30],
        "m_bp": f[..., 30:31],
        "g_st": f[..., 31:41],
        "m_st": f[..., 41:42],
        "g_ctx": f[..., 42:49],
    }
