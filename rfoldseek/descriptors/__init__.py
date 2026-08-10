"""Per-residue RNA structural descriptor.

Public API:
    compute_descriptor(coords, mask, pair_idx, stack_idx, ...) -> (tensor, info)
    parse_descriptor(tensor) -> dict[str, tensor]
    DESCRIPTOR_WIDTH                       = 49

The descriptor encodes the three forces that determine RNA tertiary
structure --- spatial proximity (top-2 nearest non-sequential
neighbours), hydrogen-bonded base pairing, and pi-stacking --- as
FRAME blocks of 10-D rigid-body transforms, plus a SCALAR context
block summarising sequence separations and local contact density.
See `compose.py` for the canonical 49-D layout.
"""

from .compose import (
    DESCRIPTOR_WIDTH,
    K_NEIGHBORS,
    compute_descriptor,
    parse_descriptor,
)
from .neighbours import (
    compute_virtual_center_from_coords,
    find_top_k_nn_with_virtual_center,
)

__all__ = [
    "DESCRIPTOR_WIDTH",
    "K_NEIGHBORS",
    "compute_descriptor",
    "parse_descriptor",
    "compute_virtual_center_from_coords",
    "find_top_k_nn_with_virtual_center",
]
