"""Deterministic tie-breaking that does not depend on database order."""
from __future__ import annotations

import hashlib

import numpy as np


def tie_keys(chain_ids, seed: int) -> np.ndarray:
    """-> float array in [0, 1), one per chain, from SHA-256 of ``seed:id``."""
    out = np.empty(len(chain_ids), dtype=np.float64)
    for i, cid in enumerate(chain_ids):
        digest = hashlib.sha256(f"{seed}:{cid}".encode()).digest()
        out[i] = int.from_bytes(digest[:8], "big") / 2.0 ** 64
    return out
