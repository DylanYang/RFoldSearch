"""Deterministic tie-breaking that does not depend on database order.

Prefilter scores are integer counts of k-mer hits on a common diagonal, so the
candidate list is heavily tied: at the ``top_n`` cut a median of ~147 targets
share the boundary score and compete for ~58 places. Breaking those ties by
database index makes the admitted pool a function of the order in which the
database happened to be built, and a relevant target can be dropped before it
is ever reranked.

Keying the tie-break on a hash of the chain id instead makes the ranking
independent of database order and of anything correlated with the labels,
while staying exactly reproducible given the seed. Passing the same seed to
different search tools gives them the same tie order, so a benchmark can
compare them without the comparison depending on either tool's storage layout.
"""
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
