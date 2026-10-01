"""Shared test helpers."""
import numpy as np


def add_coords(entries, seed=0):
    """Give every entry per-residue coordinates (a random walk, C4' at index 1)
    so the default structure-aware reranker can superpose it."""
    rng = np.random.default_rng(seed)
    for e in entries:
        n = len(e["tokens"])
        xyz = np.cumsum(rng.normal(size=(n, 3)) * 3.0, axis=0)
        e["coords"] = np.repeat(xyz[:, None, :], 5, axis=1).astype(np.float32)
        e["mask"] = np.ones(n, dtype=bool)
        e["n_valid"] = n
    return entries
