"""Shared test helpers."""
import json

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


def synthetic_evalue_params():
    """Legacy-style parameters for synthetic search tests.

    These tests exercise E-value arithmetic, not database identity. Production
    assets carry target/scoring fingerprints and must not be reused here.
    """
    from rfoldseek.search import asset

    with open(asset("evalue_params.json")) as handle:
        params = json.load(handle)
    for field in ("target_database_fingerprint",
                  "calibration_context_fingerprint", "n_clusters"):
        params.pop(field, None)
    params.pop("empirical_calibration", None)
    params.pop("validation_status", None)
    return params
