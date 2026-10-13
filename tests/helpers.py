"""Shared test helpers."""
import numpy as np

from rfoldsearch.database import (calibration_context_fingerprint,
                                cluster_median_lengths, database_fingerprint)
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, load_matrix


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


def synthetic_evalue_params(entries, clusters=None, n_codes=80):
    """Audited-format target-aware fixture bound to the supplied test database."""
    if clusters is None:
        clusters = [entry.get("cluster", f"_singleton_{i}")
                    for i, entry in enumerate(entries)]
    n_clusters = len(set(clusters))
    fingerprint = database_fingerprint(entries, clusters, n_codes)
    medians, counts = np.unique(cluster_median_lengths(entries, clusters),
                                return_counts=True)
    return {
        "scheme": "target-aware-left-censored-gumbel-power",
        "n_clusters": n_clusters,
        "target_database_fingerprint": fingerprint,
        "calibration_context_fingerprint": calibration_context_fingerprint(
            fingerprint, load_matrix(), GAP_OPEN, GAP_EXTEND),
        "validation_status": "background-audit-complete",
        "target_aware_gumbel": {
            "distribution": "Gumbel",
            "score_normalization": "sw/min(query_length,target_length)",
            "censoring": "left-censored",
            "feature_order": ["intercept", "z_log_query_length",
                              "z_log_cluster_median_length",
                              "z_log_length_difference", "z_token_entropy"],
            "centre": {
                name: {"mean": 0.0, "scale": 1.0}
                for name in ("log_query_length", "log_cluster_median_length",
                             "log_length_difference", "token_entropy")
            },
            "coefficients_mu_logbeta": [[0.5, 0.0]] + [[0.0, 0.0] for _ in range(4)],
            "cluster_median_lengths": medians.tolist(),
            "cluster_multiplicities": counts.tolist(),
        },
        "power_calibration": {
            "a": 1.0, "b": 1.0, "fit_raw_evalue_min": 0.001,
            "fit_raw_evalue_max": 100.0, "n_clusters": n_clusters,
            "validation_status": "background-audit-complete",
            "tail_rule": "proportional above fit_max; capped at cluster count",
        },
    }
