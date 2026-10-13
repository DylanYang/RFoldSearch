"""Export a database-specific target-aware E-value asset from an audited refit.

The refit stores the linear Gumbel coefficients separately from the encoded
target database.  This utility binds them to the target-cluster median-length
distribution and to the exact score context used at search time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfoldseek.database import (calibration_context_fingerprint,
                                database_fingerprint, load_database)
from rfoldseek.evalue import apply_power_calibration, validate_target_aware_params
from rfoldseek.search import GAP_EXTEND, GAP_OPEN, load_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="export database-specific target-aware E-value parameters")
    parser.add_argument("--db", required=True, help="encoded target .pt database")
    parser.add_argument("--gumbel-model", required=True,
                        help="audited full-model JSON from the refit")
    parser.add_argument("--result", required=True,
                        help="audited refit JSON containing the power fit and validation status")
    parser.add_argument("--output", required=True)
    parser.add_argument("--matrix", default=None)
    args = parser.parse_args(argv)

    db, entries, tokens, _, n_codes, clusters = load_database(args.db)
    target_digest = database_fingerprint(entries, clusters, n_codes)
    matrix = load_matrix(args.matrix)
    score_digest = calibration_context_fingerprint(
        target_digest, matrix, GAP_OPEN, GAP_EXTEND)

    model = json.loads(Path(args.gumbel_model).read_text(encoding="utf-8"))
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    model_censoring = model.get("censoring")
    if model_censoring not in {"left-censored", "none"}:
        raise ValueError("Gumbel model must record its fitted censoring mode")
    if not result.get("audit") or result["audit"][-1].get("pending") != 0:
        raise ValueError("the structural-background audit is incomplete")
    protocol = result.get("protocol", {})
    if protocol.get("deployment_ready") is not True:
        raise ValueError("refit protocol is not explicitly marked deployment_ready")
    if protocol.get("gumbel_censoring") != ("left" if model_censoring == "left-censored" else "none"):
        raise ValueError("refit protocol censoring does not match the fitted model")
    if protocol.get("score") != "SW/min(Lq,Lt), then eligible cluster max":
        raise ValueError("unsupported score definition in refit protocol")
    if protocol.get("target_length_feature") != "cluster median" or protocol.get("composition_bias") is not False:
        raise ValueError("refit protocol does not match the packaged scoring model")
    expected_source = {
        "target_database_fingerprint": target_digest,
        "calibration_context_fingerprint": score_digest,
    }
    for label, source in (("Gumbel model", model.get("source")),
                          ("refit result", result.get("source"))):
        if not isinstance(source, dict) or any(source.get(key) != value for key, value in expected_source.items()):
            raise ValueError(f"{label} does not prove the same target database and scoring context")
    if model["source"] != result["source"]:
        raise ValueError("Gumbel model and power result have different fitting provenance")
    if not model.get("fingerprint") or result.get("gumbel_model_fingerprint") != model["fingerprint"]:
        raise ValueError("power result was not fitted from this Gumbel model")
    if result.get("gumbel_model_sha256") != hashlib.sha256(Path(args.gumbel_model).read_bytes()).hexdigest():
        raise ValueError("Gumbel model contents do not match the power result")
    if model.get("fit") != ("95th-percentile left-censored weighted Gumbel likelihood"
                            if model_censoring == "left-censored"
                            else "uncensored weighted Gumbel likelihood"):
        raise ValueError("Gumbel fit description does not match censoring mode")
    coefficients = np.asarray(model["coefficients"], dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    if coefficients.shape != (2, 5) or mean.shape != (4,) or scale.shape != (4,):
        raise ValueError("unexpected target-aware Gumbel model dimensions")
    if np.any(~np.isfinite(coefficients)) or np.any(~np.isfinite(mean)) or np.any(scale <= 0):
        raise ValueError("Gumbel coefficients and standardisation must be finite")

    lengths = np.asarray([len(token) for token in tokens], dtype=float)
    labels = np.asarray(clusters, dtype=object)
    cluster_lengths = np.asarray(
        [np.median(lengths[labels == label]) for label in np.unique(labels)], dtype=float)
    values, multiplicities = np.unique(cluster_lengths, return_counts=True)
    if ("n_clusters" in protocol
            and len(cluster_lengths) != int(protocol["n_clusters"])):
        raise ValueError("result cluster count does not match the target database")
    power = dict(result["power"])
    if power.get("validation_status") != "background-audit-complete":
        raise ValueError(
            "the power fit must explicitly record validation_status "
            "'background-audit-complete'"
        )
    if "fit_min" not in power or "fit_max" not in power:
        raise ValueError("power fit must record fit_min and fit_max")
    if power.get("tail_rule") != "proportional above fit_max; capped at cluster count":
        raise ValueError("power fit must record the validated proportional capped tail rule")
    power.update({
        "scheme": "power-law",
        "n_clusters": int(len(cluster_lengths)),
        "validation_status": power["validation_status"],
        "fit_raw_evalue_min": power.pop("fit_min"),
        "fit_raw_evalue_max": power.pop("fit_max"),
    })
    if not 0 < float(power["fit_raw_evalue_min"]) <= float(power["fit_raw_evalue_max"]):
        raise ValueError("invalid power calibration fit range")
    apply_power_calibration(np.array([0.0, float(power["fit_raw_evalue_max"]),
                                      float(len(cluster_lengths))]), power)
    params = {
        "scheme": f"target-aware-{model_censoring}-gumbel-power",
        "n_chains": len(entries),
        "n_clusters": int(len(cluster_lengths)),
        "gap_open": GAP_OPEN,
        "gap_extend": GAP_EXTEND,
        "target_database_fingerprint": target_digest,
        "calibration_context_fingerprint": score_digest,
        "validation_status": "background-audit-complete",
        "target_aware_gumbel": {
            "distribution": "Gumbel",
            "score_normalization": "sw/min(query_length,target_length)",
            "feature_order": [
                "intercept", "z_log_query_length", "z_log_cluster_median_length",
                "z_log_length_difference", "z_token_entropy",
            ],
            "centre": {
                "log_query_length": {"mean": float(mean[0]), "scale": float(scale[0])},
                "log_cluster_median_length": {"mean": float(mean[1]), "scale": float(scale[1])},
                "log_length_difference": {"mean": float(mean[2]), "scale": float(scale[2])},
                "token_entropy": {"mean": float(mean[3]), "scale": float(scale[3])},
            },
            "coefficients_mu_logbeta": coefficients.T.tolist(),
            "cluster_median_lengths": values.tolist(),
            "cluster_multiplicities": multiplicities.astype(int).tolist(),
            "censoring": model_censoring,
            "fit": model["fit"],
            "auxiliary_training": model.get("auxiliary_training"),
        },
        "power_calibration": power,
        "provenance": {
            "gumbel_model_sha256": hashlib.sha256(Path(args.gumbel_model).read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256(Path(args.result).read_bytes()).hexdigest(),
            "source": model["source"],
            "scope": "database-specific target-aware E-value calibration",
            "note": "Use only with the fingerprint-matched target database and scoring context.",
        },
    }
    validate_target_aware_params(
        params, len(cluster_lengths), target_digest, score_digest, cluster_lengths)
    Path(args.output).write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(entries)} chains, {len(cluster_lengths)} clusters)")


if __name__ == "__main__":
    main()
