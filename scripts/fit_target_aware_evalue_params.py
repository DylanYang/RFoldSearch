"""Fit target-aware Gumbel and power E-value parameters from audited scores.

Input is a TSV with one row per query and eligible target cluster. Scores are
cluster-max SW/min(Lq,Lt) values produced after family/clan and structural
background exclusions. This script fits parameters; it does not score chains
or perform the US-align audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from rfoldsearch.database import (calibration_context_fingerprint,
                                database_fingerprint, load_database)
from rfoldsearch.evalue import TAIL_RULE, TAIL_RULE_POWER, apply_power_calibration
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, load_matrix


REQUIRED = {
    "query_id", "role", "query_length", "query_entropy", "family", "clan",
    "query_weight", "expansion", "cluster_id", "cluster_median_length",
    "normalized_score",
}
FEATURES = ("log_query_length", "log_cluster_median_length",
            "log_length_difference", "token_entropy")


def _tags(value):
    return frozenset(x.strip() for x in value.replace(",", ";").split(";") if x.strip())


def read_observations(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"observation table is missing columns: {sorted(missing)}")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                parsed = {
                    "query_id": row["query_id"].strip(),
                    "role": row["role"].strip().lower(),
                    "query_length": float(row["query_length"]),
                    "query_entropy": float(row["query_entropy"]),
                    "family": _tags(row["family"]), "clan": _tags(row["clan"]),
                    "query_weight": float(row["query_weight"]),
                    "expansion": float(row["expansion"]),
                    "cluster_id": row["cluster_id"].strip(),
                    "cluster_median_length": float(row["cluster_median_length"]),
                    "score": float(row["normalized_score"]),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value on TSV line {line}") from exc
            if not parsed["query_id"] or parsed["role"] not in {"gumbel", "auxiliary", "power"}:
                raise ValueError(f"invalid query_id or role on TSV line {line}")
            if (parsed["query_length"] <= 0 or parsed["cluster_median_length"] <= 0
                    or parsed["query_entropy"] < 0 or parsed["score"] < 0
                    or parsed["query_weight"] <= 0 or parsed["expansion"] <= 0
                    or not np.all(np.isfinite([parsed[k] for k in
                       ("query_length", "query_entropy", "cluster_median_length",
                        "score", "query_weight", "expansion")]))):
                raise ValueError(f"non-finite or non-positive value on TSV line {line}")
            rows.append(parsed)
    if not rows:
        raise ValueError("observation table is empty")
    return rows


def _query_groups(rows, role):
    by_query = defaultdict(list)
    for i, row in enumerate(rows):
        if row["role"] == role:
            by_query[row["query_id"]].append(i)
    metadata = {}
    for qid, indices in by_query.items():
        first = rows[indices[0]]
        for i in indices[1:]:
            row = rows[i]
            for key in ("query_length", "query_entropy", "family", "clan",
                        "query_weight", "expansion"):
                if row[key] != first[key]:
                    raise ValueError(f"inconsistent {key} for query {qid}")
        metadata[qid] = first
    return by_query, metadata


def _features(rows, indices):
    ql = np.asarray([rows[i]["query_length"] for i in indices])
    cl = np.asarray([rows[i]["cluster_median_length"] for i in indices])
    ent = np.asarray([rows[i]["query_entropy"] for i in indices])
    raw = np.column_stack([np.log(ql), np.log(cl), np.abs(np.log(ql) - np.log(cl)), ent])
    return raw, ql, cl


def _censor_thresholds(y, weights, qlength, clength, mode, quantile, bins):
    if mode == "none":
        return np.full(len(y), -1.0), {"mode": "none"}
    shorter = np.minimum(qlength, clength)
    ratio = np.maximum(qlength, clength) / shorter
    edges_l = np.unique(np.quantile(shorter, np.linspace(0, 1, bins + 1)[1:-1]))
    edges_r = np.unique(np.quantile(ratio, np.linspace(0, 1, bins + 1)[1:-1]))
    group = np.digitize(shorter, edges_l) * (len(edges_r) + 1) + np.digitize(ratio, edges_r)
    thresholds, table = np.empty(len(y)), []
    for g in np.unique(group):
        mask = group == g
        order = np.argsort(y[mask], kind="stable")
        sy, sw = y[mask][order], weights[mask][order]
        pos = min(np.searchsorted(np.cumsum(sw), quantile * sw.sum(), side="left"), len(sy) - 1)
        threshold = float(sy[pos])
        thresholds[mask] = threshold
        table.append({"bin": int(g), "threshold": threshold, "n": int(mask.sum())})
    return thresholds, {"mode": "left-censored", "quantile": quantile,
                        "shorter_length_edges": edges_l.tolist(),
                        "length_ratio_edges": edges_r.tolist(), "strata": table}


def _gumbel_objective(coef, x, y, weights, threshold, ridge):
    p = x.shape[1]
    linear = x @ coef[p:]
    log_beta = np.clip(linear, -20.0, 10.0)
    observed = np.maximum(y, threshold)
    z = (observed - x @ coef[:p]) * np.exp(-log_beta)
    neg_z = np.clip(-z, -700.0, 50.0)
    exp_neg_z = np.exp(neg_z)
    exact = (y > threshold).astype(float)
    active = ((-z > -700.0) & (-z < 50.0)).astype(float)
    loss = np.sum(weights * (exact * (log_beta + z) + exp_neg_z))
    grad_mu = x.T @ (weights * (exp_neg_z * active - exact) * np.exp(-log_beta))
    grad_beta = x.T @ (weights * (exact * (1.0 - z) + z * exp_neg_z * active)
                       * ((linear > -20.0) & (linear < 10.0)))
    grad = np.r_[grad_mu, grad_beta]
    penalized = np.r_[np.arange(1, p), np.arange(p + 1, 2 * p)]
    loss += ridge * np.sum(coef[penalized] ** 2)
    grad[penalized] += 2 * ridge * coef[penalized]
    violation = linear - log_beta
    loss += 100.0 * np.sum(weights * violation ** 2)
    grad[p:] += 200.0 * x.T @ (weights * violation)
    return float(loss), grad


def fit_gumbel(rows, indices, censoring, quantile, bins, ridge):
    by_query, _ = _query_groups(rows, "gumbel")
    aux_by_query, _ = _query_groups(rows, "auxiliary")
    active = [i for i in indices if rows[i]["role"] in {"gumbel", "auxiliary"}]
    if len(active) < 10:
        raise ValueError("at least 10 Gumbel-training observations are required")
    raw, qlength, clength = _features(rows, active)
    query_ids = [rows[i]["query_id"] for i in active]
    qweights = np.asarray([rows[i]["query_weight"] for i in active], dtype=float)
    row_counts = Counter(query_ids)
    weights = np.asarray([w / row_counts[q] for q, w in zip(query_ids, qweights)])
    weights /= weights.sum()
    mean = np.average(raw, axis=0, weights=weights)
    scale = np.sqrt(np.average((raw - mean) ** 2, axis=0, weights=weights))
    scale[scale == 0] = 1.0
    x = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
    y = np.asarray([rows[i]["score"] for i in active])
    threshold, censor_meta = _censor_thresholds(y, weights, qlength, clength,
                                                censoring, quantile, bins)
    centre = float(np.average(y, weights=weights))
    sd = math.sqrt(float(np.average((y - centre) ** 2, weights=weights)))
    beta0 = max(sd * math.sqrt(6.0) / math.pi, 1e-3)
    p = x.shape[1]
    initial = np.zeros(2 * p)
    initial[0] = centre - 0.5772156649015329 * beta0
    initial[p] = math.log(beta0)
    solution = None
    last_result = None
    for method, options in (("L-BFGS-B", {"maxiter": 2500, "ftol": 1e-12, "gtol": 1e-7}),
                            ("BFGS", {"maxiter": 2500, "gtol": 1e-7})):
        result = minimize(_gumbel_objective, initial,
                          args=(x, y, weights, threshold, ridge), jac=True,
                          method=method, options=options)
        loss, grad = _gumbel_objective(result.x, x, y, weights, threshold, ridge)
        last_result = result
        if np.isfinite(loss) and np.max(np.abs(grad)) <= 1e-4:
            solution = result.x
            break
        initial = result.x
    if solution is None:
        raise RuntimeError(
            "Gumbel optimisation did not converge "
            f"({last_result.message}; max |gradient|="
            f"{np.max(np.abs(grad)):.3g})"
        )
    return {
        "coefficients": solution.reshape(2, p).tolist(),
        "mean": mean.tolist(), "scale": scale.tolist(),
        "censoring": censoring,
        "fit": ("95th-percentile left-censored weighted Gumbel likelihood"
                if censoring == "left-censored" else "uncensored weighted Gumbel likelihood"),
        "thresholds": censor_meta,
        "objective": float(loss), "gradient_max_abs": float(np.max(np.abs(grad))),
        "n_observations": int(len(y)), "n_queries": int(len(set(query_ids))),
        "n_auxiliary_queries": int(len(aux_by_query)),
    }


@njit(parallel=True)
def _sum_gumbel_tails(scores, mu, beta, multiplicities):
    out = np.empty(len(scores), dtype=np.float64)
    for i in prange(len(scores)):
        total = 0.0
        for j in range(len(mu)):
            z = -(scores[i] - mu[j]) / beta[j]
            if z < -745.0:
                z = -745.0
            elif z > 50.0:
                z = 50.0
            total += -math.expm1(-math.exp(z)) * multiplicities[j]
        out[i] = total
    return out


def _predict_initial(rows, indices, model, cluster_lengths, multiplicities):
    mean = np.asarray(model["mean"])
    scale = np.asarray(model["scale"])
    coef = np.asarray(model["coefficients"])
    out = np.empty(len(indices))
    grouped = defaultdict(list)
    for pos, i in enumerate(indices):
        grouped[rows[i]["query_id"]].append(pos)
    for positions in grouped.values():
        row = rows[indices[positions[0]]]
        qlog = math.log(row["query_length"])
        clog = np.log(cluster_lengths)
        raw = np.column_stack([np.full(len(clog), qlog), clog,
                               np.abs(qlog - clog),
                               np.full(len(clog), row["query_entropy"])])
        x = np.column_stack([np.ones(len(clog)), (raw - mean) / scale])
        mu, log_beta = x @ coef[0], x @ coef[1]
        beta = np.exp(np.clip(log_beta, -20.0, 10.0))
        scores = np.asarray([rows[indices[p]]["score"] for p in positions])
        out[positions] = _sum_gumbel_tails(
            scores, mu, beta, multiplicities)
    return out


MIN_QUERIES_PER_LENGTH_BIN = 20


def _background_weights(rows, indices):
    """-> (row weights, queries, families) for background observations.

    Each family carries equal weight, split over its queries, and the query
    weights sum to one; a row's weight is its query's weight times its
    expansion. Summing the weights of rows with E <= e therefore gives the
    family-balanced mean number of background clusters per query at E <= e.
    """
    _, metadata = _query_groups(rows, "power")
    present = {rows[i]["query_id"] for i in indices}
    queries = sorted(q for q in metadata if q in present)
    families = set().union(*(metadata[q]["family"] for q in queries))
    if not queries or not families or any(not metadata[q]["family"] for q in queries):
        raise ValueError("power-calibration queries require family annotations")
    family_queries = defaultdict(set)
    for q in queries:
        for fam in metadata[q]["family"]:
            family_queries[fam].add(q)
    query_weight = {
        q: sum(1.0 / (len(families) * len(family_queries[f])) / len(metadata[q]["family"])
               for f in metadata[q]["family"])
        for q in queries
    }
    total_query_weight = sum(query_weight.values())
    weights = np.asarray([query_weight[rows[i]["query_id"]] / total_query_weight
                          * rows[i]["expansion"] for i in indices])
    return weights, queries, families


def fit_power(rows, indices, initial, n_clusters, maximum, n_thresholds,
              tail_rule=TAIL_RULE):
    weights, queries, families = _background_weights(rows, indices)
    values = np.asarray(initial, dtype=np.float64)
    positive = (values > 0) & (values <= maximum)
    if positive.sum() < 3:
        raise ValueError("too few positive background E0 observations in the fit range")
    e_min = float(values[positive].min())
    grid = np.geomspace(e_min, maximum, n_thresholds)
    counts = np.asarray([weights[values <= e].sum() for e in grid])
    keep = counts > 0
    if keep.sum() < 3 or len(np.unique(np.log(grid[keep]))) < 2:
        raise ValueError("insufficient positive cumulative counts to fit power law")
    slope, intercept = np.polyfit(np.log(grid[keep]), np.log(counts[keep]), 1)
    a, b = float(np.exp(intercept)), float(slope)
    if not np.isfinite(a + b) or a <= 0 or b <= 0:
        raise ValueError("power-law fit did not produce positive finite parameters")
    return {
        "a": a, "b": b, "fit_min": e_min, "fit_max": float(maximum),
        "tail_rule": tail_rule, "n_observations": int(positive.sum()),
        "n_queries": int(len(queries)), "n_families": int(len(families)),
        "thresholds": int(keep.sum()), "n_clusters": int(n_clusters),
    }


def fit_power_length_bins(rows, indices, initial, n_clusters, maximum,
                          n_thresholds, tail_rule, n_bins):
    """One power law per query-length bin (equal query counts), plus the pooled fit.

    Bins with fewer than MIN_QUERIES_PER_LENGTH_BIN queries are avoided by
    using fewer bins; one bin reproduces the pooled fit.
    """
    pooled = fit_power(rows, indices, initial, n_clusters, maximum,
                       n_thresholds, tail_rule)
    lengths = {}
    for i in indices:
        lengths[rows[i]["query_id"]] = rows[i]["query_length"]
    queries = sorted(lengths, key=lambda q: (lengths[q], q))
    n_bins = max(1, min(int(n_bins), len(queries) // MIN_QUERIES_PER_LENGTH_BIN))
    if n_bins == 1:
        return pooled
    initial = np.asarray(initial)
    position = {i: k for k, i in enumerate(indices)}
    bins = []
    for b in range(n_bins):
        part = set(queries[b * len(queries) // n_bins:(b + 1) * len(queries) // n_bins])
        sub = [i for i in indices if rows[i]["query_id"] in part]
        fit = fit_power(rows, sub, initial[[position[i] for i in sub]], n_clusters,
                        maximum, n_thresholds, tail_rule)
        centre = float(np.mean([math.log(lengths[q]) for q in part]))
        bins.append({"log_query_length": centre, "a": fit["a"], "b": fit["b"],
                     "query_length_min": float(min(lengths[q] for q in part)),
                     "query_length_max": float(max(lengths[q] for q in part)),
                     "n_queries": fit["n_queries"], "n_observations": fit["n_observations"]})
    pooled["length_bins"] = bins
    return pooled


HOLDOUT_EVALUES = (0.001, 0.01, 0.1, 1.0)
HOLDOUT_LENGTH_STRATA = 4    # reported regardless of how many bins the fit used


def _family_folds(rows, indices, n_folds):
    """Split queries into folds so that no family spans two folds.

    Queries sharing any family are joined into one group; groups go, largest
    first, to the fold with the fewest queries (ties by a stable hash).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in indices:
        q = ("q", rows[i]["query_id"])
        for fam in rows[i]["family"]:
            parent[find(q)] = find(("f", fam))
    groups = defaultdict(set)
    for i in indices:
        q = rows[i]["query_id"]
        groups[find(("q", q))].add(q)
    if len(groups) < n_folds:
        return None
    key = lambda g: (-len(g), hashlib.sha256("\0".join(sorted(g)).encode()).hexdigest())
    folds = [set() for _ in range(n_folds)]
    for group in sorted(groups.values(), key=key):
        min(folds, key=len).update(group)
    return folds


def holdout_calibration(rows, indices, initial, n_clusters, maximum, n_thresholds,
                        tail_rule, n_bins, n_folds, evalues=HOLDOUT_EVALUES,
                        n_strata=HOLDOUT_LENGTH_STRATA):
    """Family-held-out check of the calibrated E-values in the significance region.

    Each fold's power calibration is refitted on the other folds' queries
    (the Gumbel model stays fixed: it is fitted on separate queries) and
    applied to its own. For every query-length stratum and nominal E, the
    observed value is the family-balanced mean number of background clusters
    per held-out query reaching that E; a calibrated E gives observed/nominal
    near 1. ``n_hits`` against ``expected_hits`` (nominal E times the number
    of queries) shows how much a ratio rests on.
    """
    folds = _family_folds(rows, indices, n_folds)
    if folds is None:
        return {"status": "skipped",
                "reason": f"fewer than {n_folds} independent family groups"}
    initial = np.asarray(initial, dtype=np.float64)
    query_of = np.asarray([rows[i]["query_id"] for i in indices])
    length_of = {rows[i]["query_id"]: rows[i]["query_length"] for i in indices}
    calibrated = np.full(len(indices), np.nan)
    for fold in folds:
        test = np.isin(query_of, sorted(fold))
        train = np.flatnonzero(~test)
        try:
            fit = fit_power_length_bins(rows, [indices[p] for p in train], initial[train],
                                        n_clusters, maximum, n_thresholds, tail_rule, n_bins)
        except ValueError as exc:
            return {"status": "skipped", "reason": f"fold refit failed: {exc}"}
        power = {**fit, "fit_raw_evalue_max": fit["fit_max"]}
        for q in fold:
            mask = query_of == q
            calibrated[mask] = apply_power_calibration(initial[mask], power, length_of[q])

    queries = sorted(length_of, key=lambda q: (length_of[q], q))
    n_strata = max(1, min(int(n_strata), len(queries) // MIN_QUERIES_PER_LENGTH_BIN))
    strata = [queries]
    if n_strata > 1:
        strata += [queries[s * len(queries) // n_strata:(s + 1) * len(queries) // n_strata]
                   for s in range(n_strata)]
    report = []
    for part in strata:
        positions = np.flatnonzero(np.isin(query_of, part))
        weights, _, families = _background_weights(rows, [indices[p] for p in positions])
        values = calibrated[positions]
        levels = []
        for e in evalues:
            reached = values <= e
            observed = float(weights[reached].sum())
            levels.append({"evalue": float(e), "observed": observed,
                           "ratio": observed / float(e), "n_hits": int(reached.sum()),
                           "expected_hits": float(e) * len(part)})
        report.append({"query_length_min": float(length_of[part[0]]),
                       "query_length_max": float(length_of[part[-1]]),
                       "n_queries": len(part), "n_families": len(families),
                       "levels": levels})
    return {"status": "complete", "folds": int(n_folds), "grouping": "family",
            "strata": report}


def _print_holdout(holdout):
    if holdout["status"] != "complete":
        print(f"holdout calibration skipped: {holdout['reason']}")
        return
    print(f"holdout calibration ({holdout['folds']} family folds), "
          "observed/nominal [hits/expected]:")
    for s in holdout["strata"]:
        cells = "  ".join(f"E<={l['evalue']:g}: {l['ratio']:.2f} "
                          f"[{l['n_hits']}/{l['expected_hits']:.1f}]" for l in s["levels"])
        print(f"  Lq {s['query_length_min']:g}-{s['query_length_max']:g} "
              f"({s['n_queries']} q): {cells}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="clustered encoded target database")
    parser.add_argument("--observations", required=True,
                        help="audited TSV; see README E-value fitting section")
    parser.add_argument("--gumbel-model", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--censoring", required=True,
                        choices=("left-censored", "none"),
                        help="explicitly choose the Gumbel likelihood")
    parser.add_argument("--censor-quantile", type=float, default=0.95)
    parser.add_argument("--censor-strata-bins", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--power-max", type=float, default=1000.0,
                        help="largest E0 used in the power fit (default 1000; "
                             "fit well past the significance region)")
    parser.add_argument("--power-tail", choices=("power", "proportional"), default="power",
                        help="above --power-max: keep the power law (default) or "
                             "extend proportionally; both are capped at the cluster count")
    parser.add_argument("--power-length-bins", type=int, default=4,
                        help="fit the power law separately in this many query-length "
                             "bins (default 4; fewer when bins would hold <20 queries)")
    parser.add_argument("--comp-bias-window", type=int, default=0,
                        help="half-width of the query composition-bias correction the "
                             "observation scores were computed with (0: none)")
    parser.add_argument("--power-thresholds", type=int, default=128)
    parser.add_argument("--holdout-folds", type=int, default=5,
                        help="family-held-out folds for the significance-region "
                             "calibration check (default 5; 0 skips it)")
    parser.add_argument("--holdout-evalues", default=",".join(f"{e:g}" for e in HOLDOUT_EVALUES),
                        help="comma-separated nominal E-values the check reports")
    parser.add_argument("--structural-audit-complete", action="store_true",
                        help="declare that input observations already passed the complete structural audit")
    parser.add_argument("--validation-accepted", action="store_true",
                        help="declare that nested family/clan validation has been reviewed")
    args = parser.parse_args(argv)
    if not 0 < args.censor_quantile < 1 or args.censor_strata_bins < 1:
        parser.error("censor quantile must be in (0,1) and strata bins positive")
    if args.power_length_bins < 1 or args.comp_bias_window < 0:
        parser.error("power length bins must be positive and the window non-negative")
    if args.ridge < 0 or args.power_max <= 0 or args.power_thresholds < 3:
        parser.error("ridge must be nonnegative, power maximum positive, and at least 3 thresholds")
    try:
        holdout_evalues = tuple(float(x) for x in args.holdout_evalues.split(",") if x.strip())
    except ValueError:
        parser.error("holdout E-values must be comma-separated numbers")
    if (args.holdout_folds < 0 or args.holdout_folds == 1 or not holdout_evalues
            or not all(math.isfinite(e) and e > 0 for e in holdout_evalues)):
        parser.error("holdout folds must be 0 or at least 2, and holdout E-values positive")

    rows = read_observations(args.observations)
    for role in ("gumbel", "auxiliary", "power"):
        _query_groups(rows, role)
    role_queries = {
        role: {row["query_id"] for row in rows if row["role"] == role}
        for role in ("gumbel", "auxiliary", "power")
    }
    if ((role_queries["gumbel"] | role_queries["auxiliary"])
            & role_queries["power"]):
        parser.error("Gumbel-training and power-calibration query IDs must be disjoint")
    gumbel_indices = [i for i, r in enumerate(rows) if r["role"] in {"gumbel", "auxiliary"}]
    power_indices = [i for i, r in enumerate(rows) if r["role"] == "power"]
    if not gumbel_indices or not power_indices:
        parser.error("observations must include Gumbel/auxiliary and power query roles")
    if len({(row["query_id"], row["cluster_id"]) for row in rows}) != len(rows):
        parser.error("duplicate query_id/cluster_id rows are not allowed")

    _, entries, tokens, _, n_codes, clusters = load_database(args.db)
    db_digest = database_fingerprint(entries, clusters, n_codes)
    cluster_lengths = {}
    for entry, cluster in zip(entries, clusters):
        cluster_lengths.setdefault(str(cluster), []).append(len(entry["tokens"]))
    expected_lengths = {c: float(np.median(v)) for c, v in cluster_lengths.items()}
    for row in rows:
        if row["cluster_id"] not in expected_lengths:
            parser.error(f"cluster_id {row['cluster_id']!r} is absent from the target database")
        if not np.isclose(row["cluster_median_length"], expected_lengths[row["cluster_id"]],
                          rtol=0, atol=1e-8):
            parser.error(f"cluster median length mismatch for {row['cluster_id']!r}")
    matrix = load_matrix(args.matrix)
    context_digest = calibration_context_fingerprint(db_digest, matrix, GAP_OPEN, GAP_EXTEND,
                                                     args.comp_bias_window)
    composition = {"window": args.comp_bias_window} if args.comp_bias_window else False
    tail_rule = TAIL_RULE_POWER if args.power_tail == "power" else TAIL_RULE
    model = fit_gumbel(rows, gumbel_indices, args.censoring, args.censor_quantile,
                       args.censor_strata_bins, args.ridge)
    unique_lengths, length_multiplicities = np.unique(
        np.asarray(list(expected_lengths.values())), return_counts=True)
    initial = _predict_initial(rows, power_indices, model, unique_lengths,
                               length_multiplicities)
    power = fit_power_length_bins(rows, power_indices, initial, len(expected_lengths),
                                  args.power_max, args.power_thresholds, tail_rule,
                                  args.power_length_bins)
    holdout = (holdout_calibration(rows, power_indices, initial, len(expected_lengths),
                                   args.power_max, args.power_thresholds, tail_rule,
                                   args.power_length_bins, args.holdout_folds,
                                   holdout_evalues)
               if args.holdout_folds else {"status": "skipped", "reason": "disabled"})

    source = {"target_database_fingerprint": db_digest,
              "calibration_context_fingerprint": context_digest,
              "score_normalization": "sw/min(query_length,target_length)",
              "target_length_feature": "cluster median", "composition_bias": composition}
    model["source"] = source
    model["fingerprint"] = hashlib.sha256(json.dumps(
        {"model": model, "source": source}, sort_keys=True).encode()).hexdigest()
    model["auxiliary_training"] = "rows with role=auxiliary; excluded from power calibration"
    Path(args.gumbel_model).write_text(json.dumps(model, indent=2), encoding="utf-8")
    model_digest = hashlib.sha256(Path(args.gumbel_model).read_bytes()).hexdigest()
    power["validation_status"] = ("background-audit-complete"
                                  if args.structural_audit_complete and args.validation_accepted
                                  else "fit-complete-validation-pending")
    result = {
        "protocol": {
            "score": "SW/min(Lq,Lt), then eligible cluster max",
            "target_length_feature": "cluster median", "composition_bias": composition,
            "gumbel_censoring": "left" if args.censoring == "left-censored" else "none",
            "n_clusters": len(expected_lengths),
            "deployment_ready": bool(args.structural_audit_complete and args.validation_accepted),
            "observations_sha256": hashlib.sha256(Path(args.observations).read_bytes()).hexdigest(),
            "fit_max": args.power_max,
        },
        "source": source, "gumbel_model_fingerprint": model["fingerprint"],
        "gumbel_model_sha256": model_digest,
        "audit": [{"pending": 0 if args.structural_audit_complete else 1}],
        "power": power,
        "holdout_calibration": holdout,
    }
    Path(args.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.gumbel_model} and {args.result}; "
          f"deployment_ready={result['protocol']['deployment_ready']}")
    _print_holdout(holdout)


if __name__ == "__main__":
    main()
