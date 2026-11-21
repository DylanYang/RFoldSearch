"""Fresh, matched comparison of experimental RFoldSeek E-value candidates.

This script consumes frozen preprocessing/score files plus a complete,
fingerprinted member-level structural audit. It refits every model; no old
Gumbel or power coefficients are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from rfoldsearch.database import load_database


DECISIONS = (0.01, 0.1, 1.0)
LEVELS = np.unique(np.r_[np.geomspace(0.01, 100, 24), DECISIONS, 10.0])
MODES = ("raw", "normalized", "compbias_raw", "compbias_normalized")


def entropy(tokens: np.ndarray, alphabet: int) -> float:
    counts = np.bincount(tokens, minlength=alphabet).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def labels_from_tsv(path: Path) -> dict[str, tuple[str, str]]:
    result = {}
    for row in path.read_text(encoding="utf-8").splitlines()[1:]:
        chain, family, clan = (row.split("\t") + ["", ""])[:3]
        result[chain] = family, clan
    return result


def group_folds(families: np.ndarray, clans: np.ndarray, seed: int = 27) -> np.ndarray:
    parent: dict[str, str] = {}
    def root(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]
    for family, clan in zip(families, clans):
        if clan:
            parent[root("F:" + family)] = root("C:" + clan)
    groups = np.asarray([root("F:" + family) for family in families])
    rng = np.random.default_rng(seed)
    keys = list(np.unique(groups))
    rng.shuffle(keys)
    keys.sort(key=lambda key: -int(np.sum(groups == key)))
    loads = np.zeros(4, dtype=int)
    allocation = {}
    for key in keys:
        fold = int(np.argmin(loads))
        allocation[key] = fold
        loads[fold] += int(np.sum(groups == key))
    return np.asarray([allocation[key] for key in groups], dtype=int)


def select_leaders(scores: np.ndarray, query_indices: np.ndarray, ids: list[str],
                   clusters: list, members: list[list[int]], labels: dict,
                   annotated_only: bool, structural_state: np.ndarray | None = None) -> np.ndarray:
    """Select AFTER member exclusions. State: 0 unknown, 1 unrelated, 2 related."""
    leader = np.full((len(query_indices), len(members)), -1, dtype=np.int32)
    for row, qi in enumerate(query_indices):
        qfam, qclan = labels[ids[int(qi)]]
        for ci, member_indices in enumerate(members):
            if clusters[member_indices[0]] == clusters[int(qi)]:
                continue
            eligible = []
            for target in member_indices:
                tfam, tclan = labels[ids[target]]
                if annotated_only and not tfam:
                    continue
                if qfam and tfam == qfam:
                    continue
                if qclan and tclan == qclan:
                    continue
                if structural_state is not None:
                    if structural_state[row, target] == 0:
                        raise ValueError(f"missing structural audit: {ids[int(qi)]} / {ids[target]}")
                    if structural_state[row, target] == 2:
                        continue
                eligible.append(target)
            if eligible:
                leader[row, ci] = max(eligible, key=lambda j: (scores[row, j], -j))
    return leader


def positive_leaders(scores: np.ndarray, query_indices: np.ndarray,
                     ids: list[str], clusters: list, members: list[list[int]],
                     labels: dict, structural_state: np.ndarray | None = None) -> np.ndarray:
    out = np.full((len(query_indices), len(members)), -1, dtype=np.int32)
    for row, qi in enumerate(query_indices):
        family = labels[ids[int(qi)]][0]
        if not family:
            continue
        for ci, member_indices in enumerate(members):
            if clusters[member_indices[0]] == clusters[int(qi)]:
                continue
            same_family = [j for j in member_indices if labels[ids[j]][0] == family]
            if structural_state is not None:
                if any(structural_state[row, j] == 0 for j in same_family):
                    raise ValueError("missing structural audit for same-family member")
                same_family = [j for j in same_family if structural_state[row, j] == 2]
            if same_family:
                out[row, ci] = max(same_family, key=lambda j: (scores[row, j], -j))
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_structural_state(path: Path, qids: list[str], ids: list[str],
                          db_path: Path) -> np.ndarray:
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["database_sha256"] != file_sha256(db_path):
        raise ValueError("structural audit belongs to a different database")
    if metadata.get("complete") is not True or metadata.get("threshold") != .45:
        raise ValueError("complete TM=0.45 structural audit is required")
    if metadata.get("table_sha256") != file_sha256(path):
        raise ValueError("structural audit table fingerprint mismatch")
    qi, ti = {q: i for i, q in enumerate(qids)}, {t: i for i, t in enumerate(ids)}
    state = np.zeros((len(qids), len(ids)), dtype=np.uint8)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["status"] != "ok":
                continue
            q, t = qi[row["query_id"]], ti[row["target_id"]]
            tm = np.asarray([float(row["tm_query"]), float(row["tm_target"])])
            if not np.isfinite(tm).all() or np.any((tm < 0) | (tm > 1)):
                raise ValueError("invalid structural TM-score")
            if state[q, t]:
                raise ValueError("duplicate successful structural audit pair")
            state[q, t] = 2 if tm.mean() >= .45 else 1
    return state


def features(query_lengths: np.ndarray, query_entropy: np.ndarray,
             cluster_lengths: np.ndarray) -> np.ndarray:
    qlog = np.log(query_lengths)[:, None]
    clog = np.log(cluster_lengths)[None, :]
    return np.stack(np.broadcast_arrays(qlog, clog, np.abs(qlog - clog),
                                        query_entropy[:, None]), axis=-1)


def family_weights(families: np.ndarray) -> np.ndarray:
    unique, inverse, size = np.unique(families, return_inverse=True,
                                      return_counts=True)
    return 1.0 / (len(unique) * size[inverse])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    cs = np.cumsum(weights[order])
    return float(values[order[min(int(np.searchsorted(cs, probability * cs[-1])), len(order)-1)]])


def fit_gumbel(scores: np.ndarray, leaders: np.ndarray, raw_features: np.ndarray,
               families: np.ndarray, length: np.ndarray,
               cluster_length: np.ndarray, train: np.ndarray,
               n_annotated: int, censored: bool = True) -> dict:
    # A fully removed query contributes no likelihood, but stays in power/evaluation means.
    train = train[np.any(leaders[train] >= 0, axis=1)]
    rows, cols = np.where(leaders[train] >= 0)
    selected_queries = train[rows]
    targets = leaders[selected_queries, cols]
    y = np.asarray(scores[selected_queries, targets], dtype=float)
    if len(y) < 1000:
        raise ValueError("too few cluster maxima for the conditional Gumbel fit")
    family = families[train]
    annotated = train < n_annotated
    qweight = np.empty(len(train), dtype=float)
    qweight[annotated] = .8 * family_weights(family[annotated])
    qweight[~annotated] = .2 / max(int(np.sum(~annotated)), 1)
    if not np.any(~annotated):
        qweight[annotated] /= .8
    qweight /= qweight.sum()
    counts = np.sum(leaders[train] >= 0, axis=1)
    weight = qweight[rows] / counts[rows]
    xraw = raw_features[selected_queries, cols]
    mean = np.average(xraw, axis=0, weights=weight)
    scale = np.sqrt(np.average((xraw - mean) ** 2, axis=0, weights=weight))
    scale = np.maximum(scale, 1e-8)
    x = np.column_stack([np.ones(len(y)), (xraw - mean) / scale])
    if censored:
        smaller = np.minimum(length[selected_queries], cluster_length[cols])
        ratio = np.maximum(length[selected_queries], cluster_length[cols]) / smaller
        strata = (3 * np.digitize(smaller, [30, 60, 120])
                  + np.digitize(ratio, [1.5, 3]))
        threshold = np.empty(len(y))
        fallback = weighted_quantile(y, weight, .95)
        for stratum in np.unique(strata):
            mask = strata == stratum
            threshold[mask] = (weighted_quantile(y[mask], weight[mask], .95)
                               if mask.sum() >= 200 else fallback)
        delta = (y > threshold).astype(float)
        observed = np.maximum(y, threshold)
    else:
        threshold = np.full(len(y), -np.inf)
        delta = np.ones(len(y))
        observed = y
    beta_seed = max(np.sqrt(np.average((y - np.average(y, weights=weight)) ** 2,
                                       weights=weight)) * np.sqrt(6) / np.pi, 1e-4)
    start = np.zeros((2, 5))
    start[0, 0] = np.average(y, weights=weight) - .5772156649 * beta_seed
    start[1, 0] = np.log(beta_seed)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = flat.reshape(2, 5)
        mu = x @ coefficients[0]
        linear = x @ coefficients[1]
        logbeta = np.clip(linear, -20, 10)
        invbeta = np.exp(-logbeta)
        z = (observed - mu) * invbeta
        minus_z = np.clip(-z, -700, 50)
        expminus = np.exp(minus_z)
        active = ((-z > -700) & (-z < 50)).astype(float)
        loss = float(np.sum(weight * (delta * (logbeta + z) + expminus)))
        grad_mu = x.T @ (weight * (expminus * active - delta) * invbeta)
        grad_beta = x.T @ (weight * (delta * (1-z) + z * expminus * active)
                            * ((linear > -20) & (linear < 10)))
        grad = np.r_[grad_mu, grad_beta]
        indices = np.r_[1:5, 6:10]
        loss += 1e-3 * float(np.sum(flat[indices] ** 2))
        grad[indices] += 2e-3 * flat[indices]
        violation = linear - logbeta
        loss += 100 * float(np.sum(weight * violation ** 2))
        grad[5:] += 200 * (x.T @ (weight * violation))
        return loss, grad

    result = minimize(objective, start.ravel(), method="L-BFGS-B", jac=True,
                      options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-7})
    loss, gradient = objective(result.x)
    if not np.isfinite(loss) or np.max(np.abs(gradient)) > 2e-3:
        raise RuntimeError(f"Gumbel fit did not converge: {result.message}; "
                           f"max gradient={np.max(np.abs(gradient)):.4g}")
    return {"coefficients": result.x.reshape(2, 5).tolist(),
            "mean": mean.tolist(), "scale": scale.tolist(),
            "censored": censored, "n_observations": len(y),
            "n_tail_observations": int(delta.sum()),
            "n_train_queries": len(train), "loss": loss,
            "max_abs_gradient": float(np.max(np.abs(gradient)))}


def gumbel_parameters(feature_row: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([np.ones(len(feature_row)),
                         (feature_row - model["mean"]) / model["scale"]])
    coefficients = np.asarray(model["coefficients"])
    return x @ coefficients[0], np.exp(np.clip(x @ coefficients[1], -20, 10))


def log_gumbel_survival(z: np.ndarray) -> np.ndarray:
    # log(1-exp(-exp(-z))) evaluated without cancellation in the right tail.
    out = np.empty_like(z, dtype=float)
    tail = z > 16
    out[tail] = -z[tail]
    other = ~tail
    out[other] = np.log(-np.expm1(-np.exp(np.clip(-z[other], -700, 50))))
    return out


def e0_for_scores(score: np.ndarray, feature_row: np.ndarray, model: dict,
                  batch: int = 128) -> np.ndarray:
    mu, beta = gumbel_parameters(feature_row, model)
    result = np.empty(len(score), dtype=float)
    for start in range(0, len(score), batch):
        chunk = score[start:start + batch, None]
        z = (chunk - mu[None, :]) / beta[None, :]
        result[start:start + batch] = np.exp(
            np.maximum(logsumexp(log_gumbel_survival(z), axis=1), -700))
    return result


def e0_background(scores: np.ndarray, leaders: np.ndarray,
                  feature_rows: np.ndarray, model: dict, rows: np.ndarray) -> list[np.ndarray]:
    values = []
    for row in rows:
        targets = leaders[row, leaders[row] >= 0]
        values.append(e0_for_scores(scores[row, targets], feature_rows[row], model))
    return values


def fit_power(e0_rows: list[np.ndarray], families: np.ndarray,
              query_lengths: np.ndarray, expansion: np.ndarray,
              binned: bool) -> dict:
    if len(e0_rows) != len(families):
        raise ValueError("power rows and family labels are not aligned")
    low = np.min([np.min(v[(v > 0) & (v <= 100)])
                  for v in e0_rows if np.any((v > 0) & (v <= 100))])
    grid = np.geomspace(max(float(low), 1e-8), 100.0, 96)

    def counts(rows: np.ndarray, within_bin: bool) -> np.ndarray:
        local_families = families[rows]
        local_weights = ((family_weights(local_families) if within_bin
                          else family_weights(families)[rows]) * expansion[rows])
        values = np.concatenate([e0_rows[i] for i in rows])
        if not len(values):
            return np.zeros(len(grid))
        observation_weights = np.concatenate(
            [np.full(len(e0_rows[i]), local_weights[k]) for k, i in enumerate(rows)])
        order = np.argsort(values)
        cumulative = np.cumsum(observation_weights[order])
        positions = np.searchsorted(values[order], grid, side="right")
        return np.where(positions > 0, cumulative[np.maximum(positions-1, 0)], 0.0)

    full_counts = counts(np.arange(len(e0_rows)), False)
    use = full_counts > 0
    if use.sum() < 8:
        raise ValueError("insufficient positive background counts to fit power")
    b, loga = np.polyfit(np.log(grid[use]), np.log(full_counts[use]), 1)
    if b <= 0 or not np.isfinite([b, loga]).all():
        raise ValueError("invalid positive-slope power fit")
    result = {"a": float(np.exp(loga)), "b": float(b),
              "fit_min": float(grid[use][0]), "fit_max": 100.0,
              "fit_points": int(use.sum()), "n_queries": len(e0_rows)}
    if not binned:
        return result
    edges = np.unique(np.quantile(np.log(query_lengths), [.25, .5, .75]))
    assignment = np.digitize(np.log(query_lengths), edges)
    groups = [np.flatnonzero(assignment == k) for k in np.unique(assignment)]
    while len(groups) > 1:
        unsupported = [k for k, g in enumerate(groups) if len(g) < 12 or
                       len(np.unique(families[g])) < 4 or
                       np.sum(counts(g, True) > 0) < 8]
        if not unsupported:
            break
        k = unsupported[0]
        other = k+1 if k+1 < len(groups) else k-1
        lo, hi = sorted((k, other))
        groups[lo:hi+1] = [np.r_[groups[lo], groups[hi]]]
    bins = []
    for group in groups:
        curve = counts(group, True)
        valid = curve > 0
        local_loga = np.mean(np.log(curve[valid]) - b * np.log(grid[valid]))
        bins.append({"log_length_centre": float(np.mean(np.log(query_lengths[group]))),
                     "a": float(np.exp(local_loga)), "n_queries": len(group),
                     "n_families": len(np.unique(families[group]))})
    result["bins"] = bins
    return result


def apply_power(e0: np.ndarray, model: dict, length: float,
                n_clusters: int) -> np.ndarray:
    if "bins" in model:
        centre = [item["log_length_centre"] for item in model["bins"]]
        loga = [np.log(item["a"]) for item in model["bins"]]
        a = float(np.exp(np.interp(np.log(length), centre, loga)))
    else:
        a = model["a"]
    bounded = np.maximum(np.asarray(e0, dtype=float), 1e-304)
    result = a * bounded ** model["b"]
    high = bounded > model["fit_max"]
    if np.any(high):
        endpoint = a * model["fit_max"] ** model["b"]
        result = np.where(high, endpoint * bounded/model["fit_max"], result)
    return np.minimum(result, n_clusters)


def mean_family_curve(rows: list[dict], key: str, thresholds: np.ndarray,
                      bootstrap: int = 2000) -> dict:
    families = sorted({item["family"] for item in rows})
    per_family = []
    for family in families:
        group = [item for item in rows if item["family"] == family]
        per_family.append(np.mean([item[key] for item in group], axis=0))
    array = np.asarray(per_family)
    observed = array.mean(axis=0)
    rng = np.random.default_rng(27017)
    draws = rng.integers(0, len(families), (bootstrap, len(families)))
    bands = np.quantile(array[draws].mean(axis=1), [.025, .975], axis=0)
    supported = observed > 0
    rmse = (float(np.sqrt(np.mean(np.log10(observed[supported] /
                                          thresholds[supported]) ** 2)))
            if supported.any() else None)
    return {"observed": observed.tolist(), "ci95_low": bands[0].tolist(),
            "ci95_high": bands[1].tolist(), "ratio": (observed/thresholds).tolist(),
            "log10_rmse_positive_points": rmse,
            "n_zero_count_thresholds": int(np.sum(~supported)),
            "n_families": len(families), "n_queries": len(rows)}


def family_macro_metric(rows: list[dict], key: str,
                        bootstrap: int = 2000) -> dict:
    families = sorted({item["family"] for item in rows if item[key] is not None})
    if not families:
        return {"value": None, "ci95": None, "n_families": 0}
    family_mean = np.asarray([np.mean([item[key] for item in rows
                                       if item["family"] == family and item[key] is not None])
                              for family in families])
    rng = np.random.default_rng(27017)
    draws = rng.integers(0, len(families), (bootstrap, len(families)))
    ci = np.quantile(family_mean[draws].mean(axis=1), [.025, .975])
    return {"value": float(family_mean.mean()), "ci95": ci.tolist(),
            "n_families": len(families)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--uncensored", action="store_true")
    audit = parser.add_mutually_exclusive_group(required=True)
    audit.add_argument("--structural-audit", type=Path,
                       help="complete member-level audit TSV; clean BOTH training stages")
    audit.add_argument("--exploratory-unaudited", action="store_true",
                       help="reproduce the exploratory Rfam-negative proxy comparison")
    args = parser.parse_args()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    db_path = args.run / "db_nr80.pt"
    _, _, tokens, ids, alphabet, clusters = load_database(str(db_path))
    manifest = json.loads((args.run / "score_manifest.json").read_text())
    if manifest["n_targets"] != len(ids):
        raise ValueError("score matrix and database target counts differ")
    if file_sha256(db_path) != manifest["database_sha256"]:
        raise ValueError("database content differs from score manifest")
    if file_sha256(args.labels) != manifest["labels_sha256"]:
        raise ValueError("labels differ from frozen comparison inputs")
    qids = manifest["query_ids"]
    n_annotated = manifest["n_annotated_queries"]
    if len(qids) != manifest["n_queries"]:
        raise ValueError("score query manifest is incomplete")
    id_to_index = {id_: i for i, id_ in enumerate(ids)}
    qi = np.asarray([id_to_index[qid] for qid in qids])
    structural_state = (load_structural_state(args.structural_audit, qids, ids, db_path)
                        if args.structural_audit else None)
    labels = labels_from_tsv(args.labels)
    if set(ids) != set(labels):
        raise ValueError("database IDs and labels are not the same set")
    if any(not labels[qid][0] for qid in qids[:n_annotated]):
        raise ValueError("annotated query block contains unannotated chains")
    if any(labels[qid][0] for qid in qids[n_annotated:]):
        raise ValueError("auxiliary query block contains annotated chains")
    member_map = defaultdict(list)
    for index, cluster in enumerate(clusters):
        member_map[str(cluster)].append(index)
    cluster_ids = sorted(member_map)
    members = [member_map[c] for c in cluster_ids]
    n_clusters = len(members)
    length = np.asarray([len(tok) for tok in tokens], dtype=float)
    h = np.asarray([entropy(tok, alphabet) for tok in tokens])
    cluster_length = np.asarray([np.median(length[m]) for m in members])
    feature_rows = features(length[qi], h[qi], cluster_length)
    families = np.asarray([labels[q][0] for q in qids])
    clans = np.asarray([labels[q][1] for q in qids])
    folds = group_folds(families[:n_annotated], clans[:n_annotated])
    if any(not np.any(folds == fold) for fold in range(4)):
        raise ValueError("a cross-validation fold is empty")
    (args.run / "cv_folds.tsv").write_text(
        "query_id\tfamily\tclan\tfold\n" + "".join(
            f"{qids[i]}\t{families[i]}\t{clans[i]}\t{folds[i]}\n"
            for i in range(n_annotated)), encoding="utf-8")
    raw = np.load(args.run / "raw_sw.npy", mmap_mode="r")
    compbias = np.load(args.run / "compbias20_sw.npy", mmap_mode="r")
    if raw.shape != (len(qi), len(ids)) or compbias.shape != raw.shape:
        raise ValueError("freshly scored arrays have an unexpected shape")
    denominator = np.minimum(length[qi, None], length[None, :])
    score_modes = {"raw": raw, "normalized": np.asarray(raw) / denominator,
                   "compbias_raw": compbias,
                   "compbias_normalized": np.asarray(compbias) / denominator}
    method_results = {}
    audit_candidates: set[tuple[str, str, str]] = set()
    for mode in MODES:
        scores = score_modes[mode]
        candidate_records = []
        print(f"Preparing {mode}: eligible cluster maxima", flush=True)
        before = select_leaders(scores, qi, ids, clusters, members, labels, False)
        before_annotated = select_leaders(scores[:n_annotated], qi[:n_annotated],
                                           ids, clusters, members, labels, True)
        leader = select_leaders(scores, qi, ids, clusters, members, labels, False,
                                structural_state)
        annotated_state = None if structural_state is None else structural_state[:n_annotated]
        annotated_leader = select_leaders(scores[:n_annotated], qi[:n_annotated],
                                           ids, clusters, members, labels, True, annotated_state)
        positive = positive_leaders(scores[:n_annotated], qi[:n_annotated],
                                    ids, clusters, members, labels)
        structural_positive = (positive_leaders(scores[:n_annotated], qi[:n_annotated],
                                ids, clusters, members, labels, annotated_state)
                               if structural_state is not None else None)
        # Expansion is defined BEFORE structural exclusions; do not inflate it after cleaning.
        bq = np.sum(before_annotated >= 0, axis=1)
        if np.any(bq == 0):
            raise ValueError("an annotated query has no evaluable background clusters")
        expansion = n_clusters / bq
        per_query = []
        fold_models = []
        crossfit_e0: list[np.ndarray | None] = [None] * n_annotated
        for held_out in range(4):
            ptraining = np.flatnonzero(folds != held_out)
            testing = np.flatnonzero(folds == held_out)
            inner_e0: dict[int, np.ndarray] = {}
            inner_models = []
            for calibration_fold in range(4):
                if calibration_fold == held_out:
                    continue
                gtrain = np.r_[np.flatnonzero((folds != held_out) &
                                             (folds != calibration_fold)),
                               np.arange(n_annotated, len(qi))]
                inner_gumbel = fit_gumbel(scores, leader, feature_rows, families,
                                           length[qi], cluster_length, gtrain,
                                           n_annotated, not args.uncensored)
                calibration_rows = np.flatnonzero(folds == calibration_fold)
                estimates = e0_background(scores, annotated_leader, feature_rows,
                                          inner_gumbel, calibration_rows)
                inner_e0.update(zip(calibration_rows.tolist(), estimates))
                inner_models.append({"calibration_fold": calibration_fold,
                                     "gumbel": inner_gumbel})
            power_e0 = [inner_e0[int(row)] for row in ptraining]
            models = {
                "global": fit_power(power_e0, families[ptraining], length[qi[ptraining]],
                                    expansion[ptraining], False),
                "four_bin": fit_power(power_e0, families[ptraining], length[qi[ptraining]],
                                      expansion[ptraining], True),
            }
            gtrain = np.r_[ptraining, np.arange(n_annotated, len(qi))]
            gumbel = fit_gumbel(scores, leader, feature_rows, families,
                                 length[qi], cluster_length, gtrain, n_annotated,
                                 not args.uncensored)
            fold_models.append({"held_out_fold": held_out,
                                "power_training_folds": [k for k in range(4)
                                                         if k != held_out],
                                "gumbel": gumbel, "inner_gumbel": inner_models,
                                "power": models})
            for row in testing:
                negative_targets = annotated_leader[row, annotated_leader[row] >= 0]
                positive_targets = positive[row, positive[row] >= 0]
                negatives = scores[row, negative_targets]
                positives = scores[row, positive_targets]
                e0_negative = e0_for_scores(negatives, feature_rows[row], gumbel)
                e0_positive = e0_for_scores(positives, feature_rows[row], gumbel)
                structural_e0 = (e0_for_scores(scores[row, structural_positive[row,
                                    structural_positive[row] >= 0]], feature_rows[row], gumbel)
                                 if structural_positive is not None else None)
                crossfit_e0[row] = e0_negative
                global_negative = apply_power(e0_negative, models["global"],
                                              length[qi[row]], n_clusters)
                global_positive = apply_power(e0_positive, models["global"],
                                              length[qi[row]], n_clusters)
                bin_negative = apply_power(e0_negative, models["four_bin"],
                                           length[qi[row]], n_clusters)
                bin_positive = apply_power(e0_positive, models["four_bin"],
                                           length[qi[row]], n_clusters)
                for target, initial, global_e, bin_e in zip(
                        positive_targets, e0_positive, global_positive, bin_positive):
                    candidate_records.append((qids[row], ids[int(target)], "same_family",
                                              held_out, initial, global_e, bin_e))
                    audit_candidates.add((mode, qids[row], ids[int(target)]))
                for target, initial, global_e, bin_e in zip(
                        negative_targets, e0_negative, global_negative, bin_negative):
                    if min(global_e, bin_e) <= 1:
                        candidate_records.append((qids[row], ids[int(target)], "Rfam_negative",
                                                  held_out, initial, global_e, bin_e))
                        audit_candidates.add((mode, qids[row], ids[int(target)]))
                result = {"query_id": qids[row], "family": families[row],
                          "fold": int(held_out), "length": int(length[qi[row]]),
                          "entropy": float(h[qi[row]]),
                          "n_negative_clusters": len(negatives),
                          "n_positive_clusters": len(positives),
                          "expansion": float(expansion[row])}
                if len(positives) and len(negatives):
                    result["auc"] = float(np.mean(
                        (positives[:, None] > negatives[None, :]) +
                        .5 * (positives[:, None] == negatives[None, :])))
                else:
                    result["auc"] = None
                for calibration in ("initial", "global", "four_bin"):
                    if calibration == "initial":
                        negative_e = e0_negative
                        positive_e = e0_positive
                    elif calibration == "global":
                        negative_e = global_negative
                        positive_e = global_positive
                    else:
                        negative_e = bin_negative
                        positive_e = bin_positive
                    result[f"counts_{calibration}"] = (
                        expansion[row] * np.sum(negative_e[:, None] <= LEVELS, axis=0)).tolist()
                    result[f"recall_{calibration}"] = (
                        [float(np.mean(positive_e <= e)) for e in DECISIONS]
                        if len(positives) else None)
                    result[f"any_{calibration}"] = (
                        [float(np.any(positive_e <= e)) for e in DECISIONS]
                        if len(positives) else None)
                    if structural_e0 is not None:
                        se = (structural_e0 if calibration == "initial" else
                              apply_power(structural_e0, models[calibration],
                                          length[qi[row]], n_clusters))
                        result[f"structural_recall_{calibration}"] = (
                            [float(np.mean(se <= e)) for e in DECISIONS] if len(se) else None)
                        result[f"structural_any_{calibration}"] = (
                            [float(np.any(se <= e)) for e in DECISIONS] if len(se) else None)
                    result[f"any_unrelated_{calibration}"] = [
                        float(np.any(negative_e <= e)) for e in DECISIONS]
                per_query.append(result)
            print(f"  {mode}: completed outer fold {held_out + 1}/4", flush=True)
        per_query.sort(key=lambda item: item["query_id"])
        curves = {calibration: mean_family_curve(
            per_query, f"counts_{calibration}", LEVELS)
            for calibration in ("initial", "global", "four_bin")}
        sensitivity = {}
        for calibration in ("initial", "global", "four_bin"):
            sensitivity[calibration] = {
                "positive_cluster_recall": {str(threshold): family_macro_metric(
                    [{**item, "value": item[f"recall_{calibration}"][j]
                      if item[f"recall_{calibration}"] is not None else None}
                     for item in per_query], "value")
                    for j, threshold in enumerate(DECISIONS)},
                "query_with_at_least_one_homolog": {str(threshold): family_macro_metric(
                    [{**item, "value": item[f"any_{calibration}"][j]
                      if item[f"any_{calibration}"] is not None else None}
                     for item in per_query], "value")
                    for j, threshold in enumerate(DECISIONS)}
            }
            if structural_state is not None:
                for metric in ("structural_recall", "structural_any", "any_unrelated"):
                    sensitivity[calibration][metric] = {
                        str(threshold): family_macro_metric(
                            [{**item, "value": item[f"{metric}_{calibration}"][j]
                              if item[f"{metric}_{calibration}"] is not None else None}
                             for item in per_query], "value")
                        for j, threshold in enumerate(DECISIONS)}
        full_gumbel = fit_gumbel(scores, leader, feature_rows, families,
                                  length[qi], cluster_length, np.arange(len(qi)),
                                  n_annotated, not args.uncensored)
        if any(value is None for value in crossfit_e0):
            raise AssertionError("missing an out-of-fold E0 row")
        final_power = {
            "global": fit_power(crossfit_e0, families[:n_annotated],
                                length[qi[:n_annotated]], expansion, False),
            "four_bin": fit_power(crossfit_e0, families[:n_annotated],
                                  length[qi[:n_annotated]], expansion, True),
        }
        self_raw = np.load(args.run / ("self_compbias20_sw.npy" if
                                       mode.startswith("compbias") else "self_raw_sw.npy"))
        if "normalized" in mode:
            self_raw = self_raw / length
        self_features = features(length, h, cluster_length)
        self_e0 = np.asarray([e0_for_scores(np.asarray([score]), self_features[i],
                                                 full_gumbel)[0]
                              for i, score in enumerate(self_raw)])
        self_summary = {}
        for calibration in ("initial", "global", "four_bin"):
            if calibration == "initial":
                values = self_e0
            else:
                values = np.asarray([apply_power(np.asarray([e]), final_power[calibration],
                                                 length[i], n_clusters)[0]
                                     for i, e in enumerate(self_e0)])
            self_summary[calibration] = {
                "min": float(values.min()), "median": float(np.median(values)),
                "p95": float(np.quantile(values, .95)), "max": float(values.max()),
                "n_gt_1": int(np.sum(values > 1)), "n_gt_10": int(np.sum(values > 10)),
                "max_query_id": ids[int(np.argmax(values))],
                "max_query_length": int(length[np.argmax(values)]),
            }
        method_results[mode] = {"curves": curves, "sensitivity": sensitivity,
                                "auc": family_macro_metric(per_query, "auc"),
                                "self_search": self_summary,
                                "fold_models": fold_models,
                                "full_gumbel": full_gumbel,
                                "full_power": final_power,
                                "background_clusters_min": int(bq.min()),
                                "background_clusters_max": int(bq.max()),
                                "structural_cleaning": {
                                    "before_clusters": int(np.sum(before >= 0)),
                                    "after_clusters": int(np.sum(leader >= 0)),
                                    "backfilled_clusters": int(np.sum((leader >= 0) &
                                                                         (leader != before))),
                                    "emptied_clusters": int(np.sum((before >= 0) & (leader < 0))),
                                    "expansion_denominator": "before structural exclusions"},
                                "n_homolog_evaluable_queries": int(sum(
                                    item["n_positive_clusters"] > 0 for item in per_query))}
        with (args.run / f"per_query_{mode}.jsonl").open("w", encoding="utf-8") as handle:
            for item in per_query:
                handle.write(json.dumps(item) + "\n")
        with (args.run / f"candidate_hits_{mode}.tsv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["query_id", "target_id", "relationship", "fold",
                             "initial_e0", "global_e", "four_bin_e"])
            writer.writerows(candidate_records)
        (args.run / "comparison_progress.json").write_text(
            json.dumps({"completed_modes": list(method_results)}, indent=2) + "\n",
            encoding="utf-8")
    with (args.run / "structural_audit_candidates.tsv").open("w", encoding="utf-8",
                                                            newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["score_mode", "query_id", "target_id"])
        writer.writerows(sorted(audit_candidates))
    summary = {"protocol": {
        "n_structures": len(ids), "n_clusters": n_clusters,
        "n_annotated_queries": n_annotated, "n_auxiliary_queries": len(qi)-n_annotated,
        "n_families": len(np.unique(families[:n_annotated])),
        "folds": 4, "fold_seed": 27,
        "validation": "outer four-fold family/clan blocking; power trained from inner cross-fitted E0 in all three training folds; test-fold Gumbel trained on all three training folds",
        "gumbel": "target-aware linear five-feature model; stratum-specific 95th percentile left-censoring",
        "censored": not args.uncensored,
        "power": "family-balanced cumulative-count OLS; experimental N_D/B_q expansion; 96 thresholds up to 100",
        "score_modes": list(MODES),
        "structural_audit": ("complete member-level audit; exclusions and backfill before BOTH Gumbel and power fitting"
                             if structural_state is not None else "NOT YET PERFORMED"),
        "structural_audit_sha256": (file_sha256(args.structural_audit)
                                    if args.structural_audit else None),
        "n_audit_candidates_e_le_1": len(audit_candidates)},
        "levels": LEVELS.tolist(), "methods": method_results}
    (args.run / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n",
                                               encoding="utf-8")
    print(f"Wrote comparison for {len(ids)} targets and {n_annotated} annotated queries", flush=True)


if __name__ == "__main__":
    main()
