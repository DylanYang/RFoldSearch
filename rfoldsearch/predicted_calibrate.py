"""Refit the selected predicted-database E-value model on audited clusters."""
from __future__ import annotations

import numpy as np

from . import experimental_evalue as EV
from .experimental_calibrate import (_add_cluster_mates, _audit_missing,
                                     _labels, _leader_tables,
                                     _missing_high_score)
from .usalign import find_usalign


def fit_from_matrices(entries, tokens, ids, clusters, family, clan,
                      training_indices, calibration_indices, scores, tm, *,
                      auxiliary_count=0,
                      audit_file=None, usalign=None, audit_ceiling=100.0,
                      max_rounds=75, log=print):
    """Refit Gumbel on training queries and the empirical map on held-out queries.

    Rows of scores and tm are training rows followed by calibration rows.
    Related leaders are removed before selecting each cluster's replacement.
    """
    indices = np.r_[training_indices, calibration_indices]
    if (scores.shape != (len(indices), len(tokens)) or tm.shape != scores.shape
            or not len(training_indices) or not len(calibration_indices)):
        raise ValueError("predicted score and audit panels must match the database")
    if auxiliary_count < 0 or auxiliary_count >= len(training_indices):
        raise ValueError("auxiliary query count must leave core training queries")
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("predicted score matrix must cover every query-target pair")
    if np.any(np.isfinite(tm) & ((tm < 0) | (tm > 1))):
        raise ValueError("structural audit TM-scores must lie between zero and one")
    from collections import defaultdict
    groups = defaultdict(list)
    for index, label in enumerate(clusters):
        groups[str(label)].append(index)
    members = [groups[key] for key in sorted(groups)]
    column = np.empty(len(clusters), dtype=np.int64)
    for index, group in enumerate(members):
        column[group] = index
    label_sets = [(_labels(f), _labels(c)) for f, c in zip(family, clan)]
    lengths = np.asarray([len(token) for token in tokens], dtype=float)
    cluster_lengths = np.asarray([np.median(lengths[group]) for group in members])
    query_lengths = lengths[indices]
    entropies = np.asarray([EV.token_entropy(tokens[i]) for i in indices])
    labels = np.asarray([family[i] or f"auxiliary:{ids[i]}" for i in indices])
    if any(not family[i] for i in calibration_indices):
        raise ValueError("calibration queries require Rfam family labels")
    binary = None
    n_core = len(training_indices) - auxiliary_count
    training_weights = (np.r_[np.full(n_core, .8 / n_core),
                              np.full(auxiliary_count, .2 / auxiliary_count)]
                        if auxiliary_count else np.full(n_core, 1. / n_core))
    for iteration in range(1, max_rounds + 1):
        leaders, _, _ = _leader_tables(scores, indices, clusters, members,
                                       family, clan, tm)
        fitted = EV.fit_gumbel(scores[:len(training_indices)],
                               leaders[:len(training_indices)],
                               query_lengths[:len(training_indices)],
                               entropies[:len(training_indices)], cluster_lengths,
                               labels[:len(training_indices)], censored=True,
                               query_weights=training_weights, compress=True)
        unique_lengths, multiplicities = np.unique(cluster_lengths,
                                                   return_counts=True)
        fitted["cluster_median_lengths"] = unique_lengths.tolist()
        fitted["cluster_multiplicities"] = multiplicities.tolist()
        missing = _missing_high_score(scores, leaders, query_lengths,
                                      entropies, fitted, tm, indices, ids,
                                      audit_ceiling)
        _add_cluster_mates(missing, scores, tm, query_lengths, entropies, fitted,
                           indices, ids, members, column, label_sets,
                           audit_ceiling)
        log(f"round {iteration}: {len(missing)} unaudited pairs at E0<={audit_ceiling:g}")
        if not missing:
            break
        if audit_file is None:
            raise RuntimeError("high-scoring leaders need US-align audit")
        if binary is None:
            binary = find_usalign(usalign)
        _audit_missing(missing, entries, indices, ids, tm, audit_file, binary)
    else:
        raise RuntimeError(
            f"predicted structural audit did not settle after {max_rounds} rounds")
    evalue_rows = []
    for row in range(len(training_indices), len(indices)):
        targets = leaders[row, leaders[row] >= 0]
        evalue_rows.append(EV.initial_evalue(scores[row, targets],
                                              query_lengths[row], entropies[row],
                                              fitted))
    calibration = EV.fit_empirical(
        evalue_rows, family[calibration_indices],
        np.ones(len(calibration_indices)), len(members), ceiling=audit_ceiling,
        background_scaling="complete-eligible-clusters")
    return fitted, calibration, {"n_clusters": len(members),
                                 "n_core_training_queries": n_core,
                                 "n_auxiliary_training_queries": auxiliary_count,
                                 "auxiliary_likelihood_weight": (
                                     .2 if auxiliary_count else 0.),
                                 "n_calibration_queries": len(calibration_indices),
                                 "n_audited_pairs": int(np.isfinite(tm).sum()),
                                 "n_retained_background": int(np.sum(
                                     leaders[len(training_indices):] >= 0))}


def fit_files(args):
    from .experimental_calibrate import fit_files as shared_fit_files
    return shared_fit_files(args, predicted=True)
