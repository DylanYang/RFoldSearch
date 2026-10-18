"""Refit audited experimental E-values and export deployable database parameters.

The score and TM matrices use rows from ``--query-indices`` and columns from
``--db``. ``--score-manifest`` identifies the rows of ``--scores`` and records
the frozen database/matrix hashes. Missing high-scoring structural audits are
performed with US-align and saved in ``--workdir`` for a resumable refit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rfoldsearch import experimental_evalue as EV
from rfoldsearch.calibrate import audit_pairs, read_labels
from rfoldsearch.database import (calibration_context_fingerprint,
                                  database_fingerprint, load_database)
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, load_matrix
from rfoldsearch.smith_waterman import smith_waterman_numba
from rfoldsearch.usalign import find_usalign


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leader_tables(scores, query_indices, clusters, members, family, clan, tm):
    """Apply member exclusions before maxima; related leaders backfill in cluster."""
    n_queries, n_clusters = len(query_indices), len(members)
    clean = np.full((n_queries, n_clusters), -1, dtype=np.int32)
    annotated = np.full_like(clean, -1)
    before = np.zeros(n_queries, dtype=np.int32)
    for row, qi in enumerate(query_indices):
        for column, group in enumerate(members):
            if clusters[group[0]] == clusters[qi]:
                continue
            eligible = [j for j in group if np.isfinite(scores[row, j])
                        and (not family[qi] or family[j] != family[qi])
                        and (not clan[qi] or clan[j] != clan[qi])]
            if not eligible:
                continue
            if any(family[j] for j in eligible):
                before[row] += 1
            remaining = [j for j in eligible if not (np.isfinite(tm[row, j])
                                                   and tm[row, j] >= .45)]
            if remaining:
                clean[row, column] = max(remaining,
                                         key=lambda j: (scores[row, j], -j))
            annotated_members = [j for j in remaining if family[j]]
            if annotated_members:
                annotated[row, column] = max(annotated_members,
                                             key=lambda j: (scores[row, j], -j))
    return clean, annotated, before


def _missing_high_score(scores, leaders, lengths, entropies, model, tm,
                        query_indices, ids, ceiling):
    missing = {}
    for row in range(len(query_indices)):
        targets = leaders[row, leaders[row] >= 0]
        if not len(targets):
            continue
        e0 = EV.initial_evalue(scores[row, targets], lengths[row],
                              entropies[row], model)
        for target, value in zip(targets, e0):
            if value <= ceiling and not np.isfinite(tm[row, target]):
                missing[(row, int(target))] = (ids[query_indices[row]],
                                               ids[target], float(value))
    return missing


def _audit_missing(missing, entries, query_indices, ids, tm, path, binary):
    pairs = [(query, target) for query, target, _ in missing.values()]
    by_id = dict(zip(ids, entries))
    audited, related = set(), set()
    audit_pairs(binary, pairs, by_id, by_id, path, audited, related, workers=4)
    positions = {(ids[query_indices[row]], ids[target]): (row, target)
                 for row, target in missing}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            pair = (record["query_id"], record["target_id"])
            if pair in positions:
                row, target = positions[pair]
                tm[row, target] = .5 * (float(record["tm_query"])
                                          + float(record["tm_target"]))
    if any(not np.isfinite(tm[row, target]) for row, target in missing):
        raise RuntimeError("US-align did not return every requested audit pair")


def fit_from_matrices(entries, tokens, ids, clusters, family, clan, query_indices,
                      scores, tm, *, audit_ceiling=100.0, audit_file=None,
                      usalign=None, censored=True, max_rounds=20, log=print):
    """Clean, refit Gumbel after each audit, and fit the empirical map."""
    members_by_cluster = defaultdict(list)
    for index, label in enumerate(clusters):
        members_by_cluster[str(label)].append(index)
    members = [members_by_cluster[key] for key in sorted(members_by_cluster)]
    n_clusters = len(members)
    lengths_all = np.asarray([len(token) for token in tokens], dtype=float)
    cluster_lengths = np.asarray([np.median(lengths_all[group]) for group in members])
    query_lengths = lengths_all[query_indices]
    entropies = np.asarray([EV.token_entropy(tokens[i]) for i in query_indices])
    query_families = family[query_indices]
    if any(not item for item in query_families):
        raise ValueError("every calibration query needs an Rfam family")
    binary = None
    for iteration in range(1, max_rounds + 1):
        leaders, annotated, before = _leader_tables(
            scores, query_indices, clusters, members, family, clan, tm)
        fitted = EV.fit_gumbel(scores, leaders, query_lengths, entropies,
                               cluster_lengths, query_families,
                               censored=censored)
        unique_lengths, multiplicities = np.unique(cluster_lengths,
                                                   return_counts=True)
        fitted["cluster_median_lengths"] = unique_lengths.tolist()
        fitted["cluster_multiplicities"] = multiplicities.tolist()
        missing = _missing_high_score(scores, leaders, query_lengths, entropies,
                                      fitted, tm, query_indices, ids, audit_ceiling)
        missing.update(_missing_high_score(scores, annotated, query_lengths,
                                           entropies, fitted, tm, query_indices,
                                           ids, audit_ceiling))
        log(f"round {iteration}: {len(missing)} unaudited leaders at E0<={audit_ceiling:g}")
        if not missing:
            break
        if audit_file is None:
            raise RuntimeError(f"{len(missing)} high-scoring cluster leaders need "
                               "US-align; supply --workdir and --usalign")
        if binary is None:
            binary = find_usalign(usalign)
        _audit_missing(missing, entries, query_indices, ids, tm, audit_file, binary)
    else:
        raise RuntimeError(f"structural audit did not settle after {max_rounds} rounds")
    evalue_rows = []
    for row in range(len(query_indices)):
        targets = annotated[row, annotated[row] >= 0]
        evalue_rows.append(EV.initial_evalue(scores[row, targets],
                                              query_lengths[row], entropies[row], fitted))
    if np.any(before == 0):
        raise ValueError("a calibration query has no eligible annotated background")
    calibration = EV.fit_empirical(evalue_rows, query_families,
                                    n_clusters / before, n_clusters,
                                    ceiling=audit_ceiling)
    return fitted, calibration, {"n_clusters": n_clusters,
                                 "n_queries": len(query_indices),
                                 "n_families": len(set(query_families)),
                                 "n_audited_pairs": int(np.isfinite(tm).sum()),
                                 "n_retained_background": int(np.sum(annotated >= 0))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "labels", "output", "workdir"):
        parser.add_argument("--" + name, required=True)
    for name in ("query-indices", "scores", "score-manifest", "tm-matrix",
                 "tm-manifest"):
        parser.add_argument("--" + name)
    parser.add_argument("--matrix")
    parser.add_argument("--usalign")
    parser.add_argument("--uncensored", action="store_true")
    return fit_files(parser.parse_args(argv))


def fit_files(args):
    """Run the package fitting workflow from parsed CLI arguments."""
    db_path = Path(args.db)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _, entries, tokens, ids, n_codes, clusters = load_database(str(db_path))
    matrix_path = Path(args.matrix) if args.matrix else Path(__file__).resolve().parents[1] / "rfoldsearch/assets/sub_matrix.pt"
    matrix = load_matrix(str(matrix_path))
    labels = read_labels(args.labels)
    family = np.asarray([labels.get(chain_id, {}).get("family", "") for chain_id in ids])
    clan = np.asarray([labels.get(chain_id, {}).get("clan", "") for chain_id in ids])
    query_indices = (np.load(args.query_indices).astype(int)
                     if args.query_indices else np.flatnonzero(family))
    if (query_indices.ndim != 1 or len(set(query_indices)) != len(query_indices)
            or np.any(query_indices < 0) or np.any(query_indices >= len(ids))):
        raise ValueError("query index panel is invalid")
    query_ids = [ids[index] for index in query_indices]
    if bool(args.scores) != bool(args.score_manifest):
        raise ValueError("--scores and --score-manifest must be supplied together")
    if bool(args.tm_matrix) != bool(args.tm_manifest):
        raise ValueError("--tm-matrix and --tm-manifest must be supplied together")
    scores_path = Path(args.scores) if args.scores else workdir / "raw_sw.npy"
    manifest_path = (Path(args.score_manifest) if args.score_manifest
                     else workdir / "raw_sw.manifest.json")
    if not scores_path.exists() and args.scores:
        raise FileNotFoundError(scores_path)
    if not scores_path.exists():
        scores_computed = np.empty((len(query_indices), len(ids)), dtype=np.float32)
        for row, qi in enumerate(query_indices):
            for target, token in enumerate(tokens):
                scores_computed[row, target] = smith_waterman_numba(
                    tokens[qi], token, matrix, GAP_OPEN, GAP_EXTEND)
            if (row + 1) % 25 == 0:
                print(f"scored {row + 1}/{len(query_indices)} calibration queries",
                      flush=True)
        np.save(scores_path, scores_computed)
        manifest_path.write_text(json.dumps({
            "database_sha256": sha256(db_path),
            "matrix_sha256": sha256(matrix_path),
            "n_targets": len(ids), "query_ids": query_ids,
            "scores_sha256": sha256(scores_path)}, indent=2) + "\n")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("database_sha256") != sha256(db_path)
            or manifest.get("matrix_sha256") != sha256(matrix_path)
            or manifest.get("n_targets") != len(ids)
            or manifest.get("scores_sha256") != sha256(scores_path)):
        raise ValueError("score manifest does not match the target database and matrix")
    if len(manifest["query_ids"]) != len(set(manifest["query_ids"])):
        raise ValueError("score manifest contains duplicate query IDs")
    matrix_rows = {query_id: row for row, query_id in enumerate(manifest["query_ids"])}
    if any(query_id not in matrix_rows for query_id in query_ids):
        raise ValueError("score matrix lacks a calibration query")
    scores = np.asarray(np.load(scores_path)[[matrix_rows[q] for q in query_ids]],
                        dtype=np.float64)
    tm_path = Path(args.tm_matrix) if args.tm_matrix else None
    if tm_path:
        tm_manifest = json.loads(Path(args.tm_manifest).read_text(encoding="utf-8"))
        if (tm_manifest.get("database_sha256") != sha256(db_path)
                or tm_manifest.get("tm_sha256") != sha256(tm_path)
                or tm_manifest.get("n_targets") != len(ids)
                or tm_manifest.get("query_ids") != query_ids):
            raise ValueError("structural-audit manifest does not match this database and panel")
    tm = (np.asarray(np.load(tm_path), dtype=np.float64).copy() if tm_path
          else np.full(scores.shape, np.nan))
    if scores.shape != (len(query_indices), len(ids)) or tm.shape != scores.shape:
        raise ValueError("score and structural-audit matrices do not match the panel")
    audit_file = workdir / "new_usalign.tsv"
    if audit_file.exists():
        with audit_file.open(newline="") as handle:
            index = {chain_id: row for row, chain_id in enumerate(query_ids)}
            target_index = {chain_id: column for column, chain_id in enumerate(ids)}
            for record in csv.DictReader(handle, delimiter="\t"):
                if record["query_id"] in index and record["target_id"] in target_index:
                    tm[index[record["query_id"]], target_index[record["target_id"]]] = .5 * (
                        float(record["tm_query"]) + float(record["tm_target"]))
    model, calibration, audit = fit_from_matrices(
        entries, tokens, ids, np.asarray(clusters), family, clan,
        query_indices, scores, tm, audit_file=audit_file, usalign=args.usalign,
        censored=not args.uncensored)
    target_digest = database_fingerprint(entries, clusters, n_codes)
    context_digest = calibration_context_fingerprint(
        target_digest, matrix, GAP_OPEN, GAP_EXTEND)
    params = {"scheme": EV.SCHEME, "score_definition": "raw-sw",
              "n_chains": len(ids), "n_clusters": audit["n_clusters"],
              "target_database_fingerprint": target_digest,
              "calibration_context_fingerprint": context_digest,
              "validation_status": "audited-e0-le-100",
              "target_aware_gumbel": model,
              "empirical_calibration": calibration,
              "audit": audit,
              "provenance": {"database_sha256": sha256(db_path),
                             "labels_sha256": sha256(args.labels),
                             "scores_sha256": sha256(scores_path),
                             "tm_matrix_sha256": (sha256(tm_path) if tm_path else None),
                             "tm_manifest_sha256": (sha256(args.tm_manifest)
                                                    if args.tm_manifest else None),
                             "query_indices_sha256": (sha256(args.query_indices)
                                                      if args.query_indices else None),
                             "new_usalign_sha256": (sha256(audit_file)
                                                     if audit_file.exists() else None)}}
    EV.validate_parameters(params, audit["n_clusters"], target_digest,
                           context_digest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}: {audit['n_queries']} queries, "
          f"{audit['n_clusters']} clusters, {audit['n_audited_pairs']} audited pairs")
    return params


if __name__ == "__main__":
    main()
