#!/usr/bin/env python3
"""Build cluster-max background observations for empirical E-value fitting."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfoldseek import evalue as EV
from rfoldseek.database import (calibration_context_fingerprint,
                                database_fingerprint, load_database, short_id)
from rfoldseek.search import GAP_EXTEND, GAP_OPEN, load_matrix
from rfoldseek.smith_waterman import smith_waterman_numba


UNKNOWN = {"", "N/A", "NA", "None", "none", ".", "-"}


def first_present(row, names):
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def read_labels(path):
    labels = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chain_id = first_present(row, ("chain_id", "entry_id", "id",
                                           "query_id", "target_id"))
            if not chain_id:
                raise ValueError("labels TSV needs an id column")
            annotation = {
                "family": first_present(row, ("family", "rfam_acc", "rfam_family")),
                "clan": first_present(row, ("clan", "clan_acc", "rfam_clan")),
            }
            labels[chain_id] = annotation
            labels[short_id(chain_id, chain_id)] = annotation
    return labels


def read_adjudication(path, tm_threshold):
    audited, related = set(), set()
    if path is None:
        return audited, related
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query = first_present(row, ("query_id", "query"))
            target = first_present(row, ("target_id", "target"))
            marked = first_present(row, ("structurally_related",)).lower()
            tm_query = first_present(row, ("tm_query_normalized", "tm_query"))
            tm_target = first_present(row, ("tm_target_normalized", "tm_target"))
            tm = first_present(row, ("tm_mean", "tm_symmetric", "tm_sym"))
            is_related = marked in {"1", "true", "yes", "y"}
            has_result = marked in {"0", "1", "false", "true", "no", "yes", "n", "y"}
            if tm_query and tm_target:
                has_result = True
                is_related = is_related or 0.5 * (
                    float(tm_query) + float(tm_target)) >= tm_threshold
            elif tm:
                has_result = True
                is_related = is_related or float(tm) >= tm_threshold
            elif first_present(row, ("tm_symmetric_min",)):
                raise ValueError(
                    "tm_symmetric_min is not the mean TM-score; provide both "
                    "normalized scores or a tm_mean column")
            if query and target and has_result:
                aliases = {(query, target),
                           (short_id(query, query), short_id(target, target))}
                audited.update(aliases)
                audited.update((second, first) for first, second in aliases)
                if is_related:
                    related.update(aliases)
                    related.update((second, first) for first, second in aliases)
    return audited, related


def known_equal(first, second):
    return first not in UNKNOWN and second not in UNKNOWN and first == second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="target RFoldseek database")
    parser.add_argument("--queries", required=True,
                        help="independent calibration-query RFoldseek database")
    parser.add_argument("--params", required=True,
                        help="uncorrected cluster-max parameter JSON")
    parser.add_argument("--labels", required=True,
                        help="TSV labels for query and target chain ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--related-pairs", default=None,
                        help="optional US-align adjudication TSV")
    parser.add_argument("--tm-threshold", default=0.45, type=float)
    parser.add_argument("--audit-ceiling", default=100.0, type=float,
                        help="require US-align results for retained leaders at or below this E0")
    parser.add_argument("--background-scope", choices=("all", "annotated"),
                        default="all", help="use all eligible clusters or only Rfam-annotated targets")
    parser.add_argument("--query-metadata-output", default=None,
                        help="one-row-per-query denominators; default is OUTPUT.queries.tsv")
    parser.add_argument("--missing-audit-output", default=None,
                        help="unreviewed leaders; default is OUTPUT.missing-audit.tsv")
    args = parser.parse_args()

    _, target_entries, targets, target_ids, n_codes, clusters = load_database(args.db)
    _, _, queries, query_ids, _, _ = load_database(args.queries)
    matrix = load_matrix(args.matrix)
    with open(args.params) as handle:
        params = json.load(handle)
    labels = read_labels(args.labels)
    audited, related = read_adjudication(args.related_pairs, args.tm_threshold)
    n_clusters = len(set(clusters))
    target_fingerprint = database_fingerprint(target_entries, clusters, n_codes)
    fitted_target = params.get("target_database_fingerprint")
    if fitted_target is not None and fitted_target != target_fingerprint:
        raise ValueError("parameter file belongs to a different target database")
    fitted_context = params.get("calibration_context_fingerprint")
    actual_context = calibration_context_fingerprint(
        target_fingerprint, matrix, GAP_OPEN, GAP_EXTEND)
    if fitted_context is not None and fitted_context != actual_context:
        raise ValueError("parameter file belongs to a different scoring context")
    target_index = {chain_id: index for index, chain_id in enumerate(target_ids)}
    members = {}
    for index, cluster in enumerate(clusters):
        members.setdefault(str(cluster), []).append(index)

    fields = ("query_id", "family", "clan", "cluster_id", "target_id",
              "raw_score", "raw_evalue", "structurally_related",
              "n_database_clusters",
              "background_clusters_before_structural_exclusion")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    metadata_output = (Path(args.query_metadata_output) if args.query_metadata_output
                       else output.with_name(output.stem + ".queries.tsv"))
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_partial = metadata_output.with_name(metadata_output.name + ".partial")
    missing_output = (Path(args.missing_audit_output) if args.missing_audit_output
                      else output.with_name(output.stem + ".missing-audit.tsv"))
    metadata_fields = ("query_id", "family", "clan", "background_scope",
                       "structural_audit_ceiling", "structural_audit_complete",
                       "n_database_clusters",
                       "background_clusters_before_structural_exclusion",
                       "n_retained_background_clusters")
    written = 0
    missing = []
    with partial.open("w", newline="") as handle, metadata_partial.open(
            "w", newline="") as metadata_handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        metadata_writer = csv.DictWriter(
            metadata_handle, fieldnames=metadata_fields, delimiter="\t")
        writer.writeheader()
        metadata_writer.writeheader()
        for query_number, (query_id, query) in enumerate(zip(query_ids, queries), 1):
            query_label = labels.get(query_id, {"family": "", "clan": ""})
            self_cluster = None
            if query_id in target_index:
                self_cluster = str(clusters[target_index[query_id]])
            mu, beta = EV.clustermax_params_for_query(params, query)
            query_rows = []
            background_clusters_before_exclusion = 0
            for cluster, indices in members.items():
                if cluster == self_cluster:
                    continue
                scored = []
                for index in indices:
                    target_id, target = target_ids[index], targets[index]
                    target_label = labels.get(target_id, {"family": "", "clan": ""})
                    if (args.background_scope == "annotated"
                            and target_label["family"] in UNKNOWN):
                        continue
                    if known_equal(query_label["family"], target_label["family"]):
                        continue
                    if known_equal(query_label["clan"], target_label["clan"]):
                        continue
                    score = float(smith_waterman_numba(
                        query, target, matrix, GAP_OPEN, GAP_EXTEND
                    ))
                    scored.append((score, target_id))
                if not scored:
                    continue
                background_clusters_before_exclusion += 1
                best = None
                for score, target_id in sorted(scored, reverse=True):
                    aliases = ((query_id, target_id),
                               (short_id(query_id, query_id),
                                short_id(target_id, target_id)))
                    if any(pair in related for pair in aliases):
                        continue
                    raw_evalue = float(EV.evalue_clustermax(
                        score, mu, beta, n_clusters))
                    if (raw_evalue <= args.audit_ceiling
                            and not any(pair in audited for pair in aliases)):
                        missing.append({
                            "query_id": query_id,
                            "target_id": target_id,
                            "cluster_id": cluster,
                            "raw_evalue": raw_evalue,
                        })
                        break
                    best = (score, target_id, raw_evalue)
                    break
                if best is None:
                    continue
                score, target_id, raw_evalue = best
                query_rows.append({
                    "query_id": query_id,
                    "family": query_label["family"],
                    "clan": query_label["clan"],
                    "cluster_id": cluster,
                    "target_id": target_id,
                    "raw_score": score,
                    "raw_evalue": raw_evalue,
                    "structurally_related": "false",
                    "n_database_clusters": n_clusters,
                    "background_clusters_before_structural_exclusion": None,
                })
            for row in query_rows:
                row["background_clusters_before_structural_exclusion"] = (
                    background_clusters_before_exclusion)
                writer.writerow(row)
                written += 1
            metadata_writer.writerow({
                "query_id": query_id,
                "family": query_label["family"],
                "clan": query_label["clan"],
                "background_scope": args.background_scope,
                "structural_audit_ceiling": args.audit_ceiling,
                "structural_audit_complete": "true",
                "n_database_clusters": n_clusters,
                "background_clusters_before_structural_exclusion": (
                    background_clusters_before_exclusion),
                "n_retained_background_clusters": len(query_rows),
            })
            if query_number % 10 == 0:
                print(f"processed {query_number}/{len(queries)} queries", flush=True)
    if missing:
        missing_output.parent.mkdir(parents=True, exist_ok=True)
        with missing_output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("query_id", "target_id", "cluster_id",
                                    "raw_evalue"), delimiter="\t")
            writer.writeheader()
            writer.writerows(missing)
        partial.unlink(missing_ok=True)
        metadata_partial.unlink(missing_ok=True)
        raise SystemExit(
            f"{len(missing)} cluster leaders at E0<={args.audit_ceiling:g} lack "
            f"US-align adjudication; wrote {missing_output}")
    partial.replace(output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_partial.replace(metadata_output)
    missing_output.unlink(missing_ok=True)
    print(f"wrote {written} cluster-max background rows to {output} and query "
          f"denominators to {metadata_output} ({len(queries)} queries, "
          f"{n_clusters} target clusters)")


if __name__ == "__main__":
    main()
