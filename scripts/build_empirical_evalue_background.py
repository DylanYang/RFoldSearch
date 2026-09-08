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
from rfoldseek.database import load_database, short_id
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


def read_related(path, tm_threshold):
    related = set()
    if path is None:
        return related
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query = first_present(row, ("query_id", "query"))
            target = first_present(row, ("target_id", "target"))
            marked = first_present(row, ("structurally_related",)).lower()
            tm = first_present(row, ("tm_symmetric", "tm_symmetric_min", "tm_sym"))
            is_related = marked in {"1", "true", "yes", "y"}
            if tm:
                is_related = is_related or float(tm) >= tm_threshold
            if query and target and is_related:
                related.add((query, target))
                related.add((target, query))
    return related


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
    parser.add_argument("--len-ratio", default=3.0, type=float)
    parser.add_argument("--related-pairs", default=None,
                        help="optional US-align adjudication TSV")
    parser.add_argument("--tm-threshold", default=0.45, type=float)
    args = parser.parse_args()

    _, _, targets, target_ids, _, clusters = load_database(args.db)
    _, _, queries, query_ids, _, _ = load_database(args.queries)
    matrix = load_matrix(args.matrix)
    with open(args.params) as handle:
        params = json.load(handle)
    labels = read_labels(args.labels)
    related = read_related(args.related_pairs, args.tm_threshold)
    n_clusters = len(set(clusters))
    target_index = {chain_id: index for index, chain_id in enumerate(target_ids)}
    members = {}
    for index, cluster in enumerate(clusters):
        members.setdefault(str(cluster), []).append(index)

    fields = ("query_id", "family", "clan", "cluster_id", "target_id",
              "raw_score", "raw_evalue", "structurally_related")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for query_number, (query_id, query) in enumerate(zip(query_ids, queries), 1):
            query_label = labels.get(query_id, {"family": "", "clan": ""})
            self_cluster = None
            if query_id in target_index:
                self_cluster = str(clusters[target_index[query_id]])
            mu, beta = EV.clustermax_params_for_length(params, len(query))
            for cluster, indices in members.items():
                if cluster == self_cluster:
                    continue
                best = None
                for index in indices:
                    target_id, target = target_ids[index], targets[index]
                    target_label = labels.get(target_id, {"family": "", "clan": ""})
                    if known_equal(query_label["family"], target_label["family"]):
                        continue
                    if known_equal(query_label["clan"], target_label["clan"]):
                        continue
                    if (len(target) > args.len_ratio * len(query) or
                            len(query) > args.len_ratio * len(target)):
                        continue
                    score = float(smith_waterman_numba(
                        query, target, matrix, GAP_OPEN, GAP_EXTEND
                    ))
                    if best is None or score > best[0]:
                        best = (score, target_id)
                if best is None:
                    continue
                score, target_id = best
                raw_evalue = float(EV.evalue_clustermax(
                    score, mu, beta, n_clusters
                ))
                writer.writerow({
                    "query_id": query_id,
                    "family": query_label["family"],
                    "clan": query_label["clan"],
                    "cluster_id": cluster,
                    "target_id": target_id,
                    "raw_score": score,
                    "raw_evalue": raw_evalue,
                    "structurally_related": str(
                        (query_id, target_id) in related
                    ).lower(),
                })
                written += 1
            if query_number % 10 == 0:
                print(f"processed {query_number}/{len(queries)} queries", flush=True)
    print(f"wrote {written} cluster-max background rows to {output} "
          f"({len(queries)} queries, {n_clusters} target clusters)")


if __name__ == "__main__":
    main()
