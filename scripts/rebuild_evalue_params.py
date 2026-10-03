#!/usr/bin/env python3
"""Fit length-dependent cluster-max Gumbel parameters for a token database."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfoldseek import evalue as EV
from rfoldseek.database import load_database, short_id
from rfoldseek.search import load_matrix


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="clustered RFoldseek database")
    parser.add_argument("--labels", required=True,
                        help="TSV with chain ids and optional family/clan columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--lengths", default="20,30,40,60,80,120,160,220,300,500,1000,2000")
    parser.add_argument("--n-queries", default=24, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--len-ratio", default=None, type=float,
        help="optional legacy length-ratio filter; default uses the full database",
    )
    args = parser.parse_args()

    _, _, tokens, chain_ids, _, clusters = load_database(args.db)
    matrix = load_matrix(args.matrix)
    labels = read_labels(args.labels)
    family = np.asarray([labels.get(chain_id, {}).get("family", "")
                         for chain_id in chain_ids], dtype=object)
    clan = np.asarray([labels.get(chain_id, {}).get("clan", "")
                       for chain_id in chain_ids], dtype=object)

    def exclude(first, second):
        same_family = (family[first] not in UNKNOWN and family[second] not in UNKNOWN
                       and family[first] == family[second])
        same_clan = (clan[first] not in UNKNOWN and clan[second] not in UNKNOWN
                     and clan[first] == clan[second])
        return same_family or same_clan

    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    started = time.time()
    fitted = EV.calibrate_clustermax_length_dependent(
        matrix, tokens, clusters, gap_open=-2.0, gap_extend=-0.5,
        lengths=lengths, n_queries=args.n_queries, seed=args.seed,
        exclude=exclude, len_ratio=args.len_ratio,
    )
    params = {
        "scheme": "cluster-max",
        "gap_open": -2.0,
        "gap_extend": -0.5,
        "seed": args.seed,
        "len_ratio": args.len_ratio,
        "calibration_database": str(Path(args.db).resolve()),
        "n_clusters": fitted["n_clusters"],
        "n_chains": fitted["n_chains"],
        "skipped_lengths": fitted["skipped_lengths"],
        "length_table": fitted["length_table"],
        "clusters": "stored cluster labels in the RFoldseek database",
        "null": "cluster maxima over real database chains; same annotated "
                "family or clan excluded; nothing censored",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {output} in {time.time() - started:.0f}s "
          f"({fitted['n_clusters']} clusters, {fitted['n_chains']} chains)")


if __name__ == "__main__":
    main()
