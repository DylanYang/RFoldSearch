#!/usr/bin/env python3
"""Attach a database-specific empirical correction to E-value parameters.

The input TSV is produced by a calibration benchmark and must contain
``raw_evalue``. It may supply an explicit ``weight`` per row. Otherwise it must
contain ``query_id`` and weights are derived per query; when ``family`` is also
present, families are weighted equally. Rows marked true in an optional
``structurally_related`` column are excluded from the background.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfoldseek.database import load_database
from rfoldseek.evalue import fit_empirical_calibration


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="cluster-max E-value parameter JSON")
    parser.add_argument("--background", required=True,
                        help="TSV with raw_evalue and optional weight columns")
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", help="target database; cluster count is read from it")
    source.add_argument("--n-clusters", type=int,
                        help="cluster count, for calibration without a database file")
    args = parser.parse_args()

    if args.db:
        _, _, _, _, _, clusters = load_database(args.db)
        n_clusters = len(set(clusters))
    else:
        n_clusters = args.n_clusters

    rows = []
    with open(args.background, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if truthy(row.get("structurally_related", "")):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit("no background rows remain after filtering")

    values = [float(row["raw_evalue"]) for row in rows]
    explicit = [row.get("weight", "").strip() for row in rows]
    if all(explicit):
        weights = [float(value) for value in explicit]
        weighting = "explicit"
    elif any(explicit):
        raise SystemExit("weight must be present on every row or omitted from every row")
    else:
        if any(not row.get("query_id", "").strip() for row in rows):
            raise SystemExit("background TSV needs weight or query_id on every row")
        queries = sorted({row["query_id"] for row in rows})
        have_family = all(row.get("family", "").strip() for row in rows)
        if have_family:
            family_queries = defaultdict(set)
            for row in rows:
                family_queries[row["family"]].add(row["query_id"])
            weights = [1.0 / (len(family_queries) *
                              len(family_queries[row["family"]])) for row in rows]
            weighting = "equal-family, equal-query-within-family"
        else:
            weights = [1.0 / len(queries)] * len(rows)
            weighting = "equal-query"
    calibration = fit_empirical_calibration(values, weights, n_clusters)
    calibration["weighting"] = weighting
    with open(args.params) as handle:
        params = json.load(handle)
    params["empirical_calibration"] = calibration
    with open(args.output, "w") as handle:
        json.dump(params, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.output} ({calibration['n_observations']} background "
          f"observations, {n_clusters} clusters, {weighting})")


if __name__ == "__main__":
    main()
