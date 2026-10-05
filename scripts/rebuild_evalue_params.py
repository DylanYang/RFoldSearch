#!/usr/bin/env python3
"""Fit database-specific cluster-max Gumbel parameters for a token database."""

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
from rfoldseek.database import (calibration_context_fingerprint,
                                database_fingerprint, load_database, short_id)
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


def read_query_panel(path):
    if path is None:
        return None
    query_ids = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = first_present(row, ("query_id", "chain_id", "entry_id", "id"))
            if not query_id:
                raise ValueError("query panel needs a query_id column")
            query_ids.append(query_id)
    if not query_ids:
        raise ValueError("query panel is empty")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query panel contains duplicate query ids")
    return query_ids


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
            tm_mean = first_present(row, ("tm_mean", "tm_symmetric", "tm_sym"))
            is_related = marked in {"1", "true", "yes", "y"}
            has_result = marked in {"0", "1", "false", "true", "no", "yes", "n", "y"}
            if tm_query and tm_target:
                has_result = True
                is_related = is_related or 0.5 * (
                    float(tm_query) + float(tm_target)) >= tm_threshold
            elif tm_mean:
                has_result = True
                is_related = is_related or float(tm_mean) >= tm_threshold
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="clustered RFoldseek database")
    parser.add_argument("--labels", required=True,
                        help="TSV with chain ids and optional family/clan columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--matrix", default=None)
    parser.add_argument(
        "--model", choices=("query-conditional", "length-grid"),
        default="query-conditional",
        help="model to fit; length-grid is retained only as a legacy fallback",
    )
    parser.add_argument(
        "--query-panel", default=None,
        help="TSV selecting the balanced calibration queries; default uses all chains",
    )
    parser.add_argument("--related-pairs", default=None,
                        help="optional US-align adjudication TSV")
    parser.add_argument("--tm-threshold", default=0.45, type=float)
    parser.add_argument("--audit-ceiling", default=100.0, type=float,
                        help="require US-align results for fitted cluster leaders at or below this E0")
    parser.add_argument("--missing-audit-output", default=None,
                        help="unreviewed leaders; default is OUTPUT.missing-audit.tsv")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="write provisional parameters without enforcing structural audit coverage")
    parser.add_argument("--ridge", default=1e-3, type=float)
    parser.add_argument(
        "--query-fits-output", default=None,
        help="optional TSV of the per-query Gumbel fits used by the regression",
    )
    parser.add_argument("--lengths", default="20,30,40,60,80,120,160,220,300,500,1000,2000")
    parser.add_argument("--n-queries", default=24, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--len-ratio", default=None, type=float,
        help="optional legacy length-ratio filter; default uses the full database",
    )
    args = parser.parse_args()
    missing_output = (Path(args.missing_audit_output)
                      if args.missing_audit_output else
                      Path(args.output).with_name(
                          Path(args.output).stem + ".missing-audit.tsv"))
    if args.model == "query-conditional" and args.len_ratio is not None:
        parser.error("--len-ratio is only available with --model length-grid; "
                     "the deployed query-conditional model uses the full database")

    _, entries, tokens, chain_ids, n_codes, clusters = load_database(args.db)
    matrix = load_matrix(args.matrix)
    target_fingerprint = database_fingerprint(entries, clusters, n_codes)
    labels = read_labels(args.labels)
    family = np.asarray([labels.get(chain_id, {}).get("family", "")
                         for chain_id in chain_ids], dtype=object)
    clan = np.asarray([labels.get(chain_id, {}).get("clan", "")
                       for chain_id in chain_ids], dtype=object)

    audited, related = read_adjudication(args.related_pairs, args.tm_threshold)

    def exclude(first, second):
        same_family = (family[first] not in UNKNOWN and family[second] not in UNKNOWN
                       and family[first] == family[second])
        same_clan = (clan[first] not in UNKNOWN and clan[second] not in UNKNOWN
                     and clan[first] == clan[second])
        query_id, target_id = chain_ids[first], chain_ids[second]
        structural = ((query_id, target_id) in related
                      or (short_id(query_id, query_id), short_id(target_id, target_id))
                      in related)
        return same_family or same_clan or structural

    def is_audited(first, second):
        query_id, target_id = chain_ids[first], chain_ids[second]
        return ((query_id, target_id) in audited
                or (short_id(query_id, query_id), short_id(target_id, target_id))
                in audited)

    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    started = time.time()
    if args.model == "query-conditional":
        panel = read_query_panel(args.query_panel)
        aliases = {}
        for index, chain_id in enumerate(chain_ids):
            for alias in (chain_id, short_id(chain_id, chain_id)):
                if alias in aliases and aliases[alias] != index:
                    raise ValueError(f"ambiguous database chain id: {alias}")
                aliases[alias] = index
        if panel is None:
            query_indices = list(range(len(chain_ids)))
        else:
            missing = [query_id for query_id in panel if query_id not in aliases]
            if missing:
                raise ValueError(f"query-panel ids absent from database: {missing[:3]}")
            query_indices = [aliases[query_id] for query_id in panel]
        fitted = EV.calibrate_clustermax_query_conditional(
            matrix, tokens, clusters, query_indices,
            gap_open=-2.0, gap_extend=-0.5, exclude=exclude, ridge=args.ridge,
            is_audited=(None if args.allow_unreviewed else is_audited),
            audit_evalue_ceiling=(None if args.allow_unreviewed
                                  else args.audit_ceiling),
        )
        missing_audit = fitted.pop("missing_audit", [])
        if missing_audit:
            missing_output.parent.mkdir(parents=True, exist_ok=True)
            with missing_output.open("w", newline="") as handle:
                fields = ("query_id", "target_id", "cluster_id",
                          "raw_score", "raw_evalue")
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for row in missing_audit:
                    writer.writerow({
                        "query_id": chain_ids[row["query_index"]],
                        "target_id": chain_ids[row["target_index"]],
                        "cluster_id": row["cluster"],
                        "raw_score": row["raw_score"],
                        "raw_evalue": row["raw_evalue"],
                    })
            raise SystemExit(
                f"{len(missing_audit)} fitted cluster leaders at "
                f"E0<={args.audit_ceiling:g} lack US-align adjudication; "
                f"wrote {missing_output}")
        query_fits = fitted.pop("query_fits")
        for row in query_fits:
            row["query_id"] = chain_ids[row.pop("query_index")]
        if args.query_fits_output:
            path = Path(args.query_fits_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=query_fits[0].keys(),
                                        delimiter="\t")
                writer.writeheader()
                writer.writerows(query_fits)
    else:
        fitted = EV.calibrate_clustermax_length_dependent(
            matrix, tokens, clusters, gap_open=-2.0, gap_extend=-0.5,
            lengths=lengths, n_queries=args.n_queries, seed=args.seed,
            exclude=exclude, len_ratio=args.len_ratio,
        )
    params = {
        "scheme": ("cluster-max-query-conditional"
                   if args.model == "query-conditional" else "cluster-max"),
        "gap_open": -2.0,
        "gap_extend": -0.5,
        "seed": args.seed,
        "len_ratio": args.len_ratio,
        "calibration_database": str(Path(args.db).resolve()),
        "n_clusters": fitted["n_clusters"],
        "n_chains": fitted["n_chains"],
        "target_database_fingerprint": target_fingerprint,
        "calibration_context_fingerprint": calibration_context_fingerprint(
            target_fingerprint, matrix, -2.0, -0.5),
        "search_scope": ("full_database" if args.len_ratio is None
                         else f"length_ratio_{args.len_ratio:g}"),
        "clusters": "stored cluster labels in the RFoldseek database",
        "null": "cluster maxima over real database chains; same annotated "
                "family or clan and adjudicated structural relatives excluded; "
                "cluster leaders are selected after exclusions",
    }
    if args.model == "query-conditional":
        params["query_parameter_model"] = fitted["query_parameter_model"]
        params["n_calibration_queries"] = len(query_fits)
        params["validation_status"] = (
            "provisional-incomplete-structural-audit" if args.allow_unreviewed
            else "background-audit-complete")
        if not args.allow_unreviewed:
            params["structural_audit_ceiling"] = args.audit_ceiling
    else:
        params["skipped_lengths"] = fitted["skipped_lengths"]
        params["length_table"] = fitted["length_table"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(params, indent=2) + "\n")
    missing_output.unlink(missing_ok=True)
    print(f"wrote {output} in {time.time() - started:.0f}s "
          f"({fitted['n_clusters']} clusters, {fitted['n_chains']} chains)")


if __name__ == "__main__":
    main()
