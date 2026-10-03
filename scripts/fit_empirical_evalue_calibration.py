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
    parser.add_argument("--query-panel", default=None,
                        help="TSV of all calibration query_id/family pairs, including "
                             "queries with no background hit in the fitted range")
    parser.add_argument("--raw-evalue-ceiling", type=float, default=None,
                        help="largest initial E-value included in a bounded fit")
    parser.add_argument(
        "--background-scaling", choices=("none", "annotated-to-full"),
        default="none",
        help="multiply family/query weights by N_D/B_q for annotated backgrounds",
    )
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
    with open(args.params) as handle:
        params = json.load(handle)
    fitted_n = params.get("n_clusters")
    if fitted_n is not None and int(fitted_n) != n_clusters:
        raise ValueError(
            f"parametric E-value model has {fitted_n} clusters, "
            f"but the target database has {n_clusters}")

    panel = None
    if args.query_panel:
        panel = {}
        with open(args.query_panel, newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                query_id, family = row["query_id"].strip(), row["family"].strip()
                if not query_id or not family:
                    raise ValueError("query panel needs nonempty query_id and family")
                if query_id in panel:
                    raise ValueError(f"duplicate calibration query: {query_id}")
                bq = row.get("background_clusters_before_structural_exclusion", "").strip()
                panel_n = row.get("n_database_clusters", "").strip()
                scope = row.get("background_scope", "").strip()
                audit_complete = row.get("structural_audit_complete", "").strip()
                audit_ceiling = row.get("structural_audit_ceiling", "").strip()
                panel[query_id] = {
                    "family": family,
                    "bq": int(bq) if bq else None,
                    "n_clusters": int(panel_n) if panel_n else None,
                    "background_scope": scope or None,
                    "audit_complete": (truthy(audit_complete)
                                       if audit_complete else None),
                    "audit_ceiling": (float(audit_ceiling)
                                      if audit_ceiling else None),
                }
        if not panel:
            raise ValueError("query panel is empty")
    if args.background_scaling == "annotated-to-full" and panel is None:
        parser.error("annotated-to-full scaling requires --query-panel with B_q metadata")
    if (args.background_scaling == "annotated-to-full"
            and any(metadata["background_scope"] != "annotated"
                    for metadata in panel.values())):
        parser.error("annotated-to-full scaling requires query metadata produced "
                     "with --background-scope annotated")

    rows = []
    with open(args.background, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if truthy(row.get("structurally_related", "")):
                continue
            if (args.raw_evalue_ceiling is not None
                    and float(row["raw_evalue"]) > args.raw_evalue_ceiling):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit("no background rows remain after filtering")

    values = [float(row["raw_evalue"]) for row in rows]
    explicit = [row.get("weight", "").strip() for row in rows]
    if all(explicit):
        if panel is not None or args.background_scaling != "none":
            parser.error("explicit weights cannot be combined with --query-panel "
                         "or --background-scaling")
        weights = [float(value) for value in explicit]
        weighting = "explicit"
    elif any(explicit):
        raise SystemExit("weight must be present on every row or omitted from every row")
    else:
        if any(not row.get("query_id", "").strip() for row in rows):
            raise SystemExit("background TSV needs weight or query_id on every row")
        queries = sorted(panel) if panel is not None else sorted(
            {row["query_id"] for row in rows})
        if panel is not None:
            unknown = {row["query_id"] for row in rows} - set(panel)
            if unknown:
                raise ValueError(f"background queries missing from panel: {sorted(unknown)[:3]}")
        have_family = panel is not None or all(
            row.get("family", "").strip() for row in rows)
        if have_family:
            family_queries = defaultdict(set)
            if panel is not None:
                for query_id, metadata in panel.items():
                    family_queries[metadata["family"]].add(query_id)
                for row in rows:
                    expected_family = panel[row["query_id"]]["family"]
                    if row.get("family", "").strip() not in ("", expected_family):
                        raise ValueError(f"family mismatch for {row['query_id']}")
            else:
                for row in rows:
                    family_queries[row["family"]].add(row["query_id"])
            weights = []
            for row in rows:
                family = (panel[row["query_id"]]["family"]
                          if panel is not None else row["family"])
                weight = 1.0 / (len(family_queries) * len(family_queries[family]))
                if args.background_scaling == "annotated-to-full":
                    if panel is None:
                        parser.error("annotated-to-full scaling requires --query-panel "
                                     "with per-query B_q metadata")
                    metadata = panel[row["query_id"]]
                    if metadata["bq"] is None or metadata["bq"] <= 0:
                        raise ValueError(f"missing or invalid B_q for {row['query_id']}")
                    if (metadata["n_clusters"] is not None
                            and metadata["n_clusters"] != n_clusters):
                        raise ValueError(
                            f"database cluster count mismatch for {row['query_id']}")
                    weight *= n_clusters / metadata["bq"]
                weights.append(weight)
            weighting = "equal-family, equal-query-within-family"
            if args.background_scaling == "annotated-to-full":
                weighting += ", multiplied by N_D/B_q"
        else:
            weights = [1.0 / len(queries)] * len(rows)
            weighting = "equal-query"
    calibration = fit_empirical_calibration(values, weights, n_clusters)
    calibration["weighting"] = weighting
    if panel is not None:
        calibration["n_calibration_queries"] = len(panel)
        calibration["n_families"] = len({
            metadata["family"] for metadata in panel.values()})
    calibration["background_scaling"] = args.background_scaling
    audit_metadata = ([metadata["audit_complete"] for metadata in panel.values()]
                      if panel is not None else [])
    if audit_metadata and all(value is True for value in audit_metadata):
        calibration["validation_status"] = "background-audit-complete"
        ceilings = {metadata["audit_ceiling"] for metadata in panel.values()}
        if len(ceilings) != 1 or None in ceilings:
            raise ValueError("query metadata has inconsistent structural audit ceilings")
        audit_ceiling = ceilings.pop()
        if (args.raw_evalue_ceiling is None
                or args.raw_evalue_ceiling > audit_ceiling):
            raise ValueError(
                "empirical fit ceiling must not exceed the structural audit ceiling")
        calibration["structural_audit_ceiling"] = audit_ceiling
    else:
        calibration["validation_status"] = "structural-audit-not-recorded"
    if args.raw_evalue_ceiling is not None:
        calibration["candidate_raw_evalue_ceiling"] = args.raw_evalue_ceiling
    params["empirical_calibration"] = calibration
    with open(args.output, "w") as handle:
        json.dump(params, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.output} ({calibration['n_observations']} background "
          f"observations, {n_clusters} clusters, {weighting})")


if __name__ == "__main__":
    main()
