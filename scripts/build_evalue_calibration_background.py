#!/usr/bin/env python3
"""Build cluster-max background observations for power E-value fitting.

Stage 2 of ``rfoldseek calibrate``; run it on its own to inspect or rerun
this stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfoldseek import calibrate as C
from rfoldseek.database import load_database
from rfoldseek.search import load_matrix


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
    parser.add_argument("--tm-threshold", default=C.TM_THRESHOLD, type=float)
    parser.add_argument("--audit-ceiling", default=C.AUDIT_CEILING, type=float,
                        help="require US-align results for retained leaders at or below this E0")
    parser.add_argument("--background-scope", choices=("all", "annotated"),
                        default="all", help="use all eligible clusters or only Rfam-annotated targets")
    parser.add_argument("--query-metadata-output", default=None,
                        help="one-row-per-query denominators; default is OUTPUT.queries.tsv")
    parser.add_argument("--missing-audit-output", default=None,
                        help="unreviewed leaders; default is OUTPUT.missing-audit.tsv")
    args = parser.parse_args()
    output = Path(args.output)
    metadata_output = (Path(args.query_metadata_output) if args.query_metadata_output
                       else output.with_name(output.stem + ".queries.tsv"))
    missing_output = (Path(args.missing_audit_output) if args.missing_audit_output
                      else output.with_name(output.stem + ".missing-audit.tsv"))

    _, target_entries, targets, target_ids, n_codes, clusters = load_database(args.db)
    _, _, queries, query_ids, _, _ = load_database(args.queries)
    with open(args.params) as handle:
        params = json.load(handle)
    audited, related = C.read_adjudication(args.related_pairs, args.tm_threshold)
    rows, metadata, missing = C.build_background(
        params, target_entries, targets, target_ids, n_codes, clusters,
        queries, query_ids, C.read_labels(args.labels), load_matrix(args.matrix),
        audited, related, audit_ceiling=args.audit_ceiling,
        background_scope=args.background_scope,
        log=lambda message: print(message, flush=True))
    if missing:
        C._write_tsv(missing_output, C.MISSING_FIELDS, missing)
        raise SystemExit(
            f"{len(missing)} cluster leaders at E0<={args.audit_ceiling:g} lack "
            f"US-align adjudication; wrote {missing_output}")
    C._write_tsv(output, C.BACKGROUND_FIELDS, rows)
    C._write_tsv(metadata_output, C.QUERY_METADATA_FIELDS, metadata)
    missing_output.unlink(missing_ok=True)
    print(f"wrote {len(rows)} cluster-max background rows to {output} and query "
          f"denominators to {metadata_output} ({len(queries)} queries, "
          f"{len(set(clusters))} target clusters)")


if __name__ == "__main__":
    main()
