#!/usr/bin/env python3
"""Fit database-specific cluster-max Gumbel parameters for a token database.

Stage 1 of ``rfoldsearch calibrate``; run it on its own to inspect or rerun
this stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfoldsearch import calibrate as C
from rfoldsearch.database import load_database
from rfoldsearch.search import load_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="clustered RFoldSearch database")
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
    parser.add_argument("--tm-threshold", default=C.TM_THRESHOLD, type=float)
    parser.add_argument("--audit-ceiling", default=C.AUDIT_CEILING, type=float,
                        help="require US-align results for fitted cluster leaders at or below this E0")
    parser.add_argument("--missing-audit-output", default=None,
                        help="unreviewed leaders; default is OUTPUT.missing-audit.tsv")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="write provisional parameters without enforcing structural audit coverage")
    parser.add_argument("--ridge", default=1e-3, type=float)
    parser.add_argument("--comp-bias-window", default=0, type=int,
                        help="composition-bias correction half-width (0 = off)")
    parser.add_argument(
        "--query-fits-output", default=None,
        help="optional TSV of the per-query Gumbel fits used by the regression",
    )
    parser.add_argument("--lengths", default=",".join(map(str, C.DEFAULT_LENGTHS)))
    parser.add_argument("--n-queries", default=24, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--len-ratio", default=None, type=float,
        help="optional legacy length-ratio filter; default uses the full database",
    )
    args = parser.parse_args()
    output = Path(args.output)
    missing_output = (Path(args.missing_audit_output) if args.missing_audit_output
                      else output.with_name(output.stem + ".missing-audit.tsv"))

    _, entries, tokens, chain_ids, n_codes, clusters = load_database(args.db)
    audited, related = C.read_adjudication(args.related_pairs, args.tm_threshold)
    started = time.time()
    try:
        params, query_fits, missing = C.fit_clustermax_params(
            args.db, entries, tokens, chain_ids, n_codes, clusters,
            C.read_labels(args.labels), load_matrix(args.matrix), audited, related,
            query_panel=C.read_query_panel(args.query_panel), model=args.model,
            audit_ceiling=args.audit_ceiling,
            allow_unreviewed=args.allow_unreviewed, ridge=args.ridge,
            lengths=[int(value) for value in args.lengths.split(",") if value],
            n_queries=args.n_queries, seed=args.seed, len_ratio=args.len_ratio,
            comp_bias_window=args.comp_bias_window)
    except C.CalibrationInputError as error:
        parser.error(str(error))
    if missing:
        C._write_tsv(missing_output, C.MISSING_FIELDS, missing)
        raise SystemExit(
            f"{len(missing)} fitted cluster leaders at "
            f"E0<={args.audit_ceiling:g} lack US-align adjudication; "
            f"wrote {missing_output}")
    if args.query_fits_output and query_fits:
        C._write_tsv(args.query_fits_output, list(query_fits[0]), query_fits)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(params, indent=2) + "\n")
    missing_output.unlink(missing_ok=True)
    print(f"wrote {output} in {time.time() - started:.0f}s "
          f"({params['n_clusters']} clusters, {params['n_chains']} chains)")


if __name__ == "__main__":
    main()
