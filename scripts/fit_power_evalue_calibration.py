#!/usr/bin/env python3
"""Attach a database-specific power-law correction to E-value parameters.

Stage 3 of ``rfoldsearch calibrate``; run it on its own to inspect or rerun
this stage.

The input TSV is produced by a calibration benchmark and must contain
``raw_evalue``. It may supply an explicit ``weight`` per row. Otherwise it must
contain ``query_id`` and weights are derived per query; when ``family`` is also
present, families are weighted equally. Rows marked true in an optional
``structurally_related`` column are excluded from the background.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfoldsearch import calibrate as C
from rfoldsearch.database import load_database


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
    panel = C.read_power_panel(args.query_panel) if args.query_panel else None
    try:
        params = C.fit_power_params(
            params, C.read_background(args.background, args.raw_evalue_ceiling),
            n_clusters, panel=panel, raw_evalue_ceiling=args.raw_evalue_ceiling,
            background_scaling=args.background_scaling)
    except C.CalibrationInputError as error:
        parser.error(str(error))
    except ValueError as error:
        raise SystemExit(str(error))
    with open(args.output, "w") as handle:
        json.dump(params, handle, indent=2)
        handle.write("\n")
    calibration = params["power_calibration"]
    print(f"wrote {args.output} ({calibration['n_observations']} background "
          f"observations, {n_clusters} clusters, {calibration['weighting']})")


if __name__ == "__main__":
    main()
