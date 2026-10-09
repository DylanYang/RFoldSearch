#!/usr/bin/env python3
"""Build RFoldSearch calibration labels from Rfam cmscan output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_fasta_ids(paths: list[str]) -> list[str]:
    identifiers = []
    seen = set()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                identifier = line[1:].strip().split(maxsplit=1)[0]
                if not identifier:
                    raise ValueError(f"empty FASTA identifier in {path}")
                if identifier not in seen:
                    identifiers.append(identifier)
                    seen.add(identifier)
    return identifiers


def read_cmscan_hits(paths: list[str]) -> dict[str, dict[str, object]]:
    best = {}
    for path in paths:
        with open(path) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 19 or not fields[0].isdigit():
                    raise ValueError(
                        f"{path}:{line_number}: expected cmscan --fmt 2 output"
                    )
                if fields[18] != "!":
                    continue
                hit = {
                    "family": fields[2],
                    "clan": "" if fields[5] == "-" else fields[5],
                    "evalue": float(fields[17]),
                    "score": float(fields[16]),
                    "rank": int(fields[0]),
                }
                query_id = fields[3]
                order = (hit["evalue"], -hit["score"], hit["rank"])
                current = best.get(query_id)
                if current is None or order < current["order"]:
                    hit["order"] = order
                    best[query_id] = hit
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Convert Rfam cmscan --fmt 2 output into labels.tsv"
    )
    parser.add_argument(
        "--fasta", action="append", required=True,
        help="input FASTA; repeat for target and calibration sequences",
    )
    parser.add_argument(
        "--tblout", action="append", required=True,
        help="cmscan --fmt 2 table; repeat when scans were run separately",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    identifiers = read_fasta_ids(args.fasta)
    hits = read_cmscan_hits(args.tblout)
    known_ids = set(identifiers)
    unexpected = sorted(set(hits) - known_ids)
    if unexpected:
        preview = ", ".join(unexpected[:3])
        raise ValueError(
            f"{len(unexpected)} cmscan query ids are absent from the FASTA "
            f"inputs (for example: {preview})"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("chain_id", "family", "clan"), delimiter="\t"
        )
        writer.writeheader()
        for identifier in identifiers:
            hit = hits.get(identifier, {})
            writer.writerow({
                "chain_id": identifier,
                "family": hit.get("family", ""),
                "clan": hit.get("clan", ""),
            })

    print(
        f"wrote {len(identifiers)} rows to {output}; "
        f"{sum(identifier in hits for identifier in identifiers)} have Rfam labels"
    )


if __name__ == "__main__":
    main()
