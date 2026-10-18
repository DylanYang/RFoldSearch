"""Export the Rfam 15.1 PDB-chain labels used by experimental E-value fitting."""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import re

from rfoldsearch.database import load_database


def labels_for_ids(ids, regions, memberships):
    clan = {}
    with gzip.open(memberships, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                clan[fields[1]] = fields[0]
    best = {}
    with gzip.open(regions, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11 or fields[10] != "1":
                continue
            family, pdb_id, chain = fields[0], fields[1].lower(), fields[2]
            evalue = float(fields[6])
            key = (pdb_id, chain)
            if key not in best or evalue < best[key][0]:
                best[key] = (evalue, family)
    output = []
    for chain_id in ids:
        match = re.match(r"^PDB_([^_]+)_\d+_(.+)$", chain_id)
        key = ((match.group(1).lstrip("0").lower(), match.group(2))
               if match else None)
        family = best.get(key, (math.inf, ""))[1] if key else ""
        output.append((chain_id, family, clan.get(family, "")))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--pdb-regions", required=True)
    parser.add_argument("--clan-memberships", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    _, _, _, ids, _, _ = load_database(args.db)
    rows = labels_for_ids(ids, args.pdb_regions, args.clan_memberships)
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("chain_id", "family", "clan"))
        writer.writerows(rows)
    print(f"wrote {len(rows)} chain labels, {sum(bool(row[1]) for row in rows)} annotated")


if __name__ == "__main__":
    main()
