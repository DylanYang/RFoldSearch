"""Build the display sidecar for the NuFold database from NuFold's own table.

A NuFold model id names an RNAcentral sequence and a taxon
(``URS000046088B_1352_rank_1.pdb``) and nothing else. The NuFold database
publishes a statistics table with a description, the Rfam family, the sequence
length, the MSA depth and the mean pLDDT of every entry; this joins it onto the
search database and writes ``<db>.meta.json`` next to it, in the format
``app/engine.py`` reads:

    {"source": ..., "built": ..., "notice": ...,
     "entries": {id: {"desc": "Homo sapiens tRNA", "rfam": "RF00005",
                      "plddt": 78.0, "msa_depth": 836}}}

The organism is not stored here: ``app/engine.py`` resolves it from the taxon
id in the model name (``app/nufold_taxa.json``).

    python build_nufold_meta.py --db nufold_db_full.pt --table data.csv
    # table: https://nufold.kiharalab.org/downloads/nufold/nufold-<date>-data.tar.gz
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re

URS = re.compile(r"^(URS[0-9A-F]{10}_\d+)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="NuFold database (.pt)")
    ap.add_argument("--table", required=True, help="NuFold data.csv")
    ap.add_argument("--out", help="default: <db>.meta.json")
    ap.add_argument("--release", default="NuFold database", help="release label for the source line")
    ap.add_argument("--notice", help="warning shown next to this database on the site")
    args = ap.parse_args()

    import torch
    ids = [e["id"] for e in torch.load(args.db, map_location="cpu",
                                        weights_only=False)["entries"]]
    with open(args.table, newline="") as fh:
        rows = {r["urs_id"]: r for r in csv.DictReader(fh)}

    entries, hit = {}, 0
    for i in ids:                                  # keep the database order
        m = URS.match(i)
        r = rows.get(m.group(1)) if m else None
        if not r:
            entries[i] = {}
            continue
        hit += 1
        entries[i] = {"desc": r["description"], "rfam": r["rfam_family"],
                      "plddt": float(r["avg_plddt"]),
                      "msa_depth": int(r["msa_depth"])}
    out = args.out or args.db.rsplit(".", 1)[0] + ".meta.json"
    with open(out, "w") as fh:
        json.dump({"source": f"NuFold predicted structures ({args.release})",
                   "built": dt.date.today().isoformat(),
                   **({"notice": args.notice} if args.notice else {}),
                   "entries": entries}, fh, separators=(",", ":"))
    print(f"{len(ids):,} models, {hit:,} found in the table -> {out}")


if __name__ == "__main__":
    main()
