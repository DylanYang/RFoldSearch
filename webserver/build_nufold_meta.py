"""Build the display sidecar for the NuFold database: Rfam family per model.

A NuFold model id names an RNAcentral sequence and a taxon
(``URS000046088B_1352_rank_1.pdb``) and nothing else, so every hit used to be
described as "Predicted structure (NuFold)". RNAcentral publishes the Rfam
family assigned to each sequence; this joins that table onto the database and
writes ``<db>.meta.json`` next to it, in the format ``app/engine.py`` reads:

    {"source": ..., "built": ..., "entries": {id: {"desc": "tRNA (RF00005)"}}}

The organism is not stored here: ``app/engine.py`` resolves it from the taxon
id in the model name (``app/nufold_taxa.json``).

    python build_nufold_meta.py --db nufold_db_full.pt \
        --rfam rfam_annotations.tsv.gz        # from
    # https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/rfam/
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re

URS = re.compile(r"^(URS[0-9A-F]{10})_")


def best_family(path: str, wanted: set) -> dict:
    """URS id -> (Rfam accession, family name) of the lowest-E-value hit."""
    best = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[0] not in wanted:
                continue
            try:
                ev = float(f[3])
            except ValueError:
                continue
            if f[0] not in best or ev < best[f[0]][0]:
                best[f[0]] = (ev, f[1], f[8])
    return {k: (v[1], v[2]) for k, v in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="NuFold database (.pt)")
    ap.add_argument("--rfam", required=True, help="RNAcentral rfam_annotations.tsv.gz")
    ap.add_argument("--out", help="default: <db>.meta.json")
    args = ap.parse_args()

    import torch
    ids = [e["id"] for e in torch.load(args.db, map_location="cpu",
                                        weights_only=False)["entries"]]
    wanted = {m.group(1) for i in ids for m in [URS.match(i)] if m}
    fam = best_family(args.rfam, wanted)

    entries = {}
    for i in ids:                                  # keep the database order
        m = URS.match(i)
        hit = fam.get(m.group(1)) if m else None
        entries[i] = {"desc": f"{hit[1]} ({hit[0]})"} if hit else {}
    out = args.out or args.db.rsplit(".", 1)[0] + ".meta.json"
    with open(out, "w") as fh:
        json.dump({"source": "NuFold predicted structures; Rfam families from "
                             "RNAcentral rfam_annotations",
                   "built": dt.date.today().isoformat(),
                   "entries": entries}, fh, separators=(",", ":"))
    print(f"{len(ids):,} models, {len(wanted):,} RNAcentral sequences, "
          f"{len(fam):,} with an Rfam family -> {out}")


if __name__ == "__main__":
    main()
