"""Build the display sidecar for a search database: sequence + PDB annotation.

A database .pt stores tokens and backbone coordinates but no sequence and no
description, both of which the result page shows. This writes
``<db>.meta.json`` next to the database:

    {"source": ..., "built": ..., "entries": {id: {"seq": ..., "pdb": ...,
      "chain": ..., "auth_asym": ..., "desc": ..., "organism": ...,
      "title": ..., "resolution": ..., "method": ...}}}

Sequences come from the preprocessed JSONL the database was built from,
restricted to the masked residues that carry a token, so there is exactly one
letter per token and alignments map straight onto them. For ids
that name a PDB chain (``.../1ABC_D.cif``) the annotation is
fetched once from the RCSB GraphQL API; other ids get the sequence only.

    python build_metadata.py --db pdb_rna.pt --jsonl-root ~/pdb_rna \
        --source "PDB-RNA, all RCSB RNA chains" --rcsb
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict

import torch

PDB_CHAIN = re.compile(r"([0-9A-Za-z]{4})_([0-9A-Za-z]+)\.cif$")
RCSB = "https://data.rcsb.org/graphql"
# Queried per entry, not per instance: the PDB-RNA chains are named by
# label_asym_id but the RNAsolo/BGSU representatives by the author chain id,
# and the same letter can be a label id for one chain and an author id for
# another (5X8T: label W is a ribosomal protein, author W an rRNA). So only
# RNA entities are candidates, and when the label and author readings name
# different RNA chains the one whose sequence matches ours wins.
QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){
  rcsb_id struct{title} exptl{method} rcsb_entry_info{resolution_combined}
  polymer_entities{rcsb_polymer_entity{pdbx_description}
    entity_poly{rcsb_entity_polymer_type pdbx_seq_one_letter_code_can}
    rcsb_entity_source_organism{scientific_name}
    polymer_entity_instances{rcsb_polymer_entity_instance_container_identifiers{
      asym_id auth_asym_id}}}}}"""


def sequences(ids, jsonl_root):
    """id -> seq, reading each shard JSONL (dirname(id) + '.jsonl') once."""
    by_shard = defaultdict(set)
    for i in ids:
        by_shard[os.path.dirname(i) + ".jsonl"].add(i)
    out = {}
    for shard, want in sorted(by_shard.items()):
        path = os.path.join(jsonl_root, shard)
        with open(path) as fh:
            for line in fh:
                # the id is the first key; skip the JSON parse for other rows
                m = re.match(r'\{"id": "([^"]+)"', line)
                if m and m.group(1) in want:
                    row = json.loads(line)
                    # tokens exist only for residues with a complete backbone
                    # (mask), so keep exactly those letters
                    out[m.group(1)] = "".join(
                        c for c, ok in zip(row["seq"], row["mask"]) if ok)
        print(f"  {shard}: {sum(i in out for i in want)}/{len(want)}", flush=True)
    return out


def rcsb_annotations(entry_ids, batch=100):
    """entry id -> {"label": {asym: ann}, "auth": {auth_asym: ann}}."""
    out = {}
    ids = sorted(entry_ids)
    for k in range(0, len(ids), batch):
        chunk = ids[k:k + batch]
        body = json.dumps({"query": QUERY, "variables": {"ids": chunk}}).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(RCSB, body, {"Content-Type": "application/json"})
                reply = json.load(urllib.request.urlopen(req, timeout=120))
                if reply.get("errors"):          # a schema error will not heal on retry
                    raise SystemExit(f"RCSB GraphQL error: {reply['errors'][0]['message']}")
                data = reply["data"]
                break
            except SystemExit:
                raise
            except Exception as exc:                      # transient network
                print(f"  retry {attempt + 1}: {exc}", flush=True)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"RCSB query failed for batch starting {chunk[0]}")
        for entry in data["entries"] or []:
            res = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            common = {
                "title": (entry.get("struct") or {}).get("title"),
                "method": ", ".join(e["method"] for e in entry.get("exptl") or []) or None,
                "resolution": res[0] if res else None,
            }
            maps = {"label": {}, "auth": {}}
            for ent in entry.get("polymer_entities") or []:
                poly = ent.get("entity_poly") or {}
                if poly.get("rcsb_entity_polymer_type") not in ("RNA", "NA-hybrid"):
                    continue                    # proteins, DNA: never our chains
                orgs = ent.get("rcsb_entity_source_organism") or []
                ann = dict(common,
                           _canon=(poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", ""),
                           desc=(ent.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                           organism=", ".join(sorted({o["scientific_name"] for o in orgs
                                                      if o.get("scientific_name")})) or None)
                for inst in ent.get("polymer_entity_instances") or []:
                    ids_ = inst["rcsb_polymer_entity_instance_container_identifiers"]
                    maps["label"][ids_["asym_id"]] = dict(ann, auth_asym=ids_["auth_asym_id"])
                    maps["auth"][ids_["auth_asym_id"]] = dict(ann, auth_asym=ids_["auth_asym_id"])
            out[entry["rcsb_id"]] = maps
        print(f"  RCSB {min(k + batch, len(ids))}/{len(ids)} entries", flush=True)
    return out


def kmer_overlap(seq, canon, k=6):
    """Fraction of the chain's 6-mers found in an entity's canonical sequence."""
    kms = {seq[j:j + k] for j in range(len(seq) - k + 1)}
    return sum(km in canon for km in kms) / max(1, len(kms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--jsonl-root", required=True,
                    help="directory the database ids are relative to")
    ap.add_argument("--source", required=True, help="human-readable provenance")
    ap.add_argument("--rcsb", action="store_true", help="fetch PDB annotation")
    a = ap.parse_args()

    db = torch.load(a.db, map_location="cpu", weights_only=False)
    ids = [e["id"] for e in db["entries"]]
    ntok = {e["id"]: len(e["tokens"]) for e in db["entries"]}
    print(f"{a.db}: {len(ids)} entries", flush=True)

    seqs = sequences(ids, a.jsonl_root)
    bad = [i for i in ids if i not in seqs or len(seqs[i]) != ntok[i]]
    if bad:
        sys.exit(f"{len(bad)} entries lack a token-aligned sequence, e.g. {bad[:3]}")
    meta = {i: {"seq": seqs[i]} for i in ids}

    if a.rcsb:
        chains = {}
        for i in ids:
            m = PDB_CHAIN.search(i)
            if m:
                chains[i] = (m.group(1).upper(), m.group(2))
                meta[i]["pdb"], meta[i]["chain"] = chains[i]
        ann = rcsb_annotations({pdb for pdb, _ in chains.values()})
        hit = 0
        for i, (pdb, ch) in chains.items():
            maps = ann.get(pdb, {"label": {}, "auth": {}})
            cands = [c for c in (maps["label"].get(ch), maps["auth"].get(ch)) if c]
            if len(cands) == 2 and cands[0]["auth_asym"] != cands[1]["auth_asym"]:
                cands.sort(key=lambda c: -kmer_overlap(meta[i]["seq"], c["_canon"]))
            if cands:
                meta[i].update({k: v for k, v in cands[0].items() if k != "_canon"})
                hit += 1
        print(f"  annotated {hit}/{len(chains)}", flush=True)

    out = os.path.splitext(a.db)[0] + ".meta.json"
    with open(out, "w") as fh:
        # "built" is when the database itself was built (the date a user
        # needs to judge coverage), not when this sidecar was written
        json.dump({"source": a.source,
                   "built": dt.date.fromtimestamp(os.path.getmtime(a.db)).isoformat(),
                   "annotated": dt.date.today().isoformat(),
                   "n_entries": len(ids), "entries": meta}, fh)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
