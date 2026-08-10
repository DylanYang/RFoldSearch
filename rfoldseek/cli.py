"""Command line interface: rfoldseek preprocess | build | search."""
import argparse
import glob
import json
import sys

import numpy as np
import torch

from .database import build_database, short_id, structure_to_entry
from .search import (DEFAULT_MIN_DIAG, DEFAULT_TOP_N, Searcher, asset)
from .tokenizer import encode_entry, load_model


def cmd_preprocess(a):
    files = sorted(glob.glob(f"{a.pdb_dir}/*.pdb") + glob.glob(f"{a.pdb_dir}/*.cif"))
    print(f"{len(files)} structure files in {a.pdb_dir}")
    n_ok = n_fail = 0
    with open(a.output, "w") as out:
        for i, f in enumerate(files, 1):
            try:
                out.write(json.dumps(structure_to_entry(f, max_len=a.max_len)) + "\n")
                n_ok += 1
            except Exception as e:
                print(f"[SKIP] {f}: {type(e).__name__}: {e}")
                n_fail += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(files)}]", flush=True)
    print(f"Done. {n_ok} saved, {n_fail} failed -> {a.output}")


def cmd_build(a):
    build_database(a.checkpoint, a.data, a.output, min_len=a.min_len)


def cmd_search(a):
    s = Searcher(a.db, matrix_path=a.matrix)
    if a.query in s.id2i:
        qname = a.query
        hits = s.search_id(a.query, top_k=a.top_k, top_n=a.top_n,
                           min_diag=a.min_diag)
        qlen = len(s.toks[s.id2i[a.query]])
    elif a.query.endswith(".json"):
        model, ckpt, _, _ = load_model(a.model)
        entry = json.loads(open(a.query).read().splitlines()[0])
        enc = encode_entry(model, entry, torch.device("cpu"),
                           ckpt.get("norm_mean"), ckpt.get("norm_std"))
        if enc is None:
            raise SystemExit("query structure could not be encoded")
        qname, qtok = short_id(a.query, 0), np.asarray(enc["tokens"], np.int64)
        qlen = len(qtok)
        hits = s.search(qtok, top_k=a.top_k, top_n=a.top_n, min_diag=a.min_diag)
    elif a.query.endswith(".pt"):
        if not a.query_id:
            raise SystemExit("--query is a .pt database; pass --query_id CHAIN_ID")
        q = torch.load(a.query, map_location="cpu", weights_only=False)
        qmap = {short_id(e.get("id"), i): i for i, e in enumerate(q["entries"])}
        if a.query_id not in qmap:
            raise SystemExit(f"{a.query_id!r} not found in {a.query}")
        qtok = np.asarray(q["entries"][qmap[a.query_id]]["tokens"], np.int64)
        qname, qlen = a.query_id, len(qtok)
        hits = s.search(qtok, top_k=a.top_k, top_n=a.top_n, min_diag=a.min_diag)
    else:
        raise SystemExit(f"query {a.query!r} is not a db chain id, a .json, or a .pt")

    print(f"# query: {qname}  ({qlen} tokens)   database: {a.db}  ({len(s.toks)} chains)")
    if not hits:
        print("# no candidates passed the prefilter (try lowering --min_diag)")
        return
    has_ev = "evalue" in hits[0]
    header = f"{'rank':>4}  {'target':<24}{'SW_score':>10}"
    if has_ev:
        header += f"{'bits':>8}{'E-value':>11}"
    print(header + f"  {'len':>5}")
    for h in hits:
        line = f"{h['rank']:>4}  {h['target']:<24}{h['sw_score']:>10.2f}"
        if has_ev:
            line += f"{h['bits']:>8.1f}{h['evalue']:>11.2g}"
        print(line + f"  {h['length']:>5}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rfoldseek",
                                 description="Fast RNA structural search with the 3Dr alphabet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess", help="PDB/mmCIF directory -> backbone JSONL")
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_len", type=int, default=None)
    p.set_defaults(fn=cmd_preprocess)

    p = sub.add_parser("build", help="backbone JSONL -> searchable token database")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default=asset("desc_vqvae_best.pt"))
    p.add_argument("--min_len", type=int, default=20)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("search", help="search a query against a token database")
    p.add_argument("--query", required=True,
                   help="chain id in --db, a preprocessed .json, or a query .pt")
    p.add_argument("--query_id", default=None)
    p.add_argument("--db", required=True)
    p.add_argument("--model", default=asset("desc_vqvae_best.pt"))
    p.add_argument("--matrix", default=None)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--min_diag", type=int, default=DEFAULT_MIN_DIAG)
    p.set_defaults(fn=cmd_search)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
