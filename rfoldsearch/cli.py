"""Command line interface: rfoldsearch preprocess | build | calibrate | search."""
import argparse
import glob
import json
import sys

import numpy as np
import torch

from .calibrate import AuditIncomplete, calibrate
from .experimental_calibrate import fit_files as fit_experimental_evalues
from .predicted_calibrate import fit_files as fit_predicted_evalues
from .database import (attach_evalue_calibration, build_database, short_id,
                       structure_to_entry)
from .search import (DEFAULT_MIN_DIAG, DEFAULT_RERANK, DEFAULT_TOP_N, RERANKERS,
                     Searcher, asset)
from .tokenizer import encode_entry, load_model
from .usalign import USalignFailed, USalignNotFound


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
    build_database(a.checkpoint, a.data, a.output, min_len=a.min_len,
                   clusters=a.clusters, evalue_params=a.evalue_params)


def cmd_attach_calibration(a):
    result = attach_evalue_calibration(
        a.db, a.clusters, a.evalue_params, a.output,
        index_prefix=a.index_prefix)
    print(f"Saved: {result['output']}  ({result['n_chains']} chains, "
          f"{result['n_clusters']} clusters)")


def cmd_calibrate(a):
    try:
        calibrate(a.db, a.clusters, a.labels, a.queries, a.output,
                  db_type=a.db_type, related_pairs=a.related_pairs,
                  query_panel=a.query_panel, usalign=a.usalign,
                  workdir=a.workdir, index_prefix=a.index_prefix,
                  matrix=a.matrix, log=lambda m: print(m, flush=True))
    except (AuditIncomplete, FileExistsError) as e:
        raise SystemExit(str(e))


def cmd_fit_experimental_evalues(a):
    fit_experimental_evalues(a)


def cmd_fit_predicted_evalues(a):
    fit_predicted_evalues(a)


def cmd_search(a):
    s = Searcher(a.db, matrix_path=a.matrix,
                 evalue_params_path=a.evalue_params)
    kw = dict(top_k=a.top_k, top_n=a.top_n, min_diag=a.min_diag,
              rerank=a.rerank, usalign=a.usalign, tie_seed=a.tie_seed)
    if a.query in s.id2i:
        qname = a.query
        hits = s.search_id(a.query, **kw)
        qlen = len(s.toks[s.id2i[a.query]])
    elif a.query.endswith(".json"):
        model, ckpt, _, _ = load_model(a.model)
        entry = json.loads(open(a.query).read().splitlines()[0])
        enc = encode_entry(model, entry, torch.device("cpu"),
                           ckpt.get("norm_mean"), ckpt.get("norm_std"),
                           active_blocks=ckpt.get("active_blocks"))
        if enc is None:
            raise SystemExit("query structure could not be encoded")
        qname, qtok = short_id(a.query, 0), np.asarray(enc["tokens"], np.int64)
        qlen = len(qtok)
        hits = s.search(qtok, query_entry=enc, **kw)
    elif a.query.endswith(".pt"):
        if not a.query_id:
            raise SystemExit("--query is a .pt database; pass --query_id CHAIN_ID")
        q = torch.load(a.query, map_location="cpu", weights_only=False)
        qmap = {short_id(e.get("id"), i): i for i, e in enumerate(q["entries"])}
        if a.query_id not in qmap:
            raise SystemExit(f"{a.query_id!r} not found in {a.query}")
        qent = q["entries"][qmap[a.query_id]]
        qtok = np.asarray(qent["tokens"], np.int64)
        qname, qlen = a.query_id, len(qtok)
        hits = s.search(qtok, query_entry=qent, **kw)
    else:
        raise SystemExit(f"query {a.query!r} is not a db chain id, a .json, or a .pt")

    # The E-value counts the search space in clusters, so say how many there
    # are whenever one is going to be printed; TM reranking reports no E-value.
    size = f"{len(s.toks)} chains"
    if a.rerank != "tm" and s.ev_par is not None:
        size += f", {s.n_clusters} clusters"
    head = f"# query: {qname}  ({qlen} tokens)   database: {a.db}  ({size})"
    if a.rerank != "tm" and s.power_ev is not None:
        head += "   E-values: power-calibrated"
    if a.rerank == "tm":
        head += "   rerank: RFoldSearch-TM"
    elif a.rerank == "tm_a":
        head += "   rerank: R_str = S_SW x TM_a"
    print(head)
    if not hits:
        print("# no candidates passed the prefilter (try lowering --min_diag)")
        return
    skipped = getattr(s, "last_tm_skipped", None)
    if skipped:
        print(f"# {len(skipped)} candidate(s) skipped: too few resolved residues")
    if "tm_mean" in hits[0]:
        print(f"{'rank':>4}  {'target':<24}{'TM_mean':>9}{'TM_query':>10}"
              f"{'TM_target':>11}  {'len':>5}")
        for h in hits:
            print(f"{h['rank']:>4}  {h['target']:<24}{h['tm_mean']:>9.4f}"
                  f"{h['tm_query']:>10.4f}{h['tm_target']:>11.4f}"
                  f"  {h['length']:>5}")
        return
    has_ev = "evalue" in hits[0]
    if "structure_score" in hits[0]:
        header = (f"{'rank':>4}  {'target':<24}{'R_str':>10}{'TM_a':>9}"
                  f"{'S_SW':>10}")
        if has_ev:
            header += f"{'E-value':>11}"
        print(header + f"{'aligned':>10}  {'len':>5}")
        for h in hits:
            line = (f"{h['rank']:>4}  {h['target']:<24}"
                    f"{h['structure_score']:>10.2f}{h['tm_a']:>9.4f}"
                    f"{h['sw_score']:>10.2f}")
            if has_ev:
                line += f"{h['evalue']:>11.2g}"
            print(line + f"{h['aligned']:>10}  {h['length']:>5}")
        return
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
    ap = argparse.ArgumentParser(prog="rfoldsearch",
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
    p.add_argument("--min_len", type=int, default=None,
                   help="drop chains with fewer than this many valid "
                        "residues; the default keeps every chain the "
                        "tokeniser can encode. The paper's 885-chain "
                        "corpus used --min_len 20")
    p.add_argument("--clusters", default=None,
                   help="CD-HIT .clstr (or id/cluster TSV) grouping "
                        "near-duplicate chains; E-values are counted in "
                        "clusters, so without it a redundant database "
                        "reports E-values that are too small")
    p.add_argument("--evalue_params", default=None,
                   help="matching database-specific E-value parameter JSON")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("attach-calibration",
                       help="add cluster labels and E-value parameters to an existing database")
    p.add_argument("--db", required=True, help="existing encoded database")
    p.add_argument("--clusters", required=True,
                   help="CD-HIT .clstr or id/cluster TSV")
    p.add_argument("--evalue_params", required=True,
                   help="database-specific E-value parameter JSON")
    p.add_argument("--output", required=True,
                   help="new calibrated database path")
    p.add_argument("--index-prefix", default=None,
                   help="CD-HIT IDs based on database row positions, e.g. t for t0, t1, ...")
    p.set_defaults(fn=cmd_attach_calibration)

    p = sub.add_parser("calibrate",
                       help="fit database-specific E-values and attach them, in one step")
    p.add_argument("--db", required=True, help="encoded database to calibrate")
    p.add_argument("--clusters", required=True,
                   help="CD-HIT-EST .clstr (or id/cluster TSV) for --db")
    p.add_argument("--labels", required=True,
                   help="TSV of chain_id with Rfam family and clan columns, "
                        "covering --db and --queries")
    p.add_argument("--queries", required=True,
                   help="independent database of calibration queries")
    p.add_argument("--db_type", required=True, choices=("predicted", "experimental"),
                   help="predicted: background is every eligible cluster; "
                        "experimental: Rfam-annotated targets, reweighted to "
                        "the full database")
    p.add_argument("--output", required=True, help="calibrated database path")
    p.add_argument("--related_pairs", default=None,
                   help="existing US-align adjudication TSV (query_id, "
                        "target_id, tm_query, tm_target); pairs it lacks are "
                        "aligned with US-align")
    p.add_argument("--query_panel", default=None,
                   help="TSV of query_id selecting the chains of --db used to "
                        "fit the Gumbel model (default: every chain)")
    p.add_argument("--usalign", default=None,
                   help="path to the USalign binary (default: $USALIGN, then PATH)")
    p.add_argument("--workdir", default=None,
                   help="intermediate files (default: OUTPUT stem + .calibration/)")
    p.add_argument("--index_prefix", default=None,
                   help="CD-HIT IDs based on database row positions, e.g. t for t0, t1, ...")
    p.add_argument("--matrix", default=None)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("fit-experimental-evalues",
                       help="fit raw-SW target-aware Gumbel and empirical mapping")
    for name in ("db", "labels", "output", "workdir"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    for name in ("query_indices", "scores", "score_manifest", "tm_matrix",
                 "tm_manifest"):
        p.add_argument("--" + name.replace("_", "-"))
    p.add_argument("--matrix")
    p.add_argument("--usalign")
    p.add_argument("--uncensored", action="store_true",
                   help="fit all Gumbel scores exactly instead of left-censoring")
    p.add_argument("--audit-ceiling", type=float, default=100.0,
                   help="audit cluster leaders through this initial E-value (at least 100)")
    p.set_defaults(fn=cmd_fit_experimental_evalues)

    p = sub.add_parser("fit-predicted-evalues",
                       help="fit composition-adjusted SW, target-aware Gumbel and empirical map")
    for name in ("db", "labels", "training_indices", "auxiliary_indices",
                 "calibration_indices",
                 "output", "workdir"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    for name in ("scores", "score_manifest", "tm_matrix", "tm_manifest"):
        p.add_argument("--" + name.replace("_", "-"))
    p.add_argument("--matrix")
    p.add_argument("--usalign")
    p.add_argument("--audit-ceiling", type=float, default=100.0,
                   help="audit cluster leaders through this initial E-value (at least 100)")
    p.set_defaults(fn=cmd_fit_predicted_evalues)

    p = sub.add_parser("search", help="search a query against a token database")
    p.add_argument("--query", required=True,
                   help="chain id in --db, a preprocessed .json, or a query .pt")
    p.add_argument("--query_id", default=None)
    p.add_argument("--db", required=True)
    p.add_argument("--model", default=asset("desc_vqvae_best.pt"))
    p.add_argument("--matrix", default=None)
    p.add_argument("--evalue_params", default=None,
                   help="database-specific E-value parameter JSON for this search")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--min_diag", type=int, default=DEFAULT_MIN_DIAG)
    p.add_argument("--rerank", choices=RERANKERS, default=DEFAULT_RERANK,
                   help="rerank by R_str = S_SW x TM_a (tm_a, default), "
                        "token Smith-Waterman (sw), or US-align TM-score (tm)")
    p.add_argument("--usalign", default=None,
                   help="path to the USalign binary (default: $USALIGN, then PATH)")
    p.add_argument("--tie_seed", type=int, default=None,
                   help="break equal scores by a hash of the chain id seeded "
                        "with this integer, instead of by database order; "
                        "omit to keep the historical index-order behaviour")
    p.set_defaults(fn=cmd_search)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (USalignNotFound, USalignFailed) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
