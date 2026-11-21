"""Full directed member-level US-align audit for the frozen experimental comparison.

No E-value screening, reciprocal-score copying, or -fast shortcut is used.
All eligible background members (including auxiliary queries) and all non-self
same-family members are audited. Failure or missing output never means TM=0.
The successful table is fingerprinted before the refitter may consume it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rfoldsearch.database import load_database
from rfoldsearch.usalign import parse_outfmt2, write_c3_pdb
from scripts.fresh_experimental_evalue_compare import file_sha256, labels_from_tsv

INPUTS = ("db_nr80.pt", "score_manifest.json", "source_manifest.json", "structures.jsonl",
          "sequences.fasta", "raw_sw.npy", "compbias20_sw.npy", "self_raw_sw.npy",
          "self_compbias20_sw.npy")
FIELDS = ("query_id", "target_id", "tm_query", "tm_target", "status")


def prepare(source, run, labels, binary):
    run.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in INPUTS:
        src, dst = source / name, run / name
        if not src.exists():
            raise FileNotFoundError(src)
        digest = file_sha256(src)
        if dst.exists() and file_sha256(dst) != digest:
            raise ValueError(f"refusing to replace different input: {dst}")
        if not dst.exists():
            shutil.copy2(src, dst)
        hashes[name] = digest
    meta = json.loads((run / "score_manifest.json").read_text())
    if hashes["db_nr80.pt"] != meta["database_sha256"]:
        raise ValueError("source database/score fingerprint mismatch")
    if file_sha256(labels) != meta["labels_sha256"]:
        raise ValueError("labels differ from original comparison")
    labels_copy = run / "labels.tsv"
    if labels_copy.exists() and file_sha256(labels_copy) != meta["labels_sha256"]:
        raise ValueError("frozen labels changed")
    if not labels_copy.exists():
        shutil.copy2(labels, labels_copy)
    provenance = {"source": str(source.resolve()), "input_sha256": hashes,
                  "labels_sha256": file_sha256(labels), "usalign_sha256": file_sha256(binary),
                  "audit_mode": "directed default US-align; no fast mode or reciprocal reuse",
                  "threshold": .45, "old_parameters_reused": False}
    path = run / "clean_run_provenance.json"
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise ValueError("audit provenance differs; use a separate run directory")
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return meta


def required_pairs(qids, ids, clusters, labels):
    index = {id_: i for i, id_ in enumerate(ids)}
    pairs = []
    for qid in qids:
        qi = index[qid]
        qf, qc = labels[qid]
        for ti, tid in enumerate(ids):
            if clusters[qi] == clusters[ti]:
                continue
            tf, tc = labels[tid]
            positive = bool(qf and qf == tf)
            background = not ((qf and qf == tf) or (qc and qc == tc))
            if positive or background:
                pairs.append((qid, tid))
    return pairs


def checked_output(text, expected):
    parsed = parse_outfmt2(text)
    if set(parsed) != set(expected):
        raise RuntimeError(f"US-align returned {len(parsed)}/{len(expected)} pairs")
    for tm in parsed.values():
        if not np.isfinite(tm).all() or not all(0 <= value <= 1 for value in tm):
            raise RuntimeError("invalid US-align scores")
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--usalign", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-pairs", type=int, help="benchmark only; will NOT mark audit complete")
    args = ap.parse_args()
    binary = args.usalign.resolve()
    meta = prepare(args.source, args.run, args.labels, binary)
    _, entries, tokens, ids, _, clusters = load_database(str(args.run / "db_nr80.pt"))
    qids = meta["query_ids"]
    labels = labels_from_tsv(args.run / "labels.tsv")
    pairs = required_pairs(qids, ids, clusters, labels)
    required = set(pairs)
    table = args.run / "structural_members.tsv"
    done = {}
    if table.exists():
        with table.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                pair = row["query_id"], row["target_id"]
                if pair in done or pair not in required or row["status"] != "ok":
                    raise ValueError("unexpected or duplicate row in audit table")
                done[pair] = row
    state_path = table.with_suffix(".json")
    def progress(complete=False, errors=0):
        result = {"complete": complete, "threshold": .45,
                  "database_sha256": meta["database_sha256"],
                  "required_pairs": len(required), "completed_pairs": len(done),
                  "remaining_pairs": len(required)-len(done), "errors_this_run": errors,
                  "audit_mode": "default directed US-align (-mol RNA -outfmt 2)",
                  "table_sha256": file_sha256(table) if complete else None}
        state_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress()
    coordinates = args.run / "audit_coordinates"
    coordinates.mkdir(exist_ok=True)
    index = {id_: i for i, id_ in enumerate(ids)}
    for i, entry in enumerate(entries):
        if write_c3_pdb(entry["coords"], entry["mask"], str(coordinates / f"{i}.pdb")) < 3:
            raise ValueError(f"too few structural residues: {ids[i]}")
    pending = [pair for pair in pairs if pair not in done]
    pending.sort(key=lambda pair: (len(tokens[index[pair[0]]]) * len(tokens[index[pair[1]]]), pair))
    if args.limit_pairs is not None:
        pending = pending[:args.limit_pairs]
    print(f"Audit required={len(required)}, already={len(done)}, scheduled={len(pending)}", flush=True)
    started, printed, failed = time.monotonic(), time.monotonic(), 0
    # One invocation per pair isolates errors/timeouts and records progress immediately.
    def align(pair):
        qid, tid = pair
        proc = subprocess.run([str(binary), str(coordinates / f"{index[qid]}.pdb"),
                               str(coordinates / f"{index[tid]}.pdb"),
                               "-mol", "RNA", "-outfmt", "2"], capture_output=True,
                              text=True, timeout=1800,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode:
            raise RuntimeError(proc.stderr[-500:])
        tmq, tmt = checked_output(proc.stdout, [index[tid]])[index[tid]]
        return (qid, tid, tmq, tmt, "ok")
    with table.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        if stream.tell() == 0:
            writer.writerow(FIELDS)
        # Bound the queue; keep memory and cancellation predictable for large audits.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0, len(pending), 256):
                futures = {pool.submit(align, pair): pair for pair in pending[offset:offset+256]}
                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        failed += 1
                        with (args.run / "audit_errors.jsonl").open("a", encoding="utf-8") as errors:
                            errors.write(json.dumps({"pair": pair, "error": str(exc)}) + "\n")
                    else:
                        writer.writerow(row)
                        done[pair] = True
                    stream.flush()
                    if time.monotonic() - printed >= 20:
                        progress(errors=failed)
                        print(f"Audited {len(done)}/{len(required)}; errors={failed}; "
                              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
                        printed = time.monotonic()
        stream.flush()
    complete = len(done) == len(required)
    progress(complete, failed)
    print(f"Audit complete={complete}; pairs={len(done)}/{len(required)}; "
          f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    if failed:
        raise SystemExit("Audit failures remain; fitting is blocked. See audit_errors.jsonl.")


if __name__ == "__main__":
    main()
