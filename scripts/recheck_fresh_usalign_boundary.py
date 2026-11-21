"""Recheck -fast US-align pairs near the TM=0.45 decision boundary exactly."""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from rfoldsearch.database import load_database
from rfoldsearch.usalign import find_usalign, score_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--usalign", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=.05)
    ap.add_argument("--max-query-length", type=int, default=None)
    ap.add_argument("--all-fast-negatives", action="store_true",
                    help="recheck every fast-screened Rfam-negative candidate")
    args = ap.parse_args()
    fast_pairs = set()
    fast_log = args.run / "usalign_fast_pairs.tsv"
    if fast_log.exists():
        for line in fast_log.read_text().splitlines():
            qid, tid = line.split("\t")
            fast_pairs.add((qid, tid))
    selection = defaultdict(set)
    negative = set()
    if args.all_fast_negatives:
        for mode in ("raw", "normalized", "compbias_raw", "compbias_normalized"):
            with (args.run / f"candidate_hits_{mode}.tsv").open(
                    encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row["relationship"] == "Rfam_negative":
                        negative.add((row["query_id"], row["target_id"]))
    with (args.run / "usalign_fresh.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pair = row["query_id"], row["target_id"]
            if (pair in fast_pairs and row["status"] == "ok" and
                    (abs(float(row["mean_tm"]) - .45) <= args.margin or
                     (args.all_fast_negatives and pair in negative))):
                selection[pair[0]].add(pair[1])
    temp = args.run / "usalign_temp"
    temp.mkdir(exist_ok=True)
    tempfile.tempdir = str(temp.resolve())
    os.environ["TMP"] = os.environ["TEMP"] = str(temp.resolve())
    _, entries, _, ids, _, _ = load_database(str(args.run / "db_nr80.pt"))
    by_id = dict(zip(ids, entries))
    out = args.run / "usalign_exact_overrides.tsv"
    already = set()
    if out.exists():
        with out.open(encoding="utf-8", newline="") as handle:
            already = {(row["query_id"], row["target_id"])
                       for row in csv.DictReader(handle, delimiter="\t")}
    if args.max_query_length is not None:
        selection = {qid: targets for qid, targets in selection.items()
                     if len(by_id[qid]["mask"]) <= args.max_query_length}
    selection = {qid: targets - {target for query, target in already if query == qid}
                 for qid, targets in selection.items()}
    selection = {qid: targets for qid, targets in selection.items() if targets}
    with out.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if out.stat().st_size == 0:
            writer.writerow(["query_id", "target_id", "tm_query", "tm_target",
                             "mean_tm", "structurally_related", "status"])
        order = sorted(selection, key=lambda qid: (len(by_id[qid]["mask"]), qid))
        for i, qid in enumerate(order, 1):
            targets = selection[qid]
            q = by_id[qid]
            candidate = [(tid, by_id[tid]["coords"], by_id[tid]["mask"])
                         for tid in sorted(targets)]
            scored, skipped = score_pool(find_usalign(str(args.usalign.resolve())),
                                         q["coords"], q["mask"], candidate)
            for tid in sorted(targets):
                if tid in skipped or tid not in scored:
                    writer.writerow([qid, tid, "", "", "", "", "skipped"])
                    continue
                tmq, tmt = scored[tid]
                mean = (tmq + tmt) / 2
                writer.writerow([qid, tid, tmq, tmt, mean, int(mean >= .45), "ok"])
            handle.flush()
            if i % 10 == 0 or i == len(order):
                print(f"rechecked {i}/{len(order)} queries", flush=True)
    print(f"exact boundary checks: {sum(map(len, selection.values()))}", flush=True)


if __name__ == "__main__":
    main()
