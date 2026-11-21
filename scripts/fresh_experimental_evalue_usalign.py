"""Fresh US-align adjudication of held-out hits; no historical audit table."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from rfoldsearch.database import load_database
from rfoldsearch.usalign import find_usalign, parse_outfmt2, score_pool, write_c3_pdb


def fast_score_pool(binary: str, q: dict, pool: list[tuple]):
    """US-align's documented -fast mode for screening costly RNA pairs."""
    with tempfile.TemporaryDirectory(prefix="rfoldseek_tm_fast_") as temporary:
        qpath = Path(temporary) / "query.pdb"
        write_c3_pdb(q["coords"], q["mask"], str(qpath))
        tdir = Path(temporary) / "targets"
        tdir.mkdir()
        names, index_to_id, skipped = [], {}, []
        for index, (target_id, coords, mask) in enumerate(pool):
            name = f"{index}.pdb"
            if write_c3_pdb(coords, mask, str(tdir / name)) < 3:
                skipped.append(target_id)
                continue
            names.append(name)
            index_to_id[index] = target_id
        if not names:
            return {}, skipped
        listfile = Path(temporary) / "list.txt"
        listfile.write_text("\n".join(names) + "\n")
        proc = subprocess.run([binary, str(qpath), "-dir2", str(tdir) + os.sep,
                               str(listfile), "-mol", "RNA", "-outfmt", "2", "-fast"],
                              capture_output=True, text=True, timeout=180)
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip())
        parsed = parse_outfmt2(proc.stdout)
        return {target_id: parsed[index]
                for index, target_id in index_to_id.items() if index in parsed}, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--usalign", type=Path, required=True)
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="smoke test only; omit to audit every candidate")
    ap.add_argument("--relationship", choices=("all", "Rfam_negative", "same_family"),
                    default="all")
    ap.add_argument("--max-query-length", type=int, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="screen with US-align -fast; recheck TM near 0.45 exactly")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()
    if args.shard_count < 1 or (args.shard_index is not None and
                                not 0 <= args.shard_index < args.shard_count):
        ap.error("shard index must be in [0, shard count)")
    binary = find_usalign(str(args.usalign.resolve()))
    temp = args.run / "usalign_temp"
    temp.mkdir(exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temp.resolve())
    tempfile.tempdir = str(temp.resolve())
    _, entries, tokens, ids, _, _ = load_database(str(args.run / "db_nr80.pt"))
    entry_by_id = dict(zip(ids, entries))
    length_by_id = {id_: len(tok) for id_, tok in zip(ids, tokens)}
    candidates = defaultdict(set)
    for mode in ("raw", "normalized", "compbias_raw", "compbias_normalized"):
        with (args.run / f"candidate_hits_{mode}.tsv").open(
                encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if args.relationship == "all" or row["relationship"] == args.relationship:
                    candidates[row["query_id"]].add(row["target_id"])
    base_output = args.run / "usalign_fresh.tsv"
    output = (base_output if args.shard_index is None else
              args.run / f"usalign_fresh_shard{args.shard_index}.tsv")
    done = {}
    for table in (base_output, output):
        if table.exists():
            with table.open(encoding="utf-8", newline="") as handle:
                done.update({(row["query_id"], row["target_id"]): row
                             for row in csv.DictReader(handle, delimiter="\t")})
    names = sorted(candidates, key=lambda qid: (length_by_id[qid], qid))
    if args.shard_index is not None:
        names = [qid for i, qid in enumerate(names)
                 if i % args.shard_count == args.shard_index]
    if args.max_query_length is not None:
        names = [qid for qid in names if length_by_id[qid] <= args.max_query_length]
    if args.limit_queries is not None:
        names = names[:args.limit_queries]
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if output.stat().st_size == 0:
            writer.writerow(["query_id", "target_id", "tm_query", "tm_target",
                             "mean_tm", "structurally_related", "status"])
        for i, qid in enumerate(names, 1):
            if args.shard_index is not None:
                # Other shards may have completed the reciprocal comparison.
                for shard in args.run.glob("usalign_fresh_shard*.tsv"):
                    with shard.open(encoding="utf-8", newline="") as source:
                        for row in csv.DictReader(source, delimiter="\t"):
                            if (row.get("status") == "ok" and
                                    (row.get("query_id") == qid or
                                     row.get("target_id") == qid)):
                                done[(row["query_id"], row["target_id"])] = row
            pending = sorted(tid for tid in candidates[qid] if (qid, tid) not in done)
            if not pending:
                continue
            direct = []
            for tid in pending:
                reverse = done.get((tid, qid))
                if (reverse is None or reverse["status"] != "ok" or
                        abs(float(reverse["mean_tm"]) - .45) < .03):
                    direct.append(tid)
                    continue
                mirrored = {"query_id": qid, "target_id": tid,
                            "tm_query": reverse["tm_target"],
                            "tm_target": reverse["tm_query"],
                            "mean_tm": reverse["mean_tm"],
                            "structurally_related": reverse["structurally_related"],
                            "status": reverse["status"]}
                writer.writerow(list(mirrored.values()))
                done[(qid, tid)] = mirrored
            pending = direct
            if not pending:
                handle.flush()
                continue
            q = entry_by_id[qid]
            batch_size = 10 if length_by_id[qid] > 1000 else (40 if length_by_id[qid] > 500 else 200)
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                pool = [(tid, entry_by_id[tid]["coords"], entry_by_id[tid]["mask"])
                        for tid in batch]
                result, skipped = (fast_score_pool(binary, q, pool) if args.fast else
                                   score_pool(binary, q["coords"], q["mask"], pool))
                if args.fast:
                    fast_log = ("usalign_fast_pairs.tsv" if args.shard_index is None else
                                f"usalign_fast_pairs_shard{args.shard_index}.tsv")
                    with (args.run / fast_log).open("a", encoding="utf-8") as log:
                        for tid in batch:
                            log.write(f"{qid}\t{tid}\n")
                for tid in batch:
                    if tid in skipped or tid not in result:
                        writer.writerow([qid, tid, "", "", "", "", "skipped"])
                        done[(qid, tid)] = {"query_id": qid, "target_id": tid,
                                            "tm_query": "", "tm_target": "", "mean_tm": "",
                                            "structurally_related": "", "status": "skipped"}
                    else:
                        tmq, tmt = result[tid]
                        mean = (tmq + tmt) / 2
                        writer.writerow([qid, tid, tmq, tmt, mean,
                                         int(mean >= .45), "ok"])
                        done[(qid, tid)] = {"query_id": qid, "target_id": tid,
                                            "tm_query": tmq, "tm_target": tmt,
                                            "mean_tm": mean,
                                            "structurally_related": int(mean >= .45),
                                            "status": "ok"}
                handle.flush()
            if i % 10 == 0 or i == len(names):
                print(f"US-align audited queries {i}/{len(names)}", flush=True)
    print(f"Audit table: {output}", flush=True)


if __name__ == "__main__":
    main()
