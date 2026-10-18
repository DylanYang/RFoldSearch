"""Run US-align on the pairs listed in *.missing-audit.tsv and append them to adjudication.tsv."""
import argparse, csv, os, sys, torch, numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.expanduser("~/evalue_cal/pr14_par"))
from rfoldseek.database import short_id
from rfoldseek.usalign import score_pool
ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True); ap.add_argument("--queries", required=True)
ap.add_argument("--missing", required=True); ap.add_argument("--adjudication", required=True)
ap.add_argument("--expand", type=int, default=25, help="audit this many top-scoring members of each missing cluster at once (the README loop backfills one per round)")
ap.add_argument("--usalign", default=os.path.expanduser("~/rfs_bench/tools/USalign/USalign")); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
def tokens_clusters(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    ids = [short_id(e["id"], i) for i, e in enumerate(d["entries"])]
    return ids, [np.asarray(e["tokens"], np.int64) for e in d["entries"]], [e.get("cluster") for e in d["entries"]]
def entries(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {short_id(e["id"], i): (np.asarray(e["coords"], np.float64), np.asarray(e["mask"], bool)) for i, e in enumerate(d["entries"])}
T, Q = None, None
def work(item):
    qid, tids = item
    cands = [(t, *T[t]) for t in tids]
    q = Q.get(qid) or T[qid]
    try: scores, skipped = score_pool(a.usalign, q[0], q[1], cands)
    except ValueError: scores, skipped = {}, tids                      # query too short for US-align
    rows = [(qid, t, *scores.get(t, (0.0, 0.0))) for t in tids]
    return rows, len(skipped)
if __name__ == "__main__":
    T, Q = entries(a.db), entries(a.queries)
    todo = defaultdict(list)
    for r in csv.DictReader(open(a.missing), delimiter="\t"): todo[r["query_id"]].append((r["target_id"], r["cluster_id"]))
    if a.expand > 1:
        from rfoldseek.search import GAP_EXTEND, GAP_OPEN, load_matrix
        from rfoldseek.smith_waterman import smith_waterman_numba
        sub = np.asarray(load_matrix(None), np.float32)
        ids, toks, clus = tokens_clusters(a.db); qids, qtoks, _ = tokens_clusters(a.queries)
        qtok = dict(zip(qids, qtoks)); members = defaultdict(list)
        for i, c in enumerate(clus): members[str(c)].append(i)
        done = set()
        if os.path.exists(a.adjudication):
            for r in csv.DictReader(open(a.adjudication), delimiter="\t"): done.add((r["query_id"], r["target_id"]))
        for q, rows in todo.items():
            extra = []
            for _, cl in rows:
                sc = sorted(((smith_waterman_numba(qtok[q], toks[i], sub, GAP_OPEN, GAP_EXTEND), ids[i]) for i in members[str(cl)] if (q, ids[i]) not in done), reverse=True)
                extra += [t for _, t in sc[:a.expand]]
            rows.extend((t, "") for t in extra)
    todo = {q: sorted({t for t, _ in rows}) for q, rows in todo.items()}
    n_pairs = sum(len(v) for v in todo.values()); print(f"{len(todo):,} queries, {n_pairs:,} pairs to adjudicate", flush=True)
    new = not os.path.exists(a.adjudication)
    skipped = 0
    with open(a.adjudication, "a", newline="") as o, ProcessPoolExecutor(a.workers) as ex:
        w = csv.writer(o, delimiter="\t")
        if new: w.writerow(["query_id", "target_id", "tm_query_normalized", "tm_target_normalized"])
        for k, (rows, sk) in enumerate(ex.map(work, sorted(todo.items())), 1):
            w.writerows(rows); skipped += sk
            if k % 50 == 0: print(f"  {k}/{len(todo)} queries", flush=True)
    print(f"done; {skipped} pairs had a chain too short for US-align and were recorded with TM 0 (unrelated)")
