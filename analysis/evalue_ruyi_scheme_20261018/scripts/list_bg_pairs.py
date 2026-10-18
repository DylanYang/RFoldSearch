"""List every background pair (query x annotated target of another family and clan,
non-self) and which of them US-align has not yet adjudicated. usage: DIR DBFILE OUT"""
import csv, sys, torch, numpy as np
D, DBFILE, OUT = sys.argv[1:4]
short = lambda s: str(s).split("/")[-1].rsplit(".", 1)[0]
db = torch.load(f"{D}/{DBFILE}", map_location="cpu", weights_only=False)
qdb = torch.load(f"{D}/calibration_queries.pt", map_location="cpu", weights_only=False)
tid = [short(e["id"]) for e in db["entries"]]; tlen = {short(e["id"]): len(e["tokens"]) for e in db["entries"]}
tclu = {short(e["id"]): str(e["cluster"]) for e in db["entries"]}
lab = {}
for r in csv.DictReader(open(f"{D}/labels.tsv"), delimiter="\t"):
    lab[r["chain_id"]] = (frozenset(x for x in r["family"].replace(",", ";").split(";") if x),
                          frozenset(x for x in r["clan"].replace(",", ";").split(";") if x))
done = {(r["query_id"], r["target_id"]) for r in csv.DictReader(open(f"{D}/adjudication.tsv"), delimiter="\t")}
n = miss = 0; work = 0
with open(OUT, "w", newline="") as o:
    w = csv.writer(o, delimiter="\t"); w.writerow(["query_id", "target_id", "cluster_id"])
    for e in qdb["entries"]:
        q = short(e["id"]); qf, qc = lab[q]; ql = len(e["tokens"])
        for t in tid:
            tf, tc = lab.get(t, (frozenset(), frozenset()))
            if t == q or not tf or (tf & qf) or (tc & qc):
                continue
            n += 1
            if (q, t) not in done:
                miss += 1; work += ql * tlen[t]; w.writerow([q, t, tclu[t]])
print(f"{len(qdb['entries'])} queries; {n:,} background pairs; {n - miss:,} already audited; {miss:,} to audit; sum Lq*Lt {work/1e9:.2f}G")
