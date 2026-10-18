"""Leave-one-family-out check of E-value schemes, incl. Ruyi's: raw SW + target-aware null + one power law.

Background = annotated targets of another Rfam family and clan, minus pairs that
US-align finds related (mean TM >= 0.45); counts are scaled to the full database
(N_D / B_q), as the old calibration did. A calibrated E-value gives, per query,
about t unrelated clusters at E <= t.
"""
import csv, importlib.util, json, math, sys, time
from collections import defaultdict
import numpy as np
import torch

NEW = sys.argv[1]                       # worktree of origin/main
OUT = sys.argv[2]
D = sys.argv[3] if len(sys.argv) > 3 else "/home/dyaln/evalue_cal/exp/rnasolo"
DBFILE = sys.argv[4] if len(sys.argv) > 4 else "rnasolo_calibrated.pt"
ADJ = sys.argv[5] if len(sys.argv) > 5 else f"{D}/adjudication.tsv"

import rfoldsearch.evalue as old_ev     # installed (pre-PR) package
from rfoldsearch.search import load_matrix, GAP_OPEN, GAP_EXTEND
from rfoldsearch.smith_waterman import smith_waterman_numba

old_ev.TAIL_RULE = "proportional above fit_max; capped at cluster count"   # lets the fitter import
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
new_ev = load("new_evalue", f"{NEW}/rfoldsearch/evalue.py")
fitter = load("ta_fit", f"{NEW}/scripts/fit_target_aware_evalue_params.py")

db = torch.load(f"{D}/{DBFILE}", map_location="cpu", weights_only=False)
qdb = torch.load(f"{D}/calibration_queries.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
short = lambda s: str(s).split("/")[-1].rsplit(".", 1)[0]
tid = [short(e["id"]) for e in ents]
ttok = [np.asarray(e["tokens"], np.int64) for e in ents]
tclu = [str(e["cluster"]) for e in ents]
clusters = sorted(set(tclu)); NC = len(clusters)
cmed = {c: float(np.median([len(t) for t, cc in zip(ttok, tclu) if cc == c])) for c in clusters}
lab = {}
for r in csv.DictReader(open(f"{D}/labels.tsv"), delimiter="\t"):
    lab[r["chain_id"]] = (frozenset(x for x in r["family"].replace(",", ";").split(";") if x),
                          frozenset(x for x in r["clan"].replace(",", ";").split(";") if x))
related = set()
for r in csv.DictReader(open(ADJ), delimiter="\t"):
    if 0.5 * (float(r["tm_query_normalized"]) + float(r["tm_target_normalized"])) >= 0.45:
        related.add((r["query_id"], r["target_id"]))
sub = load_matrix()
ev_old = db["evalue_params"]

queries = []
for e in qdb["entries"]:
    q = short(e["id"]); fam, clan = lab[q]
    queries.append((q, np.asarray(e["tokens"], np.int64), fam, clan))
print(f"{len(queries)} queries, {len(ents)} chains, {NC} clusters", flush=True)

t0 = time.time()
rows = []      # per query x cluster: background or true positive, both scores
for qi, (q, qt, qf, qc) in enumerate(queries):
    best = {}  # (cluster, kind) -> (norm score, raw sw)
    for j in range(len(ents)):
        tf, tc = lab.get(tid[j], (frozenset(), frozenset()))
        if tid[j] == q:
            continue
        if tf & qf:
            kind = "tp"
        elif not tf or (tc & qc) or (q, tid[j]) in related:
            continue            # unannotated, same clan, or structurally related
        else:
            kind = "bg"
        sw = float(smith_waterman_numba(qt, ttok[j], sub, GAP_OPEN, GAP_EXTEND))
        ns = sw / min(len(qt), len(ttok[j]))
        k = (tclu[j], kind); b = best.get(k)
        best[k] = (max(ns, b[0]) if b else ns, max(sw, b[1]) if b else sw)
    n_bg = len({c for c, k in best if k == "bg"})
    mu, beta = old_ev.clustermax_params_for_query(ev_old, qt)
    ent = new_ev._token_entropy(qt)
    for (c, kind), (ns, sw) in best.items():
        rows.append(dict(query_id=q, family=qf, clan=qc, kind=kind, cluster_id=c, ns=ns, sw=sw,
                         qlen=len(qt), ent=ent, expansion=NC / n_bg,
                         old_raw=float(old_ev.evalue_clustermax(sw, mu, beta, NC))))
    if qi % 10 == 0:
        print(f"  scored {qi + 1}/{len(queries)} ({time.time() - t0:.0f}s)", flush=True)

fams = sorted({next(iter(sorted(r["family"]))) for r in rows})
famof = lambda r: next(iter(sorted(r["family"])))
lengths_u, mult_u = np.unique(np.asarray(list(cmed.values())), return_counts=True)

def fit_rows(sel, role):
    return [{"query_id": r["query_id"], "role": role, "query_length": float(r["qlen"]),
             "query_entropy": r["ent"], "family": r["family"], "clan": r["clan"],
             "query_weight": 1.0, "expansion": r["expansion"], "cluster_id": r["cluster_id"],
             "cluster_median_length": cmed[r["cluster_id"]], "score": r[SCORE]} for r in sel]

def new_evalues(model, power, sel):
    rr = fit_rows(sel, "power")
    e0 = fitter._predict_initial(rr, list(range(len(rr))), model, lengths_u, mult_u)
    a, b, mx = power["a"], power["b"], power["fit_max"]
    e = a * np.power(np.minimum(e0, mx), b) * np.maximum(1.0, e0 / mx)
    return np.minimum(e, NC)

def old_power(sel):
    # equal family, equal query within family, times N_D / B_q, as the deployed fit
    per_fam = defaultdict(set)
    for r in sel: per_fam[famof(r)].add(r["query_id"])
    w = [r["expansion"] / (len(per_fam) * len(per_fam[famof(r)])) for r in sel]
    return old_ev.fit_power_calibration([r["old_raw"] for r in sel], w, NC, maximum=100.0)

SCORE = "ns"
METHODS = ("old_deployed", "old_lofo", "ta_norm_left", "ta_raw_left", "ta_raw_none")
results = {m: {f: None for f in fams} for m in METHODS}
tp = {m: [] for m in results}
for fi, F in enumerate(fams):
    train = [r for r in rows if r["kind"] == "bg" and famof(r) != F]
    test = [r for r in rows if famof(r) == F]
    tfams = sorted({famof(r) for r in train})
    groles = {f: ("gumbel" if i % 2 == 0 else "power") for i, f in enumerate(tfams)}
    g_rows = fit_rows([r for r in train if groles[famof(r)] == "gumbel"], "gumbel")
    p_sel = [r for r in train if groles[famof(r)] == "power"]
    for score, cens, key in (("ns", "left-censored", "ta_norm_left"), ("sw", "left-censored", "ta_raw_left"),
                             ("sw", "none", "ta_raw_none")):
        SCORE = score
        g_rows = fit_rows([r for r in train if groles[famof(r)] == "gumbel"], "gumbel")
        model = fitter.fit_gumbel(g_rows, list(range(len(g_rows))), cens, 0.95, 3, 1e-3)
        prow = fit_rows(p_sel, "power")
        e0 = fitter._predict_initial(prow, list(range(len(prow))), model, lengths_u, mult_u)
        power = fitter.fit_power(prow, list(range(len(prow))), e0, NC, 100.0, 128)
        results[key][F] = (test, new_evalues(model, power, test))
    op = old_power(train)
    results["old_lofo"][F] = (test, np.asarray(old_ev.apply_power_calibration(
        np.array([r["old_raw"] for r in test]), op)))
    results["old_deployed"][F] = (test, np.asarray(old_ev.apply_power_calibration(
        np.array([r["old_raw"] for r in test]), ev_old["power_calibration"])))
    print(f"  fold {fi + 1}/{len(fams)} {F} done", flush=True)

TH = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
report = {}
for m, per in results.items():
    fam_fp = {t: [] for t in TH}; fam_tp = {t: [] for t in TH}
    for F, (test, ev) in per.items():
        qs = sorted({r["query_id"] for r in test})
        for t in TH:
            fp = [sum(r["expansion"] for r, e in zip(test, ev) if r["query_id"] == q and r["kind"] == "bg" and e <= t) for q in qs]
            fam_fp[t].append(float(np.mean(fp)))
            tps = [(r["query_id"], e <= t) for r, e in zip(test, ev) if r["kind"] == "tp"]
            if tps:
                fam_tp[t].append(float(np.mean([hit for _, hit in tps])))
    report[m] = {str(t): {"fp_per_query": float(np.mean(fam_fp[t])),
                          "ratio": float(np.mean(fam_fp[t]) / t),
                          "families_over_3x": int(sum(v > 3 * t for v in fam_fp[t])),
                          "tp_fraction": float(np.mean(fam_tp[t])) if fam_tp[t] else None}
                 for t in TH}
json.dump(report, open(OUT, "w"), indent=1)
print(json.dumps(report, indent=1))

