"""Recalibrate the cluster-max null and refresh the packaged asset.

Clusters come from CD-HIT-EST at 80% identity, run beforehand:

    cd-hit-est -i targets.fa -o nr80 -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1

Chains of the same Rfam family or clan are excluded from the null, so what is
fitted describes unrelated structures. The reference database is the 885-chain
NR80 benchmark; a deployment over a different database should rerun this
against that database and pass the result to Searcher as evalue_params_path.
"""
import csv, json, os, time
from pathlib import Path
import numpy as np, torch
from rfoldseek import evalue as EV
from rfoldseek.database import read_clusters, assign_clusters

PC = Path.home() / "Desktop/RFoldseek-paper-code"
DB = PC / "files/models/db.pt"
SUB = PC / "files/models/sub_matrix.pt"
LABELS = PC / "Splits/labels.tsv"
CLSTR = os.environ.get("RFS_CLSTR")          # CD-HIT .clstr for the same chains
OUT = Path(__file__).resolve().parent.parent / "rfoldseek/assets/evalue_params.json"

db = torch.load(DB, map_location="cpu", weights_only=False)
ents = db["entries"]
toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
sub = torch.load(SUB, map_location="cpu", weights_only=False)["matrix"].numpy().astype(np.float32)

lab = {r["entry_id"]: r for r in csv.DictReader(open(LABELS), delimiter="\t")}
key = [str(e["id"]) for e in ents]
fam = np.array([lab.get(k, {}).get("rfam_acc", "") or "" for k in key], dtype=object)
clan = np.array([lab.get(k, {}).get("clan_acc", "") or "" for k in key], dtype=object)
known = np.array([f not in ("", "N/A", "None") for f in fam])
hasc = np.array([c not in ("", "N/A", "None") for c in clan])

if not CLSTR:
    raise SystemExit("set RFS_CLSTR to the cd-hit-est .clstr file for these chains")
clusters = assign_clusters(ents, read_clusters(CLSTR))


def exclude(i, j):
    if known[i] and known[j] and fam[i] == fam[j]:
        return True
    if hasc[i] and hasc[j] and clan[i] == clan[j]:
        return True
    return False


t0 = time.time()
par = EV.calibrate_clustermax_length_dependent(sub, toks, clusters,
                                               gap_open=-2.0, gap_extend=-0.5,
                                               n_queries=24, seed=0,
                                               exclude=exclude, len_ratio=3.0)
print(f"calibrated {len(par['length_table'])} length points in {time.time()-t0:.0f}s "
      f"({par['n_clusters']} clusters, {par['n_chains']} chains)", flush=True)

out = {"scheme": "cluster-max",
       "gap_open": -2.0, "gap_extend": -0.5, "seed": 0,
       "len_ratio": 3.0,
       "calibration_database": str(DB),
       "n_clusters": par["n_clusters"], "n_chains": par["n_chains"],
       "skipped_lengths": par["skipped_lengths"],
       "length_table": par["length_table"],
       "clusters": "cd-hit-est -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1",
       "null": "cluster maxima over real database chains, same Rfam family or "
               "clan excluded; nothing censored",
       "note": "Gumbel (mu, beta) of the cluster-max null on a grid of query "
               "lengths, interpolated by evalue.clustermax_params_for_length. "
               "E = N_clusters * (1 - exp(-exp(-(S - mu)/beta))); the bit "
               "score is (S - mu)/(beta ln 2). Fitting maxima rather than "
               "individual pairs removes the per-database censoring fraction "
               "the earlier chain-level calibration needed, and keeps the "
               "calibration stable across databases of different redundancy."}
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
for r in par["length_table"]:
    print("  L=%-5d mu=%7.2f beta=%6.2f lambda=%.3f n=%d"
          % (r["length"], r["mu"], r["beta"], r["lambda"], r["n_maxima"]))
