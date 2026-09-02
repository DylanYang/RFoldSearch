"""Recalibrate lambda/K from real database chain pairs and refresh the asset."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from rfoldseek import evalue as EV

DB = Path.home() / "Desktop/RFoldseek-paper-code/files/models/db.pt"
SUB = Path.home() / "Desktop/RFoldseek-paper-code/files/models/sub_matrix.pt"
OUT = Path("rfoldseek/assets/evalue_params.json")

db = torch.load(DB, map_location="cpu", weights_only=False)
toks = [np.asarray(e["tokens"], dtype=np.int64) for e in db["entries"]]
sub = torch.load(SUB, map_location="cpu", weights_only=False)["matrix"].numpy().astype(np.float32)
print(f"chains={len(toks)}  alphabet={sub.shape[0]}", flush=True)

old = json.load(open(OUT))
t0 = time.time()
table = EV.calibrate_length_dependent(sub, toks, gap_open=old["gap_open"],
                                      gap_extend=old["gap_extend"], seed=old.get("seed", 0),
                                      n_pairs=1200, long_n_pairs=400)
print(f"calibrated {len(table)} length points in {time.time()-t0:.0f}s", flush=True)
ref = [r for r in table if r["length"] == old["reference_length"]]
new = {"lambda": ref[0]["lambda"] if ref else table[len(table)//2]["lambda"],
       "K": ref[0]["K"] if ref else table[len(table)//2]["K"],
       "reference_length": old["reference_length"],
       "gap_open": old["gap_open"], "gap_extend": old["gap_extend"],
       "seed": old.get("seed", 0), "length_table": table,
       "null": "real database chain pairs, top 10% censored",
       "note": ("Length-dependent Karlin-Altschul Gumbel parameters for the token "
                "Smith-Waterman score under Structure-BLOSUM. The null is the score "
                "distribution between REAL database chains with the highest-scoring "
                "10% censored, not randomised token strings: randomisation destroys "
                "the helical grammar shared by all RNA structures and makes E-values "
                "anti-conservative by orders of magnitude. Parameters are fitted by "
                "maximum likelihood and interpolated to the query length via "
                "evalue.params_for_length (lambda linearly, K in log space). "
                "E = K * m * N * exp(-lambda * S).")}
json.dump(new, open(OUT, "w"), indent=2)
print("\n len   lambda      K        null mean")
for r in table:
    print("%5d  %7.4f  %9.3g  %8.2f" % (r["length"], r["lambda"], r["K"], r["null_score_mean"]))
print(f"\nold lambda={old['lambda']:.4f} K={old['K']:.4g}   new lambda={new['lambda']:.4f} K={new['K']:.4g}")
