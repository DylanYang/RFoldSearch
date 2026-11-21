"""Compute PR-style structural sensitivity and audited FP counts at E<=1."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

def family_ci(items: list[dict], key: str) -> dict:
    families = sorted({item["family"] for item in items})
    values = np.asarray([np.mean([item[key] for item in items if item["family"] == family])
                         for family in families])
    if np.all(values == 0):
        return {"value": 0.0, "ci95": [0.0, None], "zero_events": True,
                "n_families": len(families), "n_queries": len(items)}
    rng = np.random.default_rng(27017)
    draws = rng.integers(0, len(families), (2000, len(families)))
    low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return {"value": float(values.mean()), "ci95": [float(low), float(high)],
            "zero_events": False, "n_families": len(families), "n_queries": len(items)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    audited = {}
    with (args.run / "usalign_fresh.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            audited[(row["query_id"], row["target_id"])] = row
    overrides = args.run / "usalign_exact_overrides.tsv"
    if overrides.exists():
        with overrides.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                audited[(row["query_id"], row["target_id"])] = row
    levels = np.unique(np.r_[np.geomspace(.01, 1., 24), [.1, 1.]])
    result = {"audit_complete": True, "n_audited_pairs": len(audited),
              "levels": levels.tolist(), "methods": {}}
    for mode in ("raw", "normalized", "compbias_raw", "compbias_normalized"):
        query_info = {item["query_id"]: item for item in
                      map(json.loads, (args.run / f"per_query_{mode}.jsonl").read_text().splitlines())}
        hits = defaultdict(list)
        missing = 0
        skipped = 0
        missing_by_relationship = defaultdict(int)
        skipped_by_relationship = defaultdict(int)
        with (args.run / f"candidate_hits_{mode}.tsv").open(
                encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                audit = audited.get((row["query_id"], row["target_id"]))
                if audit is None:
                    missing += 1
                    missing_by_relationship[row["relationship"]] += 1
                    continue
                if audit["status"] != "ok":
                    skipped += 1
                    skipped_by_relationship[row["relationship"]] += 1
                    continue
                row["related"] = float(audit["mean_tm"]) >= .45
                hits[row["query_id"]].append(row)
        if missing or skipped:
            result["audit_complete"] = False
        calibration_metrics = {}
        for calibration in ("global", "four_bin"):
            ekey = "global_e" if calibration == "global" else "four_bin_e"
            metrics = {}
            for threshold in levels:
                structural_recall = []
                query_recall = []
                false_positive = []
                any_false_positive = []
                structural_related_rfam_negative = []
                for qid, info in query_info.items():
                    qhits = hits[qid]
                    homologs = [h for h in qhits if h["relationship"] == "same_family"
                                and h["related"]]
                    detected = sum(float(h[ekey]) <= threshold for h in homologs)
                    if homologs:
                        structural_recall.append({"family": info["family"],
                                                  "value": detected / len(homologs)})
                        query_recall.append({"family": info["family"],
                                             "value": float(detected > 0)})
                    fp = sum(h["relationship"] == "Rfam_negative" and
                             not h["related"] and float(h[ekey]) <= threshold
                             for h in qhits)
                    related_negative = sum(h["relationship"] == "Rfam_negative" and
                                           h["related"] and float(h[ekey]) <= threshold
                                           for h in qhits)
                    false_positive.append({"family": info["family"],
                                           "value": info["expansion"] * fp})
                    any_false_positive.append({"family": info["family"],
                                               "value": float(fp > 0)})
                    structural_related_rfam_negative.append(
                        {"family": info["family"],
                         "value": info["expansion"] * related_negative})
                metrics[str(threshold)] = {
                    "structurally_similar_homolog_cluster_recall": family_ci(
                        structural_recall, "value") if structural_recall else None,
                    "queries_with_structurally_similar_homolog": family_ci(
                        query_recall, "value") if query_recall else None,
                    "audited_unrelated_background_clusters_per_query": family_ci(
                        false_positive, "value"),
                    "queries_with_at_least_one_audited_unrelated_hit": family_ci(
                        any_false_positive, "value"),
                    "Rfam_negative_but_structurally_related_per_query": family_ci(
                        structural_related_rfam_negative, "value"),
                }
            calibration_metrics[calibration] = metrics
        result["methods"][mode] = {"missing_pairs": missing, "skipped_pairs": skipped,
                                    "missing_by_relationship": dict(missing_by_relationship),
                                    "skipped_by_relationship": dict(skipped_by_relationship),
                                    "negative_audit_complete": (
                                        missing_by_relationship["Rfam_negative"] == 0 and
                                        skipped_by_relationship["Rfam_negative"] == 0),
                                    "positive_audit_complete": (
                                        missing_by_relationship["same_family"] == 0 and
                                        skipped_by_relationship["same_family"] == 0),
                                    "calibrations": calibration_metrics}
    (args.run / "structural_audit_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_complete": result["audit_complete"],
                      "n_audited_pairs": result["n_audited_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
