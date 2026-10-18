"""Replay held-out E-values and bound structural background counts through E=100.

This diagnostic uses frozen fits from comparison.json. It does not refit a
structurally cleaned null model or modify the source run.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rfoldsearch.database import load_database
from scripts.fresh_experimental_evalue_compare import (
    apply_power, e0_for_scores, entropy, features, file_sha256,
    labels_from_tsv, select_leaders,
)

THRESHOLDS = (0.01, 0.1, 1.0, 10.0, 100.0)
MODES = ("raw", "normalized", "compbias_raw", "compbias_normalized")


def family_mean(values: np.ndarray, families: np.ndarray) -> np.ndarray:
    return np.mean([np.mean(values[families == family], axis=0)
                    for family in sorted(set(families))], axis=0)


def power_at_e0(model: dict, query_length: int, e0: float,
                n_clusters: int) -> float:
    """Independent boundary calculation, including the verified linear tail."""
    if "bins" in model:
        centres = [item["log_length_centre"] for item in model["bins"]]
        loga = [np.log(item["a"]) for item in model["bins"]]
        a = float(np.exp(np.interp(np.log(query_length), centres, loga)))
    else:
        a = model["a"]
    endpoint = model["fit_max"]
    result = (a * e0 ** model["b"] if e0 <= endpoint else
              a * endpoint ** model["b"] * e0 / endpoint)
    return min(float(result), n_clusters)


def audit_states(path: Path, qids: list[str], ids: list[str], db_hash: str) -> tuple[np.ndarray, dict]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["database_sha256"] != db_hash or metadata["threshold"] != .45:
        raise ValueError("structural audit does not match the database or TM cutoff")
    qindex = {name: i for i, name in enumerate(qids)}
    tindex = {name: i for i, name in enumerate(ids)}
    state = np.zeros((len(qids), len(ids)), dtype=np.uint8)
    n_rows = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            n_rows += 1
            if row["status"] != "ok":
                raise ValueError("non-successful structural audit row")
            if row["query_id"] not in qindex:
                continue  # auxiliary query, not evaluated in this report
            qi, ti = qindex[row["query_id"]], tindex[row["target_id"]]
            if state[qi, ti]:
                raise ValueError("duplicate structural audit pair")
            tm = (float(row["tm_query"]) + float(row["tm_target"])) / 2
            if not np.isfinite(tm) or not 0 <= tm <= 1:
                raise ValueError("invalid TM-score")
            state[qi, ti] = 2 if tm >= .45 else 1
    if n_rows < metadata["completed_pairs"] or n_rows > metadata["required_pairs"]:
        raise ValueError("structural audit row count is inconsistent with metadata")
    metadata["observed_table_rows"] = n_rows
    return state, metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    comparison = json.loads((args.run / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run / "score_manifest.json").read_text(encoding="utf-8"))
    db = args.run / "db_nr80.pt"
    if file_sha256(db) != manifest["database_sha256"]:
        raise ValueError("database differs from frozen score manifest")
    if file_sha256(args.labels) != manifest["labels_sha256"]:
        raise ValueError("labels differ from frozen score manifest")
    _, _, tokens, ids, alphabet, clusters = load_database(str(db))
    qids = manifest["query_ids"][:manifest["n_annotated_queries"]]
    n_query = len(qids)
    n_clusters = comparison["protocol"]["n_clusters"]
    if len(ids) != manifest["n_targets"] or n_query != comparison["protocol"]["n_annotated_queries"]:
        raise ValueError("run dimensions differ from comparison metadata")
    labels = labels_from_tsv(args.labels)
    if set(ids) != set(labels):
        raise ValueError("labels and database IDs differ")
    qi = np.asarray([ids.index(name) for name in qids], dtype=int)
    lengths = np.asarray([len(t) for t in tokens], dtype=float)
    entropy_values = np.asarray([entropy(t, alphabet) for t in tokens])
    member_map = defaultdict(list)
    for ti, cluster in enumerate(clusters):
        member_map[str(cluster)].append(ti)
    members = [member_map[c] for c in sorted(member_map)]
    if len(members) != n_clusters:
        raise ValueError("cluster count differs from comparison metadata")
    cluster_length = np.asarray([np.median(lengths[m]) for m in members])
    features_by_query = features(lengths[qi], entropy_values[qi], cluster_length)
    state, audit_meta = audit_states(args.audit, qids, ids, manifest["database_sha256"])
    levels = np.asarray(comparison["levels"])
    threshold_indices = [int(np.flatnonzero(np.isclose(levels, e))[0]) for e in THRESHOLDS]
    families = np.asarray([labels[name][0] for name in qids])
    findings = {}
    for mode in MODES:
        saved_rows = {row["query_id"]: row for row in map(
            json.loads, (args.run / f"per_query_{mode}.jsonl").read_text(encoding="utf-8").splitlines())}
        if set(saved_rows) != set(qids):
            raise ValueError(f"{mode}: incomplete per-query rows")
        base = np.load(args.run / ("compbias20_sw.npy" if mode.startswith("compbias") else "raw_sw.npy"), mmap_mode="r")
        if base.shape != (manifest["n_queries"], len(ids)):
            raise ValueError(f"{mode}: score matrix dimensions differ")
        scores = np.asarray(base[:n_query])
        if mode.endswith("normalized"):
            scores = scores / np.minimum(lengths[qi, None], lengths[None, :])
        leader = select_leaders(scores, qi, ids, clusters, members, labels, True)
        mode_result = {}
        for calibration in ("global", "four_bin"):
            proxy = np.zeros((n_query, len(levels)))
            unrelated = np.zeros_like(proxy)
            unknown = np.zeros_like(proxy)
            possible_backfill = np.zeros_like(proxy)
            edge = np.zeros(n_query)
            maximum = np.zeros(n_query)
            lower_edge = np.zeros(n_query)
            for row, qid in enumerate(qids):
                saved = saved_rows[qid]
                model = comparison["methods"][mode]["fold_models"][saved["fold"]]
                power = model["power"][calibration]
                targets = leader[row, leader[row] >= 0]
                columns = np.flatnonzero(leader[row] >= 0)
                e0 = e0_for_scores(scores[row, targets], features_by_query[row], model["gumbel"])
                e = apply_power(e0, power, lengths[qi[row]], n_clusters)
                factor = saved["expansion"]
                hits = e[:, None] <= levels
                proxy[row] = factor * hits.sum(axis=0)
                if not np.allclose(proxy[row], saved[f"counts_{calibration}"], rtol=0, atol=1e-7):
                    raise ValueError(f"{mode}/{calibration}/{qid}: replay differs from saved counts")
                observed = state[row, targets]
                unrelated[row] = factor * ((observed == 1)[:, None] & hits).sum(axis=0)
                unknown[row] = factor * ((observed == 0)[:, None] & hits).sum(axis=0)
                ambiguous = (observed == 2) & np.asarray([len(members[c]) > 1 for c in columns])
                possible_backfill[row] = factor * (ambiguous[:, None] & hits).sum(axis=0)
                edge[row] = power_at_e0(power, int(lengths[qi[row]]), power["fit_max"], n_clusters)
                maximum[row] = power_at_e0(power, int(lengths[qi[row]]), n_clusters, n_clusters)
                lower_edge[row] = power_at_e0(power, int(lengths[qi[row]]), power["fit_min"], n_clusters)
            summary = {}
            for threshold, index in zip(THRESHOLDS, threshold_indices):
                expected = comparison["methods"][mode]["curves"][calibration]["observed"][index]
                measured = float(family_mean(proxy[:, index], families))
                if not np.isclose(measured, expected, rtol=0, atol=1e-7):
                    raise ValueError(f"{mode}/{calibration}: family mean differs at E={threshold}")
                summary[str(threshold)] = {
                    "rfam_negative_proxy_per_query": measured,
                    "proxy_ci95": [comparison["methods"][mode]["curves"][calibration][key][index]
                                   for key in ("ci95_low", "ci95_high")],
                    "proxy_over_expected": measured / threshold,
                    "audit_unrelated_lower": float(family_mean(unrelated[:, index], families)),
                    "audit_unrelated_upper": float(family_mean(
                        (unrelated + unknown + possible_backfill)[:, index], families)),
                    "queries_with_unknown_leader": int(np.sum(unknown[:, index] > 0)),
                    "fraction_queries_inside_power_fit": float(np.mean(
                        (lower_edge <= threshold) & (threshold <= edge))),
                    "fraction_queries_above_attainable_max": float(np.mean(threshold > maximum)),
                }
            mode_result[calibration] = {
                "thresholds": summary,
                "fit_upper_e_cal_median": float(np.median(edge)),
                "attainable_e_cal_max_median": float(np.median(maximum)),
            }
            print(f"verified {mode}/{calibration}: {n_query} queries, {len(levels)} levels", flush=True)
        findings[mode] = mode_result
    output = {
        "scope": "frozen exploratory fits; held-out annotated-background leaders; no structural refit",
        "database_sha256": manifest["database_sha256"],
        "audit_table_sha256": file_sha256(args.audit),
        "audit_complete": audit_meta["complete"],
        "audit_completed_pairs": audit_meta["observed_table_rows"],
        "audit_metadata_completed_pairs": audit_meta["completed_pairs"],
        "audit_required_pairs": audit_meta["required_pairs"],
        "n_query": n_query, "n_clusters": n_clusters,
        "n_grid_levels_replayed": len(levels),
        "methods": findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
