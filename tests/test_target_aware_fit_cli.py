import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


def test_fit_cli_writes_exportable_database_specific_model(tmp_path):
    root = Path(__file__).resolve().parents[1]
    db = tmp_path / "target.pt"
    entries = []
    for i, length in enumerate((12, 16, 20, 24, 28)):
        entries.append({"id": f"t{i}", "tokens": [i, (i + 1) % 5] * (length // 2),
                        "cluster": f"c{i}"})
    torch.save({"n_codes": 80, "entries": entries}, db)

    observations = tmp_path / "scores.tsv"
    columns = ("query_id", "role", "query_length", "query_entropy", "family",
               "clan", "query_weight", "expansion", "cluster_id",
               "cluster_median_length", "normalized_score")
    rows = []
    for role, qids, family in (("gumbel", tuple(f"g{i}" for i in range(8)), "gf"),
                               ("power", tuple(f"p{i}" for i in range(4)), "pf")):
        for qi, qid in enumerate(qids):
            for ci, entry in enumerate(entries):
                query_length = (13, 18, 25, 16, 29, 21, 12, 26)[qi] if role == "gumbel" else 15 + qi * 3
                entropy = (1.1, 1.8, 1.3, 2.0, 1.5, 2.2, 1.7, 1.25)[qi] if role == "gumbel" else 1.2 + qi * 0.2
                rows.append({
                    "query_id": qid, "role": role,
                    "query_length": query_length,
                    "query_entropy": entropy,
                    "family": family, "clan": "",
                    "query_weight": 1.0 / len(qids), "expansion": 1.0,
                    "cluster_id": f"c{ci}",
                    "cluster_median_length": len(entry["tokens"]),
                    "normalized_score": (0.10 + 0.018 * ci + 0.006 * qi
                                         + 0.009 * ((qi * 3 + ci * 2) % 5)),
                })
    with observations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    model_path, result_path = tmp_path / "model.json", tmp_path / "result.json"
    fit_script = root / "scripts" / "fit_target_aware_evalue_params.py"
    subprocess.run([
        sys.executable, str(fit_script), "--db", str(db),
        "--observations", str(observations), "--censoring", "none",
        "--gumbel-model", str(model_path), "--result", str(result_path),
        "--structural-audit-complete", "--validation-accepted",
    ], check=True)

    model, result = json.loads(model_path.read_text()), json.loads(result_path.read_text())
    assert model["censoring"] == "none"
    assert result["protocol"]["deployment_ready"] is True
    assert result["power"]["validation_status"] == "background-audit-complete"
    assert result["holdout_calibration"]["status"] == "skipped"   # one power family

    params_path = tmp_path / "params.json"
    export_script = root / "scripts" / "export_target_aware_evalue_params.py"
    subprocess.run([
        sys.executable, str(export_script), "--db", str(db),
        "--gumbel-model", str(model_path), "--result", str(result_path),
        "--output", str(params_path),
    ], check=True)
    params = json.loads(params_path.read_text())
    assert params["scheme"] == "target-aware-none-gumbel-power"
    assert params["n_clusters"] == len(entries)


def test_fit_cli_raw_score_bins_and_composition_bias_export_and_search(tmp_path):
    from helpers import add_coords
    from rfoldsearch.database import attach_evalue_calibration
    from rfoldsearch.search import Searcher

    root = Path(__file__).resolve().parents[1]
    db = tmp_path / "target.pt"
    entries = add_coords([{"id": f"t{i}", "tokens": [1, 2, 3, 4] * (4 + i) + [i % 7, 5, 6]}
                          for i in range(6)])
    torch.save({"n_codes": 80, "entries": entries}, db)
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("".join(f"t{i}\tc{i}\n" for i in range(6)))
    clustered = tmp_path / "clustered.pt"
    torch.save({"n_codes": 80, "entries": [{**e, "cluster": f"c{i}"} for i, e in enumerate(entries)]},
               clustered)
    columns = ("query_id", "role", "query_length", "query_entropy", "family", "clan",
               "query_weight", "expansion", "cluster_id", "cluster_median_length",
               "normalized_score")
    rows = []
    for role, n_queries in (("gumbel", 12), ("power", 44)):
        for q in range(n_queries):
            for c, entry in enumerate(entries):
                rows.append({"query_id": f"{role}{q}", "role": role,
                             "query_length": 15 + q, "query_entropy": 1.0 + 0.03 * q,
                             "family": f"f{q}", "clan": "", "query_weight": 1.0,
                             "expansion": 1.0, "cluster_id": f"c{c}",
                             "cluster_median_length": len(entry["tokens"]),
                             "normalized_score": 3.0 + 0.4 * c + 0.05 * q + 0.3 * ((q * 7 + c) % 4)})
    observations = tmp_path / "scores.tsv"
    with observations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    model, result, params = (tmp_path / n for n in ("m.json", "r.json", "p.json"))
    subprocess.run([sys.executable, str(root / "scripts" / "fit_target_aware_evalue_params.py"),
                    "--db", str(clustered), "--observations", str(observations),
                    "--censoring", "none", "--score", "raw", "--comp-bias-window", "5",
                    "--power-length-bins", "2", "--gumbel-model", str(model),
                    "--result", str(result), "--structural-audit-complete",
                    "--validation-accepted"], check=True)
    subprocess.run([sys.executable, str(root / "scripts" / "export_target_aware_evalue_params.py"),
                    "--db", str(clustered), "--gumbel-model", str(model), "--result", str(result),
                    "--output", str(params)], check=True)
    exported = json.loads(params.read_text())
    assert exported["target_aware_gumbel"]["score_normalization"] == "sw"
    assert exported["composition_bias"] == {"window": 5}
    assert len(exported["power_calibration"]["length_bins"]) == 2
    target = tmp_path / "searchable.pt"
    attach_evalue_calibration(str(db), str(clusters), str(params), str(target))
    hits = Searcher(str(target)).search_id("t0", rerank="sw", top_k=3)
    assert hits and all("evalue_sw_score" in h and h["evalue"] >= 0 for h in hits)
    assert all(h["normalized_sw_score"] == h["evalue_sw_score"] for h in hits)
