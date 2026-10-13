"""The target-aware exporter must never create a universal calibration."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import torch
import pytest
from rfoldsearch.database import (calibration_context_fingerprint,
                                database_fingerprint, load_database,
                                attach_evalue_calibration)
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, Searcher, load_matrix


def _write_inputs(tmp_path, validation_status="background-audit-complete",
                  censoring="left-censored"):
    database = tmp_path / "target.pt"
    torch.save({
        "n_codes": 80,
        "entries": [
            {"id": "a", "tokens": [1] * 20, "cluster": "c0"},
            {"id": "b", "tokens": [1] * 30, "cluster": "c1"},
        ],
    }, database)
    _, entries, _, _, n_codes, clusters = load_database(database)
    digest = database_fingerprint(entries, clusters, n_codes)
    source = {
        "target_database_fingerprint": digest,
        "calibration_context_fingerprint": calibration_context_fingerprint(
            digest, load_matrix(), GAP_OPEN, GAP_EXTEND),
        "score_normalization": "sw/min(query_length,target_length)",
        "target_length_feature": "cluster median",
        "composition_bias": False,
    }
    model = tmp_path / "model.json"
    model.write_text(json.dumps({
        "coefficients": [[0.0] * 5, [0.0] * 5],
        "mean": [0.0] * 4,
        "scale": [1.0] * 4,
        "censoring": censoring,
        "fit": ("95th-percentile left-censored weighted Gumbel likelihood"
                if censoring == "left-censored" else "uncensored weighted Gumbel likelihood"),
        "auxiliary_training": "short/low-entropy auxiliary panel",
        "source": source,
        "fingerprint": "fitted-model-identity",
    }))
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "audit": [{"pending": 0}],
        "protocol": {
            "n_clusters": 2, "deployment_ready": True,
            "gumbel_censoring": "left" if censoring == "left-censored" else "none",
            "score": "SW/min(Lq,Lt), then eligible cluster max",
            "target_length_feature": "cluster median",
            "composition_bias": False,
        },
        "source": source,
        "gumbel_model_fingerprint": "fitted-model-identity",
        "gumbel_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "power": {
            "a": 1.0, "b": 1.0, "fit_min": 0.01, "fit_max": 100.0,
            "validation_status": validation_status,
            "tail_rule": "proportional above fit_max; capped at cluster count",
        },
    }))
    return database, model, result


def test_target_aware_export_is_database_specific(tmp_path):
    database, model, result = _write_inputs(tmp_path)
    output = tmp_path / "params.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_target_aware_evalue_params.py"
    subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--gumbel-model", str(model), "--result", str(result),
        "--output", str(output),
    ], check=True)
    params = json.loads(output.read_text())
    assert params["scheme"] == "target-aware-left-censored-gumbel-power"
    assert params["target_aware_gumbel"]["censoring"] == "left-censored"
    assert params["n_clusters"] == 2
    assert params["power_calibration"]["validation_status"] == "background-audit-complete"
    assert params["provenance"]["scope"] == "database-specific target-aware E-value calibration"
    assert "33,043" not in output.read_text()


def test_export_attach_search_path_is_connected(tmp_path):
    database, model, result = _write_inputs(tmp_path)
    output = tmp_path / "params.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_target_aware_evalue_params.py"
    subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--gumbel-model", str(model), "--result", str(result),
        "--output", str(output),
    ], check=True)
    clusters = tmp_path / "nr80.clstr"
    clusters.write_text(">Cluster 0\n0\t20nt, >a... *\n"
                        ">Cluster 1\n0\t30nt, >b... *\n")
    attached = tmp_path / "attached.pt"
    attach_evalue_calibration(str(database), str(clusters), str(output),
                              str(attached))
    hits = Searcher(str(attached)).search_id("a", rerank="sw", top_k=1)
    assert hits and hits[0]["target"] == "b"
    assert hits[0]["normalized_sw_score"] == pytest.approx(
        hits[0]["sw_score"] / 20)
    assert hits[0]["raw_evalue"] >= 0
    assert hits[0]["evalue"] >= 0


def test_target_aware_export_rejects_unvalidated_power_fit(tmp_path):
    database, model, result = _write_inputs(tmp_path, "structural-audit-not-recorded")
    output = tmp_path / "params.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_target_aware_evalue_params.py"
    completed = subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--gumbel-model", str(model), "--result", str(result),
        "--output", str(output),
    ], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "background-audit-complete" in completed.stderr


def test_target_aware_export_allows_an_explicit_uncensored_model(tmp_path):
    database, model, result = _write_inputs(tmp_path, censoring="none")
    output = tmp_path / "params.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_target_aware_evalue_params.py"
    subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--gumbel-model", str(model), "--result", str(result),
        "--output", str(output),
    ], check=True)
    params = json.loads(output.read_text())
    assert params["scheme"] == "target-aware-none-gumbel-power"
    assert params["target_aware_gumbel"]["censoring"] == "none"


def test_target_aware_export_rejects_undeployable_result(tmp_path):
    database, model, result = _write_inputs(tmp_path)
    report = json.loads(result.read_text())
    report["protocol"]["deployment_ready"] = False
    result.write_text(json.dumps(report))
    output = tmp_path / "params.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_target_aware_evalue_params.py"
    completed = subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--gumbel-model", str(model), "--result", str(result),
        "--output", str(output),
    ], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "deployment_ready" in completed.stderr
    assert not output.exists()


def test_target_aware_export_rejects_missing_mode_and_wrong_source(tmp_path):
    database, model, result = _write_inputs(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_target_aware_evalue_params.py"
    output = tmp_path / "params.json"
    fitted = json.loads(model.read_text())
    del fitted["censoring"]
    model.write_text(json.dumps(fitted))
    command = [sys.executable, str(script), "--db", str(database),
               "--gumbel-model", str(model), "--result", str(result),
               "--output", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "fitted censoring mode" in completed.stderr
    fitted["censoring"] = "left-censored"
    fitted["source"]["target_database_fingerprint"] = "wrong"
    model.write_text(json.dumps(fitted))
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "same target database" in completed.stderr
    assert not output.exists()


def test_target_aware_export_rejects_power_from_another_gumbel_fit(tmp_path):
    database, model, result = _write_inputs(tmp_path)
    report = json.loads(result.read_text())
    report["gumbel_model_sha256"] = "another-model"
    result.write_text(json.dumps(report))
    output = tmp_path / "params.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_target_aware_evalue_params.py"
    completed = subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--gumbel-model", str(model), "--result", str(result),
        "--output", str(output),
    ], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "contents do not match" in completed.stderr
    assert not output.exists()
