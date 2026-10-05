"""Argument combinations for power-law E-value fitting."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fit_power_evalue_calibration.py"


def test_query_panel_rejects_explicit_weight_column(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"n_clusters": 2}))
    panel = tmp_path / "panel.tsv"
    panel.write_text("query_id\tfamily\nq1\tF1\nq2\tF1\n")
    background = tmp_path / "background.tsv"
    background.write_text("query_id\tfamily\traw_evalue\tweight\n"
                          "q1\tF1\t0.1\t1.0\n")
    output = tmp_path / "fitted.json"

    command = [sys.executable, str(SCRIPT), "--params", str(params),
               "--background", str(background), "--query-panel", str(panel),
               "--n-clusters", "2", "--output", str(output)]
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "explicit weights cannot be combined" in rejected.stderr
    assert not output.exists()

    background.write_text(
        "query_id\tfamily\traw_evalue\nq1\tF1\t0.1\nq2\tF1\t1.0\n")
    accepted = subprocess.run(command, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    fitted = json.loads(output.read_text())["power_calibration"]
    assert fitted["n_calibration_queries"] == 2
    assert fitted["weighting"] == "equal-family, equal-query-within-family"


def test_annotated_background_scaling_uses_family_balanced_nd_over_bq(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"n_clusters": 10}))
    panel = tmp_path / "panel.tsv"
    panel.write_text(
        "query_id\tfamily\tbackground_scope\tstructural_audit_ceiling\t"
        "structural_audit_complete\tn_database_clusters\t"
        "background_clusters_before_structural_exclusion\n"
        "q1\tF1\tannotated\t100\ttrue\t10\t5\n"
        "q2\tF1\tannotated\t100\ttrue\t10\t10\n"
        "q3\tF2\tannotated\t100\ttrue\t10\t5\n"
    )
    background = tmp_path / "background.tsv"
    background.write_text(
        "query_id\tfamily\traw_evalue\n"
        "q1\tF1\t0.1\n"
        "q2\tF1\t0.2\n"
    )
    output = tmp_path / "fitted.json"

    result = subprocess.run([
        sys.executable, str(SCRIPT), "--params", str(params),
        "--background", str(background), "--query-panel", str(panel),
        "--background-scaling", "annotated-to-full", "--n-clusters", "10",
        "--raw-evalue-ceiling", "100",
        "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    fitted = json.loads(output.read_text())["power_calibration"]
    assert fitted["a"] > 0
    assert fitted["b"] > 0
    assert fitted["total_weight"] == 0.75
    assert fitted["background_scaling"] == "annotated-to-full"
    assert fitted["validation_status"] == "background-audit-complete"
    assert fitted["structural_audit_ceiling"] == 100.0
