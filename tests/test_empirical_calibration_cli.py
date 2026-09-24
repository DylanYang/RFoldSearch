"""Argument combinations for empirical E-value fitting."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fit_empirical_evalue_calibration.py"


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
    assert "--query-panel cannot be combined with a weight column" in rejected.stderr
    assert not output.exists()

    background.write_text("query_id\tfamily\traw_evalue\nq1\tF1\t0.1\n")
    accepted = subprocess.run(command, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    fitted = json.loads(output.read_text())["empirical_calibration"]
    assert fitted["n_calibration_queries"] == 2
    assert fitted["weighting"] == "equal-family, equal-query-within-family"
